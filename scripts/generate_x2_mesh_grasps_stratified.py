#!/usr/bin/env python3
"""Generate many exact X2 finger strata for one mesh in one resident process.

The JSON plan contains either a top-level list or ``{"groups": [...]}``.  Each
group has exactly ``side``, ``finger_count``, ``finger_names``, ``num_grasps``,
and ``output``.  Rows from groups with the same contact count are scheduled in
round-robin order and can share one optimizer call through the row-policy API.
Four- and five-contact rows are always placed in separate calls.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_generation.x2_mesh_generator import (  # noqa: E402
    DENSE_HAND_SURFACE_SAMPLES_PER_SET,
    DENSE_OBJECT_SURFACE_SAMPLES,
    HAND_OBJECT_PENETRATION_THRESHOLD,
    make_sample_records,
    optimize_x2_mesh_batch,
)
from grasp_generation.utils.mesh_object_model import MeshObjectModel  # noqa: E402
from grasp_generation.utils.x2_config import (  # noqa: E402
    X2Config,
    load_x2_mesh_config,
)
from grasp_generation.utils.x2_hand_model import X2HandModel  # noqa: E402
from grasp_generation.utils.x2_mesh_contacts import (  # noqa: E402
    FINGER_NAMES,
    GenericDexterousContactPolicy,
    load_generic_contact_candidates,
)


DEFAULT_N_ITERATIONS = 6000
DEFAULT_BATCH_SIZE = 64
PLAN_FIELDS = frozenset(
    {"side", "finger_count", "finger_names", "num_grasps", "output"}
)


class StratifiedGenerationError(RuntimeError):
    """Raised when a plan or generated batch violates the stratified contract."""


@dataclass(frozen=True)
class PlanGroup:
    index: int
    side: str
    finger_count: int
    finger_names: tuple[str, ...]
    num_grasps: int
    output: Path

    @property
    def n_contact(self) -> int:
        return max(4, self.finger_count)


@dataclass(frozen=True)
class ScheduledRow:
    group: PlanGroup
    group_sample_index: int


@dataclass(frozen=True)
class ScheduledBatch:
    index: int
    contact_count: int
    rows: tuple[ScheduledRow, ...]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _strict_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise StratifiedGenerationError(
            f"{path} contains non-finite JSON constant {value}"
        )

    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    except StratifiedGenerationError:
        raise
    except Exception as exc:
        raise StratifiedGenerationError(f"Cannot read plan {path}: {exc}") from exc


def load_plan(path: Path | str) -> tuple[PlanGroup, ...]:
    """Load and strictly normalize a stratified generation plan."""

    plan_path = Path(path).expanduser().resolve()
    payload = _strict_json(plan_path)
    if isinstance(payload, dict):
        if set(payload) != {"groups"}:
            raise StratifiedGenerationError(
                "Plan object must contain exactly one field: groups"
            )
        values = payload["groups"]
    else:
        values = payload
    if not isinstance(values, list) or not values:
        raise StratifiedGenerationError("Plan groups must be a non-empty JSON list")

    groups: list[PlanGroup] = []
    used_destinations: set[Path] = set()
    for index, value in enumerate(values):
        label = f"plan group {index}"
        if not isinstance(value, dict) or set(value) != PLAN_FIELDS:
            actual = sorted(value) if isinstance(value, dict) else type(value).__name__
            raise StratifiedGenerationError(
                f"{label} must contain exactly {sorted(PLAN_FIELDS)}; got {actual}"
            )
        side = value["side"]
        if side not in ("front", "back"):
            raise StratifiedGenerationError(f"{label}.side must be front or back")
        finger_count = value["finger_count"]
        if (
            isinstance(finger_count, bool)
            or not isinstance(finger_count, int)
            or finger_count < 1
            or finger_count > 5
        ):
            raise StratifiedGenerationError(
                f"{label}.finger_count must be an integer in 1..5"
            )
        names = value["finger_names"]
        if (
            not isinstance(names, list)
            or len(names) != finger_count
            or any(not isinstance(name, str) for name in names)
            or len(set(names)) != finger_count
            or any(name not in FINGER_NAMES for name in names)
        ):
            raise StratifiedGenerationError(
                f"{label}.finger_names must contain {finger_count} unique known fingers"
            )
        name_set = set(names)
        normalized_names = tuple(name for name in FINGER_NAMES if name in name_set)
        num_grasps = value["num_grasps"]
        if (
            isinstance(num_grasps, bool)
            or not isinstance(num_grasps, int)
            or num_grasps <= 0
        ):
            raise StratifiedGenerationError(
                f"{label}.num_grasps must be a positive integer"
            )
        output_value = value["output"]
        if not isinstance(output_value, str) or not output_value.strip():
            raise StratifiedGenerationError(
                f"{label}.output must be a non-empty path string"
            )
        output = Path(output_value).expanduser()
        if not output.is_absolute():
            output = plan_path.parent / output
        output = output.resolve()
        raw_destination = output / f"{side}_single" / "raw"
        if raw_destination in used_destinations:
            raise StratifiedGenerationError(
                f"{label} shares its raw output directory with another group: "
                f"{raw_destination}"
            )
        used_destinations.add(raw_destination)
        groups.append(
            PlanGroup(
                index=index,
                side=side,
                finger_count=finger_count,
                finger_names=normalized_names,
                num_grasps=num_grasps,
                output=output,
            )
        )
    return tuple(groups)


def schedule_batches(
    groups: Sequence[PlanGroup], batch_size: int
) -> tuple[ScheduledBatch, ...]:
    """Round-robin groups within rectangular four/five-contact batches."""

    if batch_size <= 0:
        raise StratifiedGenerationError("batch_size must be positive")
    batches: list[ScheduledBatch] = []
    for contact_count in (4, 5):
        contact_groups = [group for group in groups if group.n_contact == contact_count]
        emitted = {group.index: 0 for group in contact_groups}
        rows: list[ScheduledRow] = []
        while True:
            added = False
            for group in contact_groups:
                local_index = emitted[group.index]
                if local_index < group.num_grasps:
                    rows.append(ScheduledRow(group, local_index))
                    emitted[group.index] += 1
                    added = True
            if not added:
                break
        for start in range(0, len(rows), batch_size):
            chunk = tuple(rows[start : start + batch_size])
            batches.append(
                ScheduledBatch(
                    index=len(batches),
                    contact_count=contact_count,
                    rows=chunk,
                )
            )
    expected = sum(group.num_grasps for group in groups)
    if sum(len(batch.rows) for batch in batches) != expected:
        raise AssertionError("Stratified scheduler lost planned rows")
    return tuple(batches)


def _with_cli_overrides(config: X2Config, args: argparse.Namespace) -> X2Config:
    data = copy.deepcopy(config.data)
    # The row policies own the actual rectangular contact count.  Keep this
    # legacy config field positive and representative of the common f1..f4 case.
    data["generation"]["n_contact"] = 4
    if args.object_scale is not None:
        data["generation"]["object_scale"] = float(args.object_scale)
    if args.surface_samples is not None:
        data["generation"]["object_surface_samples"] = int(args.surface_samples)
    return X2Config(data=data, path=config.path, project_root=config.project_root)


def _output_path(mesh_path: Path, row: ScheduledRow) -> Path:
    directory = row.group.output / f"{row.group.side}_single" / "raw"
    return directory / (
        f"{mesh_path.stem}_{row.group.side}_{row.group_sample_index:06d}.json"
    )


def _preflight_outputs(
    mesh_path: Path, groups: Sequence[PlanGroup], *, overwrite: bool
) -> None:
    if overwrite:
        return
    conflicts: list[str] = []
    for group in groups:
        directory = group.output / f"{group.side}_single" / "raw"
        pattern = f"{mesh_path.stem}_{group.side}_*.json"
        if any(directory.glob(pattern)):
            conflicts.append(f"group {group.index}: {directory / pattern}")
    if conflicts:
        raise StratifiedGenerationError(
            "Output already exists; pass --overwrite: " + ", ".join(conflicts)
        )


def _publish_records(
    *,
    mesh_path: Path,
    groups: Sequence[PlanGroup],
    records_by_group: dict[int, list[dict[str, Any]]],
    overwrite: bool,
) -> dict[int, list[Path]]:
    output_paths: dict[int, list[Path]] = {group.index: [] for group in groups}
    for group in groups:
        records = records_by_group[group.index]
        if len(records) != group.num_grasps:
            raise StratifiedGenerationError(
                f"group {group.index} produced {len(records)} records; "
                f"expected {group.num_grasps}"
            )
        directory = group.output / f"{group.side}_single" / "raw"
        directory.mkdir(parents=True, exist_ok=True)
        pattern = f"{mesh_path.stem}_{group.side}_*.json"
        if overwrite:
            for existing in directory.glob(pattern):
                existing.unlink()
        for index, record in enumerate(records):
            row = ScheduledRow(group=group, group_sample_index=index)
            path = _output_path(mesh_path, row)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(record, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
            output_paths[group.index].append(path.resolve())
    return output_paths


def run(args: argparse.Namespace) -> dict[str, Any]:
    mesh_path = Path(args.mesh_path).expanduser().resolve()
    if not mesh_path.is_file():
        raise StratifiedGenerationError(f"Mesh does not exist: {mesh_path}")
    groups = load_plan(args.plan)
    batches = schedule_batches(groups, args.batch_size)
    _preflight_outputs(mesh_path, groups, overwrite=args.overwrite)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    config = _with_cli_overrides(load_x2_mesh_config(), args)
    candidates = load_generic_contact_candidates(
        config.configured_path("contact_candidates.path", must_exist=True)
    )
    allow_thumb = bool(config.require("contact_candidates.allow_thumb"))
    policies = {
        group.index: GenericDexterousContactPolicy(
            candidates,
            active_side=group.side,
            n_contact=group.n_contact,
            allow_thumb=allow_thumb,
            target_finger_count=group.finger_count,
            required_finger_names=group.finger_names,
        )
        for group in groups
    }
    device = torch.device(args.device)
    hand = X2HandModel(
        config,
        candidates,
        device=device,
        dtype=torch.float64,
        collision_samples_per_link=int(
            config.require("generation.hand_collision_samples_per_link")
        ),
        audit_collision_samples_per_link=int(
            config.require("generation.dense_hand_surface_samples_per_set")
        ),
    )

    records_by_group: dict[int, list[dict[str, Any]]] = {
        group.index: [] for group in groups
    }
    batch_reports: list[dict[str, Any]] = []
    for batch in batches:
        active_sides = tuple(row.group.side for row in batch.rows)
        row_policies = tuple(policies[row.group.index] for row in batch.rows)
        if {policy.n_contact for policy in row_policies} != {batch.contact_count}:
            raise AssertionError("Scheduler mixed contact counts in one optimizer call")
        object_model = MeshObjectModel(
            mesh_path,
            batch_size=len(batch.rows),
            scale=float(config.require("generation.object_scale")),
            num_surface_samples=int(
                config.require("generation.object_surface_samples")
            ),
            audit_surface_samples=int(
                config.require("generation.dense_object_surface_samples")
            ),
            device=device,
            dtype=torch.float64,
            seed=args.seed + batch.index,
        )
        result = optimize_x2_mesh_batch(
            hand,
            object_model,
            active_sides,
            row_policies,
            config,
            n_iterations=args.n_iterations,
            seed=args.seed + batch.index * 100003,
            rng=rng,
        )
        batch_records = make_sample_records(
            hand, object_model, result, candidates, seed=args.seed
        )
        if len(batch_records) != len(batch.rows):
            raise StratifiedGenerationError(
                f"batch {batch.index} returned {len(batch_records)} records for "
                f"{len(batch.rows)} rows"
            )
        for row, policy, record in zip(batch.rows, row_policies, batch_records):
            if record.get("active_side") != row.group.side:
                raise StratifiedGenerationError(
                    f"batch {batch.index} returned the wrong active side for "
                    f"group {row.group.index}"
                )
            selected_contacts = record.get("selected_contacts")
            if not isinstance(selected_contacts, list):
                raise StratifiedGenerationError(
                    f"batch {batch.index} record has no selected_contacts list"
                )
            participating = tuple(
                name
                for name in FINGER_NAMES
                if any(
                    isinstance(contact, dict) and contact.get("finger_name") == name
                    for contact in selected_contacts
                )
            )
            if participating != row.group.finger_names:
                raise StratifiedGenerationError(
                    f"group {row.group.index} returned fingers {participating}; "
                    f"expected {row.group.finger_names}"
                )
            if policy.required_finger_names != row.group.finger_names:
                raise AssertionError("Row policy drifted from its exact plan mask")
            record["sample_index"] = row.group_sample_index
            record["finger_participation"] = {
                "target_count": row.group.finger_count,
                "actual_count": len(participating),
                "finger_names": list(participating),
            }
            records_by_group[row.group.index].append(record)
        batch_reports.append(
            {
                "batch_index": batch.index,
                "contact_count": batch.contact_count,
                "sample_count": len(batch.rows),
                "group_indices": [row.group.index for row in batch.rows],
            }
        )

    output_paths = _publish_records(
        mesh_path=mesh_path,
        groups=groups,
        records_by_group=records_by_group,
        overwrite=args.overwrite,
    )
    total = sum(len(values) for values in records_by_group.values())
    all_records = [
        record
        for group in groups
        for record in records_by_group[group.index]
    ]
    gate_records = [
        record.get("hand_object_penetration") for record in all_records
    ]
    if any(not isinstance(value, dict) for value in gate_records):
        raise StratifiedGenerationError(
            "Every v5 generation record must contain dense hand-object diagnostics"
        )
    first_record = next(
        record
        for group in groups
        for record in records_by_group[group.index]
    )
    return {
        "passed": True,
        "pipeline_revision": first_record.get("pipeline_revision"),
        "simulation_run": False,
        "mesh_path": str(mesh_path),
        "plan": str(Path(args.plan).expanduser().resolve()),
        "side_mode": "stratified",
        "num_output_samples": total,
        "group_count": len(groups),
        "batch_count": len(batches),
        "batch_size": args.batch_size,
        "n_iterations": args.n_iterations,
        "seed": args.seed,
        "device": args.device,
        "object_scale": float(config.require("generation.object_scale")),
        "surface_samples": int(
            config.require("generation.object_surface_samples")
        ),
        "dense_hand_object_gate": {
            "evaluation_mode": "dense_bidirectional",
            "hand_surface_samples_per_set": (
                DENSE_HAND_SURFACE_SAMPLES_PER_SET
            ),
            "hand_surface_samples_per_link": (
                3 * DENSE_HAND_SURFACE_SAMPLES_PER_SET
            ),
            "hand_surface_point_count": (
                17 * 3 * DENSE_HAND_SURFACE_SAMPLES_PER_SET
            ),
            "object_surface_samples": DENSE_OBJECT_SURFACE_SAMPLES,
            "threshold": HAND_OBJECT_PENETRATION_THRESHOLD,
            "strict_less_than": True,
            "sample_count": total,
            "evaluated_count": sum(
                bool(value.get("evaluated")) for value in gate_records
            ),
            "feasible_count": sum(
                bool(value.get("feasible")) for value in gate_records
            ),
        },
        "groups": [
            {
                "index": group.index,
                "side": group.side,
                "finger_count": group.finger_count,
                "finger_names": list(group.finger_names),
                "n_contact": group.n_contact,
                "num_output_samples": len(records_by_group[group.index]),
                "output": str(group.output),
                "output_files": [str(path) for path in output_paths[group.index]],
            }
            for group in groups
        ],
        "batches": batch_reports,
        "output_files": [
            str(path)
            for group in groups
            for path in output_paths[group.index]
        ],
        "dual_object_samples": 0,
        "valid_files_written": 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--n-iterations", type=_positive_int, default=DEFAULT_N_ITERATIONS
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--object-scale", type=_positive_float)
    parser.add_argument("--surface-samples", type=_positive_int)
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary = run(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
