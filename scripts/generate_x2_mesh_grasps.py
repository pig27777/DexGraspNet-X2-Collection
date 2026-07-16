#!/usr/bin/env python3
"""Generate side-conditioned single-object X2 grasps for a watertight mesh."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_N_ITERATIONS = 6000
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_generation.x2_mesh_generator import (
    DENSE_HAND_SURFACE_SAMPLES_PER_SET,
    DENSE_OBJECT_SURFACE_SAMPLES,
    HAND_OBJECT_PENETRATION_THRESHOLD,
    make_sample_records,
    optimize_x2_mesh_batch,
)
from grasp_generation.utils.mesh_object_model import MeshObjectModel
from grasp_generation.utils.x2_config import X2Config, load_x2_mesh_config
from grasp_generation.utils.x2_hand_model import X2HandModel
from grasp_generation.utils.x2_mesh_contacts import (
    FINGER_NAMES,
    GenericDexterousContactPolicy,
    load_generic_contact_candidates,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DexGraspNet-style generic mesh grasp generation for X2"
    )
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument("--side", choices=("front", "back", "both", "any"), default="any")
    parser.add_argument("--num-grasps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-contact", type=int, default=4)
    parser.add_argument(
        "--finger-count",
        type=int,
        choices=range(1, 6),
        help="require exactly this many distinct non-palm fingers in every sample",
    )
    parser.add_argument(
        "--finger-names",
        nargs="+",
        choices=FINGER_NAMES,
        help="require this exact non-palm finger set (implies its count)",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=DEFAULT_N_ITERATIONS,
        help=(
            "annealing iterations; defaults to the original DexGraspNet "
            f"order of magnitude ({DEFAULT_N_ITERATIONS})"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("data/x2_mesh_grasps"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--object-scale", type=float)
    parser.add_argument("--surface-samples", type=int)
    parser.add_argument("--freeze-thumb", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.num_grasps <= 0 or args.batch_size <= 0 or args.n_contact <= 0:
        parser.error("num-grasps, batch-size, and n-contact must be positive")
    if args.n_iterations <= 0:
        parser.error("n-iterations must be positive")
    if args.finger_count is not None and args.finger_count > args.n_contact:
        parser.error("finger-count cannot exceed n-contact")
    if args.finger_names:
        if len(args.finger_names) != len(set(args.finger_names)):
            parser.error("finger-names must be unique")
        if args.finger_count is None:
            args.finger_count = len(args.finger_names)
        elif len(args.finger_names) != args.finger_count:
            parser.error("finger-names count must match finger-count")
    return args


def _with_cli_overrides(config: X2Config, args: argparse.Namespace) -> X2Config:
    data = copy.deepcopy(config.data)
    data["generation"]["n_contact"] = int(args.n_contact)
    if args.object_scale is not None:
        data["generation"]["object_scale"] = float(args.object_scale)
    if args.surface_samples is not None:
        data["generation"]["object_surface_samples"] = int(args.surface_samples)
    return X2Config(data=data, path=config.path, project_root=config.project_root)


def _requested_side_batches(
    mode: str, num_grasps: int, batch_size: int, rng: np.random.Generator
) -> list[tuple[str, ...]]:
    runs: list[tuple[str, ...]] = []
    if mode == "both":
        requested = (["front"] * num_grasps, ["back"] * num_grasps)
    elif mode in ("front", "back"):
        requested = ([mode] * num_grasps,)
    else:
        requested = ([str(v) for v in rng.choice(("front", "back"), size=num_grasps)],)
    for sides in requested:
        runs.extend(tuple(sides[start : start + batch_size]) for start in range(0, len(sides), batch_size))
    return runs


def _write_records(
    records: Sequence[dict[str, Any]],
    output_root: Path,
    counters: dict[str, int],
    *,
    overwrite: bool,
) -> list[Path]:
    written: list[Path] = []
    for record in records:
        side = str(record["active_side"])
        index = counters[side]
        counters[side] += 1
        directory = output_root / f"{side}_single" / "raw"
        directory.mkdir(parents=True, exist_ok=True)
        mesh_stem = Path(record["object"]["mesh_path"]).stem
        path = directory / f"{mesh_stem}_{side}_{index:06d}.json"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}; pass --overwrite")
        path.write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        written.append(path)
    return written


def _summarize(records: Sequence[dict[str, Any]], eligible_counts: dict[str, int]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["active_side"])].append(record)
    summary: dict[str, Any] = {
        "pipeline_revision": (
            str(records[0]["pipeline_revision"])
            if records
            else "x2_mesh_grasp_unselected_finger_side_v6"
        ),
        "simulation_run": False,
        "sides": {},
    }
    gate_records = [
        item["hand_object_penetration"]
        for item in records
        if isinstance(item.get("hand_object_penetration"), dict)
    ]
    if len(gate_records) != len(records):
        raise RuntimeError(
            "Every v5 generation record must contain dense hand-object diagnostics"
        )
    summary["dense_hand_object_gate"] = {
        "evaluation_mode": "dense_bidirectional",
        "hand_surface_samples_per_set": DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        "hand_surface_samples_per_link": 3
        * DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        "hand_surface_point_count": 17
        * 3
        * DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        "object_surface_samples": DENSE_OBJECT_SURFACE_SAMPLES,
        "threshold": HAND_OBJECT_PENETRATION_THRESHOLD,
        "strict_less_than": True,
        "sample_count": len(records),
        "evaluated_count": sum(
            bool(item.get("evaluated")) for item in gate_records
        ),
        "feasible_count": sum(
            bool(item.get("feasible")) for item in gate_records
        ),
    }
    for side in ("front", "back"):
        values = grouped.get(side, [])
        if not values:
            continue
        initial = np.asarray([item["energy"]["initial_total"] for item in values], dtype=np.float64)
        final = np.asarray([item["energy"]["total"] for item in values], dtype=np.float64)
        actuator = np.asarray([item["actuator"] for item in values], dtype=np.float64)
        regions = Counter(
            contact["region"] for item in values for contact in item["selected_contacts"]
        )
        self_collision = [item["self_collision"] for item in values]
        hand_object = [item["hand_object_penetration"] for item in values]
        summary["sides"][side] = {
            "initialized": len(values),
            "optimized": len(values),
            "eligible_contact_candidates": eligible_counts[side],
            "initial_energy_mean": float(initial.mean()),
            "final_energy_mean": float(final.mean()),
            "energy_change_mean": float((final - initial).mean()),
            "energy_decreased_count": int(np.sum(final < initial)),
            "selected_contact_regions": dict(sorted(regions.items())),
            "actuator_min": float(actuator.min()),
            "actuator_max": float(actuator.max()),
            "maximum_penetration": float(
                max(item["maximum_penetration"] for item in values)
            ),
            "self_collision_maximum_penetration": float(
                max(item["maximum_penetration"] for item in self_collision)
            ),
            "self_collision_total_penetration_mean": float(
                np.mean(
                    [item["total_penetration"] for item in self_collision],
                    dtype=np.float64,
                )
            ),
            "self_collision_feasible_count": int(
                sum(bool(item["feasible"]) for item in self_collision)
            ),
            "self_collision_threshold": float(self_collision[0]["threshold"]),
            "dense_hand_object_evaluated_count": int(
                sum(bool(item["evaluated"]) for item in hand_object)
            ),
            "dense_hand_object_feasible_count": int(
                sum(bool(item["feasible"]) for item in hand_object)
            ),
            "nan_or_inf": not all(item["finite"] for item in values),
            "accepted_contact_changes": int(
                sum(item["optimization"]["accepted_contact_changes"] for item in values)
                / len(values)
            ),
        }
    return summary


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    finger_count = getattr(args, "finger_count", None)
    config = _with_cli_overrides(load_x2_mesh_config(args.config), args)
    candidates = load_generic_contact_candidates(
        config.configured_path("contact_candidates.path", must_exist=True)
    )
    policies = {
        side: GenericDexterousContactPolicy(
            candidates,
            active_side=side,
            n_contact=args.n_contact,
            allow_thumb=bool(config.require("contact_candidates.allow_thumb")),
            target_finger_count=finger_count,
            required_finger_names=getattr(args, "finger_names", None),
        )
        for side in ("front", "back")
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
        freeze_thumb=args.freeze_thumb,
    )

    records: list[dict[str, Any]] = []
    side_batches = _requested_side_batches(args.side, args.num_grasps, args.batch_size, rng)
    for batch_index, active_sides in enumerate(side_batches):
        object_model = MeshObjectModel(
            args.mesh_path,
            batch_size=len(active_sides),
            scale=float(config.require("generation.object_scale")),
            num_surface_samples=int(config.require("generation.object_surface_samples")),
            audit_surface_samples=int(
                config.require("generation.dense_object_surface_samples")
            ),
            device=device,
            dtype=torch.float64,
            seed=args.seed + batch_index,
        )
        result = optimize_x2_mesh_batch(
            hand,
            object_model,
            active_sides,
            policies,
            config,
            n_iterations=args.n_iterations,
            seed=args.seed + batch_index * 100003,
            rng=rng,
        )
        batch_records = make_sample_records(
            hand, object_model, result, candidates, seed=args.seed
        )
        for local_index, record in enumerate(batch_records):
            record["sample_index"] = len(records) + local_index
            participating = sorted(
                {
                    contact["finger_name"]
                    for contact in record["selected_contacts"]
                    if contact["finger_name"] in FINGER_NAMES
                },
                key=FINGER_NAMES.index,
            )
            record["finger_participation"] = {
                "target_count": finger_count or len(participating),
                "actual_count": len(participating),
                "finger_names": participating,
            }
            if finger_count is not None and len(participating) != finger_count:
                raise RuntimeError(
                    "Optimizer returned a contact selection outside the requested "
                    f"finger stratum: expected={finger_count}, actual={participating}"
                )
        records.extend(batch_records)

    counters = {"front": 0, "back": 0}
    paths = _write_records(
        records, args.output.expanduser().resolve(), counters, overwrite=args.overwrite
    )
    summary = _summarize(
        records,
        {side: len(policy.eligible_indices) for side, policy in policies.items()},
    )
    summary.update(
        {
            "mesh_path": str(args.mesh_path.expanduser().resolve()),
            "side_mode": args.side,
            "num_output_samples": len(records),
            "finger_count": finger_count,
            "output_files": [str(path) for path in paths],
            "dual_object_samples": 0,
            "valid_files_written": 0,
        }
    )
    return records, summary


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _, summary = run(args)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
