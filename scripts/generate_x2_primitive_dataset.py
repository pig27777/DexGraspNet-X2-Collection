#!/usr/bin/env python3
"""Generate side-separated X2 raw grasp candidates for primitive meshes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_x2_primitive_dataset import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT as DEFAULT_MESH_ROOT,
    SHAPES,
    PrimitiveSpec,
    build_dataset,
    selected_specs,
)


GENERATOR = PROJECT_ROOT / "scripts" / "generate_x2_mesh_grasps.py"
STRATIFIED_GENERATOR = (
    PROJECT_ROOT / "scripts" / "generate_x2_mesh_grasps_stratified.py"
)
DEFAULT_GRASP_ROOT = PROJECT_ROOT / "data" / "x2_primitive_grasps"
DEFAULT_GENERAL_MESH_ROOT = PROJECT_ROOT / "data" / "meshdata"
DEFAULT_SUMMARY_NAME = "summary.csv"
DEFAULT_JSON_SUMMARY_NAME = "generation_summary.json"
SIDES = ("front", "back")
DEFAULT_NUM_GRASPS = 64
DEFAULT_N_ITERATIONS = 6000
BATCH_SIZE = 8
STRATIFIED_BATCH_SIZE = 64
N_CONTACT = 4
SURFACE_SAMPLES = 512
OBJECT_SCALE = 1.0
FINGER_NAMES = ("index", "middle", "ring", "little", "thumb")
GENERATOR_PIPELINE_REVISION = "x2_mesh_grasp_unselected_finger_side_v6"
DENSE_GATE_EVALUATION_MODE = "dense_bidirectional"
DENSE_HAND_SURFACE_SAMPLES_PER_SET = 256
DENSE_HAND_SURFACE_SAMPLES_PER_LINK = 768
DENSE_HAND_SURFACE_POINT_COUNT = 13056
DENSE_OBJECT_SURFACE_SAMPLES = 8192
DENSE_HAND_OBJECT_PENETRATION_THRESHOLD = 0.001
CSV_FIELDS = (
    "shape",
    "size",
    "side",
    "finger_count",
    "sample_count",
    "finite_sample_count",
    "mean_initial_energy",
    "mean_final_energy",
    "energy_decreased_count",
    "maximum_penetration_mean",
    "maximum_penetration_min",
    "maximum_penetration_median",
)


class PrimitiveGenerationError(RuntimeError):
    """Raised when a primitive generator run or publication audit fails."""


@dataclass(frozen=True)
class GenerationTask:
    """One independently generated instance/side group in canonical order."""

    spec: PrimitiveSpec | GeneralMeshSpec
    side: str
    num_grasps: int
    finger_count: int | None = None
    finger_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class StagedResult:
    """Fully validated staging output which is safe to publish."""

    task: GenerationTask
    source_paths: tuple[Path, ...]
    row: dict[str, Any]
    generator_call: dict[str, Any]


@dataclass(frozen=True)
class GeneralMeshSpec:
    """One repository object stored as ``<object>/coacd/decomposed.obj``."""

    instance_name: str
    path: Path
    object_scale: float = 0.1
    shape: str = "general"

    @property
    def size(self) -> str:
        return f"configured_scale={self.object_scale:g}"


def discover_general_meshes(mesh_root: Path) -> tuple[GeneralMeshSpec, ...]:
    """Discover generic objects deterministically and reject ambiguous IDs."""

    root = Path(mesh_root).expanduser().resolve()
    paths = tuple(
        path.resolve()
        for path in sorted(root.glob("*/coacd/decomposed.obj"))
        if path.is_file()
    )
    scale_by_id = {path.parent.parent.name: 0.1 for path in paths}
    manifest_path = root / "x2_general_mesh_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PrimitiveGenerationError(
                f"Cannot read general mesh manifest {manifest_path}: {exc}"
            ) from exc
        records = manifest.get("meshes") if isinstance(manifest, dict) else None
        if not isinstance(records, list):
            raise PrimitiveGenerationError(
                f"General mesh manifest has no meshes list: {manifest_path}"
            )
        manifest_ids: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise PrimitiveGenerationError(
                    f"General mesh manifest contains an invalid record"
                )
            object_id = record.get("object_id")
            scale = record.get("object_scale")
            sha256 = record.get("sha256")
            if (
                not isinstance(object_id, str)
                or object_id in manifest_ids
                or isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not math.isfinite(float(scale))
                or float(scale) <= 0.0
                or not isinstance(sha256, str)
            ):
                raise PrimitiveGenerationError(
                    f"General mesh manifest contains invalid scale/hash metadata"
                )
            path = root / object_id / "coacd" / "decomposed.obj"
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
                raise PrimitiveGenerationError(
                    f"General mesh manifest hash is stale for {object_id}"
                )
            manifest_ids.add(object_id)
            scale_by_id[object_id] = float(scale)
        discovered_ids = {path.parent.parent.name for path in paths}
        if manifest_ids != discovered_ids:
            raise PrimitiveGenerationError(
                "General mesh manifest IDs do not match discovered mesh files"
            )
    specs = tuple(
        GeneralMeshSpec(
            path.parent.parent.name,
            path,
            object_scale=scale_by_id[path.parent.parent.name],
        )
        for path in paths
    )
    names = [spec.instance_name for spec in specs]
    if len(names) != len(set(names)):
        raise PrimitiveGenerationError("General mesh object IDs must be unique")
    return specs


def select_general_meshes(
    specs: Sequence[GeneralMeshSpec], object_ids: Sequence[str] | None
) -> tuple[GeneralMeshSpec, ...]:
    """Select an explicit audited subset while preserving catalog order."""

    if object_ids is None:
        return tuple(specs)
    requested = tuple(str(value) for value in object_ids)
    if not requested or len(requested) != len(set(requested)):
        raise PrimitiveGenerationError(
            "general-mesh-ids must contain unique non-empty object IDs"
        )
    by_id = {spec.instance_name: spec for spec in specs}
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise PrimitiveGenerationError(
            f"Requested general mesh IDs are missing: {missing}"
        )
    requested_set = set(requested)
    return tuple(spec for spec in specs if spec.instance_name in requested_set)


def requested_sides(mode: str) -> tuple[str, ...]:
    if mode == "both":
        return SIDES
    if mode in SIDES:
        return (mode,)
    raise PrimitiveGenerationError(f"Unsupported side mode: {mode!r}")


def plan_generation(
    specs: Sequence[PrimitiveSpec | GeneralMeshSpec],
    sides: Sequence[str],
    *,
    num_grasps: int,
    target_total: int | None = None,
    finger_targets: Mapping[int, int] | None = None,
    finger_counts: Sequence[int] | None = None,
    complementary_side_fingers: bool = False,
    seed: int = 0,
) -> tuple[GenerationTask, ...]:
    """Assign deterministic per-group counts, optionally summing to an exact total."""

    strata: tuple[int | None, ...] = (
        tuple(int(value) for value in finger_counts)
        if finger_counts
        else (None,)
    )
    if any(value is not None and (value < 1 or value > 5) for value in strata):
        raise PrimitiveGenerationError("finger-counts must contain values in 1..5")
    if len(strata) != len(set(strata)):
        raise PrimitiveGenerationError("finger-counts must be unique")
    pairs = [
        (spec, side, finger_count)
        for spec in specs
        for side in sides
        for finger_count in strata
    ]
    if not pairs:
        raise PrimitiveGenerationError("At least one primitive instance/side is required")
    if target_total is not None and finger_targets is not None:
        raise PrimitiveGenerationError(
            "target_total and finger_targets are mutually exclusive"
        )
    if finger_targets is not None:
        if any(value is None for value in strata):
            raise PrimitiveGenerationError(
                "finger_targets requires explicit finger_counts"
            )
        normalized_targets = {
            int(key): int(value) for key, value in finger_targets.items()
        }
        if set(normalized_targets) != set(strata):
            raise PrimitiveGenerationError(
                "finger_targets keys must exactly match finger_counts"
            )
        counts = [0] * len(pairs)
        for finger_count in strata:
            indices = [
                index
                for index, (_, _, value) in enumerate(pairs)
                if value == finger_count
            ]
            target = normalized_targets[int(finger_count)]
            if target < len(indices):
                raise PrimitiveGenerationError(
                    f"finger f{finger_count} target must be at least the "
                    f"{len(indices)} selected instance-side groups"
                )
            quotient, remainder = divmod(target, len(indices))
            for depth, index in enumerate(indices):
                counts[index] = quotient + (1 if depth < remainder else 0)
    elif target_total is None:
        counts = [int(num_grasps)] * len(pairs)
    else:
        if target_total < len(pairs):
            raise PrimitiveGenerationError(
                f"target-total must be at least the {len(pairs)} selected instance-side groups"
            )
        quotient, remainder = divmod(int(target_total), len(pairs))
        counts = [
            quotient + (1 if index < remainder else 0)
            for index in range(len(pairs))
        ]
    if any(count <= 0 for count in counts):
        raise PrimitiveGenerationError("Every instance-side group must receive a sample")

    def selected_names(
        spec: PrimitiveSpec | GeneralMeshSpec,
        side: str,
        finger_count: int | None,
    ) -> tuple[str, ...]:
        if not complementary_side_fingers or finger_count is None:
            return ()
        if finger_count == 5:
            return FINGER_NAMES
        front_count = finger_count if side == "front" else 5 - finger_count
        choices = tuple(itertools.combinations(FINGER_NAMES, front_count))
        digest = hashlib.sha256(
            f"{spec.instance_name}:{front_count}:{seed}".encode("utf-8")
        ).digest()
        front = choices[int.from_bytes(digest[:8], "big") % len(choices)]
        if side == "front":
            return front
        front_set = set(front)
        return tuple(name for name in FINGER_NAMES if name not in front_set)

    return tuple(
        GenerationTask(
            spec=spec,
            side=side,
            num_grasps=count,
            finger_count=finger_count,
            finger_names=selected_names(spec, side, finger_count),
        )
        for (spec, side, finger_count), count in zip(pairs, counts)
    )


def target_directory(output_root: Path, spec: PrimitiveSpec | GeneralMeshSpec, side: str) -> Path:
    return Path(output_root) / spec.shape / side / "raw"


def target_filename(
    spec: PrimitiveSpec | GeneralMeshSpec,
    side: str,
    sample_index: int,
    finger_count: int | None = None,
) -> str:
    finger_tag = f"_f{finger_count}" if finger_count is not None else ""
    return f"{spec.instance_name}{finger_tag}_{side}_{sample_index:06d}.json"


def target_glob(
    spec: PrimitiveSpec | GeneralMeshSpec, side: str, finger_count: int | None
) -> str:
    finger_tag = f"_f{finger_count}" if finger_count is not None else ""
    return f"{spec.instance_name}{finger_tag}_{side}_*.json"


def build_generator_command(
    *,
    spec: PrimitiveSpec | GeneralMeshSpec,
    side: str,
    mesh_path: Path,
    staging_output: Path,
    num_grasps: int,
    n_iterations: int,
    device: str,
    seed: int,
    object_scale: float = OBJECT_SCALE,
    finger_count: int | None = None,
    finger_names: Sequence[str] = (),
) -> list[str]:
    """Build the exact subprocess command for one instance and one side."""

    command = [
        sys.executable,
        str(GENERATOR),
        "--mesh-path",
        str(mesh_path),
        "--side",
        side,
        "--num-grasps",
        str(num_grasps),
        "--batch-size",
        str(BATCH_SIZE),
        "--n-contact",
        str(max(N_CONTACT, finger_count or 0)),
        "--n-iterations",
        str(n_iterations),
        "--surface-samples",
        str(SURFACE_SAMPLES),
        "--object-scale",
        str(object_scale),
        "--seed",
        str(seed),
        "--device",
        device,
        "--output",
        str(staging_output),
        "--overwrite",
    ]
    if finger_count is not None:
        command.extend(("--finger-count", str(finger_count)))
    if finger_names:
        command.append("--finger-names")
        command.extend(str(value) for value in finger_names)
    return command


def _reject_json_constant(value: str) -> None:
    raise PrimitiveGenerationError(f"Generated JSON contains non-finite constant {value}")


def _finite_number(value: Any, *, field: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PrimitiveGenerationError(f"{path}: {field} is not numeric") from exc
    if not math.isfinite(result):
        raise PrimitiveGenerationError(f"{path}: {field} is not finite")
    return result


def _dense_gate_contract() -> dict[str, Any]:
    """Return the immutable formal v5 dense hand/object gate settings."""

    return {
        "evaluation_mode": DENSE_GATE_EVALUATION_MODE,
        "hand_surface_samples_per_set": DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        "hand_surface_samples_per_link": DENSE_HAND_SURFACE_SAMPLES_PER_LINK,
        "hand_surface_point_count": DENSE_HAND_SURFACE_POINT_COUNT,
        "object_surface_samples": DENSE_OBJECT_SURFACE_SAMPLES,
        "threshold": DENSE_HAND_OBJECT_PENETRATION_THRESHOLD,
        "strict_less_than": True,
    }


def _audit_dense_hand_object_record(
    record: Mapping[str, Any], *, path: Path, top_level_maximum: float
) -> bool:
    """Audit one raw record against the exact formal v5 dense gate contract."""

    if record.get("pipeline_revision") != GENERATOR_PIPELINE_REVISION:
        raise PrimitiveGenerationError(
            f"{path}: pipeline_revision must be {GENERATOR_PIPELINE_REVISION}"
        )
    value = record.get("hand_object_penetration")
    if not isinstance(value, Mapping):
        raise PrimitiveGenerationError(
            f"{path}: v5 hand_object_penetration diagnostics are required"
        )
    if value.get("evaluation_mode") != DENSE_GATE_EVALUATION_MODE:
        raise PrimitiveGenerationError(
            f"{path}: hand_object_penetration.evaluation_mode must be "
            f"{DENSE_GATE_EVALUATION_MODE}"
        )
    if value.get("evaluated") is not True:
        raise PrimitiveGenerationError(
            f"{path}: hand_object_penetration.evaluated must be true"
        )

    expected_counts = {
        "hand_surface_samples_per_set": DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        "hand_surface_samples_per_link": DENSE_HAND_SURFACE_SAMPLES_PER_LINK,
        "hand_surface_point_count": DENSE_HAND_SURFACE_POINT_COUNT,
        "object_surface_samples": DENSE_OBJECT_SURFACE_SAMPLES,
    }
    for key, expected in expected_counts.items():
        actual = value.get(key)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise PrimitiveGenerationError(
                f"{path}: hand_object_penetration.{key} must equal {expected}"
            )

    numeric: dict[str, float] = {}
    for key in (
        "forward_total_penetration",
        "forward_maximum_penetration",
        "reverse_total_penetration",
        "reverse_maximum_penetration",
        "total_penetration",
        "maximum_penetration",
        "threshold",
    ):
        raw_number = value.get(key)
        if isinstance(raw_number, bool):
            raise PrimitiveGenerationError(
                f"{path}: hand_object_penetration.{key} must be finite and non-negative"
            )
        number = _finite_number(
            raw_number, field=f"hand_object_penetration.{key}", path=path
        )
        if number < 0.0:
            raise PrimitiveGenerationError(
                f"{path}: hand_object_penetration.{key} must be finite and non-negative"
            )
        numeric[key] = number
    if numeric["threshold"] != DENSE_HAND_OBJECT_PENETRATION_THRESHOLD:
        raise PrimitiveGenerationError(
            f"{path}: hand_object_penetration.threshold must equal "
            f"{DENSE_HAND_OBJECT_PENETRATION_THRESHOLD}"
        )

    def numerically_equal(actual: float, expected: float) -> bool:
        return math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-15)

    expected_total = (
        numeric["forward_total_penetration"]
        + numeric["reverse_total_penetration"]
    )
    if not numerically_equal(numeric["total_penetration"], expected_total):
        raise PrimitiveGenerationError(
            f"{path}: hand_object_penetration.total_penetration is inconsistent"
        )
    expected_maximum = max(
        numeric["forward_maximum_penetration"],
        numeric["reverse_maximum_penetration"],
    )
    if not numerically_equal(numeric["maximum_penetration"], expected_maximum):
        raise PrimitiveGenerationError(
            f"{path}: hand_object_penetration.maximum_penetration is inconsistent"
        )
    if not numerically_equal(top_level_maximum, numeric["maximum_penetration"]):
        raise PrimitiveGenerationError(
            f"{path}: top-level maximum_penetration disagrees with dense diagnostics"
        )
    feasible = value.get("feasible")
    if not isinstance(feasible, bool):
        raise PrimitiveGenerationError(
            f"{path}: hand_object_penetration.feasible must be boolean"
        )
    expected_feasible = (
        numeric["maximum_penetration"]
        < DENSE_HAND_OBJECT_PENETRATION_THRESHOLD
    )
    if feasible is not expected_feasible:
        raise PrimitiveGenerationError(
            f"{path}: hand_object_penetration.feasible is inconsistent with "
            "the strict 1 mm gate"
        )
    return feasible


def _dense_gate_summary_from_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize already-audited records without weakening the gate contract."""

    summary = _dense_gate_contract()
    summary.update(
        {
            "sample_count": len(records),
            "evaluated_count": sum(
                record["hand_object_penetration"]["evaluated"] is True
                for record in records
            ),
            "feasible_count": sum(
                record["hand_object_penetration"]["feasible"] is True
                for record in records
            ),
        }
    )
    return summary


def _audit_dense_gate_summary(
    summary: Mapping[str, Any], *, expected_sample_count: int, label: str
) -> dict[str, Any]:
    """Validate a subprocess summary and return its normalized dense gate proof."""

    if summary.get("pipeline_revision") != GENERATOR_PIPELINE_REVISION:
        raise PrimitiveGenerationError(
            f"{label}: generator summary pipeline_revision must be "
            f"{GENERATOR_PIPELINE_REVISION}"
        )
    gate = summary.get("dense_hand_object_gate")
    if not isinstance(gate, Mapping):
        raise PrimitiveGenerationError(
            f"{label}: generator summary has no dense_hand_object_gate"
        )
    for key, expected in _dense_gate_contract().items():
        actual = gate.get(key)
        if key in {
            "hand_surface_samples_per_set",
            "hand_surface_samples_per_link",
            "hand_surface_point_count",
            "object_surface_samples",
        } and (isinstance(actual, bool) or not isinstance(actual, int)):
            raise PrimitiveGenerationError(
                f"{label}: dense_hand_object_gate.{key} has an invalid type"
            )
        if key == "strict_less_than" and actual is not True:
            raise PrimitiveGenerationError(
                f"{label}: dense_hand_object_gate.strict_less_than must be true"
            )
        if key == "threshold" and isinstance(actual, bool):
            raise PrimitiveGenerationError(
                f"{label}: dense_hand_object_gate.threshold has an invalid type"
            )
        if actual != expected:
            raise PrimitiveGenerationError(
                f"{label}: dense_hand_object_gate.{key} must equal {expected!r}"
            )
    normalized = _dense_gate_contract()
    for key in ("sample_count", "evaluated_count", "feasible_count"):
        actual = gate.get(key)
        if isinstance(actual, bool) or not isinstance(actual, int):
            raise PrimitiveGenerationError(
                f"{label}: dense_hand_object_gate.{key} must be an integer"
            )
        normalized[key] = actual
    if normalized["sample_count"] != expected_sample_count:
        raise PrimitiveGenerationError(
            f"{label}: dense gate sample_count does not match generated samples"
        )
    if normalized["evaluated_count"] != expected_sample_count:
        raise PrimitiveGenerationError(
            f"{label}: every generated raw sample must be dense evaluated"
        )
    if not 0 <= normalized["feasible_count"] <= expected_sample_count:
        raise PrimitiveGenerationError(
            f"{label}: dense gate feasible_count is outside the sample range"
        )
    return normalized


def _read_and_validate_records(
    *,
    raw_directory: Path,
    mesh_path: Path,
    spec: PrimitiveSpec | GeneralMeshSpec,
    side: str,
    num_grasps: int,
    finger_count: int | None = None,
    finger_names: Sequence[str] = (),
    expected_object_scale: float | None = None,
    expected_seed: int | None = None,
    expected_n_iterations: int | None = None,
    require_finite: bool = False,
) -> list[tuple[Path, dict[str, Any]]]:
    expected_names = [
        target_filename(spec, side, index, finger_count) for index in range(num_grasps)
    ]
    paths = sorted(raw_directory.glob(target_glob(spec, side, finger_count)))
    if [path.name for path in paths] != expected_names:
        raise PrimitiveGenerationError(
            f"{spec.instance_name}/{side}: generated files are not the expected "
            f"continuous index range 0..{num_grasps - 1}"
        )
    records: list[tuple[Path, dict[str, Any]]] = []
    for expected_index, path in enumerate(paths):
        try:
            record = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise PrimitiveGenerationError(f"Cannot read generated record {path}: {exc}") from exc
        if not isinstance(record, dict):
            raise PrimitiveGenerationError(f"{path}: JSON root must be an object")
        if record.get("active_side") != side:
            raise PrimitiveGenerationError(f"{path}: active_side is not {side}")
        if int(record.get("sample_index", -1)) != expected_index:
            raise PrimitiveGenerationError(f"{path}: sample_index is not {expected_index}")
        object_record = record.get("object")
        if not isinstance(object_record, dict) or Path(
            str(object_record.get("mesh_path", ""))
        ).resolve() != mesh_path.resolve():
            raise PrimitiveGenerationError(f"{path}: object.mesh_path does not match {mesh_path}")
        if expected_object_scale is not None:
            actual_scale = _finite_number(
                object_record.get("scale"), field="object.scale", path=path
            )
            if actual_scale != expected_object_scale:
                raise PrimitiveGenerationError(
                    f"{path}: object.scale is {actual_scale}, expected "
                    f"{expected_object_scale}"
                )
        if expected_seed is not None and record.get("seed") != expected_seed:
            raise PrimitiveGenerationError(
                f"{path}: seed is {record.get('seed')!r}, expected {expected_seed}"
            )
        if expected_n_iterations is not None:
            optimization = record.get("optimization")
            if (
                not isinstance(optimization, dict)
                or optimization.get("iterations") != expected_n_iterations
            ):
                raise PrimitiveGenerationError(
                    f"{path}: optimization.iterations does not match "
                    f"{expected_n_iterations}"
                )
        if require_finite and record.get("finite") is not True:
            raise PrimitiveGenerationError(
                f"{path}: resume requires a finite, validator-ready raw sample"
            )
        if not isinstance(record.get("actuator"), list) or len(record["actuator"]) != 12:
            raise PrimitiveGenerationError(f"{path}: actuator must contain 12 values")
        if not isinstance(record.get("joint"), list) or len(record["joint"]) != 16:
            raise PrimitiveGenerationError(f"{path}: joint must contain 16 values")
        contact_ids = record.get("selected_contact_ids")
        expected_contacts = max(N_CONTACT, finger_count or 0)
        if (
            not isinstance(contact_ids, list)
            or len(contact_ids) != expected_contacts
            or len(set(contact_ids)) != expected_contacts
        ):
            raise PrimitiveGenerationError(
                f"{path}: contact IDs must contain {expected_contacts} unique values"
            )
        if finger_count is not None:
            participation = record.get("finger_participation")
            if (
                not isinstance(participation, dict)
                or participation.get("target_count") != finger_count
                or participation.get("actual_count") != finger_count
                or not isinstance(participation.get("finger_names"), list)
                or len(set(participation["finger_names"])) != finger_count
                or (
                    finger_names
                    and set(participation["finger_names"]) != set(finger_names)
                )
            ):
                raise PrimitiveGenerationError(
                    f"{path}: finger participation does not match f{finger_count}"
                )
        energy = record.get("energy")
        if not isinstance(energy, dict):
            raise PrimitiveGenerationError(f"{path}: energy must be an object")
        _finite_number(energy.get("initial_total"), field="energy.initial_total", path=path)
        _finite_number(energy.get("total"), field="energy.total", path=path)
        top_level_maximum = _finite_number(
            record.get("maximum_penetration"), field="maximum_penetration", path=path
        )
        _audit_dense_hand_object_record(
            record, path=path, top_level_maximum=top_level_maximum
        )
        if record.get("success") is not False or record.get("simulation_success") is not False:
            raise PrimitiveGenerationError(f"{path}: generator must only write unvalidated raw samples")
        validation = record.get("validation")
        if not isinstance(validation, dict) or validation.get("status") != "not_run":
            raise PrimitiveGenerationError(f"{path}: validation.status must be not_run")
        records.append((path, record))
    return records


def summarize_records(
    spec: PrimitiveSpec | GeneralMeshSpec,
    side: str,
    records: Sequence[dict[str, Any]],
    finger_count: int | None = None,
) -> dict[str, Any]:
    """Build one finite CSV row directly from published record content."""

    if not records:
        raise PrimitiveGenerationError(f"Cannot summarize empty {spec.instance_name}/{side}")
    initial = [float(record["energy"]["initial_total"]) for record in records]
    final = [float(record["energy"]["total"]) for record in records]
    penetration = [float(record["maximum_penetration"]) for record in records]
    numeric = initial + final + penetration
    if not all(math.isfinite(value) for value in numeric):
        raise PrimitiveGenerationError(f"{spec.instance_name}/{side}: summary input is non-finite")
    return {
        "shape": spec.shape,
        "size": spec.size,
        "side": side,
        "finger_count": "" if finger_count is None else finger_count,
        "sample_count": len(records),
        "finite_sample_count": sum(record.get("finite") is True for record in records),
        "mean_initial_energy": fmean(initial),
        "mean_final_energy": fmean(final),
        "energy_decreased_count": sum(
            final_value < initial_value
            for initial_value, final_value in zip(initial, final)
        ),
        "maximum_penetration_mean": fmean(penetration),
        "maximum_penetration_min": min(penetration),
        "maximum_penetration_median": median(penetration),
    }


def _parse_generator_summary(stdout: str, *, spec: PrimitiveSpec, side: str) -> dict[str, Any]:
    try:
        summary = json.loads(stdout, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise PrimitiveGenerationError(
            f"{spec.instance_name}/{side}: generator stdout is not a JSON summary\n{stdout}"
        ) from exc
    if not isinstance(summary, dict):
        raise PrimitiveGenerationError(f"{spec.instance_name}/{side}: invalid generator summary")
    return summary


def _stage_one(
    *,
    task: GenerationTask,
    mesh_path: Path,
    staging_root: Path,
    n_iterations: int,
    device: str,
    seed: int,
    strict_resume: bool = False,
) -> StagedResult:
    spec = task.spec
    side = task.side
    finger_directory = (
        f"f{task.finger_count}" if task.finger_count is not None else "legacy"
    )
    staging_output = (
        staging_root / spec.shape / spec.instance_name / side / finger_directory
    )
    command = build_generator_command(
        spec=spec,
        side=side,
        mesh_path=mesh_path,
        staging_output=staging_output,
        num_grasps=task.num_grasps,
        n_iterations=n_iterations,
        device=device,
        seed=seed,
        object_scale=(
            spec.object_scale if isinstance(spec, GeneralMeshSpec) else OBJECT_SCALE
        ),
        finger_count=task.finger_count,
        finger_names=task.finger_names,
    )
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise PrimitiveGenerationError(
            f"{spec.instance_name}/{side}: generator exited with {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    generator_summary = _parse_generator_summary(
        completed.stdout, spec=spec, side=side
    )
    if generator_summary.get("side_mode") != side or int(
        generator_summary.get("num_output_samples", -1)
    ) != task.num_grasps:
        raise PrimitiveGenerationError(
            f"{spec.instance_name}/{side}: generator summary count/side mismatch"
        )
    summary_dense_gate = _audit_dense_gate_summary(
        generator_summary,
        expected_sample_count=task.num_grasps,
        label=f"{spec.instance_name}/{side}",
    )
    raw_directory = staging_output / f"{side}_single" / "raw"
    # Generic repository objects all use the leaf name ``decomposed.obj``.
    # Give their staged records stable object IDs before auditing/publishing.
    source_prefix = "decomposed" if isinstance(spec, GeneralMeshSpec) else spec.instance_name
    for index, source in enumerate(
        sorted(raw_directory.glob(f"{source_prefix}_{side}_*.json"))
    ):
        destination = raw_directory / target_filename(
            spec, side, index, task.finger_count
        )
        if source != destination:
            source.rename(destination)
    staged_records = _read_and_validate_records(
        raw_directory=raw_directory,
        mesh_path=mesh_path,
        spec=spec,
        side=side,
        num_grasps=task.num_grasps,
        finger_count=task.finger_count,
        finger_names=task.finger_names,
        expected_object_scale=(
            spec.object_scale if isinstance(spec, GeneralMeshSpec) else OBJECT_SCALE
        ),
        expected_seed=seed if strict_resume else None,
        expected_n_iterations=n_iterations if strict_resume else None,
        require_finite=strict_resume,
    )

    records = [record for _, record in staged_records]
    record_dense_gate = _dense_gate_summary_from_records(records)
    if summary_dense_gate != record_dense_gate:
        raise PrimitiveGenerationError(
            f"{spec.instance_name}/{side}: generator dense gate summary "
            "disagrees with raw records"
        )
    return StagedResult(
        task=task,
        source_paths=tuple(path for path, _ in staged_records),
        row=summarize_records(spec, side, records, task.finger_count),
        generator_call={
            "instance_name": spec.instance_name,
            "side": side,
            "finger_count": task.finger_count,
            "finger_names": list(task.finger_names),
            "num_output_samples": generator_summary["num_output_samples"],
            "object_scale": (
                spec.object_scale if isinstance(spec, GeneralMeshSpec) else OBJECT_SCALE
            ),
            "stratified": False,
            "pipeline_revision": GENERATOR_PIPELINE_REVISION,
            "dense_hand_object_gate": record_dense_gate,
        },
    )


def _stage_stratified_object(
    *,
    tasks: Sequence[GenerationTask],
    mesh_path: Path,
    staging_root: Path,
    n_iterations: int,
    device: str,
    seed: int,
) -> tuple[StagedResult, ...]:
    """Generate all pending strata for one mesh in one resident process."""

    if not tasks:
        raise PrimitiveGenerationError("A stratified object run requires tasks")
    spec = tasks[0].spec
    if any(task.spec != spec for task in tasks):
        raise PrimitiveGenerationError(
            "A stratified object run cannot contain multiple object specs"
        )
    if any(task.finger_count is None or not task.finger_names for task in tasks):
        raise PrimitiveGenerationError(
            "Stratified batching requires exact finger counts and masks"
        )
    plan_groups: list[dict[str, Any]] = []
    staging_outputs: dict[tuple[str, int], Path] = {}
    for task in tasks:
        key = (task.side, int(task.finger_count))
        if key in staging_outputs:
            raise PrimitiveGenerationError(
                f"Duplicate stratified task for {spec.instance_name}/{key}"
            )
        staging_output = (
            staging_root
            / spec.shape
            / spec.instance_name
            / task.side
            / f"f{task.finger_count}"
        )
        staging_outputs[key] = staging_output
        plan_groups.append(
            {
                "side": task.side,
                "finger_count": task.finger_count,
                "finger_names": list(task.finger_names),
                "num_grasps": task.num_grasps,
                "output": str(staging_output.resolve()),
            }
        )
    plan_path = (
        staging_root / spec.shape / spec.instance_name / "stratified_plan.json"
    )
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps({"groups": plan_groups}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    object_scale = (
        spec.object_scale if isinstance(spec, GeneralMeshSpec) else OBJECT_SCALE
    )
    command = [
        sys.executable,
        str(STRATIFIED_GENERATOR),
        "--mesh-path",
        str(mesh_path),
        "--plan",
        str(plan_path),
        "--n-iterations",
        str(n_iterations),
        "--seed",
        str(seed),
        "--device",
        device,
        "--object-scale",
        str(object_scale),
        "--surface-samples",
        str(SURFACE_SAMPLES),
        "--batch-size",
        str(STRATIFIED_BATCH_SIZE),
        "--overwrite",
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise PrimitiveGenerationError(
            f"{spec.instance_name}: stratified generator exited with "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    summary = _parse_generator_summary(
        completed.stdout, spec=spec, side="stratified"
    )
    if (
        summary.get("passed") is not True
        or summary.get("side_mode") != "stratified"
        or int(summary.get("group_count", -1)) != len(tasks)
        or int(summary.get("num_output_samples", -1))
        != sum(task.num_grasps for task in tasks)
        or float(summary.get("object_scale", float("nan"))) != object_scale
    ):
        raise PrimitiveGenerationError(
            f"{spec.instance_name}: stratified generator summary mismatch"
        )
    summary_dense_gate = _audit_dense_gate_summary(
        summary,
        expected_sample_count=sum(task.num_grasps for task in tasks),
        label=f"{spec.instance_name}/stratified",
    )

    results: list[StagedResult] = []
    source_prefix = (
        "decomposed" if isinstance(spec, GeneralMeshSpec) else spec.instance_name
    )
    for task in tasks:
        staging_output = staging_outputs[(task.side, int(task.finger_count))]
        raw_directory = staging_output / f"{task.side}_single" / "raw"
        sources = sorted(
            raw_directory.glob(f"{source_prefix}_{task.side}_*.json")
        )
        for index, source in enumerate(sources):
            destination = raw_directory / target_filename(
                spec, task.side, index, task.finger_count
            )
            if source != destination:
                source.rename(destination)
        staged_records = _read_and_validate_records(
            raw_directory=raw_directory,
            mesh_path=mesh_path,
            spec=spec,
            side=task.side,
            num_grasps=task.num_grasps,
            finger_count=task.finger_count,
            finger_names=task.finger_names,
            expected_object_scale=object_scale,
            expected_seed=seed,
            expected_n_iterations=n_iterations,
            require_finite=True,
        )
        records = [record for _, record in staged_records]
        record_dense_gate = _dense_gate_summary_from_records(records)
        results.append(
            StagedResult(
                task=task,
                source_paths=tuple(path for path, _ in staged_records),
                row=summarize_records(
                    spec, task.side, records, task.finger_count
                ),
                generator_call={
                    "instance_name": spec.instance_name,
                    "side": task.side,
                    "finger_count": task.finger_count,
                    "finger_names": list(task.finger_names),
                    "num_output_samples": len(records),
                    "object_scale": object_scale,
                    "stratified": True,
                    "stratified_batch_count": summary.get("batch_count"),
                    "stratified_batch_size": summary.get("batch_size"),
                    "pipeline_revision": GENERATOR_PIPELINE_REVISION,
                    "dense_hand_object_gate": record_dense_gate,
                },
            )
        )
    combined_dense_gate = _dense_gate_contract()
    combined_dense_gate.update(
        {
            "sample_count": sum(
                result.generator_call["dense_hand_object_gate"]["sample_count"]
                for result in results
            ),
            "evaluated_count": sum(
                result.generator_call["dense_hand_object_gate"]["evaluated_count"]
                for result in results
            ),
            "feasible_count": sum(
                result.generator_call["dense_hand_object_gate"]["feasible_count"]
                for result in results
            ),
        }
    )
    if summary_dense_gate != combined_dense_gate:
        raise PrimitiveGenerationError(
            f"{spec.instance_name}: stratified dense gate summary disagrees "
            "with staged raw records"
        )
    return tuple(results)


def _publish_one(
    result: StagedResult,
    *,
    output_root: Path,
    overwrite: bool,
) -> None:
    """Publish one validated group without rewriting any JSON content."""

    spec = result.task.spec
    side = result.task.side
    destination = target_directory(output_root, spec, side)
    destination.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        destination.glob(target_glob(spec, side, result.task.finger_count))
    )
    if existing and not overwrite:
        raise PrimitiveGenerationError(
            f"Output already exists for {spec.instance_name}/{side}; pass --overwrite"
        )
    if overwrite:
        for path in existing:
            path.unlink()
    for source in result.source_paths:
        destination_path = destination / source.name
        source.replace(destination_path)


def _remove_validation_outputs(task: GenerationTask, output_root: Path) -> None:
    """Discard validation decisions made from a raw group being regenerated."""

    raw_directory = target_directory(output_root, task.spec, task.side)
    pattern = target_glob(task.spec, task.side, task.finger_count)
    for status in ("valid", "failed"):
        directory = raw_directory.parent / status
        for path in directory.glob(pattern):
            path.unlink()


def _audit_published_group(
    *,
    task: GenerationTask,
    mesh_path: Path,
    output_root: Path,
    n_iterations: int,
    seed: int,
) -> StagedResult:
    """Return a reusable group only after a strict current-invocation audit."""

    records_with_paths = _read_and_validate_records(
        raw_directory=target_directory(output_root, task.spec, task.side),
        mesh_path=mesh_path,
        spec=task.spec,
        side=task.side,
        num_grasps=task.num_grasps,
        finger_count=task.finger_count,
        finger_names=task.finger_names,
        expected_object_scale=(
            task.spec.object_scale
            if isinstance(task.spec, GeneralMeshSpec)
            else OBJECT_SCALE
        ),
        expected_seed=seed,
        expected_n_iterations=n_iterations,
        require_finite=True,
    )
    records = [record for _, record in records_with_paths]
    dense_gate = _dense_gate_summary_from_records(records)
    return StagedResult(
        task=task,
        source_paths=tuple(path for path, _ in records_with_paths),
        row=summarize_records(task.spec, task.side, records, task.finger_count),
        generator_call={
            "instance_name": task.spec.instance_name,
            "side": task.side,
            "finger_count": task.finger_count,
            "finger_names": list(task.finger_names),
            "num_output_samples": len(records),
            "object_scale": (
                task.spec.object_scale
                if isinstance(task.spec, GeneralMeshSpec)
                else OBJECT_SCALE
            ),
            "pipeline_revision": GENERATOR_PIPELINE_REVISION,
            "dense_hand_object_gate": dense_gate,
        },
    )


def _preflight_outputs(
    output_root: Path,
    tasks: Sequence[GenerationTask],
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    conflicts: list[str] = []
    for task in tasks:
        directory = target_directory(output_root, task.spec, task.side)
        if any(directory.glob(target_glob(task.spec, task.side, task.finger_count))):
            conflicts.append(
                f"{task.spec.instance_name}/{task.side}/f{task.finger_count}"
            )
    if conflicts:
        raise PrimitiveGenerationError(
            "Output already exists for " + ", ".join(conflicts) + "; pass --overwrite"
        )


def write_csv_summary(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Atomically write the requested summary with a stable schema/order."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json_summary(path: Path, report: Mapping[str, Any]) -> None:
    """Atomically persist the invocation settings that CSV rows cannot prove."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    primitive_specs = selected_specs(args.shapes)
    general_specs = (
        select_general_meshes(
            discover_general_meshes(args.general_mesh_root),
            getattr(args, "general_mesh_ids", None),
        )
        if args.include_general_meshes
        else ()
    )
    if args.include_general_meshes and not general_specs:
        raise PrimitiveGenerationError(
            f"No general meshes found below {args.general_mesh_root}/<object>/coacd/decomposed.obj"
        )
    specs = (*primitive_specs, *general_specs)
    sides = requested_sides(args.side)
    tasks = plan_generation(
        specs,
        sides,
        num_grasps=args.num_grasps,
        target_total=args.target_total,
        finger_targets=(
            dict(zip(args.finger_counts, args.finger_targets))
            if args.finger_targets is not None
            else None
        ),
        finger_counts=args.finger_counts,
        complementary_side_fingers=args.complementary_side_fingers,
        seed=args.seed,
    )
    mesh_root = Path(args.mesh_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    summary_csv = output_root / DEFAULT_SUMMARY_NAME
    summary_json = output_root / DEFAULT_JSON_SUMMARY_NAME
    resume = bool(getattr(args, "resume", False))
    stratified_batching = bool(getattr(args, "stratified_batching", False))
    if not resume:
        _preflight_outputs(output_root, tasks, overwrite=args.overwrite)
    mesh_reports = build_dataset(mesh_root, shapes=args.shapes, overwrite=False)
    report_by_name = {report["instance_name"]: report for report in mesh_reports}
    report_by_name.update(
        {spec.instance_name: {"path": str(spec.path)} for spec in general_specs}
    )

    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    generator_calls: list[dict[str, Any]] = []
    reused_group_count = 0
    generated_group_count = 0
    if resume:
        pending: list[tuple[int, GenerationTask]] = []
        for index, task in enumerate(tasks):
            mesh_path = Path(report_by_name[task.spec.instance_name]["path"])
            try:
                _audit_published_group(
                    task=task,
                    mesh_path=mesh_path,
                    output_root=output_root,
                    n_iterations=args.n_iterations,
                    seed=args.seed,
                )
            except Exception:
                pending.append((index, task))
            else:
                reused_group_count += 1
        print(
            f"[resume] reusable_groups={reused_group_count} "
            f"regenerate_groups={len(pending)}",
            flush=True,
        )

        if pending:
            # An older all-groups marker must not survive a partial repair run.
            summary_csv.unlink(missing_ok=True)
            summary_csv.with_suffix(summary_csv.suffix + ".tmp").unlink(missing_ok=True)
            summary_json.unlink(missing_ok=True)
            summary_json.with_suffix(summary_json.suffix + ".tmp").unlink(
                missing_ok=True
            )
            staging_parent = output_root / ".staging"
            staging_parent.mkdir(parents=True, exist_ok=True)
            try:
                with tempfile.TemporaryDirectory(
                    prefix="primitive_generation_", dir=staging_parent
                ) as temporary_directory:
                    staging_root = Path(temporary_directory)

                    if stratified_batching:
                        grouped_pending: dict[
                            str, list[tuple[int, GenerationTask]]
                        ] = {}
                        for indexed_task in pending:
                            grouped_pending.setdefault(
                                indexed_task[1].spec.instance_name, []
                            ).append(indexed_task)
                        work_items = [
                            tuple(values) for values in grouped_pending.values()
                        ]
                    else:
                        work_items = [(indexed_task,) for indexed_task in pending]

                    def execute_and_publish(
                        indexed_tasks: tuple[tuple[int, GenerationTask], ...],
                    ) -> tuple[int, ...]:
                        object_tasks = [task for _, task in indexed_tasks]
                        first_task = object_tasks[0]
                        mesh_path = Path(
                            report_by_name[first_task.spec.instance_name]["path"]
                        )
                        if stratified_batching:
                            results = _stage_stratified_object(
                                tasks=object_tasks,
                                mesh_path=mesh_path,
                                staging_root=staging_root,
                                n_iterations=args.n_iterations,
                                device=args.device,
                                seed=args.seed,
                            )
                        else:
                            results = (
                                _stage_one(
                                    task=first_task,
                                    mesh_path=mesh_path,
                                    staging_root=staging_root,
                                    n_iterations=args.n_iterations,
                                    device=args.device,
                                    seed=args.seed,
                                    strict_resume=True,
                                ),
                            )
                        for result in results:
                            task = result.task
                            _remove_validation_outputs(task, output_root)
                            _publish_one(
                                result, output_root=output_root, overwrite=True
                            )
                            print(
                                f"[resume] committed {task.spec.instance_name}/"
                                f"{task.side}/f{task.finger_count} "
                                f"samples={task.num_grasps}",
                                flush=True,
                            )
                        return tuple(index for index, _ in indexed_tasks)

                    if args.jobs == 1:
                        for work_item in work_items:
                            execute_and_publish(work_item)
                    else:
                        executor = ThreadPoolExecutor(
                            max_workers=min(args.jobs, len(work_items))
                        )
                        futures = [
                            executor.submit(execute_and_publish, work_item)
                            for work_item in work_items
                        ]
                        try:
                            for future in as_completed(futures):
                                future.result()
                        except BaseException:
                            for future in futures:
                                future.cancel()
                            executor.shutdown(wait=True, cancel_futures=True)
                            raise
                        else:
                            executor.shutdown(wait=True)
            finally:
                if staging_parent.is_dir() and not any(staging_parent.iterdir()):
                    staging_parent.rmdir()
            generated_group_count = len(pending)

        # Re-audit every expected group before publishing the all-groups summary.
        completed_results = []
        for task in tasks:
            mesh_path = Path(report_by_name[task.spec.instance_name]["path"])
            completed_results.append(
                _audit_published_group(
                    task=task,
                    mesh_path=mesh_path,
                    output_root=output_root,
                    n_iterations=args.n_iterations,
                    seed=args.seed,
                )
            )
        rows.extend(result.row for result in completed_results)
        for result in completed_results:
            result.generator_call["stratified"] = stratified_batching
            generator_calls.append(result.generator_call)
    else:
        staging_parent = output_root / ".staging"
        staging_parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                prefix="primitive_generation_", dir=staging_parent
            ) as temporary_directory:
                staging_root = Path(temporary_directory)

                def execute(task: GenerationTask) -> StagedResult:
                    mesh_path = Path(report_by_name[task.spec.instance_name]["path"])
                    return _stage_one(
                        task=task,
                        mesh_path=mesh_path,
                        staging_root=staging_root,
                        n_iterations=args.n_iterations,
                        device=args.device,
                        seed=args.seed,
                    )

                if args.jobs == 1:
                    staged_results = [execute(task) for task in tasks]
                else:
                    with ThreadPoolExecutor(
                        max_workers=min(args.jobs, len(tasks))
                    ) as executor:
                        staged_results = list(executor.map(execute, tasks))
                # Default/overwrite generation remains whole-batch transactional:
                # no group is replaced until every concurrent task passes.
                for result in staged_results:
                    _publish_one(
                        result, output_root=output_root, overwrite=args.overwrite
                    )
                    rows.append(result.row)
                    generator_calls.append(result.generator_call)
        finally:
            if staging_parent.is_dir() and not any(staging_parent.iterdir()):
                staging_parent.rmdir()
        generated_group_count = len(tasks)

    write_csv_summary(summary_csv, rows)
    object_scale_by_id = {
        spec.instance_name: (
            spec.object_scale if isinstance(spec, GeneralMeshSpec) else OBJECT_SCALE
        )
        for spec in specs
    }
    aggregate_dense_gate = _dense_gate_contract()
    aggregate_dense_gate.update(
        {
            "sample_count": sum(
                call["dense_hand_object_gate"]["sample_count"]
                for call in generator_calls
            ),
            "evaluated_count": sum(
                call["dense_hand_object_gate"]["evaluated_count"]
                for call in generator_calls
            ),
            "feasible_count": sum(
                call["dense_hand_object_gate"]["feasible_count"]
                for call in generator_calls
            ),
        }
    )
    total_samples = sum(int(row["sample_count"]) for row in rows)
    if (
        any(
            call.get("pipeline_revision") != GENERATOR_PIPELINE_REVISION
            for call in generator_calls
        )
        or aggregate_dense_gate["sample_count"] != total_samples
        or aggregate_dense_gate["evaluated_count"] != total_samples
    ):
        raise PrimitiveGenerationError(
            "Generation summary cannot prove that every raw sample passed the "
            "formal v5 dense evaluation contract"
        )
    aggregate_dense_gate["all_samples_evaluated"] = True
    report = {
        "passed": True,
        "pipeline_revision": GENERATOR_PIPELINE_REVISION,
        "mesh_root": str(mesh_root),
        "output_root": str(output_root),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "shapes": [shape for shape in SHAPES if shape in set(args.shapes)],
        "general_objects": [spec.instance_name for spec in general_specs],
        "side_mode": args.side,
        "settings": {
            "num_grasps": (
                args.num_grasps
                if args.target_total is None and args.finger_targets is None
                else None
            ),
            "target_total": args.target_total,
            "finger_targets": (
                dict(zip(args.finger_counts, args.finger_targets))
                if args.finger_targets is not None
                else None
            ),
            "finger_counts": args.finger_counts,
            "complementary_side_fingers": args.complementary_side_fingers,
            "per_group_min": min(task.num_grasps for task in tasks),
            "per_group_max": max(task.num_grasps for task in tasks),
            "batch_size": (
                STRATIFIED_BATCH_SIZE if stratified_batching else BATCH_SIZE
            ),
            "legacy_batch_size": BATCH_SIZE,
            "stratified_batching": stratified_batching,
            "stratified_batch_size": (
                STRATIFIED_BATCH_SIZE if stratified_batching else None
            ),
            "n_contact": N_CONTACT,
            "n_iterations": args.n_iterations,
            "surface_samples": SURFACE_SAMPLES,
            "object_scale": OBJECT_SCALE,
            "general_object_scales": (
                sorted({spec.object_scale for spec in general_specs})
                if general_specs
                else None
            ),
            "object_scale_by_id": object_scale_by_id,
            "device": args.device,
            "seed": args.seed,
            "jobs": args.jobs,
            "resume": resume,
            "generator_pipeline_revision": GENERATOR_PIPELINE_REVISION,
            "dense_hand_object_gate": _dense_gate_contract(),
        },
        "instance_side_runs": len(rows),
        "reused_group_count": reused_group_count,
        "generated_group_count": generated_group_count,
        "total_samples": total_samples,
        "total_finite_samples": sum(
            int(row["finite_sample_count"]) for row in rows
        ),
        "rows": rows,
        "generator_calls": generator_calls,
        "dense_hand_object_gate": aggregate_dense_gate,
        "physical_validation_run": False,
    }
    write_json_summary(summary_json, report)
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--side", choices=("front", "back", "both"), default="both")
    parser.add_argument(
        "--finger-counts",
        nargs="+",
        type=int,
        choices=range(1, 6),
        help="stratify every object/side over these distinct participating-finger counts",
    )
    parser.add_argument(
        "--complementary-side-fingers",
        action="store_true",
        help=(
            "choose deterministic disjoint masks for front fk and back f(5-k) "
            "on every object"
        ),
    )
    count_group = parser.add_mutually_exclusive_group()
    count_group.add_argument("--num-grasps", type=int, default=DEFAULT_NUM_GRASPS)
    count_group.add_argument("--target-total", type=int)
    count_group.add_argument(
        "--finger-targets",
        nargs="+",
        type=int,
        help=(
            "exact raw totals aligned with --finger-counts; permits adaptive "
            "per-stratum attempts"
        ),
    )
    parser.add_argument("--n-iterations", type=int, default=DEFAULT_N_ITERATIONS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help=(
            "strictly reuse complete published instance/side/finger groups and "
            "regenerate only missing or damaged groups"
        ),
    )
    parser.add_argument(
        "--stratified-batching",
        action="store_true",
        help=(
            "in resume mode, generate all pending exact finger strata for one "
            "mesh in a resident row-policy process"
        ),
    )
    parser.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument(
        "--include-general-meshes",
        action="store_true",
        help="also include <general-mesh-root>/*/coacd/decomposed.obj objects",
    )
    parser.add_argument(
        "--general-mesh-root", type=Path, default=DEFAULT_GENERAL_MESH_ROOT
    )
    parser.add_argument(
        "--general-mesh-ids",
        nargs="+",
        help="explicit general-mesh object IDs to include (default: all discovered)",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_GRASP_ROOT)
    args = parser.parse_args(argv)
    if args.num_grasps <= 0:
        parser.error("num-grasps must be positive")
    if args.target_total is not None and args.target_total <= 0:
        parser.error("target-total must be positive")
    if args.finger_targets is not None:
        if not args.finger_counts:
            parser.error("finger-targets requires --finger-counts")
        if len(args.finger_targets) != len(args.finger_counts):
            parser.error("finger-targets must align one-for-one with finger-counts")
        if any(value <= 0 for value in args.finger_targets):
            parser.error("finger-targets values must be positive")
    if args.n_iterations <= 0:
        parser.error("n-iterations must be positive")
    if args.jobs <= 0:
        parser.error("jobs must be positive")
    if args.general_mesh_ids is not None and not args.include_general_meshes:
        parser.error("general-mesh-ids requires --include-general-meshes")
    if args.general_mesh_ids is not None and len(args.general_mesh_ids) != len(
        set(args.general_mesh_ids)
    ):
        parser.error("general-mesh-ids must be unique")
    if args.stratified_batching and (
        not args.resume or not args.finger_counts or not args.complementary_side_fingers
    ):
        parser.error(
            "stratified-batching requires --resume, explicit --finger-counts, "
            "and --complementary-side-fingers"
        )
    if args.complementary_side_fingers and (
        args.side != "both" or not args.finger_counts
    ):
        parser.error(
            "complementary-side-fingers requires --side both and explicit "
            "--finger-counts"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "physical_validation_run": False,
                },
                indent=2,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps({"passed": True, **report}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
