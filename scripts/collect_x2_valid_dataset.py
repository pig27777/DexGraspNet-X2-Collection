#!/usr/bin/env python3
"""Collect and PhysX-validate a balanced 5k X2 multi-finger dataset.

Generation and validation are split into restartable attempts.  An attempt is
never counted until its batch generation summary and validation summary both
exist.  The collector continues sampling deficient 1..5-finger strata and
materializes an exact, deterministic final valid set only after every stratum
has reached its target.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_x2_primitive_dataset import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT as PRIMITIVE_MESH_ROOT,
    PRIMITIVE_SPECS,
)
from grasp_generation.x2_isaac_validation import (  # noqa: E402
    FORMAL_ACTUATOR_ARMATURE,
    FORMAL_ACTUATOR_DAMPING,
    FORMAL_ACTUATOR_STIFFNESS,
    PROTOCOL_REVISION as ISAAC_VALIDATION_PROTOCOL_REVISION,
)

GENERATOR = PROJECT_ROOT / "scripts" / "generate_x2_primitive_dataset.py"
VALIDATOR = PROJECT_ROOT / "scripts" / "validate_x2_primitive_dataset.py"
GENERATION_SUMMARY_NAME = "generation_summary.json"
STRATIFIED_BATCH_SIZE = 64
FORMAL_PRIMITIVE_OBJECT_SCALE = 1.0
FORMAL_GENERAL_OBJECT_SCALE = 1.0
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "x2_valid_5000"
FINGER_COUNTS = (1, 2, 3, 4, 5)
FINGER_NAMES = ("index", "middle", "ring", "little", "thumb")
EXPECTED_GENERAL_MESHES = 88
FORMAL_GENERAL_MESH_IDS = tuple(
    f"{value:03d}" for value in range(0, EXPECTED_GENERAL_MESHES, 3)
)
FORMAL_GENERAL_MESH_COUNT = 30
SIDES = ("front", "back")
COLLECTION_PROTOCOL_REVISION = "x2_balanced_complementary_30mesh_5000_v6"
GENERATOR_PIPELINE_REVISION = "x2_mesh_grasp_unselected_finger_side_v6"
VALIDATION_BACKEND = "isaac_sim_physx"
VALIDATION_PROTOCOL_REVISION = ISAAC_VALIDATION_PROTOCOL_REVISION
VALIDATION_CRITERION = "dexgraspnet-contact"
FORMAL_TARGET_VALID = 5000
FORMAL_PER_SIDE_FINGER_TARGET = 500
REQUIRED_SIM_STEPS = 100
REQUIRED_SUBSTEPS = 2
REQUIRED_PRECLOSE_PHYSICS_STEPS = 0
FORMAL_RAW_PENETRATION_CAP = 0.001
FORMAL_CLOSING_TARGET_PENETRATION_CAP = 0.0015
FORMAL_CLOSING_ALPHAS = tuple(0.5**index for index in range(9))
DENSE_HAND_SURFACE_SAMPLES_PER_SET = 256
DENSE_HAND_SURFACE_SAMPLES_PER_LINK = 768
DENSE_HAND_SURFACE_POINT_COUNT = 13056
DENSE_OBJECT_SURFACE_SAMPLES = 8192
EXPECTED_ORIENTATIONS = (
    "identity",
    "z_180",
    "z_pos_90",
    "z_neg_90",
    "x_pos_90",
    "x_neg_90",
)
EXPECTED_GRAVITY_VECTORS = (
    (0.0, -9.8, 0.0),
    (0.0, 9.8, 0.0),
    (-9.8, 0.0, 0.0),
    (9.8, 0.0, 0.0),
    (0.0, 0.0, 9.8),
    (0.0, 0.0, -9.8),
)
ATTEMPT_COMPLETION_NAME = "complete.json"


class ValidDatasetError(RuntimeError):
    """Raised when collection state or validated records are inconsistent."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValidDatasetError(f"{path} contains non-finite JSON constant {value}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    except ValidDatasetError:
        raise
    except Exception as exc:
        raise ValidDatasetError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidDatasetError(f"{path} must contain a JSON object")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_number(value: Any, expected: float, *, atol: float = 1.0e-9) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and math.isclose(
        numeric, float(expected), rel_tol=0.0, abs_tol=atol
    )


def _audit_v5_dense_gate(
    path: Path,
    payload: Mapping[str, Any],
    *,
    require_feasible: bool = True,
) -> bool:
    if payload.get("pipeline_revision") != GENERATOR_PIPELINE_REVISION:
        raise ValidDatasetError(f"{path}: generator pipeline revision is not formal v6")
    gate = payload.get("hand_object_penetration")
    if not isinstance(gate, dict):
        raise ValidDatasetError(f"{path}: v5 dense hand-object diagnostic is missing")
    if (
        gate.get("evaluation_mode") != "dense_bidirectional"
        or gate.get("evaluated") is not True
        or gate.get("hand_surface_samples_per_set")
        != DENSE_HAND_SURFACE_SAMPLES_PER_SET
        or gate.get("hand_surface_samples_per_link")
        != DENSE_HAND_SURFACE_SAMPLES_PER_LINK
        or gate.get("hand_surface_point_count") != DENSE_HAND_SURFACE_POINT_COUNT
        or gate.get("object_surface_samples") != DENSE_OBJECT_SURFACE_SAMPLES
        or not _same_number(gate.get("threshold"), FORMAL_RAW_PENETRATION_CAP)
    ):
        raise ValidDatasetError(f"{path}: v5 dense hand-object sampling contract drifted")
    numeric: dict[str, float] = {}
    for key in (
        "forward_total_penetration",
        "forward_maximum_penetration",
        "reverse_total_penetration",
        "reverse_maximum_penetration",
        "total_penetration",
        "maximum_penetration",
    ):
        value = gate.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValidDatasetError(f"{path}: dense diagnostic {key} is invalid")
        numeric[key] = float(value)
    expected_maximum = max(
        numeric["forward_maximum_penetration"],
        numeric["reverse_maximum_penetration"],
    )
    expected_total = (
        numeric["forward_total_penetration"]
        + numeric["reverse_total_penetration"]
    )
    expected_feasible = expected_maximum < FORMAL_RAW_PENETRATION_CAP
    if (
        not math.isclose(
            numeric["maximum_penetration"],
            expected_maximum,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            numeric["total_penetration"],
            expected_total,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or gate.get("feasible") is not expected_feasible
        or not _same_number(
            payload.get("maximum_penetration"), expected_maximum, atol=1.0e-12
        )
    ):
        raise ValidDatasetError(f"{path}: dense hand-object diagnostic is inconsistent")
    if require_feasible and not expected_feasible:
        raise ValidDatasetError(f"{path}: selected final record failed the v5 dense gate")
    return expected_feasible


def _finger_count(
    path: Path,
    payload: dict[str, Any],
    *,
    expected_sim_steps: int = REQUIRED_SIM_STEPS,
) -> int:
    _audit_v5_dense_gate(path, payload)
    validation = payload.get("validation")
    if not isinstance(validation, dict) or validation.get("status") != "passed":
        raise ValidDatasetError(f"{path}: selected final candidate is not passed")
    if payload.get("success") is not True or payload.get("simulation_success") is not True:
        raise ValidDatasetError(f"{path}: passed record has false success flags")
    if validation.get("backend") != VALIDATION_BACKEND:
        raise ValidDatasetError(f"{path}: validation backend is not {VALIDATION_BACKEND}")
    if validation.get("protocol_revision") != VALIDATION_PROTOCOL_REVISION:
        raise ValidDatasetError(f"{path}: validation protocol revision is stale")
    if validation.get("criterion") != VALIDATION_CRITERION:
        raise ValidDatasetError(f"{path}: validation criterion is not {VALIDATION_CRITERION}")
    thresholds = validation.get("thresholds")
    expected_thresholds = {
        "maximum_penetration_m": FORMAL_RAW_PENETRATION_CAP,
        "maximum_object_displacement_m": 0.1,
        "minimum_final_contact_force_n": 0.0,
        "maximum_active_joint_error_rad": 0.1,
        "maximum_newton_mimic_error_rad": 0.01,
    }
    if not isinstance(thresholds, dict) or any(
        not _same_number(thresholds.get(key), value)
        for key, value in expected_thresholds.items()
    ):
        raise ValidDatasetError(f"{path}: validation thresholds are not formal")
    source_value = validation.get("source_raw")
    if not isinstance(source_value, str):
        raise ValidDatasetError(f"{path}: validation.source_raw is missing")
    source_raw = Path(source_value).expanduser().resolve()
    if not source_raw.is_file():
        raise ValidDatasetError(f"{path}: validation source raw file is missing")
    if validation.get("source_sha256") != _file_sha256(source_raw):
        raise ValidDatasetError(f"{path}: validation source hash is stale")
    runtime = validation.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("simulation_steps") != expected_sim_steps
        or runtime.get("substeps") != REQUIRED_SUBSTEPS
        or runtime.get("physics_step_count")
        != expected_sim_steps * REQUIRED_SUBSTEPS
        or not isinstance(runtime.get("device"), str)
        or not runtime["device"].startswith("cuda")
    ):
        raise ValidDatasetError(
            f"{path}: validation runtime does not prove {expected_sim_steps} simulation steps"
        )
    actuator_drive = runtime.get("actuator_drive")
    if (
        not isinstance(actuator_drive, dict)
        or not _same_number(
            actuator_drive.get("stiffness_n_m_per_rad"),
            FORMAL_ACTUATOR_STIFFNESS,
        )
        or not _same_number(
            actuator_drive.get("damping_n_m_s_per_rad"),
            FORMAL_ACTUATOR_DAMPING,
        )
        or not _same_number(
            actuator_drive.get("armature_kg_m2"), FORMAL_ACTUATOR_ARMATURE
        )
        or actuator_drive.get("stiffness_source") != "cli_override"
        or actuator_drive.get("damping_source") != "cli_override"
        or actuator_drive.get("armature_source") != "cli_override"
        or actuator_drive.get("effort_limit_source") != "hand_usd"
        or actuator_drive.get("velocity_limit_source") != "hand_usd"
    ):
        raise ValidDatasetError(f"{path}: formal actuator-drive proof is missing")
    physx_solver = runtime.get("physx_solver")
    if (
        not isinstance(physx_solver, dict)
        or physx_solver.get("solver_type") != 1
        or physx_solver.get("external_forces_every_iteration") is not True
        or physx_solver.get("solve_articulation_contact_last") is not False
    ):
        raise ValidDatasetError(f"{path}: formal PhysX solver proof is missing")
    preclose = runtime.get("zero_gravity_preclose")
    if (
        not isinstance(preclose, dict)
        or preclose.get("enabled") is not False
        or preclose.get("physics_step_count") != REQUIRED_PRECLOSE_PHYSICS_STEPS
        or preclose.get("validation_physics_step_count_unchanged")
        != expected_sim_steps * REQUIRED_SUBSTEPS
        or preclose.get("displacement_reference")
        != "raw_json_object_position_before_preclose"
    ):
        raise ValidDatasetError(f"{path}: formal zero-step preclose proof is missing")
    closing = runtime.get("contact_gradient_closing")
    expected_alphas = [float(value) for value in FORMAL_CLOSING_ALPHAS]
    if (
        not isinstance(closing, dict)
        or closing.get("enabled") is not True
        or closing.get("mode") != "collision_aware_line_search"
        or not _same_number(
            closing.get("raw_penetration_cap_m"), FORMAL_RAW_PENETRATION_CAP
        )
        or not _same_number(
            closing.get("target_penetration_cap_m"),
            FORMAL_CLOSING_TARGET_PENETRATION_CAP,
        )
        or closing.get("positive_alphas") != expected_alphas
        or closing.get("bidirectional_sampled_penetration") is not True
        or closing.get("float32_quantized_and_rechecked") is not True
        or closing.get("raw_json_remains_physx_initial_state") is not True
    ):
        raise ValidDatasetError(f"{path}: formal collision-aware closing proof is missing")
    preflight = validation.get("preflight")
    sample_closing = (
        preflight.get("collision_aware_closing")
        if isinstance(preflight, dict)
        else None
    )
    if (
        not isinstance(preflight, dict)
        or preflight.get("raw_json_penetration_passed") is not True
        or preflight.get("penetration_passed") is not True
        or preflight.get("collision_aware_closing_raw_passed") is not True
        or preflight.get("self_collision_passed") is not True
        or preflight.get("hand_object_passed") is not True
        or not isinstance(sample_closing, dict)
        or sample_closing.get("enabled") is not True
        or sample_closing.get("raw_static_gate_passed") is not True
        or sample_closing.get("selected_target_safe") is not True
        or sample_closing.get("raw_penetration_passed") is not True
        or sample_closing.get("selected_penetration_passed") is not True
        or sample_closing.get("positive_alphas") != expected_alphas
        or not _same_number(
            sample_closing.get("raw_penetration_cap_m"),
            FORMAL_RAW_PENETRATION_CAP,
        )
        or not _same_number(
            sample_closing.get("target_penetration_cap_m"),
            FORMAL_CLOSING_TARGET_PENETRATION_CAP,
        )
    ):
        raise ValidDatasetError(f"{path}: selected formal static preflight is incomplete")
    orientations = validation.get("orientations")
    if (
        not isinstance(orientations, list)
        or tuple(
            item.get("name") if isinstance(item, dict) else None
            for item in orientations
        )
        != EXPECTED_ORIENTATIONS
        or not all(
            isinstance(item, dict)
            and item.get("passed") is True
            and item.get("finite") is True
            and item.get("hand_object_contact") is True
            for item in orientations
        )
        or validation.get("passed_orientation_count") != len(EXPECTED_ORIENTATIONS)
        or validation.get("required_orientation_count") != len(EXPECTED_ORIENTATIONS)
    ):
        raise ValidDatasetError(f"{path}: six-orientation PhysX proof is incomplete")
    for item, expected_gravity in zip(orientations, EXPECTED_GRAVITY_VECTORS):
        gravity = item.get("gravity_vector_object_frame_m_s2")
        if (
            not isinstance(gravity, list)
            or len(gravity) != 3
            or any(
                not _same_number(actual, expected, atol=1.0e-5)
                for actual, expected in zip(gravity, expected_gravity)
            )
        ):
            raise ValidDatasetError(f"{path}: orientation gravity vector is invalid")
    participation = payload.get("finger_participation")
    if not isinstance(participation, dict):
        raise ValidDatasetError(f"{path}: missing finger_participation")
    target = participation.get("target_count")
    actual = participation.get("actual_count")
    names = participation.get("finger_names")
    if (
        isinstance(target, bool)
        or not isinstance(target, int)
        or target not in FINGER_COUNTS
        or actual != target
        or not isinstance(names, list)
        or len(names) != target
        or len(set(names)) != target
        or not set(names) <= set(FINGER_NAMES)
    ):
        raise ValidDatasetError(f"{path}: invalid finger_participation")
    return target


@dataclass(frozen=True)
class ValidCandidate:
    path: Path
    side: str
    finger_count: int
    finger_names: frozenset[str]
    object_id: str
    object_scale: float


def _valid_candidate(
    path: Path, *, expected_sim_steps: int = REQUIRED_SIM_STEPS
) -> ValidCandidate:
    payload = _strict_json(path)
    finger_count = _finger_count(
        path, payload, expected_sim_steps=expected_sim_steps
    )
    side = payload.get("active_side")
    if side not in SIDES:
        raise ValidDatasetError(f"{path}: active_side must be front or back")
    names = frozenset(payload["finger_participation"]["finger_names"])
    source_raw = Path(payload["validation"]["source_raw"]).expanduser().resolve()
    expected_raw = (path.parent.parent / "raw" / path.name).resolve()
    if source_raw != expected_raw:
        raise ValidDatasetError(
            f"{path}: passed output is not linked to its sibling raw candidate"
        )
    object_record = payload.get("object")
    if not isinstance(object_record, dict):
        raise ValidDatasetError(f"{path}: object metadata is missing")
    mesh_path_value = object_record.get("mesh_path")
    object_scale_value = object_record.get("scale")
    if (
        not isinstance(mesh_path_value, str)
        or not mesh_path_value
        or isinstance(object_scale_value, bool)
        or not isinstance(object_scale_value, (int, float))
        or not math.isfinite(float(object_scale_value))
        or float(object_scale_value) <= 0.0
    ):
        raise ValidDatasetError(f"{path}: object mesh path/scale metadata is invalid")
    mesh_path = Path(mesh_path_value)
    object_scale = float(object_scale_value)
    object_id = (
        mesh_path.parent.parent.name
        if mesh_path.name == "decomposed.obj"
        else mesh_path.stem
    )
    return ValidCandidate(
        path.resolve(), side, finger_count, names, object_id, object_scale
    )


def _attempt_object_scales(
    metadata: Mapping[str, Any], *, label: Path | str
) -> dict[str, float]:
    """Return the complete primitive/general scale contract from an attempt."""

    objects = metadata.get("objects")
    if not isinstance(objects, dict):
        raise ValidDatasetError(f"{label}: missing object catalog")
    primitive_ids = objects.get("primitive_ids")
    primitive_meshes = objects.get("primitive_meshes")
    general_meshes = objects.get("general_meshes")
    formal_general_mesh_ids = objects.get("formal_general_mesh_ids")
    if (
        not isinstance(primitive_ids, list)
        or any(not isinstance(value, str) or not value for value in primitive_ids)
        or len(primitive_ids) != len(set(primitive_ids))
        or not isinstance(primitive_meshes, list)
        or not isinstance(general_meshes, list)
        or formal_general_mesh_ids != list(FORMAL_GENERAL_MESH_IDS)
        or len(general_meshes) != FORMAL_GENERAL_MESH_COUNT
    ):
        raise ValidDatasetError(f"{label}: invalid object catalog")
    selection = objects.get("general_selection_manifest")
    if not isinstance(selection, dict):
        raise ValidDatasetError(f"{label}: general selection manifest proof is missing")
    selection_path_value = selection.get("path")
    if not isinstance(selection_path_value, str):
        raise ValidDatasetError(f"{label}: selection manifest path is invalid")
    selection_path = Path(selection_path_value).expanduser().resolve()
    if (
        not selection_path.is_file()
        or selection.get("sha256") != _file_sha256(selection_path)
    ):
        raise ValidDatasetError(f"{label}: selection manifest hash is stale")
    expected_primitive_ids = {spec.instance_name for spec in PRIMITIVE_SPECS}
    if set(primitive_ids) != expected_primitive_ids or len(primitive_meshes) != len(
        expected_primitive_ids
    ):
        raise ValidDatasetError(f"{label}: primitive object catalog changed")
    scales: dict[str, float] = {}
    for expected_kind, records in (
        ("primitive", primitive_meshes),
        ("general", general_meshes),
    ):
        for record in records:
            if not isinstance(record, dict):
                raise ValidDatasetError(f"{label}: invalid {expected_kind} mesh record")
            object_id = record.get("object_id")
            scale = record.get("object_scale")
            sha256 = record.get("sha256")
            mesh_value = record.get("mesh_path")
            expected_scale = (
                FORMAL_PRIMITIVE_OBJECT_SCALE
                if expected_kind == "primitive"
                else FORMAL_GENERAL_OBJECT_SCALE
            )
            if (
                record.get("kind") != expected_kind
                or not isinstance(object_id, str)
                or not object_id
                or object_id in scales
                or isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not math.isfinite(float(scale))
                or float(scale) != expected_scale
                or not isinstance(sha256, str)
                or len(sha256) != 64
                or not isinstance(mesh_value, str)
            ):
                raise ValidDatasetError(f"{label}: invalid {expected_kind} mesh catalog")
            mesh_path = Path(mesh_value).expanduser().resolve()
            if not mesh_path.is_file() or _file_sha256(mesh_path) != sha256:
                raise ValidDatasetError(
                    f"{label}: {expected_kind} mesh hash is stale for {object_id}"
                )
            scales[object_id] = float(scale)
    if {record["object_id"] for record in primitive_meshes} != expected_primitive_ids:
        raise ValidDatasetError(f"{label}: primitive IDs do not match the formal catalog")
    if {
        record.get("object_id")
        for record in general_meshes
        if isinstance(record, dict)
    } != set(FORMAL_GENERAL_MESH_IDS):
        raise ValidDatasetError(f"{label}: general IDs do not match the formal 30-mesh catalog")
    for record in general_meshes:
        if not isinstance(record, dict):
            raise ValidDatasetError(f"{label}: invalid general mesh catalog record")
        object_id = record.get("object_id")
        scale = record.get("object_scale")
        if (
            not isinstance(object_id, str)
            or not object_id
            or isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) != FORMAL_GENERAL_OBJECT_SCALE
        ):
            raise ValidDatasetError(
                f"{label}: official general-mesh object_scale must be exactly "
                f"{FORMAL_GENERAL_OBJECT_SCALE}"
            )
    return scales


def discover_attempt_valid(
    attempts_root: Path,
) -> dict[tuple[str, int], list[ValidCandidate]]:
    """Return strictly audited passed outputs grouped by side and finger count."""

    grouped = {
        (side, value): [] for side in SIDES for value in FINGER_COUNTS
    }
    for attempt_root in _completed_attempt_roots(attempts_root):
        metadata = _strict_json(attempt_root / "attempt.json")
        expected_scales = _attempt_object_scales(metadata, label=attempt_root)
        validation = metadata.get("validation")
        if not isinstance(validation, dict):
            raise ValidDatasetError(f"{attempt_root}: validation metadata is missing")
        sim_steps = validation.get("sim_steps")
        if isinstance(sim_steps, bool) or not isinstance(sim_steps, int):
            raise ValidDatasetError(f"{attempt_root}: sim_steps is invalid")
        for path in sorted(attempt_root.glob("**/valid/*.json")):
            candidate = _valid_candidate(path, expected_sim_steps=sim_steps)
            expected_scale = expected_scales.get(candidate.object_id)
            if expected_scale is None or candidate.object_scale != expected_scale:
                raise ValidDatasetError(
                    f"{path}: object scale {candidate.object_scale} does not match "
                    f"attempt catalog {expected_scale}"
                )
            grouped[(candidate.side, candidate.finger_count)].append(candidate)
    return grouped


def discover_attempt_raw(attempts_root: Path) -> Counter[tuple[str, int]]:
    counts: Counter[tuple[str, int]] = Counter()
    for attempt_root in _completed_attempt_roots(attempts_root):
        for path in sorted(attempt_root.glob("**/raw/*.json")):
            payload = _strict_json(path)
            participation = payload.get("finger_participation")
            if not isinstance(participation, dict):
                raise ValidDatasetError(f"{path}: missing finger_participation")
            value = participation.get("target_count")
            side = payload.get("active_side")
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value not in FINGER_COUNTS
                or side not in SIDES
            ):
                raise ValidDatasetError(f"{path}: invalid raw side/finger stratum")
            counts[(side, value)] += 1
    return counts


def _run_checked(command: Sequence[str], *, label: str) -> None:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True)
    if completed.returncode != 0:
        raise ValidDatasetError(f"{label} exited with {completed.returncode}")


def _attempt_name(index: int) -> str:
    return f"attempt_{index:04d}"


def _next_attempt_index(attempts_root: Path) -> int:
    values = []
    for path in attempts_root.glob("attempt_*"):
        try:
            values.append(int(path.name.split("_", 2)[1]))
        except (IndexError, ValueError):
            continue
    return max(values, default=-1) + 1


def _general_mesh_catalog(mesh_root: Path) -> list[dict[str, Any]]:
    root = Path(mesh_root).expanduser().resolve()
    paths = sorted(root.glob("*/coacd/decomposed.obj"))
    if len(paths) != EXPECTED_GENERAL_MESHES:
        raise ValidDatasetError(
            f"Expected exactly {EXPECTED_GENERAL_MESHES} general meshes, "
            f"found {len(paths)}"
        )
    object_ids = [path.parent.parent.name for path in paths]
    if len(object_ids) != len(set(object_ids)):
        raise ValidDatasetError("General mesh object IDs are not unique")
    primitive_ids = {spec.instance_name for spec in PRIMITIVE_SPECS}
    collisions = sorted(primitive_ids & set(object_ids))
    if collisions:
        raise ValidDatasetError(
            f"General mesh IDs collide with primitive IDs: {collisions}"
        )
    scale_by_id: dict[str, float] = {}
    manifest_path = root / "x2_general_mesh_manifest.json"
    if manifest_path.is_file():
        manifest = _strict_json(manifest_path)
        records = manifest.get("meshes")
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict):
                    continue
                object_id = record.get("object_id")
                scale = record.get("object_scale")
                if (
                    isinstance(object_id, str)
                    and not isinstance(scale, bool)
                    and isinstance(scale, (int, float))
                    and math.isfinite(float(scale))
                    and float(scale) > 0.0
                ):
                    scale_by_id[object_id] = float(scale)
    catalog = [
        {
            "kind": "general",
            "object_id": path.parent.parent.name,
            "mesh_path": str(path.resolve()),
            "sha256": _file_sha256(path),
            "object_scale": scale_by_id.get(path.parent.parent.name),
        }
        for path in paths
    ]
    if any(
        entry["object_scale"] != FORMAL_GENERAL_OBJECT_SCALE
        for entry in catalog
    ):
        raise ValidDatasetError(
            "Every official general mesh must declare object_scale exactly "
            f"{FORMAL_GENERAL_OBJECT_SCALE} in x2_general_mesh_manifest.json"
        )
    return catalog


def _formal_general_mesh_catalog(
    catalog: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the deterministic 30-object subset of the audited 88 meshes."""

    if len(FORMAL_GENERAL_MESH_IDS) != FORMAL_GENERAL_MESH_COUNT:
        raise ValidDatasetError("Formal general-mesh ID contract is inconsistent")
    by_id = {entry.get("object_id"): entry for entry in catalog}
    missing = sorted(set(FORMAL_GENERAL_MESH_IDS) - set(by_id))
    if missing:
        raise ValidDatasetError(f"Formal general meshes are missing: {missing}")
    selected = [by_id[object_id] for object_id in FORMAL_GENERAL_MESH_IDS]
    if len({entry["object_id"] for entry in selected}) != FORMAL_GENERAL_MESH_COUNT:
        raise ValidDatasetError("Formal general-mesh selection contains duplicate IDs")
    return selected


def _primitive_mesh_catalog() -> list[dict[str, Any]]:
    root = PRIMITIVE_MESH_ROOT.expanduser().resolve()
    catalog: list[dict[str, Any]] = []
    for spec in PRIMITIVE_SPECS:
        path = (root / spec.relative_path).resolve()
        if not path.is_file():
            raise ValidDatasetError(f"Formal primitive mesh is missing: {path}")
        catalog.append(
            {
                "kind": "primitive",
                "object_id": spec.instance_name,
                "mesh_path": str(path),
                "sha256": _file_sha256(path),
                "object_scale": FORMAL_PRIMITIVE_OBJECT_SCALE,
            }
        )
    return catalog


def _verify_selection_manifest(
    mesh_root: Path, catalog: Sequence[dict[str, Any]]
) -> None:
    root = Path(mesh_root).expanduser().resolve()
    manifest_path = root / "x2_general_mesh_manifest.json"
    if not manifest_path.is_file():
        raise ValidDatasetError(
            f"General mesh selection manifest is missing: {manifest_path}"
        )
    manifest = _strict_json(manifest_path)
    manifest_scale = manifest.get("object_scale")
    if (
        manifest.get("passed") is not True
        or manifest.get("target_count") != EXPECTED_GENERAL_MESHES
        or manifest.get("selected_count") != EXPECTED_GENERAL_MESHES
        or manifest.get("license") != "CC BY-NC 4.0 (DexGraspNet 2.0 assets)"
        or isinstance(manifest_scale, bool)
        or not isinstance(manifest_scale, (int, float))
        or not math.isfinite(float(manifest_scale))
        or float(manifest_scale) != FORMAL_GENERAL_OBJECT_SCALE
    ):
        raise ValidDatasetError(f"General mesh selection manifest is invalid: {manifest_path}")
    records = manifest.get("meshes")
    if not isinstance(records, list) or len(records) != EXPECTED_GENERAL_MESHES:
        raise ValidDatasetError(f"General mesh manifest catalog is incomplete: {manifest_path}")
    actual = {
        entry["object_id"]: (entry["sha256"], entry["object_scale"])
        for entry in catalog
    }
    recorded: dict[str, tuple[str, float]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValidDatasetError(f"General mesh manifest contains a non-object record")
        object_id = record.get("object_id")
        sha256 = record.get("sha256")
        scale = record.get("object_scale")
        if (
            not isinstance(object_id, str)
            or not isinstance(sha256, str)
            or isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) != FORMAL_GENERAL_OBJECT_SCALE
        ):
            raise ValidDatasetError(f"General mesh manifest contains an invalid record")
        if object_id in recorded:
            raise ValidDatasetError(f"General mesh manifest repeats {object_id}")
        recorded[object_id] = (sha256, float(scale))
    if recorded != actual:
        raise ValidDatasetError(
            f"General mesh files no longer match their audited selection manifest"
        )


def _attempt_metadata(
    *, finger_targets: Mapping[int, int], seed: int, args: argparse.Namespace
) -> dict[str, Any]:
    normalized_targets = {
        int(key): int(value)
        for key, value in finger_targets.items()
        if int(value) > 0
    }
    if not normalized_targets or not set(normalized_targets) <= set(FINGER_COUNTS):
        raise ValidDatasetError("An attempt requires positive targets for known finger strata")
    raw_target = sum(normalized_targets.values())
    return {
        "schema_version": 4,
        "collection_protocol_revision": COLLECTION_PROTOCOL_REVISION,
        "raw_target": raw_target,
        "seed": seed,
        "generation": {
            "pipeline_revision": GENERATOR_PIPELINE_REVISION,
            "dense_hand_object_gate": {
                "evaluation_mode": "dense_bidirectional",
                "hand_surface_samples_per_set": DENSE_HAND_SURFACE_SAMPLES_PER_SET,
                "hand_surface_samples_per_link": DENSE_HAND_SURFACE_SAMPLES_PER_LINK,
                "hand_surface_point_count": DENSE_HAND_SURFACE_POINT_COUNT,
                "object_surface_samples": DENSE_OBJECT_SURFACE_SAMPLES,
                "threshold": FORMAL_RAW_PENETRATION_CAP,
                "strict_less_than": True,
            },
            "finger_counts": sorted(normalized_targets),
            "finger_targets": {
                str(key): normalized_targets[key] for key in sorted(normalized_targets)
            },
            "sides": list(SIDES),
            "complementary_side_fingers": True,
            "n_iterations": int(args.n_iterations),
            "stratified_batching": True,
            "stratified_batch_size": STRATIFIED_BATCH_SIZE,
            "primitive_object_scale": FORMAL_PRIMITIVE_OBJECT_SCALE,
        },
        "validation": {
            "backend": VALIDATION_BACKEND,
            "protocol_revision": VALIDATION_PROTOCOL_REVISION,
            "criterion": VALIDATION_CRITERION,
            "sim_steps": int(args.sim_steps),
            "substeps": REQUIRED_SUBSTEPS,
            "preclose_physics_steps": REQUIRED_PRECLOSE_PHYSICS_STEPS,
            "raw_penetration_cap_m": FORMAL_RAW_PENETRATION_CAP,
            "closing_target_penetration_cap_m": (
                FORMAL_CLOSING_TARGET_PENETRATION_CAP
            ),
            "closing_positive_alphas": [
                float(value) for value in FORMAL_CLOSING_ALPHAS
            ],
            "actuator_drive": {
                "stiffness_n_m_per_rad": FORMAL_ACTUATOR_STIFFNESS,
                "damping_n_m_s_per_rad": FORMAL_ACTUATOR_DAMPING,
                "armature_kg_m2": FORMAL_ACTUATOR_ARMATURE,
            },
            "physx_solver": {
                "solver_type": 1,
                "external_forces_every_iteration": True,
                "solve_articulation_contact_last": False,
            },
        },
        "objects": {
            "primitive_ids": [spec.instance_name for spec in PRIMITIVE_SPECS],
            "primitive_meshes": _primitive_mesh_catalog(),
            "general_meshes": _formal_general_mesh_catalog(
                _general_mesh_catalog(args.general_mesh_root)
            ),
            "formal_general_mesh_ids": list(FORMAL_GENERAL_MESH_IDS),
            "general_selection_manifest": {
                "path": str(
                    (
                        Path(args.general_mesh_root).expanduser().resolve()
                        / "x2_general_mesh_manifest.json"
                    )
                ),
                "sha256": _file_sha256(
                    Path(args.general_mesh_root).expanduser().resolve()
                    / "x2_general_mesh_manifest.json"
                ),
            },
        },
    }


def _read_csv_rows(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValidDatasetError(f"{label} has no header: {path}")
            return list(reader)
    except ValidDatasetError:
        raise
    except Exception as exc:
        raise ValidDatasetError(f"Cannot read {label} {path}: {exc}") from exc


def _inventory_sha256(paths: Sequence[Path], *, root: Path) -> str:
    entries = [
        {
            "path": str(path.resolve().relative_to(root.resolve())),
            "sha256": _file_sha256(path),
        }
        for path in sorted(path.resolve() for path in paths)
    ]
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _csv_nonnegative_int(row: dict[str, str], key: str, *, path: Path) -> int:
    try:
        value = int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidDatasetError(f"{path}: invalid integer column {key}") from exc
    if value < 0:
        raise ValidDatasetError(f"{path}: negative integer column {key}")
    return value


def _attempt_completion_payload(
    attempt_root: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    generation_summary = attempt_root / "summary.csv"
    generation_report_path = attempt_root / GENERATION_SUMMARY_NAME
    validation_summary = attempt_root / "validation_summary.csv"
    if (
        not generation_summary.is_file()
        or not generation_report_path.is_file()
        or not validation_summary.is_file()
    ):
        raise ValidDatasetError(
            f"{attempt_root}: CSV/JSON generation summaries and validation summary "
            "are all required"
        )
    if (
        metadata.get("schema_version") != 4
        or metadata.get("collection_protocol_revision")
        != COLLECTION_PROTOCOL_REVISION
    ):
        raise ValidDatasetError(f"{attempt_root}: attempt protocol metadata is stale")
    raw_target = metadata.get("raw_target")
    if isinstance(raw_target, bool) or not isinstance(raw_target, int) or raw_target <= 0:
        raise ValidDatasetError(f"{attempt_root}: invalid raw_target in attempt metadata")
    objects = metadata.get("objects")
    if not isinstance(objects, dict):
        raise ValidDatasetError(f"{attempt_root}: missing object catalog")
    primitive_ids = objects.get("primitive_ids")
    general_meshes = objects.get("general_meshes")
    if not isinstance(primitive_ids, list) or not isinstance(general_meshes, list):
        raise ValidDatasetError(f"{attempt_root}: invalid object catalog")
    expected_objects = len(primitive_ids) + len(general_meshes)
    if expected_objects != len(PRIMITIVE_SPECS) + FORMAL_GENERAL_MESH_COUNT:
        raise ValidDatasetError(f"{attempt_root}: object catalog count changed")
    expected_object_scales = _attempt_object_scales(metadata, label=attempt_root)
    if len(expected_object_scales) != expected_objects:
        raise ValidDatasetError(f"{attempt_root}: per-object scale catalog is incomplete")

    generation = metadata.get("generation")
    if not isinstance(generation, dict):
        raise ValidDatasetError(f"{attempt_root}: generation metadata is missing")
    if (
        generation.get("stratified_batching") is not True
        or generation.get("stratified_batch_size") != STRATIFIED_BATCH_SIZE
        or generation.get("primitive_object_scale")
        != FORMAL_PRIMITIVE_OBJECT_SCALE
    ):
        raise ValidDatasetError(
            f"{attempt_root}: stratified generation/scale protocol is missing"
        )
    finger_targets_value = generation.get("finger_targets")
    if not isinstance(finger_targets_value, dict):
        raise ValidDatasetError(f"{attempt_root}: finger target metadata is missing")
    try:
        finger_targets = {
            int(key): int(value) for key, value in finger_targets_value.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValidDatasetError(f"{attempt_root}: finger target metadata is invalid") from exc
    if (
        not finger_targets
        or not set(finger_targets) <= set(FINGER_COUNTS)
        or any(value <= 0 for value in finger_targets.values())
        or sum(finger_targets.values()) != raw_target
    ):
        raise ValidDatasetError(f"{attempt_root}: finger target totals are invalid")
    validation = metadata.get("validation")
    if not isinstance(validation, dict):
        raise ValidDatasetError(f"{attempt_root}: validation metadata is missing")
    expected_drive = {
        "stiffness_n_m_per_rad": FORMAL_ACTUATOR_STIFFNESS,
        "damping_n_m_s_per_rad": FORMAL_ACTUATOR_DAMPING,
        "armature_kg_m2": FORMAL_ACTUATOR_ARMATURE,
    }
    expected_solver = {
        "solver_type": 1,
        "external_forces_every_iteration": True,
        "solve_articulation_contact_last": False,
    }
    if (
        validation.get("backend") != VALIDATION_BACKEND
        or validation.get("protocol_revision") != VALIDATION_PROTOCOL_REVISION
        or validation.get("criterion") != VALIDATION_CRITERION
        or validation.get("sim_steps") != REQUIRED_SIM_STEPS
        or validation.get("substeps") != REQUIRED_SUBSTEPS
        or validation.get("preclose_physics_steps")
        != REQUIRED_PRECLOSE_PHYSICS_STEPS
        or validation.get("actuator_drive") != expected_drive
        or validation.get("physx_solver") != expected_solver
    ):
        raise ValidDatasetError(f"{attempt_root}: formal validation protocol drifted")

    generation_rows = _read_csv_rows(
        generation_summary, label="generation summary"
    )
    expected_generation_rows = expected_objects * len(SIDES) * len(finger_targets)
    if len(generation_rows) != expected_generation_rows:
        raise ValidDatasetError(
            f"{generation_summary}: has {len(generation_rows)} rows; "
            f"expected {expected_generation_rows}"
        )
    generation_count = sum(
        _csv_nonnegative_int(row, "sample_count", path=generation_summary)
        for row in generation_rows
    )
    if generation_count != raw_target:
        raise ValidDatasetError(
            f"{generation_summary}: sample total {generation_count} != {raw_target}"
        )
    per_finger = Counter()
    for row in generation_rows:
        try:
            finger_count = int(row["finger_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidDatasetError(
                f"{generation_summary}: invalid finger_count"
            ) from exc
        per_finger[finger_count] += _csv_nonnegative_int(
            row, "sample_count", path=generation_summary
        )
    if per_finger != Counter(finger_targets):
        raise ValidDatasetError(
            f"{generation_summary}: raw finger strata do not match attempt targets"
        )

    generation_report = _strict_json(generation_report_path)
    settings = generation_report.get("settings")
    if not isinstance(settings, dict):
        raise ValidDatasetError(
            f"{generation_report_path}: generation settings are missing"
        )
    try:
        reported_targets = {
            int(key): int(value)
            for key, value in settings.get("finger_targets", {}).items()
        }
        reported_scales = {
            str(key): float(value)
            for key, value in settings.get("object_scale_by_id", {}).items()
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValidDatasetError(
            f"{generation_report_path}: target/scale settings are invalid"
        ) from exc
    if (
        generation_report.get("passed") is not True
        or generation_report.get("total_samples") != raw_target
        or generation_report.get("total_finite_samples") != raw_target
        or generation_report.get("instance_side_runs")
        != expected_generation_rows
        or Path(str(generation_report.get("summary_csv", ""))).resolve()
        != generation_summary.resolve()
        or Path(str(generation_report.get("summary_json", ""))).resolve()
        != generation_report_path.resolve()
        or settings.get("resume") is not True
        or settings.get("stratified_batching") is not True
        or settings.get("batch_size") != STRATIFIED_BATCH_SIZE
        or settings.get("stratified_batch_size") != STRATIFIED_BATCH_SIZE
        or settings.get("n_iterations") != generation.get("n_iterations")
        or reported_targets != finger_targets
        or settings.get("finger_counts") != sorted(finger_targets)
        or reported_scales != expected_object_scales
    ):
        raise ValidDatasetError(
            f"{generation_report_path}: stratified generation proof is inconsistent"
        )
    generator_calls = generation_report.get("generator_calls")
    if not isinstance(generator_calls, list) or len(generator_calls) != (
        expected_generation_rows
    ):
        raise ValidDatasetError(
            f"{generation_report_path}: per-group generation proof is incomplete"
        )
    expected_call_keys = {
        (object_id, side, finger_count)
        for object_id in expected_object_scales
        for side in SIDES
        for finger_count in finger_targets
    }
    actual_call_keys: set[tuple[str, str, int]] = set()
    for call in generator_calls:
        if not isinstance(call, dict):
            raise ValidDatasetError(
                f"{generation_report_path}: invalid per-group generation proof"
            )
        object_id = call.get("instance_name")
        side = call.get("side")
        finger_count = call.get("finger_count")
        scale = call.get("object_scale")
        if (
            not isinstance(object_id, str)
            or side not in SIDES
            or isinstance(finger_count, bool)
            or not isinstance(finger_count, int)
            or call.get("stratified") is not True
            or isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or float(scale) != expected_object_scales.get(object_id)
        ):
            raise ValidDatasetError(
                f"{generation_report_path}: per-group scale/batching proof is invalid"
            )
        actual_call_keys.add((object_id, side, finger_count))
    if actual_call_keys != expected_call_keys:
        raise ValidDatasetError(
            f"{generation_report_path}: per-group generation keys are incomplete"
        )

    validation_rows = _read_csv_rows(
        validation_summary, label="validation summary"
    )
    if len(validation_rows) != expected_objects:
        raise ValidDatasetError(
            f"{validation_summary}: has {len(validation_rows)} rows; "
            f"expected {expected_objects}"
        )
    candidate_count = sum(
        _csv_nonnegative_int(row, "candidate_count", path=validation_summary)
        for row in validation_rows
    )
    valid_count = sum(
        _csv_nonnegative_int(row, "valid_count", path=validation_summary)
        for row in validation_rows
    )
    failed_count = sum(
        _csv_nonnegative_int(row, "failed_count", path=validation_summary)
        for row in validation_rows
    )
    if candidate_count != raw_target or valid_count + failed_count != raw_target:
        raise ValidDatasetError(
            f"{validation_summary}: routed {valid_count}+{failed_count} of "
            f"{candidate_count} candidates; expected {raw_target}"
        )

    raw_files = list(attempt_root.glob("**/raw/*.json"))
    valid_files = list(attempt_root.glob("**/valid/*.json"))
    failed_files = list(attempt_root.glob("**/failed/*.json"))
    if len(raw_files) != raw_target:
        raise ValidDatasetError(
            f"{attempt_root}: found {len(raw_files)} raw files; expected {raw_target}"
        )
    if len(valid_files) != valid_count or len(failed_files) != failed_count:
        raise ValidDatasetError(
            f"{attempt_root}: routed file counts disagree with validation summary"
        )
    metadata_path = attempt_root / "attempt.json"
    return {
        "passed": True,
        "collection_protocol_revision": COLLECTION_PROTOCOL_REVISION,
        "attempt_metadata_sha256": _file_sha256(metadata_path),
        "generation_summary_sha256": _file_sha256(generation_summary),
        "generation_report_sha256": _file_sha256(generation_report_path),
        "validation_summary_sha256": _file_sha256(validation_summary),
        "stratified_batching": True,
        "stratified_batch_size": STRATIFIED_BATCH_SIZE,
        "raw_count": raw_target,
        "valid_count": valid_count,
        "failed_count": failed_count,
    }


def _assert_completed_attempt(attempt_root: Path) -> dict[str, Any]:
    metadata_path = attempt_root / "attempt.json"
    marker_path = attempt_root / ATTEMPT_COMPLETION_NAME
    if not metadata_path.is_file() or not marker_path.is_file():
        raise ValidDatasetError(f"{attempt_root}: completed attempt proof is missing")
    metadata = _strict_json(metadata_path)
    marker = _strict_json(marker_path)
    expected = _attempt_completion_payload(attempt_root, metadata)
    if marker != expected:
        raise ValidDatasetError(f"{marker_path}: completion proof is stale")
    return metadata


def _completed_attempt_roots(attempts_root: Path) -> list[Path]:
    completed: list[Path] = []
    for attempt_root in sorted(attempts_root.glob("attempt_*")):
        marker = attempt_root / ATTEMPT_COMPLETION_NAME
        if not marker.is_file():
            continue
        _assert_completed_attempt(attempt_root)
        completed.append(attempt_root)
    return completed


def _planned_raw_count(
    *, deficit: int, raw_count: int, valid_count: int, minimum: int, oversample: float
) -> int:
    if deficit <= 0:
        return 0
    if raw_count > 0:
        observed_rate = valid_count / raw_count
        conservative_rate = max(0.02, min(0.95, observed_rate * 0.85))
        estimate = math.ceil(deficit / conservative_rate * oversample)
    else:
        estimate = math.ceil(deficit * oversample)
    return max(minimum, estimate)


def _generate_attempt(
    *,
    attempt_root: Path,
    finger_targets: Mapping[int, int],
    seed: int,
    args: argparse.Namespace,
) -> None:
    metadata_path = attempt_root / "attempt.json"
    expected = _attempt_metadata(
        finger_targets=finger_targets, seed=seed, args=args
    )
    raw_target = int(expected["raw_target"])
    if metadata_path.exists() and _strict_json(metadata_path) != expected:
        raise ValidDatasetError(f"Attempt metadata changed: {metadata_path}")
    _atomic_json(metadata_path, expected)

    completion_path = attempt_root / ATTEMPT_COMPLETION_NAME
    if completion_path.is_file():
        _assert_completed_attempt(attempt_root)
        return

    if not (
        (attempt_root / "summary.csv").is_file()
        and (attempt_root / GENERATION_SUMMARY_NAME).is_file()
    ):
        staging = attempt_root / ".staging"
        if staging.exists():
            shutil.rmtree(staging)
        command = [
            sys.executable,
            str(GENERATOR),
            "--shapes",
            "sphere",
            "cylinder",
            "cuboid",
            "cube",
            "--include-general-meshes",
            "--general-mesh-root",
            str(args.general_mesh_root),
            "--general-mesh-ids",
            *FORMAL_GENERAL_MESH_IDS,
            "--side",
            "both",
            "--finger-counts",
            *(str(value) for value in sorted(finger_targets)),
            "--complementary-side-fingers",
            "--finger-targets",
            *(str(finger_targets[value]) for value in sorted(finger_targets)),
            "--n-iterations",
            str(args.n_iterations),
            "--device",
            args.generation_device,
            "--jobs",
            str(args.jobs),
            "--seed",
            str(seed),
            "--output-root",
            str(attempt_root),
            "--resume",
            "--stratified-batching",
        ]
        _run_checked(command, label="complementary multi-finger generation")

    if not (attempt_root / "validation_summary.csv").is_file():
        command = [
            sys.executable,
            str(VALIDATOR),
            "--input-root",
            str(attempt_root),
            "--shapes",
            "sphere",
            "cylinder",
            "cuboid",
            "cube",
            "--include-general-meshes",
            "--general-mesh-root",
            str(args.general_mesh_root),
            "--general-mesh-ids",
            *FORMAL_GENERAL_MESH_IDS,
            "--side",
            "both",
            "--batch-size",
            str(args.validation_batch_size),
            "--sim-steps",
            str(args.sim_steps),
            "--substeps",
            str(REQUIRED_SUBSTEPS),
            "--preclose-physics-steps",
            str(REQUIRED_PRECLOSE_PHYSICS_STEPS),
            "--closing-penetration-cap",
            str(FORMAL_CLOSING_TARGET_PENETRATION_CAP),
            "--actuator-stiffness",
            str(FORMAL_ACTUATOR_STIFFNESS),
            "--actuator-damping",
            str(FORMAL_ACTUATOR_DAMPING),
            "--actuator-armature",
            str(FORMAL_ACTUATOR_ARMATURE),
            "--criterion",
            VALIDATION_CRITERION,
            "--device",
            args.validation_device,
            "--resume",
        ]
        _run_checked(command, label="PhysX validation")

    completion = _attempt_completion_payload(attempt_root, expected)
    _atomic_json(completion_path, completion)
    _assert_completed_attempt(attempt_root)


def _resume_incomplete_attempts(
    attempts_root: Path, args: argparse.Namespace
) -> None:
    for attempt_root in sorted(attempts_root.glob("attempt_*")):
        metadata_path = attempt_root / "attempt.json"
        completion_path = attempt_root / ATTEMPT_COMPLETION_NAME
        if completion_path.is_file():
            _assert_completed_attempt(attempt_root)
            continue
        if not metadata_path.is_file():
            raise ValidDatasetError(
                f"Incomplete attempt has no metadata and cannot be resumed: {attempt_root}"
            )
        metadata = _strict_json(metadata_path)
        raw_target = metadata.get("raw_target")
        seed = metadata.get("seed")
        generation = metadata.get("generation")
        finger_targets_value = (
            generation.get("finger_targets")
            if isinstance(generation, dict)
            else None
        )
        if (
            isinstance(raw_target, bool)
            or not isinstance(raw_target, int)
            or raw_target <= 0
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(finger_targets_value, dict)
        ):
            raise ValidDatasetError(f"Invalid attempt metadata: {metadata_path}")
        try:
            finger_targets = {
                int(key): int(value)
                for key, value in finger_targets_value.items()
            }
        except (TypeError, ValueError) as exc:
            raise ValidDatasetError(
                f"Invalid attempt finger targets: {metadata_path}"
            ) from exc
        if (
            not finger_targets
            or not set(finger_targets) <= set(FINGER_COUNTS)
            or any(value <= 0 for value in finger_targets.values())
            or sum(finger_targets.values()) != raw_target
        ):
            raise ValidDatasetError(f"Invalid attempt finger targets: {metadata_path}")
        print(f"[collector] resuming {attempt_root.name}", flush=True)
        _generate_attempt(
            attempt_root=attempt_root,
            finger_targets=finger_targets,
            seed=seed,
            args=args,
        )


def pair_candidates(
    grouped: dict[tuple[str, int], list[ValidCandidate]],
    per_side_finger_target: int,
) -> dict[int, list[tuple[ValidCandidate, ValidCandidate]]]:
    """Greedily pair complementary strata on the same object with disjoint fingers."""

    result: dict[int, list[tuple[ValidCandidate, ValidCandidate]]] = {}
    for front_count in (1, 2, 3, 4):
        back_count = 5 - front_count
        front_by_object: dict[str, list[ValidCandidate]] = {}
        back_by_object: dict[str, list[ValidCandidate]] = {}
        for candidate in grouped[("front", front_count)]:
            front_by_object.setdefault(candidate.object_id, []).append(candidate)
        for candidate in grouped[("back", back_count)]:
            back_by_object.setdefault(candidate.object_id, []).append(candidate)
        pairs_by_object: dict[
            str, list[tuple[ValidCandidate, ValidCandidate]]
        ] = {}
        for object_id in sorted(set(front_by_object) & set(back_by_object)):
            used_back: set[Path] = set()
            object_pairs: list[tuple[ValidCandidate, ValidCandidate]] = []
            for front in front_by_object[object_id]:
                match = next(
                    (
                        back
                        for back in back_by_object[object_id]
                        if back.path not in used_back
                        and front.finger_names.isdisjoint(back.finger_names)
                    ),
                    None,
                )
                if match is not None:
                    used_back.add(match.path)
                    object_pairs.append((front, match))
            pairs_by_object[object_id] = object_pairs
        pairs = []
        depth = 0
        while len(pairs) < per_side_finger_target:
            added = False
            for object_id in sorted(pairs_by_object):
                values = pairs_by_object[object_id]
                if depth < len(values):
                    pairs.append(values[depth])
                    added = True
                    if len(pairs) == per_side_finger_target:
                        break
            if not added:
                break
            depth += 1
        result[front_count] = pairs
    return result


def _round_robin_candidates(
    candidates: Sequence[ValidCandidate], count: int
) -> list[ValidCandidate]:
    by_object: dict[str, list[ValidCandidate]] = {}
    for candidate in candidates:
        by_object.setdefault(candidate.object_id, []).append(candidate)
    result: list[ValidCandidate] = []
    depth = 0
    while len(result) < count:
        added = False
        for object_id in sorted(by_object):
            values = by_object[object_id]
            if depth < len(values):
                result.append(values[depth])
                added = True
                if len(result) == count:
                    break
        if not added:
            break
        depth += 1
    return result


def materialize_final(
    *,
    output_root: Path,
    grouped: dict[tuple[str, int], list[ValidCandidate]],
    per_side_finger_target: int,
    required_general_ids: set[str] | None = None,
) -> dict[str, Any]:
    final_root = output_root / "final_valid"
    if final_root.exists():
        shutil.rmtree(final_root)
    records: list[dict[str, Any]] = []
    merged_entries: list[dict[str, Any]] = []
    object_ids: set[str] = set()
    object_scales: dict[str, float] = {}
    pairs = pair_candidates(grouped, per_side_finger_target)
    selected_by_stratum: dict[tuple[str, int], list[tuple[ValidCandidate, str | None]]] = {
        (side, value): [] for side in SIDES for value in FINGER_COUNTS
    }
    for front_count, values in pairs.items():
        if len(values) != per_side_finger_target:
            raise ValidDatasetError(
                f"front f{front_count}/back f{5-front_count} has "
                f"{len(values)} disjoint same-object pairs; need {per_side_finger_target}"
            )
        for index, (front, back) in enumerate(values):
            pair_id = f"front_f{front_count}_back_f{5-front_count}_{index:06d}"
            selected_by_stratum[("front", front_count)].append((front, pair_id))
            selected_by_stratum[("back", 5 - front_count)].append((back, pair_id))
            merged_entries.append(
                {
                    "pair_id": pair_id,
                    "object_id": front.object_id,
                    "front_finger_count": front_count,
                    "front_finger_names": sorted(front.finger_names),
                    "back_finger_count": 5 - front_count,
                    "back_finger_names": sorted(back.finger_names),
                    "disjoint": front.finger_names.isdisjoint(back.finger_names),
                }
            )

    for side in SIDES:
        selected_five = _round_robin_candidates(
            grouped[(side, 5)], per_side_finger_target
        )
        if len(selected_five) != per_side_finger_target:
            raise ValidDatasetError(
                f"{side}/f5 has {len(selected_five)} valid; "
                f"need {per_side_finger_target}"
            )
        selected_by_stratum[(side, 5)] = [
            (candidate, None) for candidate in selected_five
        ]
        for index, candidate in enumerate(selected_five):
            merged_entries.append(
                {
                    "pair_id": None,
                    "object_id": candidate.object_id,
                    f"{side}_finger_count": 5,
                    f"{side}_finger_names": sorted(candidate.finger_names),
                    "opposite_side": None,
                    "reason": "five fingers leave no non-empty disjoint opposite-side set",
                    "index": index,
                }
            )

    for (side, finger_count), selected in selected_by_stratum.items():
        if len(selected) != per_side_finger_target:
            raise ValidDatasetError(
                f"{side}/f{finger_count} selected {len(selected)}; "
                f"need {per_side_finger_target}"
            )
        directory = final_root / side / f"f{finger_count}"
        directory.mkdir(parents=True, exist_ok=True)
        for index, (candidate, pair_id) in enumerate(selected):
            payload = _strict_json(candidate.path)
            _finger_count(candidate.path, payload)
            object_record = payload.get("object")
            payload_scale = (
                object_record.get("scale")
                if isinstance(object_record, dict)
                else None
            )
            if (
                isinstance(payload_scale, bool)
                or not isinstance(payload_scale, (int, float))
                or not math.isfinite(float(payload_scale))
                or float(payload_scale) <= 0.0
                or float(payload_scale) != candidate.object_scale
            ):
                raise ValidDatasetError(
                    f"{candidate.path}: selected object scale proof is inconsistent"
                )
            previous_scale = object_scales.setdefault(
                candidate.object_id, candidate.object_scale
            )
            if previous_scale != candidate.object_scale:
                raise ValidDatasetError(
                    f"Object {candidate.object_id} uses inconsistent scales"
                )
            object_ids.add(candidate.object_id)
            destination = directory / f"x2_{side}_f{finger_count}_{index:06d}.json"
            os.link(candidate.path, destination)
            records.append(
                {
                    "path": str(destination.resolve()),
                    "source": str(candidate.path),
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "side": side,
                    "finger_count": finger_count,
                    "finger_names": sorted(candidate.finger_names),
                    "object_id": candidate.object_id,
                    "object_scale": candidate.object_scale,
                    "pair_id": pair_id,
                }
            )
    side_finger_counts = {
        side: {
            str(value): sum(
                record["side"] == side and record["finger_count"] == value
                for record in records
            )
            for value in FINGER_COUNTS
        }
        for side in SIDES
    }
    missing_general_ids = sorted((required_general_ids or set()) - object_ids)
    if missing_general_ids:
        raise ValidDatasetError(
            f"Final selection misses {len(missing_general_ids)} general objects: "
            f"{missing_general_ids[:10]}"
        )
    expected_valid = per_side_finger_target * len(FINGER_COUNTS) * len(SIDES)
    if len(records) != expected_valid:
        raise ValidDatasetError(
            f"Final materialization has {len(records)} records; expected {expected_valid}"
        )
    if len({record["source"] for record in records}) != len(records):
        raise ValidDatasetError("Final materialization reuses a validated source record")
    pair_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        pair_id = record["pair_id"]
        if pair_id is not None:
            pair_records.setdefault(pair_id, []).append(record)
    for pair_id, values in pair_records.items():
        if (
            len(values) != 2
            or {value["side"] for value in values} != set(SIDES)
            or len({value["object_id"] for value in values}) != 1
            or not set(values[0]["finger_names"]).isdisjoint(
                values[1]["finger_names"]
            )
        ):
            raise ValidDatasetError(f"Final pair proof is invalid: {pair_id}")
    if len(pair_records) != per_side_finger_target * 4:
        raise ValidDatasetError(
            f"Final materialization has {len(pair_records)} paired entries; "
            f"expected {per_side_finger_target * 4}"
        )
    final_files = sorted(final_root.glob("**/*.json"))
    if len(final_files) != expected_valid:
        raise ValidDatasetError(
            f"Final directory has {len(final_files)} JSON files; expected {expected_valid}"
        )
    manifest = {
        "passed": True,
        "collection_protocol_revision": COLLECTION_PROTOCOL_REVISION,
        "generation_protocol": {
            "stratified_batching": True,
            "stratified_batch_size": STRATIFIED_BATCH_SIZE,
            "rectangular_contact_count_partitions": [4, 5],
        },
        "validation_protocol": {
            "backend": VALIDATION_BACKEND,
            "protocol_revision": VALIDATION_PROTOCOL_REVISION,
            "criterion": VALIDATION_CRITERION,
            "sim_steps": REQUIRED_SIM_STEPS,
            "required_orientations": list(EXPECTED_ORIENTATIONS),
        },
        "target_valid": expected_valid,
        "valid_count": len(records),
        "per_side_finger_target": per_side_finger_target,
        "side_finger_counts": side_finger_counts,
        "paired_entry_count": sum(entry["pair_id"] is not None for entry in merged_entries),
        "single_side_five_finger_entry_count": sum(
            entry["pair_id"] is None for entry in merged_entries
        ),
        "object_count": len(object_ids),
        "object_ids": sorted(object_ids),
        "object_scale_by_id": {
            object_id: object_scales[object_id] for object_id in sorted(object_scales)
        },
        "required_general_object_count": len(required_general_ids or set()),
        "covered_general_object_count": len(
            (required_general_ids or set()) & object_ids
        ),
        "attempt_completion_proofs": [
            {
                "path": str(path.resolve()),
                "sha256": _file_sha256(path),
            }
            for path in sorted(
                (output_root / "attempts").glob(
                    f"attempt_*/{ATTEMPT_COMPLETION_NAME}"
                )
            )
        ],
        "records": records,
        "merged_entries": merged_entries,
    }
    _atomic_json(output_root / "manifest.json", manifest)
    return manifest


def _run_locked(args: argparse.Namespace) -> dict[str, Any]:
    strata_count = len(FINGER_COUNTS) * len(SIDES)
    if args.target_valid != FORMAL_TARGET_VALID:
        raise ValidDatasetError(
            f"Formal collection requires exactly {FORMAL_TARGET_VALID} valid records"
        )
    if args.target_valid % strata_count != 0:
        raise ValidDatasetError("target-valid must be divisible by ten")
    per_side_finger_target = args.target_valid // strata_count
    if per_side_finger_target != FORMAL_PER_SIDE_FINGER_TARGET:
        raise ValidDatasetError(
            "Formal per-side/per-finger quota must be exactly "
            f"{FORMAL_PER_SIDE_FINGER_TARGET}"
        )
    if args.sim_steps != REQUIRED_SIM_STEPS:
        raise ValidDatasetError(
            f"Formal collection requires exactly {REQUIRED_SIM_STEPS} PhysX simulation steps"
        )
    output_root = args.output_root.expanduser().resolve()
    attempts_root = output_root / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    official_general_catalog = _general_mesh_catalog(args.general_mesh_root)
    _verify_selection_manifest(args.general_mesh_root, official_general_catalog)
    general_catalog = _formal_general_mesh_catalog(official_general_catalog)
    required_general_ids = {entry["object_id"] for entry in general_catalog}

    _resume_incomplete_attempts(attempts_root, args)

    while True:
        grouped = discover_attempt_valid(attempts_root)
        pairs = pair_candidates(grouped, per_side_finger_target)
        pairing_complete = all(
            len(pairs[value]) >= per_side_finger_target for value in (1, 2, 3, 4)
        )
        five_complete = all(
            len(grouped[(side, 5)]) >= per_side_finger_target for side in SIDES
        )
        covered: set[str] = set()
        if pairing_complete and five_complete:
            preview_pairs = [
                candidate
                for values in pairs.values()
                for pair in values
                for candidate in pair
            ]
            preview_five = [
                candidate
                for side in SIDES
                for candidate in _round_robin_candidates(
                    grouped[(side, 5)], per_side_finger_target
                )
            ]
            covered = {
                candidate.object_id for candidate in (*preview_pairs, *preview_five)
            }
            coverage_complete = required_general_ids <= covered
            if coverage_complete:
                return materialize_final(
                    output_root=output_root,
                    grouped=grouped,
                    per_side_finger_target=per_side_finger_target,
                    required_general_ids=required_general_ids,
                )
        raw_counts = discover_attempt_raw(attempts_root)
        index = _next_attempt_index(attempts_root)
        planned_by_finger: dict[int, int] = {}
        for finger_count in FINGER_COUNTS:
            if finger_count == 5:
                deficit = max(
                    per_side_finger_target - len(grouped[(side, 5)])
                    for side in SIDES
                )
            else:
                deficit = max(
                    per_side_finger_target - len(pairs[finger_count]),
                    per_side_finger_target - len(pairs[5 - finger_count]),
                )
            if deficit <= 0:
                planned_by_finger[finger_count] = 0
                continue
            raw_count = sum(raw_counts[(side, finger_count)] for side in SIDES)
            valid_count = sum(len(grouped[(side, finger_count)]) for side in SIDES)
            planned_by_finger[finger_count] = _planned_raw_count(
                deficit=deficit * 2,
                raw_count=raw_count,
                valid_count=valid_count,
                minimum=args.minimum_attempt_raw,
                oversample=args.oversample,
            )
        finger_targets = {
            finger_count: target
            for finger_count, target in planned_by_finger.items()
            if target > 0
        }
        if not finger_targets:
            missing_coverage = sorted(required_general_ids - covered)
            if not missing_coverage:
                raise ValidDatasetError(
                    "All quotas appear complete but final materialization did not run"
                )
            # Coverage-only recovery remains broad until per-object targeting is
            # available, but it is explicit and cannot silently create a zero-work
            # attempt.
            finger_targets = {
                value: args.minimum_attempt_raw for value in FINGER_COUNTS
            }
            print(
                f"[collector] coverage-only retry for {len(missing_coverage)} "
                f"general objects: {missing_coverage[:10]}",
                flush=True,
            )
        raw_target = sum(finger_targets.values())
        attempt_root = attempts_root / _attempt_name(index)
        print(
            f"[collector] pairing={ {k: len(v) for k, v in pairs.items()} }, "
            f"f5={{'front': {len(grouped[('front', 5)])}, "
            f"'back': {len(grouped[('back', 5)])}}}, "
            f"finger_targets={finger_targets}, next_raw={raw_target}, "
            f"attempt={attempt_root.name}",
            flush=True,
        )
        _generate_attempt(
            attempt_root=attempt_root,
            finger_targets=finger_targets,
            seed=args.seed + index * 1009,
            args=args,
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".collector.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValidDatasetError(
                f"Another collector already holds {lock_path}"
            ) from exc
        return _run_locked(args)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-valid", type=_positive_int, default=FORMAL_TARGET_VALID
    )
    parser.add_argument(
        "--minimum-attempt-raw",
        type=_positive_int,
        default=FORMAL_PER_SIDE_FINGER_TARGET,
        help=(
            "Minimum raw rows requested for any still-deficient finger stratum; "
            "the adaptive acceptance-rate estimate may request more."
        ),
    )
    parser.add_argument("--oversample", type=float, default=1.25)
    parser.add_argument("--n-iterations", type=_positive_int, default=6000)
    parser.add_argument("--jobs", type=_positive_int, default=2)
    parser.add_argument("--generation-device", default="cuda")
    parser.add_argument("--validation-device", default="cuda:0")
    parser.add_argument("--validation-batch-size", type=_positive_int, default=32)
    parser.add_argument("--sim-steps", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--general-mesh-root", type=Path, default=PROJECT_ROOT / "data" / "meshdata"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if not math.isfinite(args.oversample) or args.oversample < 1.0:
        parser.error("oversample must be finite and at least 1.0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:
        print(
            json.dumps(
                {"passed": False, "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
