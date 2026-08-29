"""Contracts for joint X2 two-object PhysX validation.

The composed dataset combines two independently validated single-object
grasps.  This module deliberately treats those records only as provenance:
dual-object success requires a new shared-hand simulation in which both rigid
objects independently retain hand contact in all six gravity directions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from grasp_generation.x2_isaac_validation import (
    EXPECTED_ACTUATOR_NAMES,
    EXPECTED_JOINT_NAMES,
    GRAVITY_TESTS_WXYZ,
    PASSIVE_MIMIC_DRIVERS,
    quaternion_matrix_wxyz,
)


DUAL_VALIDATION_BACKEND = "isaac_sim_physx"
DUAL_VALIDATION_PROTOCOL_REVISION = "x2_dual_object_six_orientation_physx_v1"
DUAL_VALIDATION_PROTOCOL_REVISION_OBJECT_COLLISION = (
    "x2_dual_object_six_orientation_physx_v1_object_collision"
)
DUAL_COMPOSITION_PROTOCOL_REVISION = "x2_right_left_dual_object_warm_start_v1"
# This is intentionally not a positive-library protocol.  It exists solely
# to admit an exact-table geometry proposal into the table-supported PhysX
# acquisition worker, where its first physical static qualification is made.
TABLETOP_REPAIR_UNVALIDATED_PROTOCOL_REVISION = "x2_tabletop_repair_unvalidated_static_v1"
DUAL_VALIDATION_CRITERION = (
    "both_objects_independently_retain_hand_contact_all_six_orientations"
)
FINGER_NAMES = frozenset(("index", "middle", "ring", "little", "thumb"))
EXPECTED_GRAVITY_NAMES = tuple(name for name, _ in GRAVITY_TESTS_WXYZ)


class X2DualValidationError(RuntimeError):
    """Raised when a composed candidate or validation proof is malformed."""


@dataclass(frozen=True)
class X2DualObjectCandidate:
    path: Path
    sha256: str
    record: dict[str, Any]
    right: dict[str, Any]
    left: dict[str, Any]

    @property
    def candidate_id(self) -> str:
        return str(self.record["candidate_id"])

    @property
    def combination(self) -> str:
        return (
            f"right_f{len(self.right['finger_names'])}_"
            f"left_f{len(self.left['finger_names'])}"
        )

    @property
    def object_group(self) -> tuple[tuple[Path, float], tuple[Path, float]]:
        return (
            (
                Path(str(self.right["mesh_path"])).expanduser().resolve(),
                float(self.right["scale"]),
            ),
            (
                Path(str(self.left["mesh_path"])).expanduser().resolve(),
                float(self.left["scale"]),
            ),
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise X2DualValidationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise X2DualValidationError(f"{path}: JSON root is not an object")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_vector(
    value: Any, length: int, *, label: str, path: Path
) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise X2DualValidationError(f"{path}: {label} must have length {length}")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise X2DualValidationError(f"{path}: {label} is non-numeric") from exc
    if not all(math.isfinite(item) for item in result):
        raise X2DualValidationError(f"{path}: {label} is non-finite")
    return result


def _pose_record(value: Any, *, label: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise X2DualValidationError(f"{path}: {label} is missing")
    translation = _finite_vector(
        value.get("translation"), 3, label=f"{label}.translation", path=path
    )
    rotation = np.asarray(value.get("rotation_matrix"), dtype=np.float64)
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1.0e-6, rtol=0.0
        )
        or not math.isclose(
            float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-6, rel_tol=0.0
        )
    ):
        raise X2DualValidationError(
            f"{path}: {label}.rotation_matrix is not a proper rotation"
        )
    return {
        "translation": translation,
        "rotation_matrix": rotation.tolist(),
    }


def _audit_hand(record: Mapping[str, Any], path: Path) -> None:
    hand = record.get("hand")
    if not isinstance(hand, dict) or hand.get("frame") != "shared_hand":
        raise X2DualValidationError(f"{path}: shared-hand record is missing")
    actuator_names = hand.get("actuator_names")
    joint_names = hand.get("joint_names")
    if tuple(actuator_names or ()) != EXPECTED_ACTUATOR_NAMES:
        raise X2DualValidationError(f"{path}: actuator order is stale")
    if tuple(joint_names or ()) != EXPECTED_JOINT_NAMES:
        raise X2DualValidationError(f"{path}: joint order is stale")
    actuator = _finite_vector(
        hand.get("actuator"),
        len(EXPECTED_ACTUATOR_NAMES),
        label="hand.actuator",
        path=path,
    )
    joint = _finite_vector(
        hand.get("joint"),
        len(EXPECTED_JOINT_NAMES),
        label="hand.joint",
        path=path,
    )
    actuator_by_name = dict(zip(EXPECTED_ACTUATOR_NAMES, actuator))
    joint_by_name = dict(zip(EXPECTED_JOINT_NAMES, joint))
    for name in EXPECTED_ACTUATOR_NAMES:
        if not math.isclose(
            joint_by_name[name], actuator_by_name[name], abs_tol=1.0e-8, rel_tol=0.0
        ):
            raise X2DualValidationError(
                f"{path}: composed actuator/joint mismatch for {name}"
            )
    for follower, driver in PASSIVE_MIMIC_DRIVERS.items():
        if not math.isclose(
            joint_by_name[follower],
            actuator_by_name[driver],
            abs_tol=1.0e-8,
            rel_tol=0.0,
        ):
            raise X2DualValidationError(
                f"{path}: composed passive mimic mismatch for {follower}"
            )
    pose = _pose_record(hand.get("pose"), label="hand.pose", path=path)
    if (
        not np.allclose(pose["translation"], np.zeros(3), atol=1.0e-12, rtol=0.0)
        or not np.allclose(
            pose["rotation_matrix"], np.eye(3), atol=1.0e-12, rtol=0.0
        )
    ):
        raise X2DualValidationError(f"{path}: shared hand pose must be identity")
    owners = hand.get("finger_mode_owner")
    if (
        not isinstance(owners, dict)
        or set(owners) != FINGER_NAMES
        or not set(owners.values()) <= {"right", "left"}
    ):
        raise X2DualValidationError(f"{path}: finger ownership is malformed")


def _audit_object(
    value: Any,
    *,
    expected_slot: str,
    expected_side: str,
    path: Path,
    allow_unvalidated_tabletop_source: bool = False,
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("slot") != expected_slot
        or value.get("source_active_side") != expected_side
        or value.get("mode") != expected_slot
    ):
        raise X2DualValidationError(
            f"{path}: {expected_slot} object branch is malformed"
        )
    object_id = value.get("object_id")
    mesh_value = value.get("mesh_path")
    scale = value.get("scale")
    source_value = value.get("source_valid")
    source_sha = value.get("source_valid_sha256")
    finger_names = value.get("finger_names")
    if (
        not isinstance(object_id, str)
        or not object_id
        or not isinstance(mesh_value, str)
        or not isinstance(source_value, str)
        or not isinstance(source_sha, str)
        or isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
        or not isinstance(finger_names, list)
        or not finger_names
        or not set(finger_names) <= FINGER_NAMES
        or len(finger_names) != len(set(finger_names))
    ):
        raise X2DualValidationError(
            f"{path}: {expected_slot} object metadata is incomplete"
        )
    mesh_path = Path(mesh_value).expanduser().resolve()
    source_path = Path(source_value).expanduser().resolve()
    if not mesh_path.is_file():
        raise X2DualValidationError(f"{path}: object mesh is missing: {mesh_path}")
    source_record_available = source_path.is_file()
    # Source records can acquire non-semantic audit/provenance fields after a
    # composed candidate was made.  Their hash is useful provenance, but it
    # is not a grasp, ownership, IK, or collision constraint and must not
    # turn an otherwise self-consistent static candidate into an unexecutable
    # task.  When the source record is present its semantic validation remains
    # mandatory; when it has been pruned, preserve an explicit provenance
    # status and defer executable feasibility to the normal Plan/Certify path.
    observed_source_sha = file_sha256(source_path) if source_record_available else None
    if source_record_available:
        source = strict_json(source_path)
        source_object = source.get("object")
        validation = source.get("validation")
        source_participation = source.get("finger_participation")
        source_scale = (
            source_object.get("scale")
            if isinstance(source_object, dict)
            else None
        )
        source_matches_geometry_and_ownership = (
            source.get("active_side") == expected_side
            and isinstance(source_object, dict)
            and Path(str(source_object.get("mesh_path"))).expanduser().resolve()
            == mesh_path
            and not isinstance(source_scale, bool)
            and isinstance(source_scale, (int, float))
            and math.isfinite(float(source_scale))
            and float(source_scale) == float(scale)
            and isinstance(source_participation, dict)
            and set(source_participation.get("finger_names", ())) == set(finger_names)
        )
        source_is_positive = (
            source.get("active_side") == expected_side
            and source.get("success") is True
            and source.get("simulation_success") is True
            and isinstance(validation, dict)
            and validation.get("status") == "passed"
            and validation.get("backend") == DUAL_VALIDATION_BACKEND
            and source_matches_geometry_and_ownership
        )
        if not source_is_positive and not (
            allow_unvalidated_tabletop_source
            and source_matches_geometry_and_ownership
            and source.get("success") is False
            and source.get("simulation_success") is False
            and isinstance(validation, dict)
            and validation.get("status") == "not_run"
        ):
            raise X2DualValidationError(
                f"{path}: {expected_slot} source is not the claimed passed grasp"
            )
    result = copy.deepcopy(value)
    result["mesh_path"] = str(mesh_path)
    result["source_valid"] = str(source_path)
    result["source_valid_sha256_recorded"] = source_sha
    result["source_validation_status"] = (
        "SEMANTIC_VALIDATED"
        if source_record_available and source_is_positive
        else "UNVALIDATED_TABLETOP_RAW_FOR_PHYSX_QUALIFICATION_ONLY"
        if source_record_available and allow_unvalidated_tabletop_source
        else "PROVENANCE_RECORD_UNAVAILABLE"
    )
    result["source_valid_sha256_observed"] = observed_source_sha
    result["source_valid_sha256_matches_recorded"] = (
        None if observed_source_sha is None else observed_source_sha == source_sha
    )
    result["scale"] = float(scale)
    result["finger_names"] = sorted(finger_names)
    result["pose_in_shared_hand_frame"] = _pose_record(
        value.get("pose_in_shared_hand_frame"),
        label=f"objects[{expected_slot}].pose_in_shared_hand_frame",
        path=path,
    )
    return result


def load_candidate(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> X2DualObjectCandidate:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise X2DualValidationError(f"candidate is missing: {path}")
    sha256 = file_sha256(path)
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise X2DualValidationError(f"candidate hash is stale: {path}")
    record = strict_json(path)
    is_tabletop_repair = record.get("protocol_revision") == TABLETOP_REPAIR_UNVALIDATED_PROTOCOL_REVISION
    if (
        record.get("schema_version") != 1
        or record.get("protocol_revision") not in {
            DUAL_COMPOSITION_PROTOCOL_REVISION,
            TABLETOP_REPAIR_UNVALIDATED_PROTOCOL_REVISION,
        }
        or not isinstance(record.get("candidate_id"), str)
        or record.get("dual_object_validation", {}).get("status") != "not_run"
    ):
        raise X2DualValidationError(f"{path}: candidate protocol/status is stale")
    if is_tabletop_repair and record.get("candidate_kind") != "UNVALIDATED_TABLETOP_REPAIR_PAIR":
        raise X2DualValidationError(f"{path}: tabletop repair candidate kind is malformed")
    _audit_hand(record, path)
    objects = record.get("objects")
    if not isinstance(objects, list) or len(objects) != 2:
        raise X2DualValidationError(f"{path}: candidate must contain two objects")
    right = _audit_object(
        objects[0], expected_slot="right", expected_side="front", path=path,
        allow_unvalidated_tabletop_source=is_tabletop_repair,
    )
    left = _audit_object(
        objects[1], expected_slot="left", expected_side="back", path=path,
        allow_unvalidated_tabletop_source=is_tabletop_repair,
    )
    right_fingers = set(right["finger_names"])
    left_fingers = set(left["finger_names"])
    checks = record.get("composition_checks")
    if (
        right["object_id"] == left["object_id"]
        or right_fingers & left_fingers
        or right_fingers | left_fingers != FINGER_NAMES
        or not isinstance(checks, dict)
        or checks.get("different_objects") is not True
        or checks.get("disjoint_finger_sets") is not True
        or checks.get("all_five_fingers_assigned") is not True
    ):
        raise X2DualValidationError(
            f"{path}: objects/finger sets are not a valid cross-object composition"
        )
    normalized = copy.deepcopy(record)
    normalized["objects"] = [right, left]
    return X2DualObjectCandidate(
        path=path,
        sha256=sha256,
        record=normalized,
        right=right,
        left=left,
    )


def discover_candidates(
    dataset_root: Path,
    *,
    require_complete: bool = True,
    limit: int | None = None,
    combination: str | None = None,
) -> tuple[list[X2DualObjectCandidate], dict[str, Any]]:
    dataset_root = dataset_root.expanduser().resolve()
    manifest_path = dataset_root / "manifest.json"
    manifest = strict_json(manifest_path)
    if (
        manifest.get("protocol_revision") != DUAL_COMPOSITION_PROTOCOL_REVISION
        or manifest.get("dual_object_status") != "not_validated"
    ):
        raise X2DualValidationError("dual-object manifest protocol/status is stale")
    descriptors = manifest.get("dual_object_candidates")
    if (
        not isinstance(descriptors, list)
        or manifest.get("dual_object_candidate_count") != len(descriptors)
    ):
        raise X2DualValidationError("dual-object manifest candidate count is stale")
    if require_complete:
        counts: dict[tuple[int, int], int] = {}
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                raise X2DualValidationError("dual-object descriptor is malformed")
            key = (
                descriptor.get("right_finger_count"),
                descriptor.get("left_finger_count"),
            )
            counts[key] = counts.get(key, 0) + 1
        expected = {(1, 4): 500, (2, 3): 500, (3, 2): 500, (4, 1): 500}
        if (
            manifest.get("formal_source_completion_required") is not True
            or len(descriptors) != 2000
            or counts != expected
        ):
            raise X2DualValidationError(
                "dual-object composition is not complete: require four 500-pair "
                "strata from completed attempts"
            )
    candidates: list[X2DualObjectCandidate] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise X2DualValidationError("dual-object descriptor is malformed")
        candidate_id = descriptor.get("candidate_id")
        path_value = descriptor.get("path")
        sha256 = descriptor.get("sha256")
        if combination is not None and Path(str(path_value)).parent.name != combination:
            continue
        if (
            not isinstance(candidate_id, str)
            or candidate_id in seen_ids
            or not isinstance(path_value, str)
            or not isinstance(sha256, str)
        ):
            raise X2DualValidationError(
                "dual-object descriptor id/path/hash is malformed or duplicated"
            )
        path = Path(path_value).expanduser().resolve()
        candidate_root = (dataset_root / "dual_object_candidates").resolve()
        if candidate_root not in path.parents or path in seen_paths:
            raise X2DualValidationError(
                f"candidate path escapes root or is duplicated: {path}"
            )
        candidate = load_candidate(path, expected_sha256=sha256)
        if candidate.candidate_id != candidate_id:
            raise X2DualValidationError(f"candidate id mismatch: {path}")
        candidates.append(candidate)
        seen_ids.add(candidate_id)
        seen_paths.add(path)
        if limit is not None and len(candidates) >= limit:
            break
    if not candidates:
        selector = f" for combination {combination}" if combination is not None else ""
        raise X2DualValidationError(f"no dual-object candidates selected{selector}")
    return candidates, manifest


def group_candidates_by_objects(
    candidates: Iterable[X2DualObjectCandidate],
) -> dict[
    tuple[tuple[Path, float], tuple[Path, float]],
    list[X2DualObjectCandidate],
]:
    grouped: dict[
        tuple[tuple[Path, float], tuple[Path, float]],
        list[X2DualObjectCandidate],
    ] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.object_group, []).append(candidate)
    return grouped


def gravity_vectors_shared_hand(gravity_magnitude: float = 9.8) -> np.ndarray:
    if not math.isfinite(gravity_magnitude) or gravity_magnitude <= 0.0:
        raise X2DualValidationError(
            "gravity magnitude must be finite and positive"
        )
    world = np.asarray((0.0, -gravity_magnitude, 0.0), dtype=np.float64)
    return np.stack(
        [
            quaternion_matrix_wxyz(quaternion).T @ world
            for _, quaternion in GRAVITY_TESTS_WXYZ
        ]
    )


def dual_validation_output_path(
    candidate: X2DualObjectCandidate,
    output_root: Path,
    *,
    passed: bool,
) -> Path:
    route = "valid" if passed else "failed"
    return (
        output_root.expanduser().resolve()
        / route
        / candidate.combination
        / candidate.path.name
    )


def existing_validation_output(
    candidate: X2DualObjectCandidate,
    output_root: Path,
    *,
    protocol_revision: str = DUAL_VALIDATION_PROTOCOL_REVISION,
) -> Path | None:
    valid = dual_validation_output_path(candidate, output_root, passed=True)
    failed = dual_validation_output_path(candidate, output_root, passed=False)
    existing = [path for path in (valid, failed) if path.is_file()]
    if not existing:
        return None
    if len(existing) != 1:
        raise X2DualValidationError(
            f"candidate is routed to both valid and failed: {candidate.candidate_id}"
        )
    payload = strict_json(existing[0])
    validation = payload.get("dual_object_validation")
    expected_status = "passed" if existing[0] == valid else "failed"
    if (
        not isinstance(validation, dict)
        or validation.get("status") != expected_status
        or validation.get("backend") != DUAL_VALIDATION_BACKEND
        or validation.get("protocol_revision") != protocol_revision
        or validation.get("source_candidate_sha256") != candidate.sha256
        or validation.get("source_candidate") != str(candidate.path)
    ):
        raise X2DualValidationError(
            f"existing dual validation output is stale: {existing[0]}"
        )
    return existing[0]


def make_validation_record(
    candidate: X2DualObjectCandidate,
    *,
    passed: bool,
    simulation_ran: bool,
    static_preflight: Mapping[str, Any],
    orientations: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    failure_reasons: Sequence[str],
    protocol_revision: str = DUAL_VALIDATION_PROTOCOL_REVISION,
) -> dict[str, Any]:
    status = "passed" if passed else "failed"
    if passed and (
        not simulation_ran
        or len(orientations) != len(EXPECTED_GRAVITY_NAMES)
        or tuple(value.get("name") for value in orientations)
        != EXPECTED_GRAVITY_NAMES
        or not all(value.get("passed") is True for value in orientations)
        or failure_reasons
    ):
        raise X2DualValidationError(
            f"{candidate.candidate_id}: incomplete proof cannot be marked passed"
        )
    record = copy.deepcopy(candidate.record)
    record["dual_object_success"] = passed
    record["dual_object_validation"] = {
        "status": status,
        "backend": DUAL_VALIDATION_BACKEND,
        "protocol_revision": protocol_revision,
        "criterion": DUAL_VALIDATION_CRITERION,
        "source_candidate": str(candidate.path),
        "source_candidate_sha256": candidate.sha256,
        "simulation_ran": simulation_ran,
        "static_preflight": copy.deepcopy(dict(static_preflight)),
        "orientations": [copy.deepcopy(dict(value)) for value in orientations],
        "passed_orientation_count": sum(
            value.get("passed") is True for value in orientations
        ),
        "required_orientation_count": len(EXPECTED_GRAVITY_NAMES),
        "failure_reasons": list(failure_reasons),
        "runtime": copy.deepcopy(dict(runtime)),
    }
    return record


def write_validation_record(
    candidate: X2DualObjectCandidate,
    record: Mapping[str, Any],
    output_root: Path,
    *,
    overwrite: bool = False,
) -> Path:
    validation = record.get("dual_object_validation")
    if not isinstance(validation, Mapping) or validation.get("status") not in {
        "passed",
        "failed",
    }:
        raise X2DualValidationError("dual validation record status is invalid")
    passed = validation["status"] == "passed"
    destination = dual_validation_output_path(candidate, output_root, passed=passed)
    opposite = dual_validation_output_path(candidate, output_root, passed=not passed)
    if opposite.exists() and not overwrite:
        raise X2DualValidationError(
            f"opposite dual validation route exists: {opposite}"
        )
    if destination.exists() and not overwrite:
        raise X2DualValidationError(
            f"dual validation output already exists: {destination}"
        )
    if overwrite:
        opposite.unlink(missing_ok=True)
    atomic_json(destination, record)
    return destination


__all__ = [
    "DUAL_COMPOSITION_PROTOCOL_REVISION",
    "DUAL_VALIDATION_BACKEND",
    "DUAL_VALIDATION_CRITERION",
    "DUAL_VALIDATION_PROTOCOL_REVISION",
    "DUAL_VALIDATION_PROTOCOL_REVISION_OBJECT_COLLISION",
    "EXPECTED_GRAVITY_NAMES",
    "X2DualObjectCandidate",
    "X2DualValidationError",
    "atomic_json",
    "discover_candidates",
    "dual_validation_output_path",
    "existing_validation_output",
    "file_sha256",
    "gravity_vectors_shared_hand",
    "group_candidates_by_objects",
    "load_candidate",
    "make_validation_record",
    "strict_json",
    "write_validation_record",
]
