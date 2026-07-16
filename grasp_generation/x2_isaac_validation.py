"""Pure-data support for X2 mesh-grasp validation in Isaac Sim/PhysX.

This module deliberately has no Isaac Sim imports.  The simulator entry point lives in
``scripts/validate_x2_mesh_grasps.py``; keeping schema parsing, frame conversion and
result routing here makes those contracts unit-testable without launching Kit.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
VALIDATION_BACKEND = "isaac_sim_physx"
PROTOCOL_REVISION = "x2_object_centered_dexgraspnet_six_orientation_v7"
VALIDATION_CRITERIA = ("dexgraspnet-contact", "strict-hold")
X2_MESH_PIPELINE_REVISION_PREFIX = "x2_mesh_grasp_unselected_finger_side_v"
CLOSING_LINE_SEARCH_ALPHAS = tuple(0.5**index for index in range(9)) + (0.0,)
MAX_CLOSING_TARGET_PENETRATION_CAP = 0.002
FORMAL_ACTUATOR_STIFFNESS = 1000.0
FORMAL_ACTUATOR_DAMPING = 0.632455532
FORMAL_ACTUATOR_ARMATURE = 0.0001
V5_DENSE_HAND_SURFACE_SAMPLES_PER_SET = 256
V5_DENSE_HAND_SURFACE_SAMPLES_PER_LINK = 768
V5_DENSE_HAND_SURFACE_POINT_COUNT = 13056
V5_DENSE_OBJECT_SURFACE_SAMPLES = 8192
V5_DENSE_HAND_OBJECT_THRESHOLD = 0.001

EXPECTED_ACTUATOR_NAMES = (
    "rh_LFJ3",
    "rh_RFJ3",
    "rh_MFJ3",
    "rh_FFJ3",
    "rh_LFJ2",
    "rh_RFJ2",
    "rh_MFJ2",
    "rh_FFJ2",
    "rh_THJ4",
    "rh_THJ3",
    "rh_THJ2",
    "rh_THJ1",
)

EXPECTED_JOINT_NAMES = (
    "rh_LFJ3",
    "rh_RFJ3",
    "rh_MFJ3",
    "rh_FFJ3",
    "rh_THJ4",
    "rh_LFJ2",
    "rh_RFJ2",
    "rh_MFJ2",
    "rh_FFJ2",
    "rh_THJ3",
    "rh_LFJ1",
    "rh_RFJ1",
    "rh_MFJ1",
    "rh_FFJ1",
    "rh_THJ2",
    "rh_THJ1",
)

PASSIVE_MIMIC_DRIVERS = {
    "rh_LFJ1": "rh_LFJ2",
    "rh_RFJ1": "rh_RFJ2",
    "rh_MFJ1": "rh_MFJ2",
    "rh_FFJ1": "rh_FFJ2",
}

_SQRT_HALF = math.sqrt(0.5)
GRAVITY_TESTS_WXYZ: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("identity", (1.0, 0.0, 0.0, 0.0)),
    ("z_180", (0.0, 0.0, 0.0, 1.0)),
    ("z_pos_90", (_SQRT_HALF, 0.0, 0.0, _SQRT_HALF)),
    ("z_neg_90", (_SQRT_HALF, 0.0, 0.0, -_SQRT_HALF)),
    ("x_pos_90", (_SQRT_HALF, _SQRT_HALF, 0.0, 0.0)),
    ("x_neg_90", (_SQRT_HALF, -_SQRT_HALF, 0.0, 0.0)),
)


class X2ValidationError(RuntimeError):
    """Raised when an X2 raw candidate or validation result is invalid."""


@dataclass(frozen=True)
class ValidationThresholds:
    """Physical and static thresholds used by the six-orientation protocol."""

    penetration: float = 0.001
    retention_distance: float = 0.1
    contact_force: float = 1.0e-4
    joint_error: float = 0.1
    mimic_error: float = 0.01

    def __post_init__(self) -> None:
        values = (
            self.penetration,
            self.retention_distance,
            self.contact_force,
            self.joint_error,
            self.mimic_error,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise X2ValidationError("validation thresholds must be finite and non-negative")
        if self.penetration <= 0.0 or self.retention_distance <= 0.0:
            raise X2ValidationError("penetration and retention-distance thresholds must be positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "maximum_penetration_m": self.penetration,
            "maximum_object_displacement_m": self.retention_distance,
            "minimum_final_contact_force_n": self.contact_force,
            "maximum_active_joint_error_rad": self.joint_error,
            "maximum_newton_mimic_error_rad": self.mimic_error,
        }


@dataclass(frozen=True)
class CollisionAwareClosingSelection:
    """Per-row decision for a descending collision-aware closing line search."""

    selected_indices: np.ndarray
    selected_alphas: np.ndarray
    raw_penetration_passed: np.ndarray
    raw_state_finite: np.ndarray
    raw_actuator_limits_passed: np.ndarray
    raw_static_gate_passed: np.ndarray


def select_collision_aware_closing(
    maximum_penetration: Sequence[Sequence[float]] | np.ndarray,
    trial_finite: Sequence[Sequence[bool]] | np.ndarray,
    actuator_limits_passed: Sequence[Sequence[bool]] | np.ndarray,
    *,
    raw_penetration_cap: float,
    target_penetration_cap: float,
    alphas: Sequence[float] = CLOSING_LINE_SEARCH_ALPHAS,
) -> CollisionAwareClosingSelection:
    """Select the largest safe positive alpha independently for every row.

    Columns must follow descending positive ``alphas`` and end in the exact raw
    state ``alpha=0``.  A raw state that is non-finite, outside actuator limits,
    or not strictly below ``raw_penetration_cap`` is never repaired by closing:
    it remains at raw and is marked for the validator's static rejection gate.
    Positive target trials use the independent ``target_penetration_cap``.
    """

    raw_cap = float(raw_penetration_cap)
    target_cap = float(target_penetration_cap)
    if not math.isfinite(raw_cap) or raw_cap <= 0.0:
        raise X2ValidationError("raw penetration cap must be finite and positive")
    if (
        not math.isfinite(target_cap)
        or target_cap <= 0.0
        or target_cap > MAX_CLOSING_TARGET_PENETRATION_CAP
    ):
        raise X2ValidationError(
            "target penetration cap must be finite, positive, and at most 2 mm"
        )
    alpha_values = np.asarray(tuple(alphas), dtype=np.float64)
    if (
        alpha_values.ndim != 1
        or alpha_values.size < 2
        or not np.isfinite(alpha_values).all()
        or alpha_values[-1] != 0.0
        or np.any(alpha_values[:-1] <= 0.0)
        or np.any(alpha_values[:-2] <= alpha_values[1:-1])
    ):
        raise X2ValidationError(
            "closing alphas must be finite descending positive values followed by zero"
        )

    penetration = np.asarray(maximum_penetration, dtype=np.float64)
    finite = np.asarray(trial_finite, dtype=np.bool_)
    limits = np.asarray(actuator_limits_passed, dtype=np.bool_)
    expected_shape = (penetration.shape[0], alpha_values.size) if penetration.ndim == 2 else None
    if (
        penetration.ndim != 2
        or penetration.shape[0] <= 0
        or penetration.shape != expected_shape
        or finite.shape != penetration.shape
        or limits.shape != penetration.shape
    ):
        raise X2ValidationError(
            "closing trial matrices must share shape (batch, number_of_alphas)"
        )

    measured_finite = finite & np.isfinite(penetration)
    target_penetration_passed = measured_finite & (penetration < target_cap)
    safe = target_penetration_passed & limits
    raw_index = alpha_values.size - 1
    raw_penetration_passed = (
        measured_finite[:, raw_index]
        & (penetration[:, raw_index] < raw_cap)
    )
    raw_state_finite = measured_finite[:, raw_index]
    raw_actuator_limits_passed = limits[:, raw_index]
    raw_static_gate_passed = (
        raw_penetration_passed & raw_state_finite & raw_actuator_limits_passed
    )

    selected = np.full(penetration.shape[0], raw_index, dtype=np.int64)
    for row in np.flatnonzero(raw_static_gate_passed):
        accepted = np.flatnonzero(safe[row, :raw_index])
        if accepted.size:
            selected[row] = int(accepted[0])
    return CollisionAwareClosingSelection(
        selected_indices=selected,
        selected_alphas=alpha_values[selected],
        raw_penetration_passed=raw_penetration_passed,
        raw_state_finite=raw_state_finite,
        raw_actuator_limits_passed=raw_actuator_limits_passed,
        raw_static_gate_passed=raw_static_gate_passed,
    )


@dataclass(frozen=True)
class X2RawCandidate:
    """A schema-checked raw candidate in the generator's object/world frame."""

    path: Path
    record: dict[str, Any]
    mesh_path: Path
    object_scale: float
    active_side: str
    hand_translation: np.ndarray
    hand_quaternion_wxyz: np.ndarray
    actuator_by_name: Mapping[str, float]
    joint_by_name: Mapping[str, float]
    maximum_penetration: float
    self_collision_gate_required: bool = False
    self_collision_feasible: bool | None = None
    hand_object_gate_required: bool = False
    hand_object_feasible: bool | None = None
    source_bytes: bytes = b""
    source_sha256: str = ""

    @property
    def preflight_finite(self) -> bool:
        return bool(self.record.get("finite", False))


def _requires_self_collision_gate(pipeline_revision: Any) -> bool:
    """Return whether a generator revision carries the v4 self-collision contract."""

    if not isinstance(pipeline_revision, str):
        return False
    match = re.fullmatch(
        rf"{re.escape(X2_MESH_PIPELINE_REVISION_PREFIX)}([0-9]+)",
        pipeline_revision,
    )
    return bool(match and int(match.group(1)) >= 4)


def _requires_v5_dense_hand_object_gate(pipeline_revision: Any) -> bool:
    if not isinstance(pipeline_revision, str):
        return False
    match = re.fullmatch(
        rf"{re.escape(X2_MESH_PIPELINE_REVISION_PREFIX)}([0-9]+)",
        pipeline_revision,
    )
    return bool(match and int(match.group(1)) >= 5)


def _validate_self_collision_record(
    record: Mapping[str, Any], source: Path
) -> tuple[bool, bool | None]:
    """Validate the v4 static hull diagnostic without changing legacy v3 semantics."""

    gate_required = _requires_self_collision_gate(record.get("pipeline_revision"))
    value = record.get("self_collision")
    if value is None:
        if gate_required:
            raise X2ValidationError(
                f"{source}: v4+ self_collision diagnostics are required"
            )
        return False, None
    if not isinstance(value, dict):
        raise X2ValidationError(f"{source}: self_collision must be a JSON object")

    feasible = value.get("feasible")
    if not isinstance(feasible, bool):
        raise X2ValidationError(f"{source}: self_collision.feasible must be boolean")

    numeric: dict[str, float] = {}
    for key in ("maximum_penetration", "total_penetration", "threshold"):
        raw_number = value.get(key)
        try:
            if isinstance(raw_number, bool):
                raise TypeError
            number = float(raw_number)
        except (TypeError, ValueError) as exc:
            raise X2ValidationError(
                f"{source}: self_collision.{key} must be finite and non-negative"
            ) from exc
        if not math.isfinite(number) or number < 0.0:
            raise X2ValidationError(
                f"{source}: self_collision.{key} must be finite and non-negative"
            )
        numeric[key] = number

    worst_pair = value.get("worst_pair")
    if worst_pair is not None and not (
        isinstance(worst_pair, list)
        and len(worst_pair) == 2
        and all(isinstance(name, str) and name for name in worst_pair)
    ):
        raise X2ValidationError(
            f"{source}: self_collision.worst_pair must be null or two non-empty link names"
        )
    expected_feasible = numeric["maximum_penetration"] <= numeric["threshold"]
    if feasible is not expected_feasible:
        raise X2ValidationError(
            f"{source}: self_collision.feasible disagrees with maximum_penetration <= threshold"
        )
    return gate_required, feasible


def _validate_v5_dense_hand_object_record(
    record: Mapping[str, Any], source: Path, top_level_maximum: float
) -> tuple[bool, bool | None]:
    """Validate the exact v5 dense bidirectional hand/object gate contract."""

    required = _requires_v5_dense_hand_object_gate(
        record.get("pipeline_revision")
    )
    if not required:
        return False, None
    value = record.get("hand_object_penetration")
    if not isinstance(value, Mapping):
        raise X2ValidationError(
            f"{source}: v5 hand_object_penetration diagnostics are required"
        )
    if value.get("evaluation_mode") != "dense_bidirectional":
        raise X2ValidationError(
            f"{source}: hand_object_penetration.evaluation_mode must be dense_bidirectional"
        )
    if value.get("evaluated") is not True:
        raise X2ValidationError(
            f"{source}: hand_object_penetration.evaluated must be true"
        )

    expected_counts = {
        "hand_surface_samples_per_set": V5_DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        "hand_surface_samples_per_link": V5_DENSE_HAND_SURFACE_SAMPLES_PER_LINK,
        "hand_surface_point_count": V5_DENSE_HAND_SURFACE_POINT_COUNT,
        "object_surface_samples": V5_DENSE_OBJECT_SURFACE_SAMPLES,
    }
    for key, expected in expected_counts.items():
        actual = value.get(key)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise X2ValidationError(
                f"{source}: hand_object_penetration.{key} must equal {expected}"
            )

    numeric: dict[str, float] = {}
    numeric_keys = (
        "forward_total_penetration",
        "forward_maximum_penetration",
        "reverse_total_penetration",
        "reverse_maximum_penetration",
        "total_penetration",
        "maximum_penetration",
        "threshold",
    )
    for key in numeric_keys:
        raw_number = value.get(key)
        try:
            if isinstance(raw_number, bool):
                raise TypeError
            number = float(raw_number)
        except (TypeError, ValueError) as exc:
            raise X2ValidationError(
                f"{source}: hand_object_penetration.{key} must be finite and non-negative"
            ) from exc
        if not math.isfinite(number) or number < 0.0:
            raise X2ValidationError(
                f"{source}: hand_object_penetration.{key} must be finite and non-negative"
            )
        numeric[key] = number
    if numeric["threshold"] != V5_DENSE_HAND_OBJECT_THRESHOLD:
        raise X2ValidationError(
            f"{source}: hand_object_penetration.threshold must equal 0.001"
        )

    def numerically_equal(actual: float, expected: float) -> bool:
        return math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-15)

    expected_total = (
        numeric["forward_total_penetration"]
        + numeric["reverse_total_penetration"]
    )
    if not numerically_equal(numeric["total_penetration"], expected_total):
        raise X2ValidationError(
            f"{source}: hand_object_penetration.total_penetration is inconsistent"
        )
    expected_maximum = max(
        numeric["forward_maximum_penetration"],
        numeric["reverse_maximum_penetration"],
    )
    if not numerically_equal(numeric["maximum_penetration"], expected_maximum):
        raise X2ValidationError(
            f"{source}: hand_object_penetration.maximum_penetration is inconsistent"
        )
    if not numerically_equal(top_level_maximum, numeric["maximum_penetration"]):
        raise X2ValidationError(
            f"{source}: top-level maximum_penetration disagrees with dense diagnostics"
        )
    feasible = value.get("feasible")
    if not isinstance(feasible, bool):
        raise X2ValidationError(
            f"{source}: hand_object_penetration.feasible must be boolean"
        )
    expected_feasible = bool(
        value["evaluated"]
        and numeric["maximum_penetration"] < V5_DENSE_HAND_OBJECT_THRESHOLD
    )
    if feasible is not expected_feasible:
        raise X2ValidationError(
            f"{source}: hand_object_penetration.feasible is inconsistent"
        )

    optimization = record.get("optimization")
    if not isinstance(optimization, Mapping):
        raise X2ValidationError(f"{source}: v5 optimization diagnostics are required")
    restored_checkpoint = optimization.get("restored_checkpoint")
    allowed_checkpoints = (
        "bidirectional_feasible",
        "bidirectional_fallback",
        "fallback",
    )
    if restored_checkpoint not in allowed_checkpoints:
        raise X2ValidationError(
            f"{source}: optimization.restored_checkpoint is invalid for v5"
        )
    restored_step = optimization.get("restored_step")
    if (
        isinstance(restored_step, bool)
        or not isinstance(restored_step, int)
        or restored_step < 0
    ):
        raise X2ValidationError(
            f"{source}: optimization.restored_step must be a non-negative integer"
        )
    checkpoint_found = optimization.get(
        "bidirectional_feasible_checkpoint_found"
    )
    if not isinstance(checkpoint_found, bool) or checkpoint_found is not feasible:
        raise X2ValidationError(
            f"{source}: optimization.bidirectional_feasible_checkpoint_found "
            "must equal dense feasibility"
        )
    if feasible and restored_checkpoint != "bidirectional_feasible":
        raise X2ValidationError(
            f"{source}: a feasible v5 record must restore bidirectional_feasible"
        )
    if not feasible and restored_checkpoint == "bidirectional_feasible":
        raise X2ValidationError(
            f"{source}: an infeasible v5 record cannot restore bidirectional_feasible"
        )

    optimization_counts = {
        "dense_hand_surface_samples_per_set": V5_DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        "dense_hand_surface_samples_per_link": V5_DENSE_HAND_SURFACE_SAMPLES_PER_LINK,
        "dense_hand_surface_point_count": V5_DENSE_HAND_SURFACE_POINT_COUNT,
        "dense_object_surface_samples": V5_DENSE_OBJECT_SURFACE_SAMPLES,
    }
    for key, expected in optimization_counts.items():
        actual = optimization.get(key)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise X2ValidationError(
                f"{source}: optimization.{key} must equal {expected}"
            )
    for key in (
        "dense_bidirectional_query_calls",
        "dense_bidirectional_rows_evaluated",
    ):
        actual = optimization.get(key)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual < 1:
            raise X2ValidationError(
                f"{source}: optimization.{key} must be a positive integer"
            )
    return True, feasible


@dataclass(frozen=True)
class ObjectCenteredReplay:
    """Generator-aligned replay with the object fixed at each environment origin."""

    hand_translation: np.ndarray
    hand_quaternion_xyzw: np.ndarray
    object_translation: np.ndarray
    object_quaternion_xyzw: np.ndarray
    gravity_names: tuple[str, ...]
    gravity_vectors: np.ndarray


@dataclass(frozen=True)
class OrientationOutcome:
    """Measured result of one gravity orientation."""

    name: str
    passed: bool
    final_displacement: float
    maximum_displacement: float
    final_contact_force: float
    maximum_active_joint_error: float
    finite: bool
    gravity_vector_object_frame: tuple[float, float, float] | None = None
    hand_object_contact: bool | None = None
    maximum_newton_mimic_error: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        def measurement(value: float) -> float | None:
            numeric = float(value)
            return numeric if math.isfinite(numeric) else None

        measurements_finite = all(
            math.isfinite(float(value))
            for value in (
                self.final_displacement,
                self.maximum_displacement,
                self.final_contact_force,
                self.maximum_active_joint_error,
                self.maximum_newton_mimic_error,
            )
        )
        result = {
            "name": self.name,
            "passed": bool(
                self.passed
                and self.finite
                and measurements_finite
                and self.hand_object_contact is not False
            ),
            "final_displacement_m": measurement(self.final_displacement),
            "maximum_displacement_m": measurement(self.maximum_displacement),
            "final_contact_force_n": measurement(self.final_contact_force),
            "maximum_active_joint_error_rad": measurement(self.maximum_active_joint_error),
            "maximum_newton_mimic_error_rad": measurement(
                self.maximum_newton_mimic_error
            ),
            "finite": bool(self.finite and measurements_finite),
            "hand_object_contact": (
                bool(self.hand_object_contact)
                if self.hand_object_contact is not None
                else None
            ),
        }
        if self.gravity_vector_object_frame is not None:
            gravity = _finite_vector(
                self.gravity_vector_object_frame, 3, "gravity_vector_object_frame"
            )
            result["gravity_vector_object_frame_m_s2"] = [float(value) for value in gravity]
        _require_json_safe(result, "orientation outcome")
        return result


def _reject_json_constant(value: str) -> None:
    raise X2ValidationError(f"non-finite JSON constant is forbidden: {value}")


def _require_json_safe(value: Any, label: str) -> None:
    """Reject values that cannot be emitted with ``allow_nan=False``."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise X2ValidationError(f"{label} contains NaN or infinity")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise X2ValidationError(f"{label} contains a non-string JSON key")
            _require_json_safe(item, label)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_json_safe(item, label)
        return
    raise X2ValidationError(f"{label} contains a non-JSON value: {type(value).__name__}")


def _finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,):
        raise X2ValidationError(f"{label} must have shape ({length},), got {array.shape}")
    if not np.isfinite(array).all():
        raise X2ValidationError(f"{label} contains NaN or infinity")
    return array.copy()


def _normalise_quaternion_wxyz(value: Any, label: str) -> np.ndarray:
    quaternion = _finite_vector(value, 4, label)
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > 1.0e-6:
        raise X2ValidationError(f"{label} is not a unit quaternion: norm={norm:.12g}")
    quaternion /= norm
    return quaternion


def _validate_selected_contacts(
    record: Mapping[str, Any], active_side: str, source: Path
) -> None:
    selected_ids = record.get("selected_contact_ids")
    if not isinstance(selected_ids, list) or not selected_ids:
        raise X2ValidationError(f"{source}: selected_contact_ids must be a non-empty list")
    if not all(isinstance(point_id, str) and point_id for point_id in selected_ids):
        raise X2ValidationError(
            f"{source}: selected_contact_ids must contain non-empty strings"
        )
    if len(selected_ids) != len(set(selected_ids)):
        raise X2ValidationError(f"{source}: selected_contact_ids must be unique")

    contacts = record.get("selected_contacts")
    if not isinstance(contacts, list):
        raise X2ValidationError(f"{source}: selected_contacts must be a list")
    if len(contacts) != len(selected_ids):
        raise X2ValidationError(
            f"{source}: selected_contacts count must match selected_contact_ids"
        )

    string_fields = ("point_id", "link_name", "finger_name", "region", "source")
    participating_fingers: set[str] = set()
    known_fingers = {"index", "middle", "ring", "little", "thumb"}
    for index, (expected_id, contact) in enumerate(zip(selected_ids, contacts)):
        label = f"selected_contacts[{index}]"
        if not isinstance(contact, dict):
            raise X2ValidationError(f"{source}: {label} must be a JSON object")
        for field in string_fields:
            value = contact.get(field)
            if not isinstance(value, str) or not value:
                raise X2ValidationError(
                    f"{source}: {label}.{field} must be a non-empty string"
                )
        if contact["point_id"] != expected_id:
            raise X2ValidationError(
                f"{source}: {label}.point_id must match selected_contact_ids order"
            )
        if contact.get("enabled") is not True:
            raise X2ValidationError(f"{source}: {label}.enabled must be true")

        _finite_vector(contact.get("local_position"), 3, f"{label}.local_position")
        _finite_vector(contact.get("world_position"), 3, f"{label}.world_position")
        for field in ("local_surface_normal", "world_surface_normal"):
            normal = _finite_vector(contact.get(field), 3, f"{label}.{field}")
            norm = float(np.linalg.norm(normal))
            if abs(norm - 1.0) > 1.0e-5:
                raise X2ValidationError(
                    f"{source}: {label}.{field} is not near-unit: norm={norm:.12g}"
                )

        supported_sides = contact.get("supported_sides")
        if (
            not isinstance(supported_sides, list)
            or not supported_sides
            or not all(side in ("front", "back") for side in supported_sides)
            or len(supported_sides) != len(set(supported_sides))
        ):
            raise X2ValidationError(
                f"{source}: {label}.supported_sides must contain unique front/back values"
            )
        if active_side not in supported_sides:
            raise X2ValidationError(
                f"{source}: {label} does not support active_side={active_side}"
            )
        if contact["finger_name"] in known_fingers:
            participating_fingers.add(contact["finger_name"])

    participation = record.get("finger_participation")
    if participation is not None:
        if not isinstance(participation, dict):
            raise X2ValidationError(
                f"{source}: finger_participation must be a JSON object"
            )
        target_count = participation.get("target_count")
        actual_count = participation.get("actual_count")
        finger_names = participation.get("finger_names")
        if (
            isinstance(target_count, bool)
            or not isinstance(target_count, int)
            or target_count < 1
            or target_count > 5
        ):
            raise X2ValidationError(
                f"{source}: finger_participation.target_count must be in 1..5"
            )
        expected_names = sorted(participating_fingers)
        if actual_count != len(participating_fingers) or target_count != actual_count:
            raise X2ValidationError(
                f"{source}: finger_participation counts do not match selected contacts"
            )
        if not isinstance(finger_names, list) or sorted(finger_names) != expected_names:
            raise X2ValidationError(
                f"{source}: finger_participation.finger_names do not match selected contacts"
            )


def _validate_raw_side_directory(source: Path, active_side: str) -> None:
    if source.parent.name != "raw":
        raise X2ValidationError(
            f"{source}: raw candidate must be directly inside a raw directory"
        )
    side_directory = source.parent.parent.name
    expected = (active_side, f"{active_side}_single")
    if side_directory not in expected:
        raise X2ValidationError(
            f"{source}: raw side directory {side_directory!r} does not match "
            f"active_side={active_side}"
        )


def quaternion_matrix_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    """Convert a scalar-first unit quaternion to a 3x3 rotation matrix."""

    w, x, y, z = _normalise_quaternion_wxyz(quaternion, "quaternion")
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def quaternion_wxyz_to_xyzw(quaternion: Sequence[float]) -> np.ndarray:
    """Convert generator/JSON wxyz to the current Isaac Lab root-state xyzw order."""

    w, x, y, z = _normalise_quaternion_wxyz(quaternion, "quaternion")
    return np.asarray((x, y, z, w), dtype=np.float64)


def load_raw_candidate(path: Path | str) -> X2RawCandidate:
    """Load and strictly validate one generator raw JSON record."""

    source = Path(path).expanduser().resolve()
    try:
        source_bytes = source.read_bytes()
        record = json.loads(
            source_bytes.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except X2ValidationError:
        raise
    except Exception as exc:
        raise X2ValidationError(f"could not read {source}: {exc}") from exc
    if not isinstance(record, dict):
        raise X2ValidationError(f"{source}: JSON root must be an object")
    _require_json_safe(record, str(source))

    if record.get("schema_version") != SCHEMA_VERSION:
        raise X2ValidationError(f"{source}: expected schema_version={SCHEMA_VERSION}")
    if record.get("active_side") not in ("front", "back"):
        raise X2ValidationError(f"{source}: active_side must be front or back")
    active_side = str(record["active_side"])
    _validate_raw_side_directory(source, active_side)
    if record.get("success") is not False or record.get("simulation_success") is not False:
        raise X2ValidationError(f"{source}: validator input must be an unvalidated raw candidate")
    validation = record.get("validation")
    if not isinstance(validation, dict) or validation.get("status") != "not_run":
        raise X2ValidationError(f"{source}: validation.status must be not_run")

    object_record = record.get("object")
    if not isinstance(object_record, dict):
        raise X2ValidationError(f"{source}: object must be a JSON object")
    mesh_value = object_record.get("mesh_path")
    if not isinstance(mesh_value, str) or not mesh_value:
        raise X2ValidationError(f"{source}: object.mesh_path must be a non-empty string")
    mesh_path = Path(mesh_value).expanduser().resolve()
    if not mesh_path.is_file():
        raise X2ValidationError(f"{source}: object mesh does not exist: {mesh_path}")
    object_scale = float(object_record.get("scale"))
    if not math.isfinite(object_scale) or object_scale <= 0.0:
        raise X2ValidationError(f"{source}: object.scale must be finite and positive")
    if object_record.get("watertight") is not True:
        raise X2ValidationError(f"{source}: object.watertight must be true")

    hand_pose = record.get("hand_pose")
    if not isinstance(hand_pose, dict):
        raise X2ValidationError(f"{source}: hand_pose must be a JSON object")
    hand_translation = _finite_vector(hand_pose.get("translation"), 3, "hand_pose.translation")
    hand_quaternion = _normalise_quaternion_wxyz(
        hand_pose.get("quaternion_wxyz"), "hand_pose.quaternion_wxyz"
    )
    rotation = np.asarray(hand_pose.get("rotation_matrix"), dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise X2ValidationError(f"{source}: hand_pose.rotation_matrix must be finite 3x3")
    if not np.allclose(rotation, quaternion_matrix_wxyz(hand_quaternion), atol=1.0e-6, rtol=0.0):
        raise X2ValidationError(f"{source}: quaternion and rotation_matrix disagree")

    actuator_names = record.get("actuator_names")
    joint_names = record.get("joint_names")
    if tuple(actuator_names or ()) != EXPECTED_ACTUATOR_NAMES:
        raise X2ValidationError(f"{source}: actuator_names order does not match X2 calibration")
    if tuple(joint_names or ()) != EXPECTED_JOINT_NAMES:
        raise X2ValidationError(f"{source}: joint_names order does not match X2 runtime order")
    actuator = _finite_vector(record.get("actuator"), len(EXPECTED_ACTUATOR_NAMES), "actuator")
    joint = _finite_vector(record.get("joint"), len(EXPECTED_JOINT_NAMES), "joint")
    actuator_by_name = dict(zip(EXPECTED_ACTUATOR_NAMES, map(float, actuator)))
    joint_by_name = dict(zip(EXPECTED_JOINT_NAMES, map(float, joint)))
    for name, value in actuator_by_name.items():
        if abs(value - joint_by_name[name]) > 1.0e-8:
            raise X2ValidationError(f"{source}: actuator/joint mismatch for {name}")
    for follower, driver in PASSIVE_MIMIC_DRIVERS.items():
        if abs(joint_by_name[follower] - actuator_by_name[driver]) > 1.0e-8:
            raise X2ValidationError(f"{source}: passive mimic mismatch for {follower}->{driver}")

    maximum_penetration = float(record.get("maximum_penetration"))
    if not math.isfinite(maximum_penetration) or maximum_penetration < 0.0:
        raise X2ValidationError(f"{source}: maximum_penetration must be finite and non-negative")
    self_collision_gate_required, self_collision_feasible = (
        _validate_self_collision_record(record, source)
    )
    hand_object_gate_required, hand_object_feasible = (
        _validate_v5_dense_hand_object_record(
            record, source, maximum_penetration
        )
    )
    if record.get("finite") is not True:
        raise X2ValidationError(f"{source}: finite must be true")

    _validate_selected_contacts(record, active_side, source)

    return X2RawCandidate(
        path=source,
        record=record,
        mesh_path=mesh_path,
        object_scale=object_scale,
        active_side=active_side,
        hand_translation=hand_translation,
        hand_quaternion_wxyz=hand_quaternion,
        actuator_by_name=actuator_by_name,
        joint_by_name=joint_by_name,
        maximum_penetration=maximum_penetration,
        self_collision_gate_required=self_collision_gate_required,
        self_collision_feasible=self_collision_feasible,
        hand_object_gate_required=hand_object_gate_required,
        hand_object_feasible=hand_object_feasible,
        source_bytes=source_bytes,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )


def discover_raw_candidates(
    input_root: Path | str,
    *,
    mesh_path: Path | str | None = None,
    side: str = "both",
    limit: int | None = None,
) -> list[X2RawCandidate]:
    """Discover stable ``raw/*.json`` records in deterministic path order."""

    root = Path(input_root).expanduser().resolve()
    if not root.is_dir():
        raise X2ValidationError(f"input root does not exist: {root}")
    if side not in ("front", "back", "both"):
        raise X2ValidationError("side must be front, back, or both")
    if limit is not None and limit <= 0:
        raise X2ValidationError("limit must be positive")
    mesh_filter = Path(mesh_path).expanduser().resolve() if mesh_path is not None else None

    candidates: list[X2RawCandidate] = []
    for path in sorted(root.glob("**/raw/*.json")):
        candidate = load_raw_candidate(path)
        if side != "both" and candidate.active_side != side:
            continue
        if mesh_filter is not None and candidate.mesh_path != mesh_filter:
            continue
        candidates.append(candidate)
        if limit is not None and len(candidates) >= limit:
            break
    if not candidates:
        description = f" under {root}"
        if mesh_filter is not None:
            description += f" for mesh {mesh_filter}"
        raise X2ValidationError(f"no raw candidates found{description}")
    return candidates


def group_candidates_by_mesh(
    candidates: Iterable[X2RawCandidate],
) -> dict[tuple[Path, float], list[X2RawCandidate]]:
    """Group candidates by the exact collision asset and uniform scale."""

    grouped: dict[tuple[Path, float], list[X2RawCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate.mesh_path, candidate.object_scale), []).append(candidate)
    return grouped


def make_object_centered_replay(
    candidate: X2RawCandidate, *, gravity_magnitude: float = 9.81
) -> ObjectCenteredReplay:
    """Express all six tests in the generator's object-centred frame.

    The object remains at the environment origin and X2 is placed at the JSON
    ``hand_pose``.  Rather than rotating the complete configuration for each
    official test, the mathematically equivalent gravity vector is expressed
    in the unchanged object frame.
    """

    if not math.isfinite(gravity_magnitude) or gravity_magnitude <= 0.0:
        raise X2ValidationError("gravity_magnitude must be finite and positive")
    world_gravity = np.asarray((0.0, -gravity_magnitude, 0.0), dtype=np.float64)
    names: list[str] = []
    gravity_vectors: list[np.ndarray] = []
    for name, test_quaternion in GRAVITY_TESTS_WXYZ:
        test_rotation = quaternion_matrix_wxyz(test_quaternion)
        names.append(name)
        gravity_vectors.append(test_rotation.T @ world_gravity)
    return ObjectCenteredReplay(
        hand_translation=candidate.hand_translation.copy(),
        hand_quaternion_xyzw=quaternion_wxyz_to_xyzw(
            candidate.hand_quaternion_wxyz
        ),
        object_translation=np.zeros(3, dtype=np.float64),
        object_quaternion_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64),
        gravity_names=tuple(names),
        gravity_vectors=np.stack(gravity_vectors),
    )


def evaluate_orientation(
    *,
    name: str,
    final_displacement: float,
    maximum_displacement: float,
    final_contact_force: float,
    maximum_active_joint_error: float,
    finite: bool,
    thresholds: ValidationThresholds,
    maximum_newton_mimic_error: float = 0.0,
    criterion: str = "dexgraspnet-contact",
    gravity_vector_object_frame: Sequence[float] | None = None,
    hand_object_contact: bool | None = None,
) -> OrientationOutcome:
    """Apply the source-compatible or optional stricter hold predicate."""

    if criterion not in VALIDATION_CRITERIA:
        raise X2ValidationError(f"unknown validation criterion: {criterion}")

    values = (
        final_displacement,
        maximum_displacement,
        final_contact_force,
        maximum_active_joint_error,
        maximum_newton_mimic_error,
    )
    measured_finite = bool(finite and all(math.isfinite(float(value)) for value in values))
    contact_passed = bool(
        measured_finite
        and (
            hand_object_contact
            if hand_object_contact is not None
            else final_contact_force > thresholds.contact_force
        )
    )
    strict_hold_passed = bool(
        contact_passed
        and maximum_newton_mimic_error <= thresholds.mimic_error
        and maximum_displacement < thresholds.retention_distance
        and final_displacement < thresholds.retention_distance
        and maximum_active_joint_error < thresholds.joint_error
    )
    source_compatible_passed = bool(
        contact_passed and maximum_newton_mimic_error <= thresholds.mimic_error
    )
    passed = source_compatible_passed if criterion == "dexgraspnet-contact" else strict_hold_passed
    return OrientationOutcome(
        name=name,
        passed=passed,
        final_displacement=float(final_displacement),
        maximum_displacement=float(maximum_displacement),
        final_contact_force=float(final_contact_force),
        maximum_active_joint_error=float(maximum_active_joint_error),
        finite=measured_finite,
        maximum_newton_mimic_error=float(maximum_newton_mimic_error),
        gravity_vector_object_frame=(
            tuple(float(value) for value in _finite_vector(
                gravity_vector_object_frame, 3, "gravity_vector_object_frame"
            ))
            if gravity_vector_object_frame is not None
            else None
        ),
        hand_object_contact=(
            bool(hand_object_contact) if hand_object_contact is not None else None
        ),
    )


def summarize_newton_mimic_errors(
    errors: Iterable[float],
    *,
    threshold: float,
    orientations_per_sample: int = len(GRAVITY_TESTS_WXYZ),
) -> dict[str, Any]:
    """Summarize finite tracking violations separately from solver blow-ups.

    A non-finite mimic measurement makes its orientation invalid, but it is not
    evidence that a finite tracking error exceeded ``threshold``.  Keeping the
    two cases separate prevents an all-non-finite batch from being reported as
    both "zero maximum error" and a batch of mimic tracking violations.
    """

    if not math.isfinite(float(threshold)) or float(threshold) < 0.0:
        raise X2ValidationError("mimic audit threshold must be finite and non-negative")
    if (
        isinstance(orientations_per_sample, bool)
        or not isinstance(orientations_per_sample, int)
        or orientations_per_sample <= 0
    ):
        raise X2ValidationError("orientations_per_sample must be a positive integer")

    values = np.asarray(tuple(errors), dtype=np.float64)
    if values.ndim != 1:
        raise X2ValidationError("mimic audit errors must be one-dimensional")
    if values.size % orientations_per_sample:
        raise X2ValidationError(
            "mimic audit error count must be divisible by orientations_per_sample"
        )

    finite = np.isfinite(values)
    violations = finite & (values > float(threshold))
    nonfinite = ~finite
    sample_count = values.size // orientations_per_sample
    violation_samples = violations.reshape(sample_count, orientations_per_sample).any(axis=1)
    nonfinite_samples = nonfinite.reshape(sample_count, orientations_per_sample).any(axis=1)
    maximum_finite_error = float(values[finite].max()) if finite.any() else None
    return {
        "maximum_newton_mimic_error_rad": maximum_finite_error,
        "newton_mimic_finite_orientation_count": int(finite.sum()),
        "newton_mimic_nonfinite_orientation_count": int(nonfinite.sum()),
        "newton_mimic_nonfinite_sample_count": int(nonfinite_samples.sum()),
        "newton_mimic_violation_orientation_count": int(violations.sum()),
        "newton_mimic_violation_sample_count": int(violation_samples.sum()),
    }


def _validate_collision_aware_closing_preflight(
    value: Mapping[str, Any], thresholds: ValidationThresholds
) -> tuple[dict[str, Any], bool]:
    """Validate one v6 sampled-closing audit and return its raw static gate."""

    if not isinstance(value, Mapping):
        raise X2ValidationError("collision-aware closing preflight must be a mapping")
    audit = copy.deepcopy(dict(value))
    _require_json_safe(audit, "collision-aware closing preflight")

    required_booleans = (
        "enabled",
        "bidirectional_sampled_penetration",
        "float32_quantized_and_rechecked",
        "raw_above_raw_cap",
        "raw_state_finite",
        "raw_actuator_limits_passed",
        "raw_penetration_passed",
        "raw_static_gate_passed",
        "selected_state_finite",
        "selected_actuator_limits_passed",
        "selected_penetration_passed",
        "selected_target_safe",
        "fell_back_to_raw",
    )
    for key in required_booleans:
        if not isinstance(audit.get(key), bool):
            raise X2ValidationError(
                f"collision-aware closing preflight {key} must be boolean"
            )
    if not audit["bidirectional_sampled_penetration"]:
        raise X2ValidationError("v6 closing preflight must use bidirectional penetration")
    if not audit["float32_quantized_and_rechecked"]:
        raise X2ValidationError("v6 closing target must be rechecked after float32 quantization")

    raw_cap = audit.get("raw_penetration_cap_m")
    target_cap = audit.get("target_penetration_cap_m")
    selected_alpha = audit.get("selected_alpha")
    if (
        isinstance(raw_cap, bool)
        or isinstance(target_cap, bool)
        or isinstance(selected_alpha, bool)
    ):
        raise X2ValidationError("closing caps and selected alpha must be numeric")
    try:
        raw_cap = float(raw_cap)
        target_cap = float(target_cap)
        selected_alpha = float(selected_alpha)
    except (TypeError, ValueError) as exc:
        raise X2ValidationError("closing caps and selected alpha must be finite") from exc
    expected_raw_cap = min(float(thresholds.penetration), 0.001)
    if not math.isfinite(raw_cap) or raw_cap != expected_raw_cap:
        raise X2ValidationError(
            f"raw penetration cap must equal {expected_raw_cap}, got {raw_cap}"
        )
    if (
        not math.isfinite(target_cap)
        or target_cap <= 0.0
        or target_cap > MAX_CLOSING_TARGET_PENETRATION_CAP
    ):
        raise X2ValidationError(
            "target penetration cap must be finite, positive, and at most 2 mm"
        )
    allowed_alphas = CLOSING_LINE_SEARCH_ALPHAS if audit["enabled"] else (0.0,)
    if not math.isfinite(selected_alpha) or selected_alpha not in allowed_alphas:
        raise X2ValidationError(
            f"collision-aware closing selected an invalid alpha: {selected_alpha}"
        )
    if audit["fell_back_to_raw"] is not (selected_alpha == 0.0):
        raise X2ValidationError("closing raw fallback flag disagrees with selected alpha")

    def optional_measurement(key: str) -> float | None:
        raw = audit.get(key)
        if raw is None:
            return None
        if isinstance(raw, bool):
            raise X2ValidationError(f"closing {key} must be numeric or null")
        try:
            numeric = float(raw)
        except (TypeError, ValueError) as exc:
            raise X2ValidationError(f"closing {key} must be numeric or null") from exc
        if not math.isfinite(numeric) or numeric < 0.0:
            raise X2ValidationError(f"closing {key} must be finite non-negative or null")
        return numeric

    raw_maximum = optional_measurement("raw_maximum_penetration_m")
    selected_maximum = optional_measurement("selected_maximum_penetration_m")
    expected_raw_penetration_passed = bool(
        raw_maximum is not None and raw_maximum < raw_cap
    )
    if audit["raw_penetration_passed"] is not expected_raw_penetration_passed:
        raise X2ValidationError("closing raw penetration flag disagrees with measurement")
    expected_raw_above = bool(
        raw_maximum is not None and raw_maximum >= raw_cap
    )
    if audit["raw_above_raw_cap"] is not expected_raw_above:
        raise X2ValidationError("closing raw-above-cap flag disagrees with measurement")
    expected_raw_gate = bool(
        audit["raw_state_finite"]
        and audit["raw_actuator_limits_passed"]
        and expected_raw_penetration_passed
    )
    if audit["raw_static_gate_passed"] is not expected_raw_gate:
        raise X2ValidationError("closing raw static gate is internally inconsistent")

    expected_selected_penetration_passed = bool(
        selected_maximum is not None and selected_maximum < target_cap
    )
    if (
        audit["selected_penetration_passed"]
        is not expected_selected_penetration_passed
    ):
        raise X2ValidationError("closing selected penetration flag disagrees with measurement")
    expected_selected_safe = bool(
        expected_raw_gate
        and audit["selected_state_finite"]
        and audit["selected_actuator_limits_passed"]
        and expected_selected_penetration_passed
    )
    if audit["selected_target_safe"] is not expected_selected_safe:
        raise X2ValidationError("closing selected target safety flag is inconsistent")
    if expected_raw_gate and not expected_selected_safe:
        raise X2ValidationError("safe raw state must yield a safe selected target")
    return audit, expected_raw_gate


def compact_collision_aware_closing_batch_audit(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove per-row payloads and retain bounded batch-level closing evidence."""

    if not isinstance(value, Mapping):
        raise X2ValidationError("collision-aware closing batch audit must be a mapping")
    compact = copy.deepcopy(dict(value))
    samples = compact.pop("samples", None)
    if not isinstance(samples, list) or not samples:
        raise X2ValidationError(
            "collision-aware closing batch audit requires a non-empty samples list"
        )
    selected_alpha_counts: dict[str, int] = {}
    raw_maxima: list[float] = []
    selected_maxima: list[float] = []
    full_maxima: list[float] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise X2ValidationError("closing batch samples must be mappings")
        for key in (
            "raw_static_gate_passed",
            "selected_target_safe",
            "raw_above_raw_cap",
            "fell_back_to_raw",
        ):
            if not isinstance(sample.get(key), bool):
                raise X2ValidationError(f"closing batch sample {key} must be boolean")
        alpha = sample.get("selected_alpha")
        if isinstance(alpha, bool):
            raise X2ValidationError("closing selected alpha must be numeric")
        try:
            alpha_value = float(alpha)
        except (TypeError, ValueError) as exc:
            raise X2ValidationError("closing selected alpha must be numeric") from exc
        if not math.isfinite(alpha_value):
            raise X2ValidationError("closing selected alpha must be finite")
        label = format(alpha_value, ".9g")
        selected_alpha_counts[label] = selected_alpha_counts.get(label, 0) + 1
        for key, destination in (
            ("raw_maximum_penetration_m", raw_maxima),
            ("selected_maximum_penetration_m", selected_maxima),
            ("full_target_maximum_penetration_m", full_maxima),
        ):
            measurement = sample.get(key)
            if measurement is None:
                continue
            if isinstance(measurement, bool):
                raise X2ValidationError(f"closing batch sample {key} must be numeric or null")
            numeric = float(measurement)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise X2ValidationError(
                    f"closing batch sample {key} must be finite non-negative or null"
                )
            destination.append(numeric)

    compact["batch_aggregate"] = {
        "sample_count": len(samples),
        "raw_static_gate_failed_count": sum(
            not sample["raw_static_gate_passed"] for sample in samples
        ),
        "selected_target_unsafe_count": sum(
            not sample["selected_target_safe"] for sample in samples
        ),
        "raw_above_raw_cap_count": sum(
            sample["raw_above_raw_cap"] for sample in samples
        ),
        "fell_back_to_raw_count": sum(
            sample["fell_back_to_raw"] for sample in samples
        ),
        "selected_alpha_counts": selected_alpha_counts,
        "maximum_raw_penetration_m": max(raw_maxima, default=None),
        "maximum_full_target_penetration_m": max(full_maxima, default=None),
        "maximum_selected_penetration_m": max(selected_maxima, default=None),
    }
    _require_json_safe(compact, "compact collision-aware closing batch audit")
    return compact


def make_validated_record(
    candidate: X2RawCandidate,
    outcomes: Sequence[OrientationOutcome],
    thresholds: ValidationThresholds,
    *,
    collision_aware_closing: Mapping[str, Any],
    runtime: Mapping[str, Any] | None = None,
    criterion: str = "dexgraspnet-contact",
) -> dict[str, Any]:
    """Create a routed result while leaving the raw source record untouched."""

    if criterion not in VALIDATION_CRITERIA:
        raise X2ValidationError(f"unknown validation criterion: {criterion}")
    expected_names = [name for name, _ in GRAVITY_TESTS_WXYZ]
    actual_names = [outcome.name for outcome in outcomes]
    if actual_names != expected_names:
        raise X2ValidationError(
            f"orientation outcomes must be in protocol order: {expected_names}, got {actual_names}"
        )
    outcome_measurements_finite = [
        bool(
            outcome.finite
            and all(
                math.isfinite(float(value))
                for value in (
                    outcome.final_displacement,
                    outcome.maximum_displacement,
                    outcome.final_contact_force,
                    outcome.maximum_active_joint_error,
                    outcome.maximum_newton_mimic_error,
                )
            )
        )
        for outcome in outcomes
    ]
    outcome_contacts_passed = [
        bool(
            measured_finite
            and (
                outcome.hand_object_contact
                if outcome.hand_object_contact is not None
                else outcome.final_contact_force > thresholds.contact_force
            )
        )
        for outcome, measured_finite in zip(outcomes, outcome_measurements_finite)
    ]
    outcome_mimic_passed = [
        bool(
            measured_finite
            and outcome.maximum_newton_mimic_error <= thresholds.mimic_error
        )
        for outcome, measured_finite in zip(outcomes, outcome_measurements_finite)
    ]
    simulation_success = all(
        outcome.passed and measured_finite and contact_passed and mimic_passed
        for outcome, measured_finite, contact_passed, mimic_passed in zip(
            outcomes,
            outcome_measurements_finite,
            outcome_contacts_passed,
            outcome_mimic_passed,
        )
    )
    closing_preflight, closing_raw_passed = _validate_collision_aware_closing_preflight(
        collision_aware_closing, thresholds
    )
    raw_json_penetration_passed = (
        candidate.maximum_penetration < thresholds.penetration
    )
    penetration_passed = bool(raw_json_penetration_passed and closing_raw_passed)
    self_collision_passed = bool(
        not candidate.self_collision_gate_required
        or candidate.self_collision_feasible is True
    )
    hand_object_passed = bool(
        not candidate.hand_object_gate_required
        or candidate.hand_object_feasible is True
    )
    overall_success = bool(
        simulation_success
        and penetration_passed
        and self_collision_passed
        and hand_object_passed
    )

    failure_reasons: list[str] = []
    if not raw_json_penetration_passed:
        failure_reasons.append("maximum_penetration_not_below_threshold")
    if not closing_preflight["raw_state_finite"]:
        failure_reasons.append("collision_aware_closing_raw_state_nonfinite")
    if not closing_preflight["raw_actuator_limits_passed"]:
        failure_reasons.append("collision_aware_closing_raw_actuator_limits")
    if not closing_preflight["raw_penetration_passed"]:
        failure_reasons.append(
            "collision_aware_closing_raw_penetration_not_below_threshold"
        )
    if not self_collision_passed:
        failure_reasons.append("self_collision_not_feasible")
    if not hand_object_passed:
        failure_reasons.append("dense_hand_object_penetration_not_feasible")
    for outcome, measured_finite, contact_passed, mimic_passed in zip(
        outcomes,
        outcome_measurements_finite,
        outcome_contacts_passed,
        outcome_mimic_passed,
    ):
        if not outcome.passed or not measured_finite or not contact_passed or not mimic_passed:
            if not measured_finite:
                failure_reasons.append(f"nonfinite_simulation:{outcome.name}")
            elif (
                not outcome.hand_object_contact
                if outcome.hand_object_contact is not None
                else outcome.final_contact_force <= thresholds.contact_force
            ):
                failure_reasons.append(f"lost_contact:{outcome.name}")
            if (
                measured_finite
                and outcome.maximum_newton_mimic_error > thresholds.mimic_error
            ):
                failure_reasons.append(f"newton_mimic_tracking:{outcome.name}")
            if criterion == "strict-hold" and measured_finite:
                if (
                    outcome.maximum_displacement >= thresholds.retention_distance
                    or outcome.final_displacement >= thresholds.retention_distance
                ):
                    failure_reasons.append(f"excess_displacement:{outcome.name}")
                if outcome.maximum_active_joint_error >= thresholds.joint_error:
                    failure_reasons.append(f"joint_tracking:{outcome.name}")

    orientation_records: list[dict[str, Any]] = []
    for outcome, measured_finite, contact_passed, mimic_passed in zip(
        outcomes,
        outcome_measurements_finite,
        outcome_contacts_passed,
        outcome_mimic_passed,
    ):
        orientation_record = outcome.as_dict()
        orientation_record["passed"] = bool(
            outcome.passed and measured_finite and contact_passed and mimic_passed
        )
        orientation_records.append(orientation_record)

    result = copy.deepcopy(candidate.record)
    result["success"] = overall_success
    result["simulation_success"] = simulation_success
    result["validation"] = {
        "status": "passed" if overall_success else "failed",
        "backend": VALIDATION_BACKEND,
        "protocol_revision": PROTOCOL_REVISION,
        "criterion": criterion,
        "source_raw": str(candidate.path),
        "source_sha256": candidate.source_sha256,
        "thresholds": thresholds.as_dict(),
        "preflight": {
            "finite": candidate.preflight_finite,
            "maximum_penetration_m": candidate.maximum_penetration,
            "raw_json_penetration_passed": raw_json_penetration_passed,
            "penetration_passed": penetration_passed,
            "collision_aware_closing": closing_preflight,
            "collision_aware_closing_raw_passed": closing_raw_passed,
            "self_collision_gate_required": candidate.self_collision_gate_required,
            "self_collision_feasible": candidate.self_collision_feasible,
            "self_collision_passed": self_collision_passed,
            "hand_object_gate_required": candidate.hand_object_gate_required,
            "hand_object_feasible": candidate.hand_object_feasible,
            "hand_object_passed": hand_object_passed,
        },
        "orientations": orientation_records,
        "passed_orientation_count": sum(
            outcome.passed and measured_finite and contact_passed and mimic_passed
            for outcome, measured_finite, contact_passed, mimic_passed in zip(
                outcomes,
                outcome_measurements_finite,
                outcome_contacts_passed,
                outcome_mimic_passed,
            )
        ),
        "required_orientation_count": len(GRAVITY_TESTS_WXYZ),
        "failure_reasons": failure_reasons,
        "runtime": dict(runtime or {}),
    }
    _require_json_safe(result, "validated record")
    return result


def validation_output_path(raw_path: Path | str, passed: bool) -> Path:
    """Return the sibling ``valid`` or ``failed`` path for one raw record."""

    source = Path(raw_path).expanduser().resolve()
    if source.parent.name != "raw":
        raise X2ValidationError(f"raw record must be directly inside a raw directory: {source}")
    return source.parent.parent / ("valid" if passed else "failed") / source.name


def write_validated_record(
    candidate: X2RawCandidate,
    record: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically publish an updated copy, preserving the immutable raw input."""

    if not candidate.source_bytes or not candidate.source_sha256:
        raise X2ValidationError("raw candidate has no immutable source snapshot")
    if hashlib.sha256(candidate.source_bytes).hexdigest() != candidate.source_sha256:
        raise X2ValidationError("raw candidate source snapshot hash is inconsistent")
    try:
        current_source_bytes = candidate.path.read_bytes()
    except Exception as exc:
        raise X2ValidationError(
            f"could not re-read raw candidate before publishing: {candidate.path}: {exc}"
        ) from exc
    if current_source_bytes != candidate.source_bytes:
        raise X2ValidationError(
            f"raw candidate changed since load; refusing validation output: {candidate.path}"
        )

    passed = record.get("validation", {}).get("status") == "passed"
    target = validation_output_path(candidate.path, passed)
    opposite = validation_output_path(candidate.path, not passed)
    if not overwrite and (target.exists() or opposite.exists()):
        existing = target if target.exists() else opposite
        raise X2ValidationError(f"validation output already exists: {existing}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(target)
        if opposite.exists():
            opposite.unlink()
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise
    return target


__all__ = [
    "CLOSING_LINE_SEARCH_ALPHAS",
    "CollisionAwareClosingSelection",
    "EXPECTED_ACTUATOR_NAMES",
    "EXPECTED_JOINT_NAMES",
    "GRAVITY_TESTS_WXYZ",
    "MAX_CLOSING_TARGET_PENETRATION_CAP",
    "ObjectCenteredReplay",
    "OrientationOutcome",
    "PASSIVE_MIMIC_DRIVERS",
    "PROTOCOL_REVISION",
    "SCHEMA_VERSION",
    "VALIDATION_BACKEND",
    "VALIDATION_CRITERIA",
    "ValidationThresholds",
    "X2RawCandidate",
    "X2ValidationError",
    "discover_raw_candidates",
    "compact_collision_aware_closing_batch_audit",
    "evaluate_orientation",
    "group_candidates_by_mesh",
    "load_raw_candidate",
    "make_object_centered_replay",
    "make_validated_record",
    "quaternion_matrix_wxyz",
    "quaternion_wxyz_to_xyzw",
    "select_collision_aware_closing",
    "summarize_newton_mimic_errors",
    "validation_output_path",
    "write_validated_record",
]
