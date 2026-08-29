#!/usr/bin/env python3
"""Build a non-destructive Gold/Silver quality index for the X2 snapshot.

The frozen study population is the 60,421 raw candidates that existed before
the last four general objects (078/081/084/087) were generated.  The frozen
Gold authority is the 6,418-record archive on the desktop.  No source JSON is
modified or copied: this command emits compact JSONL indexes and an auditable
manifest.

Important label rule:

* ``gold`` means the exact archive member has a six-orientation PhysX proof.
* ``silver`` means no PhysX route exists.
* ``physx_failed`` is a known negative and is never silently relabelled Silver.
* ``gold_provisional`` is a passed route absent from the frozen Gold archive.

The static gate reuses the generator's dense bidirectional penetration and
self-collision evidence, independently replays X2 FK and actuator limits, and
checks selected-contact proximity.  The inactive-finger check is deliberately
named a *surrogate*: static geometry cannot prove which link supplied PhysX
support.  That limitation remains explicit in every output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_generation.x2_isaac_validation import (  # noqa: E402
    EXPECTED_ACTUATOR_NAMES,
    EXPECTED_JOINT_NAMES,
    GRAVITY_TESTS_WXYZ,
    X2RawCandidate,
    X2ValidationError,
    load_raw_candidate,
)


DEFAULT_ATTEMPT_ROOT = (
    PROJECT_ROOT / "data" / "x2_valid_5000" / "attempts" / "attempt_0001"
)
DEFAULT_GOLD_ARCHIVE = (
    PROJECT_ROOT.parent / "x2_physx_valid_6418_20260727_081251.zip"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "x2_cleaned_gold_silver"
DEFAULT_EXCLUDED_GENERAL_OBJECTS = ("078", "081", "084", "087")
EXPECTED_SNAPSHOT_COUNT = 60_421
EXPECTED_GOLD_COUNT = 6_418
EXPECTED_PHYSX_PASSED = 6_419
EXPECTED_PHYSX_FAILED = 54_002
EXPECTED_ORIENTATIONS = tuple(name for name, _ in GRAVITY_TESTS_WXYZ)
RECORD_PATTERN = re.compile(
    r"^(?P<object_id>.+)_f(?P<finger_count>[1-5])_"
    r"(?P<side>front|back)_(?P<index>[0-9]{6})\.json$"
)
SHA256_PATTERN = re.compile(r"^(?P<sha>[0-9a-f]{64})  (?P<path>.+)$")
FINGER_NAMES = frozenset(("index", "middle", "ring", "little", "thumb"))
PROTOCOL_REVISION = "x2_gold_silver_cleaning_v1"


class CleaningError(RuntimeError):
    """Raised when a frozen-data or cleaning contract is violated."""


@dataclass(frozen=True)
class GoldMember:
    basename: str
    member_path: str
    sha256: str
    object_id: str
    side: str
    finger_count: int
    finger_names: tuple[str, ...]


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise CleaningError(f"{label}: non-finite JSON constant {value}")

    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
    except CleaningError:
        raise
    except Exception as exc:
        raise CleaningError(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CleaningError(f"{label}: JSON root must be an object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False))
            handle.write("\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def _relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_kind(path: Path) -> str:
    return "general" if path.name == "decomposed.obj" else "primitive"


def _attempt_object_catalog(
    attempt_root: Path,
) -> dict[str, tuple[Path, float]]:
    metadata_path = attempt_root / "attempt.json"
    metadata = _strict_json_bytes(metadata_path.read_bytes(), str(metadata_path))
    objects = metadata.get("objects")
    if not isinstance(objects, Mapping):
        raise CleaningError(f"{metadata_path}: object catalog is missing")
    catalog: dict[str, tuple[Path, float]] = {}
    for collection_name in ("primitive_meshes", "general_meshes"):
        records = objects.get(collection_name)
        if not isinstance(records, list):
            raise CleaningError(
                f"{metadata_path}: {collection_name} catalog is missing"
            )
        for record in records:
            if not isinstance(record, Mapping):
                raise CleaningError(
                    f"{metadata_path}: invalid {collection_name} record"
                )
            object_id = record.get("object_id")
            mesh_value = record.get("mesh_path")
            scale_value = record.get("object_scale")
            if (
                not isinstance(object_id, str)
                or not object_id
                or object_id in catalog
                or not isinstance(mesh_value, str)
                or not mesh_value
                or isinstance(scale_value, bool)
                or not isinstance(scale_value, (int, float))
                or not math.isfinite(float(scale_value))
                or float(scale_value) <= 0.0
            ):
                raise CleaningError(
                    f"{metadata_path}: invalid object catalog entry"
                )
            mesh_path = Path(mesh_value).expanduser().resolve()
            if not mesh_path.is_file():
                raise CleaningError(f"object mesh is missing: {mesh_path}")
            catalog[object_id] = (mesh_path, float(scale_value))
    return catalog


def _record_identity(path: Path) -> tuple[str, int, str, int]:
    match = RECORD_PATTERN.fullmatch(path.name)
    if match is None:
        raise CleaningError(f"candidate filename does not match X2 schema: {path}")
    return (
        match.group("object_id"),
        int(match.group("finger_count")),
        match.group("side"),
        int(match.group("index")),
    )


def discover_snapshot_paths(
    attempt_root: Path,
    *,
    excluded_general_objects: Sequence[str],
    expected_count: int | None,
) -> list[Path]:
    attempt_root = attempt_root.expanduser().resolve()
    if not attempt_root.is_dir():
        raise CleaningError(f"attempt root does not exist: {attempt_root}")
    excluded = set(excluded_general_objects)
    result: list[Path] = []
    for path in sorted(attempt_root.glob("**/raw/*.json")):
        object_id, _, _, _ = _record_identity(path)
        if "general" in path.parts and object_id in excluded:
            continue
        result.append(path.resolve())
    if expected_count is not None and len(result) != expected_count:
        raise CleaningError(
            f"frozen candidate population must contain {expected_count} records; "
            f"found {len(result)}"
        )
    if len(result) != len(set(result)):
        raise CleaningError("raw candidate inventory contains duplicate paths")
    return result


def _validate_gold_payload(
    payload: Mapping[str, Any],
    *,
    basename: str,
) -> tuple[str, int, tuple[str, ...]]:
    match = RECORD_PATTERN.fullmatch(basename)
    if match is None:
        raise CleaningError(f"Gold filename is invalid: {basename}")
    side = match.group("side")
    finger_count = int(match.group("finger_count"))
    if payload.get("active_side") != side:
        raise CleaningError(f"{basename}: Gold active_side disagrees with filename")
    participation = payload.get("finger_participation")
    if not isinstance(participation, Mapping):
        raise CleaningError(f"{basename}: Gold finger participation is missing")
    names = participation.get("finger_names")
    if (
        participation.get("target_count") != finger_count
        or participation.get("actual_count") != finger_count
        or not isinstance(names, list)
        or len(names) != finger_count
        or len(set(names)) != finger_count
        or not set(names) <= FINGER_NAMES
    ):
        raise CleaningError(f"{basename}: Gold finger mask proof is invalid")
    validation = payload.get("validation")
    if not isinstance(validation, Mapping):
        raise CleaningError(f"{basename}: Gold validation proof is missing")
    orientations = validation.get("orientations")
    if (
        payload.get("success") is not True
        or payload.get("simulation_success") is not True
        or validation.get("status") != "passed"
        or validation.get("backend") != "isaac_sim_physx"
        or validation.get("required_orientation_count") != 6
        or validation.get("passed_orientation_count") != 6
        or not isinstance(orientations, list)
        or tuple(
            value.get("name") if isinstance(value, Mapping) else None
            for value in orientations
        )
        != EXPECTED_ORIENTATIONS
        or not all(
            isinstance(value, Mapping)
            and value.get("passed") is True
            and value.get("finite") is True
            and value.get("hand_object_contact") is True
            for value in orientations
        )
    ):
        raise CleaningError(f"{basename}: Gold six-orientation proof is invalid")
    preflight = validation.get("preflight")
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("collision_aware_closing_raw_passed") is not True
        or preflight.get("self_collision_passed") is not True
        or preflight.get("hand_object_passed") is not True
    ):
        raise CleaningError(f"{basename}: Gold static preflight proof is invalid")
    return side, finger_count, tuple(sorted(str(value) for value in names))


def load_gold_archive(
    archive: Path,
    *,
    expected_count: int | None,
) -> tuple[dict[str, GoldMember], dict[str, Any]]:
    archive = archive.expanduser().resolve()
    if not archive.is_file():
        raise CleaningError(f"Gold archive does not exist: {archive}")
    members: dict[str, GoldMember] = {}
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        manifest_name = next(
            (value for value in names if value.endswith("/manifest.json")), None
        )
        checksums_name = next(
            (value for value in names if value.endswith("/SHA256SUMS")), None
        )
        if manifest_name is None or checksums_name is None:
            raise CleaningError("Gold archive lacks manifest.json or SHA256SUMS")
        manifest = _strict_json_bytes(
            bundle.read(manifest_name), f"{archive}!{manifest_name}"
        )
        checksum_text = bundle.read(checksums_name).decode("utf-8")
        checksum_by_path: dict[str, str] = {}
        for line in checksum_text.splitlines():
            match = SHA256_PATTERN.fullmatch(line)
            if match is None:
                raise CleaningError(f"Gold SHA256SUMS line is invalid: {line!r}")
            checksum_by_path[match.group("path")] = match.group("sha")
        prefix = manifest_name[: -len("manifest.json")]
        sample_members = sorted(
            value
            for value in names
            if "/samples/" in value and value.endswith(".json")
        )
        if expected_count is not None and len(sample_members) != expected_count:
            raise CleaningError(
                f"Gold archive must contain {expected_count} samples; "
                f"found {len(sample_members)}"
            )
        if manifest.get("sample_count") != len(sample_members):
            raise CleaningError("Gold manifest sample_count disagrees with archive")
        for index, member_path in enumerate(sample_members, 1):
            relative_path = member_path[len(prefix) :]
            expected_sha = checksum_by_path.get(relative_path)
            if expected_sha is None:
                raise CleaningError(f"Gold checksum is missing: {relative_path}")
            raw = bundle.read(member_path)
            actual_sha = _sha256_bytes(raw)
            if actual_sha != expected_sha:
                raise CleaningError(f"Gold checksum mismatch: {relative_path}")
            payload = _strict_json_bytes(raw, f"{archive}!{member_path}")
            basename = Path(member_path).name
            side, finger_count, finger_names = _validate_gold_payload(
                payload, basename=basename
            )
            object_id, _, _, _ = _record_identity(Path(basename))
            if basename in members:
                raise CleaningError(f"Gold basename is duplicated: {basename}")
            members[basename] = GoldMember(
                basename=basename,
                member_path=relative_path,
                sha256=actual_sha,
                object_id=object_id,
                side=side,
                finger_count=finger_count,
                finger_names=finger_names,
            )
            if index % 1000 == 0:
                print(f"[gold-audit] {index}/{len(sample_members)}", flush=True)
    return members, {
        "archive": str(archive),
        "archive_sha256": _sha256_file(archive),
        "manifest": manifest,
        "member_count": len(members),
        "member_sha256_count": len(members),
    }


def _canonical_quaternion(values: Sequence[float]) -> list[float]:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise CleaningError("quaternion is not finite 4D")
    for value in quaternion:
        if abs(float(value)) > 1.0e-15:
            if value < 0.0:
                quaternion = -quaternion
            break
    return [float(value) for value in quaternion]


def _semantic_hash(candidate: X2RawCandidate) -> str:
    record = candidate.record
    payload = {
        "mesh": str(candidate.mesh_path),
        "scale": candidate.object_scale,
        "side": candidate.active_side,
        "finger_names": sorted(
            record["finger_participation"]["finger_names"]
        ),
        "selected_contact_ids": sorted(record["selected_contact_ids"]),
        "translation": [float(value) for value in candidate.hand_translation],
        "quaternion": _canonical_quaternion(candidate.hand_quaternion_wxyz),
        "actuator": [
            float(candidate.actuator_by_name[name])
            for name in EXPECTED_ACTUATOR_NAMES
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(raw)


def _quantize(values: Sequence[float], width: float) -> tuple[int, ...]:
    return tuple(int(round(float(value) / width)) for value in values)


def _near_key(
    candidate: X2RawCandidate,
    *,
    translation_width: float,
    quaternion_width: float,
    actuator_width: float,
) -> tuple[Any, ...]:
    record = candidate.record
    return (
        str(candidate.mesh_path),
        round(candidate.object_scale, 12),
        candidate.active_side,
        tuple(sorted(record["finger_participation"]["finger_names"])),
        tuple(sorted(record["selected_contact_ids"])),
        _quantize(candidate.hand_translation, translation_width),
        _quantize(
            _canonical_quaternion(candidate.hand_quaternion_wxyz),
            quaternion_width,
        ),
        _quantize(
            [
                candidate.actuator_by_name[name]
                for name in EXPECTED_ACTUATOR_NAMES
            ],
            actuator_width,
        ),
    )


def _load_hand_model():
    import torch

    from grasp_generation.utils.x2_config import load_x2_mesh_config
    from grasp_generation.utils.x2_hand_model import X2HandModel
    from grasp_generation.utils.x2_mesh_contacts import (
        load_generic_contact_candidates,
    )

    config = load_x2_mesh_config()
    contacts = load_generic_contact_candidates(
        config.configured_path("contact_candidates.path", must_exist=True)
    )
    hand = X2HandModel(
        config,
        contacts,
        device="cpu",
        dtype=torch.float64,
        collision_samples_per_link=8,
        audit_collision_samples_per_link=8,
        self_collision_samples_per_link=8,
    )
    return torch, hand, contacts


def _geometry_batch(
    candidates: Sequence[X2RawCandidate],
    *,
    torch: Any,
    hand: Any,
    contact_index: Mapping[str, int],
    active_contact_threshold: float,
    fk_position_tolerance: float,
    fk_normal_min_dot: float,
    inactive_flex_max: float,
) -> list[dict[str, Any]]:
    import trimesh

    if not candidates:
        return []
    mesh_keys = {(value.mesh_path, value.object_scale) for value in candidates}
    if len(mesh_keys) != 1:
        raise CleaningError("geometry batch spans multiple object meshes")
    rotations = np.stack(
        [
            np.asarray(value.record["hand_pose"]["rotation_matrix"], dtype=np.float64)
            for value in candidates
        ]
    )
    rotation6d = torch.as_tensor(
        rotations.transpose(0, 2, 1)[:, :2].reshape(len(candidates), 6),
        dtype=torch.float64,
    )
    actuator = torch.as_tensor(
        [
            [value.actuator_by_name[name] for name in EXPECTED_ACTUATOR_NAMES]
            for value in candidates
        ],
        dtype=torch.float64,
    )
    pose = torch.cat(
        (
            torch.as_tensor(
                np.stack([value.hand_translation for value in candidates]),
                dtype=torch.float64,
            ),
            rotation6d,
            actuator,
        ),
        dim=1,
    )
    try:
        ids = torch.as_tensor(
            [
                [contact_index[value] for value in candidate.record["selected_contact_ids"]]
                for candidate in candidates
            ],
            dtype=torch.long,
        )
    except KeyError as exc:
        raise CleaningError(
            f"selected contact is absent from authored pool: {exc.args[0]}"
        ) from exc
    hand.set_parameters(pose, ids)
    if hand.contact_points is None or hand.contact_normals is None:
        raise CleaningError("X2 FK did not materialize selected contacts")
    actual_points = hand.contact_points.detach().cpu().numpy()
    actual_normals = hand.contact_normals.detach().cpu().numpy()
    expected_points = np.asarray(
        [
            [contact["world_position"] for contact in value.record["selected_contacts"]]
            for value in candidates
        ],
        dtype=np.float64,
    )
    expected_normals = np.asarray(
        [
            [
                contact["world_surface_normal"]
                for contact in value.record["selected_contacts"]
            ]
            for value in candidates
        ],
        dtype=np.float64,
    )
    position_error = np.linalg.norm(actual_points - expected_points, axis=-1).max(
        axis=1
    )
    normal_dot = (
        actual_normals * expected_normals
    ).sum(axis=-1).min(axis=1)
    lower = hand.joints_lower.detach().cpu().numpy()
    upper = hand.joints_upper.detach().cpu().numpy()
    within_limits = (
        (actuator.detach().cpu().numpy() >= lower[None, :] - 1.0e-8)
        & (actuator.detach().cpu().numpy() <= upper[None, :] + 1.0e-8)
    ).all(axis=1)

    mesh_path, scale = next(iter(mesh_keys))
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not mesh.is_watertight:
        raise CleaningError(f"static proximity requires a watertight mesh: {mesh_path}")
    mesh = mesh.copy()
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
    query = trimesh.proximity.ProximityQuery(mesh)
    signed = np.asarray(
        query.signed_distance(actual_points.reshape(-1, 3)),
        dtype=np.float64,
    ).reshape(len(candidates), -1)
    if not np.isfinite(signed).all():
        raise CleaningError(f"non-finite contact proximity for {mesh_path}")

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        selected = candidate.record["selected_contacts"]
        finger_mask = np.asarray(
            [contact["finger_name"] != "palm" for contact in selected],
            dtype=bool,
        )
        if not finger_mask.any():
            raise CleaningError(f"{candidate.path}: no active-finger selected contact")
        active_max = float(np.abs(signed[index, finger_mask]).max())
        energy = candidate.record.get("energy")
        terms = energy.get("terms") if isinstance(energy, Mapping) else None
        side_energy = (
            float(terms.get("E_side")) if isinstance(terms, Mapping) else math.inf
        )
        joint_energy = (
            float(terms.get("E_joints")) if isinstance(terms, Mapping) else math.inf
        )
        inactive_flex = (
            float(terms.get("E_unselected_opposite_flex"))
            if isinstance(terms, Mapping)
            else math.inf
        )
        side_supported = all(
            candidate.active_side in contact["supported_sides"]
            for contact in selected
        )
        failures: list[str] = []
        if not bool(within_limits[index]) or not math.isfinite(joint_energy) or joint_energy > 1.0e-10:
            failures.append("joint_or_actuator_limits")
        if candidate.self_collision_feasible is not True:
            failures.append("self_collision")
        if candidate.hand_object_feasible is not True:
            failures.append("dense_hand_object_penetration")
        if (
            not side_supported
            or not math.isfinite(side_energy)
            or side_energy > 1.0e-12
        ):
            failures.append("side_specific_surface")
        if active_max > active_contact_threshold:
            failures.append("active_finger_proximity")
        if float(position_error[index]) > fk_position_tolerance:
            failures.append("fk_position")
        if float(normal_dot[index]) < fk_normal_min_dot:
            failures.append("fk_normal")
        inactive_safe = bool(
            math.isfinite(inactive_flex)
            and inactive_flex <= inactive_flex_max
        )
        if not inactive_safe:
            failures.append("inactive_finger_safety_surrogate")
        rows.append(
            {
                "static_valid": not failures,
                "static_failure_reasons": failures,
                "metrics": {
                    "maximum_selected_contact_surface_distance_m": (
                        active_max
                    ),
                    "minimum_inactive_finger_surface_distance_m": None,
                    "maximum_fk_position_error_m": float(position_error[index]),
                    "minimum_fk_normal_dot": float(normal_dot[index]),
                    "inactive_finger_opposite_flex_energy": inactive_flex,
                    "side_energy": side_energy,
                    "joint_limit_energy": joint_energy,
                    "maximum_dense_hand_object_penetration_m": float(
                        candidate.maximum_penetration
                    ),
                    "maximum_self_collision_penetration_m": float(
                        candidate.record["self_collision"]["maximum_penetration"]
                    ),
                },
                "checks": {
                    "actuator_and_joint_limits": bool(within_limits[index])
                    and joint_energy <= 1.0e-10,
                    "self_collision": candidate.self_collision_feasible is True,
                    "dense_hand_object_penetration": (
                        candidate.hand_object_feasible is True
                    ),
                    "active_finger_proximity": (
                        active_max <= active_contact_threshold
                    ),
                    "inactive_finger_safety_surrogate": inactive_safe,
                    "inactive_finger_contact_exclusion_proven": False,
                    "side_specific_surface": (
                        side_supported and side_energy <= 1.0e-12
                    ),
                    "wrist_root_and_fk": (
                        bool(position_error[index] <= fk_position_tolerance)
                        and bool(normal_dot[index] >= fk_normal_min_dot)
                    ),
                },
            }
        )
    return rows


def _route_status(
    candidate: X2RawCandidate,
    *,
    raw_sha256: str,
) -> tuple[str, Path | None, list[str]]:
    raw_path = candidate.path
    valid_path = raw_path.parent.parent / "valid" / raw_path.name
    failed_path = raw_path.parent.parent / "failed" / raw_path.name
    existing = [value for value in (valid_path, failed_path) if value.is_file()]
    if not existing:
        return "unverified", None, []
    if len(existing) != 1:
        return "conflict", None, ["both_valid_and_failed_routes_exist"]
    route_path = existing[0]
    expected_status = "passed" if route_path.parent.name == "valid" else "failed"
    try:
        payload = _strict_json_bytes(route_path.read_bytes(), str(route_path))
        validation = payload.get("validation")
        if not isinstance(validation, Mapping):
            raise CleaningError("validation object is missing")
        source_raw = validation.get("source_raw")
        source_sha = validation.get("source_sha256")
        if (
            validation.get("status") != expected_status
            or not isinstance(source_raw, str)
            or Path(source_raw).expanduser().resolve() != raw_path.resolve()
            or source_sha != raw_sha256
        ):
            raise CleaningError("validation route/source proof is inconsistent")
        orientations = validation.get("orientations")
        if not isinstance(orientations, list):
            raise CleaningError("validation orientations must be a list")
        passed_count = sum(
            int(
                isinstance(value, Mapping)
                and value.get("passed") is True
                and value.get("finite") is True
            )
            for value in orientations
        )
        if expected_status == "passed":
            if (
                payload.get("success") is not True
                or payload.get("simulation_success") is not True
                or len(orientations) != 6
                or passed_count != 6
            ):
                raise CleaningError("passed route lacks a 6/6 physical proof")
        elif (
            payload.get("success") is not False
            or not isinstance(validation.get("failure_reasons"), list)
            or not validation.get("failure_reasons")
        ):
            # A static generator gate may reject a record even when all six
            # dynamic orientations passed.  Preserve that legitimate negative
            # instead of equating simulation_success with the final label.
            raise CleaningError("failed route contradicts final failure fields")
    except Exception as exc:
        return "conflict", route_path, [f"invalid_physx_route:{type(exc).__name__}:{exc}"]
    return expected_status, route_path, []


def _base_row(candidate: X2RawCandidate) -> dict[str, Any]:
    object_id, finger_count, side, sample_index = _record_identity(candidate.path)
    participation = candidate.record["finger_participation"]
    if (
        finger_count != participation["target_count"]
        or side != candidate.active_side
    ):
        raise CleaningError(f"{candidate.path}: filename task disagrees with JSON")
    return {
        "record_id": candidate.path.stem,
        "source_raw": _relative(candidate.path),
        "object_id": object_id,
        "object_kind": _object_kind(candidate.mesh_path),
        "mesh_path": _relative(candidate.mesh_path),
        "object_scale": candidate.object_scale,
        "side": candidate.active_side,
        "finger_count": finger_count,
        "finger_names": sorted(participation["finger_names"]),
        "selected_contact_ids": list(candidate.record["selected_contact_ids"]),
        "sample_index": sample_index,
        "pipeline_revision": candidate.record.get("pipeline_revision"),
        "contract_pass": True,
    }


def clean_dataset(args: argparse.Namespace) -> dict[str, Any]:
    attempt_root = args.attempt_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise CleaningError(
                f"output root already exists (pass --overwrite): {output_root}"
            )
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        gold, gold_audit = load_gold_archive(
            args.gold_archive, expected_count=args.expected_gold_count
        )
        paths = discover_snapshot_paths(
            attempt_root,
            excluded_general_objects=args.exclude_general_object,
            expected_count=args.expected_candidate_count,
        )
        torch, hand, contacts = _load_hand_model()
        contact_index = {
            value.point_id: index for index, value in enumerate(contacts)
        }
        rows: list[dict[str, Any]] = []
        semantic_first: dict[str, str] = {}
        near_members: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        basename_inventory: set[str] = set()
        contract_errors = 0

        catalog = _attempt_object_catalog(attempt_root)
        grouped: dict[tuple[Path, float, int], list[Path]] = defaultdict(list)
        # The filename and frozen attempt catalog provide the complete grouping
        # key.  Do not parse 60k optimization traces just to discover mesh
        # metadata: Python's allocator otherwise retains tens of GiB.
        for path in paths:
            object_id, finger_count, _, _ = _record_identity(path)
            try:
                mesh_path, scale = catalog[object_id]
            except KeyError as exc:
                raise CleaningError(
                    f"{path}: object is absent from attempt catalog"
                ) from exc
            contact_count = max(4, finger_count)
            grouped[(mesh_path, scale, contact_count)].append(path)

        processed = 0
        for (mesh_path, scale, contact_count), group_paths in sorted(
            grouped.items(), key=lambda value: str(value[0])
        ):
            for offset in range(0, len(group_paths), args.batch_size):
                batch_paths = group_paths[offset : offset + args.batch_size]
                candidates: list[X2RawCandidate] = []
                candidate_rows: list[dict[str, Any]] = []
                for path in batch_paths:
                    try:
                        candidate = load_raw_candidate(path)
                        if (
                            candidate.mesh_path != mesh_path
                            or candidate.object_scale != scale
                        ):
                            raise CleaningError("discovery mesh/scale changed")
                        if (
                            len(candidate.record["selected_contact_ids"])
                            != contact_count
                        ):
                            raise CleaningError(
                                "selected contact count disagrees with "
                                "finger-count partition"
                            )
                        candidates.append(candidate)
                        candidate_rows.append(_base_row(candidate))
                    except Exception as exc:
                        contract_errors += 1
                        object_id, finger_count, side, sample_index = _record_identity(path)
                        rows.append(
                            {
                                "record_id": path.stem,
                                "source_raw": _relative(path),
                                "object_id": object_id,
                                "side": side,
                                "finger_count": finger_count,
                                "sample_index": sample_index,
                                "contract_pass": False,
                                "static_valid": False,
                                "static_failure_reasons": ["data_contract"],
                                "quality_tier": "quarantine",
                                "quality_errors": [
                                    f"{type(exc).__name__}:{exc}"
                                ],
                            }
                        )
                geometry = _geometry_batch(
                    candidates,
                    torch=torch,
                    hand=hand,
                    contact_index=contact_index,
                    active_contact_threshold=args.active_contact_threshold,
                    fk_position_tolerance=args.fk_position_tolerance,
                    fk_normal_min_dot=args.fk_normal_min_dot,
                    inactive_flex_max=args.inactive_flex_max,
                )
                for candidate, row, static in zip(
                    candidates, candidate_rows, geometry
                ):
                    raw_sha = candidate.source_sha256
                    status, route_path, route_errors = _route_status(
                        candidate, raw_sha256=raw_sha
                    )
                    gold_member = gold.get(candidate.path.name)
                    if gold_member is not None:
                        if (
                            status != "passed"
                            or route_path is None
                            or _sha256_file(route_path) != gold_member.sha256
                        ):
                            route_errors.append(
                                "frozen_gold_member_does_not_match_current_passed_route"
                            )
                        quality_tier = "gold"
                    elif status == "passed":
                        quality_tier = "gold_provisional"
                    elif status == "failed":
                        quality_tier = "physx_failed"
                    elif status == "unverified":
                        quality_tier = "silver"
                    else:
                        quality_tier = "quarantine"
                    semantic = _semantic_hash(candidate)
                    exact_duplicate_of = semantic_first.setdefault(
                        semantic, candidate.path.stem
                    )
                    if exact_duplicate_of == candidate.path.stem:
                        exact_duplicate_of = None
                    near_key = _near_key(
                        candidate,
                        translation_width=args.near_translation_width,
                        quaternion_width=args.near_quaternion_width,
                        actuator_width=args.near_actuator_width,
                    )
                    near_cluster_id = _sha256_bytes(
                        repr(near_key).encode("utf-8")
                    )[:20]
                    row.update(static)
                    row.update(
                        {
                            "source_raw_sha256": raw_sha,
                            "physx_status": status,
                            "physx_route": (
                                _relative(route_path)
                                if route_path is not None
                                else None
                            ),
                            "quality_tier": (
                                "quarantine" if route_errors else quality_tier
                            ),
                            "quality_errors": route_errors,
                            "gold_archive_member": (
                                gold_member.member_path
                                if gold_member is not None
                                else None
                            ),
                            "gold_archive_member_sha256": (
                                gold_member.sha256
                                if gold_member is not None
                                else None
                            ),
                            "semantic_pose_sha256": semantic,
                            "exact_duplicate_of": exact_duplicate_of,
                            "near_duplicate_cluster_id": near_cluster_id,
                            "near_duplicate_representative": False,
                            "pretraining_candidate": True,
                            "positive_supervision": (
                                quality_tier == "gold" and not route_errors
                            ),
                            "hard_negative": (
                                quality_tier == "physx_failed"
                                and not route_errors
                            ),
                            "paper_main_result_eligible": False,
                        }
                    )
                    row_index = len(rows)
                    rows.append(row)
                    near_members[near_key].append(row_index)
                    basename_inventory.add(candidate.path.name)
                processed += len(batch_paths)
                if processed % 1000 < len(batch_paths):
                    print(
                        f"[candidate-clean] {processed}/{len(paths)}",
                        flush=True,
                    )

        missing_gold = sorted(set(gold) - basename_inventory)
        if missing_gold:
            raise CleaningError(
                f"{len(missing_gold)} Gold members are outside the 60,421 snapshot; "
                f"first={missing_gold[0]}"
            )
        if contract_errors:
            print(
                f"[candidate-clean] contract quarantine count={contract_errors}",
                flush=True,
            )

        cluster_rows: list[dict[str, Any]] = []
        for key, indices in near_members.items():
            eligible = [
                index
                for index in indices
                if rows[index]["contract_pass"]
                and rows[index]["static_valid"]
                and rows[index]["quality_tier"] != "quarantine"
            ]
            representative: int | None = None
            if eligible:
                rank = {
                    "gold": 0,
                    "gold_provisional": 1,
                    "silver": 2,
                    "physx_failed": 3,
                }
                representative = min(
                    eligible,
                    key=lambda index: (
                        rank.get(rows[index]["quality_tier"], 9),
                        rows[index]["metrics"][
                            "maximum_dense_hand_object_penetration_m"
                        ],
                        rows[index]["record_id"],
                    ),
                )
                rows[representative]["near_duplicate_representative"] = True
            if len(indices) > 1:
                tier_counts = Counter(
                    rows[index]["quality_tier"] for index in indices
                )
                cluster_rows.append(
                    {
                        "near_duplicate_cluster_id": rows[indices[0]][
                            "near_duplicate_cluster_id"
                        ],
                        "member_count": len(indices),
                        "record_ids": [rows[index]["record_id"] for index in indices],
                        "tier_counts": dict(sorted(tier_counts.items())),
                        "physx_label_conflict": (
                            tier_counts.get("gold", 0)
                            + tier_counts.get("gold_provisional", 0)
                            > 0
                            and tier_counts.get("physx_failed", 0) > 0
                        ),
                        "representative_record_id": (
                            rows[representative]["record_id"]
                            if representative is not None
                            else None
                        ),
                    }
                )

        tier_counts = Counter(row["quality_tier"] for row in rows)
        physx_counts = Counter(
            row.get("physx_status", "contract_error") for row in rows
        )
        static_failures = Counter(
            reason
            for row in rows
            for reason in row.get("static_failure_reasons", [])
        )
        static_valid_count = sum(
            int(row.get("static_valid") is True) for row in rows
        )
        static_unique_count = sum(
            int(
                row.get("static_valid") is True
                and row.get("near_duplicate_representative") is True
            )
            for row in rows
        )
        exact_duplicate_count = sum(
            int(row.get("exact_duplicate_of") is not None) for row in rows
        )

        if args.expected_candidate_count is not None and len(rows) != args.expected_candidate_count:
            raise CleaningError("cleaned record count changed")
        if args.expected_gold_count is not None and tier_counts["gold"] != args.expected_gold_count:
            raise CleaningError(
                f"Gold reconciliation expected {args.expected_gold_count}, "
                f"got {tier_counts['gold']}"
            )
        if args.enforce_snapshot_physx_counts and (
            physx_counts["passed"] != EXPECTED_PHYSX_PASSED
            or physx_counts["failed"] != EXPECTED_PHYSX_FAILED
            or physx_counts["unverified"] != 0
        ):
            raise CleaningError(
                "60,421 snapshot PhysX counts changed: "
                f"{dict(physx_counts)}"
            )

        all_path = staging / "all_records.jsonl"
        output_counts = {
            "all_records": _write_jsonl(all_path, rows),
            "candidate_corpus": _write_jsonl(
                staging / "candidate_corpus.jsonl",
                (
                    row
                    for row in rows
                    if row.get("contract_pass") is True
                    and row.get("pretraining_candidate") is True
                ),
            ),
            "static_valid_unique": _write_jsonl(
                staging / "static_valid_unique.jsonl",
                (
                    row
                    for row in rows
                    if row.get("static_valid") is True
                    and row.get("near_duplicate_representative") is True
                ),
            ),
            "gold": _write_jsonl(
                staging / "gold.jsonl",
                (row for row in rows if row["quality_tier"] == "gold"),
            ),
            "gold_provisional": _write_jsonl(
                staging / "gold_provisional.jsonl",
                (
                    row
                    for row in rows
                    if row["quality_tier"] == "gold_provisional"
                ),
            ),
            "physx_failed": _write_jsonl(
                staging / "physx_failed.jsonl",
                (
                    row
                    for row in rows
                    if row["quality_tier"] == "physx_failed"
                ),
            ),
            "silver_unverified": _write_jsonl(
                staging / "silver_unverified.jsonl",
                (row for row in rows if row["quality_tier"] == "silver"),
            ),
            "quarantine": _write_jsonl(
                staging / "quarantine.jsonl",
                (row for row in rows if row["quality_tier"] == "quarantine"),
            ),
            "near_duplicate_clusters": _write_jsonl(
                staging / "near_duplicate_clusters.jsonl", cluster_rows
            ),
        }

        coverage: Counter[tuple[str, str, int, str]] = Counter()
        for row in rows:
            coverage[
                (
                    str(row.get("object_id")),
                    str(row.get("side")),
                    int(row.get("finger_count", 0)),
                    str(row.get("quality_tier")),
                )
            ] += 1
        with (staging / "coverage.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ("object_id", "side", "finger_count", "quality_tier", "count")
            )
            for key, count in sorted(coverage.items()):
                writer.writerow((*key, count))

        dual_root = PROJECT_ROOT / "data" / "x2_dual_object"
        dual_valid = (
            len(list((dual_root / "physx_validation" / "valid").glob("**/*.json")))
            if dual_root.is_dir()
            else 0
        )
        dual_failed = (
            len(list((dual_root / "physx_validation" / "failed").glob("**/*.json")))
            if dual_root.is_dir()
            else 0
        )
        matched_status = {
            "dataset_root": _relative(dual_root),
            "role": "matched_h1_candidate",
            "status": (
                "complete"
                if dual_valid + dual_failed == 2020
                else "in_progress"
            ),
            "candidate_count": 2020,
            "physx_valid_count": dual_valid,
            "physx_failed_count": dual_failed,
            "paper_main_result_eligible": (
                dual_valid + dual_failed == 2020
            ),
            "note": (
                "Joint front/back two-object PhysX evidence is isolated from "
                "single-object Gold and must not be merged until complete."
            ),
        }
        _atomic_json(staging / "matched_h1_status.json", matched_status)

        now = datetime.now().astimezone().isoformat(timespec="seconds")
        manifest = {
            "schema_version": 1,
            "protocol_revision": PROTOCOL_REVISION,
            "created_at": now,
            "passed": tier_counts["quarantine"] == 0,
            "source": {
                "attempt_root": _relative(attempt_root),
                "attempt_complete_sha256": _sha256_file(
                    attempt_root / "complete.json"
                ),
                "snapshot_definition": {
                    "raw_candidates": len(paths),
                    "excluded_general_objects": list(
                        args.exclude_general_object
                    ),
                },
                "gold_archive": gold_audit,
            },
            "taxonomy": {
                "gold": (
                    "Frozen archive records with 6/6 finite PhysX orientations."
                ),
                "silver": "Candidates without any PhysX route.",
                "physx_failed": (
                    "Known failed physical trials; valid only as negatives or "
                    "candidate-only pretraining examples."
                ),
                "gold_provisional": (
                    "Passed PhysX route absent from the frozen 6,418 archive."
                ),
                "matched_h1": (
                    "Separately collected joint front/back PhysX evidence."
                ),
            },
            "counts": {
                "candidate_population": len(rows),
                "quality_tiers": dict(sorted(tier_counts.items())),
                "physx_status": dict(sorted(physx_counts.items())),
                "static_valid": static_valid_count,
                "static_invalid": len(rows) - static_valid_count,
                "static_valid_near_unique": static_unique_count,
                "exact_semantic_duplicates": exact_duplicate_count,
                "near_duplicate_cluster_count": len(cluster_rows),
                "static_failure_reasons": dict(sorted(static_failures.items())),
                "outputs": output_counts,
            },
            "static_gate": {
                "active_contact_threshold_m": args.active_contact_threshold,
                "fk_position_tolerance_m": args.fk_position_tolerance,
                "fk_normal_min_dot": args.fk_normal_min_dot,
                "inactive_finger_opposite_flex_energy_max": (
                    args.inactive_flex_max
                ),
                "dense_penetration_threshold_m": 0.001,
                "self_collision_threshold_m": 0.0005,
                "inactive_finger_contact_exclusion_is_proven": False,
                "inactive_finger_limitation": (
                    "The recorded static surrogate penalizes inactive fingers "
                    "bending toward the opposite palm side. Per-link PhysX "
                    "contact ownership is required to prove they do not support "
                    "a grasp."
                ),
            },
            "near_duplicate_definition": {
                "same_object_side_mask_and_selected_contacts": True,
                "translation_bin_m": args.near_translation_width,
                "quaternion_component_bin": args.near_quaternion_width,
                "actuator_bin_rad": args.near_actuator_width,
                "representative_preference": (
                    "gold, gold_provisional, silver, physx_failed; then lower "
                    "dense penetration and deterministic record_id"
                ),
            },
            "label_policy": {
                "candidate_corpus_is_positive_supervision": False,
                "gold_is_positive_supervision": True,
                "physx_failed_is_positive_supervision": False,
                "silver_is_positive_supervision": False,
                "matched_h1_is_merged_into_gold": False,
            },
            "matched_h1": matched_status,
        }
        _atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


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


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=Path, default=DEFAULT_ATTEMPT_ROOT)
    parser.add_argument("--gold-archive", type=Path, default=DEFAULT_GOLD_ARCHIVE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--exclude-general-object",
        action="append",
        default=list(DEFAULT_EXCLUDED_GENERAL_OBJECTS),
    )
    parser.add_argument(
        "--expected-candidate-count",
        type=_positive_int,
        default=EXPECTED_SNAPSHOT_COUNT,
    )
    parser.add_argument(
        "--expected-gold-count",
        type=_positive_int,
        default=EXPECTED_GOLD_COUNT,
    )
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument(
        "--active-contact-threshold",
        type=_positive_float,
        default=0.003,
    )
    parser.add_argument(
        "--fk-position-tolerance", type=_positive_float, default=5.0e-4
    )
    parser.add_argument(
        "--fk-normal-min-dot", type=float, default=0.999
    )
    parser.add_argument(
        "--inactive-flex-max", type=_nonnegative_float, default=1.0e-4
    )
    parser.add_argument(
        "--near-translation-width", type=_positive_float, default=0.002
    )
    parser.add_argument(
        "--near-quaternion-width", type=_positive_float, default=0.01
    )
    parser.add_argument(
        "--near-actuator-width", type=_positive_float, default=0.02
    )
    parser.add_argument(
        "--enforce-snapshot-physx-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not -1.0 <= args.fk_normal_min_dot <= 1.0:
        raise SystemExit("--fk-normal-min-dot must be in [-1, 1]")
    try:
        manifest = clean_dataset(args)
    except (CleaningError, X2ValidationError) as exc:
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
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
