#!/usr/bin/env python3
"""Select tabletop-clear X2 poses from an existing PhysX-valid grasp pool.

The source PhysX result is preserved as provenance.  Adding a support plane is
only a static geometry operation, so the derived records explicitly state that
tabletop PhysX validation has not been run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_generation.utils.x2_config import load_x2_mesh_config  # noqa: E402
from grasp_generation.utils.x2_hand_model import X2HandModel  # noqa: E402
from grasp_generation.utils.x2_mesh_contacts import (  # noqa: E402
    load_generic_contact_candidates,
)


SIDES = ("front", "back")
FINGER_COUNTS = (1, 2, 3, 4, 5)


class TabletopSelectionError(RuntimeError):
    """Raised when the source pool cannot satisfy the tabletop contract."""


@dataclass(frozen=True)
class Candidate:
    path: Path
    payload: dict[str, Any]
    clearance_m: float
    maximum_displacement_m: float
    minimum_final_contact_force_n: float
    maximum_active_joint_error_rad: float
    maximum_hand_object_penetration_m: float

    @property
    def score(self) -> tuple[float, float, float, float, str]:
        """Prefer low motion, strong contact, low penetration, then clearance."""

        return (
            self.maximum_displacement_m,
            -self.minimum_final_contact_force_n,
            self.maximum_hand_object_penetration_m,
            -self.clearance_m,
            self.path.name,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise TabletopSelectionError(f"{path} contains non-finite JSON {value}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    except TabletopSelectionError:
        raise
    except Exception as exc:
        raise TabletopSelectionError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TabletopSelectionError(f"{path} must contain one JSON object")
    return payload


def _rotation_6d(rotation: np.ndarray) -> np.ndarray:
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise TabletopSelectionError("hand rotation must be one finite 3x3 matrix")
    return rotation.T[:2].reshape(6)


def _pose_tensor(hand: X2HandModel, payload: dict[str, Any]) -> torch.Tensor:
    hand_pose = payload.get("hand_pose")
    if not isinstance(hand_pose, dict):
        raise TabletopSelectionError("record has no hand_pose object")
    translation = np.asarray(hand_pose.get("translation"), dtype=np.float64)
    rotation = np.asarray(hand_pose.get("rotation_matrix"), dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise TabletopSelectionError("hand translation must be finite 3D")
    names = payload.get("actuator_names")
    values = payload.get("actuator")
    if not isinstance(names, list) or not isinstance(values, list) or len(names) != len(values):
        raise TabletopSelectionError("actuator names and values must be equal-length lists")
    actuator_by_name = dict(zip(names, values))
    if set(actuator_by_name) != set(hand.actuator_names):
        raise TabletopSelectionError("record actuators do not match the X2 model")
    actuator = np.asarray(
        [actuator_by_name[name] for name in hand.actuator_names], dtype=np.float64
    )
    pose = np.concatenate((translation, _rotation_6d(rotation), actuator))
    if not np.isfinite(pose).all():
        raise TabletopSelectionError("materialized hand pose must be finite")
    return torch.as_tensor(pose, dtype=hand.dtype).unsqueeze(0)


def _minimum_plane_clearance(
    hand: X2HandModel,
    payload: dict[str, Any],
    *,
    plane_z_m: float,
) -> float:
    hand.set_parameters(_pose_tensor(hand, payload))
    if hand.current_status is None or hand.global_rotation is None:
        raise TabletopSelectionError("X2 FK did not materialize")
    global_rotation = hand.global_rotation[0].detach().cpu().numpy()
    global_translation = hand.global_translation[0].detach().cpu().numpy()
    minimum_z = math.inf
    for link_name in hand.backend.link_names:
        collision = hand.backend.collision_meshes[link_name]
        local = np.asarray(collision.vertices_local, dtype=np.float64)
        transform = hand.current_status[link_name][0].detach().cpu().numpy()
        root_points = local @ transform[:3, :3].T + transform[:3, 3]
        world_points = root_points @ global_rotation.T + global_translation
        minimum_z = min(minimum_z, float(world_points[:, 2].min()))
    clearance = minimum_z - plane_z_m
    if not math.isfinite(clearance):
        raise TabletopSelectionError("table clearance is not finite")
    return clearance


def _source_metrics(payload: dict[str, Any]) -> tuple[float, float, float, float]:
    validation = payload.get("validation")
    if (
        not isinstance(validation, dict)
        or validation.get("status") != "passed"
        or validation.get("backend") != "isaac_sim_physx"
        or payload.get("simulation_success") is not True
    ):
        raise TabletopSelectionError("record is not an Isaac Sim/PhysX success")
    orientations = validation.get("orientations")
    required = int(validation.get("required_orientation_count", 0))
    if (
        not isinstance(orientations, list)
        or required <= 0
        or len(orientations) != required
        or validation.get("passed_orientation_count") != required
        or any(not isinstance(row, dict) or row.get("passed") is not True for row in orientations)
    ):
        raise TabletopSelectionError("record did not pass every required orientation")
    hand_object = payload.get("hand_object_penetration")
    self_collision = payload.get("self_collision")
    if (
        payload.get("finite") is not True
        or not isinstance(hand_object, dict)
        or hand_object.get("feasible") is not True
        or not isinstance(self_collision, dict)
        or self_collision.get("feasible") is not True
    ):
        raise TabletopSelectionError("record fails a static source gate")
    metrics = (
        max(float(row["maximum_displacement_m"]) for row in orientations),
        min(float(row["final_contact_force_n"]) for row in orientations),
        max(float(row["maximum_active_joint_error_rad"]) for row in orientations),
        float(hand_object["maximum_penetration"]),
    )
    if not all(math.isfinite(value) for value in metrics):
        raise TabletopSelectionError("source robustness metrics must be finite")
    return metrics


def _candidate(
    hand: X2HandModel,
    path: Path,
    *,
    side: str,
    finger_count: int,
    mesh_path: Path,
    plane_z_m: float,
) -> Candidate:
    payload = _read_json(path)
    participation = payload.get("finger_participation")
    object_record = payload.get("object")
    if (
        payload.get("active_side") != side
        or not isinstance(participation, dict)
        or participation.get("target_count") != finger_count
        or participation.get("actual_count") != finger_count
        or not isinstance(object_record, dict)
        or Path(str(object_record.get("mesh_path", ""))).resolve() != mesh_path
        or float(object_record.get("scale", math.nan)) != 1.0
    ):
        raise TabletopSelectionError(f"{path} does not match its requested stratum")
    maximum_displacement, minimum_force, maximum_joint_error, penetration = (
        _source_metrics(payload)
    )
    return Candidate(
        path=path,
        payload=payload,
        clearance_m=_minimum_plane_clearance(hand, payload, plane_z_m=plane_z_m),
        maximum_displacement_m=maximum_displacement,
        minimum_final_contact_force_n=minimum_force,
        maximum_active_joint_error_rad=maximum_joint_error,
        maximum_hand_object_penetration_m=penetration,
    )


def _derived_record(
    candidate: Candidate,
    *,
    sample_index: int,
    plane_z_m: float,
    requested_clearance_m: float,
) -> dict[str, Any]:
    record = copy.deepcopy(candidate.payload)
    source_sha256 = _sha256(candidate.path)
    original_sample_index = record.get("sample_index")
    record["sample_index"] = sample_index
    record["table_conditioning"] = {
        "status": "DERIVED_STATIC_SUPPORT_PLANE_SELECTION",
        "frame": "MESH_OBJECT_FRAME",
        "plane_normal": [0.0, 0.0, 1.0],
        "plane_offset_m": plane_z_m,
        "requested_clearance_m": requested_clearance_m,
        "penalty_weight": 0.0,
        "collision_mesh_vertex_minimum_hand_plane_clearance_m": candidate.clearance_m,
        "source_plane_nonpenetrating": candidate.clearance_m >= 0.0,
        "requested_clearance_met": candidate.clearance_m >= requested_clearance_m,
        "selection_only_no_pose_change": True,
        "not_exact_table_admission": True,
    }
    record["tabletop_static_selection"] = {
        "status": "PASS",
        "criterion": (
            "SOURCE_PHYSX_ALL_ORIENTATIONS_PASS_AND_X2_COLLISION_MESH_"
            "TABLE_CLEARANCE"
        ),
        "source_physx_valid_json": str(candidate.path),
        "source_sha256": source_sha256,
        "source_sample_index": original_sample_index,
        "maximum_source_object_displacement_m": candidate.maximum_displacement_m,
        "minimum_source_final_contact_force_n": candidate.minimum_final_contact_force_n,
        "maximum_source_active_joint_error_rad": candidate.maximum_active_joint_error_rad,
        "maximum_hand_object_penetration_m": candidate.maximum_hand_object_penetration_m,
    }
    record["tabletop_validation"] = {
        "status": "NOT_RUN",
        "reason": "TABLE_WAS_INSERTED_AFTER_SOURCE_PHYSX_VALIDATION",
        "source_physx_validation_preserved": True,
        "not_a_tabletop_dynamic_success_claim": True,
    }
    provenance = record.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise TabletopSelectionError("source provenance must be a JSON object")
    provenance["tabletop_static_source"] = {
        "path": str(candidate.path),
        "sha256": source_sha256,
    }
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.expanduser().resolve()
    mesh_path = args.mesh_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not source_root.is_dir():
        raise TabletopSelectionError(f"Source root does not exist: {source_root}")
    if not mesh_path.is_file():
        raise TabletopSelectionError(f"Mesh does not exist: {mesh_path}")
    if output_root.exists() and any(output_root.iterdir()):
        raise TabletopSelectionError(f"Output root is not empty: {output_root}")

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not mesh.is_watertight:
        raise TabletopSelectionError("Object mesh must be one watertight triangle mesh")
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise TabletopSelectionError("Object bounds must be finite")
    plane_z_m = float(bounds[0, 2])

    config = load_x2_mesh_config()
    contacts = load_generic_contact_candidates(
        config.configured_path("contact_candidates.path", must_exist=True)
    )
    hand = X2HandModel(
        config,
        contacts,
        device="cpu",
        dtype=torch.float64,
        collision_samples_per_link=1,
        self_collision_samples_per_link=1,
    )

    records: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for side in SIDES:
        for finger_count in FINGER_COUNTS:
            pattern = f"{mesh_path.stem}_f{finger_count}_{side}_*.json"
            source_paths = sorted((source_root / side / "valid").glob(pattern))
            if not source_paths:
                raise TabletopSelectionError(f"No source rows for {side}/f{finger_count}")
            candidates = [
                _candidate(
                    hand,
                    path.resolve(),
                    side=side,
                    finger_count=finger_count,
                    mesh_path=mesh_path,
                    plane_z_m=plane_z_m,
                )
                for path in source_paths
            ]
            eligible = [
                candidate
                for candidate in candidates
                if candidate.clearance_m >= args.minimum_clearance_m
            ]
            eligible.sort(key=lambda candidate: candidate.score)
            if len(eligible) < args.per_stratum:
                raise TabletopSelectionError(
                    f"{side}/f{finger_count} has {len(eligible)} rows at "
                    f"{args.minimum_clearance_m:g} m clearance; "
                    f"need {args.per_stratum}"
                )
            output_files: list[str] = []
            for sample_index, candidate in enumerate(eligible[: args.per_stratum]):
                record = _derived_record(
                    candidate,
                    sample_index=sample_index,
                    plane_z_m=plane_z_m,
                    requested_clearance_m=args.minimum_clearance_m,
                )
                directory = output_root / "raw" / side / f"f{finger_count}"
                directory.mkdir(parents=True, exist_ok=True)
                output = directory / (
                    f"{mesh_path.stem}_f{finger_count}_{side}_{sample_index:06d}.json"
                )
                output.write_text(
                    json.dumps(record, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                output_files.append(str(output))
                records.append(
                    {
                        "side": side,
                        "finger_count": finger_count,
                        "finger_names": record["finger_participation"]["finger_names"],
                        "output": str(output),
                        "output_sha256": _sha256(output),
                        "source": str(candidate.path),
                        "source_sha256": _sha256(candidate.path),
                        "table_clearance_m": candidate.clearance_m,
                        "maximum_source_object_displacement_m": (
                            candidate.maximum_displacement_m
                        ),
                        "minimum_source_final_contact_force_n": (
                            candidate.minimum_final_contact_force_n
                        ),
                    }
                )
            groups.append(
                {
                    "side": side,
                    "finger_count": finger_count,
                    "source_count": len(candidates),
                    "table_clearance_eligible_count": len(eligible),
                    "selected_count": args.per_stratum,
                    "output_files": output_files,
                }
            )

    extents_m = bounds[1] - bounds[0]
    manifest = {
        "schema_version": 1,
        "dataset": "x2_tabletop_static_physx_seeded_v1",
        "passed": True,
        "static_record_count": len(records),
        "source_root": str(source_root),
        "object": {
            "mesh_path": str(mesh_path),
            "mesh_sha256": _sha256(mesh_path),
            "scale": 1.0,
            "extents_m": extents_m.tolist(),
            "table_plane_z_m": plane_z_m,
        },
        "selection": {
            "minimum_table_clearance_m": args.minimum_clearance_m,
            "per_side_finger_count": args.per_stratum,
            "sides": list(SIDES),
            "finger_counts": list(FINGER_COUNTS),
            "source_physx_required_all_orientations_pass": True,
            "source_static_gates_required": True,
            "ranking": [
                "minimum maximum source object displacement",
                "maximum minimum source final contact force",
                "minimum dense hand-object penetration",
                "maximum table clearance",
            ],
        },
        "tabletop_validation": {
            "status": "NOT_RUN",
            "source_physx_validation_preserved": True,
            "not_a_tabletop_dynamic_success_claim": True,
        },
        "groups": groups,
        "records": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "GENERATION_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=1)
    parser.add_argument("--minimum-clearance-m", type=float, default=0.008)
    args = parser.parse_args(argv)
    if args.per_stratum <= 0:
        parser.error("per-stratum must be positive")
    if (
        not math.isfinite(args.minimum_clearance_m)
        or args.minimum_clearance_m < 0.0
    ):
        parser.error("minimum-clearance-m must be finite and non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = run(args)
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
    print(json.dumps(manifest, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
