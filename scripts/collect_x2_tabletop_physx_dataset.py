#!/usr/bin/env python3
"""Collect a mixed-object X2 tabletop-static dataset with PhysX evidence.

The campaign uses table-conditioned generation, mounted FR5/X2 inverse
kinematics, and exact FR5 moving-link/table clearance gates before running the
repository's official six-orientation Isaac Sim/PhysX validator.  Every
generator proposal is retained.  Static failures, PhysX failures, and PhysX
successes live in separate audited locations under each attempt.

Important evidence boundary: the physical protocol is the existing
object-centred, no-ground, six-orientation X2 grasp validator.  The tabletop
condition is established by generation-time collision-mesh clearance.  This is
augmented by a deterministic mounted-robot configuration proof, but is not a
simulated table-acquisition or lift-from-table protocol.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_x2_primitive_dataset import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT as PRIMITIVE_MESH_ROOT,
    PRIMITIVE_SPECS,
)


GENERATOR = PROJECT_ROOT / "scripts" / "generate_x2_mesh_grasps.py"
VALIDATOR = PROJECT_ROOT / "scripts" / "validate_x2_mesh_grasps_physx.py"
GENERAL_MESH_ROOT = PROJECT_ROOT / "data" / "meshdata"
GENERAL_MANIFEST = GENERAL_MESH_ROOT / "x2_general_mesh_manifest.json"
FORMAL_GENERAL_IDS = tuple(f"{value:03d}" for value in range(0, 88, 3))
SIDES = ("front", "back")
FINGER_COUNTS = (1, 2, 3, 4, 5)
PROTOCOL_REVISION = "x2_mixed_table_static_physx6d_collection_v2_fr5_table_safe"
PHYSX_PROTOCOL = "x2_object_centered_dexgraspnet_six_orientation_v7"
RELATION_ACT_ROOT = Path("/home/lhr/Desktop/relation_flow_matching/relation_act")
DEFAULT_FR5_VENDOR_ROOT = RELATION_ACT_ROOT / "vendor" / "frcobot_ros2"
DEFAULT_FR5_HOME = (
    RELATION_ACT_ROOT
    / "configs"
    / "retainplan_x2"
    / "authority"
    / "fr5_tabletop_home.json"
)
DEFAULT_FR5_MOUNT = (
    RELATION_ACT_ROOT
    / "configs"
    / "retainplan_x2"
    / "authority"
    / "flange_to_x2_mount.json"
)
DEFAULT_X2_WORKSPACE_BOUNDS = (
    RELATION_ACT_ROOT
    / "artifacts"
    / "x2_runtime"
    / "fr5_x2_continuous_workspace_bounds_v1.json"
)
FR5_MOVING_LINKS = (
    "shoulder_link",
    "upperarm_link",
    "forearm_link",
    "wrist1_link",
    "wrist2_link",
    "wrist3_link",
)
FR5_IK_MAX_ITERATIONS = 160
FR5_IK_POSITION_TOLERANCE_M = 2.0e-4
FR5_IK_ROTATION_TOLERANCE_RAD = 2.0e-3
FR5_IK_DAMPING = 1.0e-3


class TabletopCollectionError(RuntimeError):
    """Raised when a campaign invariant is violated."""


@dataclass(frozen=True)
class ObjectSpec:
    kind: str
    shape: str
    object_id: str
    mesh_path: Path
    mesh_sha256: str
    scale: float
    source_extents_m: tuple[float, float, float]
    physical_extents_m: tuple[float, float, float]
    table_plane_z_m: float

    @property
    def slug(self) -> str:
        return f"{self.kind}_{self.object_id}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "shape": self.shape,
            "object_id": self.object_id,
            "mesh_path": str(self.mesh_path),
            "mesh_sha256": self.mesh_sha256,
            "scale": self.scale,
            "source_extents_m": list(self.source_extents_m),
            "physical_extents_m": list(self.physical_extents_m),
            "table_plane_z_m": self.table_plane_z_m,
        }


@dataclass(frozen=True)
class GPUDevice:
    index: int
    name: str
    total_memory_mb: int
    free_memory_mb: int
    utilization_percent: int

    @property
    def torch_device(self) -> str:
        return f"cuda:{self.index}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "total_memory_mb": self.total_memory_mb,
            "free_memory_mb_at_plan_time": self.free_memory_mb,
            "utilization_percent_at_plan_time": self.utilization_percent,
        }


@dataclass(frozen=True)
class GPUExecutionPlan:
    mode: str
    inventory: tuple[GPUDevice, ...]
    generation_slots: tuple[str, ...]
    validation_slots: tuple[str, ...]
    generation_batch_size: int
    validation_batch_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "inventory": [device.as_dict() for device in self.inventory],
            "generation_slots": list(self.generation_slots),
            "validation_slots": list(self.validation_slots),
            "generation_worker_count": len(self.generation_slots),
            "validation_worker_count": len(self.validation_slots),
            "generation_batch_size": self.generation_batch_size,
            "validation_batch_size": self.validation_batch_size,
        }


def _query_nvidia_gpus() -> tuple[GPUDevice, ...]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TabletopCollectionError(f"Cannot query NVIDIA GPUs: {exc}") from exc
    if completed.returncode != 0:
        raise TabletopCollectionError(
            "nvidia-smi GPU query failed: " + completed.stderr.strip()
        )
    devices: list[GPUDevice] = []
    for raw_line in completed.stdout.splitlines():
        fields = [value.strip() for value in raw_line.split(",")]
        if len(fields) != 5:
            raise TabletopCollectionError(
                f"Unexpected nvidia-smi inventory row: {raw_line!r}"
            )
        try:
            devices.append(
                GPUDevice(
                    index=int(fields[0]),
                    name=fields[1],
                    total_memory_mb=int(fields[2]),
                    free_memory_mb=int(fields[3]),
                    utilization_percent=int(fields[4]),
                )
            )
        except ValueError as exc:
            raise TabletopCollectionError(
                f"Invalid nvidia-smi inventory row: {raw_line!r}"
            ) from exc
    if not devices:
        raise TabletopCollectionError("No NVIDIA GPU is visible")
    return tuple(sorted(devices, key=lambda value: value.index))


def _auto_gpu_execution_plan(
    inventory: Sequence[GPUDevice],
    *,
    selected_indices: Sequence[int] | None,
    requested_generation_batch_size: int,
) -> GPUExecutionPlan:
    selected_set = None if selected_indices is None else set(selected_indices)
    selected = tuple(
        device
        for device in inventory
        if selected_set is None or device.index in selected_set
    )
    if selected_set is not None and selected_set != {value.index for value in selected}:
        missing = sorted(selected_set - {value.index for value in selected})
        raise TabletopCollectionError(f"Requested GPU indices are unavailable: {missing}")
    if not selected:
        raise TabletopCollectionError("Automatic GPU plan selected no devices")

    generation_slots: list[str] = []
    validation_slots: list[str] = []
    validation_batches: list[int] = []
    for device in selected:
        # Two batch-32 optimizers already saturate an Ada/Blackwell-class
        # 24+ GiB GPU. More workers add context contention without useful
        # throughput; smaller cards receive one worker.
        generation_workers = 2 if device.total_memory_mb >= 24 * 1024 else 1
        generation_workers = min(
            generation_workers,
            max(1, (device.free_memory_mb - 2048) // 4096),
        )
        # PhysX benefits from both a larger batch and two object-level workers
        # on 24+ GiB cards because asset preparation is partly CPU-bound.
        validation_workers = 2 if device.total_memory_mb >= 24 * 1024 else 1
        validation_workers = min(
            validation_workers,
            max(1, (device.free_memory_mb - 4096) // 8192),
        )
        if device.total_memory_mb >= 24 * 1024:
            validation_batch = 32
        elif device.total_memory_mb >= 16 * 1024:
            validation_batch = 24
        elif device.total_memory_mb >= 10 * 1024:
            validation_batch = 16
        else:
            validation_batch = 8
        generation_slots.extend([device.torch_device] * generation_workers)
        validation_slots.extend([device.torch_device] * validation_workers)
        validation_batches.append(validation_batch)
    return GPUExecutionPlan(
        mode="auto",
        inventory=selected,
        generation_slots=tuple(generation_slots),
        validation_slots=tuple(validation_slots),
        generation_batch_size=requested_generation_batch_size,
        validation_batch_size=min(validation_batches),
    )


def _resolve_gpu_execution_plan(args: argparse.Namespace) -> GPUExecutionPlan:
    if args.auto_gpu:
        return _auto_gpu_execution_plan(
            _query_nvidia_gpus(),
            selected_indices=args.gpu_indices,
            requested_generation_batch_size=args.generation_batch_size,
        )
    return GPUExecutionPlan(
        mode="manual",
        inventory=(),
        generation_slots=tuple(args.generation_device for _ in range(args.jobs)),
        validation_slots=tuple(
            args.validation_device for _ in range(args.validation_jobs)
        ),
        generation_batch_size=args.generation_batch_size,
        validation_batch_size=args.validation_batch_size,
    )


def _quaternion_xyzw_matrix(values: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise TabletopCollectionError("Quaternion must be a finite xyzw 4-vector")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise TabletopCollectionError("Quaternion norm is zero")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rpy_matrix(values: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in values)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _xml_vector(raw: str | None) -> np.ndarray:
    values = [float(value) for value in (raw or "0 0 0").split()]
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise TabletopCollectionError(f"Invalid URDF 3-vector: {raw!r}")
    return vector


class FR5TableCollisionGate:
    """Existential FR5 IK plus exact moving-link/table clearance gate.

    The fixed ``base_link`` is intentionally excluded because its nominal
    mounting origin lies on the tabletop.  Every moving FR5 collision mesh is
    checked against an infinite horizontal plane, which is conservative with
    respect to the finite tabletop rectangle.
    """

    def __init__(
        self,
        *,
        vendor_root: Path,
        home_path: Path,
        mount_path: Path,
        x2_workspace_bounds_path: Path,
        object_table_xy_m: Sequence[float],
        minimum_x2_root_table_distance_m: float,
        robot_table_clearance_m: float,
        ik_seed_count: int,
    ) -> None:
        self.vendor_root = vendor_root.expanduser().resolve()
        self.home_path = home_path.expanduser().resolve()
        self.mount_path = mount_path.expanduser().resolve()
        self.x2_workspace_bounds_path = x2_workspace_bounds_path.expanduser().resolve()
        self.object_table_xy_m = np.asarray(object_table_xy_m, dtype=np.float64)
        self.minimum_x2_root_table_distance_m = float(
            minimum_x2_root_table_distance_m
        )
        self.robot_table_clearance_m = float(robot_table_clearance_m)
        self.ik_seed_count = int(ik_seed_count)
        if (
            self.object_table_xy_m.shape != (2,)
            or not np.isfinite(self.object_table_xy_m).all()
            or not math.isfinite(self.minimum_x2_root_table_distance_m)
            or self.minimum_x2_root_table_distance_m <= 0.0
            or not math.isfinite(self.robot_table_clearance_m)
            or self.robot_table_clearance_m <= 0.0
            or self.ik_seed_count < 3
        ):
            raise TabletopCollectionError("Invalid FR5/table gate configuration")

        required = (
            self.home_path,
            self.mount_path,
            self.x2_workspace_bounds_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise TabletopCollectionError(
                "FR5/table authority file is missing: " + ", ".join(missing)
            )
        relation_src = self.home_path.parents[3] / "src"
        if not relation_src.is_dir():
            relation_src = RELATION_ACT_ROOT / "src"
        if str(relation_src) not in sys.path:
            sys.path.insert(0, str(relation_src))
        try:
            from relation_act.x2.retainplan import (  # type: ignore[import-not-found]
                FR5Binding,
                so3_exp,
                solve_fr5_ik,
            )
        except Exception as exc:
            raise TabletopCollectionError(
                f"Cannot import the FR5 kinematics authority from {relation_src}: {exc}"
            ) from exc
        self._so3_exp = so3_exp
        self._solve_fr5_ik = solve_fr5_ik

        home = _strict_json(self.home_path)
        mount = _strict_json(self.mount_path)
        workspace = _strict_json(self.x2_workspace_bounds_path)
        try:
            self.table_top_z_m = float(home["tabletop_bounds_world_m"]["top_z"])
            bounds_min = np.asarray(
                home["tabletop_bounds_world_m"]["minimum_xy"], dtype=np.float64
            )
            bounds_max = np.asarray(
                home["tabletop_bounds_world_m"]["maximum_xy"], dtype=np.float64
            )
            self.q_home_rad = np.deg2rad(
                np.asarray(home["joint_positions_deg"], dtype=np.float64)
            )
            self.x2_root_subtree_radius_m = float(
                workspace["x2_root_subtree_radius_m"]
            )
            self.T_W_B = np.eye(4, dtype=np.float64)
            self.T_W_B[:3, :3] = _quaternion_xyzw_matrix(
                home["base_quaternion_xyzw"]
            )
            self.T_W_B[:3, 3] = np.asarray(
                home["base_translation_m"], dtype=np.float64
            )
            T_F_H = np.eye(4, dtype=np.float64)
            T_F_H[:3, :3] = _quaternion_xyzw_matrix(
                mount["quaternion_xyzw"]
            )
            T_F_H[:3, 3] = np.asarray(mount["translation_m"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise TabletopCollectionError(f"Invalid FR5 authority JSON: {exc}") from exc
        if (
            bounds_min.shape != (2,)
            or bounds_max.shape != (2,)
            or self.q_home_rad.shape != (6,)
            or not np.isfinite(self.q_home_rad).all()
            or not math.isfinite(self.table_top_z_m)
            or not math.isfinite(self.x2_root_subtree_radius_m)
            or self.x2_root_subtree_radius_m <= 0.0
            or np.any(self.object_table_xy_m < bounds_min)
            or np.any(self.object_table_xy_m > bounds_max)
        ):
            raise TabletopCollectionError(
                "FR5 home pose, X2 bound, or tabletop object XY is invalid"
            )
        self.tabletop_bounds_xy_m = (bounds_min, bounds_max)
        self.binding = FR5Binding.from_vendor(
            self.vendor_root,
            flange_to_x2=T_F_H,
            calibration_status=str(mount.get("source_type", "UNKNOWN")),
        )
        self._arm_vertices_by_link = self._load_arm_collision_support_vertices()
        self._authority_cache: dict[str, Any] | None = None

    def _load_arm_collision_support_vertices(self) -> dict[str, np.ndarray]:
        urdf = self.binding.urdf_path
        root = ET.parse(urdf).getroot()
        links = {str(element.get("name")): element for element in root.findall("link")}
        result: dict[str, np.ndarray] = {}
        package_root = self.vendor_root / "fairino_description"
        prefix = "package://fairino_description/"
        mesh_authority: dict[str, dict[str, Any]] = {}
        for link_name in FR5_MOVING_LINKS:
            link = links.get(link_name)
            collisions = [] if link is None else link.findall("collision")
            if len(collisions) != 1:
                raise TabletopCollectionError(
                    f"FR5 {link_name} must have exactly one collision element"
                )
            collision = collisions[0]
            mesh_element = collision.find("geometry/mesh")
            filename = None if mesh_element is None else mesh_element.get("filename")
            if not isinstance(filename, str) or not filename.startswith(prefix):
                raise TabletopCollectionError(
                    f"FR5 {link_name} collision is not an authority mesh"
                )
            mesh_path = (package_root / filename[len(prefix) :]).resolve()
            if not mesh_path.is_file():
                raise TabletopCollectionError(f"FR5 collision mesh is missing: {mesh_path}")
            loaded = trimesh.load(mesh_path, force="mesh", process=False)
            if not isinstance(loaded, trimesh.Trimesh):
                raise TabletopCollectionError(f"Invalid FR5 collision mesh: {mesh_path}")
            # Plane support is unchanged by convexification; the smaller hull
            # vertex set makes the per-IK exact minimum-height query cheap.
            vertices = np.asarray(loaded.convex_hull.vertices, dtype=np.float64)
            raw_scale = mesh_element.get("scale")
            scale = (
                np.ones(3, dtype=np.float64)
                if raw_scale is None
                else _xml_vector(raw_scale)
            )
            if np.any(scale <= 0.0):
                raise TabletopCollectionError(f"Invalid FR5 mesh scale: {mesh_path}")
            vertices = vertices * scale
            origin = collision.find("origin")
            xyz = _xml_vector(None if origin is None else origin.get("xyz"))
            rpy = _xml_vector(None if origin is None else origin.get("rpy"))
            vertices = vertices @ _rpy_matrix(rpy).T + xyz
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                raise TabletopCollectionError(f"Invalid FR5 vertices: {mesh_path}")
            result[link_name] = vertices
            mesh_authority[link_name] = {
                "path": str(mesh_path),
                "sha256": _sha256(mesh_path),
                "source_vertex_count": int(len(loaded.vertices)),
                "support_hull_vertex_count": int(len(vertices)),
                "collision_origin_xyz_m": xyz.tolist(),
                "collision_origin_rpy_rad": rpy.tolist(),
                "mesh_scale": scale.tolist(),
            }
        self._arm_mesh_authority = mesh_authority
        return result

    def as_dict(self) -> dict[str, Any]:
        if self._authority_cache is not None:
            return self._authority_cache
        bounds_min, bounds_max = self.tabletop_bounds_xy_m
        authority = {
            "gate": "existential_fr5_ik_plus_moving_link_infinite_table_plane_v1",
            "fr5_model": self.binding.as_dict(),
            "fr5_moving_link_collision_meshes": self._arm_mesh_authority,
            "home_authority": str(self.home_path),
            "home_authority_sha256": _sha256(self.home_path),
            "mount_authority": str(self.mount_path),
            "mount_authority_sha256": _sha256(self.mount_path),
            "x2_workspace_bounds": str(self.x2_workspace_bounds_path),
            "x2_workspace_bounds_sha256": _sha256(self.x2_workspace_bounds_path),
            "x2_root_subtree_radius_m": self.x2_root_subtree_radius_m,
            "table_top_z_m": self.table_top_z_m,
            "tabletop_bounds_xy_m": {
                "minimum": bounds_min.tolist(),
                "maximum": bounds_max.tolist(),
            },
            "object_table_xy_m": self.object_table_xy_m.tolist(),
            "minimum_x2_root_table_distance_m": self.minimum_x2_root_table_distance_m,
            "robot_table_clearance_m": self.robot_table_clearance_m,
            "checked_fr5_links": list(FR5_MOVING_LINKS),
            "excluded_fixed_link": "base_link",
            "table_geometry": "infinite_horizontal_plane_conservative_to_finite_table",
            "ik": {
                "seed_count": self.ik_seed_count,
                "max_iterations": FR5_IK_MAX_ITERATIONS,
                "position_tolerance_m": FR5_IK_POSITION_TOLERANCE_M,
                "rotation_tolerance_rad": FR5_IK_ROTATION_TOLERANCE_RAD,
                "damping": FR5_IK_DAMPING,
                "purpose": "existence_of_collision_free_mounted_configuration",
            },
        }
        self._authority_cache = authority
        return authority

    def record_authority(self) -> dict[str, Any]:
        authority = self.as_dict()
        return {
            "protocol_revision": PROTOCOL_REVISION,
            "gate": authority["gate"],
            "home_authority_sha256": authority["home_authority_sha256"],
            "mount_authority_sha256": authority["mount_authority_sha256"],
            "x2_workspace_bounds_sha256": authority["x2_workspace_bounds_sha256"],
            "fr5_urdf_sha256": self.binding.urdf_sha256,
            "fr5_collision_mesh_sha256": {
                link: record["sha256"]
                for link, record in self._arm_mesh_authority.items()
            },
            "table_top_z_m": self.table_top_z_m,
            "object_table_xy_m": self.object_table_xy_m.tolist(),
            "minimum_x2_root_table_distance_m": self.minimum_x2_root_table_distance_m,
            "robot_table_clearance_m": self.robot_table_clearance_m,
            "checked_fr5_links": list(FR5_MOVING_LINKS),
            "excluded_fixed_link": "base_link",
            "table_geometry": "infinite_horizontal_plane_conservative_to_finite_table",
        }

    def _ik_seeds(self, record_key: str) -> list[tuple[str, np.ndarray]]:
        q0 = self.q_home_rad
        seeds: list[tuple[str, np.ndarray]] = [
            ("tabletop_home", q0),
            ("nominal_elbow_up", q0 + np.asarray([0.0, -0.35, 0.45, 0.0, 0.0, 0.0])),
            ("nominal_elbow_down", q0 + np.asarray([0.0, 0.35, -0.45, 0.0, 0.0, 0.0])),
        ]
        digest = hashlib.sha256(record_key.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        seeds.extend(
            [
                ("deterministic_small_perturbation", q0 + rng.normal(0.0, 0.22, 6)),
                ("deterministic_broad_perturbation", q0 + rng.normal(0.0, 0.85, 6)),
            ]
        )
        while len(seeds) < self.ik_seed_count:
            seeds.append(
                (
                    f"deterministic_joint_space_seed_{len(seeds) - 4}",
                    rng.uniform(self.binding.lower_limits, self.binding.upper_limits),
                )
            )
        return [
            (
                name,
                np.minimum(
                    np.maximum(values, self.binding.lower_limits + 1.0e-6),
                    self.binding.upper_limits - 1.0e-6,
                ),
            )
            for name, values in seeds[: self.ik_seed_count]
        ]

    def _arm_table_clearances(self, q_arm: Sequence[float]) -> dict[str, float]:
        q = np.asarray(q_arm, dtype=np.float64)
        poses: dict[str, np.ndarray] = {}
        T_B_L = np.eye(4, dtype=np.float64)
        for index, joint in enumerate(self.binding.joints):
            T_B_L = T_B_L @ joint.origin
            T_B_L = T_B_L.copy()
            T_B_L[:3, :3] = T_B_L[:3, :3] @ self._so3_exp(
                joint.axis * q[index]
            )
            poses[joint.child] = T_B_L.copy()
        clearances: dict[str, float] = {}
        for link_name, vertices in self._arm_vertices_by_link.items():
            T_W_L = self.T_W_B @ poses[link_name]
            minimum_z = float(
                np.min(vertices @ T_W_L[2, :3] + T_W_L[2, 3])
            )
            clearances[link_name] = minimum_z - self.table_top_z_m
        return clearances

    def evaluate(
        self,
        payload: Mapping[str, Any],
        *,
        spec: ObjectSpec,
        record_key: str,
    ) -> dict[str, Any]:
        base: dict[str, Any] = {
            "status": "FAIL",
            "gate": "FR5_X2_MOUNTED_TABLE_COLLISION_FREE",
            "record_key": record_key,
            "authority": self.record_authority(),
            "failure_reasons": [],
            "ik_attempts": [],
        }
        try:
            hand_pose = payload["hand_pose"]
            table = payload["table_conditioning"]
            if not isinstance(hand_pose, Mapping) or not isinstance(table, Mapping):
                raise ValueError("hand_pose or table_conditioning is absent")
            translation = np.asarray(hand_pose["translation"], dtype=np.float64)
            rotation = np.asarray(hand_pose["rotation_matrix"], dtype=np.float64)
            desired_x2_clearance = float(
                table["collision_mesh_vertex_minimum_hand_plane_clearance_m"]
            )
            if (
                translation.shape != (3,)
                or rotation.shape != (3, 3)
                or not np.isfinite(translation).all()
                or not np.isfinite(rotation).all()
                or not math.isfinite(desired_x2_clearance)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6)
                or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-6)
            ):
                raise ValueError("hand pose or desired X2 clearance is invalid")
        except (KeyError, TypeError, ValueError) as exc:
            base["failure_reasons"] = ["FR5_GATE_INPUT_INVALID"]
            base["error"] = str(exc)
            return base

        T_O_H = np.eye(4, dtype=np.float64)
        T_O_H[:3, :3] = rotation
        T_O_H[:3, 3] = translation
        T_W_O = np.eye(4, dtype=np.float64)
        T_W_O[:2, 3] = self.object_table_xy_m
        T_W_O[2, 3] = self.table_top_z_m - spec.table_plane_z_m
        T_W_H = T_W_O @ T_O_H
        T_B_H = np.linalg.inv(self.T_W_B) @ T_W_H
        target_root_distance = float(T_W_H[2, 3] - self.table_top_z_m)
        base.update(
            {
                "object_world_translation_m": T_W_O[:3, 3].tolist(),
                "target_x2_root_world_translation_m": T_W_H[:3, 3].tolist(),
                "target_x2_root_table_distance_m": target_root_distance,
                "desired_x2_collision_mesh_table_clearance_m": desired_x2_clearance,
                "target_base_to_x2_transform": T_B_H.tolist(),
            }
        )
        if target_root_distance < self.minimum_x2_root_table_distance_m:
            base["failure_reasons"] = ["X2_ROOT_TABLE_DISTANCE_FAILED"]
            return base

        observed_failures: set[str] = set()
        any_converged = False
        for seed_name, seed in self._ik_seeds(record_key):
            result = self._solve_fr5_ik(
                self.binding,
                T_B_H,
                seed,
                max_iterations=FR5_IK_MAX_ITERATIONS,
                position_tolerance_m=FR5_IK_POSITION_TOLERANCE_M,
                rotation_tolerance_rad=FR5_IK_ROTATION_TOLERANCE_RAD,
                damping=FR5_IK_DAMPING,
            )
            attempt: dict[str, Any] = {
                "seed": seed_name,
                "converged": bool(result.converged),
                "iterations": int(result.iterations),
                "position_error_m": float(result.position_error_m),
                "rotation_error_rad": float(result.rotation_error_rad),
                "message": str(result.message),
            }
            if not result.converged:
                base["ik_attempts"].append(attempt)
                continue
            any_converged = True
            q_solution = np.asarray(result.q, dtype=np.float64)
            clearances = self._arm_table_clearances(q_solution)
            colliding_links = sorted(
                link
                for link, clearance in clearances.items()
                if clearance < self.robot_table_clearance_m
            )
            T_W_H_achieved = self.T_W_B @ self.binding.hand_fk(q_solution)
            achieved_root_distance = float(
                T_W_H_achieved[2, 3] - self.table_top_z_m
            )
            rotation_displacement_bound = 2.0 * self.x2_root_subtree_radius_m * math.sin(
                0.5 * float(result.rotation_error_rad)
            )
            achieved_x2_clearance_lower_bound = (
                desired_x2_clearance
                - float(result.position_error_m)
                - rotation_displacement_bound
            )
            solution_failures: list[str] = []
            if colliding_links:
                solution_failures.append("FR5_TABLE_COLLISION")
            if achieved_root_distance < self.minimum_x2_root_table_distance_m:
                solution_failures.append("X2_ROOT_TABLE_DISTANCE_FAILED")
            if achieved_x2_clearance_lower_bound < self.robot_table_clearance_m:
                solution_failures.append("MOUNTED_X2_TABLE_CLEARANCE_FAILED")
            observed_failures.update(solution_failures)
            attempt.update(
                {
                    "q_arm_rad": q_solution.tolist(),
                    "q_arm_deg": np.rad2deg(q_solution).tolist(),
                    "fr5_link_table_clearance_m": dict(sorted(clearances.items())),
                    "minimum_fr5_link_table_clearance_m": min(clearances.values()),
                    "fr5_table_collision_links": colliding_links,
                    "achieved_x2_root_world_translation_m": T_W_H_achieved[:3, 3].tolist(),
                    "achieved_x2_root_table_distance_m": achieved_root_distance,
                    "x2_rotation_displacement_bound_m": rotation_displacement_bound,
                    "achieved_x2_table_clearance_lower_bound_m": achieved_x2_clearance_lower_bound,
                    "solution_failure_reasons": solution_failures,
                }
            )
            base["ik_attempts"].append(attempt)
            if not solution_failures:
                base.update(
                    {
                        "status": "PASS",
                        "failure_reasons": [],
                        "selected_seed": seed_name,
                        "selected_q_arm_rad": q_solution.tolist(),
                        "selected_q_arm_deg": np.rad2deg(q_solution).tolist(),
                        "minimum_fr5_link_table_clearance_m": min(clearances.values()),
                        "achieved_x2_table_clearance_lower_bound_m": achieved_x2_clearance_lower_bound,
                        "achieved_x2_root_table_distance_m": achieved_root_distance,
                    }
                )
                return base

        base["failure_reasons"] = sorted(
            observed_failures if any_converged else {"FR5_IK_FAILED"}
        )
        return base


@dataclass(frozen=True)
class GenerationTask:
    attempt_index: int
    object_index: int
    spec: ObjectSpec
    finger_count: int
    num_grasps_per_side: int
    output_root: Path
    n_iterations: int
    batch_size: int
    device: str
    table_clearance_m: float
    fr5_gate: FR5TableCollisionGate

    @property
    def seed(self) -> int:
        return (
            82850000
            + self.attempt_index * 1000003
            + self.object_index * 1009
            + self.finger_count * 31
        )

    @property
    def task_root(self) -> Path:
        return (
            self.output_root
            / "objects"
            / self.spec.slug
            / "generation"
            / f"f{self.finger_count}"
        )

    @property
    def generator_output(self) -> Path:
        return self.task_root / "generator_output"

    @property
    def completion_path(self) -> Path:
        return self.task_root / "complete.json"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise TabletopCollectionError(f"{path} contains non-finite JSON {value}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    except TabletopCollectionError:
        raise
    except Exception as exc:
        raise TabletopCollectionError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TabletopCollectionError(f"{path} must contain one JSON object")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        raise


def _mesh_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or not loaded.is_watertight:
        raise TabletopCollectionError(f"Mesh is not one watertight Trimesh: {path}")
    bounds = np.asarray(loaded.bounds, dtype=np.float64)
    extents = np.asarray(loaded.extents, dtype=np.float64)
    if (
        bounds.shape != (2, 3)
        or extents.shape != (3,)
        or not np.isfinite(bounds).all()
        or not np.isfinite(extents).all()
        or np.any(extents <= 0.0)
    ):
        raise TabletopCollectionError(f"Mesh has invalid bounds: {path}")
    return bounds, extents


def _general_manifest_records() -> dict[str, dict[str, Any]]:
    manifest = _strict_json(GENERAL_MANIFEST)
    rows = manifest.get("meshes")
    if not isinstance(rows, list):
        raise TabletopCollectionError("General mesh manifest has no meshes list")
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("object_id"), str):
            raise TabletopCollectionError("General mesh manifest contains an invalid row")
        object_id = row["object_id"]
        if object_id in records:
            raise TabletopCollectionError(f"Duplicate general object ID: {object_id}")
        records[object_id] = row
    return records


def build_catalog(general_target_max_extent_m: float) -> tuple[ObjectSpec, ...]:
    """Build 12 primitives plus the formal 30 general meshes."""

    catalog: list[ObjectSpec] = []
    primitive_root = PRIMITIVE_MESH_ROOT.expanduser().resolve()
    for primitive in PRIMITIVE_SPECS:
        path = (primitive_root / primitive.relative_path).resolve()
        if not path.is_file():
            raise TabletopCollectionError(f"Primitive mesh is missing: {path}")
        bounds, extents = _mesh_geometry(path)
        catalog.append(
            ObjectSpec(
                kind="primitive",
                shape=primitive.shape,
                object_id=primitive.instance_name,
                mesh_path=path,
                mesh_sha256=_sha256(path),
                scale=1.0,
                source_extents_m=tuple(float(value) for value in extents),
                physical_extents_m=tuple(float(value) for value in extents),
                table_plane_z_m=float(bounds[0, 2]),
            )
        )

    manifest = _general_manifest_records()
    for object_id in FORMAL_GENERAL_IDS:
        row = manifest.get(object_id)
        if row is None:
            raise TabletopCollectionError(f"Formal general mesh is missing: {object_id}")
        path = (GENERAL_MESH_ROOT / object_id / "coacd" / "decomposed.obj").resolve()
        if not path.is_file() or _sha256(path) != row.get("sha256"):
            raise TabletopCollectionError(f"General mesh hash mismatch: {object_id}")
        bounds, extents = _mesh_geometry(path)
        scale = float(general_target_max_extent_m / float(extents.max()))
        if not math.isfinite(scale) or scale <= 0.0:
            raise TabletopCollectionError(f"Invalid normalized scale for {object_id}")
        physical = extents * scale
        catalog.append(
            ObjectSpec(
                kind="general",
                shape="general",
                object_id=object_id,
                mesh_path=path,
                mesh_sha256=_sha256(path),
                scale=scale,
                source_extents_m=tuple(float(value) for value in extents),
                physical_extents_m=tuple(float(value) for value in physical),
                table_plane_z_m=float(bounds[0, 2] * scale),
            )
        )
    if len(catalog) != 42 or len({spec.slug for spec in catalog}) != 42:
        raise TabletopCollectionError("Mixed catalog must contain 42 unique objects")
    return tuple(catalog)


def _settings(
    args: argparse.Namespace,
    catalog: Sequence[ObjectSpec],
    fr5_gate: FR5TableCollisionGate,
) -> dict[str, Any]:
    finger_counts = _active_finger_counts(args)
    stratum_targets = _stratum_targets(args)
    per_stratum_values = set(stratum_targets.values())
    return {
        "protocol_revision": PROTOCOL_REVISION,
        "target_total": args.target_total,
        "finger_counts": list(finger_counts),
        "per_side_target": args.target_total // len(SIDES),
        "per_side_finger_target": (
            next(iter(per_stratum_values)) if len(per_stratum_values) == 1 else None
        ),
        "side_finger_targets": {
            side: {
                str(finger): stratum_targets[(side, finger)]
                for finger in finger_counts
            }
            for side in SIDES
        },
        "n_iterations": args.n_iterations,
        "generation_device": args.generation_device,
        "generation_jobs": args.jobs,
        "generation_batch_size": args.generation_batch_size,
        "validation_device": args.validation_device,
        "validation_batch_size": args.validation_batch_size,
        "sim_steps": args.sim_steps,
        "substeps": args.substeps,
        "hand_friction": args.hand_friction,
        "object_friction": args.object_friction,
        "physx_contact_optimizer": {
            "enabled": True,
            "method": "collision_aware_contact_gradient_line_search",
            "contact_threshold_m": args.closing_contact_threshold,
            "target_displacement_m": args.closing_displacement,
            "gradient_scale": args.closing_gradient_scale,
            "maximum_penetration_m": args.closing_penetration_cap,
            "preclose_physics_steps": 0,
        },
        "table_clearance_m": args.table_clearance_m,
        "general_target_max_extent_m": args.general_target_max_extent_m,
        "expected_physx_pass_rate": args.expected_physx_pass_rate,
        "minimum_raw_per_object_stratum": args.minimum_raw_per_object_stratum,
        "maximum_raw_per_object_stratum": args.maximum_raw_per_object_stratum,
        "minimum_object_coverage": args.minimum_object_coverage,
        "minimum_general_coverage": args.minimum_general_coverage,
        "catalog": [spec.as_dict() for spec in catalog],
        "fr5_x2_table_collision_gate": fr5_gate.as_dict(),
        "physical_validation": {
            "backend": "isaac_sim_physx",
            "protocol_revision": PHYSX_PROTOCOL,
            "six_orientation": True,
            "table_collider_present": False,
            "tabletop_evidence": (
                "generation_x2_collision_mesh_clearance_plus_existential_fr5_ik_"
                "and_exact_moving_link_plane_clearance"
            ),
        },
    }


def _record_finger_count(payload: Mapping[str, Any]) -> int:
    participation = payload.get("finger_participation")
    if not isinstance(participation, Mapping):
        raise TabletopCollectionError("Record has no finger_participation object")
    value = participation.get("actual_count")
    if isinstance(value, bool) or not isinstance(value, int) or value not in FINGER_COUNTS:
        raise TabletopCollectionError("Record has an invalid participating-finger count")
    return value


def _active_finger_counts(args: argparse.Namespace) -> tuple[int, ...]:
    """Return the configured finger strata, with legacy-test compatibility."""
    return tuple(getattr(args, "finger_counts", FINGER_COUNTS))


def _stratum_targets(args: argparse.Namespace) -> dict[tuple[str, int], int]:
    """Allocate an exact total while keeping front/back quotas identical.

    When the per-side total is not divisible by the number of finger strata,
    the one-record remainder is assigned deterministically in CLI order.  For
    10,000 records and ``--finger-counts 2 3 5``, each side therefore receives
    f2=1,667, f3=1,667, and f5=1,666.
    """
    finger_counts = _active_finger_counts(args)
    per_side_total = args.target_total // len(SIDES)
    base, remainder = divmod(per_side_total, len(finger_counts))
    per_finger = {
        finger: base + int(index < remainder)
        for index, finger in enumerate(finger_counts)
    }
    return {
        (side, finger): per_finger[finger]
        for side in SIDES
        for finger in finger_counts
    }


def _object_by_mesh(catalog: Sequence[ObjectSpec]) -> dict[tuple[Path, float], ObjectSpec]:
    return {(spec.mesh_path, spec.scale): spec for spec in catalog}


def _static_failure_reasons(
    payload: Mapping[str, Any],
    *,
    spec: ObjectSpec,
    finger_count: int,
) -> list[str]:
    reasons: list[str] = []
    if payload.get("finite") is not True:
        reasons.append("NONFINITE_GENERATOR_RECORD")
    try:
        actual_finger_count = _record_finger_count(payload)
    except TabletopCollectionError:
        actual_finger_count = -1
    if actual_finger_count != finger_count:
        reasons.append("FINGER_COUNT_MISMATCH")
    side = payload.get("active_side")
    if side not in SIDES:
        reasons.append("INVALID_ACTIVE_SIDE")
    object_record = payload.get("object")
    if not isinstance(object_record, Mapping):
        reasons.append("MISSING_OBJECT_RECORD")
    else:
        try:
            mesh_matches = Path(str(object_record.get("mesh_path", ""))).resolve() == spec.mesh_path
            scale_matches = math.isclose(
                float(object_record.get("scale", math.nan)),
                spec.scale,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        except (TypeError, ValueError):
            mesh_matches = False
            scale_matches = False
        if not mesh_matches:
            reasons.append("OBJECT_MESH_MISMATCH")
        if not scale_matches:
            reasons.append("OBJECT_SCALE_MISMATCH")
    contact = payload.get("selected_contact_realization")
    if not isinstance(contact, Mapping) or contact.get("status") != "PASS":
        reasons.append("SELECTED_CONTACT_REALIZATION_FAILED")
    table = payload.get("table_conditioning")
    if not isinstance(table, Mapping):
        reasons.append("MISSING_TABLE_CONDITIONING")
    else:
        if table.get("source_plane_nonpenetrating") is not True:
            reasons.append("HAND_TABLE_PENETRATION")
        if table.get("requested_clearance_met") is not True:
            reasons.append("TABLE_CLEARANCE_MARGIN_FAILED")
        try:
            plane_matches = math.isclose(
                float(table.get("plane_offset_m", math.nan)),
                spec.table_plane_z_m,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        except (TypeError, ValueError):
            plane_matches = False
        if not plane_matches:
            reasons.append("TABLE_PLANE_MISMATCH")
    hand_object = payload.get("hand_object_penetration")
    if not isinstance(hand_object, Mapping) or hand_object.get("feasible") is not True:
        reasons.append("DENSE_HAND_OBJECT_GATE_FAILED")
    self_collision = payload.get("self_collision")
    if not isinstance(self_collision, Mapping) or self_collision.get("feasible") is not True:
        reasons.append("SELF_COLLISION_GATE_FAILED")
    development = payload.get("development_rejection")
    if isinstance(development, Mapping):
        reason = development.get("reason")
        reasons.append(str(reason) if isinstance(reason, str) else "GENERATOR_REJECTED")
    return sorted(set(reasons))


def _copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise TabletopCollectionError(f"Existing routed copy changed: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _write_static_failure(
    source: Path,
    payload: Mapping[str, Any],
    destination: Path,
    reasons: Sequence[str],
    fr5_validation: Mapping[str, Any],
) -> None:
    failed = dict(payload)
    failed["fr5_mount_table_validation"] = dict(fr5_validation)
    failed["campaign_static_validation"] = {
        "status": "failed",
        "reasons": list(reasons),
        "source_raw": str(source),
        "source_sha256": _sha256(source),
        "physx_run": False,
    }
    if destination.exists():
        existing = _strict_json(destination)
        if existing != failed:
            raise TabletopCollectionError(f"Static failure output changed: {destination}")
        return
    _atomic_json(destination, failed)


def _write_static_eligible(
    source: Path,
    payload: Mapping[str, Any],
    destination: Path,
    fr5_validation: Mapping[str, Any],
) -> None:
    eligible = dict(payload)
    eligible["fr5_mount_table_validation"] = dict(fr5_validation)
    eligible["campaign_static_validation"] = {
        "status": "passed",
        "reasons": [],
        "source_raw": str(source),
        "source_sha256": _sha256(source),
        "physx_run": True,
    }
    if destination.exists():
        existing = _strict_json(destination)
        if existing != eligible:
            raise TabletopCollectionError(f"Static eligible output changed: {destination}")
        return
    _atomic_json(destination, eligible)


def _generator_record_paths(task: GenerationTask) -> tuple[list[Path], list[Path]]:
    admitted: list[Path] = []
    rejected: list[Path] = []
    for side in SIDES:
        admitted.extend(
            sorted((task.generator_output / f"{side}_single" / "raw").glob("*.json"))
        )
        rejected.extend(
            sorted(
                (
                    task.generator_output
                    / "diagnostic_table_rejected"
                    / f"{side}_single"
                    / "raw"
                ).glob("*.json")
            )
        )
    return admitted, rejected


def _route_task(task: GenerationTask) -> dict[str, Any]:
    admitted, rejected = _generator_record_paths(task)
    expected = 2 * task.num_grasps_per_side
    if len(admitted) + len(rejected) != expected:
        raise TabletopCollectionError(
            f"{task.spec.slug}/f{task.finger_count} produced "
            f"{len(admitted) + len(rejected)} records; expected {expected}"
        )
    eligible_count = 0
    static_failed_count = 0
    object_root = task.output_root / "objects" / task.spec.slug
    for source in admitted + rejected:
        payload = _strict_json(source)
        reasons = _static_failure_reasons(
            payload, spec=task.spec, finger_count=task.finger_count
        )
        if reasons:
            fr5_validation: dict[str, Any] = {
                "status": "NOT_EVALUATED",
                "gate": "FR5_X2_MOUNTED_TABLE_COLLISION_FREE",
                "failure_reasons": ["PRECEDING_STATIC_GATE_FAILED"],
                "authority": task.fr5_gate.record_authority(),
            }
        else:
            fr5_validation = task.fr5_gate.evaluate(
                payload,
                spec=task.spec,
                record_key=(
                    f"attempt={task.attempt_index}|object={task.spec.slug}|"
                    f"finger={task.finger_count}|source={source.name}"
                ),
            )
            if fr5_validation.get("status") != "PASS":
                gate_reasons = fr5_validation.get("failure_reasons")
                if isinstance(gate_reasons, list) and gate_reasons:
                    reasons.extend(str(value) for value in gate_reasons)
                else:
                    reasons.append("FR5_MOUNT_TABLE_GATE_FAILED")
                reasons = sorted(set(reasons))
        side = payload.get("active_side")
        side_label = str(side) if side in SIDES else "invalid_side"
        if reasons:
            destination = (
                object_root
                / "static_failed"
                / f"f{task.finger_count}"
                / side_label
                / source.name
            )
            _write_static_failure(
                source,
                payload,
                destination,
                reasons,
                fr5_validation,
            )
            static_failed_count += 1
            continue
        destination = (
            object_root
            / "physx_input"
            / f"f{task.finger_count}"
            / f"{side_label}_single"
            / "raw"
            / source.name
        )
        _write_static_eligible(
            source,
            payload,
            destination,
            fr5_validation,
        )
        eligible_count += 1
    return {
        "proposal_count": expected,
        "generator_admitted_count": len(admitted),
        "generator_table_rejected_count": len(rejected),
        "static_eligible_count": eligible_count,
        "static_failed_count": static_failed_count,
    }


def _generator_command(task: GenerationTask) -> list[str]:
    n_contact = max(4, task.finger_count)
    return [
        sys.executable,
        str(GENERATOR),
        "--mesh-path",
        str(task.spec.mesh_path),
        "--side",
        "both",
        "--finger-count",
        str(task.finger_count),
        "--n-contact",
        str(n_contact),
        "--num-grasps",
        str(task.num_grasps_per_side),
        "--batch-size",
        str(min(task.batch_size, task.num_grasps_per_side)),
        "--n-iterations",
        str(task.n_iterations),
        "--seed",
        str(task.seed),
        "--device",
        task.device,
        "--object-scale",
        repr(task.spec.scale),
        "--surface-samples",
        "512",
        "--table-plane-z-m",
        repr(task.spec.table_plane_z_m),
        "--table-clearance-m",
        repr(task.table_clearance_m),
        "--table-clearance-weight",
        "10000",
        "--prefer-distal-contacts",
        "--output",
        str(task.generator_output),
    ]


def _run_generation_task(task: GenerationTask) -> dict[str, Any]:
    task.task_root.mkdir(parents=True, exist_ok=True)
    if task.completion_path.is_file():
        completion = _strict_json(task.completion_path)
        if (
            completion.get("passed") is not True
            or completion.get("protocol_revision") != PROTOCOL_REVISION
            or completion.get("seed") != task.seed
            or completion.get("num_grasps_per_side") != task.num_grasps_per_side
        ):
            raise TabletopCollectionError(f"Stale generation proof: {task.completion_path}")
        _route_task(task)
        return completion

    admitted, rejected = _generator_record_paths(task)
    expected = 2 * task.num_grasps_per_side
    command = _generator_command(task)
    log_path = task.task_root / "generator.log"
    if len(admitted) + len(rejected) == 0:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if completed.returncode != 0:
            raise TabletopCollectionError(
                f"Generator failed for {task.spec.slug}/f{task.finger_count}; "
                f"see {log_path}"
            )
    elif len(admitted) + len(rejected) != expected:
        raise TabletopCollectionError(
            f"Incomplete unproven generator output at {task.generator_output}; "
            "preserved for inspection"
        )

    routing = _route_task(task)
    completion = {
        "passed": True,
        "protocol_revision": PROTOCOL_REVISION,
        "object": task.spec.as_dict(),
        "finger_count": task.finger_count,
        "num_grasps_per_side": task.num_grasps_per_side,
        "seed": task.seed,
        "command": command,
        "routing": routing,
    }
    _atomic_json(task.completion_path, completion)
    return completion


def _validator_command(
    *,
    spec: ObjectSpec,
    object_root: Path,
    args: argparse.Namespace,
    device: str,
    batch_size: int,
) -> list[str]:
    command = [
        sys.executable,
        str(VALIDATOR),
        "--input-root",
        str(object_root / "physx_input"),
        "--mesh-path",
        str(spec.mesh_path),
        "--side",
        "both",
        "--batch-size",
        str(batch_size),
        "--sim-steps",
        str(args.sim_steps),
        "--substeps",
        str(args.substeps),
        "--hand-friction",
        str(args.hand_friction),
        "--object-friction",
        str(args.object_friction),
        "--preclose-physics-steps",
        "0",
        "--criterion",
        "dexgraspnet-contact",
        "--closing-contact-threshold",
        str(args.closing_contact_threshold),
        "--closing-displacement",
        str(args.closing_displacement),
        "--closing-gradient-scale",
        str(args.closing_gradient_scale),
        "--closing-penetration-cap",
        str(args.closing_penetration_cap),
        "--collision-approximation",
        "convex-hull" if spec.kind == "primitive" else "convex-decomposition",
        "--device",
        device,
        "--summary-json",
        str(object_root / "physx_summary.json"),
        "--resume",
        "--headless",
    ]
    return command


def _validate_object(
    spec: ObjectSpec,
    attempt_root: Path,
    args: argparse.Namespace,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    object_root = attempt_root / "objects" / spec.slug
    input_root = object_root / "physx_input"
    raw_paths = sorted(input_root.glob("**/raw/*.json")) if input_root.is_dir() else []
    if not raw_paths:
        summary = {
            "passed": True,
            "skipped": True,
            "reason": "NO_STATIC_ELIGIBLE_ROWS",
            "candidate_count": 0,
            "valid_count": 0,
            "failed_count": 0,
        }
        _atomic_json(object_root / "physx_summary.json", summary)
        return summary
    command = _validator_command(
        spec=spec,
        object_root=object_root,
        args=args,
        device=device,
        batch_size=batch_size,
    )
    log_path = object_root / "physx.log"
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        raise TabletopCollectionError(
            f"PhysX validator failed for {spec.slug}; see {log_path}"
        )
    summary = _strict_json(object_root / "physx_summary.json")
    if summary.get("passed") is not True:
        raise TabletopCollectionError(f"Invalid PhysX summary for {spec.slug}")
    return summary


def _validated_paths(output_root: Path, directory: str) -> list[Path]:
    return sorted(
        path
        for path in (output_root / "attempts").glob(
            f"attempt_*/objects/*/physx_input/**/{directory}/*.json"
        )
        if path.is_file()
    )


def _valid_pool(
    output_root: Path, catalog: Sequence[ObjectSpec]
) -> list[tuple[Path, dict[str, Any], ObjectSpec]]:
    by_mesh = _object_by_mesh(catalog)
    pool: list[tuple[Path, dict[str, Any], ObjectSpec]] = []
    for path in _validated_paths(output_root, "valid"):
        payload = _strict_json(path)
        validation = payload.get("validation")
        object_record = payload.get("object")
        if (
            not isinstance(validation, Mapping)
            or validation.get("status") != "passed"
            or validation.get("backend") != "isaac_sim_physx"
            or validation.get("protocol_revision") != PHYSX_PROTOCOL
            or not isinstance(object_record, Mapping)
        ):
            raise TabletopCollectionError(f"Invalid PhysX success record: {path}")
        key = (
            Path(str(object_record["mesh_path"])).resolve(),
            float(object_record["scale"]),
        )
        spec = by_mesh.get(key)
        if spec is None:
            raise TabletopCollectionError(f"Validated record is outside catalog: {path}")
        table = payload.get("table_conditioning")
        contact = payload.get("selected_contact_realization")
        fr5_mount = payload.get("fr5_mount_table_validation")
        if (
            not isinstance(table, Mapping)
            or table.get("requested_clearance_met") is not True
            or table.get("source_plane_nonpenetrating") is not True
            or not isinstance(contact, Mapping)
            or contact.get("status") != "PASS"
            or not isinstance(fr5_mount, Mapping)
            or fr5_mount.get("status") != "PASS"
            or fr5_mount.get("gate") != "FR5_X2_MOUNTED_TABLE_COLLISION_FREE"
        ):
            raise TabletopCollectionError(
                f"PhysX success lost a tabletop or FR5-mount gate: {path}"
            )
        pool.append((path, payload, spec))
    return pool


def _pool_summary(
    pool: Sequence[tuple[Path, dict[str, Any], ObjectSpec]],
) -> dict[str, Any]:
    strata = Counter(
        (str(payload["active_side"]), _record_finger_count(payload))
        for _, payload, _ in pool
    )
    objects = Counter(spec.slug for _, _, spec in pool)
    general = {spec.slug for _, _, spec in pool if spec.kind == "general"}
    primitive_shapes = {spec.shape for _, _, spec in pool if spec.kind == "primitive"}
    return {
        "valid_count": len(pool),
        "side_finger_counts": {
            side: {str(finger): strata[(side, finger)] for finger in FINGER_COUNTS}
            for side in SIDES
        },
        "covered_object_count": len(objects),
        "covered_general_object_count": len(general),
        "covered_primitive_shapes": sorted(primitive_shapes),
        "object_counts": dict(sorted(objects.items())),
    }


def _failure_summary(output_root: Path) -> dict[str, Any]:
    raw = sorted((output_root / "attempts").glob("attempt_*/objects/*/generation/f*/generator_output/**/*_single/raw/*.json"))
    static_failed = sorted((output_root / "attempts").glob("attempt_*/objects/*/static_failed/**/*.json"))
    physx_failed = _validated_paths(output_root, "failed")
    return {
        "generator_proposal_count": len(raw),
        "static_failed_count": len(static_failed),
        "physx_failed_count": len(physx_failed),
        "static_failed_root": str((output_root / "attempts").resolve()),
        "physx_failed_root": str((output_root / "attempts").resolve()),
    }


def _completion_status(
    pool_summary: Mapping[str, Any], args: argparse.Namespace
) -> tuple[bool, dict[tuple[str, int], int]]:
    targets = _stratum_targets(args)
    deficits: dict[tuple[str, int], int] = {}
    counts = pool_summary["side_finger_counts"]
    for side in SIDES:
        for finger in _active_finger_counts(args):
            deficits[(side, finger)] = max(
                0,
                targets[(side, finger)] - int(counts[side][str(finger)]),
            )
    coverage_ok = (
        int(pool_summary["covered_object_count"]) >= args.minimum_object_coverage
        and int(pool_summary["covered_general_object_count"]) >= args.minimum_general_coverage
        and set(pool_summary["covered_primitive_shapes"])
        == {"sphere", "cylinder", "cuboid", "cube"}
    )
    return all(value == 0 for value in deficits.values()) and coverage_ok, deficits


def _next_attempt_index(output_root: Path) -> int:
    values: list[int] = []
    for path in (output_root / "attempts").glob("attempt_*"):
        try:
            values.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(values, default=-1) + 1


def _incomplete_attempt(output_root: Path) -> Path | None:
    incomplete = sorted(
        path
        for path in (output_root / "attempts").glob("attempt_*")
        if (path / "attempt.json").is_file()
        and not (path / "complete.json").is_file()
    )
    if len(incomplete) > 1:
        raise TabletopCollectionError(
            "More than one incomplete attempt exists: "
            + ", ".join(str(path) for path in incomplete)
        )
    return incomplete[0] if incomplete else None


def _planned_rows_per_object(
    deficits: Mapping[tuple[str, int], int],
    *,
    object_count: int,
    args: argparse.Namespace,
) -> dict[int, int]:
    rows: dict[int, int] = {}
    for finger in _active_finger_counts(args):
        largest_deficit = max(deficits[(side, finger)] for side in SIDES)
        if largest_deficit <= 0:
            continue
        estimate = math.ceil(
            largest_deficit / (object_count * args.expected_physx_pass_rate)
        )
        rows[finger] = min(
            args.maximum_raw_per_object_stratum,
            max(args.minimum_raw_per_object_stratum, estimate),
        )
    return rows


def _attempt_plan(
    *,
    index: int,
    attempt_root: Path,
    catalog: Sequence[ObjectSpec],
    deficits: Mapping[tuple[str, int], int],
    pool_summary: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = _planned_rows_per_object(deficits, object_count=len(catalog), args=args)
    if not rows:
        # Quotas may already be full while diversity coverage is still short.
        coverage_finger = _active_finger_counts(args)[
            len(_active_finger_counts(args)) // 2
        ]
        rows = {coverage_finger: args.minimum_raw_per_object_stratum}
    covered = set(pool_summary.get("object_counts", {}))
    need_coverage = (
        int(pool_summary["covered_object_count"]) < args.minimum_object_coverage
        or int(pool_summary["covered_general_object_count"]) < args.minimum_general_coverage
    )
    selected = (
        [spec for spec in catalog if spec.slug not in covered]
        if need_coverage and covered
        else list(catalog)
    )
    if not selected:
        selected = list(catalog)
    tasks = [
        {
            "object_slug": spec.slug,
            "finger_count": finger,
            "num_grasps_per_side": count,
        }
        for spec in selected
        for finger, count in sorted(rows.items())
    ]
    return {
        "passed": True,
        "protocol_revision": PROTOCOL_REVISION,
        "attempt_index": index,
        "attempt_root": str(attempt_root),
        "deficits_before": {
            f"{side}_f{finger}": value
            for (side, finger), value in sorted(deficits.items())
        },
        "tasks": tasks,
    }


def _run_attempt(
    *,
    attempt_root: Path,
    plan: Mapping[str, Any],
    catalog: Sequence[ObjectSpec],
    args: argparse.Namespace,
    fr5_gate: FR5TableCollisionGate,
    gpu_plan: GPUExecutionPlan,
) -> dict[str, Any]:
    by_slug = {spec.slug: (index, spec) for index, spec in enumerate(catalog)}
    tasks: list[GenerationTask] = []
    for task_index, row in enumerate(plan["tasks"]):
        object_index, spec = by_slug[row["object_slug"]]
        tasks.append(
            GenerationTask(
                attempt_index=int(plan["attempt_index"]),
                object_index=object_index,
                spec=spec,
                finger_count=int(row["finger_count"]),
                num_grasps_per_side=int(row["num_grasps_per_side"]),
                output_root=attempt_root,
                n_iterations=args.n_iterations,
                batch_size=gpu_plan.generation_batch_size,
                device=gpu_plan.generation_slots[
                    task_index % len(gpu_plan.generation_slots)
                ],
                table_clearance_m=args.table_clearance_m,
                fr5_gate=fr5_gate,
            )
        )
    generation_results: list[dict[str, Any]] = []
    generation_workers = len(gpu_plan.generation_slots)
    generation_iterator = iter(tasks)
    generation_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=generation_workers
    )
    generation_futures: dict[
        concurrent.futures.Future[dict[str, Any]], GenerationTask
    ] = {}
    try:
        for _ in range(generation_workers):
            task = next(generation_iterator, None)
            if task is not None:
                generation_futures[
                    generation_executor.submit(_run_generation_task, task)
                ] = task
        while generation_futures:
            done, _ = concurrent.futures.wait(
                generation_futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            completed_count = 0
            for future in done:
                task = generation_futures.pop(future)
                result = future.result()
                generation_results.append(result)
                completed_count += 1
                routing = result["routing"]
                print(
                    f"[generation] {task.spec.slug}/f{task.finger_count}: "
                    f"device={task.device} "
                    f"eligible={routing['static_eligible_count']} "
                    f"static_failed={routing['static_failed_count']}",
                    flush=True,
                )
            for _ in range(completed_count):
                task = next(generation_iterator, None)
                if task is not None:
                    generation_futures[
                        generation_executor.submit(_run_generation_task, task)
                    ] = task
    except BaseException:
        for future in generation_futures:
            future.cancel()
        generation_executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        generation_executor.shutdown(wait=True)

    validation_results: dict[str, dict[str, Any]] = {}
    generated_slugs = sorted({task.spec.slug for task in tasks})
    validation_work = [
        (
            slug,
            by_slug[slug][1],
            gpu_plan.validation_slots[index % len(gpu_plan.validation_slots)],
        )
        for index, slug in enumerate(generated_slugs)
    ]
    validation_workers = len(gpu_plan.validation_slots)
    validation_iterator = iter(validation_work)
    validation_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=validation_workers
    )
    validation_futures: dict[
        concurrent.futures.Future[dict[str, Any]], str
    ] = {}
    try:
        for _ in range(validation_workers):
            work = next(validation_iterator, None)
            if work is None:
                continue
            slug, spec, device = work
            print(
                f"[physx] scheduling {slug} device={device} "
                f"batch={gpu_plan.validation_batch_size}",
                flush=True,
            )
            future = validation_executor.submit(
                _validate_object,
                spec,
                attempt_root,
                args,
                device,
                gpu_plan.validation_batch_size,
            )
            validation_futures[future] = slug
        while validation_futures:
            done, _ = concurrent.futures.wait(
                validation_futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            completed_count = 0
            for future in done:
                slug = validation_futures.pop(future)
                validation_results[slug] = future.result()
                completed_count += 1
                print(f"[physx] completed {slug}", flush=True)
            for _ in range(completed_count):
                work = next(validation_iterator, None)
                if work is None:
                    continue
                slug, spec, device = work
                print(
                    f"[physx] scheduling {slug} device={device} "
                    f"batch={gpu_plan.validation_batch_size}",
                    flush=True,
                )
                future = validation_executor.submit(
                    _validate_object,
                    spec,
                    attempt_root,
                    args,
                    device,
                    gpu_plan.validation_batch_size,
                )
                validation_futures[future] = slug
    except BaseException:
        for future in validation_futures:
            future.cancel()
        validation_executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        validation_executor.shutdown(wait=True)

    valid_count = len(
        sorted(attempt_root.glob("objects/*/physx_input/**/valid/*.json"))
    )
    failed_count = len(
        sorted(attempt_root.glob("objects/*/physx_input/**/failed/*.json"))
    )
    static_failed_count = len(sorted(attempt_root.glob("objects/*/static_failed/**/*.json")))
    complete = {
        "passed": True,
        "protocol_revision": PROTOCOL_REVISION,
        "attempt_index": int(plan["attempt_index"]),
        "generation_task_count": len(tasks),
        "generator_proposal_count": sum(
            int(result["routing"]["proposal_count"])
            for result in generation_results
        ),
        "static_failed_count": static_failed_count,
        "physx_valid_count": valid_count,
        "physx_failed_count": failed_count,
        "gpu_execution_plan": gpu_plan.as_dict(),
        "validation_results": validation_results,
    }
    _atomic_json(attempt_root / "complete.json", complete)
    return complete


def _round_robin_select(
    values: Sequence[tuple[Path, dict[str, Any], ObjectSpec]], target: int
) -> list[tuple[Path, dict[str, Any], ObjectSpec]]:
    grouped: dict[str, list[tuple[Path, dict[str, Any], ObjectSpec]]] = defaultdict(list)
    for value in sorted(values, key=lambda row: str(row[0])):
        grouped[value[2].slug].append(value)
    selected: list[tuple[Path, dict[str, Any], ObjectSpec]] = []
    depth = 0
    object_ids = sorted(grouped)
    while len(selected) < target:
        added = False
        for object_id in object_ids:
            rows = grouped[object_id]
            if depth < len(rows):
                selected.append(rows[depth])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != target:
        raise TabletopCollectionError(
            f"Round-robin selection found {len(selected)} rows; need {target}"
        )
    return selected


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _publish_final(
    *,
    output_root: Path,
    pool: Sequence[tuple[Path, dict[str, Any], ObjectSpec]],
    pool_summary: Mapping[str, Any],
    catalog: Sequence[ObjectSpec],
    settings: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    targets = _stratum_targets(args)
    by_stratum: dict[
        tuple[str, int], list[tuple[Path, dict[str, Any], ObjectSpec]]
    ] = defaultdict(list)
    for value in pool:
        _, payload, _ = value
        by_stratum[(str(payload["active_side"]), _record_finger_count(payload))].append(value)

    selected: list[tuple[Path, dict[str, Any], ObjectSpec]] = []
    for side in SIDES:
        for finger in _active_finger_counts(args):
            selected.extend(
                _round_robin_select(
                    by_stratum[(side, finger)], targets[(side, finger)]
                )
            )
    if len(selected) != args.target_total:
        raise TabletopCollectionError("Final selection count changed")

    final_root = output_root / "final"
    if final_root.exists():
        existing_manifest = final_root / "manifest.json"
        if existing_manifest.is_file():
            manifest = _strict_json(existing_manifest)
            if manifest.get("passed") is True and manifest.get("valid_count") == args.target_total:
                return manifest
        raise TabletopCollectionError(f"Unproven final directory exists: {final_root}")

    staging = Path(tempfile.mkdtemp(prefix=".final_staging_", dir=output_root))
    records: list[dict[str, Any]] = []
    counters: Counter[tuple[str, int]] = Counter()
    try:
        for source, payload, spec in selected:
            side = str(payload["active_side"])
            finger = _record_finger_count(payload)
            index = counters[(side, finger)]
            counters[(side, finger)] += 1
            destination = (
                staging
                / side
                / f"f{finger}"
                / f"x2_{spec.slug}_{side}_f{finger}_{index:06d}.json"
            )
            _link_or_copy(source, destination)
            records.append(
                {
                    "side": side,
                    "finger_count": finger,
                    "finger_names": payload["finger_participation"]["finger_names"],
                    "object_slug": spec.slug,
                    "source": str(source),
                    "source_sha256": _sha256(source),
                    "output_relative": str(destination.relative_to(staging)),
                }
            )
        selected_objects = Counter(row["object_slug"] for row in records)
        failure = _failure_summary(output_root)
        manifest = {
            "schema_version": 1,
            "protocol_revision": PROTOCOL_REVISION,
            "passed": True,
            "valid_count": args.target_total,
            "side_finger_counts": {
                side: {
                    str(finger): targets[(side, finger)]
                    for finger in _active_finger_counts(args)
                }
                for side in SIDES
            },
            "covered_object_count": len(selected_objects),
            "covered_general_object_count": len(
                {
                    slug
                    for slug in selected_objects
                    if slug.startswith("general_")
                }
            ),
            "selected_object_counts": dict(sorted(selected_objects.items())),
            "candidate_pool": dict(pool_summary),
            "failures": failure,
            "failure_records_preserved": True,
            "settings": dict(settings),
            "catalog": [spec.as_dict() for spec in catalog],
            "physical_evidence": {
                "backend": "isaac_sim_physx",
                "protocol_revision": PHYSX_PROTOCOL,
                "required_orientation_count": 6,
                "all_final_records_passed": True,
                "table_collider_present": False,
                "tabletop_static_clearance_required": True,
                "mounted_fr5_x2_ik_required": True,
                "fr5_moving_link_table_clearance_required": True,
                "mounted_x2_ik_error_bounded_clearance_required": True,
                "not_a_table_acquisition_or_lift_proof": True,
            },
            "records": records,
        }
        _atomic_json(staging / "manifest.json", manifest)
        staging.replace(final_root)
        _copy_verified(final_root / "manifest.json", output_root / "manifest.json")
        return manifest
    except Exception:
        # Preserve staging for forensic recovery; never erase generated evidence.
        raise


def _progress_payload(
    *,
    output_root: Path,
    pool: Sequence[tuple[Path, dict[str, Any], ObjectSpec]],
    pool_summary: Mapping[str, Any],
    deficits: Mapping[tuple[str, int], int],
) -> dict[str, Any]:
    return {
        "passed": True,
        "protocol_revision": PROTOCOL_REVISION,
        "candidate_pool": dict(pool_summary),
        "deficits": {
            f"{side}_f{finger}": value
            for (side, finger), value in sorted(deficits.items())
        },
        "failures": _failure_summary(output_root),
        "physx_valid_pool_count": len(pool),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-total", type=_positive_int, default=10000)
    parser.add_argument(
        "--finger-counts",
        type=int,
        choices=FINGER_COUNTS,
        nargs="+",
        default=list(FINGER_COUNTS),
        metavar="N",
        help=(
            "Participating-finger strata to collect, in deterministic quota "
            "priority order (default: 1 2 3 4 5)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "x2_tabletop_physx_10000",
    )
    parser.add_argument("--n-iterations", type=_positive_int, default=6000)
    gpu_mode = parser.add_mutually_exclusive_group()
    gpu_mode.add_argument(
        "--auto-gpu",
        dest="auto_gpu",
        action="store_true",
        default=True,
        help=(
            "Auto-detect NVIDIA GPUs and size generation/PhysX concurrency; "
            "this is the default."
        ),
    )
    gpu_mode.add_argument(
        "--manual-gpu",
        dest="auto_gpu",
        action="store_false",
        help="Use --generation-device/--jobs and validation device/job settings.",
    )
    parser.add_argument(
        "--gpu-indices",
        type=_nonnegative_int,
        nargs="+",
        help="Optional physical GPU indices used by automatic scheduling.",
    )
    parser.add_argument(
        "--generation-device",
        default="cuda",
        help="Generation device in --manual-gpu mode.",
    )
    parser.add_argument("--generation-batch-size", type=_positive_int, default=32)
    parser.add_argument(
        "--jobs",
        type=_positive_int,
        default=2,
        help="Generation worker count in --manual-gpu mode.",
    )
    parser.add_argument(
        "--validation-device",
        default="cuda:0",
        help="PhysX device in --manual-gpu mode.",
    )
    parser.add_argument(
        "--validation-jobs",
        type=_positive_int,
        default=1,
        help="Concurrent PhysX object validators in --manual-gpu mode.",
    )
    parser.add_argument("--validation-batch-size", type=_positive_int, default=8)
    parser.add_argument("--sim-steps", type=_positive_int, default=100)
    parser.add_argument("--substeps", type=_positive_int, default=2)
    parser.add_argument(
        "--hand-friction",
        type=_nonnegative_float,
        default=1.0,
        help="PhysX static and dynamic friction assigned to every X2 link.",
    )
    parser.add_argument(
        "--object-friction",
        type=_nonnegative_float,
        default=1.0,
        help="PhysX static and dynamic friction assigned to the grasped object.",
    )
    parser.add_argument(
        "--closing-contact-threshold",
        type=_positive_float,
        default=0.003,
        help="Near-surface range used by the PhysX contact-gradient optimizer.",
    )
    parser.add_argument(
        "--closing-displacement",
        type=_positive_float,
        default=0.002,
        help=(
            "Maximum contact-gradient closing proposal; the line search backs "
            "off until the penetration cap is respected."
        ),
    )
    parser.add_argument(
        "--closing-gradient-scale",
        type=_positive_float,
        default=100.0,
    )
    parser.add_argument(
        "--closing-penetration-cap",
        type=_positive_float,
        default=0.0015,
        help="Hard bidirectional penetration cap for an optimized closing target.",
    )
    parser.add_argument("--table-clearance-m", type=_positive_float, default=0.008)
    parser.add_argument("--fr5-vendor-root", type=Path, default=DEFAULT_FR5_VENDOR_ROOT)
    parser.add_argument("--fr5-home-json", type=Path, default=DEFAULT_FR5_HOME)
    parser.add_argument("--fr5-mount-json", type=Path, default=DEFAULT_FR5_MOUNT)
    parser.add_argument(
        "--x2-workspace-bounds-json",
        type=Path,
        default=DEFAULT_X2_WORKSPACE_BOUNDS,
    )
    parser.add_argument(
        "--fr5-object-table-xy-m",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("X", "Y"),
        help="World tabletop XY used to mount each generated object for the FR5 gate.",
    )
    parser.add_argument(
        "--minimum-x2-root-table-distance-m",
        type=_positive_float,
        default=0.05,
        help="Reject an X2 root target closer than this to the tabletop plane.",
    )
    parser.add_argument(
        "--robot-table-clearance-m",
        type=_positive_float,
        default=0.005,
        help=(
            "Required clearance for every moving FR5 collision mesh and the "
            "IK-error-bounded mounted X2 collision geometry."
        ),
    )
    parser.add_argument(
        "--fr5-ik-seed-count",
        type=_positive_int,
        default=8,
        help="Deterministic FR5 IK starts; at least three are required.",
    )
    parser.add_argument(
        "--general-target-max-extent-m", type=_positive_float, default=0.09
    )
    parser.add_argument(
        "--expected-physx-pass-rate", type=_positive_float, default=0.10
    )
    parser.add_argument(
        "--minimum-raw-per-object-stratum", type=_positive_int, default=8
    )
    parser.add_argument(
        "--maximum-raw-per-object-stratum", type=_positive_int, default=32
    )
    parser.add_argument("--minimum-object-coverage", type=_positive_int, default=30)
    parser.add_argument("--minimum-general-coverage", type=_nonnegative_int, default=18)
    parser.add_argument(
        "--max-attempts",
        type=_nonnegative_int,
        default=0,
        help="Stop after this many newly completed attempts; zero means no limit.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.finger_counts)) != len(args.finger_counts):
        parser.error("finger-counts must not contain duplicates")
    if args.target_total % len(SIDES) != 0:
        parser.error("target-total must be even so front/back quotas are identical")
    if args.target_total < len(SIDES) * len(args.finger_counts):
        parser.error("target-total is too small to allocate every requested stratum")
    if args.expected_physx_pass_rate > 1.0:
        parser.error("expected-physx-pass-rate must be at most 1")
    if args.closing_penetration_cap > 0.002:
        parser.error("closing-penetration-cap must be at most 0.002 m")
    if args.minimum_raw_per_object_stratum > args.maximum_raw_per_object_stratum:
        parser.error("minimum raw count cannot exceed maximum raw count")
    if args.minimum_object_coverage > 42:
        parser.error("minimum-object-coverage cannot exceed the 42-object catalog")
    if args.minimum_general_coverage > len(FORMAL_GENERAL_IDS):
        parser.error("minimum-general-coverage cannot exceed 30")
    if args.fr5_ik_seed_count < 3:
        parser.error("fr5-ik-seed-count must be at least 3")
    if not all(math.isfinite(value) for value in args.fr5_object_table_xy_m):
        parser.error("fr5-object-table-xy-m must be finite")
    if args.gpu_indices is not None and len(set(args.gpu_indices)) != len(
        args.gpu_indices
    ):
        parser.error("gpu-indices must not contain duplicates")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.expanduser().resolve()
    catalog = build_catalog(args.general_target_max_extent_m)
    gpu_plan = _resolve_gpu_execution_plan(args)
    fr5_gate = FR5TableCollisionGate(
        vendor_root=args.fr5_vendor_root,
        home_path=args.fr5_home_json,
        mount_path=args.fr5_mount_json,
        x2_workspace_bounds_path=args.x2_workspace_bounds_json,
        object_table_xy_m=args.fr5_object_table_xy_m,
        minimum_x2_root_table_distance_m=args.minimum_x2_root_table_distance_m,
        robot_table_clearance_m=args.robot_table_clearance_m,
        ik_seed_count=args.fr5_ik_seed_count,
    )
    required_generation_clearance = (
        args.robot_table_clearance_m
        + FR5_IK_POSITION_TOLERANCE_M
        + 2.0
        * fr5_gate.x2_root_subtree_radius_m
        * math.sin(0.5 * FR5_IK_ROTATION_TOLERANCE_RAD)
    )
    if args.table_clearance_m < required_generation_clearance:
        raise TabletopCollectionError(
            "table-clearance-m is too small for the mounted-X2 clearance and "
            "FR5 IK error bound; increase it to at least "
            f"{required_generation_clearance:.6f}"
        )
    settings = _settings(args, catalog, fr5_gate)
    pool = _valid_pool(output_root, catalog) if output_root.exists() else []
    pool_summary = _pool_summary(pool)
    complete, deficits = _completion_status(pool_summary, args)

    if args.dry_run:
        next_rows = _planned_rows_per_object(deficits, object_count=len(catalog), args=args)
        result = {
            "passed": True,
            "dry_run": True,
            "would_launch_generator": False,
            "would_launch_isaac_sim": False,
            "settings": settings,
            "gpu_execution_plan": gpu_plan.as_dict(),
            "current_pool": pool_summary,
            "deficits": {
                f"{side}_f{finger}": value
                for (side, finger), value in sorted(deficits.items())
            },
            "next_attempt_rows_per_object_per_side": next_rows,
        }
        print(json.dumps(result, indent=2, allow_nan=False))
        return result

    output_root.mkdir(parents=True, exist_ok=True)
    existing_entries = [path for path in output_root.iterdir() if path.name != ".collector.lock"]
    if existing_entries and not args.resume:
        raise TabletopCollectionError(
            f"Output root is not empty; pass --resume: {output_root}"
        )
    lock_path = output_root / ".collector.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TabletopCollectionError(
                f"Another tabletop collector owns {lock_path}"
            ) from exc

        settings_path = output_root / "settings.json"
        if settings_path.is_file():
            if _strict_json(settings_path) != settings:
                raise TabletopCollectionError(
                    "Resume settings differ from the existing campaign"
                )
        else:
            _atomic_json(settings_path, settings)
        _atomic_json(output_root / "runtime_gpu_plan.json", gpu_plan.as_dict())
        print(
            "[gpu] "
            f"mode={gpu_plan.mode} "
            f"generation_workers={len(gpu_plan.generation_slots)} "
            f"generation_batch={gpu_plan.generation_batch_size} "
            f"validation_workers={len(gpu_plan.validation_slots)} "
            f"validation_batch={gpu_plan.validation_batch_size} "
            f"devices={sorted(set(gpu_plan.generation_slots + gpu_plan.validation_slots))}",
            flush=True,
        )

        completed_new_attempts = 0
        while not complete:
            if args.max_attempts and completed_new_attempts >= args.max_attempts:
                progress = _progress_payload(
                    output_root=output_root,
                    pool=pool,
                    pool_summary=pool_summary,
                    deficits=deficits,
                )
                _atomic_json(output_root / "progress.json", progress)
                print(json.dumps(progress, indent=2, allow_nan=False))
                return progress

            attempt_root = _incomplete_attempt(output_root)
            if attempt_root is None:
                index = _next_attempt_index(output_root)
                attempt_root = output_root / "attempts" / f"attempt_{index:04d}"
                attempt_root.mkdir(parents=True, exist_ok=False)
                plan = _attempt_plan(
                    index=index,
                    attempt_root=attempt_root,
                    catalog=catalog,
                    deficits=deficits,
                    pool_summary=pool_summary,
                    args=args,
                )
                _atomic_json(attempt_root / "attempt.json", plan)
            else:
                plan = _strict_json(attempt_root / "attempt.json")
                index = int(plan.get("attempt_index", -1))
                if (
                    plan.get("protocol_revision") != PROTOCOL_REVISION
                    or index < 0
                    or attempt_root.name != f"attempt_{index:04d}"
                    or Path(str(plan.get("attempt_root", ""))).resolve()
                    != attempt_root.resolve()
                ):
                    raise TabletopCollectionError(
                        f"Invalid resumable attempt plan: {attempt_root / 'attempt.json'}"
                    )
                print(f"[attempt {index:04d}] resuming", flush=True)
            print(
                f"[attempt {index:04d}] tasks={len(plan['tasks'])} "
                f"pool={len(pool)}/{args.target_total}",
                flush=True,
            )
            _run_attempt(
                attempt_root=attempt_root,
                plan=plan,
                catalog=catalog,
                args=args,
                fr5_gate=fr5_gate,
                gpu_plan=gpu_plan,
            )
            completed_new_attempts += 1
            pool = _valid_pool(output_root, catalog)
            pool_summary = _pool_summary(pool)
            complete, deficits = _completion_status(pool_summary, args)
            progress = _progress_payload(
                output_root=output_root,
                pool=pool,
                pool_summary=pool_summary,
                deficits=deficits,
            )
            _atomic_json(output_root / "progress.json", progress)
            print(
                f"[attempt {index:04d}] physx_valid_pool={len(pool)} "
                f"remaining={sum(deficits.values())}",
                flush=True,
            )

        manifest = _publish_final(
            output_root=output_root,
            pool=pool,
            pool_summary=pool_summary,
            catalog=catalog,
            settings=settings,
            args=args,
        )
        print(json.dumps(manifest, indent=2, allow_nan=False))
        return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run(args)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
