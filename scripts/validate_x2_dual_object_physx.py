#!/usr/bin/env python3
"""Jointly validate composed X2 two-object candidates in Isaac Sim/PhysX.

Both objects are spawned as independent dynamic rigid bodies in a shared-hand
scene.  The historical protocol disables object-object collision after a dense
static overlap gate.  The V1 protocol can opt into physical object-object
collision with ``--enable-object-object-collisions``; this flag is deliberately
opt-in so historical outputs remain reproducible and are never silently
relabelled.  A candidate passes only when both objects retain contact in all six
gravity directions while the hand remains finite, tracks its composed actuator
target, and satisfies the X2 Newton-mimic constraints.

The top-level invocation groups candidates by their exact pair of object
assets and launches one isolated Isaac worker per group.  Outputs are written
atomically below ``physx_validation/{valid,failed}``; source candidates are
never modified.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_generation.x2_dual_object_validation import (  # noqa: E402
    DUAL_VALIDATION_BACKEND,
    DUAL_VALIDATION_CRITERION,
    DUAL_VALIDATION_PROTOCOL_REVISION,
    DUAL_VALIDATION_PROTOCOL_REVISION_OBJECT_COLLISION,
    EXPECTED_GRAVITY_NAMES,
    X2DualObjectCandidate,
    X2DualValidationError,
    atomic_json,
    discover_candidates,
    existing_validation_output,
    file_sha256,
    gravity_vectors_shared_hand,
    group_candidates_by_objects,
    make_validation_record,
    strict_json,
    write_validation_record,
)
from grasp_generation.x2_isaac_validation import (  # noqa: E402
    EXPECTED_ACTUATOR_NAMES,
    EXPECTED_JOINT_NAMES,
    FORMAL_ACTUATOR_ARMATURE,
    FORMAL_ACTUATOR_DAMPING,
    FORMAL_ACTUATOR_STIFFNESS,
    GRAVITY_TESTS_WXYZ,
    PASSIVE_MIMIC_DRIVERS,
)
from isaaclab.app import AppLauncher  # noqa: E402
from scripts.validate_x2_mesh_grasps_physx import (  # noqa: E402
    ACTUATOR_GROUP_EXPRS,
    DEFAULT_HAND_USD,
    DEFAULT_USD_CACHE,
    MAX_CONTACT_DATA_PER_OBJECT,
    _as_torch,
    _audit_converted_bounds,
    _audit_runtime_mapping,
    _convert_object_mesh,
    _read_raw_object_contacts,
)


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "x2_dual_object"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATASET_ROOT / "physx_validation"
# The joint-composition preflight is a sampled hull gate.  It is intentionally
# followed by full PhysX rather than presented as a continuous collision
# certificate.  The source records have already passed the denser single-object
# 8192/13056-point gate; this recomputation targets collisions introduced by
# mixing the two finger-owned qposes.
OBJECT_SURFACE_SAMPLES = 2048
HAND_OBJECT_PENETRATION_THRESHOLD = 0.001
OBJECT_OBJECT_PENETRATION_THRESHOLD = 0.0005


def _validation_protocol_revision(args: argparse.Namespace) -> str:
    return (
        DUAL_VALIDATION_PROTOCOL_REVISION_OBJECT_COLLISION
        if args.enable_object_object_collisions
        else DUAL_VALIDATION_PROTOCOL_REVISION
    )


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
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument("--sim-steps", type=_positive_int, default=100)
    parser.add_argument("--substeps", type=_positive_int, default=2)
    parser.add_argument("--dt", type=_positive_float, default=1.0 / 60.0)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument(
        "--combination",
        choices=("right_f1_left_f4", "right_f2_left_f3", "right_f3_left_f2", "right_f4_left_f1"),
        help="Select one exact complementary finger-count partition before applying --limit.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Pilot-only: do not require four completed 500-pair strata.",
    )
    parser.add_argument("--hand-usd", type=Path, default=DEFAULT_HAND_USD)
    parser.add_argument("--usd-cache", type=Path, default=DEFAULT_USD_CACHE)
    parser.add_argument("--density", type=_positive_float, default=500.0)
    parser.add_argument("--hand-friction", type=_nonnegative_float, default=3.0)
    parser.add_argument("--object-friction", type=_nonnegative_float, default=3.0)
    parser.add_argument("--contact-offset", type=_nonnegative_float, default=0.001)
    parser.add_argument("--rest-offset", type=_nonnegative_float, default=0.0)
    parser.add_argument(
        "--retention-distance", type=_positive_float, default=0.1
    )
    parser.add_argument(
        "--joint-error-threshold", type=_positive_float, default=0.1
    )
    parser.add_argument(
        "--mimic-error-threshold", type=_positive_float, default=0.01
    )
    parser.add_argument(
        "--hand-object-penetration-threshold",
        type=_positive_float,
        default=HAND_OBJECT_PENETRATION_THRESHOLD,
    )
    parser.add_argument(
        "--object-object-penetration-threshold",
        type=_positive_float,
        default=OBJECT_OBJECT_PENETRATION_THRESHOLD,
    )
    parser.add_argument(
        "--enable-object-object-collisions",
        action="store_true",
        help=(
            "V1 protocol: retain dynamic object-object collision. Historical "
            "runs keep collision filtering unless this flag is present."
        ),
    )
    parser.add_argument(
        "--actuator-stiffness",
        type=_nonnegative_float,
        default=FORMAL_ACTUATOR_STIFFNESS,
    )
    parser.add_argument(
        "--actuator-damping",
        type=_nonnegative_float,
        default=FORMAL_ACTUATOR_DAMPING,
    )
    parser.add_argument(
        "--actuator-armature",
        type=_nonnegative_float,
        default=FORMAL_ACTUATOR_ARMATURE,
    )
    parser.add_argument("--solver-type", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--external-forces-every-iteration",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--solve-articulation-contact-last", action="store_true")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--worker-selection", type=Path, help=argparse.SUPPRESS)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _resolved_file(path: Path, label: str) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise X2DualValidationError(f"{label} does not exist: {resolved}")
    return resolved


def _matrix_to_quaternion_xyzw(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            ) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            ) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            ) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray((x, y, z, w), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def _rotation6d_identity() -> list[float]:
    return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def _mesh_static_data(
    mesh_path: Path,
    scale: float,
    cache: dict[tuple[Path, float], dict[str, Any]],
) -> dict[str, Any]:
    key = (mesh_path, scale)
    cached = cache.get(key)
    if cached is not None:
        return cached
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not mesh.is_watertight:
        raise X2DualValidationError(
            f"dual static gate requires one watertight mesh: {mesh_path}"
        )
    mesh = mesh.copy()
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
    seed = int(file_sha256(mesh_path)[:8], 16)
    surface, _ = trimesh.sample.sample_surface(
        mesh, OBJECT_SURFACE_SAMPLES, seed=seed
    )
    cached = {
        "mesh": mesh,
        "query": trimesh.proximity.ProximityQuery(mesh),
        "surface": np.asarray(surface, dtype=np.float64),
    }
    cache[key] = cached
    return cached


def _signed_max(query: Any, points: np.ndarray) -> float:
    values = np.asarray(query.signed_distance(points), dtype=np.float64)
    if values.shape != (len(points),) or not np.isfinite(values).all():
        raise X2DualValidationError("static signed-distance query is non-finite")
    return max(float(values.max(initial=-math.inf)), 0.0)


def _static_preflight_group(
    candidates: Sequence[X2DualObjectCandidate],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    import torch

    from grasp_generation.utils.x2_config import load_x2_mesh_config
    from grasp_generation.utils.x2_hand_model import X2HandModel
    from grasp_generation.utils.x2_mesh_contacts import (
        load_generic_contact_candidates,
    )

    if not candidates:
        return []
    config = load_x2_mesh_config()
    contact_candidates = load_generic_contact_candidates(
        config.configured_path("contact_candidates.path", must_exist=True)
    )
    hand = X2HandModel(
        config,
        contact_candidates,
        device="cpu",
        dtype=torch.float64,
        collision_samples_per_link=int(
            config.require("generation.hand_collision_samples_per_link")
        ),
        self_collision_samples_per_link=int(
            config.require("self_collision.surface_samples_per_link")
        ),
    )
    actuator = torch.as_tensor(
        [candidate.record["hand"]["actuator"] for candidate in candidates],
        dtype=torch.float64,
    )
    identity = torch.as_tensor(
        _rotation6d_identity(), dtype=torch.float64
    ).expand(len(candidates), -1)
    pose = torch.cat(
        (
            torch.zeros((len(candidates), 3), dtype=torch.float64),
            identity,
            actuator,
        ),
        dim=1,
    )
    hand.set_parameters(pose)
    self_collision = hand.self_collision_diagnostics()
    hand_points = hand.collision_points_world().detach().cpu().numpy()
    mesh_cache: dict[tuple[Path, float], dict[str, Any]] = {}
    reports: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        branch_data: list[dict[str, Any]] = []
        object_world_surfaces: list[np.ndarray] = []
        object_queries: list[Any] = []
        for branch in (candidate.right, candidate.left):
            mesh_path = Path(branch["mesh_path"])
            mesh_data = _mesh_static_data(
                mesh_path, float(branch["scale"]), mesh_cache
            )
            pose_record = branch["pose_in_shared_hand_frame"]
            rotation = np.asarray(
                pose_record["rotation_matrix"], dtype=np.float64
            )
            translation = np.asarray(
                pose_record["translation"], dtype=np.float64
            )
            hand_in_object = (hand_points[index] - translation) @ rotation
            forward = _signed_max(mesh_data["query"], hand_in_object)
            object_world = mesh_data["surface"] @ rotation.T + translation
            reverse_tensor = hand.cal_distance(
                torch.as_tensor(
                    object_world[None, :, :], dtype=torch.float64
                ),
                row_indices=torch.as_tensor([index], dtype=torch.long),
            )
            reverse = max(float(reverse_tensor.max().item()), 0.0)
            maximum = max(forward, reverse)
            branch_data.append(
                {
                    "object_id": branch["object_id"],
                    "forward_maximum_penetration_m": forward,
                    "reverse_maximum_penetration_m": reverse,
                    "maximum_penetration_m": maximum,
                    "passed": maximum
                    < args.hand_object_penetration_threshold,
                }
            )
            object_world_surfaces.append(object_world)
            object_queries.append((mesh_data["query"], rotation, translation))

        right_query, right_rotation, right_translation = object_queries[0]
        left_query, left_rotation, left_translation = object_queries[1]
        right_in_left = (
            object_world_surfaces[0] - left_translation
        ) @ left_rotation
        left_in_right = (
            object_world_surfaces[1] - right_translation
        ) @ right_rotation
        object_object_forward = _signed_max(left_query, right_in_left)
        object_object_reverse = _signed_max(right_query, left_in_right)
        object_object_maximum = max(
            object_object_forward, object_object_reverse
        )
        self_maximum = float(
            self_collision.maximum_penetration[index].item()
        )
        self_passed = bool(self_collision.feasible[index].item())
        object_object_passed = (
            object_object_maximum
            < args.object_object_penetration_threshold
        )
        passed = (
            self_passed
            and object_object_passed
            and all(value["passed"] for value in branch_data)
        )
        failure_reasons: list[str] = []
        if not self_passed:
            failure_reasons.append("combined_hand_self_collision")
        for slot, value in zip(("right", "left"), branch_data):
            if not value["passed"]:
                failure_reasons.append(
                    f"{slot}_hand_object_initial_penetration"
                )
        if not object_object_passed:
            failure_reasons.append("object_object_initial_penetration")
        reports.append(
            {
                "passed": passed,
                "hand_self_collision": {
                    "maximum_penetration_m": self_maximum,
                    "threshold_m": float(self_collision.threshold),
                    "passed": self_passed,
                },
                "hand_object": {
                    "threshold_m": args.hand_object_penetration_threshold,
                    "objects": branch_data,
                    "passed": all(value["passed"] for value in branch_data),
                },
                "object_object": {
                    "forward_maximum_penetration_m": object_object_forward,
                    "reverse_maximum_penetration_m": object_object_reverse,
                    "maximum_penetration_m": object_object_maximum,
                    "threshold_m": args.object_object_penetration_threshold,
                    "passed": object_object_passed,
                },
                "failure_reasons": failure_reasons,
                "surface_samples_per_object": OBJECT_SURFACE_SAMPLES,
            }
        )
    return reports


def _make_dual_scene_cfg(
    *,
    hand_usd: Path,
    right_object_usd: Path,
    left_object_usd: Path,
    num_envs: int,
    args: argparse.Namespace,
):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils.configclass import configclass

    hand_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=args.hand_friction,
        dynamic_friction=args.hand_friction,
        restitution=0.0,
    )
    robot_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(hand_usd), physics_material=hand_material
        ),
        actuators={
            group_name: ImplicitActuatorCfg(
                joint_names_expr=expressions,
                stiffness=args.actuator_stiffness,
                damping=args.actuator_damping,
                armature=args.actuator_armature,
                # Keep the historical behaviour when a caller does not
                # provide an explicit contract.  The sequential X2 worker
                # does provide one: its canonical URDF specifies 10 Nm for
                # every active joint, whereas the imported USD currently
                # exposes a 1 Nm DriveAPI default.  Leaving this ``None`` in
                # that worker silently selects the latter and saturates the
                # close controller.
                effort_limit_sim=getattr(args, "actuator_effort_limit", None),
                velocity_limit_sim=None,
            )
            for group_name, expressions in ACTUATOR_GROUP_EXPRS.items()
        },
    )

    def object_cfg(name: str, usd: Path):
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd), activate_contact_sensors=True
            ),
        )

    def contact_cfg(name: str):
        return ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            update_period=0.0,
            history_length=1,
            track_pose=False,
            track_contact_points=False,
            track_air_time=False,
            filter_prim_paths_expr=[],
            max_contact_data_count_per_prim=MAX_CONTACT_DATA_PER_OBJECT,
        )

    # The generic batch validator is deliberately free-space by default.  The
    # sequential RetainPlan-X2 worker opts in to this fixture after it has
    # validated the independent O1/O2 support manifests.  Keep the table in
    # the environment namespace so the raw object contact view can name it
    # unambiguously as ``.../Environment/Table``.
    table_cfg = None
    front_pedestal_cfg = None
    back_pedestal_cfg = None
    if bool(getattr(args, "table_supported_scene", False)):
        table_top_z_m = float(getattr(args, "table_top_z_m"))
        table_size_xy_m = float(getattr(args, "table_size_xy_m", 1.5))
        table_thickness_m = float(getattr(args, "table_thickness_m", 0.04))
        if table_size_xy_m <= 0.0 or table_thickness_m <= 0.0:
            raise ValueError("table size and thickness must be positive")
        table_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=args.object_friction,
            dynamic_friction=args.object_friction,
            restitution=0.0,
        )
        table_cfg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Environment/Table",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, table_top_z_m - 0.5 * table_thickness_m),
            ),
            spawn=sim_utils.CuboidCfg(
                size=(table_size_xy_m, table_size_xy_m, table_thickness_m),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=table_material,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.22, 0.24, 0.28),
                ),
            ),
        )
        pedestal_height_m = float(
            getattr(args, "support_pedestal_height_m", 0.0)
        )
        if pedestal_height_m > 0.0:
            pedestal_size_xy_m = float(
                getattr(args, "support_pedestal_size_xy_m")
            )
            front_xy_m = tuple(
                float(value)
                for value in getattr(args, "support_pedestal_front_xy_m")
            )
            back_xy_m = tuple(
                float(value)
                for value in getattr(args, "support_pedestal_back_xy_m")
            )
            if (
                pedestal_size_xy_m <= 0.0
                or len(front_xy_m) != 2
                or len(back_xy_m) != 2
            ):
                raise ValueError(
                    "pedestal size must be positive and centers must be XY"
                )

            def pedestal_cfg(name: str, xy_m: tuple[float, float]):
                return AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Environment/Table/{name}",
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=(
                            xy_m[0],
                            xy_m[1],
                            table_top_z_m + 0.5 * pedestal_height_m,
                        ),
                    ),
                    spawn=sim_utils.CuboidCfg(
                        size=(
                            pedestal_size_xy_m,
                            pedestal_size_xy_m,
                            pedestal_height_m,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        physics_material=table_material,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.42, 0.44, 0.48),
                        ),
                    ),
                )

            front_pedestal_cfg = pedestal_cfg(
                "FrontPedestal", front_xy_m
            )
            back_pedestal_cfg = pedestal_cfg("BackPedestal", back_xy_m)

    if table_cfg is None:
        @configclass
        class DualValidationSceneCfg(InteractiveSceneCfg):
            robot: ArticulationCfg = robot_cfg
            right_object: RigidObjectCfg = object_cfg(
                "ObjectRight", right_object_usd
            )
            left_object: RigidObjectCfg = object_cfg(
                "ObjectLeft", left_object_usd
            )
            right_contact: ContactSensorCfg = contact_cfg("ObjectRight")
            left_contact: ContactSensorCfg = contact_cfg("ObjectLeft")
    elif front_pedestal_cfg is None:
        @configclass
        class DualValidationSceneCfg(InteractiveSceneCfg):
            robot: ArticulationCfg = robot_cfg
            right_object: RigidObjectCfg = object_cfg(
                "ObjectRight", right_object_usd
            )
            left_object: RigidObjectCfg = object_cfg(
                "ObjectLeft", left_object_usd
            )
            table: AssetBaseCfg = table_cfg
            right_contact: ContactSensorCfg = contact_cfg("ObjectRight")
            left_contact: ContactSensorCfg = contact_cfg("ObjectLeft")
    else:
        @configclass
        class DualValidationSceneCfg(InteractiveSceneCfg):
            robot: ArticulationCfg = robot_cfg
            right_object: RigidObjectCfg = object_cfg(
                "ObjectRight", right_object_usd
            )
            left_object: RigidObjectCfg = object_cfg(
                "ObjectLeft", left_object_usd
            )
            table: AssetBaseCfg = table_cfg
            front_pedestal: AssetBaseCfg = front_pedestal_cfg
            back_pedestal: AssetBaseCfg = back_pedestal_cfg
            right_contact: ContactSensorCfg = contact_cfg("ObjectRight")
            left_contact: ContactSensorCfg = contact_cfg("ObjectLeft")

    return DualValidationSceneCfg(
        num_envs=num_envs,
        env_spacing=0.8,
        replicate_physics=True,
        filter_collisions=True,
        lazy_sensor_update=False,
    )


def _disable_object_object_collisions(num_envs: int) -> None:
    import isaaclab.sim as sim_utils
    from pxr import Sdf, UsdPhysics

    stage = sim_utils.get_current_stage()
    for env_id in range(num_envs):
        right_path = f"/World/envs/env_{env_id}/ObjectRight"
        left_path = f"/World/envs/env_{env_id}/ObjectLeft"
        right_prim = stage.GetPrimAtPath(right_path)
        left_prim = stage.GetPrimAtPath(left_path)
        if not right_prim.IsValid() or not left_prim.IsValid():
            raise X2DualValidationError(
                f"cannot resolve dual object prims in env {env_id}"
            )
        relationship = (
            UsdPhysics.FilteredPairsAPI.Apply(right_prim)
            .CreateFilteredPairsRel()
        )
        relationship.AddTarget(Sdf.Path(left_path))


def _audit_second_sensor(contact_sensor, expected_env_count: int) -> None:
    if contact_sensor.num_sensors != 1:
        raise X2DualValidationError(
            "each dual object requires one sensing body per environment"
        )
    contact_view = contact_sensor.contact_view
    if (
        int(contact_view.sensor_count) != expected_env_count
        or int(contact_view.filter_count) != 0
        or not hasattr(contact_view, "get_raw_contact_data")
    ):
        raise X2DualValidationError(
            "left raw contact sensor topology/API is stale"
        )


def _prepare_batch(
    candidates: Sequence[X2DualObjectCandidate],
    *,
    capacity_samples: int,
    env_origins: Any,
    device: str,
):
    import torch

    env_count = capacity_samples * len(GRAVITY_TESTS_WXYZ)
    hand_pose = torch.zeros((env_count, 7), dtype=torch.float32, device=device)
    hand_pose[:, :3] = env_origins
    hand_pose[:, 6] = 1.0
    right_pose = hand_pose.clone()
    left_pose = hand_pose.clone()
    joint = torch.zeros((env_count, 16), dtype=torch.float32, device=device)
    target = torch.zeros((env_count, 12), dtype=torch.float32, device=device)
    gravity = torch.zeros((env_count, 3), dtype=torch.float32, device=device)
    gravity_vectors = gravity_vectors_shared_hand(9.8)
    active_env_count = len(candidates) * len(GRAVITY_TESTS_WXYZ)
    for sample_index in range(capacity_samples):
        candidate = candidates[min(sample_index, len(candidates) - 1)]
        right_record = candidate.right["pose_in_shared_hand_frame"]
        left_record = candidate.left["pose_in_shared_hand_frame"]
        for direction_index in range(len(GRAVITY_TESTS_WXYZ)):
            env_id = sample_index * len(GRAVITY_TESTS_WXYZ) + direction_index
            for destination, record in (
                (right_pose, right_record),
                (left_pose, left_record),
            ):
                destination[env_id, :3] += torch.as_tensor(
                    record["translation"],
                    dtype=torch.float32,
                    device=device,
                )
                destination[env_id, 3:7] = torch.as_tensor(
                    _matrix_to_quaternion_xyzw(record["rotation_matrix"]),
                    dtype=torch.float32,
                    device=device,
                )
            joint[env_id] = torch.as_tensor(
                candidate.record["hand"]["joint"],
                dtype=torch.float32,
                device=device,
            )
            target[env_id] = torch.as_tensor(
                candidate.record["hand"]["actuator"],
                dtype=torch.float32,
                device=device,
            )
            if env_id < active_env_count:
                gravity[env_id] = torch.as_tensor(
                    gravity_vectors[direction_index],
                    dtype=torch.float32,
                    device=device,
                )
    return (
        hand_pose,
        right_pose,
        left_pose,
        joint,
        target,
        gravity,
        active_env_count,
    )


def _object_dynamics(rigid_object: Any, active_env_count: int) -> dict[str, Any]:
    import torch

    mass = _as_torch(rigid_object.data.body_mass)[
        :active_env_count
    ].reshape(active_env_count, -1)
    inertia = _as_torch(rigid_object.data.body_inertia)[
        :active_env_count
    ].reshape(active_env_count, -1)
    if (
        not bool(torch.isfinite(mass).all().item())
        or not bool(torch.isfinite(inertia).all().item())
        or not torch.allclose(mass, mass[:1].expand_as(mass))
        or not torch.allclose(inertia, inertia[:1].expand_as(inertia))
    ):
        raise X2DualValidationError(
            "cloned dual-object mass/inertia is non-finite or inconsistent"
        )
    return {
        "body_mass_kg": float(mass[0, 0].item()),
        "body_inertia_kg_m2_flat": inertia[0].detach().cpu().tolist(),
    }


def _validate_chunk(
    *,
    candidates: Sequence[X2DualObjectCandidate],
    capacity_samples: int,
    scene: Any,
    sim: Any,
    active_joint_ids: Sequence[int],
    mimic_joint_pairs: Sequence[tuple[int, int]],
    args: argparse.Namespace,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    import torch

    robot = scene["robot"]
    right_object = scene["right_object"]
    left_object = scene["left_object"]
    right_sensor = scene["right_contact"]
    left_sensor = scene["left_contact"]
    (
        hand_pose,
        right_pose,
        left_pose,
        joint,
        target,
        gravity,
        active_env_count,
    ) = _prepare_batch(
        candidates,
        capacity_samples=capacity_samples,
        env_origins=scene.env_origins,
        device=sim.device,
    )
    env_count = hand_pose.shape[0]
    zero_object_velocity = torch.zeros((env_count, 6), device=sim.device)
    zero_hand_velocity = torch.zeros((env_count, 6), device=sim.device)
    zero_joint_velocity = torch.zeros_like(joint)
    for rigid_object, pose in (
        (right_object, right_pose),
        (left_object, left_pose),
    ):
        rigid_object.write_root_pose_to_sim_index(root_pose=pose)
        rigid_object.write_root_velocity_to_sim_index(
            root_velocity=zero_object_velocity
        )
    robot.write_root_pose_to_sim_index(root_pose=hand_pose)
    robot.write_root_velocity_to_sim_index(root_velocity=zero_hand_velocity)
    robot.write_joint_position_to_sim_index(position=joint)
    robot.write_joint_velocity_to_sim_index(velocity=zero_joint_velocity)
    robot.set_joint_position_target_index(
        target=target, joint_ids=list(active_joint_ids)
    )
    scene.reset()
    sim.forward()
    scene.update(sim.get_physics_dt())

    right_dynamics = _object_dynamics(right_object, active_env_count)
    left_dynamics = _object_dynamics(left_object, active_env_count)
    right_mass = _as_torch(right_object.data.body_mass)
    left_mass = _as_torch(left_object.data.body_mass)
    right_forces = right_mass.unsqueeze(-1) * gravity.unsqueeze(1)
    left_forces = left_mass.unsqueeze(-1) * gravity.unsqueeze(1)
    right_object.permanent_wrench_composer.set_forces_and_torques_index(
        forces=right_forces,
        torques=torch.zeros_like(right_forces),
        is_global=True,
    )
    left_object.permanent_wrench_composer.set_forces_and_torques_index(
        forces=left_forces,
        torques=torch.zeros_like(left_forces),
        is_global=True,
    )
    initial_right = right_pose[:, :3].clone()
    initial_left = left_pose[:, :3].clone()
    max_right = torch.zeros(env_count, device=sim.device)
    max_left = torch.zeros(env_count, device=sim.device)
    max_joint_error = torch.zeros(env_count, device=sim.device)
    max_mimic_error = torch.zeros(env_count, device=sim.device)
    finite = torch.ones(env_count, dtype=torch.bool, device=sim.device)
    for _ in range(args.sim_steps * args.substeps):
        robot.set_joint_position_target_index(
            target=target, joint_ids=list(active_joint_ids)
        )
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())
        right_root = _as_torch(right_object.data.root_pose_w)
        left_root = _as_torch(left_object.data.root_pose_w)
        right_velocity = _as_torch(right_object.data.root_vel_w)
        left_velocity = _as_torch(left_object.data.root_vel_w)
        joint_position = _as_torch(robot.data.joint_pos)
        joint_velocity = _as_torch(robot.data.joint_vel)
        step_finite = (
            torch.isfinite(right_root).all(dim=-1)
            & torch.isfinite(left_root).all(dim=-1)
            & torch.isfinite(right_velocity).all(dim=-1)
            & torch.isfinite(left_velocity).all(dim=-1)
            & torch.isfinite(joint_position).all(dim=-1)
            & torch.isfinite(joint_velocity).all(dim=-1)
        )
        finite &= step_finite
        right_displacement = torch.linalg.vector_norm(
            right_root[:, :3] - initial_right, dim=-1
        )
        left_displacement = torch.linalg.vector_norm(
            left_root[:, :3] - initial_left, dim=-1
        )
        right_displacement = torch.where(
            torch.isfinite(right_displacement),
            right_displacement,
            torch.full_like(right_displacement, math.inf),
        )
        left_displacement = torch.where(
            torch.isfinite(left_displacement),
            left_displacement,
            torch.full_like(left_displacement, math.inf),
        )
        max_right = torch.maximum(max_right, right_displacement)
        max_left = torch.maximum(max_left, left_displacement)
        joint_error = (
            joint_position[:, list(active_joint_ids)] - target
        ).abs().amax(dim=-1)
        joint_error = torch.where(
            torch.isfinite(joint_error),
            joint_error,
            torch.full_like(joint_error, math.inf),
        )
        max_joint_error = torch.maximum(max_joint_error, joint_error)
        mimic_error = torch.stack(
            [
                (
                    joint_position[:, follower]
                    - joint_position[:, driver]
                ).abs()
                for follower, driver in mimic_joint_pairs
            ],
            dim=-1,
        ).amax(dim=-1)
        mimic_error = torch.where(
            torch.isfinite(mimic_error),
            mimic_error,
            torch.full_like(mimic_error, math.inf),
        )
        max_mimic_error = torch.maximum(max_mimic_error, mimic_error)

    final_right = _as_torch(right_object.data.root_pos_w)
    final_left = _as_torch(left_object.data.root_pos_w)
    final_right_displacement = torch.linalg.vector_norm(
        final_right - initial_right, dim=-1
    )
    final_left_displacement = torch.linalg.vector_norm(
        final_left - initial_left, dim=-1
    )
    right_contact, right_force = _read_raw_object_contacts(
        right_sensor, env_count=env_count, dt=sim.get_physics_dt()
    )
    left_contact, left_force = _read_raw_object_contacts(
        left_sensor, env_count=env_count, dt=sim.get_physics_dt()
    )
    gravity_values = gravity.detach().cpu().tolist()
    results: list[list[dict[str, Any]]] = []
    for sample_index in range(len(candidates)):
        orientations: list[dict[str, Any]] = []
        for direction_index, name in enumerate(EXPECTED_GRAVITY_NAMES):
            env_id = sample_index * len(GRAVITY_TESTS_WXYZ) + direction_index
            common_passed = bool(
                finite[env_id].item()
                and max_joint_error[env_id].item()
                < args.joint_error_threshold
                and max_mimic_error[env_id].item()
                < args.mimic_error_threshold
            )

            def branch(
                final_displacement: Any,
                maximum_displacement: Any,
                contact: Any,
                force: Any,
            ) -> dict[str, Any]:
                final_value = float(final_displacement[env_id].item())
                maximum_value = float(maximum_displacement[env_id].item())
                contact_value = bool(contact[env_id].item())
                passed = bool(
                    common_passed
                    and contact_value
                    and final_value < args.retention_distance
                    and maximum_value < args.retention_distance
                )
                return {
                    "passed": passed,
                    "hand_contact": contact_value,
                    "final_contact_force_n": float(force[env_id].item()),
                    "final_displacement_m": final_value,
                    "maximum_displacement_m": maximum_value,
                }

            right_result = branch(
                final_right_displacement,
                max_right,
                right_contact,
                right_force,
            )
            left_result = branch(
                final_left_displacement,
                max_left,
                left_contact,
                left_force,
            )
            orientations.append(
                {
                    "name": name,
                    "passed": right_result["passed"]
                    and left_result["passed"],
                    "finite": bool(finite[env_id].item()),
                    "gravity_vector_shared_hand_frame": gravity_values[env_id],
                    "maximum_active_joint_error_rad": float(
                        max_joint_error[env_id].item()
                    ),
                    "maximum_newton_mimic_error_rad": float(
                        max_mimic_error[env_id].item()
                    ),
                    "objects": {
                        "right": right_result,
                        "left": left_result,
                    },
                }
            )
        results.append(orientations)
    return results, {
        "right_object_dynamics": right_dynamics,
        "left_object_dynamics": left_dynamics,
        "maximum_newton_mimic_error_rad": float(
            max_mimic_error[:active_env_count].amax().item()
        ),
        "maximum_active_joint_error_rad": float(
            max_joint_error[:active_env_count].amax().item()
        ),
    }


def _runtime_metadata(
    *,
    args: argparse.Namespace,
    hand_usd: Path,
    right_usd: Path,
    left_usd: Path,
    right_cache_key: str,
    left_cache_key: str,
    right_bounds: Mapping[str, Any],
    left_bounds: Mapping[str, Any],
) -> dict[str, Any]:
    import torch
    from isaaclab.utils.version import get_isaac_sim_version

    return {
        "isaac_sim_version": str(get_isaac_sim_version()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(args.device),
        "dt_s": args.dt,
        "physics_dt_s": args.dt / args.substeps,
        "substeps": args.substeps,
        "simulation_steps": args.sim_steps,
        "physics_step_count": args.sim_steps * args.substeps,
        "gravity_implementation": (
            "zero_global_gravity_plus_identical_per_environment_com_force_"
            "on_each_object"
        ),
        "replay_frame": "shared_hand",
        "hand_pose": "identity",
        "object_pose": "candidate.pose_in_shared_hand_frame",
        "initial_dof_state": "composed_candidate_joint",
        "actuator_target_state": "composed_candidate_actuator",
        "object_object_initial_collision_gate": True,
        "object_object_physx_collision_enabled": bool(
            args.enable_object_object_collisions
        ),
        "contact_semantics": (
            "raw per-object contact; object-object collision is enabled and "
            "contact provenance must be separated by the sensor audit"
            if args.enable_object_object_collisions
            else "raw per-object contact; with object-object collision filtered, "
            "every reported patch is direct X2 hand contact"
        ),
        "criterion": DUAL_VALIDATION_CRITERION,
        "thresholds": {
            "maximum_object_displacement_m": args.retention_distance,
            "maximum_active_joint_error_rad": args.joint_error_threshold,
            "maximum_newton_mimic_error_rad": args.mimic_error_threshold,
            "maximum_hand_object_penetration_m": (
                args.hand_object_penetration_threshold
            ),
            "maximum_object_object_penetration_m": (
                args.object_object_penetration_threshold
            ),
        },
        "physx_solver": {
            "solver_type": args.solver_type,
            "external_forces_every_iteration": (
                args.external_forces_every_iteration
            ),
            "solve_articulation_contact_last": (
                args.solve_articulation_contact_last
            ),
        },
        "actuator_drive": {
            "stiffness_n_m_per_rad": args.actuator_stiffness,
            "damping_n_m_s_per_rad": args.actuator_damping,
            "armature_kg_m2": args.actuator_armature,
        },
        "hand_usd": str(hand_usd),
        "objects": {
            "right": {
                "usd": str(right_usd),
                "cache_key": right_cache_key,
                "bounds_audit": dict(right_bounds),
            },
            "left": {
                "usd": str(left_usd),
                "cache_key": left_cache_key,
                "bounds_audit": dict(left_bounds),
            },
        },
    }


def _conversion_args(
    args: argparse.Namespace, mesh_path: Path
) -> argparse.Namespace:
    result = copy.copy(args)
    result.collision_approximation = (
        "convex-hull"
        if "x2_primitives" in mesh_path.parts
        else "convex-decomposition"
    )
    return result


def _worker(
    candidates: Sequence[X2DualObjectCandidate],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not candidates:
        raise X2DualValidationError("worker selection is empty")
    groups = group_candidates_by_objects(candidates)
    if len(groups) != 1:
        raise X2DualValidationError("worker selection spans multiple object pairs")
    existing: list[Path] = []
    pending: list[X2DualObjectCandidate] = []
    for candidate in candidates:
        output = existing_validation_output(
            candidate,
            args.output_root,
            protocol_revision=_validation_protocol_revision(args),
        )
        if output is not None and not (args.resume or args.overwrite):
            raise X2DualValidationError(
                f"dual validation output already exists: {output}"
            )
        if args.overwrite:
            output = None
        if output is None:
            pending.append(candidate)
        else:
            existing.append(output)
    if not pending:
        return {
            "passed": True,
            "candidate_count": len(candidates),
            "processed_count": 0,
            "skipped_existing_count": len(existing),
        }
    # Isaac Sim must own the process C++ runtime before PyTorch is imported by
    # the static preflight.  Reversing this order lets Conda's libstdc++ win and
    # can abort USD extension discovery before PhysX starts.
    hand_usd = _resolved_file(args.hand_usd, "X2 hand USD")
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        static_reports = _static_preflight_group(pending, args)
        dynamic: list[X2DualObjectCandidate] = []
        dynamic_static: list[dict[str, Any]] = []
        static_failed = 0
        for candidate, report in zip(pending, static_reports):
            if report["passed"]:
                dynamic.append(candidate)
                dynamic_static.append(report)
                continue
            record = make_validation_record(
                candidate,
                passed=False,
                simulation_ran=False,
                static_preflight=report,
                orientations=[],
                runtime={
                    "protocol_revision": _validation_protocol_revision(args),
                    "reason": (
                        "static_preflight_rejected_before_physx_scene_creation"
                    ),
                },
                failure_reasons=report["failure_reasons"],
                protocol_revision=_validation_protocol_revision(args),
            )
            write_validation_record(
                candidate, record, args.output_root, overwrite=args.overwrite
            )
            static_failed += 1
        if not dynamic:
            return {
                "passed": True,
                "candidate_count": len(candidates),
                "processed_count": len(pending),
                "skipped_existing_count": len(existing),
                "static_failed_count": static_failed,
                "physx_candidate_count": 0,
            }

        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext
        from isaaclab_physx.physics import PhysxCfg

        (right_group, left_group), _ = next(iter(groups.items()))
        right_mesh, right_scale = right_group
        left_mesh, left_scale = left_group
        right_args = _conversion_args(args, right_mesh)
        left_args = _conversion_args(args, left_mesh)
        right_usd, right_key = _convert_object_mesh(
            right_mesh, right_scale, right_args
        )
        left_usd, left_key = _convert_object_mesh(
            left_mesh, left_scale, left_args
        )
        right_bounds = _audit_converted_bounds(
            right_mesh, right_scale, right_usd
        )
        left_bounds = _audit_converted_bounds(
            left_mesh, left_scale, left_usd
        )
        capacity = min(args.batch_size, len(dynamic))
        env_count = capacity * len(GRAVITY_TESTS_WXYZ)
        scene_cfg = _make_dual_scene_cfg(
            hand_usd=hand_usd,
            right_object_usd=right_usd,
            left_object_usd=left_usd,
            num_envs=env_count,
            args=args,
        )
        sim = SimulationContext(
            sim_utils.SimulationCfg(
                device=args.device,
                dt=args.dt / args.substeps,
                gravity=(0.0, 0.0, 0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=args.object_friction,
                    dynamic_friction=args.object_friction,
                    restitution=0.0,
                ),
                physics=PhysxCfg(
                    solver_type=args.solver_type,
                    enable_external_forces_every_iteration=(
                        args.external_forces_every_iteration
                    ),
                    solve_articulation_contact_last=(
                        args.solve_articulation_contact_last
                    ),
                ),
                use_fabric=True,
                render_interval=max(1, args.sim_steps),
            )
        )
        scene = InteractiveScene(scene_cfg)
        if not args.enable_object_object_collisions:
            _disable_object_object_collisions(env_count)
        sim.reset()
        robot = scene["robot"]
        active_joint_ids, mimic_pairs = _audit_runtime_mapping(
            robot,
            scene["right_contact"],
            expected_env_count=env_count,
        )
        _audit_second_sensor(scene["left_contact"], env_count)
        runtime = _runtime_metadata(
            args=args,
            hand_usd=hand_usd,
            right_usd=right_usd,
            left_usd=left_usd,
            right_cache_key=right_key,
            left_cache_key=left_key,
            right_bounds=right_bounds,
            left_bounds=left_bounds,
        )
        physx_passed = 0
        for offset in range(0, len(dynamic), capacity):
            chunk = dynamic[offset : offset + capacity]
            print(
                f"[dual-physx] offset={offset} count={len(chunk)} "
                f"total={len(dynamic)}",
                flush=True,
            )
            orientations, batch_audit = _validate_chunk(
                candidates=chunk,
                capacity_samples=capacity,
                scene=scene,
                sim=sim,
                active_joint_ids=active_joint_ids,
                mimic_joint_pairs=mimic_pairs,
                args=args,
            )
            for local_index, (candidate, outcomes) in enumerate(
                zip(chunk, orientations)
            ):
                passed = len(outcomes) == 6 and all(
                    value["passed"] for value in outcomes
                )
                failure_reasons = (
                    []
                    if passed
                    else [
                        "one_or_more_orientations_failed_both_object_hold"
                    ]
                )
                candidate_index = dynamic.index(candidate)
                record = make_validation_record(
                    candidate,
                    passed=passed,
                    simulation_ran=True,
                    static_preflight=dynamic_static[candidate_index],
                    orientations=outcomes,
                    runtime={
                        **runtime,
                        "batch_audit": batch_audit,
                    },
                    failure_reasons=failure_reasons,
                    protocol_revision=_validation_protocol_revision(args),
                )
                write_validation_record(
                    candidate,
                    record,
                    args.output_root,
                    overwrite=args.overwrite,
                )
                physx_passed += int(passed)
        return {
            "passed": True,
            "candidate_count": len(candidates),
            "processed_count": len(pending),
            "skipped_existing_count": len(existing),
            "static_failed_count": static_failed,
            "physx_candidate_count": len(dynamic),
            "physx_passed_count": physx_passed,
        }
    finally:
        simulation_app.close()


def _worker_command(
    args: argparse.Namespace, selection_path: Path
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset-root",
        str(args.dataset_root),
        "--output-root",
        str(args.output_root),
        "--worker-selection",
        str(selection_path),
        "--batch-size",
        str(args.batch_size),
        "--sim-steps",
        str(args.sim_steps),
        "--substeps",
        str(args.substeps),
        "--dt",
        str(args.dt),
        "--hand-usd",
        str(args.hand_usd),
        "--usd-cache",
        str(args.usd_cache),
        "--density",
        str(args.density),
        "--hand-friction",
        str(args.hand_friction),
        "--object-friction",
        str(args.object_friction),
        "--contact-offset",
        str(args.contact_offset),
        "--rest-offset",
        str(args.rest_offset),
        "--retention-distance",
        str(args.retention_distance),
        "--joint-error-threshold",
        str(args.joint_error_threshold),
        "--mimic-error-threshold",
        str(args.mimic_error_threshold),
        "--hand-object-penetration-threshold",
        str(args.hand_object_penetration_threshold),
        "--object-object-penetration-threshold",
        str(args.object_object_penetration_threshold),
        "--actuator-stiffness",
        str(args.actuator_stiffness),
        "--actuator-damping",
        str(args.actuator_damping),
        "--actuator-armature",
        str(args.actuator_armature),
        "--solver-type",
        str(args.solver_type),
        "--device",
        str(args.device),
        "--headless",
    ]
    if args.allow_incomplete:
        command.append("--allow-incomplete")
    if args.external_forces_every_iteration:
        command.append("--external-forces-every-iteration")
    else:
        command.append("--no-external-forces-every-iteration")
    if args.solve_articulation_contact_last:
        command.append("--solve-articulation-contact-last")
    if args.enable_object_object_collisions:
        command.append("--enable-object-object-collisions")
    if args.combination is not None:
        command.extend(("--combination", args.combination))
    if args.overwrite:
        command.append("--overwrite")
    elif args.resume:
        command.append("--resume")
    return command


def _aggregate(
    candidates: Sequence[X2DualObjectCandidate],
    args: argparse.Namespace,
    manifest_sha256: str,
) -> dict[str, Any]:
    valid = 0
    failed = 0
    by_combination: dict[str, dict[str, int]] = {}
    for candidate in candidates:
        output = existing_validation_output(
            candidate,
            args.output_root,
            protocol_revision=_validation_protocol_revision(args),
        )
        if output is None:
            raise X2DualValidationError(
                f"candidate is not routed after validation: {candidate.candidate_id}"
            )
        status = strict_json(output)["dual_object_validation"]["status"]
        valid += int(status == "passed")
        failed += int(status == "failed")
        counts = by_combination.setdefault(
            candidate.combination, {"valid": 0, "failed": 0}
        )
        if status == "passed":
            counts["valid"] += 1
        else:
            counts["failed"] += 1
    return {
        "schema_version": 1,
        "passed": valid + failed == len(candidates),
        "protocol_revision": _validation_protocol_revision(args),
        "backend": DUAL_VALIDATION_BACKEND,
        "criterion": DUAL_VALIDATION_CRITERION,
        "source_manifest": str((args.dataset_root / "manifest.json").resolve()),
        "source_manifest_sha256": manifest_sha256,
        "candidate_count": len(candidates),
        "valid_count": valid,
        "failed_count": failed,
        "combination_counts": by_combination,
    }


def _orchestrate(
    candidates: Sequence[X2DualObjectCandidate],
    args: argparse.Namespace,
    manifest_sha256: str,
) -> dict[str, Any]:
    groups = group_candidates_by_objects(candidates)
    if args.dry_run:
        return {
            "schema_version": 1,
            "passed": True,
            "dry_run": True,
            "would_launch_isaac_sim": True,
            "protocol_revision": _validation_protocol_revision(args),
            "candidate_count": len(candidates),
            "object_pair_group_count": len(groups),
            "environment_capacity": min(args.batch_size, len(candidates)) * 6,
            "source_manifest_sha256": manifest_sha256,
        }
    worklist_root = args.output_root / "worklists"
    for index, (group, values) in enumerate(
        sorted(groups.items(), key=lambda item: str(item[0]))
    ):
        pending = [
            candidate
            for candidate in values
            if not (
                args.resume
                and existing_validation_output(
                    candidate,
                    args.output_root,
                    protocol_revision=_validation_protocol_revision(args),
                )
                is not None
            )
        ]
        if not pending:
            continue
        label = file_sha256(values[0].path)[:12]
        selection_path = worklist_root / f"group_{index:04d}_{label}.json"
        atomic_json(
            selection_path,
            {
                "schema_version": 1,
                "source_manifest_sha256": manifest_sha256,
                "object_group": [
                    [str(path), scale] for path, scale in group
                ],
                "candidate_ids": [
                    candidate.candidate_id for candidate in pending
                ],
            },
        )
        subprocess.run(
            _worker_command(args, selection_path),
            cwd=PROJECT_ROOT,
            check=True,
        )
    return _aggregate(candidates, args, manifest_sha256)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.hand_usd = _resolved_file(args.hand_usd, "X2 hand USD")
    args.usd_cache = args.usd_cache.expanduser().resolve()
    candidates, _ = discover_candidates(
        args.dataset_root,
        require_complete=not args.allow_incomplete,
        limit=args.limit if args.worker_selection is None else None,
        combination=args.combination if args.worker_selection is None else None,
    )
    manifest_path = args.dataset_root / "manifest.json"
    manifest_sha256 = file_sha256(manifest_path)
    if args.worker_selection is not None:
        selection = strict_json(args.worker_selection.expanduser().resolve())
        if selection.get("source_manifest_sha256") != manifest_sha256:
            raise X2DualValidationError("worker source manifest hash is stale")
        ids = selection.get("candidate_ids")
        if not isinstance(ids, list) or not ids:
            raise X2DualValidationError("worker candidate id list is empty")
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        try:
            selected = [by_id[value] for value in ids]
        except (KeyError, TypeError) as exc:
            raise X2DualValidationError(
                "worker candidate id is missing from source manifest"
            ) from exc
        summary = _worker(selected, args)
        summary["worker_selection"] = str(
            args.worker_selection.expanduser().resolve()
        )
    else:
        summary = _orchestrate(candidates, args, manifest_sha256)
    if args.summary_json:
        atomic_json(args.summary_json.expanduser().resolve(), summary)
    elif args.worker_selection is None and not args.dry_run:
        atomic_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except X2DualValidationError as exc:
        raise SystemExit(f"error: {exc}") from exc
