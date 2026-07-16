#!/usr/bin/env python3
"""Validate X2 mesh-grasp raw JSON with Isaac Sim/PhysX.

The protocol follows the official DexGraspNet validator's six gravity tests,
while fixing its dropped batch remainder and adapting it to the calibrated X2
USD, 12 actuators, 16 runtime joints and schema-1 JSON.  Raw files are never
modified: enriched copies are atomically written to sibling ``valid`` or
``failed`` directories.

One invocation handles exactly one ``(mesh_path, object_scale)`` group so that
all cloned environments share one collision asset.  The primitive dataset
wrapper invokes this script once per object instance.

The authored articulation keeps PhysX self-collision disabled.  For v4 raw
records, overall success additionally requires the generator's static sampled
hull ``self_collision.feasible`` gate; ``simulation_success`` remains the pure
six-orientation PhysX grasp/mimic result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

# Isaac Sim's startup probes fork the process.  Multi-threaded BLAS pools loaded
# before AppLauncher can deadlock or segfault in that fork (observed in
# scipy-openblas on this validator), so this process deliberately uses one BLAS
# thread.  Physics remains fully GPU-batched.
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

from grasp_generation.x2_isaac_validation import (  # noqa: E402
    CLOSING_LINE_SEARCH_ALPHAS,
    EXPECTED_ACTUATOR_NAMES,
    EXPECTED_JOINT_NAMES,
    FORMAL_ACTUATOR_ARMATURE,
    FORMAL_ACTUATOR_DAMPING,
    FORMAL_ACTUATOR_STIFFNESS,
    GRAVITY_TESTS_WXYZ,
    MAX_CLOSING_TARGET_PENETRATION_CAP,
    PASSIVE_MIMIC_DRIVERS,
    PROTOCOL_REVISION,
    VALIDATION_CRITERIA,
    OrientationOutcome,
    ValidationThresholds,
    X2RawCandidate,
    X2ValidationError,
    compact_collision_aware_closing_batch_audit,
    discover_raw_candidates,
    evaluate_orientation,
    group_candidates_by_mesh,
    make_object_centered_replay,
    make_validated_record,
    quaternion_matrix_wxyz,
    select_collision_aware_closing,
    summarize_newton_mimic_errors,
    validation_output_path,
    write_validated_record,
)

from isaaclab.app import AppLauncher  # noqa: E402


DEFAULT_HAND_USD = PROJECT_ROOT / "x2_mujoco" / "x2_keypoints.usda"
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "x2_primitive_grasps"
DEFAULT_USD_CACHE = PROJECT_ROOT / "data" / "cache" / "x2_physx_objects"
MAX_CONTACT_DATA_PER_OBJECT = 256

ACTUATOR_GROUP_EXPRS = {
    "finger_j3": [r"rh_(?:LF|RF|MF|FF)J3"],
    "finger_j2": [r"rh_(?:LF|RF|MF|FF)J2"],
    "thumb": [r"rh_THJ[1-4]"],
}

# PhysX filtered contact reporting is one-to-many.  The object is the single
# sensing body and these are the 17 X2 rigid-body partners below the spawned
# ``Robot`` prim.  Keeping the authored paths explicit prevents unrelated
# scene contacts from satisfying the DexGraspNet-style contact test.
HAND_RIGID_BODY_RELATIVE_PATHS = (
    "Geometry/mujoco_root/rh_palm",
    "Geometry/mujoco_root/rh_palm/rh_lfproximal",
    "Geometry/mujoco_root/rh_palm/rh_lfproximal/rh_lfmiddle",
    "Geometry/mujoco_root/rh_palm/rh_lfproximal/rh_lfmiddle/rh_lfdistal",
    "Geometry/mujoco_root/rh_palm/rh_rfproximal",
    "Geometry/mujoco_root/rh_palm/rh_rfproximal/rh_rfmiddle",
    "Geometry/mujoco_root/rh_palm/rh_rfproximal/rh_rfmiddle/rh_rfdistal",
    "Geometry/mujoco_root/rh_palm/rh_mfproximal",
    "Geometry/mujoco_root/rh_palm/rh_mfproximal/rh_mfmiddle",
    "Geometry/mujoco_root/rh_palm/rh_mfproximal/rh_mfmiddle/rh_mfdistal",
    "Geometry/mujoco_root/rh_palm/rh_ffproximal",
    "Geometry/mujoco_root/rh_palm/rh_ffproximal/rh_ffmiddle",
    "Geometry/mujoco_root/rh_palm/rh_ffproximal/rh_ffmiddle/rh_ffdistal",
    "Geometry/mujoco_root/rh_palm/rh_thbase",
    "Geometry/mujoco_root/rh_palm/rh_thbase/rh_thproximal",
    "Geometry/mujoco_root/rh_palm/rh_thbase/rh_thproximal/rh_thmiddle",
    "Geometry/mujoco_root/rh_palm/rh_thbase/rh_thproximal/rh_thmiddle/rh_thdistal",
)

# Each X2 rigid body owns one enabled, explicitly-authored low-vertex PhysX
# collision hull.  PhysX filtered contact patterns must resolve exactly one
# collision shape per environment; filtering at the rigid-body parent expands
# to both visual and collision descendants and is rejected by the tensor API.
HAND_COLLISION_SHAPE_RELATIVE_PATHS = tuple(
    f"{body_path}/x2_physx_collision_hull"
    for body_path in HAND_RIGID_BODY_RELATIVE_PATHS
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--mesh-path",
        type=Path,
        help="Validate only records for this mesh; required when input contains multiple meshes.",
    )
    parser.add_argument("--side", choices=("front", "back", "both"), default="both")
    parser.add_argument("--batch-size", type=_positive_int, default=32, help="Candidates per PhysX batch.")
    parser.add_argument("--sim-steps", type=_positive_int, default=100)
    parser.add_argument("--dt", type=_positive_float, default=1.0 / 60.0)
    parser.add_argument(
        "--substeps",
        type=_positive_int,
        default=2,
        help="Physics substeps per logical validation step; DexGraspNet uses 2.",
    )
    parser.add_argument(
        "--actuator-stiffness",
        type=_nonnegative_float,
        default=FORMAL_ACTUATOR_STIFFNESS,
        help=(
            "PhysX implicit-drive stiffness in N*m/rad; defaults to the "
            "formal X2 stability calibration."
        ),
    )
    parser.add_argument(
        "--actuator-damping",
        type=_nonnegative_float,
        default=FORMAL_ACTUATOR_DAMPING,
        help=(
            "PhysX implicit-drive damping in N*m*s/rad; defaults to the "
            "formal X2 stability calibration."
        ),
    )
    parser.add_argument(
        "--actuator-armature",
        type=_nonnegative_float,
        default=FORMAL_ACTUATOR_ARMATURE,
        help=(
            "Active-joint rotor armature in kg*m^2; defaults to the formal "
            "X2 stability calibration."
        ),
    )
    parser.add_argument(
        "--solver-type",
        type=int,
        choices=(0, 1),
        default=1,
        help="PhysX solver: 0=PGS, 1=TGS.",
    )
    parser.add_argument(
        "--external-forces-every-iteration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply the per-environment gravity force on every TGS position "
            "iteration to avoid noisy velocity updates."
        ),
    )
    parser.add_argument(
        "--solve-articulation-contact-last",
        action="store_true",
        help="Resolve dynamic contacts at the end of each articulation solve.",
    )
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--hand-usd", type=Path, default=DEFAULT_HAND_USD)
    parser.add_argument("--usd-cache", type=Path, default=DEFAULT_USD_CACHE)
    parser.add_argument(
        "--collision-approximation",
        choices=("convex-hull", "convex-decomposition"),
        default="convex-decomposition",
        help="Use convex-hull only for known convex objects such as the primitive dataset.",
    )
    parser.add_argument("--density", type=_positive_float, default=500.0)
    parser.add_argument("--hand-friction", type=_nonnegative_float, default=3.0)
    parser.add_argument("--object-friction", type=_nonnegative_float, default=3.0)
    parser.add_argument("--contact-offset", type=_nonnegative_float, default=0.001)
    parser.add_argument("--rest-offset", type=_nonnegative_float, default=0.0)
    parser.add_argument(
        "--contact-force-threshold",
        type=_nonnegative_float,
        default=0.0,
        help=(
            "Force threshold retained in validation metadata. The default criterion uses "
            "the isolated object's raw hand-contact count, matching DexGraspNet's "
            "end-of-test contact predicate."
        ),
    )
    parser.add_argument("--penetration-threshold", type=_positive_float, default=0.001)
    parser.add_argument("--retention-distance", type=_positive_float, default=0.1)
    parser.add_argument("--joint-error-threshold", type=_positive_float, default=0.1)
    parser.add_argument(
        "--mimic-error-threshold",
        type=_positive_float,
        default=0.01,
        help=(
            "Fail only the affected sample/orientation if any PhysX step violates an "
            "X2 J1=J2 follower by this many radians."
        ),
    )
    parser.add_argument("--fk-position-tolerance", type=_positive_float, default=5.0e-4)
    parser.add_argument("--fk-normal-min-dot", type=float, default=0.999)
    parser.add_argument("--criterion", choices=VALIDATION_CRITERIA, default="dexgraspnet-contact")
    parser.add_argument(
        "--no-force-closing",
        action="store_true",
        help="Disable the official-style pre-simulation contact-gradient closing step.",
    )
    parser.add_argument(
        "--closing-contact-threshold",
        type=_positive_float,
        default=0.003,
        help="X2 near-surface range whose links participate in pre-simulation closing.",
    )
    parser.add_argument("--closing-displacement", type=_positive_float, default=0.001)
    parser.add_argument(
        "--closing-gradient-scale",
        type=_positive_float,
        default=100.0,
        help="X2-calibrated scale for the source-style closing gradient.",
    )
    parser.add_argument(
        "--closing-penetration-cap",
        type=_positive_float,
        default=0.0015,
        help=(
            "Strict sampled bidirectional cap for the closing target; independent "
            "of the raw 1 mm static gate and never allowed above 2 mm."
        ),
    )
    parser.add_argument(
        "--preclose-physics-steps",
        type=_nonnegative_int,
        default=0,
        help=(
            "Experimental zero-gravity setup: linearly ramp the active joints from "
            "the raw JSON state to the closing target for this many physics steps "
            "before the unchanged six-direction validation. Zero preserves the "
            "formal raw-initial-state baseline."
        ),
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Skip raw candidates that already have a sibling valid/failed result.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional atomic JSON summary path. A full summary is always printed at the end.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run JSON/schema/output preflight without launching Isaac Sim.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _resolve_existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise X2ValidationError(f"{label} does not exist: {resolved}")
    return resolved


def _preflight_outputs(candidates: Sequence[X2RawCandidate], overwrite: bool) -> None:
    if overwrite:
        return
    for candidate in candidates:
        valid_path = validation_output_path(candidate.path, True)
        failed_path = validation_output_path(candidate.path, False)
        if valid_path.exists() or failed_path.exists():
            existing = valid_path if valid_path.exists() else failed_path
            raise X2ValidationError(f"validation output already exists: {existing}")


def _has_validation_output(candidate: X2RawCandidate) -> bool:
    return validation_output_path(candidate.path, True).exists() or validation_output_path(
        candidate.path, False
    ).exists()


def _atomic_json(path: Path, payload: Any) -> None:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
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


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _mesh_cache_key(mesh_path: Path, scale: float, args: argparse.Namespace) -> str:
    from isaaclab.utils.version import get_isaac_sim_version

    digest = hashlib.sha256()
    digest.update(mesh_path.read_bytes())
    settings = {
        "scale": scale,
        "collision_approximation": args.collision_approximation,
        "density": args.density,
        "contact_offset": args.contact_offset,
        "rest_offset": args.rest_offset,
        "isaac_sim_version": str(get_isaac_sim_version()),
        "isaaclab_version": _package_version("isaaclab"),
        "protocol_revision": PROTOCOL_REVISION,
        "converter_revision": 2,
    }
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _as_torch(value):
    return value.torch if hasattr(value, "torch") else value


def _quat_apply_xyzw(torch_module, quaternion, vector):
    xyz = quaternion[..., :3]
    w = quaternion[..., 3:4]
    cross = torch_module.linalg.cross(xyz, vector, dim=-1)
    return vector + 2.0 * (w * cross + torch_module.linalg.cross(xyz, cross, dim=-1))


def _convert_object_mesh(mesh_path: Path, scale: float, args: argparse.Namespace) -> tuple[Path, str]:
    """Convert one OBJ to a cached dynamic USD after Kit has launched."""

    import isaaclab.sim as sim_utils

    cache_key = _mesh_cache_key(mesh_path, scale, args)
    cache_dir = args.usd_cache.expanduser()
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    cache_dir = (cache_dir / cache_key).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.collision_approximation == "convex-hull":
        mesh_collision_props = sim_utils.ConvexHullPropertiesCfg()
    else:
        mesh_collision_props = sim_utils.ConvexDecompositionPropertiesCfg()
    converter_cfg = sim_utils.MeshConverterCfg(
        asset_path=str(mesh_path),
        usd_dir=str(cache_dir),
        usd_file_name="object.usd",
        force_usd_conversion=False,
        make_instanceable=True,
        scale=(scale, scale, scale),
        mass_props=sim_utils.MassPropertiesCfg(density=args.density),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=True,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=args.contact_offset,
            rest_offset=args.rest_offset,
        ),
        mesh_collision_props=mesh_collision_props,
    )
    converter = sim_utils.MeshConverter(converter_cfg)
    usd_path = Path(converter.usd_path).resolve()
    if not usd_path.is_file():
        raise X2ValidationError(f"MeshConverter did not create USD: {usd_path}")
    return usd_path, cache_key


def _audit_converted_bounds(mesh_path: Path, scale: float, usd_path: Path) -> dict[str, Any]:
    """Reject OBJ importer axis/unit changes before starting physics."""

    import trimesh
    from pxr import Usd, UsdGeom

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    expected_min = np.asarray(mesh.bounds[0], dtype=np.float64) * scale
    expected_max = np.asarray(mesh.bounds[1], dtype=np.float64) * scale
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise X2ValidationError(f"could not open converted object USD: {usd_path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise X2ValidationError(f"converted object USD has no default prim: {usd_path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    aligned = cache.ComputeWorldBound(default_prim).ComputeAlignedBox()
    actual_min = np.asarray(aligned.GetMin(), dtype=np.float64)
    actual_max = np.asarray(aligned.GetMax(), dtype=np.float64)
    tolerance = max(1.0e-8, float(np.max(expected_max - expected_min)) * 1.0e-6)
    if not (
        np.allclose(actual_min, expected_min, atol=tolerance, rtol=0.0)
        and np.allclose(actual_max, expected_max, atol=tolerance, rtol=0.0)
    ):
        raise X2ValidationError(
            "converted USD bounds disagree with scaled OBJ: "
            f"expected=({expected_min.tolist()}, {expected_max.tolist()}), "
            f"actual=({actual_min.tolist()}, {actual_max.tolist()})"
        )
    return {
        "expected_min_m": expected_min.tolist(),
        "expected_max_m": expected_max.tolist(),
        "usd_min_m": actual_min.tolist(),
        "usd_max_m": actual_max.tolist(),
        "tolerance_m": tolerance,
    }


def _make_scene_cfg(
    *,
    hand_usd: Path,
    object_usd: Path,
    num_envs: int,
    args: argparse.Namespace,
    object_prim_name: str = "Object",
):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg
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
            usd_path=str(hand_usd),
            physics_material=hand_material,
        ),
        actuators={
            group_name: ImplicitActuatorCfg(
                joint_names_expr=expressions,
                stiffness=args.actuator_stiffness,
                damping=args.actuator_damping,
                armature=args.actuator_armature,
                effort_limit_sim=None,
                velocity_limit_sim=None,
            )
            for group_name, expressions in ACTUATOR_GROUP_EXPRS.items()
        },
    )
    object_prim_path = f"{{ENV_REGEX_NS}}/{object_prim_name}"
    object_cfg = RigidObjectCfg(
        prim_path=object_prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(object_usd),
            activate_contact_sensors=True,
        ),
    )
    # The scene contains only X2 and one object per isolated environment.  Use
    # PhysX's raw, unfiltered per-sensor contact buffer: it reports contact
    # counts before any force aggregation and is supported for articulation
    # contacts on the GPU.  Filtered articulation colliders are not supported
    # by the current PhysX tensor backend.
    contact_cfg = ContactSensorCfg(
        prim_path=object_prim_path,
        update_period=0.0,
        history_length=1,
        track_pose=False,
        track_contact_points=False,
        track_air_time=False,
        filter_prim_paths_expr=[],
        max_contact_data_count_per_prim=MAX_CONTACT_DATA_PER_OBJECT,
    )

    @configclass
    class ValidationSceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg = robot_cfg
        object: RigidObjectCfg = object_cfg
        object_contact: ContactSensorCfg = contact_cfg

    return ValidationSceneCfg(
        num_envs=num_envs,
        env_spacing=0.8,
        replicate_physics=True,
        filter_collisions=True,
        lazy_sensor_update=False,
    )


def _audit_runtime_mapping(
    robot, contact_sensor, *, expected_env_count: int
) -> tuple[list[int], list[tuple[int, int]]]:
    runtime_joint_names = list(robot.joint_names)
    if runtime_joint_names != list(EXPECTED_JOINT_NAMES):
        raise X2ValidationError(
            "Isaac runtime joint order changed: "
            f"expected={list(EXPECTED_JOINT_NAMES)}, actual={runtime_joint_names}"
        )
    if robot.num_joints != 16 or robot.num_bodies != 17:
        raise X2ValidationError(
            f"X2 runtime topology changed: bodies={robot.num_bodies}, joints={robot.num_joints}"
        )
    active_ids, active_names = robot.find_joints(list(EXPECTED_ACTUATOR_NAMES), preserve_order=True)
    if list(active_names) != list(EXPECTED_ACTUATOR_NAMES):
        raise X2ValidationError(f"could not resolve X2 active joints by name: {active_names}")
    actuator_owners: dict[str, list[str]] = {}
    for actuator_name, actuator in robot.actuators.items():
        for joint_name in actuator.joint_names:
            actuator_owners.setdefault(joint_name, []).append(actuator_name)
    if set(actuator_owners) != set(EXPECTED_ACTUATOR_NAMES):
        raise X2ValidationError(
            "X2 actuator ownership changed: "
            f"expected={sorted(EXPECTED_ACTUATOR_NAMES)}, actual={sorted(actuator_owners)}"
        )
    duplicate_owners = {
        name: owners for name, owners in actuator_owners.items() if len(owners) != 1
    }
    if duplicate_owners:
        raise X2ValidationError(f"X2 joints have duplicate actuator owners: {duplicate_owners}")
    driven_followers = {
        name: actuator_owners[name]
        for name in PASSIVE_MIMIC_DRIVERS
        if name in actuator_owners
    }
    if driven_followers:
        raise X2ValidationError(
            f"X2 passive Newton-mimic followers became actuator-driven: {driven_followers}"
        )
    body_ids, body_names = robot.find_bodies(list(robot.body_names), preserve_order=True)
    if len(body_ids) != 17 or len(body_names) != 17:
        raise X2ValidationError("could not resolve all 17 X2 rigid bodies")
    if contact_sensor.num_sensors != 1:
        raise X2ValidationError(
            "X2 validator requires exactly one object sensing body per environment: "
            f"got={contact_sensor.num_sensors}"
        )
    contact_view = contact_sensor.contact_view
    if int(contact_view.sensor_count) != expected_env_count:
        raise X2ValidationError(
            "raw object contact view environment count changed: "
            f"expected={expected_env_count}, actual={contact_view.sensor_count}"
        )
    if int(contact_view.filter_count) != 0:
        raise X2ValidationError("raw object contact view must not use PhysX filters")
    if not hasattr(contact_view, "get_raw_contact_data"):
        raise X2ValidationError(
            "installed PhysX tensor backend does not expose get_raw_contact_data"
        )
    joint_index = {name: index for index, name in enumerate(runtime_joint_names)}
    mimic_pairs = [
        (joint_index[follower], joint_index[driver])
        for follower, driver in PASSIVE_MIMIC_DRIVERS.items()
    ]
    return list(active_ids), mimic_pairs


def _prepare_batch(
    candidates: Sequence[X2RawCandidate],
    *,
    capacity_samples: int,
    env_origins,
    device: str,
    closing_actuator_targets=None,
):
    """Build padded tensors in sample-major, direction-minor order."""

    import torch

    if not candidates:
        raise X2ValidationError("cannot prepare an empty PhysX batch")
    env_count = capacity_samples * len(GRAVITY_TESTS_WXYZ)
    hand_pose = torch.zeros((env_count, 7), dtype=torch.float32, device=device)
    hand_pose[:, :3] = env_origins
    hand_pose[:, 6] = 1.0
    object_pose = torch.zeros((env_count, 7), dtype=torch.float32, device=device)
    object_pose[:, :3] = env_origins
    object_pose[:, 6] = 1.0
    joint_position = torch.zeros((env_count, 16), dtype=torch.float32, device=device)
    actuator_target = torch.zeros((env_count, 12), dtype=torch.float32, device=device)
    gravity = torch.zeros((env_count, 3), dtype=torch.float32, device=device)
    active_env_count = len(candidates) * len(GRAVITY_TESTS_WXYZ)

    for sample_index in range(capacity_samples):
        candidate = candidates[min(sample_index, len(candidates) - 1)]
        replay = make_object_centered_replay(candidate, gravity_magnitude=9.8)
        for direction_index, _ in enumerate(GRAVITY_TESTS_WXYZ):
            env_id = sample_index * len(GRAVITY_TESTS_WXYZ) + direction_index
            hand_pose[env_id, :3] += torch.as_tensor(
                replay.hand_translation, dtype=torch.float32, device=device
            )
            hand_pose[env_id, 3:7] = torch.as_tensor(
                replay.hand_quaternion_xyzw, dtype=torch.float32, device=device
            )
            object_pose[env_id, :3] += torch.as_tensor(
                replay.object_translation, dtype=torch.float32, device=device
            )
            object_pose[env_id, 3:7] = torch.as_tensor(
                replay.object_quaternion_xyzw, dtype=torch.float32, device=device
            )
            joint_position[env_id] = torch.as_tensor(
                [candidate.joint_by_name[name] for name in EXPECTED_JOINT_NAMES],
                dtype=torch.float32,
                device=device,
            )
            if closing_actuator_targets is None:
                actuator_target[env_id] = torch.as_tensor(
                    [candidate.actuator_by_name[name] for name in EXPECTED_ACTUATOR_NAMES],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                actuator_target[env_id] = closing_actuator_targets[
                    min(sample_index, len(candidates) - 1)
                ].to(device=device, dtype=torch.float32)
            if env_id < active_env_count:
                gravity[env_id] = torch.as_tensor(
                    replay.gravity_vectors[direction_index], dtype=torch.float32, device=device
                )
    return (
        hand_pose,
        object_pose,
        joint_position,
        actuator_target,
        gravity,
        active_env_count,
    )


def _compute_contact_closing_targets(
    candidates: Sequence[X2RawCandidate], *, device: str, args: argparse.Namespace
):
    """Compute float32-rechecked collision-aware X2 actuator targets.

    Raw JSON q always remains the PhysX initial state.  When force closing is
    enabled, the source-style full gradient target is only a proposal: a
    descending per-row line search selects the largest alpha whose sampled
    bidirectional hand/object penetration is strictly below 1 mm, whose state
    is finite, and whose actuators remain inside authored limits.  An unsafe
    raw state is never "repaired" because PhysX must still replay that raw q;
    it falls back to raw and is rejected by the v6 static gate.
    """

    import torch

    from grasp_generation.utils.mesh_object_model import MeshObjectModel
    from grasp_generation.utils.x2_config import load_x2_mesh_config
    from grasp_generation.utils.x2_hand_model import X2HandModel
    from grasp_generation.utils.x2_mesh_contacts import load_generic_contact_candidates

    config = load_x2_mesh_config()
    authored_candidates = load_generic_contact_candidates(
        config.configured_path("contact_candidates.path", must_exist=True)
    )
    collision_samples_per_link = int(
        config.require("generation.hand_collision_samples_per_link")
    )
    object_surface_samples = int(config.require("generation.object_surface_samples"))
    hand = X2HandModel(
        config,
        authored_candidates,
        device=device,
        dtype=torch.float64,
        collision_samples_per_link=collision_samples_per_link,
        freeze_thumb=False,
    )
    rotations = np.stack(
        [quaternion_matrix_wxyz(candidate.hand_quaternion_wxyz) for candidate in candidates]
    )
    rotation6d = torch.as_tensor(
        rotations.transpose(0, 2, 1)[:, :2].reshape(len(candidates), 6),
        dtype=torch.float64,
        device=device,
    )
    raw_actuator = torch.as_tensor(
        [
            [candidate.actuator_by_name[name] for name in EXPECTED_ACTUATOR_NAMES]
            for candidate in candidates
        ],
        dtype=torch.float64,
        device=device,
    )
    point_index = {
        candidate.point_id: index for index, candidate in enumerate(authored_candidates)
    }
    try:
        contact_indices = torch.as_tensor(
            [
                [point_index[point_id] for point_id in candidate.record["selected_contact_ids"]]
                for candidate in candidates
            ],
            dtype=torch.long,
            device=device,
        )
    except KeyError as exc:
        raise X2ValidationError(
            f"raw contact ID is absent from the authored X2 pool: {exc.args[0]}"
        ) from exc
    pose_prefix = torch.cat(
        (
            torch.as_tensor(
                np.stack([candidate.hand_translation for candidate in candidates]),
                dtype=torch.float64,
                device=device,
            ),
            rotation6d,
        ),
        dim=1,
    )
    object_model = MeshObjectModel(
        candidates[0].mesh_path,
        batch_size=len(candidates),
        scale=candidates[0].object_scale,
        num_surface_samples=object_surface_samples,
        device=device,
        dtype=torch.float64,
        seed=0,
    )
    batch_ids = torch.arange(len(candidates), device=device)
    admitted = torch.zeros(
        (len(candidates), len(hand.backend.link_names)),
        dtype=torch.bool,
        device=device,
    )
    if args.no_force_closing:
        full_target = raw_actuator.detach().clone()
    else:
        pose = torch.cat((pose_prefix, raw_actuator), dim=1).requires_grad_()
        hand.set_parameters(pose, contact_indices)
        collision_points = hand.collision_points_world()
        query = object_model.query(collision_points)
        if hand.contact_points is None:
            raise X2ValidationError("X2 selected contacts were not materialized for closing")
        contact_query = object_model.query(hand.contact_points)
        link_index = {name: index for index, name in enumerate(hand.backend.link_names)}
        contact_link_indices = torch.as_tensor(
            [
                [link_index[authored_candidates[index].link_name] for index in row]
                for row in contact_indices.detach().cpu().tolist()
            ],
            dtype=torch.long,
            device=device,
        )
        selected_points: list[torch.Tensor] = []
        selected_normals: list[torch.Tensor] = []
        selected_distances: list[torch.Tensor] = []
        offset = 0
        for current_link_index, link_name in enumerate(hand.backend.link_names):
            count = len(hand.backend.collision_surface_samples_local[link_name])
            link_distances = query.signed_distance[:, offset : offset + count]
            local_ids = link_distances.argmax(dim=1)
            point_ids = local_ids + offset
            sampled_point = collision_points[batch_ids, point_ids]
            sampled_normal = query.outward_normals[batch_ids, point_ids]
            sampled_distance = query.signed_distance[batch_ids, point_ids]
            contact_distance = torch.where(
                contact_link_indices == current_link_index,
                contact_query.signed_distance,
                torch.full_like(contact_query.signed_distance, -torch.inf),
            )
            candidate_distances = torch.cat(
                (sampled_distance[:, None], contact_distance), dim=1
            )
            best = candidate_distances.argmax(dim=1)
            candidate_points = torch.cat(
                (sampled_point[:, None, :], hand.contact_points), dim=1
            )
            candidate_normals = torch.cat(
                (sampled_normal[:, None, :], contact_query.outward_normals), dim=1
            )
            selected_points.append(candidate_points[batch_ids, best])
            selected_normals.append(candidate_normals[batch_ids, best])
            selected_distances.append(candidate_distances[batch_ids, best])
            offset += count
        if offset != collision_points.shape[1]:
            raise X2ValidationError("X2 collision-point/link layout changed during closing")
        points = torch.stack(selected_points, dim=1)
        normals = torch.stack(selected_normals, dim=1)
        distances = torch.stack(selected_distances, dim=1)
        admitted = -distances < float(args.closing_contact_threshold)
        target_points = points + normals * float(args.closing_displacement)
        loss = (
            (target_points.detach() - points).square()
            * admitted.unsqueeze(-1).to(points.dtype)
        ).sum()
        loss.backward()
        if pose.grad is None:
            raise X2ValidationError("contact-gradient closing did not produce actuator gradients")
        actuator_gradient = pose.grad[:, 9:]
        full_target = raw_actuator + actuator_gradient * float(
            args.closing_gradient_scale
        )

    raw_penetration_cap = min(float(args.penetration_threshold), 0.001)
    target_penetration_cap = float(args.closing_penetration_cap)
    evaluation_alphas = (
        CLOSING_LINE_SEARCH_ALPHAS if not args.no_force_closing else (0.0,)
    )
    trial_targets: list[torch.Tensor] = []
    forward_maxima: list[torch.Tensor] = []
    reverse_maxima: list[torch.Tensor] = []
    bidirectional_maxima: list[torch.Tensor] = []
    trial_finite: list[torch.Tensor] = []
    actuator_limits_passed: list[torch.Tensor] = []
    actuator_limits = hand.backend.actuator_limits_tensor.to(raw_actuator)
    for alpha in evaluation_alphas:
        # PhysX consumes float32 targets.  Quantize first, promote back to the
        # differentiable model's float64 audit dtype, and only then accept.
        if float(alpha) == 0.0:
            unquantized_trial = raw_actuator
        else:
            unquantized_trial = raw_actuator + float(alpha) * (
                full_target.detach() - raw_actuator
            )
        trial = unquantized_trial.to(torch.float32).to(torch.float64)
        with torch.no_grad():
            hand.set_parameters(torch.cat((pose_prefix, trial), dim=1), contact_indices)
            hand_points = hand.collision_points_world()
            _, forward_maximum = object_model.penetration_summary(hand_points)
            reverse_maximum = torch.relu(
                hand.cal_distance(object_model.surface_points_tensor)
            ).amax(dim=-1)
            bidirectional_maximum = torch.maximum(
                forward_maximum, reverse_maximum
            )
            state_finite = (
                torch.isfinite(trial).all(dim=-1)
                & torch.isfinite(hand_points).all(dim=(1, 2))
                & torch.isfinite(forward_maximum)
                & torch.isfinite(reverse_maximum)
                & torch.isfinite(bidirectional_maximum)
            )
            limits_passed = (
                (trial >= actuator_limits[:, 0])
                & (trial <= actuator_limits[:, 1])
            ).all(dim=-1)
        trial_targets.append(trial.detach())
        forward_maxima.append(forward_maximum.detach())
        reverse_maxima.append(reverse_maximum.detach())
        bidirectional_maxima.append(bidirectional_maximum.detach())
        trial_finite.append(state_finite.detach())
        actuator_limits_passed.append(limits_passed.detach())

    targets_by_alpha = torch.stack(trial_targets, dim=1)
    forward_by_alpha = torch.stack(forward_maxima, dim=1)
    reverse_by_alpha = torch.stack(reverse_maxima, dim=1)
    maximum_by_alpha = torch.stack(bidirectional_maxima, dim=1)
    finite_by_alpha = torch.stack(trial_finite, dim=1)
    limits_by_alpha = torch.stack(actuator_limits_passed, dim=1)
    if args.no_force_closing:
        selected_indices = torch.zeros(
            len(candidates), dtype=torch.long, device=device
        )
        selected_alphas = np.zeros(len(candidates), dtype=np.float64)
        raw_penetration_passed = (
            finite_by_alpha[:, 0]
            & torch.isfinite(maximum_by_alpha[:, 0])
            & (maximum_by_alpha[:, 0] < raw_penetration_cap)
        ).detach().cpu().numpy()
        raw_state_finite = finite_by_alpha[:, 0].detach().cpu().numpy()
        raw_limits_passed = limits_by_alpha[:, 0].detach().cpu().numpy()
        raw_static_gate_passed = (
            raw_penetration_passed & raw_state_finite & raw_limits_passed
        )
        raw_index = 0
    else:
        selection = select_collision_aware_closing(
            maximum_by_alpha.detach().cpu().numpy(),
            finite_by_alpha.detach().cpu().numpy(),
            limits_by_alpha.detach().cpu().numpy(),
            raw_penetration_cap=raw_penetration_cap,
            target_penetration_cap=target_penetration_cap,
            alphas=evaluation_alphas,
        )
        selected_indices = torch.as_tensor(
            selection.selected_indices, dtype=torch.long, device=device
        )
        selected_alphas = selection.selected_alphas
        raw_penetration_passed = selection.raw_penetration_passed
        raw_state_finite = selection.raw_state_finite
        raw_limits_passed = selection.raw_actuator_limits_passed
        raw_static_gate_passed = selection.raw_static_gate_passed
        raw_index = len(evaluation_alphas) - 1
    target = targets_by_alpha[batch_ids, selected_indices]

    def measurement(tensor: torch.Tensor, row: int, column: int) -> float | None:
        value = float(tensor[row, column].item())
        return value if math.isfinite(value) else None

    def scalar_measurement(value: torch.Tensor) -> float | None:
        numeric = float(value.item())
        return numeric if math.isfinite(numeric) else None

    samples: list[dict[str, Any]] = []
    positive_trial_count = 0 if args.no_force_closing else len(evaluation_alphas) - 1
    for row in range(len(candidates)):
        selected_index = int(selected_indices[row].item())
        positive_finite = finite_by_alpha[row, :positive_trial_count]
        positive_limits = limits_by_alpha[row, :positive_trial_count]
        positive_penetration = maximum_by_alpha[row, :positive_trial_count]
        nonfinite_rejections = int((~positive_finite).sum().item())
        limit_rejections = int(
            (positive_finite & ~positive_limits).sum().item()
        )
        penetration_rejections = int(
            (
                positive_finite
                & positive_limits
                & ~(positive_penetration < target_penetration_cap)
            ).sum().item()
        )
        full_index = 0
        raw_maximum = measurement(maximum_by_alpha, row, raw_index)
        selected_state_finite = bool(finite_by_alpha[row, selected_index].item())
        selected_limits_passed = bool(limits_by_alpha[row, selected_index].item())
        selected_penetration = measurement(
            maximum_by_alpha, row, selected_index
        )
        selected_penetration_passed = bool(
            selected_penetration is not None
            and selected_penetration < target_penetration_cap
        )
        samples.append(
            {
                "sample_index_in_batch": row,
                "enabled": not args.no_force_closing,
                "raw_penetration_cap_m": raw_penetration_cap,
                "target_penetration_cap_m": target_penetration_cap,
                "positive_alphas": (
                    [float(value) for value in CLOSING_LINE_SEARCH_ALPHAS[:-1]]
                    if not args.no_force_closing
                    else []
                ),
                "bidirectional_sampled_penetration": True,
                "selected_alpha": float(selected_alphas[row]),
                "fell_back_to_raw": bool(selected_alphas[row] == 0.0),
                "raw_above_raw_cap": bool(
                    raw_maximum is not None
                    and raw_maximum >= raw_penetration_cap
                ),
                "raw_state_finite": bool(raw_state_finite[row]),
                "raw_actuator_limits_passed": bool(raw_limits_passed[row]),
                "raw_penetration_passed": bool(raw_penetration_passed[row]),
                "raw_static_gate_passed": bool(raw_static_gate_passed[row]),
                "selected_state_finite": selected_state_finite,
                "selected_actuator_limits_passed": selected_limits_passed,
                "selected_penetration_passed": selected_penetration_passed,
                "selected_target_safe": bool(
                    raw_static_gate_passed[row]
                    and selected_state_finite
                    and selected_limits_passed
                    and selected_penetration_passed
                ),
                "raw_forward_maximum_penetration_m": measurement(
                    forward_by_alpha, row, raw_index
                ),
                "raw_reverse_maximum_penetration_m": measurement(
                    reverse_by_alpha, row, raw_index
                ),
                "raw_maximum_penetration_m": raw_maximum,
                "full_target_forward_maximum_penetration_m": measurement(
                    forward_by_alpha, row, full_index
                ),
                "full_target_reverse_maximum_penetration_m": measurement(
                    reverse_by_alpha, row, full_index
                ),
                "full_target_maximum_penetration_m": measurement(
                    maximum_by_alpha, row, full_index
                ),
                "selected_forward_maximum_penetration_m": measurement(
                    forward_by_alpha, row, selected_index
                ),
                "selected_reverse_maximum_penetration_m": measurement(
                    reverse_by_alpha, row, selected_index
                ),
                "selected_maximum_penetration_m": measurement(
                    maximum_by_alpha, row, selected_index
                ),
                "full_target_maximum_actuator_adjustment_rad": scalar_measurement(
                    (full_target[row] - raw_actuator[row]).abs().amax()
                ),
                "selected_maximum_actuator_adjustment_rad": scalar_measurement(
                    (target[row] - raw_actuator[row]).abs().amax()
                ),
                "admitted_link_count": int(admitted[row].sum().item()),
                "positive_trial_count": positive_trial_count,
                "rejected_positive_trial_count": (
                    selected_index
                    if not args.no_force_closing and selected_index < raw_index
                    else positive_trial_count
                ),
                "rejection_counts": {
                    "nonfinite": nonfinite_rejections,
                    "actuator_limits": limit_rejections,
                    "target_penetration_cap": penetration_rejections,
                },
                "float32_quantized_and_rechecked": True,
            }
        )

    point_counts = {
        link_name: len(hand.backend.collision_surface_samples_local[link_name])
        for link_name in hand.backend.link_names
    }
    return target.detach(), {
        "enabled": not args.no_force_closing,
        "mode": (
            "raw_target" if args.no_force_closing else "collision_aware_line_search"
        ),
        "contact_threshold_m": float(args.closing_contact_threshold),
        "target_displacement_m": float(args.closing_displacement),
        "gradient_scale": float(args.closing_gradient_scale),
        "raw_penetration_cap_m": raw_penetration_cap,
        "target_penetration_cap_m": target_penetration_cap,
        "positive_alphas": (
            [float(value) for value in CLOSING_LINE_SEARCH_ALPHAS[:-1]]
            if not args.no_force_closing
            else []
        ),
        "float32_quantized_and_rechecked": True,
        "bidirectional_sampled_penetration": True,
        "hand_surface_sampling": {
            "configured_samples_per_link": collision_samples_per_link,
            "actual_point_count_per_link": point_counts,
            "total_point_count": sum(point_counts.values()),
        },
        "object_surface_sampling": {
            "point_count": object_surface_samples,
            "seed": 0,
        },
        "samples": samples,
    }


def _audit_contact_fk(
    *,
    candidates: Sequence[X2RawCandidate],
    robot,
    env_origins,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Use authored contacts to catch frame/quaternion/joint replay errors."""

    import torch

    body_index = {name: index for index, name in enumerate(robot.body_names)}
    body_pos = _as_torch(robot.data.body_pos_w)
    body_quat = _as_torch(robot.data.body_quat_w)
    max_position_error = 0.0
    minimum_normal_dot = 1.0
    for sample_index, candidate in enumerate(candidates):
        env_id = sample_index * len(GRAVITY_TESTS_WXYZ)
        origin = env_origins[env_id]
        for contact in candidate.record["selected_contacts"]:
            link_name = contact.get("link_name")
            if link_name not in body_index:
                raise X2ValidationError(f"unknown contact link in {candidate.path}: {link_name}")
            link_id = body_index[link_name]
            local_position = torch.as_tensor(
                contact["local_position"], dtype=torch.float32, device=body_pos.device
            )
            actual_position = body_pos[env_id, link_id] + _quat_apply_xyzw(
                torch, body_quat[env_id, link_id], local_position
            )
            expected_position = torch.as_tensor(
                np.asarray(contact["world_position"], dtype=np.float64),
                dtype=torch.float32,
                device=body_pos.device,
            ) + origin
            position_error = float(torch.linalg.vector_norm(actual_position - expected_position).item())
            max_position_error = max(max_position_error, position_error)

            local_normal = torch.as_tensor(
                contact["local_surface_normal"], dtype=torch.float32, device=body_pos.device
            )
            actual_normal = _quat_apply_xyzw(
                torch, body_quat[env_id, link_id], local_normal
            )
            expected_normal = torch.as_tensor(
                np.asarray(contact["world_surface_normal"], dtype=np.float64),
                dtype=torch.float32,
                device=body_pos.device,
            )
            actual_normal = actual_normal / torch.linalg.vector_norm(actual_normal)
            expected_normal = expected_normal / torch.linalg.vector_norm(expected_normal)
            normal_dot = float(torch.dot(actual_normal, expected_normal).item())
            minimum_normal_dot = min(minimum_normal_dot, normal_dot)
    if max_position_error > args.fk_position_tolerance:
        raise X2ValidationError(
            "X2 runtime FK/contact frame preflight failed: "
            f"max position error {max_position_error:.9g} m > {args.fk_position_tolerance:.9g} m"
        )
    if minimum_normal_dot < args.fk_normal_min_dot:
        raise X2ValidationError(
            "X2 runtime FK/contact normal preflight failed: "
            f"minimum dot {minimum_normal_dot:.9g} < {args.fk_normal_min_dot:.9g}"
        )
    return {
        "maximum_selected_contact_position_error_m": max_position_error,
        "minimum_selected_contact_normal_dot": minimum_normal_dot,
    }


def _read_raw_object_contacts(contact_sensor, *, env_count: int, dt: float):
    """Return exact per-object contact existence and maximum patch force.

    ``get_raw_contact_data`` reports every contact patch for each sensing body
    without a filter.  This validator deliberately creates no ground or other
    assets, so every other actor touching Object is an X2 hand body.
    """

    import torch
    import warp as wp

    contact_view = contact_sensor.contact_view
    raw = contact_view.get_raw_contact_data(dt)
    if len(raw) != 7:
        raise X2ValidationError(
            f"PhysX raw contact API returned {len(raw)} buffers instead of 7"
        )
    force_buffer, _, _, _, count_buffer, start_buffer, _ = raw
    forces = wp.to_torch(force_buffer).reshape(-1).abs()
    counts = wp.to_torch(count_buffer).reshape(-1).to(dtype=torch.long)
    starts = wp.to_torch(start_buffer).reshape(-1).to(dtype=torch.long)
    if counts.shape != (env_count,) or starts.shape != (env_count,):
        raise X2ValidationError(
            "unexpected raw object contact count/start shapes: "
            f"counts={tuple(counts.shape)}, starts={tuple(starts.shape)}, envs={env_count}"
        )
    if bool((counts < 0).any().item()) or bool((starts < 0).any().item()):
        raise X2ValidationError("PhysX raw contact buffers contain negative indices")
    if bool(((starts + counts) > forces.numel()).any().item()):
        raise X2ValidationError("PhysX raw contact buffers exceed their force storage")

    has_contact = counts > 0
    maximum_force = torch.zeros(env_count, dtype=forces.dtype, device=forces.device)
    for env_id in range(env_count):
        count = int(counts[env_id].item())
        if count:
            start = int(starts[env_id].item())
            maximum_force[env_id] = forces[start : start + count].amax()
    return has_contact, maximum_force


def _validate_batch(
    *,
    candidates: Sequence[X2RawCandidate],
    capacity_samples: int,
    scene,
    sim,
    active_joint_ids: Sequence[int],
    mimic_joint_pairs: Sequence[tuple[int, int]],
    thresholds: ValidationThresholds,
    args: argparse.Namespace,
) -> tuple[list[list[OrientationOutcome]], dict[str, float]]:
    import torch

    robot = scene["robot"]
    rigid_object = scene["object"]
    contact_sensor = scene["object_contact"]
    closing_targets, closing_audit = _compute_contact_closing_targets(
        candidates, device=sim.device, args=args
    )
    (
        hand_pose,
        object_pose,
        raw_joint_position,
        actuator_target,
        gravity,
        active_env_count,
    ) = _prepare_batch(
        candidates,
        capacity_samples=capacity_samples,
        env_origins=scene.env_origins,
        device=sim.device,
        closing_actuator_targets=closing_targets,
    )
    zero_object_velocity = torch.zeros((object_pose.shape[0], 6), device=sim.device)
    zero_hand_velocity = torch.zeros((hand_pose.shape[0], 6), device=sim.device)
    zero_joint_velocity = torch.zeros_like(raw_joint_position)

    rigid_object.write_root_pose_to_sim_index(root_pose=object_pose)
    rigid_object.write_root_velocity_to_sim_index(root_velocity=zero_object_velocity)
    robot.write_root_pose_to_sim_index(root_pose=hand_pose)
    robot.write_root_velocity_to_sim_index(root_velocity=zero_hand_velocity)
    # Replay the raw JSON state for both the FK audit and the simulation start.
    # The contact-gradient result remains a drive target approached dynamically.
    robot.write_joint_position_to_sim_index(position=raw_joint_position)
    robot.write_joint_velocity_to_sim_index(velocity=zero_joint_velocity)
    robot.set_joint_position_target_index(target=actuator_target, joint_ids=list(active_joint_ids))
    scene.reset()
    sim.forward()
    scene.update(sim.get_physics_dt())

    fk_audit = _audit_contact_fk(
        candidates=candidates,
        robot=robot,
        env_origins=scene.env_origins,
        args=args,
    )

    body_mass = _as_torch(rigid_object.data.body_mass)
    body_inertia = _as_torch(rigid_object.data.body_inertia)
    active_mass = body_mass[:active_env_count].reshape(active_env_count, -1)
    active_inertia = body_inertia[:active_env_count].reshape(active_env_count, -1)
    if not bool(torch.isfinite(active_mass).all().item()) or not bool(
        torch.isfinite(active_inertia).all().item()
    ):
        raise X2ValidationError("PhysX produced non-finite object mass or inertia")
    if not torch.allclose(active_mass, active_mass[:1].expand_as(active_mass)) or not torch.allclose(
        active_inertia, active_inertia[:1].expand_as(active_inertia)
    ):
        raise X2ValidationError("cloned object environments disagree on mass or inertia")
    object_dynamics_audit = {
        "body_mass_kg": float(active_mass[0, 0].item()),
        "body_inertia_kg_m2_flat": active_inertia[0].detach().cpu().tolist(),
    }
    initial_position = object_pose[:, :3].clone()
    maximum_displacement = torch.zeros(object_pose.shape[0], device=sim.device)
    maximum_joint_error = torch.zeros(object_pose.shape[0], device=sim.device)
    maximum_mimic_error = torch.zeros(object_pose.shape[0], device=sim.device)
    finite = torch.ones(object_pose.shape[0], dtype=torch.bool, device=sim.device)
    first_nonfinite_phase = torch.full(
        (object_pose.shape[0],), -1, dtype=torch.int8, device=sim.device
    )
    first_nonfinite_step = torch.full(
        (object_pose.shape[0],), -1, dtype=torch.int32, device=sim.device
    )
    first_nonfinite_components = torch.zeros(
        (object_pose.shape[0], 4), dtype=torch.bool, device=sim.device
    )

    # This optional setup is deliberately separate from the formal 100-step
    # gravity test.  It starts from raw JSON q, applies no gravity, ramps rather
    # than teleports the contact-gradient target, and still charges any object
    # motion/non-finite state/mimic error to the final validation outcome.
    forces = body_mass.unsqueeze(-1) * gravity.unsqueeze(1)
    if forces.ndim != 3 or forces.shape[-1] != 3:
        raise X2ValidationError(
            "per-environment object wrench must have shape (environment, body, 3), "
            f"got {tuple(forces.shape)}"
        )
    zero_forces = torch.zeros_like(forces)
    rigid_object.permanent_wrench_composer.set_forces_and_torques_index(
        forces=zero_forces,
        torques=zero_forces,
        is_global=True,
    )
    raw_actuator_position = raw_joint_position[:, list(active_joint_ids)]
    for preclose_step in range(args.preclose_physics_steps):
        progress = float(preclose_step + 1) / float(args.preclose_physics_steps)
        ramp_target = raw_actuator_position + progress * (
            actuator_target - raw_actuator_position
        )
        robot.set_joint_position_target_index(
            target=ramp_target, joint_ids=list(active_joint_ids)
        )
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())
        root_pose = _as_torch(rigid_object.data.root_pose_w)
        root_velocity = _as_torch(rigid_object.data.root_vel_w)
        joint_pos = _as_torch(robot.data.joint_pos)
        joint_vel = _as_torch(robot.data.joint_vel)
        component_finite = torch.stack(
            (
                torch.isfinite(root_pose).all(dim=-1),
                torch.isfinite(root_velocity).all(dim=-1),
                torch.isfinite(joint_pos).all(dim=-1),
                torch.isfinite(joint_vel).all(dim=-1),
            ),
            dim=-1,
        )
        step_finite = component_finite.all(dim=-1)
        newly_nonfinite = (first_nonfinite_phase < 0) & ~step_finite
        first_nonfinite_phase = torch.where(
            newly_nonfinite,
            torch.zeros_like(first_nonfinite_phase),
            first_nonfinite_phase,
        )
        first_nonfinite_step = torch.where(
            newly_nonfinite,
            torch.full_like(first_nonfinite_step, preclose_step + 1),
            first_nonfinite_step,
        )
        first_nonfinite_components = torch.where(
            newly_nonfinite[:, None],
            ~component_finite,
            first_nonfinite_components,
        )
        finite &= step_finite
        displacement = torch.linalg.vector_norm(
            root_pose[:, :3] - initial_position, dim=-1
        )
        displacement = torch.where(
            torch.isfinite(displacement),
            displacement,
            torch.full_like(displacement, math.inf),
        )
        maximum_displacement = torch.maximum(maximum_displacement, displacement)
        mimic_error = torch.stack(
            [
                (joint_pos[:, follower_id] - joint_pos[:, driver_id]).abs()
                for follower_id, driver_id in mimic_joint_pairs
            ],
            dim=-1,
        ).amax(dim=-1)
        mimic_error = torch.where(
            torch.isfinite(mimic_error),
            mimic_error,
            torch.full_like(mimic_error, math.inf),
        )
        maximum_mimic_error = torch.maximum(maximum_mimic_error, mimic_error)
    preclose_maximum_displacement = float(
        maximum_displacement[:active_env_count].amax().item()
    )
    preclose_audit = {
        "enabled": args.preclose_physics_steps > 0,
        "physics_step_count": args.preclose_physics_steps,
        "duration_s": args.preclose_physics_steps * sim.get_physics_dt(),
        "gravity_magnitude_m_s2": 0.0,
        "target_schedule": "linear_raw_to_final_actuator_target",
        "maximum_object_displacement_m": (
            preclose_maximum_displacement
            if math.isfinite(preclose_maximum_displacement)
            else None
        ),
        "nonfinite_orientation_count": int(
            (~finite[:active_env_count]).sum().item()
        ),
    }

    torques = torch.zeros_like(forces)
    rigid_object.permanent_wrench_composer.set_forces_and_torques_index(
        forces=forces,
        torques=torques,
        is_global=True,
    )

    for validation_step in range(args.sim_steps * args.substeps):
        robot.set_joint_position_target_index(target=actuator_target, joint_ids=list(active_joint_ids))
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())
        root_pose = _as_torch(rigid_object.data.root_pose_w)
        root_velocity = _as_torch(rigid_object.data.root_vel_w)
        joint_pos = _as_torch(robot.data.joint_pos)
        joint_vel = _as_torch(robot.data.joint_vel)
        component_finite = torch.stack(
            (
                torch.isfinite(root_pose).all(dim=-1),
                torch.isfinite(root_velocity).all(dim=-1),
                torch.isfinite(joint_pos).all(dim=-1),
                torch.isfinite(joint_vel).all(dim=-1),
            ),
            dim=-1,
        )
        step_finite = component_finite.all(dim=-1)
        newly_nonfinite = (first_nonfinite_phase < 0) & ~step_finite
        first_nonfinite_phase = torch.where(
            newly_nonfinite,
            torch.ones_like(first_nonfinite_phase),
            first_nonfinite_phase,
        )
        first_nonfinite_step = torch.where(
            newly_nonfinite,
            torch.full_like(first_nonfinite_step, validation_step + 1),
            first_nonfinite_step,
        )
        first_nonfinite_components = torch.where(
            newly_nonfinite[:, None],
            ~component_finite,
            first_nonfinite_components,
        )
        finite &= step_finite
        displacement = torch.linalg.vector_norm(root_pose[:, :3] - initial_position, dim=-1)
        displacement = torch.where(
            torch.isfinite(displacement), displacement, torch.full_like(displacement, math.inf)
        )
        maximum_displacement = torch.maximum(maximum_displacement, displacement)
        active_error = (joint_pos[:, list(active_joint_ids)] - actuator_target).abs().amax(dim=-1)
        active_error = torch.where(
            torch.isfinite(active_error), active_error, torch.full_like(active_error, math.inf)
        )
        maximum_joint_error = torch.maximum(maximum_joint_error, active_error)
        mimic_error = torch.stack(
            [
                (joint_pos[:, follower_id] - joint_pos[:, driver_id]).abs()
                for follower_id, driver_id in mimic_joint_pairs
            ],
            dim=-1,
        ).amax(dim=-1)
        mimic_error = torch.where(
            torch.isfinite(mimic_error), mimic_error, torch.full_like(mimic_error, math.inf)
        )
        maximum_mimic_error = torch.maximum(maximum_mimic_error, mimic_error)

    final_position = _as_torch(rigid_object.data.root_pos_w)
    final_displacement = torch.linalg.vector_norm(final_position - initial_position, dim=-1)
    hand_object_contact, final_contact_force = _read_raw_object_contacts(
        contact_sensor,
        env_count=object_pose.shape[0],
        dt=sim.get_physics_dt(),
    )
    mimic_audit = summarize_newton_mimic_errors(
        maximum_mimic_error[:active_env_count].detach().cpu().tolist(),
        threshold=args.mimic_error_threshold,
    )
    phase_values = first_nonfinite_phase[:active_env_count].detach().cpu().tolist()
    step_values = first_nonfinite_step[:active_env_count].detach().cpu().tolist()
    component_values = (
        first_nonfinite_components[:active_env_count].detach().cpu().tolist()
    )
    component_names = ("object_root_pose", "object_root_velocity", "joint_position", "joint_velocity")
    first_nonfinite_events = []
    for env_id, phase_code in enumerate(phase_values):
        if phase_code < 0:
            continue
        direction_index = env_id % len(GRAVITY_TESTS_WXYZ)
        first_nonfinite_events.append(
            {
                "sample_index_in_batch": env_id // len(GRAVITY_TESTS_WXYZ),
                "orientation": GRAVITY_TESTS_WXYZ[direction_index][0],
                "phase": "zero_gravity_preclose" if phase_code == 0 else "gravity_validation",
                "physics_step_index_1_based": int(step_values[env_id]),
                "nonfinite_components": [
                    name
                    for name, nonfinite in zip(
                        component_names, component_values[env_id]
                    )
                    if nonfinite
                ],
            }
        )

    results: list[list[OrientationOutcome]] = []
    for sample_index in range(len(candidates)):
        sample_outcomes: list[OrientationOutcome] = []
        for direction_index, (name, _) in enumerate(GRAVITY_TESTS_WXYZ):
            env_id = sample_index * len(GRAVITY_TESTS_WXYZ) + direction_index
            if env_id >= active_env_count:
                raise AssertionError("active environment indexing error")
            sample_outcomes.append(
                evaluate_orientation(
                    name=name,
                    final_displacement=float(final_displacement[env_id].item()),
                    maximum_displacement=float(maximum_displacement[env_id].item()),
                    final_contact_force=float(final_contact_force[env_id].item()),
                    maximum_active_joint_error=float(maximum_joint_error[env_id].item()),
                    maximum_newton_mimic_error=float(
                        maximum_mimic_error[env_id].item()
                    ),
                    finite=bool(finite[env_id].item()),
                    thresholds=thresholds,
                    criterion=args.criterion,
                    hand_object_contact=bool(hand_object_contact[env_id].item()),
                    gravity_vector_object_frame=gravity[env_id].detach().cpu().tolist(),
                )
            )
        results.append(sample_outcomes)
    return results, {
        **fk_audit,
        "object_dynamics": object_dynamics_audit,
        "contact_gradient_closing": closing_audit,
        "zero_gravity_preclose": preclose_audit,
        "numerical_stability": {
            "nonfinite_orientation_count": len(first_nonfinite_events),
            "first_nonfinite_events": first_nonfinite_events,
        },
        **mimic_audit,
    }


def _runtime_metadata(
    *,
    args: argparse.Namespace,
    hand_usd: Path,
    object_usd: Path,
    mesh_cache_key: str,
    bounds_audit: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from isaaclab.utils.version import get_isaac_sim_version

    return {
        "isaac_sim_version": str(get_isaac_sim_version()),
        "isaaclab_version": _package_version("isaaclab"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(args.device),
        "dt_s": args.dt,
        "physics_dt_s": args.dt / args.substeps,
        "substeps": args.substeps,
        "simulation_steps": args.sim_steps,
        "physics_step_count": args.sim_steps * args.substeps,
        "simulated_duration_s": args.dt * args.sim_steps,
        "gravity_implementation": "zero_global_gravity_plus_per_environment_com_force",
        "gravity_magnitude_m_s2": 9.8,
        "physx_solver": {
            "solver_type": args.solver_type,
            "external_forces_every_iteration": args.external_forces_every_iteration,
            "solve_articulation_contact_last": args.solve_articulation_contact_last,
        },
        "replay_frame": "object_centered_generator_frame",
        "object_pose": "identity_at_environment_origin",
        "hand_pose": "raw_json_hand_pose_in_object_frame",
        "initial_dof_state": "raw_json_joint_state",
        "actuator_drive": {
            "stiffness_n_m_per_rad": args.actuator_stiffness,
            "damping_n_m_s_per_rad": args.actuator_damping,
            "armature_kg_m2": args.actuator_armature,
            "stiffness_source": (
                "hand_usd" if args.actuator_stiffness is None else "cli_override"
            ),
            "damping_source": (
                "hand_usd" if args.actuator_damping is None else "cli_override"
            ),
            "armature_source": (
                "hand_usd" if args.actuator_armature is None else "cli_override"
            ),
            "effort_limit_source": "hand_usd",
            "velocity_limit_source": "hand_usd",
        },
        "actuator_target_state": (
            "raw_json_actuator_state"
            if args.no_force_closing
            else "collision_aware_contact_gradient_line_search"
        ),
        "hand_usd": str(hand_usd),
        "object_usd": str(object_usd),
        "mesh_cache_key": mesh_cache_key,
        "collision_approximation": args.collision_approximation,
        "density_kg_m3": args.density,
        "hand_friction": args.hand_friction,
        "object_friction": args.object_friction,
        "contact_offset_m": args.contact_offset,
        "rest_offset_m": args.rest_offset,
        "contact_detection": "physx_raw_object_contact_count",
        "controlled_contact_scene": "one_object_plus_one_x2_robot_no_ground",
        "hand_rigid_body_count": len(HAND_RIGID_BODY_RELATIVE_PATHS),
        "enabled_hand_collision_shape_count": len(HAND_COLLISION_SHAPE_RELATIVE_PATHS),
        "physx_articulation_self_collision_enabled": False,
        "static_self_collision_oracle": (
            "generator_deterministic_bidirectional_sampled_collision_hull"
        ),
        "maximum_contact_records_per_object_environment": MAX_CONTACT_DATA_PER_OBJECT,
        "mimic_runtime": "newton_mimic_api_consumed_by_physx",
        "maximum_allowed_mimic_error_rad": args.mimic_error_threshold,
        "contact_gradient_closing": {
            "enabled": not args.no_force_closing,
            "mode": (
                "raw_target"
                if args.no_force_closing
                else "collision_aware_line_search"
            ),
            "contact_threshold_m": args.closing_contact_threshold,
            "target_displacement_m": args.closing_displacement,
            "gradient_scale": args.closing_gradient_scale,
            "raw_penetration_cap_m": min(args.penetration_threshold, 0.001),
            "target_penetration_cap_m": args.closing_penetration_cap,
            "maximum_allowed_target_penetration_cap_m": (
                MAX_CLOSING_TARGET_PENETRATION_CAP
            ),
            "positive_alphas": (
                [float(value) for value in CLOSING_LINE_SEARCH_ALPHAS[:-1]]
                if not args.no_force_closing
                else []
            ),
            "bidirectional_sampled_penetration": True,
            "float32_quantized_and_rechecked": True,
            "raw_json_remains_physx_initial_state": True,
        },
        "zero_gravity_preclose": {
            "enabled": args.preclose_physics_steps > 0,
            "physics_step_count": args.preclose_physics_steps,
            "duration_s": args.preclose_physics_steps * args.dt / args.substeps,
            "target_schedule": "linear_raw_to_final_actuator_target",
            "validation_physics_step_count_unchanged": args.sim_steps * args.substeps,
            "displacement_reference": "raw_json_object_position_before_preclose",
        },
        "bounds_audit": bounds_audit,
    }


def _make_summary(
    *,
    candidates: Sequence[X2RawCandidate],
    records: Sequence[dict[str, Any]],
    outputs: Sequence[Path],
    mesh_path: Path,
    scale: float,
    args: argparse.Namespace,
    fk_audits: Sequence[dict[str, Any]],
    skipped_count: int,
) -> dict[str, Any]:
    passed = sum(record["validation"]["status"] == "passed" for record in records)
    side_summary: dict[str, dict[str, int]] = {}
    for side in ("front", "back"):
        indices = [i for i, candidate in enumerate(candidates) if candidate.active_side == side]
        if indices:
            side_summary[side] = {
                "total": len(indices),
                "passed": sum(records[i]["validation"]["status"] == "passed" for i in indices),
                "failed": sum(records[i]["validation"]["status"] == "failed" for i in indices),
            }
    max_fk_error = max(
        (audit["maximum_selected_contact_position_error_m"] for audit in fk_audits), default=0.0
    )
    min_fk_dot = min(
        (audit["minimum_selected_contact_normal_dot"] for audit in fk_audits), default=1.0
    )
    finite_mimic_maxima = [
        float(audit["maximum_newton_mimic_error_rad"])
        for audit in fk_audits
        if audit["maximum_newton_mimic_error_rad"] is not None
    ]
    max_mimic_error = max(finite_mimic_maxima, default=None)
    mimic_finite_orientation_count = sum(
        int(audit["newton_mimic_finite_orientation_count"])
        for audit in fk_audits
    )
    mimic_nonfinite_orientation_count = sum(
        int(audit["newton_mimic_nonfinite_orientation_count"])
        for audit in fk_audits
    )
    mimic_nonfinite_sample_count = sum(
        int(audit["newton_mimic_nonfinite_sample_count"])
        for audit in fk_audits
    )
    mimic_violation_orientation_count = sum(
        int(audit["newton_mimic_violation_orientation_count"])
        for audit in fk_audits
    )
    mimic_violation_sample_count = sum(
        int(audit["newton_mimic_violation_sample_count"])
        for audit in fk_audits
    )
    object_dynamics = fk_audits[0]["object_dynamics"] if fk_audits else None
    if any(audit["object_dynamics"] != object_dynamics for audit in fk_audits):
        raise X2ValidationError("validation batches disagree on object mass or inertia")
    closing_samples = [
        sample
        for audit in fk_audits
        for sample in audit["contact_gradient_closing"]["samples"]
    ]
    if len(closing_samples) != len(candidates):
        raise X2ValidationError("collision-aware closing summary lost a candidate")
    selected_alpha_counts: dict[str, int] = {}
    for sample in closing_samples:
        label = format(float(sample["selected_alpha"]), ".9g")
        selected_alpha_counts[label] = selected_alpha_counts.get(label, 0) + 1
    return {
        "passed": True,
        "input_root": str(args.input_root.expanduser().resolve()),
        "mesh_path": str(mesh_path),
        "object_scale": scale,
        "criterion": args.criterion,
        "candidate_count": len(candidates),
        "skipped_existing_count": skipped_count,
        "valid_count": passed,
        "failed_count": len(candidates) - passed,
        "static_self_collision_gate_required_count": sum(
            candidate.self_collision_gate_required for candidate in candidates
        ),
        "static_self_collision_infeasible_count": sum(
            candidate.self_collision_gate_required
            and candidate.self_collision_feasible is False
            for candidate in candidates
        ),
        "dense_hand_object_gate_required_count": sum(
            candidate.hand_object_gate_required for candidate in candidates
        ),
        "dense_hand_object_infeasible_count": sum(
            candidate.hand_object_gate_required
            and candidate.hand_object_feasible is False
            for candidate in candidates
        ),
        "physx_articulation_self_collision_enabled": False,
        "static_self_collision_oracle": (
            "generator_deterministic_bidirectional_sampled_collision_hull"
        ),
        "side_summary": side_summary,
        "fk_preflight": {
            "maximum_selected_contact_position_error_m": max_fk_error,
            "minimum_selected_contact_normal_dot": min_fk_dot,
        },
        "object_dynamics": object_dynamics,
        "collision_aware_closing": {
            "raw_static_gate_failed_count": sum(
                not sample["raw_static_gate_passed"] for sample in closing_samples
            ),
            "selected_target_unsafe_count": sum(
                not sample["selected_target_safe"] for sample in closing_samples
            ),
            "raw_above_raw_cap_count": sum(
                sample["raw_above_raw_cap"] for sample in closing_samples
            ),
            "fell_back_to_raw_count": sum(
                sample["fell_back_to_raw"] for sample in closing_samples
            ),
            "selected_alpha_counts": selected_alpha_counts,
        },
        "maximum_newton_mimic_error_rad": max_mimic_error,
        "newton_mimic_finite_orientation_count": mimic_finite_orientation_count,
        "newton_mimic_nonfinite_orientation_count": mimic_nonfinite_orientation_count,
        "newton_mimic_nonfinite_sample_count": mimic_nonfinite_sample_count,
        "newton_mimic_violation_orientation_count": mimic_violation_orientation_count,
        "newton_mimic_violation_sample_count": mimic_violation_sample_count,
        "output_files": [str(path) for path in outputs],
    }


def _dry_run_summary(
    candidates: Sequence[X2RawCandidate],
    mesh_path: Path,
    scale: float,
    args: argparse.Namespace,
    *,
    skipped_count: int,
) -> dict[str, Any]:
    return {
        "passed": True,
        "dry_run": True,
        "candidate_count": len(candidates),
        "skipped_existing_count": skipped_count,
        "mesh_path": str(mesh_path),
        "object_scale": scale,
        "side_counts": {
            side: sum(candidate.active_side == side for candidate in candidates)
            for side in ("front", "back")
        },
        "would_create_valid_or_failed": len(candidates),
        "would_launch_isaac_sim": False,
        "physx_articulation_self_collision_enabled": False,
        "static_self_collision_oracle": (
            "generator_deterministic_bidirectional_sampled_collision_hull"
        ),
        "static_self_collision_gate_required_count": sum(
            candidate.self_collision_gate_required for candidate in candidates
        ),
        "static_self_collision_infeasible_count": sum(
            candidate.self_collision_gate_required
            and candidate.self_collision_feasible is False
            for candidate in candidates
        ),
        "dense_hand_object_gate_required_count": sum(
            candidate.hand_object_gate_required for candidate in candidates
        ),
        "dense_hand_object_infeasible_count": sum(
            candidate.hand_object_gate_required
            and candidate.hand_object_feasible is False
            for candidate in candidates
        ),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not -1.0 <= args.fk_normal_min_dot <= 1.0 or not math.isfinite(args.fk_normal_min_dot):
        parser.error("--fk-normal-min-dot must be finite and in [-1, 1]")
    if args.closing_penetration_cap > MAX_CLOSING_TARGET_PENETRATION_CAP:
        parser.error("--closing-penetration-cap must be at most 0.002 m")
    args.input_root = args.input_root.expanduser().resolve()
    mesh_filter = _resolve_existing_file(args.mesh_path, "mesh") if args.mesh_path else None
    hand_usd = _resolve_existing_file(args.hand_usd, "X2 hand USD")
    candidates = discover_raw_candidates(
        args.input_root,
        mesh_path=mesh_filter,
        side=args.side,
        limit=args.limit,
    )
    groups = group_candidates_by_mesh(candidates)
    if len(groups) != 1:
        descriptions = [f"{path}@{scale}" for path, scale in groups]
        raise X2ValidationError(
            "one validator invocation must contain exactly one mesh/scale group; "
            f"got {descriptions}. Pass --mesh-path or use the primitive dataset wrapper."
        )
    (mesh_path, scale), candidates = next(iter(groups.items()))
    skipped_count = 0
    if args.resume:
        pending = [candidate for candidate in candidates if not _has_validation_output(candidate)]
        skipped_count = len(candidates) - len(pending)
        candidates = pending
        if not candidates:
            summary = {
                "passed": True,
                "dry_run": bool(args.dry_run),
                "candidate_count": 0,
                "skipped_existing_count": skipped_count,
                "mesh_path": str(mesh_path),
                "object_scale": scale,
                "would_launch_isaac_sim": False,
                "physx_articulation_self_collision_enabled": False,
                "static_self_collision_oracle": (
                    "generator_deterministic_bidirectional_sampled_collision_hull"
                ),
            }
            if args.summary_json:
                _atomic_json(args.summary_json, summary)
            print(json.dumps(summary, indent=2, allow_nan=False))
            return
    _preflight_outputs(candidates, args.overwrite)
    thresholds = ValidationThresholds(
        penetration=args.penetration_threshold,
        retention_distance=args.retention_distance,
        contact_force=args.contact_force_threshold,
        joint_error=args.joint_error_threshold,
        mimic_error=args.mimic_error_threshold,
    )
    if args.dry_run:
        summary = _dry_run_summary(
            candidates, mesh_path, scale, args, skipped_count=skipped_count
        )
        if args.summary_json:
            _atomic_json(args.summary_json, summary)
        print(json.dumps(summary, indent=2, allow_nan=False))
        return

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext

        object_usd, mesh_cache_key = _convert_object_mesh(mesh_path, scale, args)
        bounds_audit = _audit_converted_bounds(mesh_path, scale, object_usd)
        capacity_samples = min(args.batch_size, len(candidates))
        scene_cfg = _make_scene_cfg(
            hand_usd=hand_usd,
            object_usd=object_usd,
            num_envs=capacity_samples * len(GRAVITY_TESTS_WXYZ),
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
        sim.reset()
        robot = scene["robot"]
        contact_sensor = scene["object_contact"]
        active_joint_ids, mimic_joint_pairs = _audit_runtime_mapping(
            robot,
            contact_sensor,
            expected_env_count=capacity_samples * len(GRAVITY_TESTS_WXYZ),
        )
        runtime = _runtime_metadata(
            args=args,
            hand_usd=hand_usd,
            object_usd=object_usd,
            mesh_cache_key=mesh_cache_key,
            bounds_audit=bounds_audit,
        )

        validated_records: list[dict[str, Any]] = []
        outputs: list[Path] = []
        fk_audits: list[dict[str, float]] = []
        contact_groups: dict[int, list[X2RawCandidate]] = {}
        for candidate in candidates:
            contact_count = len(candidate.record["selected_contact_ids"])
            contact_groups.setdefault(contact_count, []).append(candidate)
        processed_offset = 0
        for contact_count in sorted(contact_groups):
            contact_candidates = contact_groups[contact_count]
            for offset in range(0, len(contact_candidates), capacity_samples):
                chunk = contact_candidates[offset : offset + capacity_samples]
                print(
                    f"[physx] batch offset={processed_offset} count={len(chunk)} "
                    f"contacts={contact_count} total={len(candidates)}",
                    flush=True,
                )
                print(
                    f"[physx] contact-stratum={contact_count}",
                    flush=True,
                )
                try:
                    chunk_outcomes, fk_audit = _validate_batch(
                        candidates=chunk,
                        capacity_samples=capacity_samples,
                        scene=scene,
                        sim=sim,
                        active_joint_ids=active_joint_ids,
                        mimic_joint_pairs=mimic_joint_pairs,
                        thresholds=thresholds,
                        args=args,
                    )
                except BaseException as exc:
                    print(
                        json.dumps(
                            {
                                "physx_batch_error": type(exc).__name__,
                                "message": str(exc),
                                "offset": processed_offset,
                                "count": len(chunk),
                                "contact_count": contact_count,
                            },
                            allow_nan=False,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    raise
                fk_audits.append(fk_audit)
                observed_mimic_error = fk_audit["maximum_newton_mimic_error_rad"]
                mimic_error_text = (
                    f"{observed_mimic_error:.9g}"
                    if observed_mimic_error is not None
                    else "unobserved"
                )
                print(
                    f"[physx] batch offset={processed_offset} completed "
                    f"mimic_error={mimic_error_text}",
                    flush=True,
                )
                runtime_batch_audit = dict(fk_audit)
                runtime_batch_audit["contact_gradient_closing"] = (
                    compact_collision_aware_closing_batch_audit(
                        fk_audit["contact_gradient_closing"]
                    )
                )
                batch_runtime = {**runtime, "batch_audit": runtime_batch_audit}
                closing_samples = fk_audit["contact_gradient_closing"]["samples"]
                if len(closing_samples) != len(chunk):
                    raise X2ValidationError(
                        "collision-aware closing audit lost a batch sample"
                    )
                chunk_records: list[dict[str, Any]] = []
                for sample_index, (candidate, outcomes) in enumerate(
                    zip(chunk, chunk_outcomes)
                ):
                    chunk_records.append(
                        make_validated_record(
                            candidate,
                            outcomes,
                            thresholds,
                            collision_aware_closing=closing_samples[sample_index],
                            runtime=batch_runtime,
                            criterion=args.criterion,
                        )
                    )
                chunk_outputs = [
                    write_validated_record(candidate, record, overwrite=args.overwrite)
                    for candidate, record in zip(chunk, chunk_records)
                ]
                validated_records.extend(chunk_records)
                outputs.extend(chunk_outputs)
                processed_offset += len(chunk)

        if len(validated_records) != len(candidates):
            raise AssertionError("validator lost a batch remainder")
        summary = _make_summary(
            candidates=candidates,
            records=validated_records,
            outputs=outputs,
            mesh_path=mesh_path,
            scale=scale,
            args=args,
            fk_audits=fk_audits,
            skipped_count=skipped_count,
        )
        if args.summary_json:
            _atomic_json(args.summary_json, summary)
        print(json.dumps(summary, indent=2, allow_nan=False))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
