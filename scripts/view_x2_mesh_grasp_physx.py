#!/usr/bin/env python3
"""Open one X2 mesh-grasp candidate in an interactive Isaac Sim window.

By default this viewer reruns the validator's contact-gradient closing and all
six PhysX gravity directions, then freezes the identity environment at its
final simulated pose.  ``--state raw`` instead displays the optimizer output
before validator closing.  Neither mode writes validation results or modifies
the source JSON.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
    EXPECTED_JOINT_NAMES,
    FORMAL_ACTUATOR_ARMATURE,
    FORMAL_ACTUATOR_DAMPING,
    FORMAL_ACTUATOR_STIFFNESS,
    GRAVITY_TESTS_WXYZ,
    ValidationThresholds,
    load_raw_candidate,
    make_object_centered_replay,
)
from isaaclab.app import AppLauncher  # noqa: E402
from scripts.validate_x2_mesh_grasps_physx import (  # noqa: E402
    DEFAULT_HAND_USD,
    DEFAULT_USD_CACHE,
    _audit_converted_bounds,
    _audit_runtime_mapping,
    _convert_object_mesh,
    _make_scene_cfg,
    _validate_batch,
)


DEFAULT_RAW = (
    PROJECT_ROOT
    / "data"
    / "x2_primitive_grasps"
    / "sphere"
    / "front"
    / "raw"
    / "sphere_r020_front_000000.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--state",
        choices=("validated-final", "raw"),
        default="validated-final",
        help="Replay the final PhysX state (default) or show the raw optimized pose.",
    )
    parser.add_argument("--hand-usd", type=Path, default=DEFAULT_HAND_USD)
    parser.add_argument("--usd-cache", type=Path, default=DEFAULT_USD_CACHE)
    parser.add_argument(
        "--collision-approximation",
        choices=("convex-hull", "convex-decomposition"),
        default="convex-hull",
    )
    parser.add_argument("--density", type=float, default=500.0)
    parser.add_argument("--hand-friction", type=float, default=3.0)
    parser.add_argument("--object-friction", type=float, default=3.0)
    parser.add_argument("--contact-offset", type=float, default=0.001)
    parser.add_argument("--rest-offset", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--sim-steps", type=int, default=100)
    parser.add_argument(
        "--actuator-stiffness", type=float, default=FORMAL_ACTUATOR_STIFFNESS
    )
    parser.add_argument(
        "--actuator-damping", type=float, default=FORMAL_ACTUATOR_DAMPING
    )
    parser.add_argument(
        "--actuator-armature", type=float, default=FORMAL_ACTUATOR_ARMATURE
    )
    parser.add_argument("--solver-type", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--external-forces-every-iteration",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--solve-articulation-contact-last", action="store_true"
    )
    parser.add_argument(
        "--criterion",
        choices=("dexgraspnet-contact", "strict-hold"),
        default="dexgraspnet-contact",
    )
    parser.add_argument("--contact-force-threshold", type=float, default=0.0)
    parser.add_argument("--penetration-threshold", type=float, default=0.001)
    parser.add_argument("--retention-distance", type=float, default=0.1)
    parser.add_argument("--joint-error-threshold", type=float, default=0.1)
    parser.add_argument("--mimic-error-threshold", type=float, default=0.01)
    parser.add_argument("--fk-position-tolerance", type=float, default=5.0e-4)
    parser.add_argument("--fk-normal-min-dot", type=float, default=0.999)
    parser.add_argument("--no-force-closing", action="store_true")
    parser.add_argument("--closing-contact-threshold", type=float, default=0.003)
    parser.add_argument("--closing-displacement", type=float, default=0.001)
    parser.add_argument("--closing-gradient-scale", type=float, default=100.0)
    parser.add_argument("--closing-penetration-cap", type=float, default=0.0015)
    parser.add_argument("--preclose-physics-steps", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    candidate = load_raw_candidate(args.raw_json)
    hand_usd = args.hand_usd.expanduser().resolve()
    if not hand_usd.is_file():
        raise FileNotFoundError(f"X2 hand USD does not exist: {hand_usd}")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext

        object_usd, _ = _convert_object_mesh(
            candidate.mesh_path, candidate.object_scale, args
        )
        _audit_converted_bounds(
            candidate.mesh_path, candidate.object_scale, object_usd
        )
        shape_prefix = candidate.mesh_path.stem.split("_", 1)[0]
        object_prim_name = (
            shape_prefix.capitalize()
            if shape_prefix.isidentifier()
            else "MeshObject"
        )
        validated_final = args.state == "validated-final"
        num_envs = len(GRAVITY_TESTS_WXYZ) if validated_final else 1
        scene_cfg = _make_scene_cfg(
            hand_usd=hand_usd,
            object_usd=object_usd,
            num_envs=num_envs,
            args=args,
            object_prim_name=object_prim_name,
        )
        sim = SimulationContext(
            sim_utils.SimulationCfg(
                device=args.device,
                dt=args.dt / args.substeps if validated_final else args.dt,
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
                render_interval=1,
            )
        )
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2500.0, color=(0.8, 0.8, 0.8)
        )
        light_cfg.func("/World/X2ViewerLight", light_cfg)
        scene = InteractiveScene(scene_cfg)
        sim.reset()

        robot = scene["robot"]
        rigid_object = scene["object"]
        active_joint_ids, _ = _audit_runtime_mapping(
            robot, scene["object_contact"], expected_env_count=num_envs
        )
        if validated_final:
            thresholds = ValidationThresholds(
                penetration=args.penetration_threshold,
                retention_distance=args.retention_distance,
                contact_force=args.contact_force_threshold,
                joint_error=args.joint_error_threshold,
                mimic_error=args.mimic_error_threshold,
            )
            outcomes, audit = _validate_batch(
                candidates=[candidate],
                capacity_samples=1,
                scene=scene,
                sim=sim,
                active_joint_ids=active_joint_ids,
                mimic_joint_pairs=_audit_runtime_mapping(
                    robot, scene["object_contact"], expected_env_count=num_envs
                )[1],
                thresholds=thresholds,
                args=args,
            )
            identity_outcome = outcomes[0][0]
            identity_joint_position = robot.data.joint_pos[0].detach().cpu().tolist()
            identity_object_position = (
                rigid_object.data.root_pos_w[0].detach().cpu().numpy()
            )
        else:
            replay = make_object_centered_replay(candidate)
            hand_pose = torch.zeros((1, 7), dtype=torch.float32, device=sim.device)
            hand_pose[0, :3] = torch.as_tensor(
                replay.hand_translation, dtype=torch.float32, device=sim.device
            ) + scene.env_origins[0]
            hand_pose[0, 3:] = torch.as_tensor(
                replay.hand_quaternion_xyzw, dtype=torch.float32, device=sim.device
            )
            object_pose = torch.zeros((1, 7), dtype=torch.float32, device=sim.device)
            object_pose[0, :3] = torch.as_tensor(
                replay.object_translation, dtype=torch.float32, device=sim.device
            ) + scene.env_origins[0]
            object_pose[0, 3:] = torch.as_tensor(
                replay.object_quaternion_xyzw, dtype=torch.float32, device=sim.device
            )
            joint_position = torch.as_tensor(
                [[candidate.joint_by_name[name] for name in EXPECTED_JOINT_NAMES]],
                dtype=torch.float32,
                device=sim.device,
            )
            active_target = joint_position[:, active_joint_ids]
            rigid_object.write_root_pose_to_sim_index(root_pose=object_pose)
            rigid_object.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=sim.device)
            )
            robot.write_root_pose_to_sim_index(root_pose=hand_pose)
            robot.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=sim.device)
            )
            robot.write_joint_position_to_sim_index(position=joint_position)
            robot.write_joint_velocity_to_sim_index(
                velocity=torch.zeros_like(joint_position)
            )
            robot.set_joint_position_target_index(
                target=active_target, joint_ids=active_joint_ids
            )
            scene.reset()
            sim.forward()
            scene.update(sim.get_physics_dt())
            identity_object_position = object_pose[0, :3].detach().cpu().numpy()

        object_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.02, 0.35, 1.0),
            emissive_color=(0.0, 0.04, 0.15),
            roughness=0.28,
            metallic=0.15,
        )
        object_material_path = "/World/Looks/X2ViewerObjectBlue"
        object_material.func(object_material_path, object_material)
        sim_utils.bind_visual_material(
            f"/World/envs/env_0/{object_prim_name}",
            object_material_path,
            stronger_than_descendants=True,
        )

        contact_positions = np.stack(
            [
                np.asarray(contact["world_position"], dtype=np.float64)
                for contact in candidate.record["selected_contacts"]
            ]
        ) + scene.env_origins[0].detach().cpu().numpy()
        if not validated_final:
            marker_cfg = sim_utils.SphereCfg(
                radius=0.004,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.02, 0.02),
                    emissive_color=(0.5, 0.0, 0.0),
                ),
            )
            for index, position in enumerate(contact_positions):
                marker_cfg.func(
                    f"/World/SelectedContact_{index}",
                    marker_cfg,
                    translation=tuple(float(value) for value in position),
                )

        target = np.asarray(identity_object_position, dtype=np.float64)
        eye = target + np.asarray((0.34, -0.34, 0.24), dtype=np.float64)
        sim.set_camera_view(tuple(eye), tuple(target))
        lines = [
            f"[viewer] raw={candidate.path}",
            f"[viewer] state={args.state}",
            f"[viewer] active_side={candidate.active_side}",
            f"[viewer] stage_object=/World/envs/env_0/{object_prim_name}",
            f"[viewer] selected_contact_ids={candidate.record['selected_contact_ids']}",
        ]
        if validated_final:
            lines.extend(
                (
                    f"[viewer] identity_passed={identity_outcome.passed}",
                    f"[viewer] identity_final_displacement_m={identity_outcome.final_displacement:.9g}",
                    f"[viewer] identity_final_contact_force_n={identity_outcome.final_contact_force:.9g}",
                    f"[viewer] closing_audit={audit['contact_gradient_closing']}",
                    f"[viewer] identity_final_joint={identity_joint_position}",
                    "[viewer] env_0 shows the frozen identity validation result; "
                    "the other five gravity-test environments are spaced 0.8 m away.",
                )
            )
        else:
            lines.append(
                "[viewer] blue mesh is the object; red spheres mark the raw selected contacts."
            )
        lines.append("[viewer] close the Isaac Sim window to exit.")
        print("\n".join(lines), flush=True)
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
