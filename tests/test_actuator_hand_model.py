"""Regression tests for the real X2 12-actuator hand parameterization."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from grasp_generation.utils.actuator_hand_model import (
    ActuatorHandModel,
    _sample_triangle_surfaces,
)
from grasp_generation.utils.x2_config import load_x2_mesh_config


class ActuatorHandModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # This loads and validates the composed X2 USD, but does not launch
        # Isaac Sim.  A small vertex sample keeps the test inexpensive while
        # still exercising every authored collision link.
        cls.config = load_x2_mesh_config()
        cls.hand = ActuatorHandModel(
            cls.config,
            device="cpu",
            dtype=torch.float64,
            collision_samples_per_link=8,
        )

    def test_twelve_actuators_expand_to_sixteen_joints(self) -> None:
        actuators = torch.linspace(
            -0.35, 0.35, 12, dtype=torch.float64, requires_grad=True
        )
        joints = self.hand.expand_actuators(actuators, check_limits=True)

        self.assertEqual(tuple(actuators.shape), (12,))
        self.assertEqual(tuple(joints.shape), (16,))

        actuator_index = {name: index for index, name in enumerate(self.hand.actuator_names)}
        joint_index = {name: index for index, name in enumerate(self.hand.full_joint_names)}
        for name in self.hand.actuator_names:
            self.assertEqual(
                joints[joint_index[name]].item(), actuators[actuator_index[name]].item()
            )
        for passive_name, (driver, multiplier, offset) in self.hand.passive_joint_coupling.items():
            self.assertEqual(multiplier, 1.0)
            self.assertEqual(offset, 0.0)
            self.assertEqual(
                joints[joint_index[passive_name]].item(),
                joints[joint_index[driver]].item(),
                msg=f"{passive_name} must exactly mimic {driver}",
            )

        # Each J2 drives its passive J1 follower; every other actuator appears
        # once in the materialized 16-joint state.
        joints.sum().backward()
        self.assertIsNotNone(actuators.grad)
        torch.testing.assert_close(
            actuators.grad,
            torch.tensor([1.0] * 4 + [2.0] * 4 + [1.0] * 4, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_thumb_values_are_fixed_and_do_not_receive_optimizer_gradient(self) -> None:
        seed = torch.linspace(-0.4, 0.4, 12, dtype=torch.float64, requires_grad=True)
        fixed = self.hand.with_fixed_thumb(seed)

        for name in self.hand.non_thumb_actuator_names:
            index = self.hand.actuator_names.index(name)
            self.assertEqual(fixed[index].item(), seed[index].item())
        for name, expected in self.hand.fixed_thumb_position.items():
            index = self.hand.actuator_names.index(name)
            self.assertEqual(fixed[index].item(), expected)

        fixed.sum().backward()
        self.assertIsNotNone(seed.grad)
        expected_gradient = torch.tensor([1.0] * 8 + [0.0] * 4, dtype=torch.float64)
        torch.testing.assert_close(seed.grad, expected_gradient, rtol=0.0, atol=0.0)

        batch = np.stack((np.linspace(-0.2, 0.2, 12), np.linspace(0.2, -0.2, 12)))
        batch_actuators = self.hand.with_fixed_thumb(batch)
        self.assertIsInstance(batch_actuators, np.ndarray)
        for name, expected in self.hand.fixed_thumb_position.items():
            index = self.hand.actuator_names.index(name)
            np.testing.assert_array_equal(batch_actuators[:, index], expected)

    def test_thumb_collision_geometry_remains_in_the_model(self) -> None:
        self.assertEqual(len(self.hand.link_names), 17)
        self.assertEqual(set(self.hand.collision_meshes), set(self.hand.link_names))
        self.assertTrue(
            all(mesh.approximation == "convexHull" for mesh in self.hand.collision_meshes.values())
        )

    def test_optimizer_and_physx_share_explicit_low_vertex_collision_hulls(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        manifest_path = (
            project_root / "x2_mujoco" / "payloads" / "x2_physx_collision_hulls.json"
        )
        manifest = json.loads(manifest_path.read_text())

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["max_vertices_per_hull"], 64)
        self.assertEqual(manifest["source_asset"], "x2_mujoco/x2_keypoints.usda")
        self.assertEqual(
            manifest["output_overlay"],
            "x2_mujoco/payloads/x2_physx_collision_hulls.usda",
        )
        self.assertEqual(set(manifest["links"]), set(self.hand.link_names))
        # Fourteen render subtrees originate as instanceable references.  The
        # saved roots are required so a second overlay rebuild keeps them
        # de-instanced and can continue to disable their original colliders.
        self.assertEqual(
            sum(
                record["instance_root_path"] is not None
                for record in manifest["links"].values()
            ),
            14,
        )
        for link_name, mesh in self.hand.collision_meshes.items():
            with self.subTest(link=link_name):
                self.assertTrue(mesh.prim_path.endswith("/x2_physx_collision_hull"))
                self.assertLessEqual(mesh.authored_vertex_count, 64)
                self.assertEqual(
                    mesh.authored_vertex_count,
                    manifest["links"][link_name]["proxy_vertex_count"],
                )
                self.assertEqual(
                    mesh.authored_triangle_count,
                    manifest["links"][link_name]["proxy_triangle_count"],
                )

    def test_area_weighted_face_samples_cover_broad_triangle_interiors(self) -> None:
        vertices = np.array(
            [
                [0.1603, 0.04152, -0.00023],
                [0.31791, 0.01296, 0.02036],
                [0.31135, 0.01320, -0.03071],
            ],
            dtype=np.float64,
        )
        triangles = np.array([[0, 1, 2]], dtype=np.int64)
        samples = _sample_triangle_surfaces(vertices, triangles, 512)
        self.assertEqual(samples.shape, (512, 3))
        self.assertTrue(np.isfinite(samples).all())
        barycentric_normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        plane_error = np.abs((samples - vertices[0]) @ barycentric_normal)
        self.assertLess(float(plane_error.max()), 1.0e-12)

        for link_name in self.hand.link_names:
            with self.subTest(link=link_name):
                self.assertEqual(
                    self.hand.collision_face_samples_local[link_name].shape,
                    (self.hand.collision_samples_per_link, 3),
                )
                self.assertTrue(
                    np.isfinite(self.hand.collision_face_samples_local[link_name]).all()
                )
        zero_actuators = torch.zeros(12, dtype=torch.float64)
        collision_points = self.hand.transform_collision_points(
            self.hand.expand_actuators(zero_actuators)
        )
        self.assertEqual(collision_points.shape[-1], 3)
        self.assertTrue(torch.isfinite(collision_points).all().item())
        self.assertEqual(
            collision_points.shape[-2],
            sum(len(points) for points in self.hand.collision_surface_samples_local.values()),
        )
        expected_audit_per_link = 3 * self.hand.audit_collision_samples_per_link
        for link_name in self.hand.link_names:
            with self.subTest(audit_link=link_name):
                self.assertEqual(
                    self.hand.audit_collision_surface_samples_local[
                        link_name
                    ].shape,
                    (expected_audit_per_link, 3),
                )
        audit_points = self.hand.transform_audit_collision_points(
            self.hand.expand_actuators(zero_actuators)
        )
        self.assertEqual(
            audit_points.shape[-2],
            len(self.hand.link_names) * expected_audit_per_link,
        )
        self.assertTrue(torch.isfinite(audit_points).all().item())

        # Thumb geometry participates in the same self-collision model and is
        # also available to the generic contact policy.
        self.assertTrue(
            any(
                first in self.hand.thumb_links or second in self.hand.thumb_links
                for first, second in self.hand.self_collision_pairs
            )
        )

    def test_hull_pair_filtering_and_inflated_capsules_match_audited_contract(self) -> None:
        self.assertEqual(len(self.hand.self_collision_proxy_pairs), 98)
        self.assertEqual(len(self.hand.self_collision_hull_pairs), 119)
        proxy_pairs = {frozenset(pair) for pair in self.hand.self_collision_proxy_pairs}
        hull_pairs = {frozenset(pair) for pair in self.hand.self_collision_hull_pairs}

        # Direct parents remain excluded.  Graph-distance-two pairs move into the
        # accurate hull set, except for the one explicit canonical structural
        # overlap.  palm--thmiddle is only a capsule exclusion.
        self.assertNotIn(frozenset(("rh_palm", "rh_ffproximal")), hull_pairs)
        self.assertNotIn(frozenset(("rh_ffproximal", "rh_ffdistal")), proxy_pairs)
        self.assertIn(frozenset(("rh_ffproximal", "rh_ffdistal")), hull_pairs)
        self.assertNotIn(frozenset(("rh_palm", "rh_thproximal")), hull_pairs)
        self.assertNotIn(frozenset(("rh_palm", "rh_thmiddle")), proxy_pairs)
        self.assertIn(frozenset(("rh_palm", "rh_thmiddle")), hull_pairs)

        thumb_links = set(self.hand.thumb_links[1:])
        index_links = set(self.hand.finger_links["index"])
        weighted = {
            frozenset(pair)
            for pair, weight in zip(
                self.hand.self_collision_hull_pairs,
                self.hand.self_collision_hull_pair_weights,
            )
            if weight == 2.0
        }
        self.assertEqual(
            weighted,
            {
                frozenset((thumb, index))
                for thumb in thumb_links
                for index in index_links
            },
        )

        expected_samples = 3 * self.hand.self_collision_samples_per_link
        for link_name, proxy in self.hand.capsule_proxy_by_link.items():
            with self.subTest(link=link_name):
                self.assertEqual(
                    self.hand.self_collision_surface_samples_local[link_name].shape,
                    (expected_samples, 3),
                )
                vertices = self.hand.collision_vertices_local[link_name]
                start = np.asarray(proxy.point_a_local)
                end = np.asarray(proxy.point_b_local)
                segment = end - start
                fraction = ((vertices - start) @ segment) / max(
                    float(segment @ segment), 1.0e-18
                )
                closest = start + fraction.clip(0.0, 1.0)[:, None] * segment
                required_radius = float(
                    np.linalg.norm(vertices - closest, axis=1).max()
                )
                inflated_radius = (
                    proxy.radius
                    + self.hand.capsule_enclosure_residual_by_link[link_name]
                    + self.hand.self_collision_broadphase_margin
                )
                self.assertLessEqual(required_radius, inflated_radius + 1.0e-12)

    def test_sampled_hulls_detect_capsule_false_negative_and_canonical_is_clear(self) -> None:
        false_negative = torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                -0.8944927476456583,
                0.0,
                0.0,
                0.0,
                1.172519862657409,
                0.1919020405061742,
                -1.0315357827695095,
                1.141518901458947,
                -1.0143883279069341,
            ],
            dtype=torch.float64,
        )
        joints = self.hand.expand_actuators(false_negative)
        capsule = self.hand.self_collision_depths(joints)
        self.assertEqual(float(torch.relu(capsule).sum()), 0.0)

        first_depths, second_depths = self.hand.self_collision_hull_signed_depths(joints)
        pair_index = self.hand.self_collision_hull_pairs.index(
            ("rh_ffproximal", "rh_thmiddle")
        )
        target_depth = torch.maximum(
            first_depths[pair_index].amax(), second_depths[pair_index].amax()
        )
        self.assertGreater(float(target_depth), 0.0005)

        # The conservative broadphase must preserve every positive sampled depth.
        full_first, full_second = self.hand.self_collision_hull_signed_depths(
            joints, use_broadphase=False
        )
        torch.testing.assert_close(
            torch.relu(first_depths), torch.relu(full_first), rtol=0.0, atol=1.0e-12
        )
        torch.testing.assert_close(
            torch.relu(second_depths), torch.relu(full_second), rtol=0.0, atol=1.0e-12
        )

        canonical = self.hand.expand_actuators(torch.zeros(12, dtype=torch.float64))
        canonical_first, canonical_second = self.hand.self_collision_hull_signed_depths(
            canonical
        )
        self.assertEqual(float(torch.relu(canonical_first).sum()), 0.0)
        self.assertEqual(float(torch.relu(canonical_second).sum()), 0.0)

    def test_formal_sample_31_reproduces_thumb_index_penetration(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        fixture = (
            project_root
            / "data/x2_formal_grasps_6000/sphere_r020_seed1/front_single/raw"
            / "sphere_r020_front_000031.json"
        )
        self.assertEqual(
            hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "d4b6bd97826490e7356b3c536997c8390fe3ee9ca12f7eb09eb20e38a160b489",
        )
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        joints = self.hand.expand_actuators(
            torch.tensor(payload["actuator"], dtype=torch.float64)
        )
        first_depths, second_depths = self.hand.self_collision_hull_signed_depths(joints)
        pair_maximum = torch.maximum(
            first_depths.amax(dim=-1), second_depths.amax(dim=-1)
        )
        worst_index = int(pair_maximum.argmax())
        self.assertEqual(
            self.hand.self_collision_hull_pairs[worst_index],
            ("rh_ffmiddle", "rh_thdistal"),
        )
        self.assertGreater(float(pair_maximum[worst_index]), 0.008)

    def test_short_collision_only_optimization_halves_severe_penetration(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        fixture = (
            project_root
            / "data/x2_formal_grasps_6000/sphere_r020_seed1/front_single/raw"
            / "sphere_r020_front_000031.json"
        )
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        actuators = torch.tensor(
            payload["actuator"], dtype=torch.float64, requires_grad=True
        )
        clearance = float(self.config.require("self_collision.clearance_margin"))
        smoothness = float(self.config.require("self_collision.smoothness"))
        pair_weights = torch.as_tensor(
            self.hand.self_collision_hull_pair_weights, dtype=torch.float64
        )

        def collision_terms(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            joints = self.hand.expand_actuators(value)
            first_depths, second_depths = self.hand.self_collision_hull_signed_depths(
                joints
            )
            maximum = torch.maximum(
                torch.relu(first_depths).amax(dim=-1),
                torch.relu(second_depths).amax(dim=-1),
            ).amax()

            def penalty(depth: torch.Tensor) -> torch.Tensor:
                q = smoothness * torch.nn.functional.softplus(
                    (depth + clearance) / smoothness
                )
                return q + q.square() / clearance

            first_penalty = penalty(first_depths)
            second_penalty = penalty(second_depths)
            pair_energy = torch.maximum(
                first_penalty.amax(dim=-1), second_penalty.amax(dim=-1)
            ) + 0.5 * (
                first_penalty.mean(dim=-1) + second_penalty.mean(dim=-1)
            )
            return (pair_energy * pair_weights).sum(), maximum

        optimizer = torch.optim.Adam((actuators,), lr=0.02)
        initial_maximum = float(collision_terms(actuators)[1].detach())
        self.assertGreater(initial_maximum, 0.005)
        for _ in range(12):
            optimizer.zero_grad()
            energy, _ = collision_terms(actuators)
            self.assertTrue(bool(torch.isfinite(energy)))
            energy.backward()
            self.assertTrue(bool(torch.isfinite(actuators.grad).all()))
            optimizer.step()
            with torch.no_grad():
                actuators.clamp_(
                    self.hand.actuator_limits_tensor[:, 0],
                    self.hand.actuator_limits_tensor[:, 1],
                )

        final_energy, final_maximum = collision_terms(actuators)
        joints = self.hand.expand_actuators(actuators, check_limits=True)
        self.assertTrue(bool(torch.isfinite(final_energy)))
        self.assertTrue(bool(torch.isfinite(actuators).all()))
        self.assertTrue(bool(torch.isfinite(joints).all()))
        self.assertLess(float(final_maximum.detach()), 0.5 * initial_maximum)


if __name__ == "__main__":
    unittest.main()
