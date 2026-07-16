"""Regression tests for side-conditioned generic X2 mesh grasp generation."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from grasp_generation.x2_mesh_generator import (
    MeshEnergy,
    X2MeshAnnealing,
    _valid_checkpoint_rows,
    cal_unselected_opposite_flex_energy,
    cal_x2_mesh_energy,
    initialize_x2_convex_hull,
)
from grasp_generation.utils.mesh_object_model import MeshObjectModel
from grasp_generation.utils.x2_config import load_x2_mesh_config
from grasp_generation.utils.x2_hand_model import X2HandModel
from grasp_generation.utils.x2_mesh_contacts import (
    GenericDexterousContactPolicy,
    load_generic_contact_candidates,
)
from scripts.generate_x2_mesh_grasps import _parse_args, run
from scripts.build_x2_mesh_contact_candidates import build as build_contact_candidates


class X2MeshGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.mesh_path = (
            cls.root
            / "data/meshdata/000/coacd/decomposed.obj"
        )
        cls.object_scale = 1.0
        cls.config = load_x2_mesh_config()
        cls.candidates = load_generic_contact_candidates(
            cls.config.configured_path("contact_candidates.path", must_exist=True)
        )
        cls.hand = X2HandModel(
            cls.config,
            cls.candidates,
            device="cpu",
            dtype=torch.float64,
            collision_samples_per_link=4,
        )
        cls.policies = {
            side: GenericDexterousContactPolicy(
                cls.candidates, active_side=side, n_contact=4, allow_thumb=True
            )
            for side in ("front", "back")
        }

    def _object(self, batch_size: int = 2, samples: int = 16) -> MeshObjectModel:
        return MeshObjectModel(
            self.mesh_path,
            batch_size=batch_size,
            scale=self.object_scale,
            num_surface_samples=samples,
            device="cpu",
            dtype=torch.float64,
            seed=17,
        )

    def test_front_back_frames_share_fk_and_normals_are_antiparallel(self) -> None:
        torch.testing.assert_close(
            self.hand.back_palm_normal,
            -self.hand.front_palm_normal,
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertIs(self.hand.backend, self.hand.backend)
        self.assertEqual(
            self.config.require("palm_sides.front.frame_path"),
            self.config.require("palm_sides.back.frame_path"),
        )

    def test_front_and_back_initialization_put_object_on_requested_side(self) -> None:
        torch.manual_seed(4)
        active_sides = ("front", "back")
        initialize_x2_convex_hull(
            self.hand,
            self._object(),
            active_sides,
            self.policies,
            self.config,
            np.random.default_rng(4),
        )
        object_root = torch.einsum(
            "bi,bij->bj", -self.hand.global_translation, self.hand.global_rotation
        )
        for index, side in enumerate(active_sides):
            normal = (
                self.hand.front_palm_normal if side == "front" else self.hand.back_palm_normal
            )
            projection = torch.dot(object_root[index] - self.hand.palm_centers[side], normal)
            self.assertGreater(float(projection.detach()), 0.0)

    def test_unselected_fingers_are_penalized_only_when_bending_to_opposite_side(self) -> None:
        def selection(side: str, *, include_index: bool) -> list[int]:
            eligible = self.policies[side].eligible_indices
            chosen: list[int] = []
            if include_index:
                chosen.append(
                    next(
                        value
                        for value in eligible
                        if self.candidates[value].finger_name == "index"
                    )
                )
            chosen.extend(
                value
                for value in eligible
                if value not in chosen
                and (
                    include_index
                    or self.candidates[value].finger_name != "index"
                )
            )
            return chosen[:4]

        # rh_FFJ3 > 0 moves the real distal hull toward front (+palm normal),
        # while rh_FFJ3 < 0 moves it toward back.  The energy itself derives
        # this from FK displacement rather than relying on this sign.
        poses = torch.zeros(5, self.hand.POSE_DIMENSION, dtype=torch.float64)
        poses[:, 3:9] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64
        )
        index_actuator = self.hand.actuator_names.index("rh_FFJ3")
        poses[0, 9 + index_actuator] = 0.75   # back: wrong (toward front)
        poses[1, 9 + index_actuator] = -0.75  # back: allowed
        poses[2, 9 + index_actuator] = -0.75  # front: wrong (toward back)
        poses[3, 9 + index_actuator] = 0.75   # front: allowed
        poses[4, 9 + index_actuator] = 0.75   # back: selected, therefore exempt
        poses.requires_grad_()
        contacts = torch.tensor(
            [
                selection("back", include_index=False),
                selection("back", include_index=False),
                selection("front", include_index=False),
                selection("front", include_index=False),
                selection("back", include_index=True),
            ],
            dtype=torch.long,
        )
        self.hand.set_parameters(poses, contacts)
        energy = cal_unselected_opposite_flex_energy(
            self.hand,
            ("back", "back", "front", "front", "back"),
            margin=0.0,
            displacement_scale=0.02,
        )

        self.assertGreater(float(energy[0].detach()), 1.0)
        self.assertGreater(float(energy[2].detach()), 1.0)
        torch.testing.assert_close(
            energy[[1, 3, 4]],
            torch.zeros(3, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-20,
        )
        energy.sum().backward()
        self.assertGreater(float(poses.grad[0, 9 + index_actuator]), 0.0)
        self.assertLess(float(poses.grad[2, 9 + index_actuator]), 0.0)

    def test_unselected_opposite_flex_constraint_covers_all_five_fingers(self) -> None:
        actuator_by_finger = {
            "index": "rh_FFJ3",
            "middle": "rh_MFJ3",
            "ring": "rh_RFJ3",
            "little": "rh_LFJ3",
            "thumb": "rh_THJ4",
        }
        self.assertEqual(set(self.hand.finger_names), set(actuator_by_finger))
        poses = torch.zeros(5, self.hand.POSE_DIMENSION, dtype=torch.float64)
        poses[:, 3:9] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64
        )
        selections: list[list[int]] = []
        for row, finger_name in enumerate(self.hand.finger_names):
            actuator_name = actuator_by_finger[finger_name]
            actuator_index = self.hand.actuator_names.index(actuator_name)
            poses[row, 9 + actuator_index] = 0.75
            selections.append(
                [
                    index
                    for index in self.policies["back"].eligible_indices
                    if self.candidates[index].finger_name != finger_name
                ][:4]
            )
        poses.requires_grad_()
        self.hand.set_parameters(poses, torch.tensor(selections, dtype=torch.long))
        energy = cal_unselected_opposite_flex_energy(
            self.hand,
            ("back",) * 5,
            margin=0.0,
            displacement_scale=0.02,
        )
        self.assertTrue(torch.all(energy > 1.0))

    def test_side_policy_filters_authored_candidates_and_keeps_shared_tips(self) -> None:
        front = set(self.policies["front"].eligible_indices)
        back = set(self.policies["back"].eligible_indices)
        self.assertNotEqual(front, back)
        for index in front:
            self.assertIn("front", self.candidates[index].supported_sides)
            self.assertNotEqual(self.candidates[index].region, "back_palm")
        for index in back:
            self.assertIn("back", self.candidates[index].supported_sides)
            self.assertNotEqual(self.candidates[index].region, "front_palm")
        shared = {
            index for index, value in enumerate(self.candidates)
            if value.region == "shared_fingertip"
        }
        self.assertTrue(shared)
        self.assertTrue(shared <= front & back)

    def test_finger_strata_survive_sampling_and_slot_resampling(self) -> None:
        rng = np.random.default_rng(123)
        known = {"index", "middle", "ring", "little", "thumb"}
        for side in ("front", "back"):
            for finger_count in range(1, 6):
                policy = GenericDexterousContactPolicy(
                    self.candidates,
                    active_side=side,
                    n_contact=max(4, finger_count),
                    allow_thumb=True,
                    target_finger_count=finger_count,
                )
                values = policy.sample(rng)
                for slot in range(policy.n_contact):
                    values = policy.resample_slot(values, slot, rng)
                    policy.validate(values)
                    actual = {
                        self.candidates[index].finger_name
                        for index in values
                    } & known
                    self.assertEqual(len(actual), finger_count)

        required = ("thumb", "index")
        policy = GenericDexterousContactPolicy(
            self.candidates,
            active_side="front",
            n_contact=4,
            allow_thumb=True,
            target_finger_count=2,
            required_finger_names=required,
        )
        values = policy.sample(rng)
        for slot in range(policy.n_contact):
            values = policy.resample_slot(values, slot, rng)
            actual = {
                self.candidates[index].finger_name for index in values
            } & known
            self.assertEqual(actual, set(required))

    def test_row_policies_mix_same_side_exact_masks_and_front_back(self) -> None:
        class TrackingPolicy(GenericDexterousContactPolicy):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                self.sample_calls = 0
                self.resample_calls = 0

            def sample(self, rng: np.random.Generator) -> tuple[int, ...]:
                self.sample_calls += 1
                return super().sample(rng)

            def resample_slot(
                self,
                indices: tuple[int, ...],
                slot: int,
                rng: np.random.Generator,
            ) -> tuple[int, ...]:
                self.resample_calls += 1
                return super().resample_slot(indices, slot, rng)

        requests = (
            ("front", ("thumb",)),
            ("front", ("index", "middle")),
            ("back", ("ring", "little", "thumb")),
        )
        row_policies = tuple(
            TrackingPolicy(
                self.candidates,
                active_side=side,
                n_contact=4,
                allow_thumb=True,
                target_finger_count=len(names),
                required_finger_names=names,
            )
            for side, names in requests
        )
        active_sides = tuple(side for side, _ in requests)
        torch.manual_seed(71)
        _, contacts = initialize_x2_convex_hull(
            self.hand,
            self._object(batch_size=len(requests)),
            active_sides,
            row_policies,
            self.config,
            np.random.default_rng(71),
        )

        def actual_fingers(row: torch.Tensor) -> set[str]:
            return {
                self.candidates[int(index)].finger_name
                for index in row.detach().cpu()
                if self.candidates[int(index)].finger_name != "palm"
            }

        for row, (policy, (_, expected_names)) in enumerate(
            zip(row_policies, requests)
        ):
            self.assertEqual(policy.sample_calls, 1)
            policy.validate(contacts[row].tolist())
            self.assertEqual(actual_fingers(contacts[row]), set(expected_names))

        annealing_data = copy.deepcopy(self.config.data)
        annealing_data["optimization"]["switch_possibility"] = 1.0
        annealing_config = type(self.config)(
            data=annealing_data,
            path=self.config.path,
            project_root=self.config.project_root,
        )
        assert self.hand.hand_pose is not None
        self.hand.hand_pose.grad = torch.zeros_like(self.hand.hand_pose)
        annealer = X2MeshAnnealing(
            self.hand,
            row_policies,
            active_sides,
            annealing_config,
            seed=72,
        )
        _, _, _, proposed = annealer.try_step()
        for row, (policy, (_, expected_names)) in enumerate(
            zip(row_policies, requests)
        ):
            self.assertEqual(policy.resample_calls, policy.n_contact)
            policy.validate(proposed[row].tolist())
            self.assertEqual(actual_fingers(proposed[row]), set(expected_names))

        batch_size = len(requests)
        zero = torch.zeros(batch_size, dtype=self.hand.dtype)
        energy = MeshEnergy(
            **{
                name: zero.clone()
                for name in MeshEnergy.__dataclass_fields__
            }
        )
        pair_zero = torch.zeros(batch_size, 1, dtype=self.hand.dtype)
        self_collision = SimpleNamespace(
            pair_total_penetration=pair_zero.clone(),
            pair_maximum_penetration=pair_zero.clone(),
            total_penetration=zero.clone(),
            maximum_penetration=zero.clone(),
            worst_pair_indices=torch.zeros(batch_size, dtype=torch.long),
            feasible=torch.ones(batch_size, dtype=torch.bool),
            pair_energy=pair_zero.clone(),
        )
        self.assertTrue(
            bool(
                _valid_checkpoint_rows(
                    self.hand,
                    energy,
                    self_collision,
                    row_policies,
                    active_sides,
                ).all()
            )
        )

        # Both swapped rows are still unique, front-side-eligible selections;
        # only their row-specific exact finger-mask contracts make them invalid.
        swapped = proposed.clone()
        swapped[[0, 1]] = swapped[[1, 0]]
        self.hand.set_parameters(self.hand.hand_pose, swapped)
        torch.testing.assert_close(
            _valid_checkpoint_rows(
                self.hand,
                energy,
                self_collision,
                row_policies,
                active_sides,
            ),
            torch.tensor([False, False, True]),
        )

    def test_thumb_candidates_come_from_seventeen_authored_usd_keypoints(self) -> None:
        thumb = [value for value in self.candidates if value.finger_name == "thumb"]
        self.assertEqual(len(thumb), 17)
        self.assertTrue(
            all(value.source.startswith("authored_keypoint:/robot/") for value in thumb)
        )
        self.assertTrue(
            all(value.supported_sides == ("front", "back") for value in thumb)
        )
        counts = {
            link_name: sum(value.link_name == link_name for value in thumb)
            for link_name in self.hand.backend.thumb_links
        }
        self.assertEqual(
            counts,
            {
                "rh_thbase": 0,
                "rh_thproximal": 4,
                "rh_thmiddle": 4,
                "rh_thdistal": 9,
            },
        )

    def test_contact_builder_rebuilds_all_regions_directly_from_x2_usd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = copy.deepcopy(self.config.data)
            output = Path(directory) / "contact_points_x2_mesh.json"
            data["contact_candidates"]["path"] = str(output)
            config_path = Path(directory) / "x2_mesh_grasp.yaml"
            config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            path, counts = build_contact_candidates(config_path)
            self.assertEqual(path, output)
            self.assertEqual(
                dict(counts),
                {
                    "front_palm": 41,
                    "back_palm": 41,
                    "shared_fingertip": 20,
                    "front_finger_surface": 84,
                    "back_finger_surface": 84,
                    "thumb": 17,
                },
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["candidate_count"], 287)

    def test_generic_contact_sampling_is_unique_and_not_fixed_two_plus_four(self) -> None:
        rng = np.random.default_rng(123)
        observed_palm_counts: set[int] = set()
        observed_thumb = False
        for _ in range(300):
            selection = self.policies["front"].sample(rng)
            self.assertEqual(len(selection), 4)
            self.assertEqual(len(selection), len(set(selection)))
            observed_palm_counts.add(
                sum(self.candidates[index].finger_name == "palm" for index in selection)
            )
            observed_thumb |= any(
                self.candidates[index].finger_name == "thumb" for index in selection
            )
            selection = self.policies["front"].resample_slot(selection, 0, rng)
            self.policies["front"].validate(selection)
        self.assertGreater(len(observed_palm_counts), 1)
        self.assertTrue(observed_thumb)

    def test_twelve_actuators_expand_and_receive_fk_gradients(self) -> None:
        actuator = torch.linspace(-0.2, 0.2, 12, dtype=torch.float64, requires_grad=True)
        joints = self.hand.backend.expand_actuators(actuator)
        self.assertEqual(tuple(joints.shape), (16,))
        collision_points = self.hand.backend.transform_collision_points(joints)
        weights = torch.linspace(
            1.0, 2.0, collision_points.numel(), dtype=torch.float64
        ).reshape_as(collision_points)
        objective = (collision_points.square() * weights).sum()
        objective.backward()
        self.assertTrue(torch.isfinite(actuator.grad).all())
        self.assertTrue(torch.all(actuator.grad.abs() > 1.0e-10))

    def test_freeze_thumb_is_optional_and_default_optimizes_thumb(self) -> None:
        self.assertFalse(self.hand.freeze_thumb)
        frozen = X2HandModel(
            self.config,
            self.candidates,
            device="cpu",
            dtype=torch.float64,
            collision_samples_per_link=2,
            freeze_thumb=True,
        )
        raw = torch.zeros(1, 12, dtype=torch.float64, requires_grad=True)
        materialized = frozen._materialize_actuators(raw)
        materialized.sum().backward()
        torch.testing.assert_close(raw.grad[0, 8:], torch.zeros(4, dtype=torch.float64))
        self.assertTrue(torch.all(raw.grad[0, :8] == 1.0))

    def test_mesh_triangle_distance_and_reverse_penetration_have_finite_gradients(self) -> None:
        object_model = self._object()
        points = torch.tensor(
            [[[0.2, 0.0, 0.0]], [[0.0, 0.2, 0.0]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        distance, _ = object_model.cal_distance(points)
        distance.abs().sum().backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertTrue(torch.any(points.grad != 0.0))

        torch.manual_seed(8)
        initialize_x2_convex_hull(
            self.hand,
            object_model,
            ("front", "back"),
            self.policies,
            self.config,
            np.random.default_rng(8),
        )
        reverse = self.hand.cal_distance(object_model.surface_points_tensor)
        self_collision = self.hand.self_penetration()
        self.assertTrue(torch.isfinite(reverse).all())
        self.assertTrue(torch.isfinite(self_collision).all())

    def test_front_and_back_use_same_dexgraspnet_mesh_energy(self) -> None:
        object_model = self._object()
        torch.manual_seed(19)
        initialize_x2_convex_hull(
            self.hand,
            object_model,
            ("front", "back"),
            self.policies,
            self.config,
            np.random.default_rng(19),
        )
        energy = cal_x2_mesh_energy(
            self.hand,
            object_model,
            ("front", "back"),
            self.config.require("optimization.weights"),
            side_margin=float(self.config.require("optimization.side_margin")),
            normal_opposition_weight=float(
                self.config.require("optimization.normal_opposition_in_force_closure")
            ),
        )
        for name in (
            "total",
            "E_fc",
            "E_dis",
            "E_pen",
            "E_spen",
            "E_joints",
            "E_side",
            "E_unselected_opposite_flex",
        ):
            value = getattr(energy, name)
            self.assertEqual(tuple(value.shape), (2,))
            self.assertTrue(torch.isfinite(value).all())

    def test_generic_mesh_modules_have_no_legacy_domain_dependencies(self) -> None:
        paths = (
            Path("grasp_generation/x2_mesh_generator.py"),
            Path("grasp_generation/utils/mesh_object_model.py"),
            Path("grasp_generation/utils/x2_hand_model.py"),
            Path("scripts/generate_x2_mesh_grasps.py"),
        )
        forbidden_text = ("CylinderObjectModel", "2 palm + 4 finger", "75-point", "certificate")
        for path in paths:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            imports: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            self.assertFalse([name for name in imports if "cylinder" in name.lower()])
            self.assertFalse([text for text in forbidden_text if text in source])

    def test_default_iterations_match_original_dexgraspnet(self) -> None:
        source = Path("grasp_generation/scripts/generate_grasps.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        original_default = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--n_iter"
            ):
                continue
            for keyword in node.keywords:
                if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                    original_default = keyword.value.value
        self.assertEqual(original_default, 6000)
        args = _parse_args(["--mesh-path", str(self.mesh_path)])
        self.assertEqual(args.n_iterations, original_default)

    def _cli_args(self, output: Path, side: str, num_grasps: int) -> argparse.Namespace:
        return argparse.Namespace(
            mesh_path=self.mesh_path,
            side=side,
            num_grasps=num_grasps,
            batch_size=2,
            n_contact=4,
            n_iterations=1,
            seed=31,
            device="cpu",
            output=output,
            config=None,
            object_scale=self.object_scale,
            surface_samples=8,
            freeze_thumb=False,
            overwrite=True,
        )

    def test_side_both_outputs_two_single_object_groups_not_dual_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records, summary = run(self._cli_args(Path(directory), "both", 1))
            self.assertEqual({record["active_side"] for record in records}, {"front", "back"})
            self.assertEqual(len(records), 2)
            self.assertEqual(summary["dual_object_samples"], 0)
            for record in records:
                self.assertIn("mesh_path", record["object"])
                self.assertNotIn("front_object", record)
                self.assertNotIn("back_object", record)
                self.assertFalse(record["simulation_success"])
            self.assertFalse(list(Path(directory).glob("**/valid/*.json")))

    def test_side_any_persists_and_routes_actual_active_side(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records, summary = run(self._cli_args(Path(directory), "any", 2))
            self.assertEqual(len(records), 2)
            for path_text, record in zip(summary["output_files"], records):
                path = Path(path_text)
                self.assertEqual(path.parents[1].name, f"{record['active_side']}_single")
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(saved["active_side"], record["active_side"])
                self.assertFalse(saved["success"])


if __name__ == "__main__":
    unittest.main()
