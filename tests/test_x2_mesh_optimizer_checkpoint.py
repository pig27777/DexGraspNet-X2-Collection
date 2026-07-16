"""Checkpoint, feasibility-protection, and v4 wire regressions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from grasp_generation.x2_mesh_generator import (
    BIDIRECTIONAL_CHECKPOINT_PERIOD,
    BIDIRECTIONAL_CHECKPOINT_TOP_K,
    DENSE_HAND_SURFACE_SAMPLES_PER_SET,
    DENSE_OBJECT_SURFACE_SAMPLES,
    HAND_OBJECT_PENETRATION_THRESHOLD,
    HandObjectPenetrationDiagnostics,
    MeshEnergy,
    X2MeshAnnealing,
    _CheckpointBank,
    _SparseCheckpointPool,
    _bidirectional_fallback_improvement_mask,
    _bidirectional_feasible_improvement_mask,
    _fallback_improvement_mask,
    _feasible_improvement_mask,
    _proposal_protection_masks,
)
from grasp_generation.utils.x2_config import (
    X2Config,
    X2ConfigurationError,
    _validate_x2_mesh_config,
    load_x2_mesh_config,
)
from scripts.generate_x2_mesh_grasps import run


def _energy(values: list[float]) -> MeshEnergy:
    total = torch.tensor(values, dtype=torch.float64)
    zero = torch.zeros_like(total)
    return MeshEnergy(
        total=total,
        E_fc=zero.clone(),
        E_dis=zero.clone(),
        E_pen=zero.clone(),
        reverse_maximum_penetration=zero.clone(),
        E_spen=zero.clone(),
        E_spen_capsule=zero.clone(),
        E_spen_hull=zero.clone(),
        E_joints=zero.clone(),
        E_side=zero.clone(),
        E_unselected_opposite_flex=zero.clone(),
        normal_opposition=zero.clone(),
        normal_opposition_penalty=zero.clone(),
    )


def _self_collision(
    maximum: list[float], total: list[float], feasible: list[bool]
) -> SimpleNamespace:
    maximum_tensor = torch.tensor(maximum, dtype=torch.float64)
    total_tensor = torch.tensor(total, dtype=torch.float64)
    return SimpleNamespace(
        pair_total_penetration=total_tensor[:, None],
        pair_maximum_penetration=maximum_tensor[:, None],
        total_penetration=total_tensor,
        maximum_penetration=maximum_tensor,
        worst_pair_indices=torch.zeros(len(maximum), dtype=torch.long),
        feasible=torch.tensor(feasible, dtype=torch.bool),
        pair_energy=torch.zeros(len(maximum), 1, dtype=torch.float64),
    )


def _hand_object(
    maximum: list[float],
    total: list[float] | None = None,
    evaluated: list[bool] | None = None,
) -> HandObjectPenetrationDiagnostics:
    maximum_tensor = torch.tensor(maximum, dtype=torch.float64)
    total_tensor = torch.tensor(
        maximum if total is None else total, dtype=torch.float64
    )
    evaluated_tensor = torch.tensor(
        [True] * len(maximum) if evaluated is None else evaluated,
        dtype=torch.bool,
    )
    zero = torch.zeros_like(maximum_tensor)
    return HandObjectPenetrationDiagnostics(
        forward_total_penetration=total_tensor,
        forward_maximum_penetration=maximum_tensor,
        reverse_total_penetration=zero.clone(),
        reverse_maximum_penetration=zero.clone(),
        total_penetration=total_tensor,
        maximum_penetration=maximum_tensor,
        feasible=(
            evaluated_tensor
            & (maximum_tensor < HAND_OBJECT_PENETRATION_THRESHOLD)
        ),
        evaluated=evaluated_tensor,
        threshold=HAND_OBJECT_PENETRATION_THRESHOLD,
    )


class _DummyHand:
    def __init__(self, batch_size: int) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.hand_pose = torch.zeros(batch_size, 21, dtype=torch.float64)
        self.contact_point_indices = torch.arange(4).repeat(batch_size, 1)
        self.actuator_positions = torch.zeros(batch_size, 12, dtype=torch.float64)
        self.joint_positions = torch.zeros(batch_size, 16, dtype=torch.float64)

    def set_parameters(
        self, pose: torch.Tensor, contact_indices: torch.Tensor
    ) -> None:
        self.hand_pose = pose
        self.contact_point_indices = contact_indices


class CheckpointRankingTest(unittest.TestCase):
    def test_sparse_pool_preserves_energy_and_penetration_candidates(self) -> None:
        hand = _DummyHand(1)
        pool = _SparseCheckpointPool.empty(hand, capacity=3)

        def add(
            pose_id: float,
            maximum: float,
            total: float,
            energy: float,
            step: int,
        ) -> None:
            hand.hand_pose[0, 0] = pose_id
            pool.update(
                torch.tensor([True]),
                hand,
                _energy([energy]),
                _hand_object([maximum], [total]),
                step=step,
            )

        add(1.0, 0.0003, 0.0003, 1.0, 1)
        add(2.0, 0.0002, 0.0020, 9.0, 2)
        add(3.0, 0.0002, 0.0010, 8.0, 3)
        add(4.0, 0.0002, 0.0010, 7.0, 4)
        add(5.0, 0.0002, 0.0010, 7.0, 5)
        torch.testing.assert_close(pool.step[0], torch.tensor([1, 4, 5]))
        torch.testing.assert_close(
            pool.pose[0, :, 0],
            torch.tensor([1.0, 4.0, 5.0], dtype=torch.float64),
        )

        # Re-seeing an identical pose/contact snapshot cannot consume a second
        # top-K slot or replace its earlier deterministic tie.
        hand.hand_pose[0, 0] = 4.0
        pool.update(
            torch.tensor([True]),
            hand,
            _energy([7.0]),
            _hand_object([0.0002], [0.0010]),
            step=99,
        )
        torch.testing.assert_close(pool.step[0], torch.tensor([1, 4, 5]))

    def test_formal_revision_and_dense_density_are_non_relaxable(self) -> None:
        config = load_x2_mesh_config()
        self.assertEqual(
            config.require("pipeline_revision"),
            "x2_mesh_grasp_unselected_finger_side_v6",
        )
        self.assertEqual(
            config.require("generation.dense_hand_surface_samples_per_set"),
            DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        )
        self.assertEqual(
            config.require("generation.dense_object_surface_samples"),
            DENSE_OBJECT_SURFACE_SAMPLES,
        )

        for field, invalid in (
            ("pipeline_revision", "x2_mesh_grasp_unselected_finger_side_v4"),
            ("dense_hand_surface_samples_per_set", 255),
            ("dense_object_surface_samples", 8191),
            ("dense_hand_object_penetration_threshold", 0.00101),
        ):
            with self.subTest(field=field):
                data = copy.deepcopy(config.data)
                if field == "pipeline_revision":
                    data[field] = invalid
                else:
                    data["generation"][field] = invalid
                candidate = X2Config(
                    data=data,
                    path=config.path,
                    project_root=config.project_root,
                )
                with self.assertRaises(X2ConfigurationError):
                    _validate_x2_mesh_config(candidate)

    def test_bidirectional_gate_is_strict_and_joint_feasible_ranks_energy(self) -> None:
        hand = _DummyHand(3)
        energy = _energy([10.0, 10.0, 10.0])
        collision = _self_collision(
            [0.0002, 0.0002, 0.0002],
            [0.0003, 0.0003, 0.0003],
            [True, True, False],
        )
        initial_hand_object = _hand_object([0.0008, 0.0010, 0.0004])
        self.assertEqual(
            initial_hand_object.feasible.tolist(), [True, False, True]
        )
        bank = _CheckpointBank.empty(
            hand, energy, collision, initial_hand_object
        )
        initial_mask = _bidirectional_feasible_improvement_mask(
            bank,
            torch.ones(3, dtype=torch.bool),
            energy,
            collision,
            initial_hand_object,
        )
        torch.testing.assert_close(
            initial_mask, torch.tensor([True, False, False])
        )
        bank.update(
            initial_mask,
            hand,
            energy,
            collision,
            initial_hand_object,
            step=0,
        )

        candidate_energy = _energy([5.0, 2.0, 1.0])
        candidate_hand_object = _hand_object([0.0009, 0.0007, 0.0001])
        candidate_mask = _bidirectional_feasible_improvement_mask(
            bank,
            torch.ones(3, dtype=torch.bool),
            candidate_energy,
            collision,
            candidate_hand_object,
        )
        torch.testing.assert_close(
            candidate_mask, torch.tensor([True, True, False])
        )

    def test_bidirectional_self_feasible_fallback_ranks_max_total_energy(self) -> None:
        hand = _DummyHand(4)
        baseline_energy = _energy([3.0, 3.0, 3.0, 3.0])
        collision = _self_collision(
            [0.0002] * 4, [0.0003] * 4, [True] * 4
        )
        baseline_hand_object = _hand_object(
            [0.002] * 4, [0.004] * 4
        )
        bank = _CheckpointBank.empty(
            hand, baseline_energy, collision, baseline_hand_object
        )
        bank.update(
            torch.ones(4, dtype=torch.bool),
            hand,
            baseline_energy,
            collision,
            baseline_hand_object,
            step=0,
        )
        candidate_energy = _energy([99.0, 99.0, 2.0, 3.0])
        candidate_hand_object = _hand_object(
            [0.0019, 0.002, 0.002, 0.002],
            [99.0, 0.0039, 0.004, 0.004],
        )
        mask = _bidirectional_fallback_improvement_mask(
            bank,
            torch.ones(4, dtype=torch.bool),
            candidate_energy,
            collision,
            candidate_hand_object,
        )
        torch.testing.assert_close(
            mask, torch.tensor([True, True, True, False])
        )

    def test_checkpoint_update_freezes_unmasked_rows_and_full_materialization(self) -> None:
        hand = _DummyHand(3)
        hand.hand_pose[:] = torch.arange(3, dtype=torch.float64)[:, None]
        hand.contact_point_indices[:] += 10 * torch.arange(3)[:, None]
        hand.actuator_positions[:] = 20 * torch.arange(3, dtype=torch.float64)[:, None]
        hand.joint_positions[:] = 30 * torch.arange(3, dtype=torch.float64)[:, None]
        energy = _energy([1.0, 2.0, 3.0])
        collision = _self_collision([0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [False] * 3)
        bank = _CheckpointBank.empty(hand, energy, collision)
        bank.update(
            torch.tensor([True, False, True]),
            hand,
            energy,
            collision,
            step=4,
        )
        row_zero_pose = bank.pose[0].clone()
        row_two_contacts = bank.contact_indices[2].clone()
        row_two_actuators = bank.actuator_positions[2].clone()
        row_two_joints = bank.joint_positions[2].clone()

        hand.hand_pose.fill_(99.0)
        hand.contact_point_indices.fill_(99)
        hand.actuator_positions.fill_(99.0)
        hand.joint_positions.fill_(99.0)
        bank.update(
            torch.tensor([False, True, False]),
            hand,
            _energy([9.0, 9.0, 9.0]),
            _self_collision([0.9] * 3, [9.0] * 3, [False] * 3),
            step=8,
        )

        torch.testing.assert_close(bank.valid, torch.tensor([True, True, True]))
        torch.testing.assert_close(bank.step, torch.tensor([4, 8, 4]))
        torch.testing.assert_close(bank.pose[0], row_zero_pose)
        torch.testing.assert_close(bank.contact_indices[2], row_two_contacts)
        torch.testing.assert_close(bank.actuator_positions[2], row_two_actuators)
        torch.testing.assert_close(bank.joint_positions[2], row_two_joints)
        self.assertTrue(torch.all(bank.pose[1] == 99.0))
        self.assertTrue(torch.all(bank.contact_indices[1] == 99))

    def test_feasible_checkpoint_keeps_rejected_proposal_and_earliest_tie(self) -> None:
        hand = _DummyHand(1)
        initial_energy = _energy([10.0])
        initial_collision = _self_collision([0.0004], [0.001], [True])
        bank = _CheckpointBank.empty(hand, initial_energy, initial_collision)
        bank.update(
            _feasible_improvement_mask(
                bank,
                torch.tensor([True]),
                initial_energy,
                initial_collision,
            ),
            hand,
            initial_energy,
            initial_collision,
            step=0,
        )

        # This proposal is checkpointed before the annealer decides whether to
        # accept it.  Resetting the live hand simulates a Metropolis rejection.
        hand.hand_pose.fill_(2.0)
        proposal_energy = _energy([5.0])
        proposal_collision = _self_collision([0.0003], [0.0008], [True])
        bank.update(
            _feasible_improvement_mask(
                bank,
                torch.tensor([True]),
                proposal_energy,
                proposal_collision,
            ),
            hand,
            proposal_energy,
            proposal_collision,
            step=1,
        )
        hand.hand_pose.zero_()
        self.assertEqual(int(bank.step[0]), 1)
        self.assertTrue(torch.all(bank.pose[0] == 2.0))

        # An exact tie at a later step must not replace the earlier snapshot.
        hand.hand_pose.fill_(3.0)
        bank.update(
            _feasible_improvement_mask(
                bank,
                torch.tensor([True]),
                proposal_energy,
                proposal_collision,
            ),
            hand,
            proposal_energy,
            proposal_collision,
            step=2,
        )
        self.assertEqual(int(bank.step[0]), 1)
        self.assertTrue(torch.all(bank.pose[0] == 2.0))

    def test_fallback_ranking_is_maximum_then_total_then_energy(self) -> None:
        hand = _DummyHand(4)
        baseline_energy = _energy([3.0, 3.0, 3.0, 3.0])
        baseline_collision = _self_collision(
            [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [False] * 4
        )
        bank = _CheckpointBank.empty(hand, baseline_energy, baseline_collision)
        bank.update(
            torch.ones(4, dtype=torch.bool),
            hand,
            baseline_energy,
            baseline_collision,
            step=0,
        )
        candidate_energy = _energy([99.0, 99.0, 2.0, 3.0])
        candidate_collision = _self_collision(
            [0.9, 1.0, 1.0, 1.0],
            [99.0, 1.9, 2.0, 2.0],
            [False] * 4,
        )
        mask = _fallback_improvement_mask(
            bank,
            torch.ones(4, dtype=torch.bool),
            candidate_energy,
            candidate_collision,
        )
        torch.testing.assert_close(
            mask, torch.tensor([True, True, True, False])
        )

    def test_protection_rejects_only_finite_severe_regressions(self) -> None:
        old = torch.tensor([0.0004, 0.0020, 0.0004, 0.0004])
        new = torch.tensor([0.0011, 0.0015, 0.0009, float("nan")])
        finite = torch.tensor([True, True, True, False])
        allowed, severe = _proposal_protection_masks(
            old,
            new,
            finite,
            enabled=True,
            hard_threshold=0.001,
            maximum_allowed_increase=0.0001,
        )
        torch.testing.assert_close(
            allowed, torch.tensor([False, True, True, False])
        )
        torch.testing.assert_close(
            severe, torch.tensor([True, False, False, False])
        )

    def test_nonfinite_proposal_rolls_back_only_its_batch_row(self) -> None:
        hand = _DummyHand(2)
        config = load_x2_mesh_config()
        annealer = X2MeshAnnealing(hand, {}, (), config, seed=123)
        old_pose = hand.hand_pose.detach().clone().requires_grad_(True)
        old_indices = hand.contact_point_indices.detach().clone()
        old_gradient = torch.zeros_like(old_pose)
        proposal = old_pose.detach().clone()
        proposal[0, 0] = float("nan")
        proposal[1] = 2.0
        proposal.requires_grad_(True)
        proposed_indices = old_indices.clone()
        proposed_indices[:, 0] += 10
        hand.set_parameters(proposal, proposed_indices)

        accepted, _ = annealer.accept_step(
            torch.tensor([1.0, 1.0], dtype=torch.float64),
            torch.tensor([float("nan"), 0.0], dtype=torch.float64),
            old_pose,
            old_indices,
            old_gradient,
            proposed_indices,
            torch.tensor([True, True]),
        )

        torch.testing.assert_close(accepted, torch.tensor([False, True]))
        self.assertTrue(bool(torch.isfinite(hand.hand_pose).all()))
        torch.testing.assert_close(hand.hand_pose[0], old_pose[0])
        self.assertTrue(torch.all(hand.hand_pose[1] == 2.0))
        torch.testing.assert_close(hand.contact_point_indices[0], old_indices[0])
        self.assertEqual(
            int(hand.contact_point_indices[1, 0]), int(old_indices[1, 0]) + 10
        )


class V4RecordTest(unittest.TestCase):
    def test_generated_record_round_trips_self_collision_and_provenance(self) -> None:
        root = Path(__file__).resolve().parents[1]
        mesh_path = (
            root
            / "data/meshdata/000/coacd/decomposed.obj"
        )
        config = load_x2_mesh_config()
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                mesh_path=mesh_path,
                side="front",
                num_grasps=1,
                batch_size=1,
                n_contact=4,
                n_iterations=1,
                seed=43,
                device="cpu",
                output=Path(directory),
                config=None,
                object_scale=1.0,
                surface_samples=8,
                freeze_thumb=False,
                overwrite=True,
            )
            records, summary = run(args)
            saved = json.loads(
                Path(summary["output_files"][0]).read_text(encoding="utf-8")
            )

        self.assertEqual(records[0], saved)
        self.assertEqual(
            summary["dense_hand_object_gate"],
            {
                "evaluation_mode": "dense_bidirectional",
                "hand_surface_samples_per_set": (
                    DENSE_HAND_SURFACE_SAMPLES_PER_SET
                ),
                "hand_surface_samples_per_link": (
                    3 * DENSE_HAND_SURFACE_SAMPLES_PER_SET
                ),
                "hand_surface_point_count": (
                    17 * 3 * DENSE_HAND_SURFACE_SAMPLES_PER_SET
                ),
                "object_surface_samples": DENSE_OBJECT_SURFACE_SAMPLES,
                "threshold": HAND_OBJECT_PENETRATION_THRESHOLD,
                "strict_less_than": True,
                "sample_count": 1,
                "evaluated_count": 1,
                "feasible_count": int(
                    saved["hand_object_penetration"]["feasible"]
                ),
            },
        )
        self.assertEqual(
            saved["pipeline_revision"],
            "x2_mesh_grasp_unselected_finger_side_v6",
        )
        collision = saved["self_collision"]
        self.assertEqual(
            set(collision),
            {
                "maximum_penetration",
                "total_penetration",
                "worst_pair",
                "feasible",
                "threshold",
            },
        )
        self.assertEqual(
            collision["feasible"],
            collision["maximum_penetration"] <= collision["threshold"],
        )
        self.assertEqual(
            collision["threshold"],
            float(config.require("self_collision.feasibility_threshold")),
        )
        self.assertEqual(
            saved["energy"]["terms"]["E_spen"],
            saved["energy"]["terms"]["E_spen_capsule"],
        )
        self.assertIn(
            saved["optimization"]["restored_checkpoint"],
            (
                "bidirectional_feasible",
                "bidirectional_fallback",
                "feasible",
                "fallback",
            ),
        )
        self.assertEqual(
            saved["optimization"]["feasible_checkpoint_found"],
            saved["self_collision"]["feasible"],
        )
        penetration = saved["hand_object_penetration"]
        self.assertEqual(penetration["evaluation_mode"], "dense_bidirectional")
        self.assertIs(penetration["evaluated"], True)
        self.assertEqual(
            penetration["hand_surface_samples_per_set"],
            DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        )
        self.assertEqual(
            penetration["hand_surface_samples_per_link"],
            3 * DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        )
        self.assertEqual(
            penetration["hand_surface_point_count"],
            17 * 3 * DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        )
        self.assertEqual(
            penetration["object_surface_samples"],
            DENSE_OBJECT_SURFACE_SAMPLES,
        )
        self.assertEqual(
            penetration["threshold"], HAND_OBJECT_PENETRATION_THRESHOLD
        )
        self.assertEqual(
            penetration["maximum_penetration"],
            max(
                penetration["forward_maximum_penetration"],
                penetration["reverse_maximum_penetration"],
            ),
        )
        self.assertEqual(
            penetration["feasible"],
            penetration["maximum_penetration"]
            < HAND_OBJECT_PENETRATION_THRESHOLD,
        )
        self.assertEqual(
            saved["maximum_penetration"], penetration["maximum_penetration"]
        )
        self.assertEqual(
            saved["optimization"]["bidirectional_feasible_checkpoint_found"],
            penetration["feasible"],
        )
        self.assertEqual(
            saved["optimization"]["bidirectional_checkpoint_period"],
            BIDIRECTIONAL_CHECKPOINT_PERIOD,
        )
        self.assertEqual(
            saved["optimization"]["bidirectional_checkpoint_pool_capacity"],
            BIDIRECTIONAL_CHECKPOINT_TOP_K,
        )
        self.assertEqual(
            saved["optimization"]["dense_hand_surface_samples_per_set"],
            DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        )
        self.assertEqual(
            saved["optimization"]["dense_hand_surface_samples_per_link"],
            3 * DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        )
        self.assertEqual(
            saved["optimization"]["dense_hand_surface_point_count"],
            17 * 3 * DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        )
        self.assertEqual(
            saved["optimization"]["dense_object_surface_samples"],
            DENSE_OBJECT_SURFACE_SAMPLES,
        )
        self.assertGreaterEqual(
            saved["optimization"]["dense_bidirectional_query_calls"], 1
        )
        self.assertLessEqual(
            saved["optimization"]["dense_bidirectional_query_calls"],
            BIDIRECTIONAL_CHECKPOINT_TOP_K + 2,
        )
        self.assertGreaterEqual(
            saved["optimization"]["dense_bidirectional_rows_evaluated"], 1
        )
        self.assertGreaterEqual(saved["optimization"]["restored_step"], 0)
        candidate_path = config.configured_path(
            "contact_candidates.path", must_exist=True
        )
        expected_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        self.assertEqual(
            saved["provenance"]["contact_candidates_sha256"], expected_sha
        )


if __name__ == "__main__":
    unittest.main()
