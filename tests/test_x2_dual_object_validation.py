from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from grasp_generation.x2_dual_object_validation import (
    DUAL_VALIDATION_PROTOCOL_REVISION,
    EXPECTED_GRAVITY_NAMES,
    X2DualObjectCandidate,
    X2DualValidationError,
    existing_validation_output,
    gravity_vectors_shared_hand,
    make_validation_record,
    write_validation_record,
)
from scripts.validate_x2_dual_object_physx import (
    _matrix_to_quaternion_xyzw,
    _prepare_batch,
)


def _candidate(path: Path) -> X2DualObjectCandidate:
    right = {
        "object_id": "object-a",
        "mesh_path": str(path.parent / "a.obj"),
        "scale": 1.0,
        "finger_names": ["index"],
        "pose_in_shared_hand_frame": {
            "translation": [0.1, 0.0, 0.0],
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
    }
    left = {
        "object_id": "object-b",
        "mesh_path": str(path.parent / "b.obj"),
        "scale": 1.0,
        "finger_names": ["little", "middle", "ring", "thumb"],
        "pose_in_shared_hand_frame": {
            "translation": [-0.1, 0.0, 0.0],
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
    }
    record = {
        "candidate_id": "right_f1_left_f4_000000",
        "objects": [right, left],
        "hand": {
            "joint": [0.0] * 16,
            "actuator": [0.0] * 12,
        },
        "dual_object_validation": {"status": "not_run"},
    }
    return X2DualObjectCandidate(
        path=path,
        sha256="source-sha",
        record=record,
        right=right,
        left=left,
    )


def _orientation(name: str) -> dict[str, object]:
    return {
        "name": name,
        "passed": True,
        "finite": True,
        "objects": {
            "right": {"hand_contact": True, "passed": True},
            "left": {"hand_contact": True, "passed": True},
        },
    }


class X2DualObjectValidationTests(unittest.TestCase):
    def test_matrix_to_quaternion_uses_isaac_xyzw_order(self) -> None:
        quaternion = _matrix_to_quaternion_xyzw(np.eye(3))
        np.testing.assert_allclose(
            quaternion, np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1.0e-12
        )

    def test_prepare_batch_repeats_both_object_poses_for_six_directions(
        self,
    ) -> None:
        import torch

        candidate = _candidate(Path("/tmp/candidate.json"))
        origins = torch.zeros((6, 3), dtype=torch.float32)
        (
            hand,
            right,
            left,
            joint,
            target,
            gravity,
            active,
        ) = _prepare_batch(
            [candidate],
            capacity_samples=1,
            env_origins=origins,
            device="cpu",
        )
        self.assertEqual(active, 6)
        self.assertEqual(tuple(hand.shape), (6, 7))
        self.assertTrue(torch.allclose(right[:, 0], torch.full((6,), 0.1)))
        self.assertTrue(torch.allclose(left[:, 0], torch.full((6,), -0.1)))
        self.assertEqual(tuple(joint.shape), (6, 16))
        self.assertEqual(tuple(target.shape), (6, 12))
        self.assertEqual(tuple(gravity.shape), (6, 3))
        self.assertTrue(torch.isfinite(gravity).all())

    def test_gravity_vectors_are_six_finite_equal_magnitude_directions(self) -> None:
        values = gravity_vectors_shared_hand(9.8)
        self.assertEqual(values.shape, (6, 3))
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(
            np.linalg.norm(values, axis=1),
            np.full(6, 9.8),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_pass_requires_all_six_orientation_proofs(self) -> None:
        candidate = _candidate(Path("/tmp/candidate.json"))
        with self.assertRaises(X2DualValidationError):
            make_validation_record(
                candidate,
                passed=True,
                simulation_ran=True,
                static_preflight={"passed": True},
                orientations=[
                    _orientation(name) for name in EXPECTED_GRAVITY_NAMES[:-1]
                ],
                runtime={},
                failure_reasons=[],
            )
        record = make_validation_record(
            candidate,
            passed=True,
            simulation_ran=True,
            static_preflight={"passed": True},
            orientations=[
                _orientation(name) for name in EXPECTED_GRAVITY_NAMES
            ],
            runtime={},
            failure_reasons=[],
        )
        self.assertTrue(record["dual_object_success"])
        self.assertEqual(
            record["dual_object_validation"]["passed_orientation_count"], 6
        )

    def test_failed_static_preflight_does_not_claim_simulation(self) -> None:
        candidate = _candidate(Path("/tmp/candidate.json"))
        record = make_validation_record(
            candidate,
            passed=False,
            simulation_ran=False,
            static_preflight={"passed": False},
            orientations=[],
            runtime={},
            failure_reasons=["object_object_initial_penetration"],
        )
        self.assertFalse(record["dual_object_success"])
        self.assertEqual(
            record["dual_object_validation"]["failure_reasons"],
            ["object_object_initial_penetration"],
        )

    def test_atomic_route_and_resume_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = _candidate(root / "candidate.json")
            record = make_validation_record(
                candidate,
                passed=True,
                simulation_ran=True,
                static_preflight={"passed": True},
                orientations=[
                    _orientation(name) for name in EXPECTED_GRAVITY_NAMES
                ],
                runtime={},
                failure_reasons=[],
            )
            output = write_validation_record(candidate, record, root / "routes")
            self.assertIn("/valid/right_f1_left_f4/", str(output))
            resumed = existing_validation_output(candidate, root / "routes")
            self.assertEqual(resumed, output)
            self.assertEqual(
                record["dual_object_validation"]["protocol_revision"],
                DUAL_VALIDATION_PROTOCOL_REVISION,
            )


if __name__ == "__main__":
    unittest.main()
