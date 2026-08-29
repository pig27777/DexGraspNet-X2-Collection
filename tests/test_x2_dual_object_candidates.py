from __future__ import annotations

import unittest
from pathlib import Path

from scripts.build_x2_dual_object_candidates import (
    _combined_actuators,
    _combined_joints,
    _object_pose_in_hand_frame,
    _round_robin_different_object_pairs,
)
from scripts.collect_x2_valid_dataset import ValidCandidate


def _candidate(
    name: str, side: str, count: int, fingers: set[str], object_id: str
) -> ValidCandidate:
    return ValidCandidate(
        path=Path(f"/tmp/{name}.json"),
        side=side,
        finger_count=count,
        finger_names=frozenset(fingers),
        object_id=object_id,
        object_scale=1.0,
    )


class DualObjectCandidateTests(unittest.TestCase):
    def test_object_pose_is_inverse_of_hand_pose(self) -> None:
        payload = {
            "hand_pose": {
                "translation": [1.0, 2.0, 3.0],
                "rotation_matrix": [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        }
        result = _object_pose_in_hand_frame(payload, Path("source.json"))
        self.assertEqual(
            result["rotation_matrix"],
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        )
        self.assertEqual(result["translation"], [-2.0, 1.0, -3.0])

    def test_combined_actuators_take_values_from_finger_owner(self) -> None:
        names = [
            "rh_LFJ3",
            "rh_RFJ3",
            "rh_MFJ3",
            "rh_FFJ3",
            "rh_THJ4",
        ]
        right_payload = {"actuator_names": names, "actuator": [1, 2, 3, 4, 5]}
        left_payload = {"actuator_names": names, "actuator": [-1, -2, -3, -4, -5]}
        right = _candidate(
            "right", "front", 2, {"index", "thumb"}, "object-a"
        )
        left = _candidate(
            "left", "back", 3, {"middle", "ring", "little"}, "object-b"
        )
        result_names, values, owners = _combined_actuators(
            right_payload, left_payload, right, left
        )
        self.assertEqual(result_names, names)
        self.assertEqual(values, [-1.0, -2.0, -3.0, 4.0, 5.0])
        self.assertEqual(owners["index"], "right")
        self.assertEqual(owners["little"], "left")

    def test_combined_joints_take_values_from_finger_owner(self) -> None:
        names = [
            "rh_LFJ3",
            "rh_RFJ3",
            "rh_MFJ3",
            "rh_FFJ3",
            "rh_THJ4",
            "rh_THJ1",
        ]
        right_payload = {"joint_names": names, "joint": [1, 2, 3, 4, 5, 6]}
        left_payload = {"joint_names": names, "joint": [-1, -2, -3, -4, -5, -6]}
        right = _candidate(
            "right", "front", 2, {"index", "thumb"}, "object-a"
        )
        left = _candidate(
            "left", "back", 3, {"middle", "ring", "little"}, "object-b"
        )
        result_names, values = _combined_joints(
            right_payload, left_payload, right, left
        )
        self.assertEqual(result_names, names)
        self.assertEqual(values, [-1.0, -2.0, -3.0, 4.0, 5.0, 6.0])

    def test_pairing_requires_different_objects_and_disjoint_fingers(self) -> None:
        right = [
            _candidate("r-a", "front", 1, {"index"}, "object-a"),
            _candidate("r-b", "front", 1, {"middle"}, "object-b"),
        ]
        left = [
            _candidate(
                "l-b",
                "back",
                4,
                {"middle", "ring", "little", "thumb"},
                "object-b",
            ),
            _candidate(
                "l-c",
                "back",
                4,
                {"index", "ring", "little", "thumb"},
                "object-c",
            ),
        ]
        pairs = _round_robin_different_object_pairs(right, left, None)
        self.assertEqual(len(pairs), 2)
        for first, second in pairs:
            self.assertNotEqual(first.object_id, second.object_id)
            self.assertTrue(first.finger_names.isdisjoint(second.finger_names))

    def test_pairing_reaches_maximum_when_object_counts_are_imbalanced(self) -> None:
        right = [
            _candidate(f"r-a-{index}", "front", 1, {"index"}, "object-a")
            for index in range(2)
        ] + [_candidate("r-b", "front", 1, {"index"}, "object-b")]
        left = [
            _candidate(
                "l-a",
                "back",
                4,
                {"middle", "ring", "little", "thumb"},
                "object-a",
            )
        ] + [
            _candidate(
                f"l-b-{index}",
                "back",
                4,
                {"middle", "ring", "little", "thumb"},
                "object-b",
            )
            for index in range(2)
        ]
        pairs = _round_robin_different_object_pairs(right, left, None)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(len({first.path for first, _ in pairs}), 3)
        self.assertEqual(len({second.path for _, second in pairs}), 3)


if __name__ == "__main__":
    unittest.main()
