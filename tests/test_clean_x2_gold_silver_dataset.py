from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.clean_x2_gold_silver_dataset import (
    EXPECTED_ORIENTATIONS,
    _canonical_quaternion,
    discover_snapshot_paths,
    load_gold_archive,
)


class X2GoldSilverCleaningTests(unittest.TestCase):
    def test_snapshot_inventory_excludes_only_requested_general_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = (
                root
                / "general"
                / "front"
                / "raw"
                / "078_f1_front_000000.json",
                root
                / "general"
                / "back"
                / "raw"
                / "075_f2_back_000001.json",
                root
                / "cube"
                / "front"
                / "raw"
                / "cube_e040_f3_front_000002.json",
            )
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            result = discover_snapshot_paths(
                root,
                excluded_general_objects=("078",),
                expected_count=2,
            )
            self.assertEqual(
                [path.name for path in result],
                [
                    "cube_e040_f3_front_000002.json",
                    "075_f2_back_000001.json",
                ],
            )

    def test_gold_archive_requires_exact_six_orientation_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "gold.zip"
            basename = "cube_e040_f1_front_000000.json"
            payload = {
                "active_side": "front",
                "success": True,
                "simulation_success": True,
                "finger_participation": {
                    "target_count": 1,
                    "actual_count": 1,
                    "finger_names": ["index"],
                },
                "validation": {
                    "status": "passed",
                    "backend": "isaac_sim_physx",
                    "required_orientation_count": 6,
                    "passed_orientation_count": 6,
                    "orientations": [
                        {
                            "name": name,
                            "passed": True,
                            "finite": True,
                            "hand_object_contact": True,
                        }
                        for name in EXPECTED_ORIENTATIONS
                    ],
                    "preflight": {
                        "collision_aware_closing_raw_passed": True,
                        "self_collision_passed": True,
                        "hand_object_passed": True,
                    },
                },
            }
            raw = json.dumps(payload).encode("utf-8")
            sha256 = hashlib.sha256(raw).hexdigest()
            sample_path = f"samples/cube/front/valid/{basename}"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(
                    "frozen/manifest.json",
                    json.dumps({"sample_count": 1}),
                )
                bundle.writestr(
                    "frozen/SHA256SUMS",
                    f"{sha256}  {sample_path}\n",
                )
                bundle.writestr(f"frozen/{sample_path}", raw)
            members, audit = load_gold_archive(archive, expected_count=1)
            self.assertEqual(set(members), {basename})
            self.assertEqual(members[basename].sha256, sha256)
            self.assertEqual(audit["member_count"], 1)

    def test_quaternion_sign_is_canonical_for_duplicate_keys(self) -> None:
        positive = _canonical_quaternion((0.5, 0.5, 0.5, 0.5))
        negative = _canonical_quaternion((-0.5, -0.5, -0.5, -0.5))
        self.assertEqual(positive, negative)


if __name__ == "__main__":
    unittest.main()
