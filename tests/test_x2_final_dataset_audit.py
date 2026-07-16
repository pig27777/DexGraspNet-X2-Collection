from __future__ import annotations

import unittest

from scripts.audit_x2_valid_dataset import _audit_pairing
from scripts.collect_x2_valid_dataset import ValidDatasetError


class X2FinalDatasetAuditTest(unittest.TestCase):
    @staticmethod
    def _balanced_records() -> tuple[list[dict], list[dict]]:
        fingers = ["index", "middle", "ring", "little", "thumb"]
        records: list[dict] = []
        merged: list[dict] = []
        for front_count in (1, 2, 3, 4):
            back_count = 5 - front_count
            pair_id = f"front_f{front_count}_back_f{back_count}_000000"
            front_names = fingers[:front_count]
            back_names = fingers[front_count:]
            records.extend(
                [
                    {
                        "path": f"/final/front/f{front_count}/x2_front_f{front_count}_000000.json",
                        "side": "front",
                        "finger_count": front_count,
                        "finger_names": front_names,
                        "object_id": "object",
                        "pair_id": pair_id,
                    },
                    {
                        "path": f"/final/back/f{back_count}/x2_back_f{back_count}_000000.json",
                        "side": "back",
                        "finger_count": back_count,
                        "finger_names": back_names,
                        "object_id": "object",
                        "pair_id": pair_id,
                    },
                ]
            )
            merged.append(
                {
                    "pair_id": pair_id,
                    "object_id": "object",
                    "front_finger_count": front_count,
                    "front_finger_names": sorted(front_names),
                    "back_finger_count": back_count,
                    "back_finger_names": sorted(back_names),
                    "disjoint": True,
                }
            )
        for side in ("front", "back"):
            records.append(
                {
                    "path": f"/final/{side}/f5/x2_{side}_f5_000000.json",
                    "side": side,
                    "finger_count": 5,
                    "finger_names": fingers,
                    "object_id": "object",
                    "pair_id": None,
                }
            )
            merged.append(
                {
                    "pair_id": None,
                    "object_id": "object",
                    f"{side}_finger_count": 5,
                    f"{side}_finger_names": sorted(fingers),
                    "opposite_side": None,
                    "index": 0,
                }
            )
        return records, merged

    def test_pairing_audit_accepts_exact_complementary_dataset(self) -> None:
        records, merged = self._balanced_records()
        report = _audit_pairing(records, merged, per_side_finger_target=1)
        self.assertEqual(report["paired_entry_count"], 4)
        self.assertEqual(report["single_side_five_finger_entry_count"], 2)

    def test_pairing_audit_rejects_overlapping_fingers(self) -> None:
        records, merged = self._balanced_records()
        pair_id = "front_f1_back_f4_000000"
        back = next(
            record
            for record in records
            if record.get("pair_id") == pair_id and record["side"] == "back"
        )
        back["finger_names"] = ["index", "middle", "ring", "little"]
        with self.assertRaisesRegex(ValidDatasetError, "overlapping"):
            _audit_pairing(records, merged, per_side_finger_target=1)


if __name__ == "__main__":
    unittest.main()
