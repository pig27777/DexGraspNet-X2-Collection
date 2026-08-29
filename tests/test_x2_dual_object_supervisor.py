from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.supervise_x2_dual_object_validation import (
    _completed_attempt_count,
    dual_composition_ready,
    formal_collection_ready,
    validation_complete,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _dual_records(per_stratum: int = 500) -> list[dict[str, int]]:
    records = []
    for right in (1, 2, 3, 4):
        for _ in range(per_stratum):
            records.append(
                {
                    "right_finger_count": right,
                    "left_finger_count": 5 - right,
                }
            )
    return records


class X2DualObjectSupervisorTests(unittest.TestCase):
    def test_completed_attempt_count_ignores_incomplete_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "attempts" / "attempt_0000" / "complete.json", {})
            (root / "attempts" / "attempt_0001").mkdir(parents=True)
            self.assertEqual(_completed_attempt_count(root), 1)

    def test_readiness_requires_formal_and_four_complete_strata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = root / "formal"
            dual = root / "dual"
            self.assertFalse(formal_collection_ready(formal))
            _write(formal / "manifest.json", {"passed": True, "valid_count": 5000})
            self.assertTrue(formal_collection_ready(formal))
            _write(
                dual / "manifest.json",
                {
                    "protocol_revision": (
                        "x2_right_left_dual_object_warm_start_v1"
                    ),
                    "dual_object_status": "not_validated",
                    "formal_source_completion_required": True,
                    "dual_object_candidates": _dual_records(),
                },
            )
            self.assertTrue(dual_composition_ready(dual))
            manifest = json.loads(
                (dual / "manifest.json").read_text(encoding="utf-8")
            )
            manifest["dual_object_candidates"] = _dual_records(503)
            _write(dual / "manifest.json", manifest)
            self.assertTrue(dual_composition_ready(dual))

    def test_validation_summary_is_bound_to_current_composition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            records = _dual_records(503)
            candidate_count = len(records)
            _write(
                manifest,
                {
                    "manifest": "current",
                    "dual_object_candidates": records,
                },
            )
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            _write(
                root / "physx_validation" / "summary.json",
                {
                    "passed": True,
                    "protocol_revision": (
                        "x2_dual_object_six_orientation_physx_v1"
                    ),
                    "candidate_count": candidate_count,
                    "valid_count": 123,
                    "failed_count": candidate_count - 123,
                    "source_manifest_sha256": digest,
                },
            )
            self.assertTrue(validation_complete(root))


if __name__ == "__main__":
    unittest.main()
