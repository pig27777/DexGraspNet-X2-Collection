from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from scripts.watch_x2_collection import (
    DashboardSnapshot,
    ProcessStatus,
    _object_id_from_command,
    parse_collector_progress,
    render_dashboard,
    scan_attempts,
)


class X2CollectionWatchTest(unittest.TestCase):
    def test_scan_attempts_uses_proof_and_ignores_staging_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt0 = root / "attempts" / "attempt_0000"
            attempt0.mkdir(parents=True)
            (attempt0 / "attempt.json").write_text(
                json.dumps({"raw_target": 10, "seed": 0}), encoding="utf-8"
            )
            (attempt0 / "complete.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "raw_count": 10,
                        "valid_count": 3,
                        "failed_count": 7,
                    }
                ),
                encoding="utf-8",
            )

            attempt1 = root / "attempts" / "attempt_0001"
            (attempt1 / "object" / "front" / "raw").mkdir(parents=True)
            (attempt1 / "object" / "front" / "valid").mkdir(parents=True)
            (attempt1 / "object" / "front" / "failed").mkdir(parents=True)
            (attempt1 / ".staging" / "object" / "raw").mkdir(parents=True)
            (attempt1 / "attempt.json").write_text(
                json.dumps({"raw_target": 20, "seed": 1009}), encoding="utf-8"
            )
            for index in range(4):
                (attempt1 / "object" / "front" / "raw" / f"{index}.json").write_text(
                    "{}", encoding="utf-8"
                )
            (attempt1 / "object" / "front" / "valid" / "0.json").write_text(
                "{}", encoding="utf-8"
            )
            (attempt1 / "object" / "front" / "failed" / "1.json").write_text(
                "{}", encoding="utf-8"
            )
            (attempt1 / ".staging" / "object" / "raw" / "hidden.json").write_text(
                "{}", encoding="utf-8"
            )

            statuses = scan_attempts(root)
            self.assertEqual(len(statuses), 2)
            self.assertEqual(
                (statuses[0].phase, statuses[0].raw, statuses[0].valid, statuses[0].failed),
                ("complete", 10, 3, 7),
            )
            self.assertEqual(
                (statuses[1].phase, statuses[1].raw, statuses[1].valid, statuses[1].failed),
                ("generating", 4, 1, 1),
            )

    def test_parse_collector_progress_uses_latest_structured_lines(self) -> None:
        pairing, f5, reusable, regenerate = parse_collector_progress(
            (
                "[collector] pairing={1: 32, 2: 35, 3: 41, 4: 33}, "
                "f5={'front': 74, 'back': 63}, finger_targets={1: 1}",
                "[resume] reusable_groups=12 regenerate_groups=408",
            )
        )
        self.assertEqual(pairing, {1: 32, 2: 35, 3: 41, 4: 33})
        self.assertEqual(f5, {"front": 74, "back": 63})
        self.assertEqual((reusable, regenerate), (12, 408))

    def test_object_id_handles_general_and_primitive_meshes(self) -> None:
        self.assertEqual(
            _object_id_from_command(
                "python worker.py --mesh-path /data/meshdata/030/coacd/decomposed.obj"
            ),
            "030",
        )
        self.assertEqual(
            _object_id_from_command(
                "python worker.py --mesh-path /data/x2_primitives/sphere/sphere_r020.obj"
            ),
            "sphere_r020",
        )

    def test_render_distinguishes_candidate_pool_from_final_manifest(self) -> None:
        snapshot = DashboardSnapshot(
            captured_at=datetime.now().astimezone(),
            service_state="active",
            attempts=(),
            processes=(
                ProcessStatus(
                    pid=123,
                    elapsed_seconds=90,
                    cpu_percent=100.0,
                    rss_kib=1024,
                    role="generator",
                    object_id="030",
                    command="worker",
                ),
            ),
            gpu_lines=("GPU 0 test",),
            manifest_complete=False,
            manifest_valid=0,
            final_audit_complete=False,
            completed_valid_pool=642,
            pairing={1: 32, 2: 35, 3: 41, 4: 33},
            f5_counts={"front": 74, "back": 63},
            reusable_groups=0,
            regenerate_groups=420,
            recent_events=("[collector] test",),
        )
        rendered = render_dashboard(snapshot)
        self.assertIn("manifest pending", rendered)
        self.assertIn("audited candidate pool 642/5000", rendered)
        self.assertIn("f1=32/500", rendered)
        self.assertIn("generator", rendered)
        self.assertNotIn("FINAL: COMPLETE", rendered)


if __name__ == "__main__":
    unittest.main()
