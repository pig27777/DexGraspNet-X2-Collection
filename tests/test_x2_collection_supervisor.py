from __future__ import annotations

import fcntl
import json
import tempfile
import unittest
from pathlib import Path

from scripts.supervise_x2_valid_collection import (
    PROJECT_ROOT,
    audit_report_proves_complete,
    collector_lock_is_held,
    formal_audit_command,
    formal_collector_command,
    manifest_proves_complete,
    route_counts,
)


class X2CollectionSupervisorTest(unittest.TestCase):
    def test_lock_probe_distinguishes_free_and_held_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / ".collector.lock"
            self.assertFalse(collector_lock_is_held(lock_path))
            with lock_path.open("a+", encoding="utf-8") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertTrue(collector_lock_is_held(lock_path))

    def test_manifest_requires_every_formal_headline_quota(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            payload = {
                "passed": True,
                "target_valid": 5000,
                "valid_count": 5000,
                "side_finger_counts": {
                    side: {str(value): 500 for value in range(1, 6)}
                    for side in ("front", "back")
                },
                "paired_entry_count": 2000,
                "single_side_five_finger_entry_count": 1000,
                "required_general_object_count": 30,
                "covered_general_object_count": 30,
                "records": [{} for _ in range(5000)],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(manifest_proves_complete(path))
            payload["side_finger_counts"]["front"]["3"] = 499
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(manifest_proves_complete(path))

    def test_final_audit_report_is_bound_to_current_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            report = root / "final_audit.json"
            manifest.write_text("{}", encoding="utf-8")
            import hashlib

            manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
            report.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "valid_count": 5000,
                        "paired_entry_count": 2000,
                        "single_side_five_finger_entry_count": 1000,
                        "required_general_object_count": 30,
                        "covered_general_object_count": 30,
                        "audited_record_sha256_count": 5000,
                        "manifest_sha256": manifest_sha256,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(audit_report_proves_complete(report, manifest))
            manifest.write_text('{"changed": true}', encoding="utf-8")
            self.assertFalse(audit_report_proves_complete(report, manifest))

    def test_route_counts_only_route_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for route, count in (("raw", 3), ("valid", 2), ("failed", 1)):
                directory = root / "attempts" / "attempt_0000" / "object" / route
                directory.mkdir(parents=True)
                for index in range(count):
                    (directory / f"{index}.json").write_text("{}", encoding="utf-8")
            (root / "attempts" / "attempt_0000" / "attempt.json").write_text(
                "{}", encoding="utf-8"
            )
            self.assertEqual(route_counts(root), (3, 2, 1))

    def test_formal_command_preserves_collection_contract(self) -> None:
        command = formal_collector_command(
            conda_executable=Path("/opt/conda"), output_root=Path("/tmp/output")
        )
        self.assertEqual(command[:6], [
            "/opt/conda", "run", "-n", "isaaclab", "--no-capture-output", "python"
        ])
        self.assertIn(str(PROJECT_ROOT / "scripts" / "collect_x2_valid_dataset.py"), command)
        for flag, value in (
            ("--target-valid", "5000"),
            ("--n-iterations", "6000"),
            ("--jobs", "2"),
            ("--validation-batch-size", "32"),
            ("--sim-steps", "100"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)

        audit_command = formal_audit_command(
            conda_executable=Path("/opt/conda"), output_root=Path("/tmp/output")
        )
        self.assertIn(
            str(PROJECT_ROOT / "scripts" / "audit_x2_valid_dataset.py"),
            audit_command,
        )
        self.assertEqual(
            audit_command[audit_command.index("--output-root") + 1],
            "/tmp/output",
        )


if __name__ == "__main__":
    unittest.main()
