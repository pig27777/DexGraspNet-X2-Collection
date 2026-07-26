from __future__ import annotations

import fcntl
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.supervise_x2_valid_collection import (
    PROJECT_ROOT,
    _wait_for_collector_child,
    audit_report_proves_complete,
    collector_lock_is_held,
    formal_audit_command,
    formal_collector_command,
    manifest_proves_complete,
    route_counts,
)


class X2CollectionSupervisorTest(unittest.TestCase):
    def test_owned_child_wait_logs_route_counts_at_health_interval(self) -> None:
        child = Mock()
        child.pid = 1234
        child.wait.side_effect = [
            subprocess.TimeoutExpired("collector", 5.0),
            subprocess.TimeoutExpired("collector", 5.0),
            0,
        ]

        with (
            patch(
                "scripts.supervise_x2_valid_collection.time.monotonic",
                side_effect=[10.0, 20.0],
            ),
            patch(
                "scripts.supervise_x2_valid_collection.route_counts",
                side_effect=[(11, 2, 9), (23, 4, 19)],
            ) as mocked_counts,
            patch("scripts.supervise_x2_valid_collection._log") as mocked_log,
        ):
            return_code, last_health_log = _wait_for_collector_child(
                child,
                output_root=Path("/tmp/output"),
                supervisor_log=Mock(),
                poll_seconds=5.0,
                health_log_seconds=10.0,
                last_health_log=0.0,
            )

        self.assertEqual((return_code, last_health_log), (0, 20.0))
        self.assertEqual(child.wait.call_count, 3)
        for call in child.wait.call_args_list:
            self.assertEqual(call.kwargs, {"timeout": 5.0})
        self.assertEqual(mocked_counts.call_count, 2)
        messages = [call.args[1] for call in mocked_log.call_args_list]
        self.assertIn(
            "collector child running pid=1234; published raw/valid/failed=11/2/9",
            messages,
        )
        self.assertIn(
            "collector child running pid=1234; published raw/valid/failed=23/4/19",
            messages,
        )
        child.terminate.assert_not_called()
        child.kill.assert_not_called()
        child.send_signal.assert_not_called()

    def test_owned_child_wait_preserves_immediate_exit_without_intervention(
        self,
    ) -> None:
        child = Mock()
        child.pid = 5678
        child.wait.return_value = 23

        with (
            patch("scripts.supervise_x2_valid_collection.route_counts") as counts,
            patch("scripts.supervise_x2_valid_collection._log") as mocked_log,
        ):
            return_code, last_health_log = _wait_for_collector_child(
                child,
                output_root=Path("/tmp/output"),
                supervisor_log=Mock(),
                poll_seconds=15.0,
                health_log_seconds=300.0,
                last_health_log=42.0,
            )

        self.assertEqual((return_code, last_health_log), (23, 42.0))
        child.wait.assert_called_once_with(timeout=15.0)
        counts.assert_not_called()
        mocked_log.assert_not_called()
        child.terminate.assert_not_called()
        child.kill.assert_not_called()
        child.send_signal.assert_not_called()

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
            ("--validation-batch-size", "8"),
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
