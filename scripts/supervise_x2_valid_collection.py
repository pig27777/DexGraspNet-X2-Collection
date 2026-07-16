#!/usr/bin/env python3
"""Keep the formal X2 5000-valid collector alive across terminal restarts.

The collector itself owns ``.collector.lock``.  This supervisor never takes
that lock for longer than a non-blocking liveness probe, so it can safely watch
an already-running collector.  If the lock becomes free before a complete
manifest exists, it starts the exact formal collection command again; the
collector then resumes the existing attempt from its atomic file checkpoints.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "x2_valid_5000"
DEFAULT_CONDA = Path.home() / "miniconda3" / "bin" / "conda"
FORMAL_TARGET_VALID = 5000
FORMAL_GENERAL_MESH_COUNT = 30
SIDES = ("front", "back")
FINGER_COUNTS = (1, 2, 3, 4, 5)


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(stream: TextIO, message: str) -> None:
    stream.write(f"[{_timestamp()}] {message}\n")
    stream.flush()
    os.fsync(stream.fileno())


def collector_lock_is_held(lock_path: Path) -> bool:
    """Return whether a live collector currently owns ``lock_path``."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return False


def manifest_proves_complete(path: Path) -> bool:
    """Apply the cheap, independent completion checks needed to stop restarts.

    The collector performs the authoritative record-by-record audit before it
    writes the manifest.  This guard deliberately checks the headline quotas
    again so a stray or partial JSON file cannot stop the supervisor.
    """

    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    expected_counts = {str(value): 500 for value in FINGER_COUNTS}
    records = payload.get("records")
    return (
        payload.get("passed") is True
        and payload.get("target_valid") == FORMAL_TARGET_VALID
        and payload.get("valid_count") == FORMAL_TARGET_VALID
        and payload.get("side_finger_counts")
        == {side: expected_counts for side in SIDES}
        and payload.get("paired_entry_count") == 2000
        and payload.get("single_side_five_finger_entry_count") == 1000
        and payload.get("required_general_object_count")
        == FORMAL_GENERAL_MESH_COUNT
        and payload.get("covered_general_object_count")
        == FORMAL_GENERAL_MESH_COUNT
        and isinstance(records, list)
        and len(records) == FORMAL_TARGET_VALID
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_report_proves_complete(report_path: Path, manifest_path: Path) -> bool:
    try:
        report: Any = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(report, dict)
        and report.get("passed") is True
        and report.get("valid_count") == FORMAL_TARGET_VALID
        and report.get("paired_entry_count") == 2000
        and report.get("single_side_five_finger_entry_count") == 1000
        and report.get("required_general_object_count")
        == FORMAL_GENERAL_MESH_COUNT
        and report.get("covered_general_object_count")
        == FORMAL_GENERAL_MESH_COUNT
        and report.get("audited_record_sha256_count") == FORMAL_TARGET_VALID
        and manifest_path.is_file()
        and report.get("manifest_sha256") == _file_sha256(manifest_path)
    )


def route_counts(output_root: Path) -> tuple[int, int, int]:
    counts = {"raw": 0, "valid": 0, "failed": 0}
    attempts = output_root / "attempts"
    if not attempts.is_dir():
        return 0, 0, 0
    for path in attempts.rglob("*.json"):
        if path.parent.name in counts:
            counts[path.parent.name] += 1
    return counts["raw"], counts["valid"], counts["failed"]


def formal_collector_command(
    *, conda_executable: Path, output_root: Path
) -> list[str]:
    return [
        str(conda_executable),
        "run",
        "-n",
        "isaaclab",
        "--no-capture-output",
        "python",
        str(PROJECT_ROOT / "scripts" / "collect_x2_valid_dataset.py"),
        "--target-valid",
        "5000",
        "--n-iterations",
        "6000",
        "--generation-device",
        "cuda",
        "--jobs",
        "2",
        "--validation-device",
        "cuda:0",
        "--validation-batch-size",
        "32",
        "--sim-steps",
        "100",
        "--general-mesh-root",
        str(PROJECT_ROOT / "data" / "meshdata"),
        "--output-root",
        str(output_root),
    ]


def formal_audit_command(
    *, conda_executable: Path, output_root: Path
) -> list[str]:
    return [
        str(conda_executable),
        "run",
        "-n",
        "isaaclab",
        "--no-capture-output",
        "python",
        str(PROJECT_ROOT / "scripts" / "audit_x2_valid_dataset.py"),
        "--output-root",
        str(output_root),
        "--general-mesh-root",
        str(PROJECT_ROOT / "data" / "meshdata"),
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--conda-executable", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--health-log-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0.0 or args.health_log_seconds <= 0.0:
        parser.error("poll and health-log intervals must be positive")
    return args


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    conda_executable = args.conda_executable.expanduser().resolve()
    if not conda_executable.is_file():
        raise FileNotFoundError(f"Conda executable does not exist: {conda_executable}")

    lock_path = output_root / ".collector.lock"
    stop_path = output_root / ".stop_supervisor"
    manifest_path = output_root / "manifest.json"
    final_audit_path = output_root / "final_audit.json"
    failed_audit_path = output_root / "final_audit_failed.json"
    audit_stderr_path = output_root / "final_audit.stderr.log"
    supervisor_log_path = output_root / "collector_supervisor.log"
    collector_log_path = output_root / "collector_console.log"
    command = formal_collector_command(
        conda_executable=conda_executable, output_root=output_root
    )
    audit_command = formal_audit_command(
        conda_executable=conda_executable, output_root=output_root
    )

    last_health_log = 0.0
    restart_delay = args.poll_seconds
    last_failed_manifest_sha256: str | None = None
    with supervisor_log_path.open("a", encoding="utf-8") as supervisor_log:
        _log(
            supervisor_log,
            f"supervisor started pid={os.getpid()} output_root={output_root}",
        )
        while True:
            if stop_path.exists():
                _log(supervisor_log, f"stop sentinel detected: {stop_path}; exiting")
                return 0
            if manifest_proves_complete(manifest_path):
                if audit_report_proves_complete(final_audit_path, manifest_path):
                    _log(
                        supervisor_log,
                        "independent full audit proves the final dataset complete; exiting",
                    )
                    return 0
                manifest_sha256 = _file_sha256(manifest_path)
                if manifest_sha256 != last_failed_manifest_sha256:
                    temporary_audit_path = final_audit_path.with_suffix(".json.tmp")
                    _log(
                        supervisor_log,
                        "headline manifest quotas are complete; running independent "
                        "record-by-record audit",
                    )
                    with (
                        temporary_audit_path.open("wb") as audit_output,
                        audit_stderr_path.open("ab", buffering=0) as audit_error,
                    ):
                        audit_result = subprocess.run(
                            audit_command,
                            cwd=PROJECT_ROOT,
                            stdin=subprocess.DEVNULL,
                            stdout=audit_output,
                            stderr=audit_error,
                            check=False,
                        )
                    if (
                        audit_result.returncode == 0
                        and audit_report_proves_complete(
                            temporary_audit_path, manifest_path
                        )
                    ):
                        temporary_audit_path.replace(final_audit_path)
                        failed_audit_path.unlink(missing_ok=True)
                        _log(
                            supervisor_log,
                            "independent full audit passed and final_audit.json "
                            "was published; exiting",
                        )
                        return 0
                    temporary_audit_path.replace(failed_audit_path)
                    last_failed_manifest_sha256 = manifest_sha256
                    _log(
                        supervisor_log,
                        "independent full audit failed "
                        f"code={audit_result.returncode}; evidence saved to "
                        f"{failed_audit_path}",
                    )

            if collector_lock_is_held(lock_path):
                now = time.monotonic()
                if now - last_health_log >= args.health_log_seconds:
                    raw, valid, failed = route_counts(output_root)
                    _log(
                        supervisor_log,
                        "collector lock held; "
                        f"published raw/valid/failed={raw}/{valid}/{failed}",
                    )
                    last_health_log = now
                time.sleep(args.poll_seconds)
                restart_delay = args.poll_seconds
                continue

            _log(
                supervisor_log,
                "collector lock is free and final manifest is incomplete; "
                "starting the exact formal resume command",
            )
            started = time.monotonic()
            with collector_log_path.open("ab", buffering=0) as collector_log:
                child = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=collector_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                _log(supervisor_log, f"collector child started pid={child.pid}")
                return_code = child.wait()
            elapsed = time.monotonic() - started
            _log(
                supervisor_log,
                f"collector child exited code={return_code} after {elapsed:.1f}s",
            )
            if elapsed >= args.health_log_seconds:
                restart_delay = args.poll_seconds
            else:
                restart_delay = min(max(args.poll_seconds, restart_delay * 2.0), 300.0)
            last_failed_manifest_sha256 = None
            _log(supervisor_log, f"next liveness check in {restart_delay:.1f}s")
            time.sleep(restart_delay)


def main(argv: Sequence[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
