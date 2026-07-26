#!/usr/bin/env python3
"""Render a read-only terminal dashboard for the formal X2 collector.

The dashboard never acquires the collector lock, writes collection files, or
changes process state.  It is safe to leave running in a separate terminal and
continues working after Codex or the IDE is closed.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "x2_valid_5000"
DEFAULT_SERVICE = "x2-valid-collector-supervisor.service"
TARGET_VALID = 5000
ROUTES = ("raw", "valid", "failed")
PROCESS_MARKERS = (
    "supervise_x2_valid_collection.py",
    "collect_x2_valid_dataset.py",
    "generate_x2_primitive_dataset.py",
    "generate_x2_mesh_grasps_stratified.py",
    "validate_x2_primitive_dataset.py",
    "validate_x2_mesh_grasps_physx.py",
)


@dataclass(frozen=True)
class AttemptStatus:
    name: str
    raw_target: int
    raw: int
    valid: int
    failed: int
    phase: str
    complete: bool
    validation_objects: int
    seed: int | None
    updated_at: float | None


@dataclass(frozen=True)
class ProcessStatus:
    pid: int
    elapsed_seconds: int
    cpu_percent: float
    rss_kib: int
    role: str
    object_id: str | None
    command: str


@dataclass(frozen=True)
class DashboardSnapshot:
    captured_at: datetime
    service_state: str
    attempts: tuple[AttemptStatus, ...]
    processes: tuple[ProcessStatus, ...]
    gpu_lines: tuple[str, ...]
    manifest_complete: bool
    manifest_valid: int
    final_audit_complete: bool
    completed_valid_pool: int
    pairing: dict[int, int] | None
    f5_counts: dict[str, int] | None
    reusable_groups: int | None
    regenerate_groups: int | None
    recent_events: tuple[str, ...]


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _integer(value: Any, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _scan_incomplete_routes(attempt: Path) -> tuple[dict[str, int], float | None]:
    counts = {route: 0 for route in ROUTES}
    newest: float | None = None
    for directory, children, files in os.walk(attempt):
        current = Path(directory)
        if current.name == ".staging":
            children[:] = []
            continue
        if current.name not in counts:
            continue
        children[:] = []
        json_names = [name for name in files if name.endswith(".json")]
        counts[current.name] += len(json_names)
        if json_names:
            try:
                modified = current.stat().st_mtime
            except OSError:
                continue
            newest = modified if newest is None else max(newest, modified)
    return counts, newest


def scan_attempts(output_root: Path) -> tuple[AttemptStatus, ...]:
    attempts_root = output_root / "attempts"
    if not attempts_root.is_dir():
        return ()
    statuses: list[AttemptStatus] = []
    for attempt in sorted(attempts_root.glob("attempt_[0-9][0-9][0-9][0-9]")):
        metadata = _load_json(attempt / "attempt.json") or {}
        proof = _load_json(attempt / "complete.json")
        complete = bool(proof and proof.get("passed") is True)
        if complete:
            counts = {
                "raw": _integer(proof.get("raw_count")),
                "valid": _integer(proof.get("valid_count")),
                "failed": _integer(proof.get("failed_count")),
            }
            try:
                updated_at = (attempt / "complete.json").stat().st_mtime
            except OSError:
                updated_at = None
        else:
            counts, updated_at = _scan_incomplete_routes(attempt)
        summaries = attempt / "validation_summaries"
        validation_objects = 0
        if summaries.is_dir():
            validation_objects = sum(
                1
                for path in summaries.glob("[!.]*.json")
                if path.is_file() and path.stat().st_size > 0
            )
        raw_target = _integer(metadata.get("raw_target"))
        if complete:
            phase = "complete"
        elif (attempt / "validation_summary.csv").is_file() or validation_objects:
            phase = "validating"
        elif (attempt / "generation_summary.json").is_file() or (
            raw_target > 0 and counts["raw"] >= raw_target
        ):
            phase = "ready-to-validate"
        else:
            phase = "generating"
        seed_value = metadata.get("seed")
        statuses.append(
            AttemptStatus(
                name=attempt.name,
                raw_target=raw_target,
                raw=counts["raw"],
                valid=counts["valid"],
                failed=counts["failed"],
                phase=phase,
                complete=complete,
                validation_objects=validation_objects,
                seed=seed_value if isinstance(seed_value, int) else None,
                updated_at=updated_at,
            )
        )
    return tuple(statuses)


def _run_text(command: Sequence[str], timeout: float = 3.0) -> str:
    try:
        completed = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip()


def _object_id_from_command(command: str) -> str | None:
    match = re.search(r"(?:^|\s)--mesh-path\s+(\S+)", command)
    if not match:
        return None
    mesh = Path(match.group(1))
    if mesh.stem == "decomposed" and mesh.parent.name == "coacd":
        return mesh.parent.parent.name
    return mesh.stem


def _process_role(command: str) -> str:
    if "supervise_x2_valid_collection.py" in command:
        return "supervisor"
    if "collect_x2_valid_dataset.py" in command:
        return "collector"
    if "generate_x2_primitive_dataset.py" in command:
        return "generation-parent"
    if "generate_x2_mesh_grasps_stratified.py" in command:
        return "generator"
    if "validate_x2_primitive_dataset.py" in command:
        return "validation-parent"
    if "validate_x2_mesh_grasps_physx.py" in command:
        return "validator"
    return "related"


def scan_processes() -> tuple[ProcessStatus, ...]:
    output = _run_text(("ps", "-eo", "pid=,etimes=,pcpu=,rss=,args="))
    rows: list[ProcessStatus] = []
    for line in output.splitlines():
        if not any(marker in line for marker in PROCESS_MARKERS):
            continue
        if "watch_x2_collection.py" in line:
            continue
        parts = line.strip().split(None, 4)
        if len(parts) != 5:
            continue
        try:
            pid = int(parts[0])
            elapsed = int(parts[1])
            cpu_percent = float(parts[2])
            rss_kib = int(parts[3])
        except ValueError:
            continue
        command = parts[4]
        rows.append(
            ProcessStatus(
                pid=pid,
                elapsed_seconds=elapsed,
                cpu_percent=cpu_percent,
                rss_kib=rss_kib,
                role=_process_role(command),
                object_id=_object_id_from_command(command),
                command=command,
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.role, row.pid)))


def scan_gpu() -> tuple[str, ...]:
    output = _run_text(
        (
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        )
    )
    lines: list[str] = []
    for row in output.splitlines():
        values = [value.strip() for value in row.split(",")]
        if len(values) != 7:
            continue
        index, name, used, total, utilization, temperature, power = values
        lines.append(
            f"GPU {index} {name}: {used}/{total} MiB | util {utilization}% | "
            f"{temperature} C | {power} W"
        )
    return tuple(lines)


def _tail_lines(path: Path, limit: int = 80) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            return tuple(line.rstrip() for line in deque(stream, maxlen=limit))
    except OSError:
        return ()


def parse_collector_progress(
    lines: Sequence[str],
) -> tuple[dict[int, int] | None, dict[str, int] | None, int | None, int | None]:
    pairing: dict[int, int] | None = None
    f5_counts: dict[str, int] | None = None
    reusable: int | None = None
    regenerate: int | None = None
    for line in reversed(lines):
        if pairing is None and "[collector] pairing=" in line:
            match = re.search(r"pairing=(\{.*?\}), f5=(\{.*?\}),", line)
            if match:
                try:
                    parsed_pairing = ast.literal_eval(match.group(1))
                    parsed_f5 = ast.literal_eval(match.group(2))
                except (SyntaxError, ValueError):
                    pass
                else:
                    if isinstance(parsed_pairing, dict) and isinstance(parsed_f5, dict):
                        pairing = {
                            int(key): int(value)
                            for key, value in parsed_pairing.items()
                            if isinstance(key, int) and isinstance(value, int)
                        }
                        f5_counts = {
                            str(key): int(value)
                            for key, value in parsed_f5.items()
                            if isinstance(key, str) and isinstance(value, int)
                        }
        if reusable is None and line.startswith("[resume] reusable_groups="):
            match = re.search(r"reusable_groups=(\d+) regenerate_groups=(\d+)", line)
            if match:
                reusable, regenerate = int(match.group(1)), int(match.group(2))
        if pairing is not None and reusable is not None:
            break
    return pairing, f5_counts, reusable, regenerate


def collect_snapshot(output_root: Path, service_name: str) -> DashboardSnapshot:
    attempts = scan_attempts(output_root)
    manifest = _load_json(output_root / "manifest.json") or {}
    final_audit = _load_json(output_root / "final_audit.json") or {}
    console_lines = _tail_lines(output_root / "collector_console.log")
    pairing, f5_counts, reusable, regenerate = parse_collector_progress(console_lines)
    interesting = tuple(
        line
        for line in console_lines
        if line.startswith(("[collector]", "[resume]", "[validation]", "[generator]"))
        or "Traceback" in line
        or "OOM" in line
        or "error" in line.casefold()
    )[-5:]
    service_state = _run_text(("systemctl", "--user", "is-active", service_name))
    return DashboardSnapshot(
        captured_at=datetime.now().astimezone(),
        service_state=service_state or "unknown",
        attempts=attempts,
        processes=scan_processes(),
        gpu_lines=scan_gpu(),
        manifest_complete=manifest.get("passed") is True
        and _integer(manifest.get("valid_count")) == TARGET_VALID,
        manifest_valid=_integer(manifest.get("valid_count")),
        final_audit_complete=final_audit.get("passed") is True,
        completed_valid_pool=sum(row.valid for row in attempts if row.complete),
        pairing=pairing,
        f5_counts=f5_counts,
        reusable_groups=reusable,
        regenerate_groups=regenerate,
        recent_events=interesting,
    )


def _duration(seconds: int) -> str:
    days, remainder = divmod(max(0, seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def _age(timestamp: float | None, now: datetime) -> str:
    if timestamp is None:
        return "-"
    return _duration(int(max(0.0, now.timestamp() - timestamp)))


def _bar(value: int, target: int, width: int = 24) -> str:
    if target <= 0:
        return "[" + ("?" * width) + "]"
    fraction = min(1.0, max(0.0, value / target))
    filled = int(round(fraction * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def render_dashboard(snapshot: DashboardSnapshot, *, width: int = 120) -> str:
    del width  # Reserved for future compact layouts.
    lines = [
        "X2 FORMAL COLLECTION DASHBOARD (read-only)",
        snapshot.captured_at.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "=" * 78,
    ]
    if snapshot.manifest_complete:
        lines.append(
            f"FINAL: COMPLETE {snapshot.manifest_valid}/{TARGET_VALID} | "
            f"independent audit={'PASS' if snapshot.final_audit_complete else 'PENDING'}"
        )
    else:
        lines.append(
            f"FINAL: manifest pending | audited candidate pool "
            f"{snapshot.completed_valid_pool}/{TARGET_VALID} "
            f"{_bar(snapshot.completed_valid_pool, TARGET_VALID)}"
        )
    if snapshot.pairing:
        pair_text = " ".join(
            f"f{finger}={snapshot.pairing.get(finger, 0)}/500"
            for finger in range(1, 5)
        )
        f5 = snapshot.f5_counts or {}
        lines.append(
            f"PAIR: {pair_text} | f5 front={f5.get('front', 0)}/500 "
            f"back={f5.get('back', 0)}/500"
        )
    lines.extend(("", "ATTEMPTS"))
    lines.append(
        f"{'name':<14} {'phase':<18} {'raw':>14} {'valid':>8} "
        f"{'failed':>8} {'objects':>9} {'last write':>12}"
    )
    if not snapshot.attempts:
        lines.append("(no attempts found)")
    for row in snapshot.attempts:
        raw_text = f"{row.raw}/{row.raw_target}" if row.raw_target else str(row.raw)
        lines.append(
            f"{row.name:<14} {row.phase:<18} {raw_text:>14} {row.valid:>8} "
            f"{row.failed:>8} {row.validation_objects:>5}/42 "
            f"{_age(row.updated_at, snapshot.captured_at):>12}"
        )
    if snapshot.reusable_groups is not None:
        lines.append(
            f"resume groups: reusable={snapshot.reusable_groups} "
            f"regenerate={snapshot.regenerate_groups}"
        )

    lines.extend(("", "RUNTIME"))
    lines.append(f"systemd service: {snapshot.service_state}")
    if not snapshot.processes:
        lines.append("(no collector-related processes found)")
    else:
        lines.append(
            f"{'pid':>7} {'role':<18} {'elapsed':>8} {'cpu':>7} "
            f"{'rss MiB':>9} object"
        )
        for process in snapshot.processes:
            lines.append(
                f"{process.pid:>7} {process.role:<18} "
                f"{_duration(process.elapsed_seconds):>8} "
                f"{process.cpu_percent:>6.1f}% {process.rss_kib / 1024:>9.0f} "
                f"{process.object_id or '-'}"
            )
    lines.extend(snapshot.gpu_lines or ("GPU: unavailable",))

    lines.extend(("", "RECENT EVENTS"))
    lines.extend(snapshot.recent_events or ("(no recent structured event)",))

    warnings: list[str] = []
    if not snapshot.manifest_complete and snapshot.service_state != "active":
        warnings.append(f"service is {snapshot.service_state}, but final manifest is incomplete")
    roles = {process.role for process in snapshot.processes}
    if not snapshot.manifest_complete and not roles.intersection(
        {"collector", "generation-parent", "generator", "validation-parent", "validator"}
    ):
        warnings.append("no active collector/generator/validator process")
    if warnings:
        lines.extend(("", "WARNINGS"))
        lines.extend(f"! {warning}" for warning in warnings)
    lines.extend(("", "Ctrl+C to exit. This dashboard never changes collection state."))
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--service-name", default=DEFAULT_SERVICE)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear the terminal between refreshes.",
    )
    args = parser.parse_args(argv)
    if args.interval <= 0.0:
        parser.error("--interval must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    if not output_root.is_dir():
        print(f"output root does not exist: {output_root}", file=sys.stderr)
        return 2
    try:
        while True:
            snapshot = collect_snapshot(output_root, args.service_name)
            dashboard = render_dashboard(
                snapshot, width=shutil.get_terminal_size((120, 40)).columns
            )
            if not args.once and not args.no_clear:
                print("\033[2J\033[H", end="")
            print(dashboard, flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nDashboard stopped; collector was not changed.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
