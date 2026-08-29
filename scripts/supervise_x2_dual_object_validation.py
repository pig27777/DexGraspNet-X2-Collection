#!/usr/bin/env python3
"""Wait for formal collection, rebuild dual compositions, then run PhysX.

This lightweight supervisor is safe to run as a persistent user service.  It
does not touch an in-progress collection.  Once the formal 5000-valid manifest
exists, it deterministically rebuilds ``data/x2_dual_object`` from completed
attempts, requires at least 500 candidates in each of the four cross-object
strata, and launches the resume-capable joint dual-object PhysX validator for
every candidate rather than truncating each stratum at its formal minimum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSITION_PROTOCOL = "x2_right_left_dual_object_warm_start_v1"
VALIDATION_PROTOCOL = "x2_dual_object_six_orientation_physx_v1"


class DualValidationSupervisorError(RuntimeError):
    """Raised when a completed artifact is malformed or validation fails."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DualValidationSupervisorError(
            f"Cannot read completed artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise DualValidationSupervisorError(f"{path}: JSON root is not an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def formal_collection_ready(formal_root: Path) -> bool:
    manifest_path = formal_root / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _json(manifest_path)
    if manifest.get("passed") is not True or manifest.get("valid_count") != 5000:
        raise DualValidationSupervisorError(
            "formal manifest exists but does not prove 5000 valid records"
        )
    return True


def dual_composition_ready(dataset_root: Path) -> bool:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _json(manifest_path)
    if (
        manifest.get("protocol_revision") != COMPOSITION_PROTOCOL
        or manifest.get("dual_object_status") != "not_validated"
        or manifest.get("formal_source_completion_required") is not True
    ):
        return False
    records = manifest.get("dual_object_candidates")
    if not isinstance(records, list) or len(records) < 2000:
        return False
    counts: dict[tuple[int, int], int] = {}
    for value in records:
        if not isinstance(value, dict):
            raise DualValidationSupervisorError(
                "dual composition descriptor is malformed"
            )
        key = (
            value.get("right_finger_count"),
            value.get("left_finger_count"),
        )
        counts[key] = counts.get(key, 0) + 1
    expected = {(1, 4), (2, 3), (3, 2), (4, 1)}
    return set(counts) == expected and all(
        counts[key] >= 500 for key in expected
    )


def validation_complete(dataset_root: Path) -> bool:
    composition_path = dataset_root / "manifest.json"
    summary_path = dataset_root / "physx_validation" / "summary.json"
    if not composition_path.is_file() or not summary_path.is_file():
        return False
    composition = _json(composition_path)
    records = composition.get("dual_object_candidates")
    if not isinstance(records, list) or not records:
        raise DualValidationSupervisorError(
            "dual composition manifest has no candidate inventory"
        )
    candidate_count = len(records)
    summary = _json(summary_path)
    if (
        summary.get("passed") is not True
        or summary.get("protocol_revision") != VALIDATION_PROTOCOL
        or summary.get("candidate_count") != candidate_count
        or summary.get("valid_count", 0) + summary.get("failed_count", 0)
        != candidate_count
        or summary.get("source_manifest_sha256") != _sha256(composition_path)
    ):
        raise DualValidationSupervisorError(
            "dual PhysX summary exists but its completion proof is stale"
        )
    return True


def _commands(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    conda = str(args.conda_executable)
    common = [conda, "run", "-n", args.environment, "--no-capture-output", "python"]
    build = [
        *common,
        str(args.repo_root / "scripts" / "build_x2_dual_object_candidates.py"),
        "--overwrite",
        "--output-root",
        str(args.dataset_root),
    ]
    validate = [
        *common,
        str(args.repo_root / "scripts" / "validate_x2_dual_object_physx.py"),
        "--dataset-root",
        str(args.dataset_root),
        "--output-root",
        str(args.dataset_root / "physx_validation"),
        "--batch-size",
        str(args.batch_size),
        "--sim-steps",
        "100",
        "--substeps",
        "2",
        "--resume",
        "--headless",
        "--device",
        args.device,
        "--summary-json",
        str(args.dataset_root / "physx_validation" / "summary.json"),
    ]
    return build, validate


def _completed_attempt_count(formal_root: Path) -> int:
    attempts = formal_root / "attempts"
    return sum(
        1
        for path in attempts.glob("attempt_*/complete.json")
        if path.is_file()
    )


def run(args: argparse.Namespace) -> int:
    completed_attempts = _completed_attempt_count(args.formal_root)
    while not formal_collection_ready(args.formal_root):
        time.sleep(args.poll_seconds)
        current_completed = _completed_attempt_count(args.formal_root)
        if current_completed > completed_attempts:
            build, _ = _commands(args)
            subprocess.run(build, cwd=args.repo_root, check=True)
            completed_attempts = current_completed
    if validation_complete(args.dataset_root):
        return 0
    if not dual_composition_ready(args.dataset_root):
        build, _ = _commands(args)
        subprocess.run(build, cwd=args.repo_root, check=True)
    if not dual_composition_ready(args.dataset_root):
        raise DualValidationSupervisorError(
            "formal collection is complete but cross-object composition did not "
            "produce at least 500 candidates in each of the four strata"
        )
    _, validate = _commands(args)
    subprocess.run(validate, cwd=args.repo_root, check=True)
    if not validation_complete(args.dataset_root):
        raise DualValidationSupervisorError(
            "dual PhysX validator exited without a complete bound summary"
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "x2_valid_5000",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "x2_dual_object",
    )
    parser.add_argument(
        "--conda-executable",
        type=Path,
        default=Path("/home/lhr/miniconda3/bin/conda"),
    )
    parser.add_argument("--environment", default="isaaclab")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.repo_root = args.repo_root.expanduser().resolve()
    args.formal_root = args.formal_root.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.conda_executable = args.conda_executable.expanduser().resolve()
    if args.batch_size <= 0 or args.poll_seconds <= 0.0:
        raise SystemExit("batch-size and poll-seconds must be positive")
    try:
        return run(args)
    except (DualValidationSupervisorError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
