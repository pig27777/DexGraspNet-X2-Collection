#!/usr/bin/env python3
"""Run the X2 Isaac Sim/PhysX validator over the primitive dataset.

Each object instance is a separate Isaac Sim process so its cloned environments
share exactly one converted collision asset.  Processes run sequentially on the
selected device; this wrapper is intended to run after grasp generation has
finished, not concurrently with it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_x2_primitive_dataset import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT as DEFAULT_MESH_ROOT,
    SHAPES,
    PrimitiveSpec,
    build_dataset,
    selected_specs,
)
from scripts.generate_x2_primitive_dataset import (  # noqa: E402
    DEFAULT_GENERAL_MESH_ROOT,
    GeneralMeshSpec,
    discover_general_meshes,
    select_general_meshes,
)
from grasp_generation.x2_isaac_validation import (  # noqa: E402
    CLOSING_LINE_SEARCH_ALPHAS,
    FORMAL_ACTUATOR_ARMATURE,
    FORMAL_ACTUATOR_DAMPING,
    FORMAL_ACTUATOR_STIFFNESS,
    PROTOCOL_REVISION,
    VALIDATION_BACKEND,
    X2RawCandidate,
    discover_raw_candidates,
    validation_output_path,
)


VALIDATOR = PROJECT_ROOT / "scripts" / "validate_x2_mesh_grasps_physx.py"
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "x2_primitive_grasps"
SUMMARY_FIELDS = (
    "shape",
    "size",
    "instance_name",
    "candidate_count",
    "skipped_existing_count",
    "valid_count",
    "failed_count",
    "front_total",
    "front_valid",
    "back_total",
    "back_valid",
    "maximum_fk_position_error_m",
    "minimum_fk_normal_dot",
    "maximum_newton_mimic_error_rad",
)


class PrimitiveValidationError(RuntimeError):
    """Raised when an object-level Isaac validation process fails."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def build_validator_command(
    *,
    spec: PrimitiveSpec | GeneralMeshSpec,
    mesh_root: Path,
    input_root: Path,
    summary_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(VALIDATOR),
        "--input-root",
        str(input_root),
        "--mesh-path",
        str(
            spec.path.resolve()
            if isinstance(spec, GeneralMeshSpec)
            else (mesh_root / spec.relative_path).resolve()
        ),
        "--side",
        args.side,
        "--batch-size",
        str(args.batch_size),
        "--sim-steps",
        str(args.sim_steps),
        "--substeps",
        str(args.substeps),
        "--preclose-physics-steps",
        str(args.preclose_physics_steps),
        "--closing-penetration-cap",
        str(args.closing_penetration_cap),
        "--actuator-stiffness",
        str(getattr(args, "actuator_stiffness", FORMAL_ACTUATOR_STIFFNESS)),
        "--actuator-damping",
        str(getattr(args, "actuator_damping", FORMAL_ACTUATOR_DAMPING)),
        "--actuator-armature",
        str(getattr(args, "actuator_armature", FORMAL_ACTUATOR_ARMATURE)),
        "--external-forces-every-iteration",
        "--criterion",
        args.criterion,
        "--collision-approximation",
        "convex-decomposition"
        if isinstance(spec, GeneralMeshSpec)
        else "convex-hull",
        "--device",
        args.device,
        "--summary-json",
        str(summary_path),
        "--viz",
        "none",
    ]
    if args.limit_per_object is not None:
        command.extend(("--limit", str(args.limit_per_object)))
    if args.overwrite:
        command.append("--overwrite")
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        command.append("--dry-run")
    return command


def _summary_row(
    spec: PrimitiveSpec | GeneralMeshSpec, report: dict[str, Any]
) -> dict[str, Any]:
    sides = report.get("side_summary", {})
    fk = report.get("fk_preflight", {})
    if report.get("dry_run"):
        side_counts = report.get("side_counts", {})
        sides = {
            side: {"total": int(side_counts.get(side, 0)), "passed": 0}
            for side in ("front", "back")
        }
    return {
        "shape": spec.shape,
        "size": spec.size,
        "instance_name": spec.instance_name,
        "candidate_count": int(report.get("candidate_count", 0)),
        "skipped_existing_count": int(report.get("skipped_existing_count", 0)),
        "valid_count": int(report.get("valid_count", 0)),
        "failed_count": int(report.get("failed_count", 0)),
        "front_total": int(sides.get("front", {}).get("total", 0)),
        "front_valid": int(sides.get("front", {}).get("passed", 0)),
        "back_total": int(sides.get("back", {}).get("total", 0)),
        "back_valid": int(sides.get("back", {}).get("passed", 0)),
        "maximum_fk_position_error_m": fk.get("maximum_selected_contact_position_error_m", ""),
        "minimum_fk_normal_dot": fk.get("minimum_selected_contact_normal_dot", ""),
        "maximum_newton_mimic_error_rad": report.get(
            "maximum_newton_mimic_error_rad", ""
        ),
    }


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise PrimitiveValidationError(f"{label} contains non-finite JSON constant {value}")

    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream, parse_constant=reject_constant)
    except PrimitiveValidationError:
        raise
    except Exception as exc:
        raise PrimitiveValidationError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PrimitiveValidationError(f"{label} must contain a JSON object: {path}")
    return payload


def _report_count(report: dict[str, Any], key: str, *, default: int | None = None) -> int:
    value = report.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PrimitiveValidationError(f"validator summary {key} must be a non-negative integer")
    return value


def _validate_child_report(
    report: dict[str, Any],
    *,
    expected_mesh: Path,
    expected_scale: float,
    expected_count: int,
    resume: bool,
    dry_run: bool,
) -> tuple[int, int]:
    """Reject a missing, stale, or internally inconsistent child summary."""

    if report.get("passed") is not True:
        raise PrimitiveValidationError("validator summary did not pass")
    if bool(report.get("dry_run", False)) != dry_run:
        raise PrimitiveValidationError("validator summary dry_run does not match the invocation")

    mesh_value = report.get("mesh_path")
    if not isinstance(mesh_value, str) or Path(mesh_value).expanduser().resolve() != expected_mesh:
        raise PrimitiveValidationError(
            f"validator summary mesh mismatch: expected={expected_mesh}, actual={mesh_value!r}"
        )
    scale_value = report.get("object_scale")
    if isinstance(scale_value, bool):
        raise PrimitiveValidationError("validator summary object_scale must be finite and positive")
    try:
        actual_scale = float(scale_value)
    except (TypeError, ValueError) as exc:
        raise PrimitiveValidationError(
            "validator summary object_scale must be finite and positive"
        ) from exc
    if not math.isfinite(actual_scale) or actual_scale <= 0.0 or actual_scale != expected_scale:
        raise PrimitiveValidationError(
            f"validator summary scale mismatch: expected={expected_scale}, actual={actual_scale}"
        )

    processed_count = _report_count(report, "candidate_count")
    skipped_count = _report_count(report, "skipped_existing_count", default=0)
    if resume:
        reported_total = processed_count + skipped_count
    else:
        if skipped_count != 0:
            raise PrimitiveValidationError(
                "validator summary skipped candidates without a --resume invocation"
            )
        reported_total = processed_count
    if reported_total != expected_count:
        raise PrimitiveValidationError(
            "validator summary candidate count mismatch: "
            f"expected={expected_count}, processed={processed_count}, skipped={skipped_count}"
        )

    if not dry_run:
        valid_count = _report_count(report, "valid_count", default=0)
        failed_count = _report_count(report, "failed_count", default=0)
        if valid_count + failed_count != processed_count:
            raise PrimitiveValidationError(
                "validator summary routed count mismatch: "
                f"processed={processed_count}, valid={valid_count}, failed={failed_count}"
            )
    return processed_count, skipped_count


def _validated_output(
    candidate: X2RawCandidate,
    *,
    path: Path,
    expected_status: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = _strict_json(path, "validated output")
    validation = payload.get("validation")
    if not isinstance(validation, dict) or validation.get("status") != expected_status:
        raise PrimitiveValidationError(
            f"validated output status/path mismatch for {path}: expected {expected_status}"
        )
    expected_success = expected_status == "passed"
    if payload.get("success") is not expected_success:
        raise PrimitiveValidationError(f"validated output success/status mismatch: {path}")
    if payload.get("active_side") != candidate.active_side:
        raise PrimitiveValidationError(f"validated output active_side mismatch: {path}")

    object_record = payload.get("object")
    if not isinstance(object_record, dict):
        raise PrimitiveValidationError(f"validated output has no object record: {path}")
    mesh_value = object_record.get("mesh_path")
    if not isinstance(mesh_value, str) or Path(mesh_value).expanduser().resolve() != candidate.mesh_path:
        raise PrimitiveValidationError(f"validated output mesh mismatch: {path}")
    try:
        scale = float(object_record.get("scale"))
    except (TypeError, ValueError) as exc:
        raise PrimitiveValidationError(f"validated output scale is invalid: {path}") from exc
    if not math.isfinite(scale) or scale != candidate.object_scale:
        raise PrimitiveValidationError(f"validated output scale mismatch: {path}")

    source_value = validation.get("source_raw")
    if not isinstance(source_value, str) or Path(source_value).expanduser().resolve() != candidate.path:
        raise PrimitiveValidationError(f"validated output source_raw mismatch: {path}")
    source_sha256 = hashlib.sha256(candidate.path.read_bytes()).hexdigest()
    if validation.get("source_sha256") != source_sha256:
        raise PrimitiveValidationError(f"validated output source_sha256 is stale: {path}")

    if (
        validation.get("backend") != VALIDATION_BACKEND
        or validation.get("protocol_revision") != PROTOCOL_REVISION
        or validation.get("criterion") != args.criterion
    ):
        raise PrimitiveValidationError(
            f"validated output protocol/backend/criterion is stale: {path}"
        )
    runtime = validation.get("runtime")
    expected_physics_steps = args.sim_steps * args.substeps
    if (
        not isinstance(runtime, dict)
        or runtime.get("device") != args.device
        or runtime.get("simulation_steps") != args.sim_steps
        or runtime.get("substeps") != args.substeps
        or runtime.get("physics_step_count") != expected_physics_steps
    ):
        raise PrimitiveValidationError(
            f"validated output runtime step/device contract is stale: {path}"
        )
    actuator_drive = runtime.get("actuator_drive")
    if (
        not isinstance(actuator_drive, dict)
        or actuator_drive.get("stiffness_n_m_per_rad")
        != getattr(args, "actuator_stiffness", FORMAL_ACTUATOR_STIFFNESS)
        or actuator_drive.get("damping_n_m_s_per_rad")
        != getattr(args, "actuator_damping", FORMAL_ACTUATOR_DAMPING)
        or actuator_drive.get("armature_kg_m2")
        != getattr(args, "actuator_armature", FORMAL_ACTUATOR_ARMATURE)
        or actuator_drive.get("stiffness_source") != "cli_override"
        or actuator_drive.get("damping_source") != "cli_override"
        or actuator_drive.get("armature_source") != "cli_override"
    ):
        raise PrimitiveValidationError(
            f"validated output actuator-drive contract is stale: {path}"
        )
    physx_solver = runtime.get("physx_solver")
    if (
        not isinstance(physx_solver, dict)
        or physx_solver.get("solver_type") != 1
        or physx_solver.get("external_forces_every_iteration") is not True
        or physx_solver.get("solve_articulation_contact_last") is not False
    ):
        raise PrimitiveValidationError(
            f"validated output PhysX solver contract is stale: {path}"
        )
    preclose = runtime.get("zero_gravity_preclose")
    if (
        not isinstance(preclose, dict)
        or preclose.get("enabled") is not (args.preclose_physics_steps > 0)
        or preclose.get("physics_step_count") != args.preclose_physics_steps
        or preclose.get("validation_physics_step_count_unchanged")
        != expected_physics_steps
    ):
        raise PrimitiveValidationError(
            f"validated output preclose contract is stale: {path}"
        )
    closing = runtime.get("contact_gradient_closing")
    expected_alphas = [float(value) for value in CLOSING_LINE_SEARCH_ALPHAS[:-1]]
    if (
        not isinstance(closing, dict)
        or closing.get("enabled") is not True
        or closing.get("mode") != "collision_aware_line_search"
        or closing.get("raw_penetration_cap_m") != 0.001
        or closing.get("target_penetration_cap_m")
        != args.closing_penetration_cap
        or closing.get("positive_alphas") != expected_alphas
        or closing.get("bidirectional_sampled_penetration") is not True
        or closing.get("float32_quantized_and_rechecked") is not True
        or closing.get("raw_json_remains_physx_initial_state") is not True
    ):
        raise PrimitiveValidationError(
            f"validated output collision-aware closing contract is stale: {path}"
        )
    preflight = validation.get("preflight")
    sample_closing = (
        preflight.get("collision_aware_closing")
        if isinstance(preflight, dict)
        else None
    )
    if (
        not isinstance(sample_closing, dict)
        or sample_closing.get("enabled") is not True
        or sample_closing.get("raw_penetration_cap_m") != 0.001
        or sample_closing.get("target_penetration_cap_m")
        != args.closing_penetration_cap
        or sample_closing.get("positive_alphas") != expected_alphas
        or sample_closing.get("bidirectional_sampled_penetration") is not True
        or sample_closing.get("float32_quantized_and_rechecked") is not True
    ):
        raise PrimitiveValidationError(
            f"validated output per-sample closing proof is stale: {path}"
        )
    return payload


def _scan_validation_outputs(
    candidates: Sequence[X2RawCandidate], args: argparse.Namespace
) -> dict[str, Any]:
    """Audit sibling outputs and return complete counts for the selected raw set."""

    side_summary = {
        "front": {"total": 0, "passed": 0, "failed": 0},
        "back": {"total": 0, "passed": 0, "failed": 0},
    }
    valid_count = 0
    failed_count = 0
    pending: list[str] = []
    output_files: list[str] = []
    for candidate in candidates:
        side_summary[candidate.active_side]["total"] += 1
        valid_path = validation_output_path(candidate.path, True)
        failed_path = validation_output_path(candidate.path, False)
        if valid_path.exists() and failed_path.exists():
            raise PrimitiveValidationError(
                f"raw candidate has both valid and failed outputs: {candidate.path}"
            )
        if valid_path.exists():
            _validated_output(
                candidate, path=valid_path, expected_status="passed", args=args
            )
            valid_count += 1
            side_summary[candidate.active_side]["passed"] += 1
            output_files.append(str(valid_path))
        elif failed_path.exists():
            _validated_output(
                candidate, path=failed_path, expected_status="failed", args=args
            )
            failed_count += 1
            side_summary[candidate.active_side]["failed"] += 1
            output_files.append(str(failed_path))
        else:
            pending.append(str(candidate.path))
    side_summary = {
        side: counts for side, counts in side_summary.items() if counts["total"] > 0
    }
    return {
        "candidate_count": len(candidates),
        "valid_count": valid_count,
        "failed_count": failed_count,
        "pending_count": len(pending),
        "pending_raw": pending,
        "side_summary": side_summary,
        "output_files": output_files,
    }


def _complete_report(
    report: dict[str, Any],
    *,
    candidates: Sequence[X2RawCandidate],
    processed_count: int,
    dry_run: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Convert a child-run report into an unambiguous full-selection report."""

    completed = dict(report)
    completed["processed_candidate_count"] = processed_count
    completed["candidate_count"] = len(candidates)
    if dry_run:
        side_counts = {
            side: sum(candidate.active_side == side for candidate in candidates)
            for side in ("front", "back")
        }
        completed["side_counts"] = side_counts
        completed["would_validate_count"] = processed_count
        completed["would_create_valid_or_failed"] = processed_count
        return completed

    audit = _scan_validation_outputs(candidates, args)
    if audit["pending_count"] != 0:
        preview = audit["pending_raw"][:3]
        raise PrimitiveValidationError(
            f"validator left {audit['pending_count']} selected raw candidates unrouted: {preview}"
        )
    completed.update(audit)
    return completed


def _remove_stale_resume_outputs(
    candidates: Sequence[X2RawCandidate], args: argparse.Namespace
) -> int:
    """Remove only routed records which cannot prove the current v6 contract."""

    if not args.resume or args.dry_run:
        return 0
    removed = 0
    for candidate in candidates:
        valid_path = validation_output_path(candidate.path, True)
        failed_path = validation_output_path(candidate.path, False)
        if valid_path.exists() and failed_path.exists():
            raise PrimitiveValidationError(
                f"raw candidate has both valid and failed outputs: {candidate.path}"
            )
        for path, status in ((valid_path, "passed"), (failed_path, "failed")):
            if not path.exists():
                continue
            try:
                _validated_output(
                    candidate, path=path, expected_status=status, args=args
                )
            except PrimitiveValidationError as exc:
                path.unlink()
                removed += 1
                print(f"[resume] removed stale route {path}: {exc}", flush=True)
    return removed


def _new_temporary_summary(summary_dir: Path, instance_name: str) -> Path:
    descriptor, value = tempfile.mkstemp(
        dir=summary_dir,
        prefix=f".{instance_name}.",
        suffix=".summary.json",
    )
    os.close(descriptor)
    return Path(value)


def _publish_summary(path: Path, target: Path, report: dict[str, Any]) -> None:
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.replace(target)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument(
        "--include-general-meshes",
        action="store_true",
        help="also validate <general-mesh-root>/*/coacd/decomposed.obj objects",
    )
    parser.add_argument(
        "--general-mesh-root", type=Path, default=DEFAULT_GENERAL_MESH_ROOT
    )
    parser.add_argument(
        "--general-mesh-ids",
        nargs="+",
        help="explicit general-mesh object IDs to include (default: all discovered)",
    )
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--side", choices=("front", "back", "both"), default="both")
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--sim-steps", type=_positive_int, default=100)
    parser.add_argument("--substeps", type=_positive_int, default=2)
    parser.add_argument(
        "--preclose-physics-steps", type=_nonnegative_int, default=0
    )
    parser.add_argument(
        "--closing-penetration-cap", type=_positive_float, default=0.0015
    )
    parser.add_argument(
        "--actuator-stiffness",
        type=_positive_float,
        default=FORMAL_ACTUATOR_STIFFNESS,
    )
    parser.add_argument(
        "--actuator-damping",
        type=_positive_float,
        default=FORMAL_ACTUATOR_DAMPING,
    )
    parser.add_argument(
        "--actuator-armature",
        type=_positive_float,
        default=FORMAL_ACTUATOR_ARMATURE,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--criterion",
        choices=("dexgraspnet-contact", "strict-hold"),
        default="dexgraspnet-contact",
    )
    parser.add_argument("--limit-per-object", type=_positive_int)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Default: <input-root>/validation_summary.csv.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        help="Default: <input-root>/validation_summaries.",
    )
    args = parser.parse_args(argv)
    if args.general_mesh_ids is not None and not args.include_general_meshes:
        parser.error("general-mesh-ids requires --include-general-meshes")
    if args.general_mesh_ids is not None and len(args.general_mesh_ids) != len(
        set(args.general_mesh_ids)
    ):
        parser.error("general-mesh-ids must be unique")
    return args


def _resolve_summary_paths(
    input_root: Path, args: argparse.Namespace
) -> tuple[Path, Path]:
    summary_dir_value = args.summary_dir or input_root / "validation_summaries"
    summary_csv_value = args.summary_csv or input_root / "validation_summary.csv"
    return (
        summary_dir_value.expanduser().resolve(),
        summary_csv_value.expanduser().resolve(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    input_root = args.input_root.expanduser().resolve()
    mesh_root = args.mesh_root.expanduser().resolve()
    summary_dir, summary_csv = _resolve_summary_paths(input_root, args)
    general_specs = (
        select_general_meshes(
            discover_general_meshes(args.general_mesh_root),
            getattr(args, "general_mesh_ids", None),
        )
        if args.include_general_meshes
        else ()
    )
    if args.include_general_meshes and not general_specs:
        raise PrimitiveValidationError(
            f"No general meshes found below {args.general_mesh_root}/<object>/coacd/decomposed.obj"
        )
    specs = (*selected_specs(args.shapes), *general_specs)
    build_dataset(mesh_root, shapes=args.shapes, overwrite=False)
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        summary_path = summary_dir / f"{spec.instance_name}.json"
        expected_mesh = (
            spec.path.resolve()
            if isinstance(spec, GeneralMeshSpec)
            else (mesh_root / spec.relative_path).resolve()
        )
        candidates = discover_raw_candidates(
            input_root,
            mesh_path=expected_mesh,
            side=args.side,
            limit=args.limit_per_object,
        )
        scales = {candidate.object_scale for candidate in candidates}
        if len(scales) != 1:
            raise PrimitiveValidationError(
                f"{spec.instance_name}: selected raw candidates contain multiple scales: {sorted(scales)}"
            )
        expected_scale = next(iter(scales))
        _remove_stale_resume_outputs(candidates, args)
        temporary_summary = _new_temporary_summary(summary_dir, spec.instance_name)
        try:
            command = build_validator_command(
                spec=spec,
                mesh_root=mesh_root,
                input_root=input_root,
                summary_path=temporary_summary,
                args=args,
            )
            print(f"[{index}/{len(specs)}] validating {spec.instance_name}", flush=True)
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if completed.returncode != 0:
                tail = "\n".join(completed.stdout.splitlines()[-120:])
                raise PrimitiveValidationError(
                    f"{spec.instance_name}: validator exited {completed.returncode}\n{tail}"
                )
            report = _strict_json(temporary_summary, "temporary validator summary")
            processed_count, _ = _validate_child_report(
                report,
                expected_mesh=expected_mesh,
                expected_scale=expected_scale,
                expected_count=len(candidates),
                resume=args.resume,
                dry_run=args.dry_run,
            )
            report = _complete_report(
                report,
                candidates=candidates,
                processed_count=processed_count,
                dry_run=args.dry_run,
                args=args,
            )
            _publish_summary(temporary_summary, summary_path, report)
        finally:
            temporary_summary.unlink(missing_ok=True)
        reports.append(report)
        rows.append(_summary_row(spec, report))
        print(
            f"[{index}/{len(specs)}] {spec.instance_name}: "
            f"valid={report.get('valid_count', 0)} failed={report.get('failed_count', 0)} "
            f"skipped={report.get('skipped_existing_count', 0)}",
            flush=True,
        )

    _write_csv(summary_csv, rows)
    result = {
        "passed": True,
        "dry_run": bool(args.dry_run),
        "input_root": str(input_root),
        "objects": len(specs),
        "candidate_count": sum(int(row["candidate_count"]) for row in rows),
        "skipped_existing_count": sum(int(row["skipped_existing_count"]) for row in rows),
        "valid_count": sum(int(row["valid_count"]) for row in rows),
        "failed_count": sum(int(row["failed_count"]) for row in rows),
        "summary_csv": str(summary_csv),
        "reports": reports,
    }
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
