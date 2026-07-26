#!/usr/bin/env python3
"""Run the existing X2 generic mesh generator on a deterministic cylinder mesh."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = PROJECT_ROOT / "scripts" / "generate_x2_mesh_grasps.py"
DEFAULT_MESH_PATH = (
    PROJECT_ROOT / "data" / "meshdata" / "test_cylinder" / "cylinder_r30_h80.obj"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "x2_mesh_grasps" / "cylinder_smoke_seed0"

RADIUS = 0.03
HEIGHT = 0.08
SECTIONS = 64
NUM_GRASPS = 8
BATCH_SIZE = 8
N_CONTACT = 4
N_ITERATIONS = 100
OBJECT_SCALE = 1.0
SURFACE_SAMPLES = 256
SEED = 0


class CylinderSmokeError(RuntimeError):
    """Raised when mesh construction, generation, or output validation fails."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("timeout-seconds must be positive")
    return args


def _create_cylinder_mesh(path: Path) -> dict[str, Any]:
    """Create and audit a 30 mm radius, 80 mm high, local-Z cylinder."""

    mesh = trimesh.creation.cylinder(radius=RADIUS, height=HEIGHT, sections=SECTIONS)
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise CylinderSmokeError("trimesh produced a non-watertight cylinder")
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    expected_extents = np.array([2.0 * RADIUS, 2.0 * RADIUS, HEIGHT])
    if not np.allclose(extents, expected_extents, rtol=0.0, atol=1.0e-10):
        raise CylinderSmokeError(
            f"Cylinder extents are {extents.tolist()}, expected {expected_extents.tolist()}"
        )
    if not np.allclose(
        bounds[:, 2], np.array([-HEIGHT / 2.0, HEIGHT / 2.0]), rtol=0.0, atol=1.0e-10
    ):
        raise CylinderSmokeError(f"Cylinder is not centered on its local Z axis: {bounds}")
    radial = np.linalg.vector_norm(np.asarray(mesh.vertices)[:, :2], axis=1)
    if not math.isclose(float(radial.max()), RADIUS, rel_tol=0.0, abs_tol=1.0e-10):
        raise CylinderSmokeError("Cylinder vertex radius does not equal 0.03 m")

    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path, file_type="obj")
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or not loaded.is_watertight:
        raise CylinderSmokeError(f"Exported OBJ is not a watertight mesh: {path}")
    if not np.allclose(loaded.extents, expected_extents, rtol=0.0, atol=1.0e-8):
        raise CylinderSmokeError("Exported OBJ dimensions changed after reload")
    return {
        "path": str(path),
        "radius_m": RADIUS,
        "height_m": HEIGHT,
        "axis": "Z",
        "sections": SECTIONS,
        "vertices": int(len(loaded.vertices)),
        "faces": int(len(loaded.faces)),
        "watertight": bool(loaded.is_watertight),
    }


def _raw_directory(output_root: Path, side: str) -> Path:
    return output_root / f"{side}_single" / "raw"


def _clear_previous_records(output_root: Path, side: str, mesh_stem: str) -> None:
    directory = _raw_directory(output_root, side)
    if not directory.is_dir():
        return
    for path in directory.glob(f"{mesh_stem}_{side}_*.json"):
        path.unlink()


def _run_generator(
    *,
    side: str,
    mesh_path: Path,
    output_root: Path,
    device: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(GENERATOR),
        "--mesh-path",
        str(mesh_path),
        "--side",
        side,
        "--num-grasps",
        str(NUM_GRASPS),
        "--batch-size",
        str(BATCH_SIZE),
        "--n-contact",
        str(N_CONTACT),
        "--n-iterations",
        str(N_ITERATIONS),
        "--object-scale",
        str(OBJECT_SCALE),
        "--surface-samples",
        str(SURFACE_SAMPLES),
        "--seed",
        str(SEED),
        "--device",
        device,
        "--output",
        str(output_root),
        "--overwrite",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise CylinderSmokeError(
            f"{side} generator exceeded {timeout_seconds} seconds"
        ) from exc
    if completed.returncode != 0:
        raise CylinderSmokeError(
            f"{side} generator exited with code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    try:
        summary = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CylinderSmokeError(
            f"{side} generator did not emit a JSON summary:\n{completed.stdout}"
        ) from exc
    if int(summary.get("num_output_samples", -1)) != NUM_GRASPS:
        raise CylinderSmokeError(
            f"{side} summary reported {summary.get('num_output_samples')} samples, "
            f"expected {NUM_GRASPS}"
        )
    if summary.get("side_mode") != side:
        raise CylinderSmokeError(
            f"{side} summary has side_mode={summary.get('side_mode')!r}"
        )
    return summary


def _finite_number(value: Any, label: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CylinderSmokeError(f"{path}: {label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise CylinderSmokeError(f"{path}: {label} is not finite: {result}")
    return result


def _validate_side_records(
    *, side: str, mesh_path: Path, output_root: Path
) -> dict[str, Any]:
    directory = _raw_directory(output_root, side)
    paths = sorted(directory.glob(f"{mesh_path.stem}_{side}_*.json"))
    if len(paths) != NUM_GRASPS:
        raise CylinderSmokeError(
            f"{side} generated {len(paths)} raw JSON files in {directory}; "
            f"expected {NUM_GRASPS}"
        )

    initial_energy: list[float] = []
    final_energy: list[float] = []
    maximum_penetration: list[float] = []
    self_collision_maximum: list[float] = []
    self_collision_total: list[float] = []
    selected_regions: Counter[str] = Counter()
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CylinderSmokeError(f"Cannot read generated record {path}: {exc}") from exc
        if record.get("active_side") != side:
            raise CylinderSmokeError(
                f"{path}: active_side={record.get('active_side')!r}, expected {side!r}"
            )
        if record.get("finite") is not True:
            raise CylinderSmokeError(f"{path}: finite must be true")
        actuator = record.get("actuator")
        joint = record.get("joint")
        contact_ids = record.get("selected_contact_ids")
        if not isinstance(actuator, list) or len(actuator) != 12:
            raise CylinderSmokeError(f"{path}: actuator must contain 12 values")
        if not isinstance(joint, list) or len(joint) != 16:
            raise CylinderSmokeError(f"{path}: joint must contain 16 values")
        if (
            not isinstance(contact_ids, list)
            or len(contact_ids) != N_CONTACT
            or len(set(contact_ids)) != N_CONTACT
        ):
            raise CylinderSmokeError(f"{path}: contact IDs must contain 4 unique values")
        contacts = record.get("selected_contacts")
        if not isinstance(contacts, list) or len(contacts) != N_CONTACT:
            raise CylinderSmokeError(f"{path}: selected_contacts must contain 4 records")
        for contact in contacts:
            supported = contact.get("supported_sides") if isinstance(contact, dict) else None
            if not isinstance(supported, list) or side not in supported:
                raise CylinderSmokeError(f"{path}: selected contact is illegal for {side}")
            selected_regions[str(contact.get("region"))] += 1

        energy = record.get("energy")
        if not isinstance(energy, dict):
            raise CylinderSmokeError(f"{path}: energy must be an object")
        initial = _finite_number(energy.get("initial_total"), "initial_total", path)
        final = _finite_number(energy.get("total"), "final_total", path)
        if not final < initial:
            raise CylinderSmokeError(
                f"{path}: final energy {final:.9g} is not below initial energy {initial:.9g}"
            )
        terms = energy.get("terms")
        if not isinstance(terms, dict) or not terms:
            raise CylinderSmokeError(f"{path}: energy terms are missing")
        for name, value in terms.items():
            _finite_number(value, f"energy.terms.{name}", path)
        penetration = _finite_number(
            record.get("maximum_penetration"), "maximum_penetration", path
        )
        if penetration < 0.0:
            raise CylinderSmokeError(f"{path}: maximum_penetration is negative")
        self_collision = record.get("self_collision")
        if not isinstance(self_collision, dict):
            raise CylinderSmokeError(f"{path}: self_collision must be an object")
        self_maximum = _finite_number(
            self_collision.get("maximum_penetration"),
            "self_collision.maximum_penetration",
            path,
        )
        self_total = _finite_number(
            self_collision.get("total_penetration"),
            "self_collision.total_penetration",
            path,
        )
        self_threshold = _finite_number(
            self_collision.get("threshold"), "self_collision.threshold", path
        )
        if min(self_maximum, self_total, self_threshold) < 0.0:
            raise CylinderSmokeError(
                f"{path}: self-collision distances and threshold must be non-negative"
            )
        expected_feasible = self_maximum <= self_threshold
        if self_collision.get("feasible") is not expected_feasible:
            raise CylinderSmokeError(
                f"{path}: self_collision.feasible disagrees with its threshold"
            )
        if not expected_feasible:
            raise CylinderSmokeError(
                f"{path}: self collision {self_maximum:.9g} exceeds "
                f"threshold {self_threshold:.9g}"
            )
        worst_pair = self_collision.get("worst_pair")
        if worst_pair is not None and (
            not isinstance(worst_pair, list)
            or len(worst_pair) != 2
            or not all(isinstance(name, str) and name for name in worst_pair)
        ):
            raise CylinderSmokeError(
                f"{path}: self_collision.worst_pair must be null or two link names"
            )
        if record.get("success") is not False or record.get("simulation_success") is not False:
            raise CylinderSmokeError(f"{path}: smoke output must remain unvalidated")
        validation = record.get("validation")
        if not isinstance(validation, dict) or validation.get("status") != "not_run":
            raise CylinderSmokeError(f"{path}: validation.status must be not_run")

        initial_energy.append(initial)
        final_energy.append(final)
        maximum_penetration.append(penetration)
        self_collision_maximum.append(self_maximum)
        self_collision_total.append(self_total)

    valid_directory = output_root / f"{side}_single" / "valid"
    valid_paths = (
        list(valid_directory.glob(f"{mesh_path.stem}_{side}_*.json"))
        if valid_directory.is_dir()
        else []
    )
    if valid_paths:
        raise CylinderSmokeError(f"{side} smoke unexpectedly wrote valid records")
    return {
        "passed": True,
        "json_count": len(paths),
        "initial_energy_mean": fmean(initial_energy),
        "final_energy_mean": fmean(final_energy),
        "energy_decreased_count": sum(
            final < initial for initial, final in zip(initial_energy, final_energy)
        ),
        "maximum_penetration_mean": fmean(maximum_penetration),
        "maximum_penetration_min": min(maximum_penetration),
        "self_collision_maximum_penetration": max(self_collision_maximum),
        "self_collision_total_penetration_mean": fmean(self_collision_total),
        "self_collision_feasible_count": len(self_collision_maximum),
        "selected_contact_regions": dict(sorted(selected_regions.items())),
        "output_directory": str(directory),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    mesh_path = DEFAULT_MESH_PATH.resolve()
    output_root = args.output.expanduser().resolve()
    mesh_report = _create_cylinder_mesh(mesh_path)
    side_reports: dict[str, Any] = {}
    for side in ("front", "back"):
        _clear_previous_records(output_root, side, mesh_path.stem)
        generator_summary = _run_generator(
            side=side,
            mesh_path=mesh_path,
            output_root=output_root,
            device=args.device,
            timeout_seconds=args.timeout_seconds,
        )
        side_report = _validate_side_records(
            side=side, mesh_path=mesh_path, output_root=output_root
        )
        side_report["generator_summary"] = generator_summary["sides"][side]
        side_reports[side] = side_report
    return {
        "passed": True,
        "mesh": mesh_report,
        "settings": {
            "num_grasps": NUM_GRASPS,
            "batch_size": BATCH_SIZE,
            "n_contact": N_CONTACT,
            "n_iterations": N_ITERATIONS,
            "object_scale": OBJECT_SCALE,
            "surface_samples": SURFACE_SAMPLES,
            "seed": SEED,
            "device": args.device,
        },
        "sides": side_reports,
        "physical_validation_run": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:
        failure = {
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "physical_validation_run": False,
        }
        print(json.dumps(failure, indent=2, allow_nan=False))
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
