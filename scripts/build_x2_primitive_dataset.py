#!/usr/bin/env python3
"""Deterministically build and audit the X2 primitive mesh dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "meshdata" / "x2_primitives"
SHAPES = ("sphere", "cylinder", "cuboid", "cube")
SPHERE_SUBDIVISIONS = 3
CYLINDER_SECTIONS = 64
AUDIT_ATOL = 1.0e-9


class PrimitiveDatasetError(RuntimeError):
    """Raised when a primitive cannot be built or audited exactly."""


def _millimetre_tag(value: float) -> str:
    millimetres = int(round(float(value) * 1000.0))
    if not np.isclose(value, millimetres / 1000.0, rtol=0.0, atol=1.0e-12):
        raise PrimitiveDatasetError(f"Dimension {value!r} is not an integer millimetre")
    return f"{millimetres:03d}"


@dataclass(frozen=True)
class PrimitiveSpec:
    """One immutable primitive definition, with all dimensions in metres."""

    shape: str
    instance_name: str
    dimensions: tuple[float, ...]
    size: str

    @property
    def relative_path(self) -> Path:
        return Path(self.shape) / f"{self.instance_name}.obj"

    @property
    def expected_extents(self) -> tuple[float, float, float]:
        if self.shape == "sphere":
            diameter = 2.0 * self.dimensions[0]
            return (diameter, diameter, diameter)
        if self.shape == "cylinder":
            radius, height = self.dimensions
            return (2.0 * radius, 2.0 * radius, height)
        if self.shape == "cuboid":
            return (self.dimensions[0], self.dimensions[1], self.dimensions[2])
        if self.shape == "cube":
            edge = self.dimensions[0]
            return (edge, edge, edge)
        raise PrimitiveDatasetError(f"Unsupported primitive shape: {self.shape}")


def _sphere(radius: float) -> PrimitiveSpec:
    tag = _millimetre_tag(radius)
    return PrimitiveSpec("sphere", f"sphere_r{tag}", (radius,), f"radius={radius:.3f}")


def _cylinder(radius: float, height: float) -> PrimitiveSpec:
    r_tag = _millimetre_tag(radius)
    h_tag = _millimetre_tag(height)
    return PrimitiveSpec(
        "cylinder",
        f"cylinder_r{r_tag}_h{h_tag}",
        (radius, height),
        f"radius={radius:.3f};height={height:.3f}",
    )


def _cuboid(x: float, y: float, z: float) -> PrimitiveSpec:
    return PrimitiveSpec(
        "cuboid",
        f"cuboid_x{_millimetre_tag(x)}_y{_millimetre_tag(y)}_z{_millimetre_tag(z)}",
        (x, y, z),
        f"{x:.3f}x{y:.3f}x{z:.3f}",
    )


def _cube(edge: float) -> PrimitiveSpec:
    tag = _millimetre_tag(edge)
    return PrimitiveSpec("cube", f"cube_e{tag}", (edge,), f"edge={edge:.3f}")


PRIMITIVE_SPECS = (
    _sphere(0.020),
    _sphere(0.030),
    _sphere(0.040),
    _cylinder(0.018, 0.100),
    _cylinder(0.025, 0.100),
    _cylinder(0.032, 0.080),
    _cuboid(0.035, 0.055, 0.090),
    _cuboid(0.045, 0.065, 0.110),
    _cuboid(0.055, 0.075, 0.130),
    _cube(0.040),
    _cube(0.050),
    _cube(0.060),
)


def selected_specs(shapes: Sequence[str] | None = None) -> tuple[PrimitiveSpec, ...]:
    """Return catalog entries in stable order, optionally filtered by shape."""

    selected = set(SHAPES if shapes is None else (str(value) for value in shapes))
    unknown = sorted(selected - set(SHAPES))
    if unknown:
        raise PrimitiveDatasetError(f"Unknown primitive shapes: {unknown}")
    return tuple(spec for spec in PRIMITIVE_SPECS if spec.shape in selected)


def create_mesh(spec: PrimitiveSpec) -> trimesh.Trimesh:
    """Construct one centered primitive without any random operation."""

    if spec.shape == "sphere":
        mesh = trimesh.creation.icosphere(
            subdivisions=SPHERE_SUBDIVISIONS, radius=spec.dimensions[0]
        )
    elif spec.shape == "cylinder":
        radius, height = spec.dimensions
        mesh = trimesh.creation.cylinder(
            radius=radius, height=height, sections=CYLINDER_SECTIONS
        )
    elif spec.shape in ("cuboid", "cube"):
        mesh = trimesh.creation.box(extents=spec.expected_extents)
    else:
        raise PrimitiveDatasetError(f"Unsupported primitive shape: {spec.shape}")
    mesh.remove_unreferenced_vertices()
    return mesh


def audit_mesh(mesh: trimesh.Trimesh, spec: PrimitiveSpec, *, path: Path | None = None) -> dict[str, Any]:
    """Strictly audit topology, finiteness, centering, and authored dimensions."""

    label = str(path) if path is not None else spec.instance_name
    if not isinstance(mesh, trimesh.Trimesh):
        raise PrimitiveDatasetError(f"{label}: expected one Trimesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4:
        raise PrimitiveDatasetError(f"{label}: invalid vertex array {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) < 4:
        raise PrimitiveDatasetError(f"{label}: invalid triangle array {faces.shape}")
    if not np.isfinite(vertices).all():
        raise PrimitiveDatasetError(f"{label}: vertices contain NaN or infinity")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise PrimitiveDatasetError(f"{label}: face index is outside the vertex array")
    triangles = vertices[faces]
    doubled_area = np.linalg.vector_norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if not np.isfinite(doubled_area).all() or np.any(doubled_area <= 1.0e-18):
        raise PrimitiveDatasetError(f"{label}: mesh contains a degenerate triangle")
    if not mesh.is_watertight:
        raise PrimitiveDatasetError(f"{label}: mesh is not watertight")
    if not mesh.is_winding_consistent:
        raise PrimitiveDatasetError(f"{label}: winding is inconsistent")
    if not mesh.is_volume or not np.isfinite(float(mesh.volume)) or float(mesh.volume) <= 0.0:
        raise PrimitiveDatasetError(f"{label}: mesh does not enclose a positive finite volume")

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    expected_extents = np.asarray(spec.expected_extents, dtype=np.float64)
    if not np.isfinite(bounds).all():
        raise PrimitiveDatasetError(f"{label}: bounds contain NaN or infinity")
    if not np.allclose(bounds.mean(axis=0), 0.0, rtol=0.0, atol=AUDIT_ATOL):
        raise PrimitiveDatasetError(f"{label}: mesh is not centered at the origin")
    if not np.allclose(extents, expected_extents, rtol=0.0, atol=AUDIT_ATOL):
        raise PrimitiveDatasetError(
            f"{label}: extents {extents.tolist()} != {expected_extents.tolist()}"
        )

    if spec.shape == "sphere":
        radius = spec.dimensions[0]
        vertex_radii = np.linalg.vector_norm(vertices, axis=1)
        if not np.allclose(vertex_radii, radius, rtol=0.0, atol=AUDIT_ATOL):
            raise PrimitiveDatasetError(f"{label}: sphere vertices are not on radius {radius}")
    elif spec.shape == "cylinder":
        radius, height = spec.dimensions
        radial = np.linalg.vector_norm(vertices[:, :2], axis=1)
        if not np.isclose(radial.max(), radius, rtol=0.0, atol=AUDIT_ATOL):
            raise PrimitiveDatasetError(f"{label}: cylinder radius is not {radius}")
        if not np.allclose(
            bounds[:, 2], (-height / 2.0, height / 2.0), rtol=0.0, atol=AUDIT_ATOL
        ):
            raise PrimitiveDatasetError(f"{label}: cylinder axis/height is not local Z/{height}")

    return {
        "shape": spec.shape,
        "instance_name": spec.instance_name,
        "size": spec.size,
        "path": str(path.resolve()) if path is not None else None,
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "watertight": True,
        "winding_consistent": True,
        "volume_m3": float(mesh.volume),
        "extents_m": [float(value) for value in extents],
    }


def _export_obj(mesh: trimesh.Trimesh) -> str:
    exported = trimesh.exchange.obj.export_obj(
        mesh,
        include_normals=False,
        include_color=False,
        include_texture=False,
        digits=12,
        header="X2 deterministic primitive dataset; units=m",
    )
    if not isinstance(exported, str):
        raise PrimitiveDatasetError("trimesh OBJ exporter did not return text")
    return exported.rstrip() + "\n"


def build_dataset(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    shapes: Sequence[str] | None = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Build missing meshes, optionally replace them, then audit OBJ reloads."""

    output_root = Path(output_root).expanduser().resolve()
    reports: list[dict[str, Any]] = []
    for spec in selected_specs(shapes):
        path = output_root / spec.relative_path
        status = "verified"
        if overwrite or not path.exists():
            mesh = create_mesh(spec)
            audit_mesh(mesh, spec)
            payload = _export_obj(mesh)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(path)
            status = "written"
        loaded = trimesh.load(path, force="mesh", process=False)
        report = audit_mesh(loaded, spec, path=path)
        report["status"] = status
        reports.append(report)
    return reports


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    reports = build_dataset(
        args.output_root, shapes=args.shapes, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.expanduser().resolve()),
                "mesh_count": len(reports),
                "all_watertight": all(report["watertight"] for report in reports),
                "instances": reports,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
