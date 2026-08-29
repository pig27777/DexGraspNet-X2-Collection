#!/usr/bin/env python3
"""Render one table-conditioned X2 static grasp candidate.

The image reconstructs the stored X2 collision meshes, object mesh, selected
contacts, and the support plane recorded by the generator.  It is a static
diagnostic only and does not imply tabletop or PhysX validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_generation.utils.x2_config import load_x2_mesh_config  # noqa: E402
from grasp_generation.utils.x2_hand_model import X2HandModel  # noqa: E402
from grasp_generation.utils.x2_mesh_contacts import (  # noqa: E402
    load_generic_contact_candidates,
)
from scripts.render_x2_cylinder_smoke import (  # noqa: E402
    X2CylinderRenderError,
    _materialize_geometry,
)


class X2TableRenderError(RuntimeError):
    """Raised when a static record lacks a usable table contract."""


FINGER_COLORS = {
    "rh_th": (0.95, 0.50, 0.10, 0.88),
    "rh_ff": (0.88, 0.22, 0.24, 0.88),
    "rh_mf": (0.36, 0.64, 0.34, 0.88),
    "rh_rf": (0.66, 0.42, 0.72, 0.88),
    "rh_lf": (0.20, 0.64, 0.72, 0.88),
}


def _link_color(link_name: str) -> tuple[float, float, float, float]:
    for prefix, color in FINGER_COLORS.items():
        if link_name.startswith(prefix):
            return color
    if link_name == "rh_palm":
        return (0.52, 0.56, 0.64, 0.72)
    return (0.68, 0.71, 0.76, 0.60)


def _object_id(mesh_path: Path) -> str:
    if mesh_path.stem == "decomposed" and mesh_path.parent.name == "coacd":
        return mesh_path.parent.parent.name
    return mesh_path.stem


def _load_record(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise X2TableRenderError(f"Cannot read {path}: {exc}") from exc
    table = record.get("table_conditioning")
    if not isinstance(table, dict):
        raise X2TableRenderError("record has no table_conditioning block")
    if table.get("source_plane_nonpenetrating") is not True:
        raise X2TableRenderError("record crosses the source support plane")
    if record.get("finite") is not True:
        raise X2TableRenderError("record is not finite")
    return record


def _plane_quad(
    table: dict[str, Any], center: np.ndarray, half_extent: float
) -> np.ndarray:
    normal = np.asarray(table["plane_normal"], dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise X2TableRenderError("table plane normal must be finite 3D")
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise X2TableRenderError("table plane normal must be non-zero")
    normal /= norm
    offset = float(table["plane_offset_m"])
    plane_center = center + (offset - float(normal @ center)) * normal
    reference = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    if abs(float(reference @ normal)) > 0.9:
        reference = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    tangent = np.cross(normal, reference)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    return np.asarray(
        [
            plane_center - half_extent * tangent - half_extent * bitangent,
            plane_center + half_extent * tangent - half_extent * bitangent,
            plane_center + half_extent * tangent + half_extent * bitangent,
            plane_center - half_extent * tangent + half_extent * bitangent,
        ]
    )


def _draw_scene(
    axis: Any,
    geometry: dict[str, Any],
    *,
    elevation: float,
    azimuth: float,
) -> None:
    record = geometry["record"]
    object_vertices = geometry["object_vertices"]
    object_faces = geometry["object_faces"]
    all_points = [object_vertices]
    for _, vertices, _ in geometry["links"]:
        all_points.append(vertices)
    points = np.concatenate(all_points, axis=0)
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.58 * float((upper - lower).max()), 0.09)

    table_quad = _plane_quad(record["table_conditioning"], center, 1.35 * radius)
    axis.add_collection3d(
        Poly3DCollection(
            [table_quad],
            facecolor=(0.78, 0.67, 0.50, 0.58),
            edgecolor=(0.37, 0.28, 0.18, 0.72),
            linewidth=0.8,
        )
    )
    axis.add_collection3d(
        Poly3DCollection(
            object_vertices[object_faces],
            facecolor=(0.10, 0.45, 0.95, 0.58),
            edgecolor=(0.03, 0.18, 0.42, 0.30),
            linewidth=0.18,
        )
    )
    for link_name, vertices, faces in geometry["links"]:
        axis.add_collection3d(
            Poly3DCollection(
                vertices[faces],
                facecolor=_link_color(link_name),
                edgecolor=(0.06, 0.06, 0.08, 0.20),
                linewidth=0.12,
            )
        )
    contacts = geometry["contacts"]
    axis.scatter(
        contacts[:, 0],
        contacts[:, 1],
        contacts[:, 2],
        color="#ffe45e",
        edgecolors="#3c3100",
        linewidths=0.75,
        s=34,
        depthshade=False,
    )
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(
        min(float(table_quad[:, 2].min()), center[2] - radius),
        center[2] + radius,
    )
    axis.set_box_aspect((1.0, 1.0, 0.9))
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()


def render(raw_json: Path, output: Path) -> dict[str, Any]:
    record = _load_record(raw_json)
    config = load_x2_mesh_config()
    candidates = load_generic_contact_candidates(
        config.configured_path("contact_candidates.path", must_exist=True)
    )
    hand = X2HandModel(
        config,
        candidates,
        device="cpu",
        dtype=torch.float64,
        collision_samples_per_link=1,
        self_collision_samples_per_link=1,
    )
    try:
        geometry = _materialize_geometry(hand, raw_json, record)
    except X2CylinderRenderError as exc:
        raise X2TableRenderError(str(exc)) from exc

    side = str(record["active_side"])
    main_azimuth = -64.0 if side == "front" else 116.0
    views = (
        (24.0, main_azimuth, "Main view"),
        (22.0, main_azimuth + 90.0, "Side view"),
        (70.0, main_azimuth, "Top view"),
    )
    figure = plt.figure(figsize=(15, 5.4), constrained_layout=True)
    for index, (elevation, azimuth, title) in enumerate(views, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        _draw_scene(axis, geometry, elevation=elevation, azimuth=azimuth)
        axis.set_title(title, fontsize=10)

    table = record["table_conditioning"]
    contact = record.get("selected_contact_realization", {})
    tabletop_validation = record.get("tabletop_validation")
    source_validation = record.get("validation")
    source_physx_passed = (
        isinstance(source_validation, dict)
        and source_validation.get("status") == "passed"
        and source_validation.get("backend") == "isaac_sim_physx"
    )
    tabletop_physx_not_run = (
        isinstance(tabletop_validation, dict)
        and tabletop_validation.get("status") == "NOT_RUN"
    )
    if source_physx_passed and tabletop_physx_not_run:
        physical_status = "SOURCE_PHYSX_PASS_TABLETOP_PHYSX_NOT_RUN"
        physical_caption = "SOURCE PHYSX PASS — TABLETOP PHYSX NOT RUN"
    elif source_physx_passed:
        physical_status = "SOURCE_SIX_ORIENTATION_PHYSX_PASS"
        physical_caption = "6-ORIENTATION PHYSX PASS — STATIC TABLE RENDER"
    else:
        physical_status = "STATIC_ONLY_PHYSX_NOT_RUN"
        physical_caption = "STATIC ONLY — PHYSX NOT RUN"
    contact_distance = contact.get("maximum_distance_m")
    if contact_distance is None:
        selection = record.get("tabletop_static_selection", {})
        diagnostic_caption = (
            f"source max displacement={1e3 * float(selection.get('maximum_source_object_displacement_m', float('nan'))):.2f} mm | "
            f"source min contact force={float(selection.get('minimum_source_final_contact_force_n', float('nan'))):.2f} N"
        )
    else:
        diagnostic_caption = (
            f"selected-contact max distance={1e3 * float(contact_distance):.2f} mm"
        )
    object_path = Path(str(record["object"]["mesh_path"]))
    object_extents_mm = np.ptp(geometry["object_vertices"], axis=0) * 1.0e3
    figure.suptitle(
        (
            f"X2 tabletop static grasp | {_object_id(object_path)} | {side} | "
            f"object={object_extents_mm[0]:.1f} × {object_extents_mm[1]:.1f} × "
            f"{object_extents_mm[2]:.1f} mm | scale={float(record['object']['scale']):g}\n"
            f"table clearance={1e3 * float(table['collision_mesh_vertex_minimum_hand_plane_clearance_m']):.2f} mm | "
            f"{diagnostic_caption} | {physical_caption}"
        ),
        fontsize=11,
        color="#8f1d1d",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_sha256 = hashlib.sha256(raw_json.read_bytes()).hexdigest()
    figure.savefig(
        output,
        dpi=190,
        facecolor="white",
        metadata={
            "RawJSON": str(raw_json),
            "RawSHA256": raw_sha256,
            "PhysicalStatus": physical_status,
        },
    )
    plt.close(figure)
    return {
        "schema_version": 1,
        "renderer": "x2_table_static_collision_mesh_multiview_v1",
        "raw_json": str(raw_json),
        "raw_sha256": raw_sha256,
        "image": str(output),
        "image_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "physical_status": physical_status,
        "object_scale": float(record["object"]["scale"]),
        "source_plane_nonpenetrating": True,
        "requested_clearance_met": bool(table["requested_clearance_met"]),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = render(
        args.raw_json.expanduser().resolve(), args.output.expanduser().resolve()
    )
    manifest_path = args.output.expanduser().resolve().with_suffix(".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
