#!/usr/bin/env python3
"""Render the best front/back cylinder smoke records with SHA provenance.

The renderer is intentionally static: it reconstructs the same explicit
low-vertex collision hulls used by the X2 optimizer and overlays the object and
selected contact points.  It does not launch PhysX and does not mutate raw
candidate JSON files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "data" / "x2_mesh_grasps" / "cylinder_smoke_seed0"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_generation.utils.x2_config import load_x2_mesh_config  # noqa: E402
from grasp_generation.utils.x2_hand_model import X2HandModel  # noqa: E402
from grasp_generation.utils.x2_mesh_contacts import (  # noqa: E402
    load_generic_contact_candidates,
)


class X2CylinderRenderError(RuntimeError):
    """Raised when a smoke record cannot be reconstructed deterministically."""


def _rotation6d(rotation: np.ndarray) -> np.ndarray:
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise X2CylinderRenderError("hand_pose.rotation_matrix must be finite 3x3")
    return rotation.T[:2].reshape(6)


def _read_records(root: Path, side: str) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted((root / f"{side}_single" / "raw").glob(f"*_{side}_*.json"))
    if not paths:
        raise X2CylinderRenderError(f"No {side} raw records found below {root}")
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise X2CylinderRenderError(f"Cannot read {path}: {exc}") from exc
        collision = record.get("self_collision")
        energy = record.get("energy")
        if (
            record.get("active_side") != side
            or not isinstance(collision, dict)
            or collision.get("feasible") is not True
            or not isinstance(energy, dict)
        ):
            raise X2CylinderRenderError(f"{path} is not a feasible v4 {side} record")
        records.append((path, record))
    return records


def _best_record(root: Path, side: str) -> tuple[Path, dict[str, Any]]:
    records = _read_records(root, side)
    return min(records, key=lambda item: (float(item[1]["energy"]["total"]), item[0].name))


def _materialize_geometry(
    hand: X2HandModel,
    raw_path: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    rotation = np.asarray(record["hand_pose"]["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(record["hand_pose"]["translation"], dtype=np.float64)
    actuator = np.asarray(record["actuator"], dtype=np.float64)
    pose = np.concatenate((translation, _rotation6d(rotation), actuator))
    hand.set_parameters(torch.as_tensor(pose, dtype=hand.dtype).unsqueeze(0))
    if hand.current_status is None or hand.global_rotation is None:
        raise X2CylinderRenderError("X2 FK did not materialize")

    link_meshes: list[tuple[str, np.ndarray, np.ndarray]] = []
    global_rotation = hand.global_rotation[0].detach().cpu().numpy()
    global_translation = hand.global_translation[0].detach().cpu().numpy()
    for link_name in hand.backend.link_names:
        collision = hand.backend.collision_meshes[link_name]
        local = np.asarray(collision.vertices_local, dtype=np.float64)
        transform = hand.current_status[link_name][0].detach().cpu().numpy()
        root_points = local @ transform[:3, :3].T + transform[:3, 3]
        world_points = root_points @ global_rotation.T + global_translation
        link_meshes.append(
            (link_name, world_points, np.asarray(collision.triangles, dtype=np.int64))
        )

    object_record = record["object"]
    object_mesh = trimesh.load(
        Path(str(object_record["mesh_path"])).expanduser().resolve(),
        force="mesh",
        process=False,
    )
    if not isinstance(object_mesh, trimesh.Trimesh):
        raise X2CylinderRenderError("record object did not load as one triangle mesh")
    object_vertices = np.asarray(object_mesh.vertices, dtype=np.float64) * float(
        object_record["scale"]
    )
    contacts = np.asarray(
        [contact["world_position"] for contact in record["selected_contacts"]],
        dtype=np.float64,
    )
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    return {
        "raw_path": raw_path,
        "raw_sha256": digest,
        "record": record,
        "links": link_meshes,
        "object_vertices": object_vertices,
        "object_faces": np.asarray(object_mesh.faces, dtype=np.int64),
        "contacts": contacts,
    }


def _link_color(link_name: str) -> tuple[float, float, float, float]:
    if link_name.startswith("rh_th"):
        return (0.95, 0.50, 0.10, 0.72)
    if link_name.startswith("rh_ff"):
        return (0.85, 0.16, 0.18, 0.72)
    if link_name == "rh_palm":
        return (0.68, 0.70, 0.76, 0.45)
    return (0.36, 0.45, 0.60, 0.58)


def _draw_scene(
    axis: Any,
    geometry: dict[str, Any],
    *,
    elevation: float,
    azimuth: float,
    closeup: bool,
) -> None:
    object_vertices = geometry["object_vertices"]
    object_faces = geometry["object_faces"]
    axis.add_collection3d(
        Poly3DCollection(
            object_vertices[object_faces],
            facecolor=(0.08, 0.45, 0.95, 0.42),
            edgecolor=(0.04, 0.20, 0.50, 0.25),
            linewidth=0.18,
        )
    )
    all_points = [object_vertices]
    for link_name, vertices, faces in geometry["links"]:
        axis.add_collection3d(
            Poly3DCollection(
                vertices[faces],
                facecolor=_link_color(link_name),
                edgecolor=(0.08, 0.08, 0.10, 0.18),
                linewidth=0.12,
            )
        )
        all_points.append(vertices)
    contacts = geometry["contacts"]
    axis.scatter(
        contacts[:, 0], contacts[:, 1], contacts[:, 2],
        color="#00e6b8", edgecolors="#003d33", linewidths=0.6, s=24,
        depthshade=False,
    )
    if closeup:
        center = object_vertices.mean(axis=0)
        radius = max(float(np.ptp(object_vertices, axis=0).max()) * 1.55, 0.07)
    else:
        points = np.concatenate(all_points, axis=0)
        lower, upper = points.min(axis=0), points.max(axis=0)
        center = 0.5 * (lower + upper)
        radius = 0.55 * float((upper - lower).max())
    radius = max(radius, 1.0e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()


def _caption(geometry: dict[str, Any]) -> str:
    record = geometry["record"]
    collision = record["self_collision"]
    return (
        f"{geometry['raw_path'].name}  SHA-256 {geometry['raw_sha256']}\n"
        f"self max={1.0e3 * float(collision['maximum_penetration']):.4f} mm; "
        f"energy={float(record['energy']['total']):.6g}"
    )


def _save_multiview(geometry: dict[str, Any], output: Path) -> None:
    views = ((24.0, -64.0), (24.0, 26.0), (72.0, -90.0), (4.0, -90.0))
    figure = plt.figure(figsize=(12, 10), constrained_layout=True)
    for index, (elevation, azimuth) in enumerate(views, start=1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        _draw_scene(
            axis, geometry, elevation=elevation, azimuth=azimuth, closeup=False
        )
    figure.suptitle(_caption(geometry), fontsize=9)
    figure.savefig(
        output,
        dpi=180,
        metadata={"RawSHA256": geometry["raw_sha256"], "RawJSON": str(geometry["raw_path"])},
    )
    plt.close(figure)


def _save_comparison(
    front: dict[str, Any],
    back: dict[str, Any],
    output: Path,
    *,
    closeup: bool,
) -> None:
    figure = plt.figure(figsize=(13, 6), constrained_layout=True)
    for index, (side, geometry, azimuth) in enumerate(
        (("front", front, -64.0), ("back", back, 116.0)), start=1
    ):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        _draw_scene(
            axis,
            geometry,
            elevation=26.0,
            azimuth=azimuth,
            closeup=closeup,
        )
        axis.set_title(f"{side}: {geometry['raw_path'].name}", fontsize=9)
    figure.suptitle(f"{_caption(front)}\n{_caption(back)}", fontsize=8)
    figure.savefig(
        output,
        dpi=180,
        metadata={
            "FrontRawSHA256": front["raw_sha256"],
            "BackRawSHA256": back["raw_sha256"],
        },
    )
    plt.close(figure)


def render(input_root: Path, output_dir: Path) -> dict[str, Any]:
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
    front_path, front_record = _best_record(input_root, "front")
    back_path, back_record = _best_record(input_root, "back")
    front = _materialize_geometry(hand, front_path, front_record)
    back = _materialize_geometry(hand, back_path, back_record)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "front_best_multiview.png": front,
        "back_best_multiview.png": back,
    }
    for filename, geometry in outputs.items():
        _save_multiview(geometry, output_dir / filename)
    _save_comparison(
        front, back, output_dir / "front_back_best_overview.png", closeup=False
    )
    _save_comparison(
        front,
        back,
        output_dir / "front_back_best_contact_closeup.png",
        closeup=True,
    )

    def source_record(geometry: dict[str, Any]) -> dict[str, Any]:
        raw_path = geometry["raw_path"].resolve()
        try:
            portable_path = str(raw_path.relative_to(PROJECT_ROOT))
        except ValueError:
            portable_path = str(raw_path)
        record = geometry["record"]
        return {
            "raw_json": portable_path,
            "raw_sha256": geometry["raw_sha256"],
            "energy_total": float(record["energy"]["total"]),
            "self_collision_maximum_penetration_m": float(
                record["self_collision"]["maximum_penetration"]
            ),
        }

    manifest = {
        "schema_version": 1,
        "renderer": "deterministic_x2_collision_hull_matplotlib",
        "geometry": "optimizer_and_physx_shared_low_vertex_collision_hulls",
        "physx_simulation_run": False,
        "sources": {"front": source_record(front), "back": source_record(back)},
        "images": {
            "front_best_multiview.png": ["front"],
            "back_best_multiview.png": ["back"],
            "front_back_best_overview.png": ["front", "back"],
            "front_back_best_contact_closeup.png": ["front", "back"],
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_root = args.input_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_root / "visualizations"
    )
    manifest = render(input_root, output_dir)
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
