#!/usr/bin/env python3
"""Render deterministic 1--5-finger samples from an X2 collection attempt.

This is a read-only visualization helper.  It reconstructs the exact X2
collision hull pose stored in each raw JSON, overlays the object and selected
contact points, and records SHA-256 provenance.  It does not run PhysX and it
never labels raw optimizer output as valid data.
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "x2_valid_5000" / "attempts" / "attempt_0000"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "x2_valid_5000" / "sample_visualizations"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_generation.utils.x2_config import load_x2_mesh_config  # noqa: E402
from grasp_generation.utils.x2_hand_model import X2HandModel  # noqa: E402
from grasp_generation.utils.x2_mesh_contacts import (  # noqa: E402
    load_generic_contact_candidates,
)
from scripts.render_x2_cylinder_smoke import _materialize_geometry  # noqa: E402


class X2SampleRenderError(RuntimeError):
    """Raised when representative raw samples cannot be rendered safely."""


FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
FINGER_LABELS = {
    "thumb": "Thumb",
    "index": "Index",
    "middle": "Middle",
    "ring": "Ring",
    "little": "Little",
}
FINGER_COLORS = {
    "thumb": "#f28e2b",
    "index": "#e15759",
    "middle": "#59a14f",
    "ring": "#af7aa1",
    "little": "#4eafc0",
}
SIDE_BY_FINGER_COUNT = {1: "front", 2: "back", 3: "front", 4: "back", 5: "front"}


def _portable(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _finger_for_link(link_name: str) -> str | None:
    prefixes = {
        "rh_th": "thumb",
        "rh_ff": "index",
        "rh_mf": "middle",
        "rh_rf": "ring",
        "rh_lf": "little",
    }
    for prefix, finger in prefixes.items():
        if link_name.startswith(prefix):
            return finger
    return None


def _load_candidate(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    participation = payload.get("finger_participation")
    self_collision = payload.get("self_collision")
    penetration = payload.get("hand_object_penetration")
    validation = payload.get("validation")
    if not (
        str(payload.get("pipeline_revision", "")).endswith("_v6")
        and payload.get("finite") is True
        and isinstance(participation, dict)
        and participation.get("target_count") == participation.get("actual_count")
        and isinstance(self_collision, dict)
        and self_collision.get("feasible") is True
        and isinstance(penetration, dict)
        and penetration.get("evaluation_mode") == "dense_bidirectional"
        and penetration.get("evaluated") is True
        and penetration.get("feasible") is True
        and isinstance(validation, dict)
        and validation.get("status") == "not_run"
    ):
        return None
    return payload


def _object_id(payload: dict[str, Any]) -> str:
    mesh_path = Path(str(payload["object"]["mesh_path"]))
    if mesh_path.stem == "decomposed" and mesh_path.parent.name == "coacd":
        return mesh_path.parent.parent.name
    return mesh_path.stem


def _select_samples(input_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    pools: dict[int, list[tuple[Path, dict[str, Any]]]] = {
        count: [] for count in SIDE_BY_FINGER_COUNT
    }
    for path in sorted(input_root.glob("**/raw/*.json")):
        payload = _load_candidate(path)
        if payload is None:
            continue
        count = int(payload["finger_participation"]["actual_count"])
        if count in pools and payload.get("active_side") == SIDE_BY_FINGER_COUNT[count]:
            pools[count].append((path, payload))

    selected: list[tuple[Path, dict[str, Any]]] = []
    used_objects: set[str] = set()
    for count in SIDE_BY_FINGER_COUNT:
        ranked = sorted(
            pools[count],
            key=lambda item: (float(item[1]["energy"]["total"]), str(item[0])),
        )
        choice = next(
            (item for item in ranked if _object_id(item[1]) not in used_objects),
            None,
        )
        if choice is None:
            raise X2SampleRenderError(
                f"Cannot select a unique-object f{count} {SIDE_BY_FINGER_COUNT[count]} sample"
            )
        selected.append(choice)
        used_objects.add(_object_id(choice[1]))
    return selected


def _link_color(link_name: str, active: set[str]) -> tuple[str, float]:
    finger = _finger_for_link(link_name)
    if finger is not None:
        if finger in active:
            return FINGER_COLORS[finger], 0.88
        return "#c5c9d0", 0.24
    if link_name == "rh_palm":
        return "#7f8794", 0.52
    return "#aeb4be", 0.32


def _draw_scene(
    axis: Any,
    geometry: dict[str, Any],
    *,
    elevation: float,
    azimuth: float,
    closeup: bool,
) -> None:
    payload = geometry["record"]
    active = set(payload["finger_participation"]["finger_names"])
    object_vertices = geometry["object_vertices"]
    object_faces = geometry["object_faces"]
    axis.add_collection3d(
        Poly3DCollection(
            object_vertices[object_faces],
            facecolor=(0.10, 0.42, 0.92, 0.34),
            edgecolor=(0.03, 0.14, 0.38, 0.30),
            linewidth=0.20,
        )
    )
    all_points = [object_vertices]
    for link_name, vertices, faces in geometry["links"]:
        color, alpha = _link_color(link_name, active)
        axis.add_collection3d(
            Poly3DCollection(
                vertices[faces],
                facecolor=color,
                alpha=alpha,
                edgecolor=(0.06, 0.06, 0.08, min(alpha, 0.24)),
                linewidth=0.12,
            )
        )
        all_points.append(vertices)

    contacts = geometry["contacts"]
    axis.scatter(
        contacts[:, 0],
        contacts[:, 1],
        contacts[:, 2],
        color="#ffe34f",
        edgecolors="#302800",
        linewidths=0.75,
        s=34,
        depthshade=False,
    )
    if closeup:
        lower, upper = object_vertices.min(axis=0), object_vertices.max(axis=0)
        center = 0.5 * (lower + upper)
        radius = max(0.72 * float((upper - lower).max()), 0.055)
    else:
        points = np.concatenate(all_points, axis=0)
        lower, upper = points.min(axis=0), points.max(axis=0)
        center = 0.5 * (lower + upper)
        radius = 0.56 * float((upper - lower).max())
    radius = max(radius, 1.0e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()


def _sample_title(geometry: dict[str, Any]) -> str:
    payload = geometry["record"]
    participation = payload["finger_participation"]
    names = ", ".join(FINGER_LABELS[name] for name in participation["finger_names"])
    penetration_mm = 1.0e3 * float(
        payload["hand_object_penetration"]["maximum_penetration"]
    )
    return (
        f"f{participation['actual_count']} | {payload['active_side']} | "
        f"{_object_id(payload)}\n"
        f"Active: {names} | energy={float(payload['energy']['total']):.4f} | "
        f"dense max={penetration_mm:.3f} mm"
    )


def _save_individual(geometry: dict[str, Any], output: Path) -> None:
    side = geometry["record"]["active_side"]
    primary_azimuth = -64.0 if side == "front" else 116.0
    views = (
        (26.0, primary_azimuth, False, "Whole hand"),
        (26.0, primary_azimuth + 90.0, False, "Side view"),
        (26.0, primary_azimuth, True, "Contact close-up"),
        (72.0, primary_azimuth, True, "Top close-up"),
    )
    figure = plt.figure(figsize=(12, 10), constrained_layout=True)
    for index, (elevation, azimuth, closeup, title) in enumerate(views, start=1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        _draw_scene(
            axis,
            geometry,
            elevation=elevation,
            azimuth=azimuth,
            closeup=closeup,
        )
        axis.set_title(title, fontsize=9)
    figure.suptitle(
        f"{_sample_title(geometry)}\nRAW OPTIMIZER POSE — NOT YET PHYSX VALIDATED",
        fontsize=11,
        color="#9c1c1c",
    )
    figure.savefig(
        output,
        dpi=180,
        facecolor="white",
        metadata={
            "RawSHA256": geometry["raw_sha256"],
            "RawJSON": _portable(geometry["raw_path"]),
            "PhysXValidated": "false",
        },
    )
    plt.close(figure)


def _save_overview(geometries: list[dict[str, Any]], output: Path) -> None:
    figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    for index, geometry in enumerate(geometries, start=1):
        side = geometry["record"]["active_side"]
        axis = figure.add_subplot(2, 3, index, projection="3d")
        _draw_scene(
            axis,
            geometry,
            elevation=26.0,
            azimuth=-64.0 if side == "front" else 116.0,
            closeup=False,
        )
        axis.set_title(_sample_title(geometry), fontsize=9)

    legend_axis = figure.add_subplot(2, 3, 6)
    legend_axis.axis("off")
    handles = [
        Patch(facecolor=FINGER_COLORS[name], label=FINGER_LABELS[name])
        for name in FINGER_ORDER
    ]
    handles.extend(
        (
            Patch(facecolor="#246bd6", alpha=0.45, label="Object mesh"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#ffe34f",
                markeredgecolor="#302800",
                markersize=8,
                label="Selected contact",
            ),
            Patch(facecolor="#c5c9d0", alpha=0.35, label="Inactive finger"),
        )
    )
    legend_axis.legend(handles=handles, loc="center", frameon=False, fontsize=10)
    legend_axis.text(
        0.5,
        0.08,
        "RAW / NOT YET PHYSX VALIDATED",
        ha="center",
        va="center",
        color="#9c1c1c",
        fontsize=12,
        fontweight="bold",
        transform=legend_axis.transAxes,
    )
    figure.suptitle(
        "X2 formal collection: deterministic 1–5 finger pose samples",
        fontsize=14,
    )
    figure.savefig(
        output,
        dpi=180,
        facecolor="white",
        metadata={"PhysXValidated": "false"},
    )
    plt.close(figure)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(input_root: Path, output_dir: Path) -> dict[str, Any]:
    chosen = _select_samples(input_root)
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
    geometries = [
        _materialize_geometry(hand, raw_path, payload)
        for raw_path, payload in chosen
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_entries: list[dict[str, Any]] = []
    for geometry in geometries:
        payload = geometry["record"]
        participation = payload["finger_participation"]
        object_id = _object_id(payload)
        filename = (
            f"sample_f{participation['actual_count']}_"
            f"{payload['active_side']}_{object_id}.png"
        )
        image_path = output_dir / filename
        _save_individual(geometry, image_path)
        sample_entries.append(
            {
                "raw_json": _portable(geometry["raw_path"]),
                "raw_sha256": geometry["raw_sha256"],
                "pipeline_revision": payload["pipeline_revision"],
                "validation_status": payload["validation"]["status"],
                "physx_validation_run": False,
                "object_id": object_id,
                "object_mesh": _portable(Path(payload["object"]["mesh_path"])),
                "active_side": payload["active_side"],
                "active_finger_count": participation["actual_count"],
                "active_fingers": participation["finger_names"],
                "energy_total": float(payload["energy"]["total"]),
                "dense_maximum_penetration_m": float(
                    payload["hand_object_penetration"]["maximum_penetration"]
                ),
                "image": filename,
                "image_sha256": _sha256(image_path),
            }
        )

    overview_path = output_dir / "grasp_samples_overview.png"
    _save_overview(geometries, overview_path)
    manifest = {
        "schema_version": 1,
        "renderer": "deterministic_x2_collision_hull_matplotlib",
        "source_root": _portable(input_root),
        "selection": (
            "lowest energy per f1..f5 on alternating front/back sides, "
            "with unique object ids"
        ),
        "status": "raw_unvalidated",
        "physx_simulation_run": False,
        "overview_image": overview_path.name,
        "overview_sha256": _sha256(overview_path),
        "samples": sample_entries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = render(
        args.input_root.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
