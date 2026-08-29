#!/usr/bin/env python3
"""Render representative X2 cross-object composed warm-start candidates.

The renderer reconstructs the composed hand qpose in its shared hand frame,
places both object meshes using the stored inverse source hand poses, and
transforms each source record's selected contacts into that same frame.  It is
read-only and does not run the required joint dual-object PhysX validation.
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
import trimesh  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "x2_dual_object_20260727_105600"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "visualizations"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_generation.utils.x2_config import load_x2_mesh_config  # noqa: E402
from grasp_generation.utils.x2_hand_model import X2HandModel  # noqa: E402
from grasp_generation.utils.x2_mesh_contacts import (  # noqa: E402
    load_generic_contact_candidates,
)


class X2DualRenderError(RuntimeError):
    """Raised when a composed candidate cannot be reconstructed."""


RIGHT_COLOR = "#e76f51"
LEFT_COLOR = "#2a9d8f"
PALM_COLOR = "#8d99ae"
OBJECT_COLORS = ("#3a86ff", "#9b5de5")
CONTACT_COLORS = ("#ffd166", "#ff9f1c")
FINGER_LABELS = {
    "thumb": "Thumb",
    "index": "Index",
    "middle": "Middle",
    "ring": "Ring",
    "little": "Little",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _rotation6d(rotation: np.ndarray) -> np.ndarray:
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise X2DualRenderError("hand rotation must be a finite 3x3 matrix")
    return rotation.T[:2].reshape(6)


def _finger_for_link(link_name: str) -> str | None:
    prefixes = {
        "rh_th": "thumb",
        "rh_ff": "index",
        "rh_mf": "middle",
        "rh_rf": "ring",
        "rh_lf": "little",
    }
    return next(
        (finger for prefix, finger in prefixes.items() if link_name.startswith(prefix)),
        None,
    )


def _strict_candidate(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise X2DualRenderError(f"Cannot read {path}: {exc}") from exc
    hand = payload.get("hand")
    objects = payload.get("objects")
    checks = payload.get("composition_checks")
    validation = payload.get("dual_object_validation")
    if (
        not isinstance(hand, dict)
        or not isinstance(hand.get("actuator"), list)
        or len(hand["actuator"]) != 12
        or not isinstance(hand.get("joint"), list)
        or len(hand["joint"]) != 16
        or not isinstance(objects, list)
        or len(objects) != 2
        or objects[0].get("object_id") == objects[1].get("object_id")
        or not isinstance(checks, dict)
        or checks.get("different_objects") is not True
        or checks.get("disjoint_finger_sets") is not True
        or checks.get("all_five_fingers_assigned") is not True
        or not isinstance(validation, dict)
        or validation.get("status") != "not_run"
    ):
        raise X2DualRenderError(f"{path}: malformed dual-object candidate")
    return payload


def _select_candidates(input_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    manifest_path = input_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise X2DualRenderError(f"Cannot read {manifest_path}: {exc}") from exc
    descriptors = manifest.get("dual_object_candidates")
    if not isinstance(descriptors, list):
        raise X2DualRenderError("manifest dual_object_candidates is missing")
    selected: list[tuple[Path, dict[str, Any]]] = []
    used_objects: set[str] = set()
    for right_count in (1, 2, 3, 4):
        pool = [
            descriptor
            for descriptor in descriptors
            if descriptor.get("right_finger_count") == right_count
        ]
        if not pool:
            raise X2DualRenderError(f"No right f{right_count} candidate")

        def rank(descriptor: dict[str, Any]) -> tuple[int, int, str]:
            object_ids = {
                str(descriptor.get("right_object_id")),
                str(descriptor.get("left_object_id")),
            }
            primitive_count = sum(not value.isdigit() for value in object_ids)
            new_count = len(object_ids - used_objects)
            return (-primitive_count, -new_count, str(descriptor.get("path")))

        descriptor = min(pool, key=rank)
        path = Path(str(descriptor["path"])).expanduser().resolve()
        if not path.is_file() or _sha256(path) != descriptor.get("sha256"):
            raise X2DualRenderError(f"Candidate path/hash is stale: {path}")
        payload = _strict_candidate(path)
        selected.append((path, payload))
        used_objects.update(str(value["object_id"]) for value in payload["objects"])
    return selected


def _materialize_hand(
    hand: X2HandModel,
    payload: dict[str, Any],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    pose_record = payload["hand"]["pose"]
    translation = np.asarray(pose_record["translation"], dtype=np.float64)
    rotation = np.asarray(pose_record["rotation_matrix"], dtype=np.float64)
    actuator = np.asarray(payload["hand"]["actuator"], dtype=np.float64)
    pose = np.concatenate((translation, _rotation6d(rotation), actuator))
    hand.set_parameters(torch.as_tensor(pose, dtype=hand.dtype).unsqueeze(0))
    if hand.current_status is None or hand.global_rotation is None:
        raise X2DualRenderError("X2 FK did not materialize")
    global_rotation = hand.global_rotation[0].detach().cpu().numpy()
    global_translation = hand.global_translation[0].detach().cpu().numpy()
    result: list[tuple[str, np.ndarray, np.ndarray]] = []
    for link_name in hand.backend.link_names:
        collision = hand.backend.collision_meshes[link_name]
        local = np.asarray(collision.vertices_local, dtype=np.float64)
        transform = hand.current_status[link_name][0].detach().cpu().numpy()
        root_points = local @ transform[:3, :3].T + transform[:3, 3]
        world_points = root_points @ global_rotation.T + global_translation
        result.append(
            (
                link_name,
                world_points,
                np.asarray(collision.triangles, dtype=np.int64),
            )
        )
    return result


def _materialize_object(record: dict[str, Any]) -> dict[str, Any]:
    mesh_path = Path(str(record["mesh_path"])).expanduser().resolve()
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise X2DualRenderError(f"Object mesh is invalid: {mesh_path}")
    pose = record["pose_in_shared_hand_frame"]
    rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(pose["translation"], dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(record["scale"])
    vertices = vertices @ rotation.T + translation

    source_path = Path(str(record["source_valid"])).expanduser().resolve()
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise X2DualRenderError(f"Cannot read source {source_path}: {exc}") from exc
    contacts = np.asarray(
        [value["world_position"] for value in source["selected_contacts"]],
        dtype=np.float64,
    )
    contacts = contacts @ rotation.T + translation
    return {
        "record": record,
        "vertices": vertices,
        "faces": np.asarray(mesh.faces, dtype=np.int64),
        "contacts": contacts,
    }


def _geometry(
    hand: X2HandModel,
    path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": _sha256(path),
        "payload": payload,
        "links": _materialize_hand(hand, payload),
        "objects": [_materialize_object(value) for value in payload["objects"]],
    }


def _draw_scene(
    axis: Any,
    geometry: dict[str, Any],
    *,
    elevation: float,
    azimuth: float,
) -> None:
    payload = geometry["payload"]
    owners = payload["hand"]["finger_mode_owner"]
    all_points: list[np.ndarray] = []
    for index, object_geometry in enumerate(geometry["objects"]):
        vertices = object_geometry["vertices"]
        faces = object_geometry["faces"]
        axis.add_collection3d(
            Poly3DCollection(
                vertices[faces],
                facecolor=OBJECT_COLORS[index],
                alpha=0.36,
                edgecolor=OBJECT_COLORS[index],
                linewidth=0.15,
            )
        )
        contacts = object_geometry["contacts"]
        axis.scatter(
            contacts[:, 0],
            contacts[:, 1],
            contacts[:, 2],
            color=CONTACT_COLORS[index],
            edgecolors="#332500",
            linewidths=0.55,
            s=25,
            depthshade=False,
        )
        all_points.append(vertices)
    for link_name, vertices, faces in geometry["links"]:
        finger = _finger_for_link(link_name)
        owner = owners.get(finger) if finger is not None else None
        color = RIGHT_COLOR if owner == "right" else LEFT_COLOR if owner == "left" else PALM_COLOR
        alpha = 0.84 if owner is not None else 0.45
        axis.add_collection3d(
            Poly3DCollection(
                vertices[faces],
                facecolor=color,
                alpha=alpha,
                edgecolor=(0.06, 0.06, 0.08, 0.16),
                linewidth=0.10,
            )
        )
        # Keep the long palm/forearm visible but do not let it dominate the
        # camera bounds; the composed objects and finger posture are the
        # subject of this diagnostic image.
        if finger is not None:
            all_points.append(vertices)
    points = np.concatenate(all_points, axis=0)
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.58 * float((upper - lower).max()), 0.06)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()


def _title(geometry: dict[str, Any]) -> str:
    payload = geometry["payload"]
    right, left = payload["objects"]
    right_fingers = ", ".join(FINGER_LABELS[value] for value in right["finger_names"])
    left_fingers = ", ".join(FINGER_LABELS[value] for value in left["finger_names"])
    return (
        f"{payload['candidate_id']}\n"
        f"Front/right: {right['object_id']} [{right_fingers}]\n"
        f"Back/left: {left['object_id']} [{left_fingers}]"
    )


def _save_individual(geometry: dict[str, Any], output: Path) -> None:
    views = ((24.0, -62.0), (24.0, 28.0), (68.0, -62.0), (8.0, 118.0))
    figure = plt.figure(figsize=(12, 10), constrained_layout=True)
    for index, (elevation, azimuth) in enumerate(views, start=1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        _draw_scene(axis, geometry, elevation=elevation, azimuth=azimuth)
        axis.set_title(("Main", "Side", "Top", "Reverse")[index - 1], fontsize=9)
    figure.suptitle(
        f"{_title(geometry)}\n"
        "COMPOSED WARM-START — JOINT DUAL-OBJECT PHYSX NOT RUN",
        color="#a61b1b",
        fontsize=11,
    )
    figure.savefig(
        output,
        dpi=180,
        facecolor="white",
        metadata={
            "CandidateSHA256": geometry["sha256"],
            "CandidateJSON": _portable(geometry["path"]),
            "DualObjectPhysXValidated": "false",
            "RenderedState": "composed_input_qpose",
        },
    )
    plt.close(figure)


def _save_overview(geometries: list[dict[str, Any]], output: Path) -> None:
    figure = plt.figure(figsize=(15, 12), constrained_layout=True)
    for index, geometry in enumerate(geometries, start=1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        _draw_scene(axis, geometry, elevation=24.0, azimuth=-62.0)
        axis.set_title(_title(geometry), fontsize=9)
    handles = [
        Patch(facecolor=RIGHT_COLOR, label="Front/right-owned fingers"),
        Patch(facecolor=LEFT_COLOR, label="Back/left-owned fingers"),
        Patch(facecolor=OBJECT_COLORS[0], alpha=0.40, label="Front/right object"),
        Patch(facecolor=OBJECT_COLORS[1], alpha=0.40, label="Back/left object"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=CONTACT_COLORS[0],
            markeredgecolor="#332500",
            markersize=7,
            label="Source selected contacts",
        ),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    figure.suptitle(
        "X2 cross-object composed qposes\n"
        "WARM-START CANDIDATES — JOINT DUAL-OBJECT PHYSX NOT RUN",
        fontsize=15,
        color="#a61b1b",
    )
    figure.savefig(
        output,
        dpi=180,
        facecolor="white",
        metadata={
            "DualObjectPhysXValidated": "false",
            "RenderedState": "composed_input_qpose",
        },
    )
    plt.close(figure)


def render(input_root: Path, output_dir: Path) -> dict[str, Any]:
    selected = _select_candidates(input_root)
    config = load_x2_mesh_config()
    contacts = load_generic_contact_candidates(
        config.configured_path("contact_candidates.path", must_exist=True)
    )
    hand = X2HandModel(
        config,
        contacts,
        device="cpu",
        dtype=torch.float64,
        collision_samples_per_link=1,
        self_collision_samples_per_link=1,
    )
    geometries = [
        _geometry(hand, path, payload) for path, payload in selected
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    for geometry in geometries:
        candidate_id = geometry["payload"]["candidate_id"]
        image_path = output_dir / f"{candidate_id}.png"
        _save_individual(geometry, image_path)
        samples.append(
            {
                "candidate_id": candidate_id,
                "candidate_json": _portable(geometry["path"]),
                "candidate_sha256": geometry["sha256"],
                "image": image_path.name,
                "image_sha256": _sha256(image_path),
                "dual_object_validation": "not_run",
            }
        )
    overview = output_dir / "dual_object_composed_overview.png"
    _save_overview(geometries, overview)
    manifest = {
        "schema_version": 1,
        "renderer": "deterministic_x2_dual_object_collision_hull_matplotlib",
        "source_root": _portable(input_root),
        "rendered_state": "composed_input_qpose",
        "dual_object_physx_validated": False,
        "overview_image": overview.name,
        "overview_sha256": _sha256(overview),
        "samples": samples,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = render(
        args.input_root.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )
    print(json.dumps(manifest, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
