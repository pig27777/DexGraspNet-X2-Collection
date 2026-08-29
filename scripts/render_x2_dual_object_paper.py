#!/usr/bin/env python3
"""Render paper-style images of PhysX-valid X2 dual-object grasps.

Unlike :mod:`render_x2_dual_object_samples`, this renderer is intended for
figures rather than diagnostics.  It uses the authored high-resolution X2
visual meshes, an orthographic camera, opaque materials, and no annotations or
contact markers.  The default input is restricted to records that passed all
six dual-object PhysX orientations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "data" / "x2_dual_object" / "physx_validation" / "valid"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "x2_dual_object" / "paper_renders"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_generation.utils.x2_config import load_x2_mesh_config  # noqa: E402
from grasp_generation.utils.x2_hand_model import X2HandModel  # noqa: E402
from grasp_generation.utils.x2_mesh_contacts import (  # noqa: E402
    load_generic_contact_candidates,
)
from scripts.render_x2_dual_object_samples import (  # noqa: E402
    X2DualRenderError,
    _finger_for_link,
    _materialize_object,
    _portable,
    _rotation6d,
    _sha256,
)


VALIDATION_PROTOCOL = "x2_dual_object_six_orientation_physx_v1"
MAXIMUM_RENDER_PENETRATION_M = 0.0
FINGER_COLORS = {
    "right": "#d95f59",
    "left": "#3b8fc4",
    "palm": "#7f8896",
}
OBJECT_COLORS = ("#e1dfd8", "#a9b2bf")
CAMERA_VIEWS = {
    "main": (24.0, -62.0),
    "side": (20.0, 28.0),
    "top": (68.0, -62.0),
    "reverse": (10.0, 118.0),
}
DEXGRASPNET_GRID_SIZE = (2288, 1232)
DEXGRASPNET_ROW_SIZE = (2288, 411)
DEXGRASPNET_GRID_COLUMNS = 6
DEXGRASPNET_GRID_ROWS = 3
DEXGRASPNET_CONTRAST_FACTOR = 1.25
DEXGRASPNET_COLOR_FACTOR = 1.25
DEXGRASPNET_UNSHARP_RADIUS_PX = 0.8
DEXGRASPNET_UNSHARP_PERCENT = 60
DEXGRASPNET_UNSHARP_THRESHOLD = 2


def _strict_valid_record(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise X2DualRenderError(f"Cannot read {path}: {exc}") from exc
    hand = payload.get("hand")
    objects = payload.get("objects")
    checks = payload.get("composition_checks")
    validation = payload.get("dual_object_validation")
    orientations = validation.get("orientations") if isinstance(validation, dict) else None
    static_preflight = (
        validation.get("static_preflight") if isinstance(validation, dict) else None
    )
    try:
        static_penetrations = [
            float(static_preflight["hand_self_collision"]["maximum_penetration_m"]),
            float(static_preflight["object_object"]["maximum_penetration_m"]),
            *[
                float(value["maximum_penetration_m"])
                for value in static_preflight["hand_object"]["objects"]
            ],
        ]
    except (KeyError, TypeError, ValueError):
        static_penetrations = []
    if (
        not isinstance(hand, dict)
        or not isinstance(hand.get("actuator"), list)
        or len(hand["actuator"]) != 12
        or not isinstance(objects, list)
        or len(objects) != 2
        or objects[0].get("object_id") == objects[1].get("object_id")
        or not isinstance(checks, dict)
        or checks.get("different_objects") is not True
        or checks.get("disjoint_finger_sets") is not True
        or checks.get("all_five_fingers_assigned") is not True
        or not isinstance(validation, dict)
        or validation.get("status") != "passed"
        or validation.get("backend") != "isaac_sim_physx"
        or validation.get("protocol_revision") != VALIDATION_PROTOCOL
        or validation.get("passed_orientation_count") != 6
        or validation.get("required_orientation_count") != 6
        or not isinstance(orientations, list)
        or len(orientations) != 6
        or not all(
            isinstance(value, dict)
            and value.get("passed") is True
            and value.get("finite") is True
            for value in orientations
        )
        or payload.get("dual_object_success") is not True
        or len(static_penetrations) != 4
        or not all(
            value <= MAXIMUM_RENDER_PENETRATION_M for value in static_penetrations
        )
    ):
        raise X2DualRenderError(
            f"{path}: record is not a zero-penetration, six-orientation "
            "PhysX-valid dual grasp"
        )
    return payload


def _valid_paths(input_root: Path) -> list[Path]:
    paths = sorted(input_root.rglob("*.json"))
    return [path for path in paths if path.name not in {"manifest.json", "summary.json"}]


def _representative_records(
    input_root: Path,
    requested: Sequence[str],
    target_count: int,
) -> list[tuple[Path, dict[str, Any]]]:
    paths = _valid_paths(input_root)
    if not paths:
        raise X2DualRenderError(f"No JSON records found below {input_root}")

    if requested:
        by_stem = {path.stem: path for path in paths}
        selected_paths: list[Path] = []
        for value in requested:
            direct = Path(value).expanduser()
            if direct.is_file():
                selected_paths.append(direct.resolve())
                continue
            candidate_id = direct.stem
            if candidate_id not in by_stem:
                raise X2DualRenderError(
                    f"Candidate {value!r} was not found below {input_root}"
                )
            selected_paths.append(by_stem[candidate_id])
        return [(path, _strict_valid_record(path)) for path in selected_paths]

    pools: dict[int, list[tuple[Path, dict[str, Any]]]] = {
        count: [] for count in (1, 2, 3, 4)
    }
    for path in paths:
        try:
            payload = _strict_valid_record(path)
        except X2DualRenderError:
            continue
        right_count = len(payload["objects"][0]["finger_names"])
        if right_count in pools:
            pools[right_count].append((path, payload))

    if target_count < 1:
        raise X2DualRenderError("target_count must be positive")
    if target_count == 4:
        return _one_per_finger_split(pools)
    return _diverse_grid_records(pools, target_count)


def _apply_grid_replacements(
    input_root: Path,
    selected: list[tuple[Path, dict[str, Any]]],
    replacements: dict[int, str],
) -> list[tuple[Path, dict[str, Any]]]:
    if not replacements:
        return selected
    by_stem = {path.stem: path for path in _valid_paths(input_root)}
    result = list(selected)
    for index, candidate_id in sorted(replacements.items()):
        if index < 0 or index >= len(result):
            raise X2DualRenderError(f"Replacement cell index is out of range: {index}")
        if candidate_id not in by_stem:
            raise X2DualRenderError(
                f"Replacement candidate {candidate_id!r} was not found below {input_root}"
            )
        path = by_stem[candidate_id]
        payload = _strict_valid_record(path)
        duplicate_at = next(
            (
                position
                for position, (_, value) in enumerate(result)
                if position != index and value["candidate_id"] == candidate_id
            ),
            None,
        )
        if duplicate_at is not None:
            raise X2DualRenderError(
                f"Replacement candidate {candidate_id!r} is already in cell {duplicate_at + 1}"
            )
        result[index] = (path, payload)
    return result


def _shape_family(object_id: str) -> str:
    if object_id.startswith(("cube_", "cuboid_")):
        return "box"
    return object_id.split("_")[0]


def _one_per_finger_split(
    pools: dict[int, list[tuple[Path, dict[str, Any]]]],
) -> list[tuple[Path, dict[str, Any]]]:
    selected: list[tuple[Path, dict[str, Any]]] = []
    used_objects: set[str] = set()
    for right_count in (1, 2, 3, 4):
        pool = pools[right_count]
        if not pool:
            raise X2DualRenderError(f"No valid right-f{right_count} record found")

        def rank(item: tuple[Path, dict[str, Any]]) -> tuple[int, int, int, str]:
            object_ids = {str(value["object_id"]) for value in item[1]["objects"]}
            primitive_penalty = sum(value.isdigit() for value in object_ids)
            shape_families = {_shape_family(value) for value in object_ids}
            shape_diversity_penalty = -len(shape_families)
            new_object_count = len(object_ids - used_objects)
            return (
                primitive_penalty,
                shape_diversity_penalty,
                -new_object_count,
                str(item[0]),
            )

        choice = min(pool, key=rank)
        selected.append(choice)
        used_objects.update(str(value["object_id"]) for value in choice[1]["objects"])
    return selected


def _diverse_grid_records(
    pools: dict[int, list[tuple[Path, dict[str, Any]]]],
    target_count: int,
) -> list[tuple[Path, dict[str, Any]]]:
    """Select a balanced, object-diverse sequence for a qualitative grid."""

    base = target_count // 4
    quotas = {
        right_count: min(base, len(pools[right_count]))
        for right_count in (1, 2, 3, 4)
    }
    remaining = target_count - sum(quotas.values())
    while remaining:
        candidates = [
            right_count
            for right_count in (1, 2, 3, 4)
            if quotas[right_count] < len(pools[right_count])
        ]
        if not candidates:
            raise X2DualRenderError(
                f"Only {target_count - remaining} zero-penetration records are "
                f"available for a {target_count}-image grid"
            )
        for right_count in candidates:
            if not remaining:
                break
            quotas[right_count] += 1
            remaining -= 1
    order: list[int] = []
    while len(order) < target_count:
        for right_count in (1, 2, 3, 4):
            if order.count(right_count) < quotas[right_count]:
                order.append(right_count)

    selected: list[tuple[Path, dict[str, Any]]] = []
    selected_paths: set[Path] = set()
    used_objects: set[str] = set()
    used_pairs: set[tuple[str, str]] = set()
    used_families: set[str] = set()
    for right_count in order:
        pool = [item for item in pools[right_count] if item[0] not in selected_paths]
        if not pool:
            raise X2DualRenderError(
                f"Only {len(selected)} records are available for a {target_count}-image grid"
            )

        def rank(item: tuple[Path, dict[str, Any]]) -> tuple[Any, ...]:
            object_ids = tuple(str(value["object_id"]) for value in item[1]["objects"])
            families = {_shape_family(value) for value in object_ids}
            new_object_count = len(set(object_ids) - used_objects)
            new_family_count = len(families - used_families)
            pair_reuse = int(object_ids in used_pairs or object_ids[::-1] in used_pairs)
            # Numeric IDs are the non-primitive mesh objects and most closely
            # match the varied object silhouettes in the paper figure.
            primitive_count = sum(not value.isdigit() for value in object_ids)
            return (
                -new_object_count,
                pair_reuse,
                primitive_count,
                -new_family_count,
                -len(families),
                str(item[0]),
            )

        choice = min(pool, key=rank)
        selected.append(choice)
        selected_paths.add(choice[0])
        object_ids = tuple(str(value["object_id"]) for value in choice[1]["objects"])
        used_objects.update(object_ids)
        used_pairs.add(object_ids)
        used_families.update(_shape_family(value) for value in object_ids)
    return selected


def _gf_matrix_to_column(matrix: Any) -> np.ndarray:
    """Convert USD's row-vector Gf matrix convention to a column transform."""

    return np.asarray(matrix, dtype=np.float64).T


def _triangles(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if counts.ndim != 1 or indices.ndim != 1 or int(counts.sum()) != len(indices):
        raise X2DualRenderError("Visual mesh has malformed polygon topology")
    if np.all(counts == 3):
        return indices.reshape(-1, 3).copy()
    result = np.empty((int(np.sum(counts - 2)), 3), dtype=np.int64)
    source_cursor = 0
    output_cursor = 0
    for count in counts.tolist():
        face = indices[source_cursor : source_cursor + count]
        source_cursor += count
        for face_index in range(1, count - 1):
            result[output_cursor] = (face[0], face[face_index], face[face_index + 1])
            output_cursor += 1
    return result


def _cluster_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simplify a dense visual mesh with deterministic vertex clustering."""

    if voxel_size <= 0.0:
        return vertices, faces
    origin = vertices.min(axis=0)
    keys = np.floor((vertices - origin) / voxel_size + 0.5).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    vertex_count = int(inverse.max()) + 1
    clustered = np.zeros((vertex_count, 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=vertex_count).astype(np.float64)
    for axis in range(3):
        np.add.at(clustered[:, axis], inverse, vertices[:, axis])
    clustered /= counts[:, None]

    mapped = inverse[faces]
    nondegenerate = (
        (mapped[:, 0] != mapped[:, 1])
        & (mapped[:, 1] != mapped[:, 2])
        & (mapped[:, 0] != mapped[:, 2])
    )
    mapped = mapped[nondegenerate]
    canonical = np.sort(mapped, axis=1)
    _, first = np.unique(canonical, axis=0, return_index=True)
    mapped = mapped[np.sort(first)]
    used = np.unique(mapped)
    remap = np.full(len(clustered), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return clustered[used], remap[mapped]


def _visual_link_templates(
    hand: X2HandModel,
    usd_path: Path,
    voxel_size: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise X2DualRenderError(
            "The paper renderer needs pxr/Isaac Sim; run it in the isaaclab environment"
        ) from exc

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise X2DualRenderError(f"Could not open X2 USD stage: {usd_path}")
    path_to_link = {
        str(path): name for name, path in hand.backend.rigid_body_paths.items()
    }
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    templates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputePurpose() != "guide":
            continue
        owner = prim
        while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        if not owner or str(owner.GetPath()) not in path_to_link:
            continue
        link_name = path_to_link[str(owner.GetPath())]
        if link_name in templates:
            raise X2DualRenderError(f"Multiple X2 visual meshes found for {link_name}")
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        faces = _triangles(counts, indices)
        mesh_to_world = cache.GetLocalToWorldTransform(prim)
        world_to_owner = cache.GetLocalToWorldTransform(owner).GetInverse()
        mesh_to_owner = _gf_matrix_to_column(mesh_to_world * world_to_owner)
        vertices = points @ mesh_to_owner[:3, :3].T + mesh_to_owner[:3, 3]
        templates[link_name] = _cluster_mesh(vertices, faces, voxel_size)

    expected = set(hand.backend.link_names)
    if set(templates) != expected:
        raise X2DualRenderError(
            "X2 visual mesh coverage is incomplete: "
            f"missing={sorted(expected - set(templates))}, "
            f"extra={sorted(set(templates) - expected)}"
        )
    return templates


def _materialize_visual_hand(
    hand: X2HandModel,
    templates: dict[str, tuple[np.ndarray, np.ndarray]],
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
        vertices, faces = templates[link_name]
        transform = hand.current_status[link_name][0].detach().cpu().numpy()
        root_points = vertices @ transform[:3, :3].T + transform[:3, 3]
        world_points = root_points @ global_rotation.T + global_translation
        result.append((link_name, world_points, faces))
    return result


def _geometry(
    hand: X2HandModel,
    templates: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": _sha256(path),
        "payload": payload,
        "links": _materialize_visual_hand(hand, templates, payload),
        "objects": [_materialize_object(value) for value in payload["objects"]],
    }


def _shaded_facecolors(
    vertices: np.ndarray,
    faces: np.ndarray,
    color: str,
) -> np.ndarray:
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1.0e-12)
    light = np.asarray((0.25, -0.45, 0.86), dtype=np.float64)
    light /= np.linalg.norm(light)
    # The source meshes contain a few inconsistently wound triangles.  Using
    # two-sided lighting avoids salt-and-pepper artifacts while retaining the
    # softly faceted look of the figures in the DexGraspNet paper.
    diffuse = np.abs(normals @ light)
    # The reference qualitative figure uses a strongly directional studio
    # light: broad surfaces remain readable, while creases and opposing faces
    # are substantially darker.  A 62/38 ambient-to-diffuse split reproduces
    # that foreground luminance range without introducing hard black facets.
    intensity = 0.62 + 0.38 * diffuse
    base = np.asarray(mcolors.to_rgb(color), dtype=np.float64)
    rgb = np.clip(base[None, :] * intensity[:, None], 0.0, 1.0)
    return np.column_stack((rgb, np.ones(len(rgb), dtype=np.float64)))


def _add_mesh(
    axis: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: str,
) -> None:
    axis.add_collection3d(
        Poly3DCollection(
            vertices[faces],
            facecolors=_shaded_facecolors(vertices, faces, color),
            edgecolors="none",
            linewidths=0.0,
            antialiaseds=False,
            rasterized=True,
            zsort="average",
        )
    )


def _draw_scene(
    axis: Any,
    geometry: dict[str, Any],
    elevation: float,
    azimuth: float,
) -> None:
    payload = geometry["payload"]
    owners = payload["hand"]["finger_mode_owner"]
    focus_points: list[np.ndarray] = []
    for index, object_geometry in enumerate(geometry["objects"]):
        vertices = object_geometry["vertices"]
        _add_mesh(axis, vertices, object_geometry["faces"], OBJECT_COLORS[index])
        focus_points.append(vertices)
    for link_name, vertices, faces in geometry["links"]:
        finger = _finger_for_link(link_name)
        owner = owners.get(finger) if finger is not None else None
        color = FINGER_COLORS[owner] if owner in {"right", "left"} else FINGER_COLORS["palm"]
        _add_mesh(axis, vertices, faces, color)
        if finger is not None:
            focus_points.append(vertices)

    points = np.concatenate(focus_points, axis=0)
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.54 * float((upper - lower).max()), 0.055)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_proj_type("ortho")
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()
    axis.set_facecolor("white")


def _save_individual(
    geometry: dict[str, Any],
    output: Path,
    elevation: float,
    azimuth: float,
    dpi: int,
) -> None:
    figure = plt.figure(figsize=(5.0, 5.0), facecolor="white")
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0), projection="3d")
    _draw_scene(axis, geometry, elevation, azimuth)
    figure.savefig(
        output,
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.0,
        metadata={
            "CandidateSHA256": geometry["sha256"],
            "CandidateJSON": _portable(geometry["path"]),
            "DualObjectPhysXValidated": "true",
            "RenderedState": "physx_validated_input_qpose",
            "Style": "dexgraspnet_paper",
        },
    )
    plt.close(figure)


def _save_overview(
    geometries: Sequence[dict[str, Any]],
    output: Path,
    elevation: float,
    azimuth: float,
    dpi: int,
) -> None:
    width = 4.4 * len(geometries)
    figure = plt.figure(figsize=(width, 4.4), facecolor="white")
    for index, geometry in enumerate(geometries):
        axis = figure.add_axes(
            (index / len(geometries), 0.0, 1.0 / len(geometries), 1.0),
            projection="3d",
        )
        _draw_scene(axis, geometry, elevation, azimuth)
    figure.savefig(
        output,
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.0,
        metadata={
            "DualObjectPhysXValidated": "true",
            "RenderedState": "physx_validated_input_qpose",
            "Style": "dexgraspnet_paper",
        },
    )
    plt.close(figure)


def _save_dexgraspnet_grid(
    image_paths: Sequence[Path], output: Path, *, rows: int = DEXGRASPNET_GRID_ROWS
) -> None:
    """Stitch one or three six-image rows in the qualitative-results layout."""

    try:
        from PIL import Image, ImageEnhance, ImageFilter
    except ImportError as exc:
        raise X2DualRenderError(
            "The DexGraspNet grid layout requires Pillow in the render environment"
        ) from exc

    if rows not in {1, DEXGRASPNET_GRID_ROWS}:
        raise X2DualRenderError(f"DexGraspNet layout does not support {rows} rows")
    expected = DEXGRASPNET_GRID_COLUMNS * rows
    if len(image_paths) != expected:
        raise X2DualRenderError(
            f"DexGraspNet grid requires exactly {expected} images, got {len(image_paths)}"
        )
    width, height = DEXGRASPNET_ROW_SIZE if rows == 1 else DEXGRASPNET_GRID_SIZE
    canvas = Image.new("RGB", (width, height), "white")
    for index, path in enumerate(image_paths):
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            pixels = np.asarray(rgb)
            foreground = np.any(pixels < 248, axis=2)
            foreground_rows, columns = np.nonzero(foreground)
            if not len(columns):
                raise X2DualRenderError(f"Rendered image is blank: {path}")
            padding = max(4, round(0.018 * max(rgb.size)))
            left = max(0, int(columns.min()) - padding)
            top = max(0, int(foreground_rows.min()) - padding)
            right = min(rgb.width, int(columns.max()) + 1 + padding)
            bottom = min(rgb.height, int(foreground_rows.max()) + 1 + padding)
            cropped = rgb.crop((left, top, right, bottom))

        row, column = divmod(index, DEXGRASPNET_GRID_COLUMNS)
        x0 = round(column * width / DEXGRASPNET_GRID_COLUMNS)
        x1 = round((column + 1) * width / DEXGRASPNET_GRID_COLUMNS)
        y0 = round(row * height / rows)
        y1 = round((row + 1) * height / rows)
        cell_width, cell_height = x1 - x0, y1 - y0
        maximum = (round(0.91 * cell_width), round(0.87 * cell_height))
        scale = min(maximum[0] / cropped.width, maximum[1] / cropped.height)
        resized = cropped.resize(
            (
                max(1, round(cropped.width * scale)),
                max(1, round(cropped.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        paste_x = x0 + (cell_width - resized.width) // 2
        paste_y = y0 + (cell_height - resized.height) // 2
        canvas.paste(resized, (paste_x, paste_y))
    # Match the reference figure's foreground statistics after the final
    # downsample: its saturated hand colors and studio-lit objects are more
    # separated than an unprocessed Matplotlib raster.  The mild sub-pixel
    # unsharp mask restores edge definition lost during 1200 px -> cell-size
    # reduction without producing the bright halos of a large-radius filter.
    canvas = ImageEnhance.Contrast(canvas).enhance(DEXGRASPNET_CONTRAST_FACTOR)
    canvas = ImageEnhance.Color(canvas).enhance(DEXGRASPNET_COLOR_FACTOR)
    canvas = canvas.filter(
        ImageFilter.UnsharpMask(
            radius=DEXGRASPNET_UNSHARP_RADIUS_PX,
            percent=DEXGRASPNET_UNSHARP_PERCENT,
            threshold=DEXGRASPNET_UNSHARP_THRESHOLD,
        )
    )
    canvas.save(output, format="PNG", optimize=True)


def render(
    input_root: Path,
    output_dir: Path,
    *,
    requested: Sequence[str] = (),
    view: str = "side",
    voxel_size: float = 0.001,
    dpi: int = 240,
    layout: str = "strip",
    replacements: dict[int, str] | None = None,
) -> dict[str, Any]:
    if view not in CAMERA_VIEWS:
        raise X2DualRenderError(f"Unknown view {view!r}; choose from {sorted(CAMERA_VIEWS)}")
    if layout not in {"strip", "dexgraspnet", "dexgraspnet_row"}:
        raise X2DualRenderError(f"Unknown layout {layout!r}")
    default_count = {"strip": 4, "dexgraspnet": 18, "dexgraspnet_row": 6}
    target_count = len(requested) if requested else default_count[layout]
    if layout == "dexgraspnet" and target_count != 18:
        raise X2DualRenderError(
            "DexGraspNet layout requires exactly 18 explicit candidates or no explicit set"
        )
    if layout == "dexgraspnet_row" and target_count != 6:
        raise X2DualRenderError(
            "DexGraspNet row layout requires exactly 6 explicit candidates or no explicit set"
        )
    selected = _representative_records(input_root, requested, target_count)
    replacements = replacements or {}
    if replacements and layout != "dexgraspnet":
        raise X2DualRenderError("--replace-cell requires --layout dexgraspnet")
    selected = _apply_grid_replacements(input_root, selected, replacements)
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
    templates = _visual_link_templates(
        hand,
        config.configured_path("robot.usd_path", must_exist=True),
        voxel_size,
    )
    geometries = [
        _geometry(hand, templates, path, payload) for path, payload in selected
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    elevation, azimuth = CAMERA_VIEWS[view]
    samples: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    for geometry in geometries:
        payload = geometry["payload"]
        candidate_id = payload["candidate_id"]
        image_path = output_dir / f"{candidate_id}_{view}.png"
        _save_individual(geometry, image_path, elevation, azimuth, dpi)
        image_paths.append(image_path)
        samples.append(
            {
                "candidate_id": candidate_id,
                "candidate_json": _portable(geometry["path"]),
                "candidate_sha256": geometry["sha256"],
                "objects": [
                    {
                        "slot": value["slot"],
                        "object_id": value["object_id"],
                        "finger_names": value["finger_names"],
                    }
                    for value in payload["objects"]
                ],
                "image": image_path.name,
                "image_sha256": _sha256(image_path),
                "dual_object_validation": "passed_6_of_6",
            }
        )
    if layout == "dexgraspnet":
        overview = output_dir / "dual_object_qualitative_results.png"
        _save_dexgraspnet_grid(image_paths, overview)
    elif layout == "dexgraspnet_row":
        overview = output_dir / "dual_object_qualitative_row.png"
        _save_dexgraspnet_grid(image_paths, overview, rows=1)
    else:
        overview = output_dir / f"dual_object_paper_overview_{view}.png"
        _save_overview(geometries, overview, elevation, azimuth, dpi)
    selected_index = output_dir / "selected_physx_valid.jsonl"
    selected_index.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": 1,
                    "candidate_id": geometry["payload"]["candidate_id"],
                    "record": _portable(geometry["path"]),
                    "record_sha256": geometry["sha256"],
                    "validation": {
                        "status": geometry["payload"]["dual_object_validation"]["status"],
                        "backend": geometry["payload"]["dual_object_validation"]["backend"],
                        "protocol_revision": geometry["payload"]["dual_object_validation"][
                            "protocol_revision"
                        ],
                        "passed_orientation_count": geometry["payload"][
                            "dual_object_validation"
                        ]["passed_orientation_count"],
                        "required_orientation_count": geometry["payload"][
                            "dual_object_validation"
                        ]["required_orientation_count"],
                        "all_orientations_finite": all(
                            value["finite"] is True
                            for value in geometry["payload"]["dual_object_validation"][
                                "orientations"
                            ]
                        ),
                        "maximum_static_penetration_m": max(
                            geometry["payload"]["dual_object_validation"][
                                "static_preflight"
                            ]["hand_self_collision"]["maximum_penetration_m"],
                            geometry["payload"]["dual_object_validation"][
                                "static_preflight"
                            ]["object_object"]["maximum_penetration_m"],
                            *[
                                value["maximum_penetration_m"]
                                for value in geometry["payload"][
                                    "dual_object_validation"
                                ]["static_preflight"]["hand_object"]["objects"]
                            ],
                        ),
                    },
                    "right_finger_count": len(
                        geometry["payload"]["objects"][0]["finger_names"]
                    ),
                    "objects": [
                        {
                            "slot": value["slot"],
                            "object_id": value["object_id"],
                            "mesh_path": _portable(Path(value["mesh_path"])),
                            "finger_names": value["finger_names"],
                        }
                        for value in geometry["payload"]["objects"]
                    ],
                },
                allow_nan=False,
            )
            + "\n"
            for geometry in geometries
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "renderer": "x2_dual_object_visual_mesh_matplotlib_paper_v1",
        "source_root": _portable(input_root),
        "selection": (
            "explicit"
            if requested
            else (
                "balanced_diverse_18"
                if layout == "dexgraspnet"
                else (
                    "balanced_diverse_6"
                    if layout == "dexgraspnet_row"
                    else "one_representative_per_finger_split"
                )
            )
        ),
        "layout": layout,
        "rendered_state": "physx_validated_input_qpose",
        "dual_object_physx_validated": True,
        "required_passed_orientations": 6,
        "maximum_allowed_static_penetration_m": MAXIMUM_RENDER_PENETRATION_M,
        "view": view,
        "elevation_deg": elevation,
        "azimuth_deg": azimuth,
        "visual_mesh_voxel_size_m": voxel_size,
        "dpi": dpi,
        "overview_image": overview.name,
        "overview_sha256": _sha256(overview),
        "selected_dataset_index": selected_index.name,
        "selected_dataset_index_sha256": _sha256(selected_index),
        "samples": samples,
    }
    if layout in {"dexgraspnet", "dexgraspnet_row"}:
        reference = PROJECT_ROOT / "images" / "qualitative_results.png"
        grid_rows = 1 if layout == "dexgraspnet_row" else DEXGRASPNET_GRID_ROWS
        grid_size = (
            DEXGRASPNET_ROW_SIZE
            if layout == "dexgraspnet_row"
            else DEXGRASPNET_GRID_SIZE
        )
        manifest["grid"] = {
            "reference_image": _portable(reference),
            "reference_sha256": _sha256(reference),
            "canvas_width_px": grid_size[0],
            "canvas_height_px": grid_size[1],
            "columns": DEXGRASPNET_GRID_COLUMNS,
            "rows": grid_rows,
            "column_boundaries_px": [
                round(index * grid_size[0] / DEXGRASPNET_GRID_COLUMNS)
                for index in range(DEXGRASPNET_GRID_COLUMNS + 1)
            ],
            "row_boundaries_px": [
                round(index * grid_size[1] / grid_rows)
                for index in range(grid_rows + 1)
            ],
            "background": "#ffffff",
            "titles": False,
            "borders": False,
            "reference_matched_postprocess": {
                "contrast_factor": DEXGRASPNET_CONTRAST_FACTOR,
                "color_factor": DEXGRASPNET_COLOR_FACTOR,
                "unsharp_radius_px": DEXGRASPNET_UNSHARP_RADIUS_PX,
                "unsharp_percent": DEXGRASPNET_UNSHARP_PERCENT,
                "unsharp_threshold": DEXGRASPNET_UNSHARP_THRESHOLD,
            },
            "replacements": [
                {
                    "row": index // DEXGRASPNET_GRID_COLUMNS + 1,
                    "column": index % DEXGRASPNET_GRID_COLUMNS + 1,
                    "candidate_id": candidate_id,
                }
                for index, candidate_id in sorted(replacements.items())
            ],
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
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="candidate id or JSON path; repeat to render an explicit set",
    )
    parser.add_argument("--view", choices=sorted(CAMERA_VIEWS), default="side")
    parser.add_argument(
        "--visual-voxel-mm",
        type=float,
        default=1.0,
        help="deterministic visual-mesh simplification grid in millimetres",
    )
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--layout",
        choices=("strip", "dexgraspnet", "dexgraspnet_row"),
        default="strip",
        help=(
            "use dexgraspnet for the exact 6x3, 2288x1232 layout, or "
            "dexgraspnet_row for one 1x6, 2288x411 row"
        ),
    )
    parser.add_argument(
        "--replace-cell",
        action="append",
        default=[],
        metavar="ROW,COLUMN=CANDIDATE_ID",
        help="replace one 1-based DexGraspNet grid cell; may be repeated",
    )
    return parser


def _parse_replacements(values: Sequence[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    for value in values:
        try:
            position, candidate_id = value.split("=", 1)
            row_text, column_text = position.split(",", 1)
            row, column = int(row_text), int(column_text)
        except ValueError as exc:
            raise X2DualRenderError(
                f"Invalid --replace-cell {value!r}; use ROW,COLUMN=CANDIDATE_ID"
            ) from exc
        if not (
            1 <= row <= DEXGRASPNET_GRID_ROWS
            and 1 <= column <= DEXGRASPNET_GRID_COLUMNS
            and candidate_id
        ):
            raise X2DualRenderError(f"Invalid --replace-cell value: {value!r}")
        index = (row - 1) * DEXGRASPNET_GRID_COLUMNS + column - 1
        if index in result:
            raise X2DualRenderError(f"Grid cell {row},{column} was replaced more than once")
        result[index] = candidate_id
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.visual_voxel_mm <= 0.0:
        raise X2DualRenderError("--visual-voxel-mm must be positive")
    if args.dpi < 72:
        raise X2DualRenderError("--dpi must be at least 72")
    manifest = render(
        args.input_root.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        requested=args.candidate,
        view=args.view,
        voxel_size=args.visual_voxel_mm / 1000.0,
        dpi=args.dpi,
        layout=args.layout,
        replacements=_parse_replacements(args.replace_cell),
    )
    print(json.dumps(manifest, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
