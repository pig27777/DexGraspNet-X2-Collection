#!/usr/bin/env python3
"""Select a deterministic, diverse, audited subset of general-object meshes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "meshdata"
DEFAULT_TARGET_COUNT = 88
DEFAULT_OBJECT_SCALE = 1.0
MIN_SCALED_EXTENT_M = 0.003
MAX_SCALED_EXTENT_M = 0.35
MIN_SCALED_VOLUME_M3 = 1.0e-8
ARCHIVE_DIRECTORY_NAME = "_excluded_general_meshes"


class MeshSelectionError(RuntimeError):
    """Raised when the official assets cannot produce the requested subset."""


@dataclass(frozen=True)
class AuditedMesh:
    object_id: str
    category: str
    path: Path
    sha256: str
    vertices: int
    faces: int
    components: int
    extents: tuple[float, float, float]
    volume: float
    object_scale: float

    @property
    def scaled_extents(self) -> tuple[float, float, float]:
        return tuple(value * self.object_scale for value in self.extents)

    @property
    def scaled_volume(self) -> float:
        return self.volume * self.object_scale**3

    def record(self, destination: Path, *, preexisting: bool) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "category": self.category,
            "source": str(self.path),
            "destination": str(destination.resolve()),
            "sha256": self.sha256,
            "vertices": self.vertices,
            "faces": self.faces,
            "components": self.components,
            "unscaled_extents": list(self.extents),
            "object_scale": self.object_scale,
            "scaled_extents_m": list(self.scaled_extents),
            "volume": self.volume,
            "unscaled_volume": self.volume,
            "scaled_volume_m3": self.scaled_volume,
            "preexisting": preexisting,
        }


def _category(object_id: str) -> str:
    parts = object_id.split("-")
    if len(parts) >= 3 and parts[0] in {"sem", "core"}:
        return f"{parts[0]}-{parts[1]}"
    if len(parts) >= 2:
        return parts[0]
    return "unknown"


def audit_mesh(
    path: Path, *, object_scale: float = DEFAULT_OBJECT_SCALE
) -> AuditedMesh:
    path = path.expanduser().resolve()
    object_id = path.parent.parent.name
    if not math.isfinite(object_scale) or object_scale <= 0.0:
        raise MeshSelectionError(f"object_scale must be finite and positive: {object_scale}")
    if object_id == ARCHIVE_DIRECTORY_NAME:
        raise MeshSelectionError(f"{path}: reserved object ID {object_id!r}")
    try:
        loaded = trimesh.load(path, force="mesh", process=False)
    except Exception as exc:
        raise MeshSelectionError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(loaded, trimesh.Trimesh):
        raise MeshSelectionError(f"{path}: not a triangle mesh")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or len(vertices) < 4
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or len(faces) < 4
        or not np.isfinite(vertices).all()
    ):
        raise MeshSelectionError(f"{path}: invalid finite triangle arrays")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise MeshSelectionError(f"{path}: triangle indices are out of range")
    extents = tuple(float(value) for value in loaded.extents)
    scaled_extents = tuple(value * object_scale for value in extents)
    if (
        not all(math.isfinite(value) for value in scaled_extents)
        or any(value < MIN_SCALED_EXTENT_M for value in scaled_extents)
        or max(scaled_extents) > MAX_SCALED_EXTENT_M
    ):
        raise MeshSelectionError(
            f"{path}: scaled extents {scaled_extents} m are outside "
            f"[{MIN_SCALED_EXTENT_M}, {MAX_SCALED_EXTENT_M}] m at "
            f"object_scale={object_scale}"
        )
    try:
        components = tuple(loaded.split(only_watertight=False))
    except Exception as exc:
        raise MeshSelectionError(f"{path}: cannot split convex components: {exc}") from exc
    if not components or not all(component.is_watertight for component in components):
        raise MeshSelectionError(f"{path}: convex components are not watertight")
    volume = float(sum(abs(component.volume) for component in components))
    scaled_volume = volume * object_scale**3
    if not math.isfinite(scaled_volume) or scaled_volume <= MIN_SCALED_VOLUME_M3:
        raise MeshSelectionError(
            f"{path}: scaled volume {scaled_volume} m^3 is not greater than "
            f"{MIN_SCALED_VOLUME_M3} m^3"
        )
    return AuditedMesh(
        object_id=object_id,
        category=_category(object_id),
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        vertices=len(vertices),
        faces=len(faces),
        components=len(components),
        extents=extents,
        volume=volume,
        object_scale=object_scale,
    )


def discover_and_audit(
    source_root: Path,
    *,
    object_scale: float = DEFAULT_OBJECT_SCALE,
    excluded_root: Path | None = None,
) -> tuple[list[AuditedMesh], list[dict[str, str]]]:
    candidates = sorted(source_root.expanduser().resolve().glob("**/coacd/decomposed.obj"))
    if excluded_root is not None:
        excluded_root = excluded_root.expanduser().resolve()
        candidates = [
            path for path in candidates if not path.resolve().is_relative_to(excluded_root)
        ]
    accepted: list[AuditedMesh] = []
    rejected: list[dict[str, str]] = []
    paths_by_normalized_id: dict[str, list[Path]] = defaultdict(list)
    for path in candidates:
        paths_by_normalized_id[path.parent.parent.name.casefold()].append(path)
    conflicting_paths = {
        path.resolve()
        for paths in paths_by_normalized_id.values()
        if len(paths) > 1
        for path in paths
    }
    for path in candidates:
        if path.resolve() in conflicting_paths:
            rejected.append(
                {
                    "path": str(path),
                    "reason": "duplicate or case-conflicting object ID",
                }
            )
            continue
        try:
            audited = audit_mesh(path, object_scale=object_scale)
        except MeshSelectionError as exc:
            rejected.append({"path": str(path), "reason": str(exc)})
            continue
        accepted.append(audited)
    return accepted, rejected


def diverse_selection(candidates: Sequence[AuditedMesh], count: int) -> list[AuditedMesh]:
    by_category: dict[str, list[AuditedMesh]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda value: value.object_id):
        by_category[candidate.category].append(candidate)
    selected: list[AuditedMesh] = []
    categories = sorted(by_category)
    depth = 0
    while len(selected) < count:
        added = False
        for category in categories:
            values = by_category[category]
            if depth < len(values):
                selected.append(values[depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != count:
        raise MeshSelectionError(
            f"Only {len(selected)} audited meshes available; need {count}"
        )
    return selected


def _assert_case_unique(object_ids: Sequence[str], *, label: str) -> None:
    normalized: dict[str, str] = {}
    for object_id in object_ids:
        previous = normalized.setdefault(object_id.casefold(), object_id)
        if previous != object_id:
            raise MeshSelectionError(
                f"{label} contains case-conflicting IDs: {previous!r} and {object_id!r}"
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    destination_root = args.destination.expanduser().resolve()
    object_scale = float(getattr(args, "object_scale", DEFAULT_OBJECT_SCALE))
    replace_existing = bool(getattr(args, "replace_existing", False))
    if not math.isfinite(object_scale) or object_scale <= 0.0:
        raise MeshSelectionError(
            f"object_scale must be finite and positive: {object_scale}"
        )
    destination_root.mkdir(parents=True, exist_ok=True)
    existing_paths = sorted(destination_root.glob("*/coacd/decomposed.obj"))
    existing_path_by_id = {path.parent.parent.name: path for path in existing_paths}
    if len(existing_path_by_id) != len(existing_paths):
        raise MeshSelectionError("Destination contains duplicate direct object IDs")
    _assert_case_unique(list(existing_path_by_id), label="destination")

    audited, rejected = discover_and_audit(
        args.source_root,
        object_scale=object_scale,
        # In replacement mode the source selection must be independent of every
        # old destination object, even if source-root happens to contain it.
        excluded_root=destination_root if replace_existing else None,
    )

    archived_ids: list[str] = []
    archived_unselected_ids: list[str] = []
    archived_conflict_ids: list[str] = []
    retained_ids: set[str] = set()
    initial_existing_count = len(existing_paths)

    if replace_existing:
        # Complete the strict source audit and selection before moving anything.
        selected = sorted(
            diverse_selection(audited, args.target_count),
            key=lambda value: value.object_id,
        )
        selected_by_id = {value.object_id: value for value in selected}
        _assert_case_unique(list(selected_by_id), label="selected source")

        existing_case_map = {value.casefold(): value for value in existing_path_by_id}
        for selected_id in selected_by_id:
            existing_spelling = existing_case_map.get(selected_id.casefold())
            if existing_spelling is not None and existing_spelling != selected_id:
                raise MeshSelectionError(
                    "Selected source ID conflicts by case with destination ID: "
                    f"{selected_id!r} versus {existing_spelling!r}"
                )

        for object_id, path in existing_path_by_id.items():
            source = selected_by_id.get(object_id)
            if source is None:
                archived_unselected_ids.append(object_id)
            elif hashlib.sha256(path.read_bytes()).hexdigest() == source.sha256:
                retained_ids.add(object_id)
            else:
                archived_conflict_ids.append(object_id)
        archived_ids = sorted([*archived_unselected_ids, *archived_conflict_ids])
        archived_unselected_ids.sort()
        archived_conflict_ids.sort()

        # A selected target must either be absent, be an identical retained
        # direct mesh, or be one of the old directories that will be archived.
        for value in selected:
            target_directory = destination_root / value.object_id
            if (
                target_directory.exists()
                and value.object_id not in existing_path_by_id
            ):
                raise MeshSelectionError(
                    f"Selected target directory already exists without a direct mesh: "
                    f"{target_directory}"
                )

        archive_root = destination_root / ARCHIVE_DIRECTORY_NAME
        if archived_ids and archive_root.exists() and not archive_root.is_dir():
            raise MeshSelectionError(f"Archive path is not a directory: {archive_root}")
        for object_id in archived_ids:
            archive_target = archive_root / object_id
            if archive_target.exists():
                raise MeshSelectionError(
                    f"Refusing to overwrite archived object directory: {archive_target}"
                )

        if archived_ids:
            archive_root.mkdir(parents=True, exist_ok=True)
        for object_id in archived_ids:
            source_directory = existing_path_by_id[object_id].parent.parent
            archive_target = archive_root / object_id
            shutil.move(str(source_directory), str(archive_target))
        selected_new = [
            value for value in selected if value.object_id not in retained_ids
        ]
    else:
        existing = [
            audit_mesh(path, object_scale=object_scale) for path in existing_paths
        ]
        existing_ids = {value.object_id for value in existing}
        if len(existing) > args.target_count:
            raise MeshSelectionError(
                f"Destination already contains {len(existing)} meshes; "
                f"target is {args.target_count}"
            )
        audited_case_map = {value.object_id.casefold(): value.object_id for value in audited}
        for object_id in existing_ids:
            source_spelling = audited_case_map.get(object_id.casefold())
            if source_spelling is not None and source_spelling != object_id:
                raise MeshSelectionError(
                    "Source ID conflicts by case with destination ID: "
                    f"{source_spelling!r} versus {object_id!r}"
                )
        available = [value for value in audited if value.object_id not in existing_ids]
        selected_new = diverse_selection(available, args.target_count - len(existing))
        for value in selected_new:
            target_directory = destination_root / value.object_id
            if target_directory.exists():
                raise MeshSelectionError(
                    f"Selected target directory already exists: {target_directory}"
                )
        selected = sorted([*existing, *selected_new], key=lambda value: value.object_id)
        retained_ids = existing_ids

    records = []
    for value in selected:
        target = destination_root / value.object_id / "coacd" / "decomposed.obj"
        preexisting = value.object_id in retained_ids
        if not preexisting:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(value.path, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != value.sha256:
            raise MeshSelectionError(f"Copied mesh hash mismatch: {target}")
        records.append(value.record(target, preexisting=preexisting))

    final_paths = sorted(destination_root.glob("*/coacd/decomposed.obj"))
    if len(final_paths) != args.target_count:
        raise MeshSelectionError(
            f"Final general mesh count is {len(final_paths)}, expected {args.target_count}"
        )
    final_ids = {path.parent.parent.name for path in final_paths}
    selected_ids = {value.object_id for value in selected}
    if final_ids != selected_ids:
        raise MeshSelectionError(
            "Final direct object IDs do not exactly match the audited selection: "
            f"missing={sorted(selected_ids - final_ids)}, "
            f"unexpected={sorted(final_ids - selected_ids)}"
        )
    category_counts: dict[str, int] = dict(
        sorted(
            (category, sum(value.category == category for value in selected))
            for category in {value.category for value in selected}
        )
    )
    report = {
        "passed": True,
        "source_root": str(args.source_root.expanduser().resolve()),
        "destination": str(destination_root),
        "target_count": args.target_count,
        "object_scale": object_scale,
        "replace_existing": replace_existing,
        "selected_count": len(records),
        "initial_existing_count": initial_existing_count,
        "preexisting_count": len(retained_ids),
        "new_count": len(selected_new),
        "archived_count": len(archived_ids),
        "archived_ids": archived_ids,
        "archived_unselected_ids": archived_unselected_ids,
        "archived_conflict_ids": archived_conflict_ids,
        "category_count": len(category_counts),
        "category_counts": category_counts,
        "audited_source_count": len(audited),
        "rejected_source_count": len(rejected),
        "license": "CC BY-NC 4.0 (DexGraspNet 2.0 assets)",
        "source_url": "https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0",
        "meshes": records,
        "rejected": rejected,
    }
    manifest = args.manifest or destination_root / "x2_general_mesh_manifest.json"
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest)
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument(
        "--object-scale",
        type=float,
        default=DEFAULT_OBJECT_SCALE,
        help="Scale applied to mesh vertices when auditing physical dimensions.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help=(
            "Select exclusively from source-root and archive old direct destination "
            "objects that are not identical selected assets."
        ),
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if args.target_count <= 0:
        parser.error("target-count must be positive")
    if not math.isfinite(args.object_scale) or args.object_scale <= 0.0:
        parser.error("object-scale must be finite and positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:
        print(
            json.dumps(
                {"passed": False, "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"meshes", "rejected"}},
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
