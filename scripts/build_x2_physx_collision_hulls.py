#!/usr/bin/env python3
"""Build explicit low-vertex X2 convex colliders shared with PhysX and PyTorch.

The imported X2 visual meshes contain tens of thousands of points.  Declaring
those meshes as ``convexHull`` leaves PhysX free to cook a much smaller hull,
while the analytic generator previously reconstructed the unsimplified SciPy
hull.  Contact candidates could therefore be on a surface that did not exist
in simulation.  This script creates one deterministic, at-most-64-vertex mesh
directly below each rigid body, disables the original visual-mesh collider, and
records the construction provenance in a small USD overlay.

Run with the same ``isaaclab`` conda environment used by generation.  The
output is referenced by ``x2_keypoints.usda`` and can be rebuilt idempotently.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PROXY_NAME = "x2_physx_collision_hull"


class CollisionHullBuildError(RuntimeError):
    """Raised when the composed X2 collision source is incomplete."""


def _portable_manifest_path(path: Path) -> str:
    """Prefer repository-relative provenance while preserving custom paths."""

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _previous_instance_roots(manifest_path: Path) -> dict[str, str]:
    """Retain de-instancing opinions across idempotent overlay rebuilds.

    Once the overlay is composed, the formerly instanced render meshes appear
    as ordinary prims, so OpenUSD can no longer rediscover their instance-root
    paths.  The prior manifest is therefore part of the deterministic build
    state, not merely a report.
    """

    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    links = payload.get("links")
    if not isinstance(links, dict):
        return {}
    result: dict[str, str] = {}
    for link_name, record in links.items():
        if not isinstance(link_name, str) or not isinstance(record, dict):
            continue
        path = record.get("instance_root_path")
        if isinstance(path, str) and path.startswith("/"):
            result[link_name] = path
    return result


def _gf_matrix_to_column(matrix: Any) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64).T.copy()


def _owner_rigid_body(prim, UsdPhysics):
    owner = prim
    while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
        owner = owner.GetParent()
    return owner if owner and owner.IsValid() else None


def _initial_support_indices(vertices: np.ndarray) -> list[int]:
    directions: list[np.ndarray] = []
    for x in (-1.0, 0.0, 1.0):
        for y in (-1.0, 0.0, 1.0):
            for z in (-1.0, 0.0, 1.0):
                vector = np.asarray((x, y, z), dtype=np.float64)
                norm = float(np.linalg.norm(vector))
                if norm > 0.0:
                    directions.append(vector / norm)
    selected: list[int] = []
    seen: set[int] = set()
    for direction in directions:
        index = int(np.argmax(vertices @ direction))
        if index not in seen:
            selected.append(index)
            seen.add(index)
    return selected


def _simplify_convex_hull(
    vertices: np.ndarray,
    *,
    max_vertices: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from scipy.spatial import ConvexHull, QhullError

    try:
        source_hull = ConvexHull(vertices)
    except QhullError as exc:
        raise CollisionHullBuildError(f"source convex hull failed: {exc}") from exc
    source_indices = np.unique(np.asarray(source_hull.simplices, dtype=np.int64))
    source_vertices = np.ascontiguousarray(vertices[source_indices], dtype=np.float64)
    selected = _initial_support_indices(source_vertices)
    if len(selected) < 4:
        raise CollisionHullBuildError("source hull produced fewer than four support points")

    # Iteratively add the source point with the largest supporting-plane
    # violation.  This directly reduces the one-sided shrink introduced by an
    # inscribed low-vertex hull and is deterministic under stable NumPy argmax.
    while len(selected) < max_vertices:
        try:
            candidate_hull = ConvexHull(source_vertices[selected])
        except QhullError as exc:
            raise CollisionHullBuildError(f"simplified convex hull failed: {exc}") from exc
        normals = np.asarray(candidate_hull.equations[:, :3], dtype=np.float64)
        offsets = np.asarray(candidate_hull.equations[:, 3], dtype=np.float64)
        violation = np.max(source_vertices @ normals.T + offsets[None, :], axis=1)
        violation[np.asarray(selected, dtype=np.int64)] = -math.inf
        index = int(np.argmax(violation))
        if not math.isfinite(float(violation[index])) or float(violation[index]) <= 1.0e-12:
            break
        selected.append(index)

    selected_vertices = source_vertices[np.asarray(selected, dtype=np.int64)]
    final_hull = ConvexHull(selected_vertices)
    used = np.unique(np.asarray(final_hull.simplices, dtype=np.int64))
    remap = np.full(len(selected_vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    triangles = remap[np.asarray(final_hull.simplices, dtype=np.int64)]
    result_vertices = np.ascontiguousarray(selected_vertices[used], dtype=np.float64)

    triangle_vertices = result_vertices[triangles]
    triangle_normals = np.cross(
        triangle_vertices[:, 1] - triangle_vertices[:, 0],
        triangle_vertices[:, 2] - triangle_vertices[:, 0],
    )
    outward = np.asarray(final_hull.equations[:, :3], dtype=np.float64)
    inward = np.einsum("ij,ij->i", triangle_normals, outward) < 0.0
    triangles[inward, 1], triangles[inward, 2] = (
        triangles[inward, 2].copy(),
        triangles[inward, 1].copy(),
    )

    final_normals = np.asarray(final_hull.equations[:, :3], dtype=np.float64)
    final_offsets = np.asarray(final_hull.equations[:, 3], dtype=np.float64)
    support_violation = np.max(
        source_vertices @ final_normals.T + final_offsets[None, :], axis=1
    )
    maximum_support_shrink = max(0.0, float(support_violation.max(initial=0.0)))
    return result_vertices, triangles, {
        "source_authored_vertex_count": int(len(vertices)),
        "source_convex_hull_vertex_count": int(len(source_vertices)),
        "proxy_vertex_count": int(len(result_vertices)),
        "proxy_triangle_count": int(len(triangles)),
        "maximum_source_support_plane_violation": maximum_support_shrink,
    }


def _collect_sources(source_path: Path) -> list[dict[str, Any]]:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(source_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise CollisionHullBuildError(f"OpenUSD could not open {source_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    sources: list[dict[str, Any]] = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if (
            not prim.IsA(UsdGeom.Mesh)
            or not prim.HasAPI(UsdPhysics.CollisionAPI)
            or prim.GetName() == PROXY_NAME
        ):
            continue
        owner = _owner_rigid_body(prim, UsdPhysics)
        if owner is None:
            raise CollisionHullBuildError(
                f"collision mesh has no rigid-body ancestor: {prim.GetPath()}"
            )
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 4:
            raise CollisionHullBuildError(f"invalid points on {prim.GetPath()}")
        mesh_to_world = cache.GetLocalToWorldTransform(prim)
        world_to_owner = cache.GetLocalToWorldTransform(owner).GetInverse()
        mesh_to_owner = _gf_matrix_to_column(mesh_to_world * world_to_owner)
        owner_points = points @ mesh_to_owner[:3, :3].T + mesh_to_owner[:3, 3]
        instance_root = prim
        while instance_root and not instance_root.IsInstance():
            instance_root = instance_root.GetParent()
        sources.append(
            {
                "link_name": str(owner.GetName()),
                "owner_path": str(owner.GetPath()),
                "source_prim_path": str(prim.GetPath()),
                "instance_root_path": (
                    str(instance_root.GetPath())
                    if instance_root and instance_root.IsValid()
                    else None
                ),
                "vertices_owner": owner_points,
            }
        )
    if len(sources) != 17:
        raise CollisionHullBuildError(
            f"expected 17 original X2 collision meshes, found {len(sources)}"
        )
    names = [entry["link_name"] for entry in sources]
    if len(set(names)) != 17:
        raise CollisionHullBuildError(f"collision owners are not unique: {names}")
    return sorted(sources, key=lambda entry: entry["link_name"])


def _author_overlay(
    output_path: Path,
    records: list[dict[str, Any]],
    *,
    max_vertices: int,
) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

    if output_path.exists():
        output_path.unlink()
    stage = Usd.Stage.CreateNew(str(output_path))
    if stage is None:
        raise CollisionHullBuildError(f"could not create {output_path}")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)

    for record in records:
        if record["instance_root_path"] is not None:
            # Opinions cannot target an instance proxy.  De-instance only the
            # referenced visual subtree that owns this collider, preserving the
            # articulation and all rigid-body paths while making the child mesh
            # authorable in this stronger layer.
            stage.OverridePrim(record["instance_root_path"]).SetInstanceable(False)
        source_override = stage.OverridePrim(record["source_prim_path"])
        UsdPhysics.CollisionAPI.Apply(source_override).CreateCollisionEnabledAttr(False)

        proxy_path = f"{record['owner_path']}/{PROXY_NAME}"
        proxy = UsdGeom.Mesh.Define(stage, proxy_path)
        vertices = record["vertices"]
        triangles = record["triangles"]
        proxy.CreatePointsAttr(
            Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32, copy=False))
        )
        proxy.CreateFaceVertexCountsAttr(
            Vt.IntArray([3] * len(triangles))
        )
        proxy.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(triangles.astype(np.int32, copy=False).reshape(-1))
        )
        lower = vertices.min(axis=0)
        upper = vertices.max(axis=0)
        proxy.CreateExtentAttr(
            Vt.Vec3fArray(
                [Gf.Vec3f(*[float(value) for value in lower]),
                 Gf.Vec3f(*[float(value) for value in upper])]
            )
        )
        proxy.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        proxy.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        collision = UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
        collision.CreateCollisionEnabledAttr(True)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(proxy.GetPrim())
        mesh_collision.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        prim = proxy.GetPrim()
        prim.CreateAttribute(
            "x2Collision:role", Sdf.ValueTypeNames.Token, custom=True
        ).Set("explicitLowVertexPhysxHull")
        prim.CreateAttribute(
            "x2Collision:sourcePrimPath", Sdf.ValueTypeNames.String, custom=True
        ).Set(record["source_prim_path"])
        prim.CreateAttribute(
            "x2Collision:maxCookVertices", Sdf.ValueTypeNames.Int, custom=True
        ).Set(int(max_vertices))
        prim.CreateAttribute(
            "x2Collision:sourceConvexHullVertexCount",
            Sdf.ValueTypeNames.Int,
            custom=True,
        ).Set(int(record["audit"]["source_convex_hull_vertex_count"]))
        prim.CreateAttribute(
            "x2Collision:maximumSourceSupportPlaneViolation",
            Sdf.ValueTypeNames.Double,
            custom=True,
        ).Set(float(record["audit"]["maximum_source_support_plane_violation"]))

    stage.GetRootLayer().customLayerData = {
        "generator": "scripts/build_x2_physx_collision_hulls.py",
        "algorithm": "support_extrema_plus_greedy_max_plane_violation",
        "maxVerticesPerHull": int(max_vertices),
        "sourceAsset": "x2_mujoco/x2_keypoints.usda",
    }
    stage.GetRootLayer().Save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "x2_mujoco" / "x2_keypoints.usda",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "x2_mujoco" / "payloads" / "x2_physx_collision_hulls.usda",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--max-vertices", type=int, default=64)
    args = parser.parse_args()
    if args.max_vertices < 16 or args.max_vertices > 64:
        raise ValueError("--max-vertices must be in [16, 64] for deterministic PhysX cooking")
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else output.with_suffix(".json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    previous_instance_roots = _previous_instance_roots(manifest)
    sources = _collect_sources(source)
    for source_record in sources:
        if source_record["instance_root_path"] is None:
            source_record["instance_root_path"] = previous_instance_roots.get(
                source_record["link_name"]
            )
    records: list[dict[str, Any]] = []
    for source_record in sources:
        vertices, triangles, audit = _simplify_convex_hull(
            source_record["vertices_owner"],
            max_vertices=args.max_vertices,
        )
        records.append(
            {
                **source_record,
                "vertices": vertices,
                "triangles": triangles,
                "audit": audit,
            }
        )
    _author_overlay(output, records, max_vertices=args.max_vertices)
    payload = {
        "schema_version": 1,
        "source_asset": _portable_manifest_path(source),
        "output_overlay": _portable_manifest_path(output),
        "max_vertices_per_hull": args.max_vertices,
        "algorithm": "support_extrema_plus_greedy_max_plane_violation",
        "links": {
            record["link_name"]: {
                "owner_path": record["owner_path"],
                "source_prim_path": record["source_prim_path"],
                "instance_root_path": record["instance_root_path"],
                **record["audit"],
            }
            for record in records
        },
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"COLLISION_HULL_BUILD_COMPLETE links={len(records)} "
        f"max_vertices={args.max_vertices}"
    )
    print(f"overlay: {output}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
