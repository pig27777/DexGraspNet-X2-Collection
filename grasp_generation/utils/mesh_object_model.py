"""Thin differentiable mesh adapter for the X2 DexGraspNet generator.

The original repository's optional ``torchsdf``/PyTorch3D extensions are not
available in every supported environment.  This module therefore implements
the same point-to-triangle query directly in vectorized PyTorch.  It never
substitutes nearest-vertex distance and never converts optimization tensors to
NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh


class MeshObjectError(RuntimeError):
    """Raised for invalid mesh assets or query tensors."""


@dataclass(frozen=True)
class MeshQuery:
    signed_distance: torch.Tensor
    closest_points: torch.Tensor
    outward_normals: torch.Tensor


def _spatial_face_order(
    triangles: np.ndarray, maximum_chunk_size: int
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return deterministic KD-style face order and spatial leaf offsets."""

    centroids = triangles.mean(axis=1)
    leaves: list[np.ndarray] = []

    def split(indices: np.ndarray) -> None:
        if len(indices) <= maximum_chunk_size:
            leaves.append(indices)
            return
        values = centroids[indices]
        axis = int(np.argmax(np.ptp(values, axis=0)))
        ordered = indices[
            np.argsort(values[:, axis], kind="stable")
        ]
        middle = len(ordered) // 2
        split(ordered[:middle])
        split(ordered[middle:])

    split(np.arange(len(triangles), dtype=np.int64))
    order = np.concatenate(leaves)
    offsets = [0]
    for leaf in leaves:
        offsets.append(offsets[-1] + len(leaf))
    return order, tuple(offsets)


def _deterministic_surface_samples(
    triangles: np.ndarray, areas: np.ndarray, count: int
) -> np.ndarray:
    """Area-stratified low-discrepancy samples independent of run seed."""

    cumulative = np.cumsum(areas)
    targets = (np.arange(count, dtype=np.float64) + 0.5) * (
        float(cumulative[-1]) / count
    )
    faces = np.searchsorted(cumulative, targets, side="left")
    sequence = np.arange(count, dtype=np.float64) + 0.5
    u = np.mod(sequence * 0.7548776662466927, 1.0)
    v = np.mod(sequence * 0.5698402909980532, 1.0)
    root = np.sqrt(u)
    weights = np.stack(
        (1.0 - root, root * (1.0 - v), root * v), axis=1
    )
    return np.einsum("ni,nij->nj", weights, triangles[faces])


def _segment_closest(points: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    segment = end - start
    denominator = torch.sum(segment * segment, dim=-1).clamp_min(1.0e-24)
    fraction = torch.sum((points - start) * segment, dim=-1) / denominator
    fraction = fraction.clamp(0.0, 1.0)
    return start + fraction[..., None] * segment


def _triangle_chunk_closest(
    points: torch.Tensor, triangles: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closest point/squared distance/face index for one triangle chunk."""

    query = points[:, None, :]
    a = triangles[None, :, 0, :]
    b = triangles[None, :, 1, :]
    c = triangles[None, :, 2, :]
    ab = b - a
    ac = c - a
    normal = torch.cross(ab, ac, dim=-1)
    normal_norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1.0e-24)
    unit_normal = normal / normal_norm

    plane_offset = torch.sum((query - a) * unit_normal, dim=-1, keepdim=True)
    projection = query - plane_offset * unit_normal
    relative = projection - a
    d00 = torch.sum(ab * ab, dim=-1)
    d01 = torch.sum(ab * ac, dim=-1)
    d11 = torch.sum(ac * ac, dim=-1)
    d20 = torch.sum(relative * ab, dim=-1)
    d21 = torch.sum(relative * ac, dim=-1)
    denominator = (d00 * d11 - d01 * d01).clamp_min(1.0e-24)
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    inside = (v >= 0.0) & (w >= 0.0) & (v + w <= 1.0)

    edge_ab = _segment_closest(query, a, b)
    edge_bc = _segment_closest(query, b, c)
    edge_ca = _segment_closest(query, c, a)
    candidates = torch.stack((projection, edge_ab, edge_bc, edge_ca), dim=-2)
    squared_raw = torch.sum((query.unsqueeze(-2) - candidates) ** 2, dim=-1)
    plane_squared = torch.where(
        inside,
        squared_raw[..., 0],
        torch.full_like(squared_raw[..., 0], torch.inf),
    )
    squared = torch.cat((plane_squared[..., None], squared_raw[..., 1:]), dim=-1)
    kind = squared.argmin(dim=-1)
    best_per_face = torch.gather(
        candidates, -2, kind[..., None, None].expand(-1, -1, 1, 3)
    ).squeeze(-2)
    squared_per_face = torch.gather(squared, -1, kind[..., None]).squeeze(-1)
    face = squared_per_face.argmin(dim=-1)
    closest = best_per_face[torch.arange(len(points), device=points.device), face]
    minimum = squared_per_face[torch.arange(len(points), device=points.device), face]
    return closest, minimum, face


class MeshObjectModel:
    """One watertight mesh, loaded once and shared by a grasp batch."""

    def __init__(
        self,
        mesh_path: str | Path,
        *,
        batch_size: int,
        scale: float = 0.1,
        num_surface_samples: int = 256,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
        face_chunk_size: int = 512,
        point_chunk_size: int = 1024,
        audit_surface_samples: int = 8192,
    ) -> None:
        self.mesh_path = Path(mesh_path).expanduser().resolve()
        if not self.mesh_path.is_file():
            raise MeshObjectError(f"Mesh does not exist: {self.mesh_path}")
        if (
            batch_size <= 0
            or num_surface_samples <= 0
            or audit_surface_samples <= 0
        ):
            raise MeshObjectError(
                "batch_size, num_surface_samples, and audit_surface_samples "
                "must be positive"
            )
        if not np.isfinite(scale) or scale <= 0.0:
            raise MeshObjectError("scale must be finite and positive")
        self.batch_size_each = int(batch_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.scale = float(scale)
        self.num_surface_samples = int(num_surface_samples)
        self.audit_surface_samples = int(audit_surface_samples)
        self.face_chunk_size = int(face_chunk_size)
        self.point_chunk_size = int(point_chunk_size)

        loaded = trimesh.load(self.mesh_path, force="mesh", process=False)
        if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
            raise MeshObjectError(f"{self.mesh_path} did not load as a non-empty triangle mesh")
        self.object_mesh = loaded.copy()
        self.object_mesh.remove_unreferenced_vertices()
        self.object_mesh.fix_normals(multibody=True)
        if not self.object_mesh.is_watertight:
            raise MeshObjectError(f"Generic grasp generation requires a watertight mesh: {self.mesh_path}")
        vertices = np.asarray(self.object_mesh.vertices, dtype=np.float64) * self.scale
        faces = np.asarray(self.object_mesh.faces, dtype=np.int64)
        triangles = vertices[faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        norms = np.linalg.norm(cross, axis=1)
        if np.any(norms <= 1.0e-15):
            raise MeshObjectError("Mesh contains a degenerate triangle")
        normals = cross / norms[:, None]

        face_order, face_chunk_offsets = _spatial_face_order(
            triangles, self.face_chunk_size
        )
        faces = faces[face_order]
        triangles = triangles[face_order]
        normals = normals[face_order]
        norms = norms[face_order]
        self.face_chunk_offsets = face_chunk_offsets

        self.vertices = torch.tensor(vertices, device=self.device, dtype=dtype)
        self.bounds_lower = self.vertices.amin(dim=0)
        self.bounds_upper = self.vertices.amax(dim=0)
        self.faces = torch.tensor(faces, device=self.device, dtype=torch.long)
        self.triangles = torch.tensor(triangles, device=self.device, dtype=dtype)
        self.face_normals = torch.tensor(normals, device=self.device, dtype=dtype)
        chunk_lower = []
        chunk_upper = []
        for start, stop in zip(face_chunk_offsets[:-1], face_chunk_offsets[1:]):
            values = triangles[start:stop]
            chunk_lower.append(values.min(axis=(0, 1)))
            chunk_upper.append(values.max(axis=(0, 1)))
        self.face_chunk_lower = torch.tensor(
            np.stack(chunk_lower), device=self.device, dtype=dtype
        )
        self.face_chunk_upper = torch.tensor(
            np.stack(chunk_upper), device=self.device, dtype=dtype
        )
        self.object_scale_tensor = torch.full(
            (1, self.batch_size_each), self.scale, device=self.device, dtype=dtype
        )
        self.object_mesh_list = [self.object_mesh]

        rng = np.random.default_rng(seed)
        areas = norms * 0.5
        chosen = rng.choice(len(faces), size=self.num_surface_samples, p=areas / areas.sum())
        random = rng.random((self.num_surface_samples, 2))
        root = np.sqrt(random[:, 0])
        weights = np.stack((1.0 - root, root * (1.0 - random[:, 1]), root * random[:, 1]), axis=1)
        surface = np.einsum("ni,nij->nj", weights, triangles[chosen])
        one_surface = torch.tensor(surface, device=self.device, dtype=dtype)
        self.surface_points_tensor = one_surface.unsqueeze(0).expand(
            self.batch_size_each, -1, -1
        ).clone()
        audit_surface = _deterministic_surface_samples(
            triangles,
            norms * 0.5,
            self.audit_surface_samples,
        )
        self.audit_surface_points = torch.tensor(
            audit_surface, device=self.device, dtype=dtype
        )
        self.convex_hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).convex_hull

    def _query_flat(self, points: torch.Tensor) -> MeshQuery:
        best_squared = torch.full(
            (len(points),), torch.inf, device=points.device, dtype=points.dtype
        )
        best_closest = torch.zeros_like(points)
        best_normal = torch.zeros_like(points)
        triangles = self.triangles.to(points)
        normals = self.face_normals.to(points)

        # A nearest vertex is a point on the mesh and therefore supplies a safe
        # upper bound on point-to-surface distance.  Spatial leaf AABBs whose
        # lower bound exceeds it cannot contain the nearest triangle.  No face
        # is approximated or dropped unless this proof holds.
        vertex_upper = torch.full_like(best_squared, torch.inf)
        vertices = self.vertices.to(points)
        for start in range(0, len(vertices), 1024):
            squared = torch.sum(
                (points[:, None, :] - vertices[None, start : start + 1024]) ** 2,
                dim=-1,
            )
            vertex_upper = torch.minimum(vertex_upper, squared.amin(dim=1))

        chunk_lower = self.face_chunk_lower.to(points)
        chunk_upper = self.face_chunk_upper.to(points)
        for chunk_index, (start, stop) in enumerate(
            zip(self.face_chunk_offsets[:-1], self.face_chunk_offsets[1:])
        ):
            lower = chunk_lower[chunk_index]
            upper = chunk_upper[chunk_index]
            outside = torch.relu(lower - points) + torch.relu(points - upper)
            lower_bound = torch.sum(outside.square(), dim=-1)
            pruning_bound = torch.minimum(best_squared.detach(), vertex_upper)
            candidate_mask = lower_bound <= (
                pruning_bound * (1.0 + 1.0e-12) + 1.0e-24
            )
            candidate_indices = torch.nonzero(
                candidate_mask, as_tuple=False
            ).flatten()
            if candidate_indices.numel() == 0:
                continue
            selected_points = points.index_select(0, candidate_indices)
            closest, squared, local_face = _triangle_chunk_closest(
                selected_points, triangles[start:stop]
            )
            previous_squared = best_squared.index_select(0, candidate_indices)
            update = squared < previous_squared
            selected_normal = normals[start:stop][local_face]
            best_squared = best_squared.index_copy(
                0,
                candidate_indices,
                torch.where(update, squared, previous_squared),
            )
            previous_closest = best_closest.index_select(0, candidate_indices)
            best_closest = best_closest.index_copy(
                0,
                candidate_indices,
                torch.where(update[:, None], closest, previous_closest),
            )
            previous_normal = best_normal.index_select(0, candidate_indices)
            best_normal = best_normal.index_copy(
                0,
                candidate_indices,
                torch.where(update[:, None], selected_normal, previous_normal),
            )

        distance = torch.sqrt(best_squared.clamp_min(1.0e-18))
        orientation = torch.sum((points - best_closest) * best_normal, dim=-1)
        # Match original DexGraspNet: inside positive, outside negative.
        signed = torch.where(orientation <= 0.0, distance, -distance)
        return MeshQuery(signed, best_closest, best_normal)

    def query(self, points: torch.Tensor) -> MeshQuery:
        if not isinstance(points, torch.Tensor) or points.shape[-1] != 3:
            raise MeshObjectError("points must be a tensor ending in dimension 3")
        original_shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        signed_parts: list[torch.Tensor] = []
        closest_parts: list[torch.Tensor] = []
        normal_parts: list[torch.Tensor] = []
        for start in range(0, len(flat), self.point_chunk_size):
            result = self._query_flat(flat[start : start + self.point_chunk_size])
            signed_parts.append(result.signed_distance)
            closest_parts.append(result.closest_points)
            normal_parts.append(result.outward_normals)
        return MeshQuery(
            torch.cat(signed_parts).reshape(original_shape),
            torch.cat(closest_parts).reshape(*original_shape, 3),
            torch.cat(normal_parts).reshape(*original_shape, 3),
        )

    def penetration_depth(self, points: torch.Tensor) -> torch.Tensor:
        """Return positive inside depth using a conservative closed-AABB filter.

        A point strictly outside the scaled mesh's axis-aligned bounds cannot
        be inside the watertight mesh, so its penetration is exactly zero.
        Points on the closed bounds, and all non-finite points, retain the full
        triangle query.  The returned tensor is numerically equivalent to
        ``relu(query(points).signed_distance)`` and has shape ``points.shape[:-1]``.
        """

        if not isinstance(points, torch.Tensor) or points.shape[-1] != 3:
            raise MeshObjectError("points must be a tensor ending in dimension 3")
        original_shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        lower = self.bounds_lower.to(flat)
        upper = self.bounds_upper.to(flat)
        finite = torch.isfinite(flat).all(dim=-1)
        within_closed_bounds = ((flat >= lower) & (flat <= upper)).all(dim=-1)
        candidate_mask = within_closed_bounds | ~finite
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()

        # Preserve a zero gradient connection to every culled point.  This
        # also handles a genuinely empty input without calling query(), whose
        # concatenation contract requires at least one point.
        penetration = flat[:, 0] * 0.0
        if candidate_indices.numel() == 0:
            return penetration.reshape(original_shape)
        candidate_depth = torch.relu(
            self.query(flat[candidate_indices]).signed_distance
        )
        return penetration.index_copy(
            0, candidate_indices, candidate_depth
        ).reshape(original_shape)

    def penetration_summary(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact per-row total and maximum inside depth.

        ``points`` must be ``(B,N,3)`` with a non-empty point dimension.  The
        closed-AABB filter in :meth:`penetration_depth` removes points that
        provably cannot be inside before the full triangle query, while the
        returned reductions remain exactly equivalent to reducing the dense
        signed-distance result.
        """

        if (
            not isinstance(points, torch.Tensor)
            or points.ndim != 3
            or points.shape[-1] != 3
            or points.shape[1] <= 0
        ):
            raise MeshObjectError("points must have shape (B,N,3) with N > 0")
        depth = self.penetration_depth(points)
        return depth.sum(dim=-1), depth.amax(dim=-1)

    def cal_distance(
        self, points: torch.Tensor, with_closest_points: bool = False
    ) -> tuple[torch.Tensor, ...]:
        if points.ndim != 3 or points.shape[0] != self.batch_size_each:
            raise MeshObjectError(
                f"points must have shape ({self.batch_size_each},N,3), got {tuple(points.shape)}"
            )
        result = self.query(points)
        if with_closest_points:
            return result.signed_distance, result.outward_normals, result.closest_points
        return result.signed_distance, result.outward_normals


__all__ = ["MeshObjectError", "MeshObjectModel", "MeshQuery"]
