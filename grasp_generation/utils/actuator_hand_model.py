"""Strict, differentiable X2 hand model backed by the composed USD asset.

This module intentionally has no Isaac Gym dependency.  OpenUSD is used once at
construction time to validate and extract the real ``x2_keypoints.usda``
articulation; all runtime kinematics and collision-proxy calculations use
PyTorch so gradients can flow to all twelve actuators when a caller chooses to
optimize them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .x2_config import X2Config


class ActuatorHandModelError(RuntimeError):
    """Raised when the X2 configuration, USD asset, or runtime input is invalid."""


@dataclass(frozen=True)
class CollisionMesh:
    """One cooked-shape approximation expressed in its rigid-link frame."""

    link_name: str
    prim_path: str
    approximation: str
    authored_vertex_count: int
    authored_triangle_count: int
    vertices_local: np.ndarray
    triangles: np.ndarray
    plane_normals_local: np.ndarray
    plane_offsets_local: np.ndarray


@dataclass(frozen=True)
class CapsuleProxy:
    """Legacy PCA capsule used by the self-penetration energy."""

    link_name: str
    point_a_local: tuple[float, float, float]
    point_b_local: tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class _JointSpec:
    name: str
    parent_link: str
    child_link: str
    full_state_index: int
    axis: np.ndarray
    parent_joint_frame: np.ndarray
    child_joint_frame_inverse: np.ndarray


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _gf_matrix_to_column(matrix: Any) -> np.ndarray:
    """Convert an OpenUSD row-vector matrix to a standard column-vector matrix."""

    return np.asarray(matrix, dtype=np.float64).T.copy()


def _pose_matrix_from_usd(position: Any, rotation: Any, Gf: Any) -> np.ndarray:
    xyz = np.asarray(position, dtype=np.float64)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ActuatorHandModelError(f"Invalid USD joint local position: {position!r}")
    imaginary = np.asarray(rotation.GetImaginary(), dtype=np.float64)
    real = float(rotation.GetReal())
    quaternion = np.concatenate(([real], imaginary))
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ActuatorHandModelError(f"Invalid USD joint local rotation: {rotation!r}")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ActuatorHandModelError(f"Zero-length USD joint quaternion: {rotation!r}")
    quaternion /= norm
    usd_quaternion = Gf.Quatd(
        float(quaternion[0]),
        Gf.Vec3d(*[float(item) for item in quaternion[1:]]),
    )
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(usd_quaternion)
    matrix.SetTranslateOnly(Gf.Vec3d(*[float(item) for item in xyz]))
    return _gf_matrix_to_column(matrix)


def _sample_vertices(vertices: np.ndarray, count: int) -> np.ndarray:
    """Deterministic farthest-point sample from a bounded candidate subset."""

    if len(vertices) <= count:
        return _readonly(vertices.copy())

    # The source meshes are deliberately high resolution (roughly 30k--82k
    # vertices/link).  A stratified candidate subset keeps startup bounded while
    # retaining coverage across the authored vertex order.
    candidate_count = min(len(vertices), max(4096, count * 16))
    candidate_indices = np.linspace(0, len(vertices) - 1, candidate_count, dtype=np.int64)
    extrema = np.concatenate(
        [np.argmin(vertices, axis=0), np.argmax(vertices, axis=0)]
    ).astype(np.int64)
    candidate_indices = np.unique(np.concatenate([candidate_indices, extrema]))
    candidates = vertices[candidate_indices]

    centroid = candidates.mean(axis=0)
    first = int(np.argmax(np.sum((candidates - centroid) ** 2, axis=1)))
    chosen = np.empty(count, dtype=np.int64)
    chosen[0] = first
    min_distance_sq = np.sum((candidates - candidates[first]) ** 2, axis=1)
    for index in range(1, count):
        selected = int(np.argmax(min_distance_sq))
        chosen[index] = selected
        distance_sq = np.sum((candidates - candidates[selected]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
    return _readonly(candidates[chosen].copy())


def _sample_vertices_fixed(vertices: np.ndarray, count: int) -> np.ndarray:
    """Return exactly ``count`` deterministic samples, repeating only if needed."""

    sampled = np.asarray(_sample_vertices(vertices, count), dtype=np.float64)
    if len(sampled) == count:
        return _readonly(sampled.copy())
    if len(sampled) == 0:
        raise ActuatorHandModelError("Cannot sample an empty collision point set")
    indices = np.arange(count, dtype=np.int64) % len(sampled)
    return _readonly(sampled[indices].copy())


def _sample_triangle_surfaces(
    vertices: np.ndarray,
    triangles: np.ndarray,
    count: int,
) -> np.ndarray:
    """Deterministically sample face interiors in proportion to surface area.

    Convex-hull triangulation can create a very large, thin triangle whose
    vertices and centroid both miss a narrow intersection through another part
    of its interior. Area-stratified samples expose the full face to the mesh
    penetration energy.
    """
    triangle_vertices = vertices[triangles]
    cross = np.cross(
        triangle_vertices[:, 1] - triangle_vertices[:, 0],
        triangle_vertices[:, 2] - triangle_vertices[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = np.isfinite(areas) & (areas > 1.0e-18)
    if not bool(valid.any()):
        raise ActuatorHandModelError("Convex-hull collision mesh has no finite positive-area face")
    valid_indices = np.flatnonzero(valid)
    valid_areas = areas[valid]
    cumulative = np.cumsum(valid_areas)
    total_area = float(cumulative[-1])
    targets = (np.arange(count, dtype=np.float64) + 0.5) * (total_area / count)
    selected_valid = np.searchsorted(cumulative, targets, side="left")
    selected_faces = valid_indices[np.minimum(selected_valid, len(valid_indices) - 1)]

    # Two irrational rotations form a deterministic low-discrepancy sequence.
    # The square-root map converts the unit square to uniform triangle area.
    sequence = np.arange(count, dtype=np.float64) + 0.5
    u = np.mod(sequence * 0.7548776662466927, 1.0)
    v = np.mod(sequence * 0.5698402909980532, 1.0)
    root_u = np.sqrt(u)
    weights = np.stack(
        (1.0 - root_u, root_u * (1.0 - v), root_u * v), axis=1
    )
    sampled_triangles = triangle_vertices[selected_faces]
    points = np.einsum("ni,nij->nj", weights, sampled_triangles)
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise ActuatorHandModelError("Could not construct finite area-weighted collision samples")
    return _readonly(points)


def _fit_capsule(link_name: str, vertices: np.ndarray) -> CapsuleProxy:
    """Fit the deterministic legacy PCA capsule to link-local vertices.

    The endpoint retraction matches the historical X2 energy and is deliberately
    retained for compatibility.  It is not a conservative enclosure; the
    separately computed enclosure residual inflates it for safe broadphase use.
    """

    center = vertices.mean(axis=0)
    centered = vertices - center
    covariance = centered.T @ centered / max(len(vertices), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0.0:
        axis = -axis
    projections = centered @ axis
    perpendicular = centered - projections[:, None] * axis[None, :]
    radius = float(np.sqrt(np.max(np.sum(perpendicular * perpendicular, axis=1))))
    if not math.isfinite(radius):
        raise ActuatorHandModelError(f"Could not fit finite capsule radius for {link_name}")
    radius = max(radius, 1.0e-6)

    lower = float(projections.min())
    upper = float(projections.max())
    midpoint = 0.5 * (lower + upper)
    half_segment = max(0.5 * (upper - lower) - radius, 0.0)
    point_a = center + (midpoint - half_segment) * axis
    point_b = center + (midpoint + half_segment) * axis
    return CapsuleProxy(
        link_name=link_name,
        point_a_local=tuple(float(item) for item in point_a),
        point_b_local=tuple(float(item) for item in point_b),
        radius=radius,
    )


def _capsule_enclosure_residual(proxy: CapsuleProxy, vertices: np.ndarray) -> float:
    """Maximum distance by which hull vertices extend beyond a capsule."""

    start = np.asarray(proxy.point_a_local, dtype=np.float64)
    end = np.asarray(proxy.point_b_local, dtype=np.float64)
    segment = end - start
    denominator = max(float(segment @ segment), 1.0e-18)
    fraction = ((vertices - start) @ segment / denominator).clip(0.0, 1.0)
    closest = start + fraction[:, None] * segment[None, :]
    distance = np.linalg.norm(vertices - closest, axis=1)
    residual = max(float(distance.max(initial=0.0)) - proxy.radius, 0.0)
    if not math.isfinite(residual):
        raise ActuatorHandModelError(
            f"Could not compute finite capsule enclosure residual for {proxy.link_name}"
        )
    return residual


class ActuatorHandModel:
    """Validated X2 actuator mapping, differentiable FK, and collision geometry.

    Parameters
    ----------
    config:
        A fully loaded :class:`X2Config`. Paths, names, limits, and couplings
        are never guessed or silently replaced.
    device, dtype:
        Storage defaults for exposed tensors.  Runtime torch inputs may use a
        different floating dtype/device; immutable geometry is converted to
        match without detaching the input.
    collision_samples_per_link:
        Number of convex-hull vertices, face centroids, and area-stratified
        face-interior points retained per rigid link for mesh penetration
        energy. Sampling all three covers broad convex faces as well as edges.
    audit_collision_samples_per_link:
        Independent dense vertex, centroid, and face-interior sample count used
        only by the final hand-object penetration gate.
    """

    EXPECTED_ACTUATOR_COUNT = 12
    EXPECTED_NON_THUMB_COUNT = 8
    EXPECTED_JOINT_COUNT = 16
    EXPECTED_RIGID_BODY_COUNT = 17
    EXPECTED_COLLISION_MESH_COUNT = 17
    EXPECTED_CAPSULE_PAIR_COUNT = 98
    EXPECTED_HULL_PAIR_COUNT = 119

    def __init__(
        self,
        config: X2Config,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        collision_samples_per_link: int = 128,
        audit_collision_samples_per_link: int = 256,
        self_collision_samples_per_link: int | None = None,
        fk_tolerance: float = 2.0e-5,
    ) -> None:
        if not all(hasattr(config, name) for name in ("require", "configured_path")):
            raise TypeError(
                "ActuatorHandModel requires an X2 configuration with require() "
                "and configured_path() methods"
            )
        if not isinstance(collision_samples_per_link, int) or collision_samples_per_link <= 0:
            raise ValueError("collision_samples_per_link must be a positive integer")
        if (
            not isinstance(audit_collision_samples_per_link, int)
            or audit_collision_samples_per_link <= 0
        ):
            raise ValueError(
                "audit_collision_samples_per_link must be a positive integer"
            )
        if self_collision_samples_per_link is None:
            self_collision_samples_per_link = int(
                config.require("self_collision.surface_samples_per_link")
            )
        if (
            not isinstance(self_collision_samples_per_link, int)
            or self_collision_samples_per_link <= 0
        ):
            raise ValueError("self_collision_samples_per_link must be a positive integer")
        if not isinstance(dtype, torch.dtype) or not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError(f"dtype must be a floating torch dtype, got {dtype!r}")
        if not math.isfinite(float(fk_tolerance)) or fk_tolerance <= 0.0:
            raise ValueError("fk_tolerance must be finite and positive")

        self.config = config
        self.usd_path = config.configured_path("robot.usd_path", must_exist=True)
        if self.usd_path.name != "x2_keypoints.usda":
            raise ActuatorHandModelError(
                "robot.usd_path must select the real x2_keypoints.usda asset, "
                f"got {self.usd_path}"
            )
        self.collision_samples_per_link = collision_samples_per_link
        self.audit_collision_samples_per_link = audit_collision_samples_per_link
        self.self_collision_samples_per_link = self_collision_samples_per_link
        self.self_collision_broadphase_margin = float(
            config.require("self_collision.broadphase_margin")
        )
        if (
            not math.isfinite(self.self_collision_broadphase_margin)
            or self.self_collision_broadphase_margin <= 0.0
        ):
            raise ActuatorHandModelError(
                "self_collision.broadphase_margin must be finite and positive"
            )

        self.actuator_names = self._read_name_list(
            "robot.actuator_names", self.EXPECTED_ACTUATOR_COUNT
        )
        self.non_thumb_actuator_names = self._read_name_list(
            "robot.active_non_thumb_actuators", self.EXPECTED_NON_THUMB_COUNT
        )
        self.thumb_actuator_names = self._read_name_list("robot.thumb_actuators", 4)
        if self.actuator_names != self.non_thumb_actuator_names + self.thumb_actuator_names:
            raise ActuatorHandModelError(
                "robot.actuator_names must contain the configured 8 non-thumb actuators "
                "followed by the 4 thumb actuators"
            )
        self.full_joint_names = self._read_name_list(
            "robot.full_joint_names", self.EXPECTED_JOINT_COUNT
        )

        self.palm_link_name = str(config.require("robot.palm_link_name"))
        self.palm_link_path = str(config.require("robot.palm_link_path"))
        finger_links_raw = config.require("robot.finger_links")
        if not isinstance(finger_links_raw, Mapping):
            raise ActuatorHandModelError("robot.finger_links must be a mapping")
        self.finger_links: dict[str, tuple[str, ...]] = {}
        for finger_name in ("index", "middle", "ring", "little"):
            value = finger_links_raw.get(finger_name)
            if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 3:
                raise ActuatorHandModelError(
                    f"robot.finger_links.{finger_name} must contain exactly three link names"
                )
            self.finger_links[finger_name] = tuple(str(item) for item in value)
        self.thumb_links = self._read_name_list("robot.thumb_links", 4)
        finger_link_names = tuple(
            link for finger_name in ("index", "middle", "ring", "little")
            for link in self.finger_links[finger_name]
        )
        self.link_names = (self.palm_link_name,) + finger_link_names + self.thumb_links
        if len(self.link_names) != self.EXPECTED_RIGID_BODY_COUNT or len(set(self.link_names)) != len(
            self.link_names
        ):
            raise ActuatorHandModelError(
                "Configured palm, four-finger, and thumb links must define 17 unique rigid bodies"
            )

        coupling_raw = config.require("robot.passive_joint_coupling")
        if not isinstance(coupling_raw, Mapping) or len(coupling_raw) != 4:
            raise ActuatorHandModelError(
                "robot.passive_joint_coupling must define exactly four passive J1 joints"
            )
        self.passive_joint_coupling: dict[str, tuple[str, float, float]] = {}
        for passive_name, raw in coupling_raw.items():
            if not isinstance(raw, Mapping):
                raise ActuatorHandModelError(
                    f"Passive coupling for {passive_name!r} must be a mapping"
                )
            try:
                driver = str(raw["driver"])
                multiplier = float(raw["multiplier"])
                offset = float(raw["offset"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ActuatorHandModelError(
                    f"Invalid passive coupling for {passive_name!r}: {raw!r}"
                ) from exc
            if driver not in self.actuator_names:
                raise ActuatorHandModelError(
                    f"Passive joint {passive_name} references non-actuator driver {driver}"
                )
            if multiplier != 1.0 or offset != 0.0:
                raise ActuatorHandModelError(
                    f"X2 requires {passive_name}={driver}; got multiplier={multiplier}, "
                    f"offset={offset}"
                )
            self.passive_joint_coupling[str(passive_name)] = (driver, multiplier, offset)
        if set(self.full_joint_names) != set(self.actuator_names) | set(
            self.passive_joint_coupling
        ):
            raise ActuatorHandModelError(
                "The 16 full joint names must be exactly 12 actuators plus four passive J1 joints"
            )

        fixed_thumb_raw = config.require("robot.fixed_thumb_position")
        if not isinstance(fixed_thumb_raw, Mapping) or set(fixed_thumb_raw) != set(
            self.thumb_actuator_names
        ):
            raise ActuatorHandModelError(
                "robot.fixed_thumb_position must define exactly the four configured thumb actuators"
            )
        self.fixed_thumb_position = {
            name: float(fixed_thumb_raw[name]) for name in self.thumb_actuator_names
        }
        if not all(math.isfinite(value) for value in self.fixed_thumb_position.values()):
            raise ActuatorHandModelError("Fixed thumb positions must be finite")

        actuator_limits_raw = config.require("robot.actuator_limits")
        if not isinstance(actuator_limits_raw, Mapping) or set(actuator_limits_raw) != set(
            self.actuator_names
        ):
            raise ActuatorHandModelError(
                "robot.actuator_limits must define exactly the 12 configured actuator names"
            )
        actuator_limits_np = np.empty((self.EXPECTED_ACTUATOR_COUNT, 2), dtype=np.float64)
        for index, name in enumerate(self.actuator_names):
            bounds = actuator_limits_raw[name]
            if not isinstance(bounds, Sequence) or isinstance(bounds, str) or len(bounds) != 2:
                raise ActuatorHandModelError(f"Actuator limit for {name} must contain [lower, upper]")
            try:
                lower, upper = (float(bounds[0]), float(bounds[1]))
            except (TypeError, ValueError) as exc:
                raise ActuatorHandModelError(f"Non-numeric actuator limit for {name}: {bounds!r}") from exc
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ActuatorHandModelError(f"Invalid actuator limit for {name}: {bounds!r}")
            actuator_limits_np[index] = (lower, upper)
        for name, value in self.fixed_thumb_position.items():
            lower, upper = actuator_limits_np[self.actuator_names.index(name)]
            if value < lower or value > upper:
                raise ActuatorHandModelError(
                    f"Fixed thumb value {name}={value} is outside [{lower}, {upper}]"
                )

        self._actuator_index = {name: index for index, name in enumerate(self.actuator_names)}
        self._joint_index = {name: index for index, name in enumerate(self.full_joint_names)}
        usd_data = self._load_and_validate_usd(actuator_limits_np)
        self._joint_specs: tuple[_JointSpec, ...] = usd_data["joint_specs"]
        self.rigid_body_paths: dict[str, str] = usd_data["rigid_body_paths"]
        self.collision_meshes: dict[str, CollisionMesh] = usd_data["collision_meshes"]
        self.collision_vertices_local: dict[str, np.ndarray] = {
            name: self.collision_meshes[name].vertices_local for name in self.link_names
        }
        self.collision_triangle_centroids_local: dict[str, np.ndarray] = {
            name: _readonly(
                self.collision_vertices_local[name][
                    self.collision_meshes[name].triangles
                ].mean(axis=1)
            )
            for name in self.link_names
        }
        self.collision_vertex_samples_local: dict[str, np.ndarray] = {
            name: _sample_vertices(
                self.collision_vertices_local[name], self.collision_samples_per_link
            )
            for name in self.link_names
        }
        self.collision_centroid_samples_local: dict[str, np.ndarray] = {
            name: _sample_vertices(
                self.collision_triangle_centroids_local[name],
                self.collision_samples_per_link,
            )
            for name in self.link_names
        }
        self.collision_face_samples_local: dict[str, np.ndarray] = {
            name: _sample_triangle_surfaces(
                self.collision_vertices_local[name],
                self.collision_meshes[name].triangles,
                self.collision_samples_per_link,
            )
            for name in self.link_names
        }
        self.collision_surface_samples_local: dict[str, np.ndarray] = {
            name: _readonly(
                np.concatenate(
                    (
                        self.collision_vertex_samples_local[name],
                        self.collision_centroid_samples_local[name],
                        self.collision_face_samples_local[name],
                    ),
                    axis=0,
                )
            )
            for name in self.link_names
        }
        self.audit_collision_vertex_samples_local: dict[str, np.ndarray] = {
            name: _sample_vertices_fixed(
                self.collision_vertices_local[name],
                self.audit_collision_samples_per_link,
            )
            for name in self.link_names
        }
        self.audit_collision_centroid_samples_local: dict[str, np.ndarray] = {
            name: _sample_vertices_fixed(
                self.collision_triangle_centroids_local[name],
                self.audit_collision_samples_per_link,
            )
            for name in self.link_names
        }
        self.audit_collision_face_samples_local: dict[str, np.ndarray] = {
            name: _sample_triangle_surfaces(
                self.collision_vertices_local[name],
                self.collision_meshes[name].triangles,
                self.audit_collision_samples_per_link,
            )
            for name in self.link_names
        }
        self.audit_collision_surface_samples_local: dict[str, np.ndarray] = {
            name: _readonly(
                np.concatenate(
                    (
                        self.audit_collision_vertex_samples_local[name],
                        self.audit_collision_centroid_samples_local[name],
                        self.audit_collision_face_samples_local[name],
                    ),
                    axis=0,
                )
            )
            for name in self.link_names
        }
        # Self-collision uses a fixed 64x3 (configurable) point layout so every
        # pair has the same tensor shape.  Collision meshes with fewer vertices
        # are deterministically repeated; centroid and face samples preserve
        # broad-face coverage.
        self.self_collision_vertex_samples_local: dict[str, np.ndarray] = {
            name: _sample_vertices_fixed(
                self.collision_vertices_local[name], self.self_collision_samples_per_link
            )
            for name in self.link_names
        }
        self.self_collision_centroid_samples_local: dict[str, np.ndarray] = {
            name: _sample_vertices_fixed(
                self.collision_triangle_centroids_local[name],
                self.self_collision_samples_per_link,
            )
            for name in self.link_names
        }
        self.self_collision_face_samples_local: dict[str, np.ndarray] = {
            name: _sample_triangle_surfaces(
                self.collision_vertices_local[name],
                self.collision_meshes[name].triangles,
                self.self_collision_samples_per_link,
            )
            for name in self.link_names
        }
        self.self_collision_surface_samples_local: dict[str, np.ndarray] = {
            name: _readonly(
                np.concatenate(
                    (
                        self.self_collision_vertex_samples_local[name],
                        self.self_collision_centroid_samples_local[name],
                        self.self_collision_face_samples_local[name],
                    ),
                    axis=0,
                )
            )
            for name in self.link_names
        }
        self.capsule_proxies = tuple(
            _fit_capsule(name, self.collision_vertices_local[name]) for name in self.link_names
        )
        self.capsule_proxy_by_link = {proxy.link_name: proxy for proxy in self.capsule_proxies}
        self.capsule_enclosure_residual_by_link = {
            proxy.link_name: _capsule_enclosure_residual(
                proxy, self.collision_vertices_local[proxy.link_name]
            )
            for proxy in self.capsule_proxies
        }

        self._rest_link_transforms: dict[str, np.ndarray] = usd_data["rest_link_transforms"]
        joint_limits_np: np.ndarray = usd_data["joint_limits"]

        target_device = torch.device("cpu" if device is None else device)
        self._device = target_device
        self._dtype = dtype
        self.actuator_limits_tensor = torch.as_tensor(
            actuator_limits_np, device=target_device, dtype=dtype
        )
        self.joint_limits_tensor = torch.as_tensor(
            joint_limits_np, device=target_device, dtype=dtype
        )
        self._joint_parent_frames_tensor = torch.as_tensor(
            np.stack([spec.parent_joint_frame for spec in self._joint_specs]),
            device=target_device,
            dtype=dtype,
        )
        self._joint_child_frames_inverse_tensor = torch.as_tensor(
            np.stack([spec.child_joint_frame_inverse for spec in self._joint_specs]),
            device=target_device,
            dtype=dtype,
        )
        self._joint_axes_tensor = torch.as_tensor(
            np.stack([spec.axis for spec in self._joint_specs]),
            device=target_device,
            dtype=dtype,
        )
        self._joint_state_indices_tensor = torch.as_tensor(
            [spec.full_state_index for spec in self._joint_specs],
            device=target_device,
            dtype=torch.long,
        )
        self._collision_sample_tensors = {
            name: torch.tensor(
                self.collision_surface_samples_local[name], device=target_device, dtype=dtype
            )
            for name in self.link_names
        }
        self._audit_collision_sample_tensors = {
            name: torch.tensor(
                self.audit_collision_surface_samples_local[name],
                device=target_device,
                dtype=dtype,
            )
            for name in self.link_names
        }
        self._self_collision_sample_tensors = {
            name: torch.tensor(
                self.self_collision_surface_samples_local[name],
                device=target_device,
                dtype=dtype,
            )
            for name in self.link_names
        }
        self._collision_plane_normal_tensors = {
            name: torch.tensor(
                np.array(self.collision_meshes[name].plane_normals_local, copy=True),
                device=target_device,
                dtype=dtype,
            )
            for name in self.link_names
        }
        self._collision_plane_offset_tensors = {
            name: torch.tensor(
                np.array(self.collision_meshes[name].plane_offsets_local, copy=True),
                device=target_device,
                dtype=dtype,
            )
            for name in self.link_names
        }
        self._capsule_a_tensor = torch.as_tensor(
            [proxy.point_a_local for proxy in self.capsule_proxies],
            device=target_device,
            dtype=dtype,
        )
        self._capsule_b_tensor = torch.as_tensor(
            [proxy.point_b_local for proxy in self.capsule_proxies],
            device=target_device,
            dtype=dtype,
        )
        self._capsule_radius_tensor = torch.as_tensor(
            [proxy.radius for proxy in self.capsule_proxies],
            device=target_device,
            dtype=dtype,
        )
        self._capsule_enclosure_residual_tensor = torch.as_tensor(
            [
                self.capsule_enclosure_residual_by_link[proxy.link_name]
                for proxy in self.capsule_proxies
            ],
            device=target_device,
            dtype=dtype,
        )
        self._capsule_broadphase_radius_tensor = (
            self._capsule_radius_tensor
            + self._capsule_enclosure_residual_tensor
            + self.self_collision_broadphase_margin
        )

        neighbors: dict[str, set[str]] = {name: set() for name in self.link_names}
        directly_adjacent: set[frozenset[str]] = set()
        for spec in self._joint_specs:
            neighbors[spec.parent_link].add(spec.child_link)
            neighbors[spec.child_link].add(spec.parent_link)
            directly_adjacent.add(frozenset((spec.parent_link, spec.child_link)))
        structurally_near = set(directly_adjacent)
        # Preserve the historical capsule pair set: links one or two edges apart
        # share a structural joint region, where the coarse proxy produces false
        # penetration rather than a useful optimization signal.
        for linked_names in neighbors.values():
            ordered = sorted(linked_names)
            for first_index in range(len(ordered)):
                for second_index in range(first_index + 1, len(ordered)):
                    structurally_near.add(
                        frozenset((ordered[first_index], ordered[second_index]))
                    )
        legacy_pair_names: list[tuple[str, str]] = []
        legacy_pair_indices: list[tuple[int, int]] = []
        hull_pair_names: list[tuple[str, str]] = []
        hull_pair_indices: list[tuple[int, int]] = []
        for first in range(len(self.link_names)):
            for second in range(first + 1, len(self.link_names)):
                names = (self.link_names[first], self.link_names[second])
                pair = frozenset(names)
                if pair not in directly_adjacent:
                    hull_pair_names.append(names)
                    hull_pair_indices.append((first, second))
                if pair not in structurally_near:
                    legacy_pair_names.append(names)
                    legacy_pair_indices.append((first, second))
        self.self_collision_pairs = tuple(legacy_pair_names)
        pair_set = {frozenset(pair) for pair in self.self_collision_pairs}
        proxy_exclusions_raw = self.config.require("robot.self_collision_proxy_exclusions")
        if not isinstance(proxy_exclusions_raw, Sequence) or isinstance(
            proxy_exclusions_raw, str
        ):
            raise ActuatorHandModelError(
                "robot.self_collision_proxy_exclusions must be a sequence"
            )
        proxy_exclusions: set[frozenset[str]] = set()
        for entry in proxy_exclusions_raw:
            if not isinstance(entry, Mapping):
                raise ActuatorHandModelError(
                    "Each self-collision proxy exclusion must be a mapping"
                )
            links = entry.get("links")
            reason = str(entry.get("reason", "")).strip()
            if (
                not isinstance(links, Sequence)
                or isinstance(links, str)
                or len(links) != 2
                or not reason
            ):
                raise ActuatorHandModelError(
                    f"Invalid self-collision proxy exclusion: {entry!r}"
                )
            pair = frozenset(str(name) for name in links)
            if len(pair) != 2 or pair not in pair_set or pair in proxy_exclusions:
                raise ActuatorHandModelError(
                    f"Unknown or duplicate self-collision proxy exclusion: {entry!r}"
                )
            proxy_exclusions.add(pair)
        self.self_collision_proxy_exclusions = tuple(
            pair for pair in self.self_collision_pairs if frozenset(pair) in proxy_exclusions
        )
        proxy_pair_indices = [
            indices
            for names, indices in zip(legacy_pair_names, legacy_pair_indices)
            if frozenset(names) not in proxy_exclusions
        ]
        self.self_collision_proxy_pairs = tuple(
            names
            for names in legacy_pair_names
            if frozenset(names) not in proxy_exclusions
        )
        if len(self.self_collision_proxy_pairs) != self.EXPECTED_CAPSULE_PAIR_COUNT:
            raise ActuatorHandModelError(
                "X2 capsule filtering must produce exactly "
                f"{self.EXPECTED_CAPSULE_PAIR_COUNT} pairs, got "
                f"{len(self.self_collision_proxy_pairs)}"
            )
        self._self_collision_pair_indices_tensor = torch.as_tensor(
            proxy_pair_indices, device=target_device, dtype=torch.long
        )

        hull_pair_set = {frozenset(pair) for pair in hull_pair_names}
        hull_exclusions_raw = self.config.require("robot.self_collision_hull_exclusions")
        if not isinstance(hull_exclusions_raw, Sequence) or isinstance(
            hull_exclusions_raw, str
        ):
            raise ActuatorHandModelError(
                "robot.self_collision_hull_exclusions must be a sequence"
            )
        hull_exclusions: set[frozenset[str]] = set()
        for entry in hull_exclusions_raw:
            if not isinstance(entry, Mapping):
                raise ActuatorHandModelError(
                    "Each self-collision hull exclusion must be a mapping"
                )
            links = entry.get("links")
            reason = str(entry.get("reason", "")).strip()
            if (
                not isinstance(links, Sequence)
                or isinstance(links, str)
                or len(links) != 2
                or not reason
            ):
                raise ActuatorHandModelError(
                    f"Invalid self-collision hull exclusion: {entry!r}"
                )
            pair = frozenset(str(name) for name in links)
            if len(pair) != 2 or pair not in hull_pair_set or pair in hull_exclusions:
                raise ActuatorHandModelError(
                    f"Unknown or duplicate self-collision hull exclusion: {entry!r}"
                )
            hull_exclusions.add(pair)
        self.self_collision_hull_exclusions = tuple(
            pair for pair in hull_pair_names if frozenset(pair) in hull_exclusions
        )
        self.self_collision_hull_pairs = tuple(
            names for names in hull_pair_names if frozenset(names) not in hull_exclusions
        )
        filtered_hull_indices = [
            indices
            for names, indices in zip(hull_pair_names, hull_pair_indices)
            if frozenset(names) not in hull_exclusions
        ]
        if len(self.self_collision_hull_pairs) != self.EXPECTED_HULL_PAIR_COUNT:
            raise ActuatorHandModelError(
                "X2 hull filtering must produce exactly "
                f"{self.EXPECTED_HULL_PAIR_COUNT} pairs, got "
                f"{len(self.self_collision_hull_pairs)}"
            )
        self._self_collision_hull_pair_indices_tensor = torch.as_tensor(
            filtered_hull_indices, device=target_device, dtype=torch.long
        )
        thumb_index_weight = float(
            self.config.require("self_collision.pair_weight_overrides.thumb_index")
        )
        weighted_thumb_links = set(self.thumb_links[1:])
        index_links = set(self.finger_links["index"])
        self.self_collision_hull_pair_weights = tuple(
            thumb_index_weight
            if (
                (first in weighted_thumb_links and second in index_links)
                or (second in weighted_thumb_links and first in index_links)
            )
            else 1.0
            for first, second in self.self_collision_hull_pairs
        )

        self.zero_pose_fk_max_error = self.validate_zero_pose_fk(tolerance=fk_tolerance)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def _read_name_list(self, key: str, expected_count: int) -> tuple[str, ...]:
        value = self.config.require(key)
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise ActuatorHandModelError(f"{key} must be a sequence of names")
        result = tuple(str(item) for item in value)
        if len(result) != expected_count or len(set(result)) != expected_count:
            raise ActuatorHandModelError(
                f"{key} must contain exactly {expected_count} unique names, got {result!r}"
            )
        if any(not name for name in result):
            raise ActuatorHandModelError(f"{key} contains an empty name")
        return result

    def _load_and_validate_usd(self, actuator_limits_np: np.ndarray) -> dict[str, Any]:
        try:
            from pxr import Gf, Usd, UsdGeom, UsdPhysics
        except ImportError as exc:
            raise ActuatorHandModelError(
                "OpenUSD Python bindings are required. Run this model with "
                "`conda run -n isaaclab python ...`."
            ) from exc

        try:
            stage = Usd.Stage.Open(str(self.usd_path), load=Usd.Stage.LoadAll)
        except Exception as exc:
            raise ActuatorHandModelError(f"OpenUSD failed to open {self.usd_path}: {exc}") from exc
        if stage is None:
            raise ActuatorHandModelError(f"OpenUSD returned no stage for {self.usd_path}")

        configured_root = str(self.config.require("robot.source_prim_path"))
        default_prim = stage.GetDefaultPrim()
        if not default_prim or str(default_prim.GetPath()) != configured_root:
            actual = str(default_prim.GetPath()) if default_prim else "<unset>"
            raise ActuatorHandModelError(
                f"USD defaultPrim mismatch: config={configured_root}, composed USD={actual}"
            )
        if str(self.config.require("robot.units")) != "m":
            raise ActuatorHandModelError("robot.units must be exactly 'm'")
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not math.isclose(meters_per_unit, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ActuatorHandModelError(
                f"x2_keypoints.usda must use metersPerUnit=1, got {meters_per_unit}"
            )

        articulation_path = str(self.config.require("robot.articulation_root_path"))
        articulation_prim = stage.GetPrimAtPath(articulation_path)
        if not articulation_prim or not articulation_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ActuatorHandModelError(
                f"Configured articulation root is missing PhysicsArticulationRootAPI: "
                f"{articulation_path}"
            )

        rigid_prims = [
            prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if len(rigid_prims) != self.EXPECTED_RIGID_BODY_COUNT:
            raise ActuatorHandModelError(
                f"Expected 17 rigid bodies in composed USD, found {len(rigid_prims)}: "
                f"{[str(prim.GetPath()) for prim in rigid_prims]}"
            )
        rigid_by_name: dict[str, Any] = {}
        for prim in rigid_prims:
            name = str(prim.GetName())
            if name in rigid_by_name:
                raise ActuatorHandModelError(f"Duplicate rigid-body basename in USD: {name}")
            rigid_by_name[name] = prim
        expected_link_set = set(self.link_names)
        if set(rigid_by_name) != expected_link_set:
            raise ActuatorHandModelError(
                "Rigid-body names do not match config: "
                f"missing={sorted(expected_link_set - set(rigid_by_name))}, "
                f"extra={sorted(set(rigid_by_name) - expected_link_set)}"
            )
        if str(rigid_by_name[self.palm_link_name].GetPath()) != self.palm_link_path:
            raise ActuatorHandModelError(
                f"Palm rigid path mismatch: config={self.palm_link_path}, "
                f"USD={rigid_by_name[self.palm_link_name].GetPath()}"
            )
        rigid_body_paths = {
            name: str(rigid_by_name[name].GetPath()) for name in self.link_names
        }
        path_to_link = {path: name for name, path in rigid_body_paths.items()}

        revolute_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)]
        if len(revolute_prims) != self.EXPECTED_JOINT_COUNT:
            raise ActuatorHandModelError(
                f"Expected 16 revolute joints in composed USD, found {len(revolute_prims)}"
            )
        joint_by_name: dict[str, Any] = {}
        for prim in revolute_prims:
            name = str(prim.GetName())
            if name in joint_by_name:
                raise ActuatorHandModelError(f"Duplicate revolute-joint name in USD: {name}")
            joint_by_name[name] = prim
        if set(joint_by_name) != set(self.full_joint_names):
            raise ActuatorHandModelError(
                "Revolute-joint names do not match config: "
                f"missing={sorted(set(self.full_joint_names) - set(joint_by_name))}, "
                f"extra={sorted(set(joint_by_name) - set(self.full_joint_names))}"
            )

        driven_names = {
            name
            for name, prim in joint_by_name.items()
            if prim.HasAPI(UsdPhysics.DriveAPI, "angular")
        }
        if driven_names != set(self.actuator_names):
            raise ActuatorHandModelError(
                "USD angular-drive set is not the configured 12 actuators: "
                f"missing={sorted(set(self.actuator_names) - driven_names)}, "
                f"extra={sorted(driven_names - set(self.actuator_names))}"
            )

        joint_limits = np.empty((self.EXPECTED_JOINT_COUNT, 2), dtype=np.float64)
        for joint_name in self.full_joint_names:
            revolute = UsdPhysics.RevoluteJoint(joint_by_name[joint_name])
            lower_degrees = revolute.GetLowerLimitAttr().Get()
            upper_degrees = revolute.GetUpperLimitAttr().Get()
            if lower_degrees is None or upper_degrees is None:
                raise ActuatorHandModelError(f"USD joint {joint_name} has no finite limits")
            lower = math.radians(float(lower_degrees))
            upper = math.radians(float(upper_degrees))
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ActuatorHandModelError(
                    f"USD joint {joint_name} has invalid degree limits "
                    f"[{lower_degrees}, {upper_degrees}]"
                )
            joint_limits[self._joint_index[joint_name]] = (lower, upper)
            if joint_name in self._actuator_index:
                configured = actuator_limits_np[self._actuator_index[joint_name]]
                if not np.allclose(configured, (lower, upper), rtol=0.0, atol=2.0e-6):
                    raise ActuatorHandModelError(
                        f"Actuator limits for {joint_name} disagree with composed USD: "
                        f"config={configured.tolist()} rad, USD={[lower, upper]} rad"
                    )

        for passive_name, (driver, _, _) in self.passive_joint_coupling.items():
            prim = joint_by_name.get(passive_name)
            if prim is None:
                raise ActuatorHandModelError(f"Passive joint is absent from USD: {passive_name}")
            relation = prim.GetRelationship("newton:mimicJoint")
            targets = list(relation.GetTargets()) if relation else []
            if len(targets) != 1 or str(targets[0].name) != driver:
                raise ActuatorHandModelError(
                    f"USD mimic relation for {passive_name} must target {driver}, got {targets}"
                )
            passive_bounds = joint_limits[self._joint_index[passive_name]]
            driver_bounds = joint_limits[self._joint_index[driver]]
            if not np.allclose(passive_bounds, driver_bounds, rtol=0.0, atol=2.0e-6):
                raise ActuatorHandModelError(
                    f"Coupled limits differ for {passive_name} and {driver}: "
                    f"{passive_bounds.tolist()} vs {driver_bounds.tolist()}"
                )

        raw_specs: dict[str, _JointSpec] = {}
        incoming_children: set[str] = set()
        for joint_name, prim in joint_by_name.items():
            revolute = UsdPhysics.RevoluteJoint(prim)
            body0 = list(revolute.GetBody0Rel().GetTargets())
            body1 = list(revolute.GetBody1Rel().GetTargets())
            if len(body0) != 1 or len(body1) != 1:
                raise ActuatorHandModelError(
                    f"Joint {joint_name} must have exactly one body0 and one body1 target; "
                    f"got body0={body0}, body1={body1}"
                )
            parent_path, child_path = str(body0[0]), str(body1[0])
            if parent_path not in path_to_link or child_path not in path_to_link:
                raise ActuatorHandModelError(
                    f"Joint {joint_name} endpoint is not a configured rigid body: "
                    f"body0={parent_path}, body1={child_path}"
                )
            parent_link, child_link = path_to_link[parent_path], path_to_link[child_path]
            if child_link in incoming_children:
                raise ActuatorHandModelError(f"Rigid body {child_link} has multiple revolute parents")
            incoming_children.add(child_link)
            axis_token = str(revolute.GetAxisAttr().Get()).upper()
            axis_by_token = {
                "X": np.array([1.0, 0.0, 0.0], dtype=np.float64),
                "Y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
                "Z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
            }
            if axis_token not in axis_by_token:
                raise ActuatorHandModelError(
                    f"Unsupported axis {axis_token!r} on USD joint {joint_name}"
                )
            parent_frame = _pose_matrix_from_usd(
                revolute.GetLocalPos0Attr().Get(), revolute.GetLocalRot0Attr().Get(), Gf
            )
            child_frame = _pose_matrix_from_usd(
                revolute.GetLocalPos1Attr().Get(), revolute.GetLocalRot1Attr().Get(), Gf
            )
            raw_specs[joint_name] = _JointSpec(
                name=joint_name,
                parent_link=parent_link,
                child_link=child_link,
                full_state_index=self._joint_index[joint_name],
                axis=axis_by_token[axis_token],
                parent_joint_frame=parent_frame,
                child_joint_frame_inverse=np.linalg.inv(child_frame),
            )

        expected_children = expected_link_set - {self.palm_link_name}
        if incoming_children != expected_children:
            raise ActuatorHandModelError(
                "Revolute graph does not cover exactly the 16 non-palm links: "
                f"missing={sorted(expected_children - incoming_children)}, "
                f"extra={sorted(incoming_children - expected_children)}"
            )

        children_by_parent: dict[str, list[_JointSpec]] = {}
        for spec in raw_specs.values():
            children_by_parent.setdefault(spec.parent_link, []).append(spec)
        joint_specs: list[_JointSpec] = []
        visited_links = {self.palm_link_name}
        frontier = [self.palm_link_name]
        while frontier:
            parent = frontier.pop(0)
            children = sorted(
                children_by_parent.get(parent, []), key=lambda item: item.full_state_index
            )
            for spec in children:
                if spec.child_link in visited_links:
                    raise ActuatorHandModelError(
                        f"Cycle found in revolute graph at {spec.child_link}"
                    )
                joint_specs.append(spec)
                visited_links.add(spec.child_link)
                frontier.append(spec.child_link)
        if visited_links != expected_link_set or len(joint_specs) != self.EXPECTED_JOINT_COUNT:
            raise ActuatorHandModelError(
                "Revolute graph is not a 17-link tree rooted at the configured palm: "
                f"visited={sorted(visited_links)}"
            )

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        palm_world = cache.GetLocalToWorldTransform(rigid_by_name[self.palm_link_name])
        world_to_palm = palm_world.GetInverse()
        rest_link_transforms: dict[str, np.ndarray] = {}
        for name in self.link_names:
            link_to_world = cache.GetLocalToWorldTransform(rigid_by_name[name])
            rest_link_transforms[name] = _readonly(
                _gf_matrix_to_column(link_to_world * world_to_palm)
            )

        collision_prims = [
            prim
            for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
            if (
                prim.IsA(UsdGeom.Mesh)
                and prim.HasAPI(UsdPhysics.CollisionAPI)
                and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                is not False
            )
        ]
        if len(collision_prims) != self.EXPECTED_COLLISION_MESH_COUNT:
            raise ActuatorHandModelError(
                f"Expected 17 collision meshes (including instance proxies), found "
                f"{len(collision_prims)}: {[str(prim.GetPath()) for prim in collision_prims]}"
            )
        collision_meshes: dict[str, CollisionMesh] = {}
        for prim in collision_prims:
            owner = prim
            while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            if not owner:
                raise ActuatorHandModelError(
                    f"Collision mesh has no rigid-body ancestor: {prim.GetPath()}"
                )
            owner_path = str(owner.GetPath())
            if owner_path not in path_to_link:
                raise ActuatorHandModelError(
                    f"Collision mesh owner is not a configured rigid body: {owner_path}"
                )
            link_name = path_to_link[owner_path]
            if link_name in collision_meshes:
                raise ActuatorHandModelError(
                    f"Expected exactly one collision mesh per rigid body; {link_name} has at least two"
                )
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
                raise ActuatorHandModelError(
                    f"Collision mesh {prim.GetPath()} has invalid points shape {points.shape}"
                )
            mesh_to_world = cache.GetLocalToWorldTransform(prim)
            world_to_owner = cache.GetLocalToWorldTransform(owner).GetInverse()
            mesh_to_owner = _gf_matrix_to_column(mesh_to_world * world_to_owner)
            vertices_local = (
                points @ mesh_to_owner[:3, :3].T + mesh_to_owner[:3, 3][None, :]
            )
            if not np.isfinite(vertices_local).all():
                raise ActuatorHandModelError(
                    f"Collision mesh {prim.GetPath()} produced non-finite link-local vertices"
                )
            mesh = UsdGeom.Mesh(prim)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
            if (
                counts.ndim != 1
                or indices.ndim != 1
                or len(counts) == 0
                or int(counts.sum()) != len(indices)
                or np.any(counts < 3)
                or np.any(indices < 0)
                or np.any(indices >= len(vertices_local))
            ):
                raise ActuatorHandModelError(
                    f"Collision mesh {prim.GetPath()} has invalid polygon topology"
                )
            if bool(np.all(counts == 3)):
                triangles = indices.reshape(-1, 3).copy()
            else:
                triangle_count = int(np.sum(counts - 2))
                triangles = np.empty((triangle_count, 3), dtype=np.int64)
                source_cursor = 0
                output_cursor = 0
                for count in counts.tolist():
                    face = indices[source_cursor : source_cursor + count]
                    source_cursor += count
                    for face_index in range(1, count - 1):
                        triangles[output_cursor] = (
                            int(face[0]),
                            int(face[face_index]),
                            int(face[face_index + 1]),
                        )
                        output_cursor += 1
            authored_vertex_count = len(vertices_local)
            authored_triangle_count = len(triangles)
            approximation = str(
                UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            )
            if approximation != "convexHull":
                raise ActuatorHandModelError(
                    f"Collision mesh {prim.GetPath()} must use the PhysX convexHull "
                    f"approximation, got {approximation!r}"
                )

            # Each active collider is the explicit, at-most-64-vertex proxy
            # authored by build_x2_physx_collision_hulls.py.  Reconstruct its
            # convex surface once here so the differentiable optimizer and
            # PhysX cook the same vertex set instead of independently reducing
            # the much denser render mesh.
            try:
                from scipy.spatial import ConvexHull, QhullError

                hull = ConvexHull(vertices_local)
            except (ImportError, QhullError) as exc:
                raise ActuatorHandModelError(
                    f"Could not reconstruct convexHull collider {prim.GetPath()}: {exc}"
                ) from exc
            hull_triangles = np.asarray(hull.simplices, dtype=np.int64).copy()
            hull_normals = np.asarray(hull.equations[:, :3], dtype=np.float64)
            triangle_vertices = vertices_local[hull_triangles]
            triangle_normals = np.cross(
                triangle_vertices[:, 1] - triangle_vertices[:, 0],
                triangle_vertices[:, 2] - triangle_vertices[:, 0],
            )
            inward = np.einsum("ij,ij->i", triangle_normals, hull_normals) < 0.0
            hull_triangles[inward, 1], hull_triangles[inward, 2] = (
                hull_triangles[inward, 2].copy(),
                hull_triangles[inward, 1].copy(),
            )
            used_vertices = np.unique(hull_triangles)
            remap = np.full(len(vertices_local), -1, dtype=np.int64)
            remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
            vertices_local = vertices_local[used_vertices]
            triangles = remap[hull_triangles]
            collision_meshes[link_name] = CollisionMesh(
                link_name=link_name,
                prim_path=str(prim.GetPath()),
                approximation=approximation,
                authored_vertex_count=authored_vertex_count,
                authored_triangle_count=authored_triangle_count,
                vertices_local=_readonly(vertices_local),
                triangles=_readonly(triangles),
                plane_normals_local=_readonly(hull_normals.copy()),
                plane_offsets_local=_readonly(
                    np.asarray(hull.equations[:, 3], dtype=np.float64).copy()
                ),
            )
        if set(collision_meshes) != expected_link_set:
            raise ActuatorHandModelError(
                "Collision meshes do not cover all 17 rigid bodies: "
                f"missing={sorted(expected_link_set - set(collision_meshes))}, "
                f"extra={sorted(set(collision_meshes) - expected_link_set)}"
            )
        missing_thumb = sorted(set(self.thumb_links) - set(collision_meshes))
        if missing_thumb:
            raise ActuatorHandModelError(
                f"Thumb collision geometry is mandatory but missing: {missing_thumb}"
            )

        return {
            "joint_specs": tuple(joint_specs),
            "joint_limits": joint_limits,
            "rigid_body_paths": rigid_body_paths,
            "collision_meshes": collision_meshes,
            "rest_link_transforms": rest_link_transforms,
        }

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ActuatorHandModel":
        """Move model tensors in place and return ``self``."""

        target_device = self._device if device is None else torch.device(device)
        target_dtype = self._dtype if dtype is None else dtype
        if not isinstance(target_dtype, torch.dtype) or not torch.empty(
            (), dtype=target_dtype
        ).is_floating_point():
            raise TypeError(f"dtype must be a floating torch dtype, got {target_dtype!r}")
        floating_names = (
            "actuator_limits_tensor",
            "joint_limits_tensor",
            "_joint_parent_frames_tensor",
            "_joint_child_frames_inverse_tensor",
            "_joint_axes_tensor",
            "_capsule_a_tensor",
            "_capsule_b_tensor",
            "_capsule_radius_tensor",
            "_capsule_enclosure_residual_tensor",
            "_capsule_broadphase_radius_tensor",
        )
        for name in floating_names:
            setattr(self, name, getattr(self, name).to(device=target_device, dtype=target_dtype))
        self._joint_state_indices_tensor = self._joint_state_indices_tensor.to(target_device)
        self._self_collision_pair_indices_tensor = self._self_collision_pair_indices_tensor.to(
            target_device
        )
        self._self_collision_hull_pair_indices_tensor = (
            self._self_collision_hull_pair_indices_tensor.to(target_device)
        )
        self._collision_sample_tensors = {
            name: value.to(device=target_device, dtype=target_dtype)
            for name, value in self._collision_sample_tensors.items()
        }
        self._audit_collision_sample_tensors = {
            name: value.to(device=target_device, dtype=target_dtype)
            for name, value in self._audit_collision_sample_tensors.items()
        }
        self._self_collision_sample_tensors = {
            name: value.to(device=target_device, dtype=target_dtype)
            for name, value in self._self_collision_sample_tensors.items()
        }
        self._collision_plane_normal_tensors = {
            name: value.to(device=target_device, dtype=target_dtype)
            for name, value in self._collision_plane_normal_tensors.items()
        }
        self._collision_plane_offset_tensors = {
            name: value.to(device=target_device, dtype=target_dtype)
            for name, value in self._collision_plane_offset_tensors.items()
        }
        self._device = target_device
        self._dtype = target_dtype
        return self

    @staticmethod
    def _validate_last_dimension(value: Any, expected: int, label: str) -> None:
        shape = tuple(value.shape) if hasattr(value, "shape") else tuple(np.asarray(value).shape)
        if not shape or shape[-1] != expected:
            raise ActuatorHandModelError(
                f"{label} must have shape (..., {expected}), got {shape}"
            )

    def expand_actuators(
        self, actuator_positions: np.ndarray | torch.Tensor, *, check_limits: bool = False
    ) -> np.ndarray | torch.Tensor:
        """Map 12 logical actuators to 16 joints, enforcing ``J1 == J2``.

        The output preserves numpy versus torch, leading batch dimensions,
        floating dtype, torch device, and torch autograd connectivity.
        """

        is_torch = torch.is_tensor(actuator_positions)
        if is_torch:
            values = actuator_positions
            if not values.is_floating_point():
                values = values.to(dtype=self.dtype)
        else:
            values = np.asarray(actuator_positions)
            if not np.issubdtype(values.dtype, np.floating):
                values = values.astype(np.float64)
        self._validate_last_dimension(values, self.EXPECTED_ACTUATOR_COUNT, "actuator_positions")
        if check_limits:
            self.validate_actuator_positions(values)

        outputs: list[Any] = []
        for joint_name in self.full_joint_names:
            if joint_name in self._actuator_index:
                outputs.append(values[..., self._actuator_index[joint_name]])
            else:
                driver, multiplier, offset = self.passive_joint_coupling[joint_name]
                outputs.append(values[..., self._actuator_index[driver]] * multiplier + offset)
        if is_torch:
            return torch.stack(outputs, dim=-1)
        return np.stack(outputs, axis=-1)

    def with_fixed_thumb(
        self, actuator_positions: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Return a 12-D state with all four thumb entries set to configured constants.

        Non-thumb values retain their autograd connection.  This is useful when
        constructing a full actuator state from an existing seed before the
        optimizer scatters its eight active parameters into that seed.
        """

        is_torch = torch.is_tensor(actuator_positions)
        if is_torch:
            values = actuator_positions
            if not values.is_floating_point():
                values = values.to(dtype=self.dtype)
        else:
            values = np.asarray(actuator_positions)
            if not np.issubdtype(values.dtype, np.floating):
                values = values.astype(np.float64)
        self._validate_last_dimension(values, self.EXPECTED_ACTUATOR_COUNT, "actuator_positions")

        outputs: list[Any] = []
        for index, name in enumerate(self.actuator_names):
            if name in self.fixed_thumb_position:
                fixed = self.fixed_thumb_position[name]
                if is_torch:
                    outputs.append(torch.full_like(values[..., index], fixed))
                else:
                    outputs.append(np.full(values.shape[:-1], fixed, dtype=values.dtype))
            else:
                outputs.append(values[..., index])
        return torch.stack(outputs, dim=-1) if is_torch else np.stack(outputs, axis=-1)

    def validate_actuator_positions(self, actuator_positions: np.ndarray | torch.Tensor) -> None:
        """Raise an explicit error if a 12-D actuator state is non-finite or out of bounds."""

        self._validate_last_dimension(
            actuator_positions, self.EXPECTED_ACTUATOR_COUNT, "actuator_positions"
        )
        if torch.is_tensor(actuator_positions):
            values = actuator_positions
            limits = self.actuator_limits_tensor.to(device=values.device, dtype=values.dtype)
            finite = bool(torch.isfinite(values).all().detach().cpu())
            valid = bool(
                ((values >= limits[..., 0]) & (values <= limits[..., 1]))
                .all()
                .detach()
                .cpu()
            )
        else:
            values = np.asarray(actuator_positions)
            limits = self.actuator_limits_tensor.detach().cpu().numpy()
            finite = bool(np.isfinite(values).all())
            valid = bool(((values >= limits[:, 0]) & (values <= limits[:, 1])).all())
        if not finite:
            raise ActuatorHandModelError("actuator_positions contains NaN or infinity")
        if not valid:
            raise ActuatorHandModelError(
                "actuator_positions violates configured limits in actuator_names order"
            )

    def _as_joint_tensor(self, joint_positions: np.ndarray | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(joint_positions):
            values = joint_positions
            if not values.is_floating_point():
                values = values.to(dtype=self.dtype)
        else:
            values = torch.as_tensor(joint_positions, device=self.device, dtype=self.dtype)
        self._validate_last_dimension(values, self.EXPECTED_JOINT_COUNT, "joint_positions")
        return values

    @staticmethod
    def _axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Rodrigues rotation for a fixed unit axis and arbitrary angle batch."""

        axis = axis.to(device=angle.device, dtype=angle.dtype)
        x, y, z = axis.unbind()
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        one_minus_cosine = 1.0 - cosine
        row0 = torch.stack(
            (
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ),
            dim=-1,
        )
        row1 = torch.stack(
            (
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ),
            dim=-1,
        )
        row2 = torch.stack(
            (
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ),
            dim=-1,
        )
        return torch.stack((row0, row1, row2), dim=-2)

    def forward_kinematics(
        self, joint_positions: np.ndarray | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return differentiable palm-relative transforms for all 17 rigid links.

        ``joint_positions`` is in ``full_joint_names`` order and radians.  Each
        returned matrix has shape ``(..., 4, 4)`` and maps link-local column
        vectors into the configured palm frame.
        """

        values = self._as_joint_tensor(joint_positions)
        batch_shape = values.shape[:-1]
        identity = torch.eye(4, device=values.device, dtype=values.dtype)
        transforms: dict[str, torch.Tensor] = {
            self.palm_link_name: identity.expand(*batch_shape, 4, 4)
        }
        parent_frames = self._joint_parent_frames_tensor.to(
            device=values.device, dtype=values.dtype
        )
        child_frame_inverses = self._joint_child_frames_inverse_tensor.to(
            device=values.device, dtype=values.dtype
        )
        axes = self._joint_axes_tensor.to(device=values.device, dtype=values.dtype)
        for spec_index, spec in enumerate(self._joint_specs):
            angle = values[..., spec.full_state_index]
            rotation3 = self._axis_angle_matrix(axes[spec_index], angle)
            rotation4 = identity.expand(*batch_shape, 4, 4).clone()
            rotation4[..., :3, :3] = rotation3
            parent_joint = parent_frames[spec_index]
            child_joint_inverse = child_frame_inverses[spec_index]
            transforms[spec.child_link] = (
                transforms[spec.parent_link]
                @ parent_joint
                @ rotation4
                @ child_joint_inverse
            )
        if set(transforms) != set(self.link_names):
            raise ActuatorHandModelError(
                f"Internal FK graph omitted links: {sorted(set(self.link_names) - set(transforms))}"
            )
        return transforms

    def forward_kinematics_from_actuators(
        self, actuator_positions: np.ndarray | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Convenience FK from the logical 12-actuator state."""

        return self.forward_kinematics(self.expand_actuators(actuator_positions))

    def validate_zero_pose_fk(self, *, tolerance: float = 2.0e-5) -> float:
        """Check torch FK at q=0 against OpenUSD ``UsdGeom.XformCache`` transforms."""

        if not math.isfinite(float(tolerance)) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        zero = torch.zeros(
            self.EXPECTED_JOINT_COUNT, device=self.device, dtype=torch.float64
        )
        actual = self.forward_kinematics(zero)
        errors: dict[str, float] = {}
        for name in self.link_names:
            expected = torch.tensor(
                self._rest_link_transforms[name], device=zero.device, dtype=zero.dtype
            )
            errors[name] = float(torch.max(torch.abs(actual[name] - expected)).detach().cpu())
        maximum = max(errors.values(), default=0.0)
        if maximum > tolerance:
            worst = sorted(errors.items(), key=lambda item: item[1], reverse=True)[:5]
            raise ActuatorHandModelError(
                f"Torch FK q=0 disagrees with UsdGeom.XformCache (max={maximum:.3e}, "
                f"tolerance={tolerance:.3e}); worst={worst}"
            )
        return maximum

    @staticmethod
    def _transform_points_with_matrix(
        transform: torch.Tensor, points_local: torch.Tensor
    ) -> torch.Tensor:
        rotation = transform[..., :3, :3]
        translation = transform[..., :3, 3]
        return torch.einsum("...ij,nj->...ni", rotation, points_local) + translation[..., None, :]

    def transform_collision_points(
        self, joint_positions: np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """Transform sampled real collision vertices for all 17 links.

        Palm, four fingers, and all four thumb collision meshes are included.
        """

        values = self._as_joint_tensor(joint_positions)
        transforms = self.forward_kinematics(values)
        result: list[torch.Tensor] = []
        for name in self.link_names:
            local = self._collision_sample_tensors[name].to(
                device=values.device, dtype=values.dtype
            )
            result.append(self._transform_points_with_matrix(transforms[name], local))
        return torch.cat(result, dim=-2)

    def transform_audit_collision_points(
        self, joint_positions: np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """Transform the independent 256/set dense audit surface."""

        values = self._as_joint_tensor(joint_positions)
        transforms = self.forward_kinematics(values)
        result: list[torch.Tensor] = []
        for name in self.link_names:
            local = self._audit_collision_sample_tensors[name].to(
                device=values.device, dtype=values.dtype
            )
            result.append(
                self._transform_points_with_matrix(transforms[name], local)
            )
        return torch.cat(result, dim=-2)

    def _transform_capsules(
        self,
        values: torch.Tensor,
        transforms: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_a = self._capsule_a_tensor.to(device=values.device, dtype=values.dtype)
        local_b = self._capsule_b_tensor.to(device=values.device, dtype=values.dtype)
        point_a: list[torch.Tensor] = []
        point_b: list[torch.Tensor] = []
        for index, name in enumerate(self.link_names):
            transform = transforms[name]
            point_a.append(
                transform[..., :3, :3] @ local_a[index] + transform[..., :3, 3]
            )
            point_b.append(
                transform[..., :3, :3] @ local_b[index] + transform[..., :3, 3]
            )
        return torch.stack(point_a, dim=-2), torch.stack(point_b, dim=-2)

    @staticmethod
    def _point_segment_distance_squared(
        point: torch.Tensor, start: torch.Tensor, end: torch.Tensor
    ) -> torch.Tensor:
        segment = end - start
        denominator = torch.sum(segment * segment, dim=-1).clamp_min(1.0e-18)
        fraction = torch.sum((point - start) * segment, dim=-1) / denominator
        fraction = fraction.clamp(0.0, 1.0)
        closest = start + fraction[..., None] * segment
        return torch.sum((point - closest) ** 2, dim=-1)

    @classmethod
    def _segment_segment_distance(
        cls,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
    ) -> torch.Tensor:
        """Exact segment distance from endpoint and valid interior candidates."""

        candidates = [
            cls._point_segment_distance_squared(a, c, d),
            cls._point_segment_distance_squared(b, c, d),
            cls._point_segment_distance_squared(c, a, b),
            cls._point_segment_distance_squared(d, a, b),
        ]
        u = b - a
        v = d - c
        w = a - c
        uu = torch.sum(u * u, dim=-1)
        uv = torch.sum(u * v, dim=-1)
        vv = torch.sum(v * v, dim=-1)
        uw = torch.sum(u * w, dim=-1)
        vw = torch.sum(v * w, dim=-1)
        denominator = uu * vv - uv * uv
        safe_denominator = torch.where(
            torch.abs(denominator) > 1.0e-18,
            denominator,
            torch.ones_like(denominator),
        )
        first_fraction = (uv * vw - vv * uw) / safe_denominator
        second_fraction = (uu * vw - uv * uw) / safe_denominator
        first_point = a + first_fraction[..., None] * u
        second_point = c + second_fraction[..., None] * v
        interior_distance_sq = torch.sum((first_point - second_point) ** 2, dim=-1)
        valid_interior = (
            (torch.abs(denominator) > 1.0e-18)
            & (first_fraction >= 0.0)
            & (first_fraction <= 1.0)
            & (second_fraction >= 0.0)
            & (second_fraction <= 1.0)
        )
        interior_distance_sq = torch.where(
            valid_interior,
            interior_distance_sq,
            torch.full_like(interior_distance_sq, torch.inf),
        )
        candidates.append(interior_distance_sq)
        minimum_sq = torch.stack(candidates, dim=-1).amin(dim=-1)
        return torch.sqrt(minimum_sq.clamp_min(1.0e-18))

    def self_collision_depths(
        self, joint_positions: np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """Return capsule overlaps in ``self_collision_proxy_pairs`` order.

        Positive values mean capsule overlap; zero is tangency; negative values
        mean separation.  Pairs at kinematic graph distance one or two are
        intentionally omitted because their conservative capsules share a
        structural joint region.  Audited broad-shape false positives may also
        be excluded from this optimization proxy without changing the complete
        PhysX self-contact pair set.
        """

        values = self._as_joint_tensor(joint_positions)
        transforms = self.forward_kinematics(values)
        all_a, all_b = self._transform_capsules(values, transforms)
        pair_indices = self._self_collision_pair_indices_tensor.to(values.device)
        first, second = pair_indices[:, 0], pair_indices[:, 1]
        distance = self._segment_segment_distance(
            all_a.index_select(-2, first),
            all_b.index_select(-2, first),
            all_a.index_select(-2, second),
            all_b.index_select(-2, second),
        )
        radii = self._capsule_radius_tensor.to(device=values.device, dtype=values.dtype)
        return radii.index_select(0, first) + radii.index_select(0, second) - distance

    def _self_collision_hull_broadphase_mask_from_transforms(
        self,
        values: torch.Tensor,
        transforms: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return rows whose conservative inflated capsules can overlap."""

        all_a, all_b = self._transform_capsules(values, transforms)
        pair_indices = self._self_collision_hull_pair_indices_tensor.to(values.device)
        first, second = pair_indices[:, 0], pair_indices[:, 1]
        distance = self._segment_segment_distance(
            all_a.index_select(-2, first),
            all_b.index_select(-2, first),
            all_a.index_select(-2, second),
            all_b.index_select(-2, second),
        )
        radii = self._capsule_broadphase_radius_tensor.to(
            device=values.device, dtype=values.dtype
        )
        maximum_distance = radii.index_select(0, first) + radii.index_select(0, second)
        return distance <= maximum_distance

    def self_collision_hull_broadphase_mask(
        self, joint_positions: np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """Conservative candidate mask in ``self_collision_hull_pairs`` order."""

        values = self._as_joint_tensor(joint_positions)
        transforms = self.forward_kinematics(values)
        return self._self_collision_hull_broadphase_mask_from_transforms(values, transforms)

    def _directional_hull_depth_for_rows(
        self,
        source_name: str,
        target_name: str,
        source_transform: torch.Tensor,
        target_transform: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample source hull points and return positive-inside target depths."""

        source_local = self._self_collision_sample_tensors[source_name].to(
            device=device, dtype=dtype
        )
        source_root = self._transform_points_with_matrix(source_transform, source_local)
        target_local = torch.einsum(
            "bni,bij->bnj",
            source_root - target_transform[:, None, :3, 3],
            target_transform[:, :3, :3],
        )
        normals = self._collision_plane_normal_tensors[target_name].to(
            device=device, dtype=dtype
        )
        offsets = self._collision_plane_offset_tensors[target_name].to(
            device=device, dtype=dtype
        )
        plane_values = torch.einsum("bni,ki->bnk", target_local, normals) + offsets
        return -plane_values.amax(dim=-1)

    def self_collision_hull_signed_depths(
        self,
        joint_positions: np.ndarray | torch.Tensor,
        *,
        use_broadphase: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return bidirectional sampled convex-hull depths for 119 link pairs.

        Both tensors have shape ``(..., 119, 3 * surface_samples_per_link)`` and
        follow :attr:`self_collision_hull_pairs`.  The first tensor samples the
        first link inside the second; the second reverses that direction.
        Positive values are penetration.  The inflated-capsule broadphase is a
        conservative enclosure of every convex hull.  Inactive rows receive a
        finite negative sentinel, so all positive depths remain exact while the
        expensive point-plane query is skipped.
        """

        if not isinstance(use_broadphase, bool):
            raise TypeError("use_broadphase must be boolean")
        values = self._as_joint_tensor(joint_positions)
        transforms = self.forward_kinematics(values)
        batch_shape = tuple(values.shape[:-1])
        batch_count = int(values[..., 0].numel())
        pair_count = len(self.self_collision_hull_pairs)
        sample_count = 3 * self.self_collision_samples_per_link
        if use_broadphase:
            broadphase = self._self_collision_hull_broadphase_mask_from_transforms(
                values, transforms
            )
        else:
            broadphase = torch.ones(
                *batch_shape,
                pair_count,
                device=values.device,
                dtype=torch.bool,
            )
        broadphase_flat = broadphase.reshape(batch_count, pair_count)
        transform_flat = {
            name: transform.reshape(batch_count, 4, 4)
            for name, transform in transforms.items()
        }
        # One device-to-host synchronization identifies active pairs.  Avoid a
        # per-pair nonzero()/dynamic-row query, which is disproportionately slow
        # on CUDA.  An active pair is queried for the whole small grasp batch and
        # then masked per row; wholly inactive pairs skip all point-plane work.
        if use_broadphase:
            pair_active = (
                broadphase_flat.detach().any(dim=0).cpu().tolist()
            )
        else:
            pair_active = [True] * pair_count

        first_to_second: list[torch.Tensor] = []
        second_to_first: list[torch.Tensor] = []
        for pair_index, (first_name, second_name) in enumerate(
            self.self_collision_hull_pairs
        ):
            forward = torch.full(
                (batch_count, sample_count),
                -1.0,
                device=values.device,
                dtype=values.dtype,
            )
            reverse = torch.full_like(forward, -1.0)
            if pair_active[pair_index]:
                first_transform = transform_flat[first_name]
                second_transform = transform_flat[second_name]
                active_forward = self._directional_hull_depth_for_rows(
                    first_name,
                    second_name,
                    first_transform,
                    second_transform,
                    device=values.device,
                    dtype=values.dtype,
                )
                active_reverse = self._directional_hull_depth_for_rows(
                    second_name,
                    first_name,
                    second_transform,
                    first_transform,
                    device=values.device,
                    dtype=values.dtype,
                )
                active_rows = broadphase_flat[:, pair_index, None]
                forward = torch.where(active_rows, active_forward, forward)
                reverse = torch.where(active_rows, active_reverse, reverse)
            first_to_second.append(forward.reshape(*batch_shape, sample_count))
            second_to_first.append(reverse.reshape(*batch_shape, sample_count))
        return (
            torch.stack(first_to_second, dim=-2),
            torch.stack(second_to_first, dim=-2),
        )


__all__ = [
    "ActuatorHandModel",
    "ActuatorHandModelError",
    "CapsuleProxy",
    "CollisionMesh",
    "_capsule_enclosure_residual",
    "_sample_vertices_fixed",
]
