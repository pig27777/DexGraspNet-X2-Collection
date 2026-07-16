"""DexGraspNet-compatible differentiable hand facade for the X2 articulation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import numpy as np
import torch

from .actuator_hand_model import ActuatorHandModel, ActuatorHandModelError
from .rot6d import robust_compute_rotation_matrix_from_ortho6d
from .x2_config import X2Config
from .x2_mesh_contacts import GenericContactCandidate


@dataclass(frozen=True)
class SelfCollisionDiagnostics:
    """Differentiable sampled-hull collision terms for one hand-pose batch.

    ``pair_energy`` includes configured per-pair overrides (currently 2x for
    thumb-index pairs), but intentionally excludes the global ``hull_weight``.
    Penetration fields are raw metres from ``relu(signed_depth)`` and therefore
    do not include the energy's clearance margin.
    """

    pair_names: tuple[tuple[str, str], ...]
    pair_total_penetration: torch.Tensor
    pair_maximum_penetration: torch.Tensor
    total_penetration: torch.Tensor
    maximum_penetration: torch.Tensor
    worst_pair_indices: torch.Tensor
    feasible: torch.Tensor
    threshold: float
    pair_energy: torch.Tensor
    surface_samples_per_link: int

    @property
    def energy(self) -> torch.Tensor:
        return self.pair_energy.sum(dim=-1)

    # Short aliases keep tensor-oriented optimizer code concise.
    @property
    def pair_sum(self) -> torch.Tensor:
        return self.pair_total_penetration

    @property
    def pair_max(self) -> torch.Tensor:
        return self.pair_maximum_penetration

    @property
    def total(self) -> torch.Tensor:
        return self.total_penetration

    @property
    def maximum(self) -> torch.Tensor:
        return self.maximum_penetration

    @cached_property
    def worst_pair(self) -> tuple[tuple[str, str] | None, ...]:
        """Resolve names lazily so optimization does not synchronize every step."""

        packed = torch.stack(
            (self.worst_pair_indices, (self.maximum_penetration > 0.0).to(torch.long)),
            dim=-1,
        )
        return tuple(
            self.pair_names[int(pair_index)] if int(has_penetration) else None
            for pair_index, has_penetration in packed.detach().reshape(-1, 2).cpu().tolist()
        )

    def worst_pair_names(self, index: int) -> tuple[str, str] | None:
        return self.worst_pair[index]

    def detached(self) -> "SelfCollisionDiagnostics":
        """Return an immutable snapshot safe to retain as a checkpoint."""

        return SelfCollisionDiagnostics(
            pair_names=self.pair_names,
            pair_total_penetration=self.pair_total_penetration.detach().clone(),
            pair_maximum_penetration=self.pair_maximum_penetration.detach().clone(),
            total_penetration=self.total_penetration.detach().clone(),
            maximum_penetration=self.maximum_penetration.detach().clone(),
            worst_pair_indices=self.worst_pair_indices.detach().clone(),
            feasible=self.feasible.detach().clone(),
            threshold=self.threshold,
            pair_energy=self.pair_energy.detach().clone(),
            surface_samples_per_link=self.surface_samples_per_link,
        )


class X2HandModel:
    """One X2 FK/collision model shared by front and back grasp conditions.

    A batch pose is ``translation(3) + rotation6d(6) + actuator(12)``.  The
    actuator state is expanded through the verified 12-to-16 follower mapping
    before every FK call.  No mirrored hand or side-specific articulation is
    constructed.
    """

    POSE_DIMENSION = 21

    def __init__(
        self,
        config: X2Config,
        candidates: Sequence[GenericContactCandidate],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        collision_samples_per_link: int = 24,
        audit_collision_samples_per_link: int = 256,
        self_collision_samples_per_link: int | None = None,
        freeze_thumb: bool = False,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.freeze_thumb = bool(freeze_thumb)
        self.backend = ActuatorHandModel(
            config,
            device=self.device,
            dtype=dtype,
            collision_samples_per_link=collision_samples_per_link,
            audit_collision_samples_per_link=(
                audit_collision_samples_per_link
            ),
            self_collision_samples_per_link=self_collision_samples_per_link,
        )
        self.candidates = tuple(candidates)
        if not self.candidates:
            raise ActuatorHandModelError("X2 mesh contact candidate pool is empty")
        unknown = sorted({point.link_name for point in self.candidates} - set(self.backend.link_names))
        if unknown:
            raise ActuatorHandModelError(f"Contact metadata references unknown links: {unknown}")

        self.actuator_names = self.backend.actuator_names
        self.full_joint_names = self.backend.full_joint_names
        self.thumb_actuator_names = self.backend.thumb_actuator_names
        self.n_dofs = 12
        self.n_contact_candidates = len(self.candidates)
        self.joints_lower = self.backend.actuator_limits_tensor[:, 0]
        self.joints_upper = self.backend.actuator_limits_tensor[:, 1]
        self.self_collision_capsule_weight = float(
            self.config.require("self_collision.capsule_weight")
        )
        self.self_collision_hull_weight = float(
            self.config.require("self_collision.hull_weight")
        )
        self.self_collision_clearance_margin = float(
            self.config.require("self_collision.clearance_margin")
        )
        self.self_collision_smoothness = float(
            self.config.require("self_collision.smoothness")
        )
        self.self_collision_feasibility_threshold = float(
            self.config.require("self_collision.feasibility_threshold")
        )
        self._self_collision_pair_weights = torch.tensor(
            self.backend.self_collision_hull_pair_weights,
            device=self.device,
            dtype=self.dtype,
        )

        self.front_grasp_frame = self._matrix("palm_sides.front.grasp_frame")
        self.back_grasp_frame = self._matrix("palm_sides.back.grasp_frame")
        self.front_palm_normal = self._vector("palm_sides.front.normal_local")
        self.back_palm_normal = self._vector("palm_sides.back.normal_local")
        dot = torch.dot(self.front_palm_normal, self.back_palm_normal)
        if float(dot.detach().cpu()) > -0.999:
            raise ActuatorHandModelError("front/back palm normals must be antiparallel")
        self.palm_centers = {
            side: self._vector(f"palm_sides.{side}.center_local")
            for side in ("front", "back")
        }

        finger_links = self.config.require("robot.finger_links")
        self.finger_names = tuple(str(name) for name in finger_links) + ("thumb",)
        self.finger_distal_links = tuple(
            str(finger_links[name][-1]) for name in finger_links
        ) + (str(self.backend.thumb_links[-1]),)
        if len(self.finger_names) != 5 or set(self.finger_names) != {
            "index",
            "middle",
            "ring",
            "little",
            "thumb",
        }:
            raise ActuatorHandModelError(
                "robot.finger_links plus thumb must identify the five X2 fingers"
            )
        self._finger_distal_centers_local = tuple(
            torch.tensor(
                np.array(
                    self.backend.collision_meshes[link_name].vertices_local,
                    copy=True,
                ).mean(axis=0),
                device=self.device,
                dtype=self.dtype,
            )
            for link_name in self.finger_distal_links
        )
        canonical_actuators = torch.tensor(
            self.config.require("initialization.canonical_open_actuator"),
            device=self.device,
            dtype=self.dtype,
        )
        canonical_actuators = self._materialize_actuators(canonical_actuators)
        canonical_status = self.backend.forward_kinematics(
            self.backend.expand_actuators(canonical_actuators)
        )
        self.canonical_finger_distal_points_root = torch.stack(
            tuple(
                canonical_status[link_name][:3, :3] @ local_center
                + canonical_status[link_name][:3, 3]
                for link_name, local_center in zip(
                    self.finger_distal_links, self._finger_distal_centers_local
                )
            ),
            dim=0,
        ).detach()

        self._candidate_positions = torch.tensor(
            [point.local_position for point in self.candidates],
            device=self.device,
            dtype=dtype,
        )
        self._candidate_normals = torch.tensor(
            [point.local_surface_normal for point in self.candidates],
            device=self.device,
            dtype=dtype,
        )
        link_index = {name: index for index, name in enumerate(self.backend.link_names)}
        self._candidate_link_indices = torch.tensor(
            [link_index[point.link_name] for point in self.candidates],
            device=self.device,
            dtype=torch.long,
        )
        finger_index = {name: index for index, name in enumerate(self.finger_names)}
        unknown_fingers = sorted(
            {point.finger_name for point in self.candidates}
            - set(self.finger_names)
            - {"palm"}
        )
        if unknown_fingers:
            raise ActuatorHandModelError(
                f"Contact metadata references unknown fingers: {unknown_fingers}"
            )
        self._candidate_finger_indices = torch.tensor(
            [finger_index.get(point.finger_name, -1) for point in self.candidates],
            device=self.device,
            dtype=torch.long,
        )

        self.hand_pose: torch.Tensor | None = None
        self.contact_point_indices: torch.Tensor | None = None
        self.global_translation: torch.Tensor | None = None
        self.global_rotation: torch.Tensor | None = None
        self.actuator_positions: torch.Tensor | None = None
        self.joint_positions: torch.Tensor | None = None
        self.current_status: dict[str, torch.Tensor] | None = None
        self.contact_points: torch.Tensor | None = None
        self.contact_normals: torch.Tensor | None = None
        self._self_collision_diagnostics_cache: SelfCollisionDiagnostics | None = None

    def _vector(self, key: str) -> torch.Tensor:
        value = torch.tensor(self.config.require(key), device=self.device, dtype=self.dtype)
        if value.shape != (3,) or not bool(torch.isfinite(value).all()):
            raise ActuatorHandModelError(f"{key} must contain three finite values")
        return value

    def _matrix(self, key: str) -> torch.Tensor:
        value = torch.tensor(self.config.require(key), device=self.device, dtype=self.dtype)
        if value.shape != (3, 3) or not bool(torch.isfinite(value).all()):
            raise ActuatorHandModelError(f"{key} must be a finite 3x3 matrix")
        identity = torch.eye(3, device=self.device, dtype=self.dtype)
        if not torch.allclose(value.transpose(0, 1) @ value, identity, atol=1.0e-8, rtol=0.0):
            raise ActuatorHandModelError(f"{key} must be orthonormal")
        if float(torch.det(value).detach().cpu()) < 0.999:
            raise ActuatorHandModelError(f"{key} must be right handed")
        return value

    def _materialize_actuators(self, raw: torch.Tensor) -> torch.Tensor:
        if not self.freeze_thumb:
            return raw
        return self.backend.with_fixed_thumb(raw)

    def set_parameters(
        self, hand_pose: torch.Tensor, contact_point_indices: torch.Tensor | None = None
    ) -> None:
        self._self_collision_diagnostics_cache = None
        if hand_pose.ndim != 2 or hand_pose.shape[1] != self.POSE_DIMENSION:
            raise ActuatorHandModelError(
                f"hand_pose must have shape (B,{self.POSE_DIMENSION}), got {tuple(hand_pose.shape)}"
            )
        if not hand_pose.is_floating_point():
            raise ActuatorHandModelError("hand_pose must be floating point")
        self.hand_pose = hand_pose
        if hand_pose.requires_grad:
            hand_pose.retain_grad()
        self.global_translation = hand_pose[:, :3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(hand_pose[:, 3:9])
        self.actuator_positions = self._materialize_actuators(hand_pose[:, 9:])
        self.joint_positions = self.backend.expand_actuators(self.actuator_positions)
        self.current_status = self.backend.forward_kinematics(self.joint_positions)
        if contact_point_indices is not None:
            if (
                contact_point_indices.ndim != 2
                or contact_point_indices.shape[0] != hand_pose.shape[0]
                or contact_point_indices.dtype != torch.long
            ):
                raise ActuatorHandModelError("contact_point_indices must be a (B,N) long tensor")
            if contact_point_indices.numel() and (
                int(contact_point_indices.min()) < 0
                or int(contact_point_indices.max()) >= self.n_contact_candidates
            ):
                raise ActuatorHandModelError("contact_point_indices contains an invalid candidate")
            self.contact_point_indices = contact_point_indices
            local_points = self._candidate_positions.to(hand_pose)[contact_point_indices]
            local_normals = self._candidate_normals.to(hand_pose)[contact_point_indices]
            link_indices = self._candidate_link_indices.to(hand_pose.device)[contact_point_indices]
            batch_size, n_contact = contact_point_indices.shape
            palm_points = torch.empty(
                batch_size, n_contact, 3, device=hand_pose.device, dtype=hand_pose.dtype
            )
            palm_normals = torch.empty_like(palm_points)
            for index, link_name in enumerate(self.backend.link_names):
                mask = link_indices == index
                if not bool(mask.any()):
                    continue
                transform = self.current_status[link_name]
                rotation = transform[:, :3, :3].unsqueeze(1).expand(-1, n_contact, -1, -1)
                translation = transform[:, :3, 3].unsqueeze(1).expand(-1, n_contact, -1)
                transformed_points = torch.einsum("bnij,bnj->bni", rotation, local_points) + translation
                transformed_normals = torch.einsum("bnij,bnj->bni", rotation, local_normals)
                palm_points[mask] = transformed_points[mask]
                palm_normals[mask] = transformed_normals[mask]
            self.contact_points = (
                torch.einsum("bij,bnj->bni", self.global_rotation, palm_points)
                + self.global_translation[:, None, :]
            )
            self.contact_normals = torch.nn.functional.normalize(
                torch.einsum("bij,bnj->bni", self.global_rotation, palm_normals), dim=-1
            )

    def finger_distal_points_root(self) -> torch.Tensor:
        """Return one real-FK distal collision-hull center per X2 finger.

        The points are expressed in the shared hand root frame.  They are used
        to measure front/back bending without assuming an actuator sign.
        """

        if self.current_status is None or self.hand_pose is None:
            raise ActuatorHandModelError("set_parameters must be called first")
        points = []
        for link_name, local_center in zip(
            self.finger_distal_links, self._finger_distal_centers_local
        ):
            transform = self.current_status[link_name]
            center = local_center.to(self.hand_pose)
            points.append(
                torch.einsum("bij,j->bi", transform[:, :3, :3], center)
                + transform[:, :3, 3]
            )
        return torch.stack(points, dim=1)

    def selected_finger_mask(self) -> torch.Tensor:
        """Return ``(B,5)`` mask for fingers represented by selected contacts."""

        if self.contact_point_indices is None or self.hand_pose is None:
            raise ActuatorHandModelError("Contact selections have not been materialized")
        selected_finger_indices = self._candidate_finger_indices.to(
            self.contact_point_indices.device
        )[self.contact_point_indices]
        return torch.stack(
            tuple(
                (selected_finger_indices == index).any(dim=1)
                for index in range(len(self.finger_names))
            ),
            dim=1,
        )

    def collision_points_world(self) -> torch.Tensor:
        if self.joint_positions is None or self.global_rotation is None or self.global_translation is None:
            raise ActuatorHandModelError("set_parameters must be called first")
        palm_points = self.backend.transform_collision_points(self.joint_positions)
        return (
            torch.einsum("bij,bnj->bni", self.global_rotation, palm_points)
            + self.global_translation[:, None, :]
        )

    def audit_collision_points_world(
        self, row_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the independent dense 256/set hand collision surface."""

        if (
            self.joint_positions is None
            or self.global_rotation is None
            or self.global_translation is None
        ):
            raise ActuatorHandModelError("set_parameters must be called first")
        joints = self.joint_positions
        rotation = self.global_rotation
        translation = self.global_translation
        if row_indices is not None:
            if row_indices.ndim != 1 or row_indices.dtype != torch.long:
                raise ActuatorHandModelError(
                    "row_indices must be a one-dimensional long tensor"
                )
            joints = joints.index_select(0, row_indices)
            rotation = rotation.index_select(0, row_indices)
            translation = translation.index_select(0, row_indices)
        root_points = self.backend.transform_audit_collision_points(joints)
        return (
            torch.einsum("bij,bnj->bni", rotation, root_points)
            + translation[:, None, :]
        )

    def cal_distance(
        self,
        world_points: torch.Tensor,
        row_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Signed depth against the union of 17 real convex collision hulls.

        Positive values are inside the hand.  Only the positive part is used by
        DexGraspNet reverse penetration, for which convex plane slack is exact.
        """

        if self.current_status is None or self.global_rotation is None or self.global_translation is None:
            raise ActuatorHandModelError("set_parameters must be called first")
        if row_indices is None:
            expected_batch = self.hand_pose.shape[0]
            global_translation = self.global_translation
            global_rotation = self.global_rotation
            status = self.current_status
        else:
            if row_indices.ndim != 1 or row_indices.dtype != torch.long:
                raise ActuatorHandModelError(
                    "row_indices must be a one-dimensional long tensor"
                )
            expected_batch = len(row_indices)
            global_translation = self.global_translation.index_select(
                0, row_indices
            )
            global_rotation = self.global_rotation.index_select(0, row_indices)
            status = {
                name: value.index_select(0, row_indices)
                for name, value in self.current_status.items()
            }
        if world_points.ndim != 3 or world_points.shape[0] != expected_batch or world_points.shape[-1] != 3:
            raise ActuatorHandModelError("world_points must have shape (B,N,3)")
        root_points = torch.einsum(
            "bni,bij->bnj",
            world_points - global_translation[:, None, :],
            global_rotation,
        )
        depths: list[torch.Tensor] = []
        for link_name in self.backend.link_names:
            transform = status[link_name]
            local = torch.einsum(
                "bni,bij->bnj", root_points - transform[:, None, :3, 3], transform[:, :3, :3]
            )
            normals = torch.tensor(
                np.array(
                    self.backend.collision_meshes[link_name].plane_normals_local,
                    copy=True,
                ),
                device=world_points.device,
                dtype=world_points.dtype,
            )
            offsets = torch.tensor(
                np.array(
                    self.backend.collision_meshes[link_name].plane_offsets_local,
                    copy=True,
                ),
                device=world_points.device,
                dtype=world_points.dtype,
            )
            plane_values = torch.einsum("bni,ki->bnk", local, normals) + offsets
            depths.append(-plane_values.amax(dim=-1))
        return torch.stack(depths, dim=0).amax(dim=0)

    def self_penetration(self) -> torch.Tensor:
        """Legacy capsule overlap sum (the historical ``E_spen`` semantics)."""

        if self.joint_positions is None:
            raise ActuatorHandModelError("set_parameters must be called first")
        return torch.relu(self.backend.self_collision_depths(self.joint_positions)).sum(dim=-1)

    def self_collision_diagnostics(self) -> SelfCollisionDiagnostics:
        """Evaluate bidirectional sampled-hull energy and physical penetration."""

        if self.joint_positions is None:
            raise ActuatorHandModelError("set_parameters must be called first")
        if self._self_collision_diagnostics_cache is not None:
            return self._self_collision_diagnostics_cache
        first_depths, second_depths = self.backend.self_collision_hull_signed_depths(
            self.joint_positions
        )
        first_penetration = torch.relu(first_depths)
        second_penetration = torch.relu(second_depths)
        pair_total = first_penetration.sum(dim=-1) + second_penetration.sum(dim=-1)
        pair_maximum = torch.maximum(
            first_penetration.amax(dim=-1), second_penetration.amax(dim=-1)
        )
        total = pair_total.sum(dim=-1)
        maximum, worst_indices = pair_maximum.max(dim=-1)
        threshold = self.self_collision_feasibility_threshold
        feasible = maximum <= threshold

        smoothness = self.self_collision_smoothness
        clearance = self.self_collision_clearance_margin

        def penalty(depth: torch.Tensor) -> torch.Tensor:
            q = smoothness * torch.nn.functional.softplus(
                (depth + clearance) / smoothness
            )
            return q + q.square() / clearance

        first_penalty = penalty(first_depths)
        second_penalty = penalty(second_depths)
        pair_energy = torch.maximum(
            first_penalty.amax(dim=-1), second_penalty.amax(dim=-1)
        ) + 0.5 * (
            first_penalty.mean(dim=-1) + second_penalty.mean(dim=-1)
        )
        pair_weights = self._self_collision_pair_weights.to(pair_energy)
        pair_energy = pair_energy * pair_weights

        diagnostics = SelfCollisionDiagnostics(
            pair_names=self.backend.self_collision_hull_pairs,
            pair_total_penetration=pair_total,
            pair_maximum_penetration=pair_maximum,
            total_penetration=total,
            maximum_penetration=maximum,
            worst_pair_indices=worst_indices,
            feasible=feasible,
            threshold=threshold,
            pair_energy=pair_energy,
            surface_samples_per_link=self.backend.self_collision_samples_per_link,
        )
        self._self_collision_diagnostics_cache = diagnostics
        return diagnostics

    def self_collision_hull_energy(
        self, diagnostics: SelfCollisionDiagnostics | None = None
    ) -> torch.Tensor:
        """Return unweighted-global hull energy, preserving pair overrides."""

        if diagnostics is None:
            diagnostics = self.self_collision_diagnostics()
        if diagnostics.pair_names != self.backend.self_collision_hull_pairs:
            raise ActuatorHandModelError(
                "SelfCollisionDiagnostics pair order does not match this X2 model"
            )
        return diagnostics.energy


__all__ = ["SelfCollisionDiagnostics", "X2HandModel"]
