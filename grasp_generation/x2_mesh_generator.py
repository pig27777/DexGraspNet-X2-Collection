"""DexGraspNet-style generic mesh grasp generation for the X2 hand."""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import trimesh

from .utils.mesh_object_model import MeshObjectModel
from .utils.x2_config import X2Config
from .utils.x2_hand_model import SelfCollisionDiagnostics, X2HandModel
from .utils.x2_mesh_contacts import (
    FINGER_NAMES,
    GenericContactCandidate,
    GenericDexterousContactPolicy,
)
from .utils.rot6d import robust_compute_rotation_matrix_from_ortho6d


ContactPolicyInput = (
    Mapping[str, GenericDexterousContactPolicy]
    | Sequence[GenericDexterousContactPolicy]
)


# This is the unchanged raw hand-object penetration gate used before PhysX.
# Checkpoints must be strictly below it; equality is not accepted.
HAND_OBJECT_PENETRATION_THRESHOLD = 0.001
# Full hand-surface/object-triangle queries are exact but expensive.  Evaluate
# them periodically for checkpoint selection rather than in every annealing
# energy call.  The final restored pose is always evaluated exactly as well.
BIDIRECTIONAL_CHECKPOINT_PERIOD = 50
BIDIRECTIONAL_CHECKPOINT_TOP_K = 16
DENSE_HAND_SURFACE_SAMPLES_PER_SET = 256
DENSE_OBJECT_SURFACE_SAMPLES = 8192


def _resolve_row_policies(
    policies: ContactPolicyInput,
    active_sides: Sequence[str],
) -> tuple[GenericDexterousContactPolicy, ...]:
    """Normalize legacy side mappings and new batch-aligned policy sequences.

    A side mapping preserves the original public API by expanding
    ``policies[active_sides[row]]``.  A sequence is interpreted as one policy
    per batch row, which permits two rows on the same side to use different
    exact finger masks.  Contact counts still have to be rectangular within an
    optimizer call and are checked by the initialization path.
    """

    sides = tuple(str(side) for side in active_sides)
    if isinstance(policies, Mapping):
        missing = sorted(set(sides) - set(policies))
        if missing:
            raise ValueError(f"Contact policies are missing active sides: {missing}")
        resolved = tuple(policies[side] for side in sides)
    else:
        resolved = tuple(policies)
        if len(resolved) != len(sides):
            raise ValueError(
                "A row policy sequence must contain exactly one policy per "
                f"active side: got {len(resolved)} policies for {len(sides)} rows"
            )
    for row, (side, policy) in enumerate(zip(sides, resolved)):
        if not isinstance(policy, GenericDexterousContactPolicy):
            raise TypeError(
                f"Contact policy for batch row {row} is not a "
                "GenericDexterousContactPolicy"
            )
        if policy.active_side != side:
            raise ValueError(
                f"Contact policy for batch row {row} targets "
                f"{policy.active_side!r}, but active_sides requests {side!r}"
            )
    return resolved


@dataclass(frozen=True)
class MeshEnergy:
    total: torch.Tensor
    E_fc: torch.Tensor
    E_dis: torch.Tensor
    E_pen: torch.Tensor
    reverse_maximum_penetration: torch.Tensor
    E_spen: torch.Tensor
    E_spen_capsule: torch.Tensor
    E_spen_hull: torch.Tensor
    E_joints: torch.Tensor
    E_side: torch.Tensor
    E_unselected_opposite_flex: torch.Tensor
    normal_opposition: torch.Tensor
    normal_opposition_penalty: torch.Tensor
    # Optional because the generic object-only generator remains a supported
    # public API.  When present this is a *generation* penalty, not a motion
    # proof; the sequential runtime still performs exact tabletop
    # admission and continuous-motion certification independently.
    E_table: torch.Tensor | None = None
    # Generation-only excess-distance barrier for contact-conditioned output.
    E_contact: torch.Tensor | None = None
    # Forward hand-surface-in-object penetration complements the historical
    # reverse object-surface-in-hand term.  Both remain source optimization
    # evidence; the independent dense bidirectional gate is authoritative.
    E_forward_pen: torch.Tensor | None = None
    forward_maximum_penetration: torch.Tensor | None = None
    E_hand_object_gate: torch.Tensor | None = None
    # Minimum signed clearance over every calibrated collision-mesh vertex.
    # This is a source-generation diagnostic used to retain useful optimizer
    # states; it is not an exact scene/table proof.
    minimum_table_clearance: torch.Tensor | None = None

    def detached_terms(self, index: int) -> dict[str, float]:
        terms = {
            name: float(getattr(self, name)[index].detach().cpu())
            for name in (
                "E_fc", "E_dis", "E_pen", "E_spen", "E_spen_capsule",
                "E_spen_hull", "E_joints", "E_side", "E_unselected_opposite_flex",
            )
        }
        if self.E_table is not None:
            terms["E_table"] = float(self.E_table[index].detach().cpu())
        if self.E_contact is not None:
            terms["E_contact"] = float(self.E_contact[index].detach().cpu())
        if self.E_forward_pen is not None:
            terms["E_forward_pen"] = float(
                self.E_forward_pen[index].detach().cpu()
            )
        if self.E_hand_object_gate is not None:
            terms["E_hand_object_gate"] = float(
                self.E_hand_object_gate[index].detach().cpu()
            )
        if self.minimum_table_clearance is not None:
            terms["minimum_table_clearance"] = float(
                self.minimum_table_clearance[index].detach().cpu()
            )
        return terms


@dataclass(frozen=True)
class ReachabilityPoseAnchor:
    """A soft root-pose prior from a known arm-reachable hand pose.

    This is deliberately a candidate-generation term only.  It neither
    asserts arm feasibility nor replaces downstream IK/collision
    certification.  A small anchor keeps mesh optimization in the useful
    basin of a reachable wrist configuration instead of sampling arbitrary
    free-space hand poses and discovering arm infeasibility afterward.
    """

    pose: torch.Tensor
    translation_weight: float
    rotation_weight: float
    actuator: torch.Tensor | None = None
    contact_indices: torch.Tensor | None = None


def _with_reachability_anchor(
    energy: MeshEnergy, hand: X2HandModel, anchor: ReachabilityPoseAnchor | None
) -> MeshEnergy:
    if anchor is None:
        return energy
    if hand.global_translation is None or hand.global_rotation is None:
        raise RuntimeError("reachability anchor requires a materialized hand pose")
    target = anchor.pose.to(device=hand.device, dtype=hand.dtype)
    batch = hand.global_translation.shape[0]
    if target.shape != (batch, hand.POSE_DIMENSION):
        raise ValueError(
            "reachability anchor pose must match the optimizer batch and X2 pose dimension"
        )
    if not math.isfinite(anchor.translation_weight) or not math.isfinite(anchor.rotation_weight):
        raise ValueError("reachability anchor weights must be finite")
    if anchor.translation_weight < 0.0 or anchor.rotation_weight < 0.0:
        raise ValueError("reachability anchor weights must be non-negative")
    target_rotation = robust_compute_rotation_matrix_from_ortho6d(target[:, 3:9])
    translation = (hand.global_translation - target[:, :3]).square().sum(dim=-1)
    relative = torch.einsum("bij,bjk->bik", target_rotation.transpose(1, 2), hand.global_rotation)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    rotation = 1.0 - cosine
    penalty = float(anchor.translation_weight) * translation + float(anchor.rotation_weight) * rotation
    return replace(energy, total=energy.total + penalty)

@dataclass(frozen=True)
class TabletopPlaneConstraint:
    """A finite horizontal/oblique table plane used only during generation.

    The constraint is expressed in the same frame as the mesh and generated
    hand pose: a point ``p`` clears the table by
    ``normal.dot(p) - offset_m``.  It intentionally models only the support
    plane.  Table extents, exact link meshes, object contact, and an arm path
    belong to the sequential exact-admission and PlanR/CertifyR layers.
    """

    normal: tuple[float, float, float]
    offset_m: float
    clearance_m: float
    weight: float

    def normalized(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        normal = torch.tensor(self.normal, device=device, dtype=dtype)
        if normal.shape != (3,) or not bool(torch.isfinite(normal).all()):
            raise ValueError("table plane normal must contain three finite values")
        norm = torch.linalg.vector_norm(normal)
        if not bool(torch.isfinite(norm)) or float(norm.detach().cpu()) <= 0.0:
            raise ValueError("table plane normal must be non-zero")
        for name, value in (
            ("offset_m", self.offset_m),
            ("clearance_m", self.clearance_m),
            ("weight", self.weight),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"table plane {name} must be finite")
        if self.clearance_m < 0.0:
            raise ValueError("table plane clearance_m must be non-negative")
        if self.weight <= 0.0:
            raise ValueError("table plane weight must be positive")
        return normal / norm


@dataclass(frozen=True)
class MeshBatchResult:
    initial_pose: torch.Tensor
    final_pose: torch.Tensor
    initial_energy: MeshEnergy
    final_energy: MeshEnergy
    contact_indices: torch.Tensor
    active_sides: tuple[str, ...]
    optimizer_diagnostics: dict[str, Any]
    maximum_penetration: torch.Tensor
    hand_object_penetration: HandObjectPenetrationDiagnostics
    self_collision_diagnostics: SelfCollisionDiagnostics
    checkpoint_type: tuple[str, ...]
    restored_step: torch.Tensor
    actuator_positions: torch.Tensor
    joint_positions: torch.Tensor


@dataclass(frozen=True)
class HandObjectPenetrationDiagnostics:
    """Exact sampled bidirectional hand-object penetration reductions."""

    forward_total_penetration: torch.Tensor
    forward_maximum_penetration: torch.Tensor
    reverse_total_penetration: torch.Tensor
    reverse_maximum_penetration: torch.Tensor
    total_penetration: torch.Tensor
    maximum_penetration: torch.Tensor
    feasible: torch.Tensor
    evaluated: torch.Tensor
    threshold: float = HAND_OBJECT_PENETRATION_THRESHOLD

    def detached(self) -> "HandObjectPenetrationDiagnostics":
        return HandObjectPenetrationDiagnostics(
            **{
                name: value.detach().clone()
                for name, value in vars(self).items()
                if isinstance(value, torch.Tensor)
            },
            threshold=float(self.threshold),
        )


def _detach_mesh_energy(energy: MeshEnergy) -> MeshEnergy:
    return MeshEnergy(
        **{
            name: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for name, value in energy.__dict__.items()
        }
    )


def _batch_tensor_fields(value: Any, batch_size: int) -> dict[str, torch.Tensor]:
    """Return batch tensor references without copying their storage."""

    return {
        name: field
        for name, field in vars(value).items()
        if isinstance(field, torch.Tensor)
        and field.ndim > 0
        and field.shape[0] == batch_size
    }


_SELF_COLLISION_CHECKPOINT_FIELDS = (
    "pair_total_penetration",
    "pair_maximum_penetration",
    "total_penetration",
    "maximum_penetration",
    "worst_pair_indices",
    "feasible",
    "pair_energy",
)

_HAND_OBJECT_CHECKPOINT_FIELDS = (
    "forward_total_penetration",
    "forward_maximum_penetration",
    "reverse_total_penetration",
    "reverse_maximum_penetration",
    "total_penetration",
    "maximum_penetration",
    "feasible",
    "evaluated",
)


def _self_collision_checkpoint_tensors(
    value: SelfCollisionDiagnostics, batch_size: int
) -> dict[str, torch.Tensor]:
    """Return reduced diagnostics needed to rank and verify checkpoints.

    Directional sample depths are deliberately excluded: all finiteness,
    ranking, wire, and restore checks operate on these reductions, while
    retaining both ``B x pairs x samples`` tensors would make each bank much
    larger without adding checkpoint information.
    """

    result: dict[str, torch.Tensor] = {}
    for name in _SELF_COLLISION_CHECKPOINT_FIELDS:
        field = getattr(value, name)
        if (
            not isinstance(field, torch.Tensor)
            or field.ndim == 0
            or field.shape[0] != batch_size
        ):
            raise RuntimeError(
                f"Self-collision diagnostic {name} is not batch-aligned"
            )
        result[name] = field
    return result


def _hand_object_checkpoint_tensors(
    value: HandObjectPenetrationDiagnostics, batch_size: int
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    if float(value.threshold) != HAND_OBJECT_PENETRATION_THRESHOLD:
        raise RuntimeError(
            "Hand-object checkpoint threshold does not match the raw gate"
        )
    for name in _HAND_OBJECT_CHECKPOINT_FIELDS:
        field = getattr(value, name)
        if (
            not isinstance(field, torch.Tensor)
            or field.ndim == 0
            or field.shape[0] != batch_size
        ):
            raise RuntimeError(
                f"Hand-object penetration diagnostic {name} is not batch-aligned"
            )
        result[name] = field
    return result


@dataclass
class _CheckpointBank:
    """Per-row optimizer snapshots; pose/contact are authoritative on restore."""

    valid: torch.Tensor
    pose: torch.Tensor
    contact_indices: torch.Tensor
    actuator_positions: torch.Tensor
    joint_positions: torch.Tensor
    energy: dict[str, torch.Tensor]
    self_collision: dict[str, torch.Tensor]
    hand_object: dict[str, torch.Tensor]
    step: torch.Tensor

    @classmethod
    def empty(
        cls,
        hand: X2HandModel,
        energy: MeshEnergy,
        self_collision: SelfCollisionDiagnostics,
        hand_object: HandObjectPenetrationDiagnostics | None = None,
    ) -> "_CheckpointBank":
        if hand.hand_pose is None or hand.contact_point_indices is None:
            raise RuntimeError("Hand state must be materialized before checkpoint creation")
        if hand.actuator_positions is None or hand.joint_positions is None:
            raise RuntimeError("Actuator and joint state must be materialized")
        batch_size = hand.hand_pose.shape[0]
        return cls(
            valid=torch.zeros(batch_size, device=hand.device, dtype=torch.bool),
            pose=torch.empty_like(hand.hand_pose),
            contact_indices=torch.empty_like(hand.contact_point_indices),
            actuator_positions=torch.empty_like(hand.actuator_positions),
            joint_positions=torch.empty_like(hand.joint_positions),
            energy={
                name: torch.empty_like(value)
                for name, value in _batch_tensor_fields(energy, batch_size).items()
            },
            self_collision={
                name: torch.empty_like(value)
                for name, value in _self_collision_checkpoint_tensors(
                    self_collision, batch_size
                ).items()
            },
            hand_object=(
                {
                    name: torch.empty_like(value)
                    for name, value in _hand_object_checkpoint_tensors(
                        hand_object, batch_size
                    ).items()
                }
                if hand_object is not None
                else {}
            ),
            step=torch.full(
                (batch_size,), -1, device=hand.device, dtype=torch.long
            ),
        )

    def update(
        self,
        mask: torch.Tensor,
        hand: X2HandModel,
        energy: MeshEnergy,
        self_collision: SelfCollisionDiagnostics,
        hand_object: HandObjectPenetrationDiagnostics | None = None,
        *,
        step: int,
    ) -> None:
        if hand.hand_pose is None or hand.contact_point_indices is None:
            raise RuntimeError("Cannot checkpoint an unmaterialized hand")
        if hand.actuator_positions is None or hand.joint_positions is None:
            raise RuntimeError("Cannot checkpoint unmaterialized joint state")
        with torch.no_grad():
            self.valid[mask] = True
            self.pose[mask] = hand.hand_pose.detach()[mask]
            self.contact_indices[mask] = hand.contact_point_indices.detach()[mask]
            self.actuator_positions[mask] = hand.actuator_positions.detach()[mask]
            self.joint_positions[mask] = hand.joint_positions.detach()[mask]
            for name, value in _batch_tensor_fields(
                energy, len(self.valid)
            ).items():
                self.energy[name][mask] = value[mask]
            for name, value in _self_collision_checkpoint_tensors(
                self_collision, len(self.valid)
            ).items():
                self.self_collision[name][mask] = value[mask]
            if self.hand_object:
                if hand_object is None:
                    raise RuntimeError(
                        "This checkpoint bank requires hand-object diagnostics"
                    )
                for name, value in _hand_object_checkpoint_tensors(
                    hand_object, len(self.valid)
                ).items():
                    self.hand_object[name][mask] = value[mask]
            self.step[mask] = int(step)


@dataclass
class _SparseCheckpointPool:
    """Per-row top-K self-feasible candidates with hybrid ranking.

    Half of the slots preserve the lowest-energy grasp candidates and half
    preserve the lowest sparse-penetration candidates.  The final dense strict
    1 mm audit remains authoritative.  This avoids both failure modes of a
    single ranking: penetration-only selection prefers separated poses, while
    energy-only selection can evict every collision-safe fallback.
    """

    valid: torch.Tensor
    pose: torch.Tensor
    contact_indices: torch.Tensor
    energy_total: torch.Tensor
    sparse_total_penetration: torch.Tensor
    sparse_maximum_penetration: torch.Tensor
    step: torch.Tensor

    @classmethod
    def empty(
        cls, hand: X2HandModel, *, capacity: int
    ) -> "_SparseCheckpointPool":
        if capacity <= 0:
            raise ValueError("Checkpoint pool capacity must be positive")
        if hand.hand_pose is None or hand.contact_point_indices is None:
            raise RuntimeError("Hand state must be materialized before pool creation")
        batch_size = hand.hand_pose.shape[0]
        return cls(
            valid=torch.zeros(
                batch_size, capacity, device=hand.device, dtype=torch.bool
            ),
            pose=torch.empty(
                batch_size,
                capacity,
                hand.hand_pose.shape[1],
                device=hand.device,
                dtype=hand.dtype,
            ),
            contact_indices=torch.empty(
                batch_size,
                capacity,
                hand.contact_point_indices.shape[1],
                device=hand.device,
                dtype=torch.long,
            ),
            energy_total=torch.full(
                (batch_size, capacity),
                torch.inf,
                device=hand.device,
                dtype=hand.dtype,
            ),
            sparse_total_penetration=torch.full(
                (batch_size, capacity),
                torch.inf,
                device=hand.device,
                dtype=hand.dtype,
            ),
            sparse_maximum_penetration=torch.full(
                (batch_size, capacity),
                torch.inf,
                device=hand.device,
                dtype=hand.dtype,
            ),
            step=torch.full(
                (batch_size, capacity),
                -1,
                device=hand.device,
                dtype=torch.long,
            ),
        )

    @property
    def capacity(self) -> int:
        return self.valid.shape[1]

    @staticmethod
    def _gather(tensor: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        shape = order.shape + (1,) * (tensor.ndim - 2)
        expanded = order.reshape(shape).expand(
            *order.shape, *tensor.shape[2:]
        )
        return torch.gather(tensor, 1, expanded)

    def update(
        self,
        mask: torch.Tensor,
        hand: X2HandModel,
        energy: MeshEnergy,
        hand_object: HandObjectPenetrationDiagnostics,
        *,
        step: int,
    ) -> None:
        if hand.hand_pose is None or hand.contact_point_indices is None:
            raise RuntimeError("Cannot pool an unmaterialized hand")
        if mask.shape != (len(self.valid),) or mask.dtype != torch.bool:
            raise ValueError("Pool update mask must be one boolean per batch row")
        candidate_valid = (
            mask
            & hand_object.evaluated
            & torch.isfinite(hand_object.total_penetration)
            & torch.isfinite(hand_object.maximum_penetration)
        )
        duplicate = (
            self.valid
            & (self.pose == hand.hand_pose.detach()[:, None, :]).all(dim=-1)
            & (
                self.contact_indices
                == hand.contact_point_indices.detach()[:, None, :]
            ).all(dim=-1)
        ).any(dim=1)
        candidate_valid &= ~duplicate
        infinity = torch.full_like(energy.total.detach(), torch.inf)
        with torch.no_grad():
            combined: dict[str, torch.Tensor] = {
                "valid": torch.cat(
                    (self.valid, candidate_valid[:, None]), dim=1
                ),
                "pose": torch.cat(
                    (self.pose, hand.hand_pose.detach()[:, None, :]), dim=1
                ),
                "contact_indices": torch.cat(
                    (
                        self.contact_indices,
                        hand.contact_point_indices.detach()[:, None, :],
                    ),
                    dim=1,
                ),
                "energy_total": torch.cat(
                    (
                        self.energy_total,
                        torch.where(
                            candidate_valid, energy.total.detach(), infinity
                        )[:, None],
                    ),
                    dim=1,
                ),
                "sparse_total_penetration": torch.cat(
                    (
                        self.sparse_total_penetration,
                        torch.where(
                            candidate_valid,
                            hand_object.total_penetration,
                            infinity,
                        )[:, None],
                    ),
                    dim=1,
                ),
                "sparse_maximum_penetration": torch.cat(
                    (
                        self.sparse_maximum_penetration,
                        torch.where(
                            candidate_valid,
                            hand_object.maximum_penetration,
                            infinity,
                        )[:, None],
                    ),
                    dim=1,
                ),
                "step": torch.cat(
                    (
                        self.step,
                        torch.full(
                            (len(self.valid), 1),
                            int(step),
                            device=self.step.device,
                            dtype=self.step.dtype,
                        ),
                    ),
                    dim=1,
                ),
            }
            batch_size, combined_count = combined["valid"].shape
            origin = torch.arange(
                combined_count, device=self.valid.device, dtype=torch.long
            ).expand(batch_size, -1)

            def ranked(keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
                values = {**combined, "_origin": origin}
                for key in keys:
                    order = torch.argsort(values[key], dim=1, stable=True)
                    values = {
                        name: self._gather(value, order)
                        for name, value in values.items()
                    }
                return values

            # Keys are supplied least-significant to most-significant.
            by_energy = ranked(
                (
                    "sparse_total_penetration",
                    "sparse_maximum_penetration",
                    "energy_total",
                )
            )
            by_penetration = ranked(
                (
                    "energy_total",
                    "sparse_total_penetration",
                    "sparse_maximum_penetration",
                )
            )
            energy_slots = (self.capacity + 1) // 2
            penetration_slots = self.capacity - energy_slots
            energy_origin = by_energy["_origin"][:, :energy_slots]
            if penetration_slots:
                already_selected = (
                    by_penetration["_origin"][:, :, None]
                    == energy_origin[:, None, :]
                ).any(dim=2)
                unique_first = torch.argsort(
                    already_selected.to(torch.int8), dim=1, stable=True
                )
                by_penetration = {
                    name: self._gather(value, unique_first)
                    for name, value in by_penetration.items()
                }
                selected_origin = torch.cat(
                    (
                        energy_origin,
                        by_penetration["_origin"][:, :penetration_slots],
                    ),
                    dim=1,
                )
            else:
                selected_origin = energy_origin
            combined = {
                name: self._gather(value, selected_origin)
                for name, value in combined.items()
            }
            for name in vars(self):
                getattr(self, name).copy_(combined[name])


def _finite_candidate_rows(
    hand: X2HandModel,
    energy: MeshEnergy,
    self_collision: SelfCollisionDiagnostics,
) -> torch.Tensor:
    if hand.hand_pose is None or hand.actuator_positions is None or hand.joint_positions is None:
        raise RuntimeError("Hand state has not been materialized")
    batch_size = hand.hand_pose.shape[0]
    finite = torch.isfinite(hand.hand_pose).reshape(batch_size, -1).all(dim=1)
    finite &= torch.isfinite(hand.actuator_positions).reshape(batch_size, -1).all(dim=1)
    finite &= torch.isfinite(hand.joint_positions).reshape(batch_size, -1).all(dim=1)
    for value in _batch_tensor_fields(energy, batch_size).values():
        finite &= torch.isfinite(value).reshape(batch_size, -1).all(dim=1)
    for value in _self_collision_checkpoint_tensors(
        self_collision, batch_size
    ).values():
        if value.is_floating_point():
            finite &= torch.isfinite(value).reshape(batch_size, -1).all(dim=1)
    return finite


def _valid_checkpoint_rows(
    hand: X2HandModel,
    energy: MeshEnergy,
    self_collision: SelfCollisionDiagnostics,
    policies: ContactPolicyInput,
    active_sides: Sequence[str],
) -> torch.Tensor:
    """Require finite, in-limit materialized states and legal unique contacts."""

    if hand.contact_point_indices is None:
        raise RuntimeError("Hand contacts have not been materialized")
    if hand.actuator_positions is None or hand.joint_positions is None:
        raise RuntimeError("Hand joint state has not been materialized")
    valid = _finite_candidate_rows(hand, energy, self_collision)
    actuator_limits = hand.backend.actuator_limits_tensor.to(hand.actuator_positions)
    joint_limits = hand.backend.joint_limits_tensor.to(hand.joint_positions)
    valid &= (
        (hand.actuator_positions >= actuator_limits[:, 0])
        & (hand.actuator_positions <= actuator_limits[:, 1])
    ).all(dim=1)
    valid &= (
        (hand.joint_positions >= joint_limits[:, 0])
        & (hand.joint_positions <= joint_limits[:, 1])
    ).all(dim=1)
    contacts = hand.contact_point_indices.detach()
    ordered_contacts = contacts.sort(dim=1).values
    unique = (
        ordered_contacts[:, 1:] != ordered_contacts[:, :-1]
    ).all(dim=1)
    legal = unique.clone()
    row_policies = _resolve_row_policies(policies, active_sides)
    if len(row_policies) != contacts.shape[0]:
        raise ValueError(
            "active_sides and row policies must match the materialized hand batch"
        )
    grouped_rows: dict[int, tuple[GenericDexterousContactPolicy, list[int]]] = {}
    for row, policy in enumerate(row_policies):
        key = id(policy)
        if key not in grouped_rows:
            grouped_rows[key] = (policy, [])
        grouped_rows[key][1].append(row)

    # Validate the complete policy contract on-device.  Copying contacts to the
    # host and calling policy.validate row by row here would introduce B CUDA
    # synchronizations for every annealing proposal.
    finger_index = {name: index for index, name in enumerate(FINGER_NAMES)}
    for policy, rows in grouped_rows.values():
        row_indices = torch.tensor(
            rows,
            device=contacts.device,
            dtype=torch.long,
        )
        if contacts.shape[1] != policy.n_contact:
            legal[row_indices] = False
            continue
        if len(policy.candidates) != hand.n_contact_candidates:
            raise ValueError(
                "Every row policy must reference the hand's complete contact "
                "candidate catalog"
            )
        selected = contacts.index_select(0, row_indices)
        eligible = torch.zeros(
            hand.n_contact_candidates, device=contacts.device, dtype=torch.bool
        )
        eligible[
            torch.as_tensor(
                policy.eligible_indices, device=contacts.device, dtype=torch.long
            )
        ] = True
        row_legal = eligible[selected].all(dim=1)
        if policy.target_finger_count is not None:
            candidate_fingers = torch.tensor(
                [
                    finger_index.get(candidate.finger_name, -1)
                    for candidate in policy.candidates
                ],
                device=contacts.device,
                dtype=torch.long,
            )
            selected_fingers = candidate_fingers[selected]
            participation = torch.stack(
                tuple(
                    (selected_fingers == index).any(dim=1)
                    for index in range(len(FINGER_NAMES))
                ),
                dim=1,
            )
            row_legal &= (
                participation.sum(dim=1) == policy.target_finger_count
            )
            if policy.required_finger_names:
                expected = torch.tensor(
                    [name in policy.required_finger_names for name in FINGER_NAMES],
                    device=contacts.device,
                    dtype=torch.bool,
                )
                row_legal &= (participation == expected).all(dim=1)
        legal[row_indices] = legal[row_indices] & row_legal
    return valid & legal


def _hand_object_penetration_diagnostics(
    hand: X2HandModel,
    object_model: MeshObjectModel,
    energy: MeshEnergy,
    evaluate_rows: torch.Tensor,
    *,
    threshold: float = HAND_OBJECT_PENETRATION_THRESHOLD,
) -> HandObjectPenetrationDiagnostics:
    """Evaluate the sparse ranking metric for selected batch rows.

    Reverse object-surface-in-hand reductions are reused from the current
    energy evaluation.  This metric is suitable for periodic top-K ranking,
    not for the final hard gate, because it uses the optimization densities.
    """

    if hand.hand_pose is None:
        raise RuntimeError("Hand state has not been materialized")
    batch_size = hand.hand_pose.shape[0]
    if (
        evaluate_rows.shape != (batch_size,)
        or evaluate_rows.dtype != torch.bool
    ):
        raise ValueError("evaluate_rows must be one boolean per batch row")
    if float(threshold) != HAND_OBJECT_PENETRATION_THRESHOLD:
        raise ValueError(
            "The hand-object feasibility threshold must remain exactly 1 mm"
        )
    device = hand.hand_pose.device
    dtype = hand.hand_pose.dtype
    evaluated = evaluate_rows.detach().to(device=device, dtype=torch.bool)
    infinity = torch.full((batch_size,), torch.inf, device=device, dtype=dtype)
    forward_total = infinity.clone()
    forward_maximum = infinity.clone()
    selected_rows = torch.nonzero(evaluated, as_tuple=False).flatten()
    if selected_rows.numel() > 0:
        with torch.no_grad():
            points = hand.collision_points_world().detach().index_select(
                0, selected_rows
            )
            selected_total, selected_maximum = object_model.penetration_summary(
                points
            )
            forward_total.index_copy_(0, selected_rows, selected_total)
            forward_maximum.index_copy_(0, selected_rows, selected_maximum)

    reverse_total = torch.where(
        evaluated, energy.E_pen.detach(), infinity
    )
    reverse_maximum = torch.where(
        evaluated,
        energy.reverse_maximum_penetration.detach(),
        infinity,
    )
    total = forward_total + reverse_total
    maximum = torch.maximum(forward_maximum, reverse_maximum)
    finite = torch.isfinite(total) & torch.isfinite(maximum)
    feasible = evaluated & finite & (maximum < float(threshold))
    return HandObjectPenetrationDiagnostics(
        forward_total_penetration=forward_total,
        forward_maximum_penetration=forward_maximum,
        reverse_total_penetration=reverse_total,
        reverse_maximum_penetration=reverse_maximum,
        total_penetration=total,
        maximum_penetration=maximum,
        feasible=feasible,
        evaluated=evaluated,
        threshold=float(threshold),
    )


def _dense_hand_object_penetration_diagnostics(
    hand: X2HandModel,
    object_model: MeshObjectModel,
    evaluate_rows: torch.Tensor,
    *,
    threshold: float = HAND_OBJECT_PENETRATION_THRESHOLD,
) -> HandObjectPenetrationDiagnostics:
    """Evaluate the final hard gate at 256/set hand and 8192 object points."""

    if hand.hand_pose is None:
        raise RuntimeError("Hand state has not been materialized")
    batch_size = hand.hand_pose.shape[0]
    if (
        evaluate_rows.shape != (batch_size,)
        or evaluate_rows.dtype != torch.bool
    ):
        raise ValueError("evaluate_rows must be one boolean per batch row")
    if float(threshold) != HAND_OBJECT_PENETRATION_THRESHOLD:
        raise ValueError(
            "The hand-object feasibility threshold must remain exactly 1 mm"
        )
    if (
        hand.backend.audit_collision_samples_per_link
        != DENSE_HAND_SURFACE_SAMPLES_PER_SET
        or object_model.audit_surface_samples != DENSE_OBJECT_SURFACE_SAMPLES
    ):
        raise RuntimeError("Dense hand/object audit sample density drifted")
    device = hand.hand_pose.device
    dtype = hand.hand_pose.dtype
    evaluated = evaluate_rows.detach().to(device=device, dtype=torch.bool)
    infinity = torch.full((batch_size,), torch.inf, device=device, dtype=dtype)
    forward_total = infinity.clone()
    forward_maximum = infinity.clone()
    reverse_total = infinity.clone()
    reverse_maximum = infinity.clone()
    selected_rows = torch.nonzero(evaluated, as_tuple=False).flatten()
    if selected_rows.numel() > 0:
        with torch.no_grad():
            hand_points = hand.audit_collision_points_world(selected_rows).detach()
            selected_forward_total, selected_forward_maximum = (
                object_model.penetration_summary(hand_points)
            )
            object_points = object_model.audit_surface_points.to(
                device=device, dtype=dtype
            ).unsqueeze(0).expand(len(selected_rows), -1, -1)
            selected_reverse_depth = torch.relu(
                hand.cal_distance(object_points, row_indices=selected_rows)
            )
            selected_reverse_total = selected_reverse_depth.sum(dim=-1)
            selected_reverse_maximum = selected_reverse_depth.amax(dim=-1)
            forward_total.index_copy_(
                0, selected_rows, selected_forward_total
            )
            forward_maximum.index_copy_(
                0, selected_rows, selected_forward_maximum
            )
            reverse_total.index_copy_(
                0, selected_rows, selected_reverse_total
            )
            reverse_maximum.index_copy_(
                0, selected_rows, selected_reverse_maximum
            )
    total = forward_total + reverse_total
    maximum = torch.maximum(forward_maximum, reverse_maximum)
    finite = torch.isfinite(total) & torch.isfinite(maximum)
    feasible = evaluated & finite & (maximum < float(threshold))
    return HandObjectPenetrationDiagnostics(
        forward_total_penetration=forward_total,
        forward_maximum_penetration=forward_maximum,
        reverse_total_penetration=reverse_total,
        reverse_maximum_penetration=reverse_maximum,
        total_penetration=total,
        maximum_penetration=maximum,
        feasible=feasible,
        evaluated=evaluated,
        threshold=float(threshold),
    )


def _bidirectional_feasible_improvement_mask(
    bank: _CheckpointBank,
    candidate_valid: torch.Tensor,
    energy: MeshEnergy,
    self_collision: SelfCollisionDiagnostics,
    hand_object: HandObjectPenetrationDiagnostics,
) -> torch.Tensor:
    """Prefer minimum energy among strictly joint-feasible checkpoints."""

    if not bank.hand_object:
        raise RuntimeError("Bidirectional checkpoint bank has no diagnostics")
    jointly_feasible = (
        candidate_valid
        & self_collision.feasible.detach().to(dtype=torch.bool)
        & hand_object.evaluated
        & hand_object.feasible
    )
    candidate_energy = energy.total.detach()
    bank_energy = bank.energy["total"]
    better_energy = candidate_energy < bank_energy
    equal_energy = candidate_energy == bank_energy
    better_maximum = (
        hand_object.maximum_penetration
        < bank.hand_object["maximum_penetration"]
    )
    better = better_energy | (equal_energy & better_maximum)
    return jointly_feasible & (~bank.valid | better)


def _bidirectional_fallback_improvement_mask(
    bank: _CheckpointBank,
    candidate_valid: torch.Tensor,
    energy: MeshEnergy,
    self_collision: SelfCollisionDiagnostics,
    hand_object: HandObjectPenetrationDiagnostics,
) -> torch.Tensor:
    """For self-feasible rows, minimize exact bidirectional penetration."""

    if not bank.hand_object:
        raise RuntimeError("Bidirectional checkpoint bank has no diagnostics")
    candidate = (
        candidate_valid
        & self_collision.feasible.detach().to(dtype=torch.bool)
        & hand_object.evaluated
    )
    candidate_maximum = hand_object.maximum_penetration
    candidate_total = hand_object.total_penetration
    candidate_energy = energy.total.detach()
    bank_maximum = bank.hand_object["maximum_penetration"]
    bank_total = bank.hand_object["total_penetration"]
    bank_energy = bank.energy["total"]
    better_maximum = candidate_maximum < bank_maximum
    equal_maximum = candidate_maximum == bank_maximum
    better_total = candidate_total < bank_total
    equal_total = candidate_total == bank_total
    better_energy = candidate_energy < bank_energy
    better = better_maximum | (
        equal_maximum & (better_total | (equal_total & better_energy))
    )
    return candidate & (~bank.valid | better)


def _fallback_improvement_mask(
    bank: _CheckpointBank,
    candidate_valid: torch.Tensor,
    energy: MeshEnergy,
    self_collision: SelfCollisionDiagnostics,
) -> torch.Tensor:
    """Lexicographic min of maximum, total, energy; an earlier tie wins."""

    candidate_maximum = self_collision.maximum_penetration.detach()
    candidate_total = self_collision.total_penetration.detach()
    candidate_energy = energy.total.detach()
    bank_maximum = bank.self_collision["maximum_penetration"]
    bank_total = bank.self_collision["total_penetration"]
    bank_energy = bank.energy["total"]
    better_maximum = candidate_maximum < bank_maximum
    equal_maximum = candidate_maximum == bank_maximum
    better_total = candidate_total < bank_total
    equal_total = candidate_total == bank_total
    better_energy = candidate_energy < bank_energy
    better = better_maximum | (
        equal_maximum & (better_total | (equal_total & better_energy))
    )
    return candidate_valid & (~bank.valid | better)


def _feasible_improvement_mask(
    bank: _CheckpointBank,
    candidate_valid: torch.Tensor,
    energy: MeshEnergy,
    self_collision: SelfCollisionDiagnostics,
) -> torch.Tensor:
    feasible = self_collision.feasible.detach().to(dtype=torch.bool)
    better = energy.total.detach() < bank.energy["total"]
    return candidate_valid & feasible & (~bank.valid | better)


def _table_contact_improvement_mask(
    bank: _CheckpointBank,
    candidate_valid: torch.Tensor,
    energy: MeshEnergy,
    self_collision: SelfCollisionDiagnostics,
    *,
    requested_clearance_m: float,
) -> torch.Tensor:
    """Retain the best source-table-admissible contact checkpoint per row.

    The ordinary checkpoint pool ranks grasp energy and sampled hand-object
    penetration. Neither quantity guarantees that the complete calibrated
    hand collision mesh stays above the source support plane. This auxiliary
    bank prevents a rare useful state from being evicted before the existing
    dense hand-object audit. It grants no downstream planning authority.
    """

    if energy.minimum_table_clearance is None or energy.E_contact is None:
        raise RuntimeError(
            "table-contact checkpointing requires table clearance and contact energy"
        )
    if not math.isfinite(float(requested_clearance_m)) or requested_clearance_m < 0.0:
        raise ValueError("requested_clearance_m must be finite and non-negative")
    if energy.forward_maximum_penetration is None:
        raise RuntimeError(
            "table-contact checkpointing requires forward penetration diagnostics"
        )
    candidate_clearance = energy.minimum_table_clearance.detach()
    candidate_contact = energy.E_contact.detach()
    candidate = (
        candidate_valid
        & self_collision.feasible.detach().to(dtype=torch.bool)
        & torch.isfinite(candidate_clearance)
        & torch.isfinite(candidate_contact)
        & (candidate_clearance >= 0.0)
        & (candidate_contact <= 0.0)
    )
    candidate_requested = candidate_clearance >= float(requested_clearance_m)
    bank_clearance = bank.energy["minimum_table_clearance"]
    bank_requested = bank_clearance >= float(requested_clearance_m)

    candidate_maximum = torch.maximum(
        energy.reverse_maximum_penetration.detach(),
        energy.forward_maximum_penetration.detach(),
    )
    bank_maximum = torch.maximum(
        bank.energy["reverse_maximum_penetration"],
        bank.energy["forward_maximum_penetration"],
    )
    better_requested_class = candidate_requested & ~bank_requested
    same_requested_class = candidate_requested == bank_requested
    better_maximum = candidate_maximum < bank_maximum
    equal_maximum = candidate_maximum == bank_maximum
    better_energy = energy.total.detach() < bank.energy["total"]
    equal_energy = energy.total.detach() == bank.energy["total"]
    better_clearance = candidate_clearance > bank_clearance
    better = better_requested_class | (
        same_requested_class
        & (
            better_maximum
            | (
                equal_maximum
                & (better_energy | (equal_energy & better_clearance))
            )
        )
    )
    return candidate & (~bank.valid | better)


def _selected_bank_tensor(
    feasible: _CheckpointBank,
    fallback: _CheckpointBank,
    use_feasible: torch.Tensor,
    field: str,
) -> torch.Tensor:
    first = getattr(feasible, field)
    second = getattr(fallback, field)
    shape = (len(use_feasible),) + (1,) * (first.ndim - 1)
    return torch.where(use_feasible.reshape(shape), first, second)


def _select_rows(
    mask: torch.Tensor, selected: torch.Tensor, fallback: torch.Tensor
) -> torch.Tensor:
    shape = (len(mask),) + (1,) * (selected.ndim - 1)
    return torch.where(mask.reshape(shape), selected, fallback)


def _selected_priority_bank_tensor(
    *,
    bidirectional_feasible: _CheckpointBank,
    bidirectional_fallback: _CheckpointBank,
    feasible: _CheckpointBank,
    fallback: _CheckpointBank,
    use_bidirectional_feasible: torch.Tensor,
    use_bidirectional_fallback: torch.Tensor,
    use_feasible: torch.Tensor,
    field: str,
) -> torch.Tensor:
    result = getattr(fallback, field)
    result = _select_rows(use_feasible, getattr(feasible, field), result)
    result = _select_rows(
        use_bidirectional_fallback,
        getattr(bidirectional_fallback, field),
        result,
    )
    return _select_rows(
        use_bidirectional_feasible,
        getattr(bidirectional_feasible, field),
        result,
    )


def _proposal_protection_masks(
    old_maximum: torch.Tensor,
    new_maximum: torch.Tensor,
    finite: torch.Tensor,
    *,
    enabled: bool,
    hard_threshold: float,
    maximum_allowed_increase: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return allowed rows and rows rejected for severe collision regression."""

    serious = (new_maximum > hard_threshold) & (
        new_maximum > old_maximum + maximum_allowed_increase
    )
    if not enabled:
        serious = torch.zeros_like(serious)
    serious &= finite
    return finite & ~serious, serious


def _sample_surface(mesh: trimesh.Trimesh, count: int, rng: np.random.Generator) -> np.ndarray:
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    chosen = rng.choice(len(triangles), size=count, p=areas / areas.sum())
    random = rng.random((count, 2))
    root = np.sqrt(random[:, 0])
    weights = np.stack((1.0 - root, root * (1.0 - random[:, 1]), root * random[:, 1]), axis=1)
    return np.einsum("ni,nij->nj", weights, triangles[chosen])


def _table_exposed_surface_points(
    points: np.ndarray,
    table_plane: TabletopPlaneConstraint,
    *,
    exposed_fraction: float,
    minimum_count: int,
) -> np.ndarray:
    """Keep the support-normal-exposed portion of an object surface.

    A local ``z`` filter is not meaningful once an object is placed on an
    arbitrary stable face. The table normal, expressed in the mesh frame,
    defines the physically exposed direction. ``exposed_fraction=0.5`` keeps
    the upper half of the sampled support-height range; deterministic top-k
    fallback handles very coarse or degenerate samples.
    """

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("surface points must be a finite Nx3 array")
    if not math.isfinite(float(exposed_fraction)) or not 0.0 <= exposed_fraction <= 1.0:
        raise ValueError("exposed_fraction must lie in [0, 1]")
    if minimum_count <= 0:
        raise ValueError("minimum_count must be positive")
    normal = np.asarray(table_plane.normal, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError("table plane normal must contain three finite values")
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError("table plane normal must be non-zero")
    normal /= norm
    heights = values @ normal - float(table_plane.offset_m)
    lower = float(heights.min())
    upper = float(heights.max())
    threshold = lower + float(exposed_fraction) * (upper - lower)
    selected = values[heights >= threshold]
    if len(selected) >= minimum_count:
        return selected
    count = min(minimum_count, len(values))
    order = np.argsort(-heights, kind="stable")[:count]
    return values[order]


def _farthest_points(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministic FPS matching the role of the original PyTorch3D call."""

    selected = np.empty(count, dtype=np.int64)
    selected[0] = 0
    minimum = np.sum((points - points[0]) ** 2, axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(minimum))
        distance = np.sum((points - points[selected[index]]) ** 2, axis=1)
        minimum = np.minimum(minimum, distance)
    return points[selected]


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise RuntimeError("Cannot normalize a zero direction")
    return value / norm


def _approach_with_cone_jitter(
    inward: np.ndarray, theta: float, azimuth: float
) -> np.ndarray:
    inward = _unit(inward)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(inward @ reference)) > 0.9:
        reference = np.array([0.0, 0.0, 1.0])
    tangent = _unit(np.cross(inward, reference))
    bitangent = np.cross(inward, tangent)
    return _unit(
        math.cos(theta) * inward
        + math.sin(theta) * (math.cos(azimuth) * tangent + math.sin(azimuth) * bitangent)
    )


def _world_grasp_frame(approach: np.ndarray, roll: float) -> np.ndarray:
    z_axis = _unit(approach)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(z_axis @ reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    x_base = _unit(np.cross(reference, z_axis))
    y_base = np.cross(z_axis, x_base)
    x_axis = math.cos(roll) * x_base + math.sin(roll) * y_base
    y_axis = -math.sin(roll) * x_base + math.cos(roll) * y_base
    return np.stack((x_axis, y_axis, z_axis), axis=1)


_NONTHUMB_FINGER_CLOSURE_ACTUATORS = {
    "little": ("rh_LFJ3", "rh_LFJ2"),
    "ring": ("rh_RFJ3", "rh_RFJ2"),
    "middle": ("rh_MFJ3", "rh_MFJ2"),
    "index": ("rh_FFJ3", "rh_FFJ2"),
}


def _freeze_unselected_finger_actuators(
    hand: X2HandModel,
    actuators: torch.Tensor,
    contact_indices: torch.Tensor,
    canonical_open: torch.Tensor,
) -> torch.Tensor:
    """Keep acquisition-reserved fingers at the canonical open command.

    A contact policy is also an ownership policy.  Letting the optimizer move
    unselected fingers produced records labelled ``index`` whose middle/ring
    collision hulls already touched the object.  Downstream Admit0 must reject
    those rows.  Freezing only unselected actuators here avoids generating the
    contradiction while leaving hand pose and every selected finger mutable.
    """

    if actuators.ndim != 2 or actuators.shape[1] != len(hand.actuator_names):
        raise ValueError("actuators must be a [B, n_actuator] tensor")
    if contact_indices.ndim != 2 or contact_indices.shape[0] != actuators.shape[0]:
        raise ValueError("contact indices must align with actuator rows")
    canonical = canonical_open.to(actuators).reshape(-1)
    if canonical.shape != (len(hand.actuator_names),):
        raise ValueError("canonical open command has the wrong actuator count")

    selected_finger_indices = hand._candidate_finger_indices.to(
        contact_indices.device
    )[contact_indices]
    selected = torch.stack(
        tuple(
            (selected_finger_indices == finger_index).any(dim=1)
            for finger_index in range(len(hand.finger_names))
        ),
        dim=1,
    )
    actuator_index = {name: index for index, name in enumerate(hand.actuator_names)}
    by_finger = {
        **_NONTHUMB_FINGER_CLOSURE_ACTUATORS,
        "thumb": tuple(hand.thumb_actuator_names),
    }
    mutable = torch.zeros_like(actuators, dtype=torch.bool)
    finger_index = {name: index for index, name in enumerate(hand.finger_names)}
    for finger_name, names in by_finger.items():
        finger_selected = selected[:, finger_index[finger_name]]
        for name in names:
            mutable[:, actuator_index[name]] = finger_selected
    return torch.where(mutable, actuators, canonical.unsqueeze(0))


def _seed_selected_finger_closure(
    hand: X2HandModel,
    actuators: torch.Tensor,
    contact_indices: torch.Tensor,
    active_sides: Sequence[str],
    config: X2Config,
) -> torch.Tensor:
    """Seed selected non-thumb fingers in their physical closing direction.

    The generic initializer already locates the palm beside the sampled object
    surface.  For a palm-plus-fingers selection, leaving every selected finger
    at the canonical open command puts the finger contacts tens of millimetres
    away and makes the optimizer cross a poor collision basin.  This bounded
    seed preserves all policy-selected contact identities and only supplies a
    side-aware initial posture; optimization and the explicit output gate stay
    authoritative.
    """

    if contact_indices.ndim != 2 or contact_indices.shape[0] != len(active_sides):
        raise ValueError("contact_indices must contain one selected tuple per active side")
    actuator_index = {name: index for index, name in enumerate(hand.actuator_names)}
    j3_seed = float(config.require("initialization.contact_closure_seed_j3"))
    j2_seed = float(config.require("initialization.contact_closure_seed_j2"))
    seeded = actuators.clone()
    lower = hand.joints_lower.to(seeded)
    upper = hand.joints_upper.to(seeded)
    for row, side in enumerate(active_sides):
        if side not in ("front", "back"):
            raise ValueError(f"Unsupported active side {side!r}")
        direction = 1.0 if side == "front" else -1.0
        selected_fingers = {
            hand.candidates[int(candidate)].finger_name
            for candidate in contact_indices[row].detach().cpu().tolist()
        }
        for finger_name in selected_fingers:
            actuator_names = _NONTHUMB_FINGER_CLOSURE_ACTUATORS.get(finger_name)
            if actuator_names is None:
                continue
            j3_index, j2_index = (actuator_index[name] for name in actuator_names)
            seeded[row, j3_index] = direction * j3_seed
            seeded[row, j2_index] = direction * j2_seed
    return torch.maximum(torch.minimum(seeded, upper), lower)


def _table_roll_is_better(
    *,
    best_clearance: torch.Tensor,
    best_contact_distance: torch.Tensor,
    proposed_clearance: torch.Tensor,
    proposed_contact_distance: torch.Tensor,
    required_clearance: float,
) -> torch.Tensor:
    """Lexicographically prefer table admission, then contact realization.

    Clearance remains the only fallback criterion when no sampled roll reaches
    the source support-plane margin.  Once a roll is admissible, excess
    clearance has no task value at initialization, while selected-contact
    distance determines whether continuous optimization starts in the useful
    contact basin.
    """

    threshold = best_clearance.new_tensor(float(required_clearance))
    best_admissible = best_clearance >= threshold
    proposed_admissible = proposed_clearance >= threshold
    better_contact = proposed_contact_distance < best_contact_distance
    tied_contact = proposed_contact_distance == best_contact_distance
    return (
        (proposed_admissible & ~best_admissible)
        | (
            proposed_admissible
            & best_admissible
            & (better_contact | (tied_contact & (proposed_clearance > best_clearance)))
        )
        | (
            ~proposed_admissible
            & ~best_admissible
            & (proposed_clearance > best_clearance)
        )
    )


def initialize_x2_convex_hull(
    hand: X2HandModel,
    object_model: MeshObjectModel,
    active_sides: Sequence[str],
    policies: ContactPolicyInput,
    config: X2Config,
    rng: np.random.Generator,
    *,
    table_plane: TabletopPlaneConstraint | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Original convex-hull initialization with a side-specific grasp frame."""

    batch_size = len(active_sides)
    if batch_size != object_model.batch_size_each:
        raise ValueError("active_sides length must equal object batch size")
    if any(side not in ("front", "back") for side in active_sides):
        raise ValueError("Every active side must be front or back")
    row_policies = _resolve_row_policies(policies, active_sides)
    contact_counts = {policy.n_contact for policy in row_policies}
    if len(contact_counts) != 1:
        raise ValueError(
            "Every row policy in one optimizer batch must use the same contact count"
        )

    hull_origin = object_model.convex_hull.copy()
    vertices = np.asarray(hull_origin.vertices, dtype=np.float64).copy()
    inflation = float(config.require("initialization.hull_inflation"))
    if not math.isfinite(inflation) or inflation < 0.0:
        raise ValueError("initialization.hull_inflation must be finite and non-negative")
    # ``hull_inflation`` is a dimensionless radial fraction.  Treating its
    # configured 0.20 value as metres displaced every mesh by 20 cm, which
    # is outside the contact basin for tabletop objects.  Centering also
    # avoids assuming that every mesh is authored about the origin.
    center = vertices.mean(axis=0, keepdims=True)
    direction = vertices - center
    inflated_vertices = center + (1.0 + inflation) * direction
    inflated = trimesh.Trimesh(
        vertices=inflated_vertices,
        faces=np.asarray(hull_origin.faces),
        process=False,
    ).convex_hull
    dense = _sample_surface(inflated, max(100 * batch_size, batch_size), rng)
    minimum_surface_z = config.data.get("initialization", {}).get(
        "minimum_surface_z_m"
    )
    if table_plane is not None:
        dense = _table_exposed_surface_points(
            dense,
            table_plane,
            exposed_fraction=float(
                config.require("initialization.table_exposed_surface_fraction")
            ),
            minimum_count=batch_size,
        )
    elif minimum_surface_z is not None:
        minimum_surface_z = float(minimum_surface_z)
        if not math.isfinite(minimum_surface_z):
            raise ValueError("initialization.minimum_surface_z_m must be finite")
        dense = dense[dense[:, 2] >= minimum_surface_z]
        if len(dense) < batch_size:
            raise ValueError(
                "initialization.minimum_surface_z_m leaves fewer convex-hull "
                f"samples ({len(dense)}) than the batch size ({batch_size})"
            )
    sampled = _farthest_points(dense, batch_size)
    closest, _, _ = trimesh.proximity.closest_point_naive(hull_origin, sampled)
    inward = closest - sampled
    inward /= np.linalg.norm(inward, axis=1, keepdims=True).clip(1.0e-12)

    distance_lower = float(config.require("initialization.distance_lower"))
    distance_upper = float(config.require("initialization.distance_upper"))
    theta_lower = float(config.require("initialization.theta_lower"))
    theta_upper = float(config.require("initialization.theta_upper"))
    distances = rng.uniform(distance_lower, distance_upper, size=batch_size)
    theta = rng.uniform(theta_lower, theta_upper, size=batch_size)
    process = rng.uniform(0.0, 2.0 * math.pi, size=batch_size)
    roll = rng.uniform(0.0, 2.0 * math.pi, size=batch_size)

    approaches = np.empty((batch_size, 3), dtype=np.float64)
    palm_centers_world = np.empty((batch_size, 3), dtype=np.float64)
    local_frames = np.empty((batch_size, 3, 3), dtype=np.float64)
    palm_centers = np.empty((batch_size, 3), dtype=np.float64)
    for index, side in enumerate(active_sides):
        approach = _approach_with_cone_jitter(inward[index], theta[index], process[index])
        approaches[index] = approach
        local_frames[index] = (
            hand.front_grasp_frame if side == "front" else hand.back_grasp_frame
        ).detach().cpu().numpy()
        palm_centers[index] = hand.palm_centers[side].detach().cpu().numpy()
        palm_centers_world[index] = sampled[index] - distances[index] * approach

    contact_indices = torch.tensor(
        [policy.sample(rng) for policy in row_policies],
        device=hand.device,
        dtype=torch.long,
    )
    canonical = torch.tensor(
        config.require("initialization.canonical_open_actuator"),
        device=hand.device,
        dtype=hand.dtype,
    )
    lower = hand.joints_lower.to(hand.device, hand.dtype)
    upper = hand.joints_upper.to(hand.device, hand.dtype)
    sigma = float(config.require("initialization.jitter_strength")) * (upper - lower)
    actuators = torch.empty(batch_size, 12, device=hand.device, dtype=hand.dtype)
    for actuator_index in range(12):
        torch.nn.init.trunc_normal_(
            actuators[:, actuator_index],
            mean=float(canonical[actuator_index]),
            std=float(sigma[actuator_index]),
            a=float(lower[actuator_index]) + 1.0e-6,
            b=float(upper[actuator_index]) - 1.0e-6,
        )
    if hand.freeze_thumb:
        actuators = hand.backend.with_fixed_thumb(actuators)
    actuators = _seed_selected_finger_closure(
        hand, actuators, contact_indices, active_sides, config
    )
    actuators = _freeze_unselected_finger_actuators(
        hand,
        actuators,
        contact_indices,
        canonical,
    )

    def pose_for_rolls(roll_values: np.ndarray) -> torch.Tensor:
        translations = np.empty((batch_size, 3), dtype=np.float64)
        rotations = np.empty((batch_size, 3, 3), dtype=np.float64)
        for row in range(batch_size):
            world_frame = _world_grasp_frame(approaches[row], float(roll_values[row]))
            rotation = world_frame @ local_frames[row].T
            translations[row] = palm_centers_world[row] - rotation @ palm_centers[row]
            rotations[row] = rotation
        rotation_tensor = torch.tensor(
            rotations, device=hand.device, dtype=hand.dtype
        )
        # ``robust_compute_rotation_matrix_from_ortho6d`` consumes the first
        # two *columns* as two three-vectors.  Encoding the first two rows
        # silently transposes a general rotation.
        rotation6d = (
            rotation_tensor[:, :, :2].transpose(1, 2).reshape(batch_size, 6)
        )
        return torch.cat(
            (
                torch.tensor(
                    translations, device=hand.device, dtype=hand.dtype
                ),
                rotation6d,
                actuators,
            ),
            dim=1,
        )

    pose = pose_for_rolls(roll)
    if table_plane is not None:
        roll_samples = int(config.require("initialization.table_aware_roll_samples"))
        best_pose = pose.detach().clone()
        hand.set_parameters(best_pose, contact_indices)
        best_clearance = tabletop_plane_collision_mesh_minimum_clearance(
            hand, table_plane
        ).detach()
        if hand.contact_points is None:
            raise RuntimeError("hand contact points are unavailable after initialization")
        best_contact_distance = object_model.cal_distance(
            hand.contact_points
        )[0].abs().amax(dim=-1).detach()
        # A finite roll family changes neither the sampled object surface nor
        # the palm-side approach.  First find a source-table-admissible roll,
        # then preserve the best selected-contact basin among admissible rolls.
        # Exact tabletop admission remains downstream and no source clearance
        # is relabelled as proof.
        for sample in range(1, roll_samples):
            offset = 2.0 * math.pi * float(sample) / float(roll_samples)
            proposed = pose_for_rolls(roll + offset)
            hand.set_parameters(proposed, contact_indices)
            clearance = tabletop_plane_collision_mesh_minimum_clearance(
                hand, table_plane
            ).detach()
            if hand.contact_points is None:
                raise RuntimeError("hand contact points are unavailable after initialization")
            contact_distance = object_model.cal_distance(
                hand.contact_points
            )[0].abs().amax(dim=-1).detach()
            improve = _table_roll_is_better(
                best_clearance=best_clearance,
                best_contact_distance=best_contact_distance,
                proposed_clearance=clearance,
                proposed_contact_distance=contact_distance,
                required_clearance=table_plane.clearance_m,
            )
            best_pose[improve] = proposed[improve]
            best_clearance = torch.where(improve, clearance, best_clearance)
            best_contact_distance = torch.where(
                improve, contact_distance, best_contact_distance
            )
        pose = best_pose
    pose = pose.detach().requires_grad_()
    hand.set_parameters(pose, contact_indices)
    return pose, contact_indices


def cal_x2_mesh_energy(
    hand: X2HandModel,
    object_model: MeshObjectModel,
    active_sides: Sequence[str],
    weights: Mapping[str, float],
    *,
    side_margin: float = 0.005,
    normal_opposition_weight: float = 5.0,
    unselected_finger_margin: float = 0.0,
    unselected_finger_scale: float = 0.02,
    table_plane: TabletopPlaneConstraint | None = None,
    selected_contact_max_distance_m: float = 0.003,
) -> MeshEnergy:
    """DexGraspNet mesh terms plus X2 side-conditioning penalties."""

    if hand.contact_points is None or hand.contact_normals is None:
        raise RuntimeError("Hand contacts have not been materialized")
    batch_size, n_contact, _ = hand.contact_points.shape
    distance, object_normal = object_model.cal_distance(hand.contact_points)
    E_dis = distance.abs().sum(dim=-1)
    if (
        not math.isfinite(float(selected_contact_max_distance_m))
        or float(selected_contact_max_distance_m) < 0.0
    ):
        raise ValueError(
            "selected_contact_max_distance_m must be finite and non-negative"
        )
    # The output contract is pointwise: every selected contact must realize.
    # Keep this differentiable barrier separate from aggregate E_dis so a
    # distant finger cannot be hidden by other near selected contacts.
    E_contact = torch.relu(
        distance.abs() - distance.new_tensor(float(selected_contact_max_distance_m))
    ).square().sum(dim=-1)

    contact_normal = object_normal.reshape(batch_size, 1, 3 * n_contact)
    transformation_matrix = torch.tensor(
        [[0, 0, 0, 0, 0, -1, 0, 1, 0],
         [0, 0, 1, 0, 0, 0, -1, 0, 0],
         [0, -1, 0, 1, 0, 0, 0, 0, 0]],
        device=hand.device,
        dtype=hand.dtype,
    )
    grasp = torch.cat(
        (
            torch.eye(3, device=hand.device, dtype=hand.dtype)
            .expand(batch_size, n_contact, 3, 3)
            .reshape(batch_size, 3 * n_contact, 3),
            (hand.contact_points @ transformation_matrix).reshape(
                batch_size, 3 * n_contact, 3
            ),
        ),
        dim=2,
    )
    closure_norm = torch.linalg.vector_norm(contact_normal @ grasp, dim=(1, 2))
    normal_dot = torch.sum(hand.contact_normals * object_normal, dim=-1)
    normal_opposition_penalty = (normal_dot + 1.0).square().sum(dim=-1)
    E_fc = closure_norm.square() + float(normal_opposition_weight) * normal_opposition_penalty

    actuators = hand.actuator_positions
    joints = hand.joint_positions
    actuator_limits = hand.backend.actuator_limits_tensor.to(actuators)
    joint_limits = hand.backend.joint_limits_tensor.to(joints)
    actuator_violation = torch.relu(actuator_limits[:, 0] - actuators) + torch.relu(
        actuators - actuator_limits[:, 1]
    )
    joint_violation = torch.relu(joint_limits[:, 0] - joints) + torch.relu(
        joints - joint_limits[:, 1]
    )
    E_joints = actuator_violation.sum(dim=-1) + joint_violation.sum(dim=-1)

    reverse_depth = torch.relu(
        hand.cal_distance(object_model.surface_points_tensor)
    )
    E_pen = reverse_depth.sum(dim=-1)
    reverse_maximum_penetration = reverse_depth.amax(dim=-1)
    forward_depth = object_model.penetration_depth(
        _collision_mesh_vertices_world(hand)
    )
    # Normalize the collision-vertex sum to the reverse surface-sample count,
    # so the two directions share the existing E_pen weight without making
    # that weight depend on the hand mesh tessellation density.
    forward_scale = float(object_model.num_surface_samples) / float(
        forward_depth.shape[1]
    )
    E_forward_pen = forward_depth.sum(dim=-1) * forward_scale
    forward_maximum_penetration = forward_depth.amax(dim=-1)
    maximum_bidirectional_penetration = torch.maximum(
        forward_maximum_penetration, reverse_maximum_penetration
    )
    E_hand_object_gate = torch.relu(
        maximum_bidirectional_penetration
        / maximum_bidirectional_penetration.new_tensor(
            HAND_OBJECT_PENETRATION_THRESHOLD
        )
        - 1.0
    ).square()
    # Keep E_spen as the legacy capsule proxy.  Hull self-collision is a
    # separate term so old consumers can still interpret E_spen unchanged.
    E_spen_capsule = hand.self_penetration()
    E_spen = E_spen_capsule
    E_spen_hull = hand.self_collision_hull_energy()

    object_center_root = torch.einsum(
        "bi,bij->bj", -hand.global_translation, hand.global_rotation
    )
    side_penalties: list[torch.Tensor] = []
    for index, side in enumerate(active_sides):
        normal = hand.front_palm_normal if side == "front" else hand.back_palm_normal
        relative = object_center_root[index] - hand.palm_centers[side]
        projection = torch.dot(relative, normal.to(relative))
        side_penalties.append(torch.relu(relative.new_tensor(side_margin) - projection).square())
    E_side = torch.stack(side_penalties)
    E_unselected_opposite_flex = cal_unselected_opposite_flex_energy(
        hand,
        active_sides,
        margin=unselected_finger_margin,
        displacement_scale=unselected_finger_scale,
    )
    if table_plane is None:
        E_table = torch.zeros_like(E_dis)
        minimum_table_clearance = None
    else:
        normal = table_plane.normalized(device=hand.device, dtype=hand.dtype)
        signed_clearance = (
            _tabletop_collision_mesh_clearances(hand, normal)
            - float(table_plane.offset_m)
        )
        minimum_table_clearance = signed_clearance.amin(dim=-1)
        # The hard all-vertex clearance conjunction is governed by its worst
        # violation. Averaging over thousands of safe vertices diluted a
        # deeply penetrating link and systematically returned unusable raw
        # poses. The squared maximum hinge is zero iff every calibrated
        # collision-mesh vertex reaches the requested source-plane margin.
        E_table = torch.relu(
            signed_clearance.new_tensor(float(table_plane.clearance_m))
            - signed_clearance
        ).amax(dim=-1).square()

    normal_opposition = normal_dot.mean(dim=-1)
    total = (
        E_fc
        + float(weights["E_dis"]) * E_dis
        + float(weights.get("E_contact", 0.0)) * E_contact
        + float(weights["E_pen"]) * (E_pen + E_forward_pen)
        + float(weights.get("E_hand_object_gate", 0.0)) * E_hand_object_gate
        + hand.self_collision_capsule_weight * E_spen_capsule
        + hand.self_collision_hull_weight * E_spen_hull
        + float(weights["E_joints"]) * E_joints
        + float(weights.get("E_side", 0.0)) * E_side
        + float(weights.get("E_unselected_opposite_flex", 0.0))
        * E_unselected_opposite_flex
        + (0.0 if table_plane is None else float(table_plane.weight)) * E_table
    )
    return MeshEnergy(
        total=total,
        E_fc=E_fc,
        E_dis=E_dis,
        E_pen=E_pen,
        reverse_maximum_penetration=reverse_maximum_penetration,
        E_spen=E_spen,
        E_spen_capsule=E_spen_capsule,
        E_spen_hull=E_spen_hull,
        E_joints=E_joints,
        E_side=E_side,
        E_unselected_opposite_flex=E_unselected_opposite_flex,
        normal_opposition=normal_opposition,
        normal_opposition_penalty=normal_opposition_penalty,
        E_table=E_table if table_plane is not None else None,
        E_contact=E_contact,
        E_forward_pen=E_forward_pen,
        forward_maximum_penetration=forward_maximum_penetration,
        E_hand_object_gate=E_hand_object_gate,
        minimum_table_clearance=minimum_table_clearance,
    )


def tabletop_plane_minimum_clearance(
    hand: X2HandModel,
    table_plane: TabletopPlaneConstraint,
    *,
    dense_audit: bool = True,
) -> torch.Tensor:
    """Return one sampled hand/table signed clearance per optimized row.

    This is an auditable generator diagnostic.  It is deliberately not named
    an exact collision result and must not be used as a substitute for the
    sequential exact tabletop admission.
    """

    normal = table_plane.normalized(device=hand.device, dtype=hand.dtype)
    points = (
        hand.audit_collision_points_world()
        if dense_audit
        else hand.collision_points_world()
    )
    return (
        torch.einsum("bni,i->bn", points, normal) - float(table_plane.offset_m)
    ).amin(dim=-1)


def _collision_mesh_vertices_world(hand: X2HandModel) -> torch.Tensor:
    """Transform every calibrated collision-hull vertex through live FK."""

    if (
        hand.current_status is None
        or hand.global_rotation is None
        or hand.global_translation is None
    ):
        raise RuntimeError("hand parameters must be materialized before table evaluation")
    values: list[torch.Tensor] = []
    for link_name in hand.backend.link_names:
        vertices = torch.as_tensor(
            np.array(
                hand.backend.collision_meshes[link_name].vertices_local,
                copy=True,
            ),
            device=hand.device,
            dtype=hand.dtype,
        )
        transform = hand.current_status[link_name]
        root = (
            torch.einsum("bij,nj->bni", transform[:, :3, :3], vertices)
            + transform[:, None, :3, 3]
        )
        world = (
            torch.einsum("bij,bnj->bni", hand.global_rotation, root)
            + hand.global_translation[:, None, :]
        )
        values.append(world)
    return torch.cat(values, dim=1)


def _tabletop_collision_mesh_clearances(
    hand: X2HandModel, normal: torch.Tensor
) -> torch.Tensor:
    """Return support-plane clearances of every vertex in every link mesh.

    Sampling a link surface can miss a low collision-hull vertex, exactly the
    failure mode that makes a table-conditioned raw grasp useless downstream.
    Each link mesh is convex in the X2 collision model, so a positive vertex
    clearance is a conservative support-plane clearance for its full hull.
    """

    return torch.einsum(
        "bni,i->bn", _collision_mesh_vertices_world(hand), normal
    )


def tabletop_plane_collision_mesh_minimum_clearance(
    hand: X2HandModel, table_plane: TabletopPlaneConstraint
) -> torch.Tensor:
    """Conservative collision-mesh-vertex support-plane diagnostic per row."""

    normal = table_plane.normalized(device=hand.device, dtype=hand.dtype)
    return (
        _tabletop_collision_mesh_clearances(hand, normal)
        - float(table_plane.offset_m)
    ).amin(dim=-1)


TABLE_PROJECTION_NUMERICAL_EPSILON_M = 1.0e-9


def _project_pose_to_table_clearance(
    hand: X2HandModel,
    pose: torch.Tensor,
    contact_indices: torch.Tensor,
    table_plane: TabletopPlaneConstraint,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project root translation onto the table-clearance half-space.

    The support plane and hand pose are expressed in the same object frame.
    Consequently, translating the root by ``delta * normal`` increases every
    calibrated collision-hull vertex clearance by exactly ``delta`` without
    changing root rotation, actuators, or selected contacts. This is a
    candidate-generation feasibility projection, not an exact table-motion
    safety result; the sequential runtime still performs its independent exact
    endpoint and continuous-motion checks.

    A fixed 1 nm numerical epsilon prevents floating-point roundoff from
    placing a projected vertex infinitesimally below the requested boundary.
    It is numerical tolerance, not a physical safety margin.
    """

    if pose.ndim != 2 or pose.shape[1] != X2HandModel.POSE_DIMENSION:
        raise ValueError("pose must have shape [batch, X2HandModel.POSE_DIMENSION]")
    hand.set_parameters(pose, contact_indices)
    minimum = tabletop_plane_collision_mesh_minimum_clearance(
        hand, table_plane
    ).detach()
    required = minimum.new_tensor(
        float(table_plane.clearance_m) + TABLE_PROJECTION_NUMERICAL_EPSILON_M
    )
    correction = torch.relu(required - minimum)
    normal = table_plane.normalized(device=pose.device, dtype=pose.dtype)
    projected = pose.detach().clone()
    projected[:, :3] += correction[:, None] * normal[None, :]
    projected.requires_grad_(True)
    hand.set_parameters(projected, contact_indices)
    return projected, correction


def cal_unselected_opposite_flex_energy(
    hand: X2HandModel,
    active_sides: Sequence[str],
    *,
    margin: float = 0.0,
    displacement_scale: float = 0.02,
) -> torch.Tensor:
    """Penalize unselected fingers that bend toward the inactive palm side.

    Bending is measured from the canonical open pose using each finger's real
    distal collision-hull center and FK.  For a back grasp the forbidden
    direction is the front palm normal; for a front grasp it is the back palm
    normal.  A finger with at least one selected contact is exempt.
    """

    if hand.hand_pose is None:
        raise RuntimeError("Hand parameters have not been materialized")
    if len(active_sides) != hand.hand_pose.shape[0]:
        raise ValueError("active_sides length must equal the hand batch size")
    if any(side not in ("front", "back") for side in active_sides):
        raise ValueError("Every active side must be front or back")
    if not math.isfinite(float(margin)) or margin < 0.0:
        raise ValueError("unselected finger margin must be finite and non-negative")
    if not math.isfinite(float(displacement_scale)) or displacement_scale <= 0.0:
        raise ValueError("unselected finger displacement scale must be finite and positive")

    current = hand.finger_distal_points_root()
    canonical = hand.canonical_finger_distal_points_root.to(current)
    displacement = current - canonical.unsqueeze(0)
    forbidden_normals = torch.stack(
        tuple(
            hand.back_palm_normal if side == "front" else hand.front_palm_normal
            for side in active_sides
        ),
        dim=0,
    ).to(displacement)
    forbidden_projection = torch.einsum(
        "bfi,bi->bf", displacement, forbidden_normals
    )
    normalized_excess = torch.relu(
        forbidden_projection - displacement.new_tensor(float(margin))
    ) / float(displacement_scale)
    unselected = ~hand.selected_finger_mask()
    return (normalized_excess.square() * unselected.to(displacement.dtype)).sum(dim=-1)


class X2MeshAnnealing:
    """Original gradient proposal/contact switch/simulated-annealing loop."""

    def __init__(
        self,
        hand: X2HandModel,
        policies: ContactPolicyInput,
        active_sides: Sequence[str],
        config: X2Config,
        *,
        seed: int,
    ) -> None:
        self.hand = hand
        self.config = config
        # Retain the caller-supplied value for compatibility/debugging while
        # all proposal logic uses the unambiguous batch-aligned expansion.
        self.policies = policies
        self.active_sides = tuple(active_sides)
        self.row_policies = _resolve_row_policies(policies, self.active_sides)
        self.switch_possibility = float(config.require("optimization.switch_possibility"))
        self.contact_closure_warmup_iterations = int(
            config.require("optimization.contact_closure_warmup_iterations")
        )
        self.starting_temperature = float(config.require("optimization.starting_temperature"))
        self.temperature_decay = float(config.require("optimization.temperature_decay"))
        self.annealing_period = int(config.require("optimization.annealing_period"))
        # Pose coordinates have different physical units.  Applying one
        # RMS-normalized scalar to all 21 coordinates makes a nominal 0.005
        # actuator/rotation step also become a 5 mm root translation.  That
        # can leap over the 3 mm selected-contact basin and restore the
        # initialization checkpoint even when a descent direction exists.
        # Explicit blockwise trust-region limits preserve the original
        # rotational and actuator scale while bounding translation in metres.
        self.step_size = float(config.require("optimization.step_size"))
        translation_step = float(
            config.require("optimization.coordinate_step_limits.translation_m")
        )
        rotation_step = float(
            config.require("optimization.coordinate_step_limits.rotation_6d")
        )
        actuator_step = float(
            config.require("optimization.coordinate_step_limits.actuator_rad")
        )
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (translation_step, rotation_step, actuator_step)
        ):
            raise ValueError("coordinate step limits must be finite and positive")
        self.coordinate_step_size = torch.tensor(
            [translation_step] * 3
            + [rotation_step] * 6
            + [actuator_step] * 12,
            device=hand.device,
            dtype=hand.dtype,
        )
        self.stepsize_period = int(config.require("optimization.stepsize_period"))
        self.mu = float(config.require("optimization.mu"))
        self.step = 0
        self.ema_grad = torch.zeros(X2HandModel.POSE_DIMENSION, device=hand.device, dtype=hand.dtype)
        self.rng = np.random.default_rng(seed)
        self.proposed_contact_changes = 0
        self.accepted_contact_changes = torch.zeros(
            (), device=hand.device, dtype=torch.long
        )
        self.steps_with_contact_resampling: list[int] = []

    def try_step(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.hand.hand_pose is None or self.hand.hand_pose.grad is None:
            raise RuntimeError("Energy.backward() must precede try_step()")
        old_pose = self.hand.hand_pose
        old_indices = self.hand.contact_point_indices
        old_grad = old_pose.grad.detach().clone()
        decay = self.temperature_decay ** (self.step // self.stepsize_period)
        scale = self.coordinate_step_size * decay
        self.ema_grad = self.mu * (old_pose.grad.square().mean(dim=0)) + (1.0 - self.mu) * self.ema_grad
        proposal = (
            old_pose
            - scale * old_pose.grad / (torch.sqrt(self.ema_grad) + 1.0e-6)
        ).detach()
        proposed_values = old_indices.detach().cpu().tolist()
        if len(self.row_policies) != len(proposed_values):
            raise RuntimeError(
                "Annealer row policies do not match the materialized hand batch"
            )
        changed = 0
        # A selected-contact realization gate is pointwise: all selected
        # markers must converge to the object, not merely have good aggregate
        # energy.  Let the hand close on one legal tuple first.  Without this
        # bounded warmup a 50% per-slot switch probability changes the target
        # on almost every early proposal, preventing precisely that closure.
        if self.step >= self.contact_closure_warmup_iterations:
            for batch_index, policy in enumerate(self.row_policies):
                values = tuple(int(v) for v in proposed_values[batch_index])
                for slot in range(old_indices.shape[1]):
                    if self.rng.random() < self.switch_possibility:
                        replacement = policy.resample_slot(values, slot, self.rng)
                        if replacement[slot] != values[slot]:
                            changed += 1
                        values = replacement
                proposed_values[batch_index] = list(values)
        indices = torch.tensor(
            proposed_values, device=old_indices.device, dtype=torch.long
        )
        canonical = torch.tensor(
            self.config.require("initialization.canonical_open_actuator"),
            device=proposal.device,
            dtype=proposal.dtype,
        )
        proposal = torch.cat(
            (
                proposal[:, :9],
                _freeze_unselected_finger_actuators(
                    self.hand,
                    proposal[:, 9:],
                    indices,
                    canonical,
                ),
            ),
            dim=1,
        ).requires_grad_(True)
        self.proposed_contact_changes += changed
        if changed:
            self.steps_with_contact_resampling.append(self.step)
        self.hand.set_parameters(proposal, indices)
        self.step += 1
        return old_pose, old_indices, old_grad, indices

    def accept_step(
        self,
        old_energy: torch.Tensor,
        new_energy: torch.Tensor,
        old_pose: torch.Tensor,
        old_indices: torch.Tensor,
        old_grad: torch.Tensor,
        proposed_indices: torch.Tensor,
        proposal_allowed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        temperature = self.starting_temperature * self.temperature_decay ** (
            self.step // self.annealing_period
        )
        alpha = torch.tensor(
            self.rng.random(len(old_energy)), device=old_energy.device, dtype=old_energy.dtype
        )
        finite = torch.isfinite(old_energy) & torch.isfinite(new_energy)
        safe_new_energy = torch.where(finite, new_energy, old_energy)
        probability = torch.exp(
            ((old_energy - safe_new_energy) / temperature).clamp(max=80.0)
        )
        accept = finite & (alpha < probability)
        if proposal_allowed is not None:
            if proposal_allowed.shape != accept.shape:
                raise ValueError("proposal_allowed must have one boolean per batch row")
            accept &= proposal_allowed.to(device=accept.device, dtype=torch.bool)
        proposal = self.hand.hand_pose
        with torch.no_grad():
            reject = ~accept
            proposal[reject] = old_pose[reject]
            proposed_indices[reject] = old_indices[reject]
            if proposal.grad is not None:
                proposal.grad[reject] = old_grad[reject]
        accepted_changed = (
            (proposed_indices != old_indices) & accept[:, None]
        ).sum()
        self.accepted_contact_changes += accepted_changed.detach()
        self.hand.set_parameters(proposal, proposed_indices)
        return accept, temperature

    def zero_grad(self) -> None:
        if self.hand.hand_pose is not None and self.hand.hand_pose.grad is not None:
            self.hand.hand_pose.grad.zero_()


def optimize_x2_mesh_batch(
    hand: X2HandModel,
    object_model: MeshObjectModel,
    active_sides: Sequence[str],
    policies: ContactPolicyInput,
    config: X2Config,
    *,
    n_iterations: int,
    seed: int,
    rng: np.random.Generator,
    table_plane: TabletopPlaneConstraint | None = None,
    reachability_anchor: ReachabilityPoseAnchor | None = None,
) -> MeshBatchResult:
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive")
    row_policies = _resolve_row_policies(policies, active_sides)
    pose, contact_indices = initialize_x2_convex_hull(
        hand,
        object_model,
        active_sides,
        row_policies,
        config,
        rng,
        table_plane=table_plane,
    )
    if reachability_anchor is not None:
        target = reachability_anchor.pose.to(device=hand.device, dtype=hand.dtype)
        if target.shape != pose.shape:
            raise ValueError(
                "reachability anchor pose must match the optimizer batch and X2 pose dimension"
            )
        actuator = pose[:, 9:]
        if reachability_anchor.actuator is not None:
            anchored_actuator = reachability_anchor.actuator.to(
                device=hand.device, dtype=hand.dtype
            )
            if anchored_actuator.shape != actuator.shape:
                raise ValueError(
                    "reachability anchor actuator seed must match the optimizer batch and X2 actuator dimension"
                )
            if not bool(torch.isfinite(anchored_actuator).all()):
                raise ValueError("reachability anchor actuator seed must be finite")
            actuator = anchored_actuator
        if reachability_anchor.contact_indices is not None:
            anchored_contacts = reachability_anchor.contact_indices.to(
                device=hand.device, dtype=torch.long
            )
            if anchored_contacts.shape != contact_indices.shape:
                raise ValueError(
                    "initial contact seed must match the optimizer batch and contact count"
                )
            for policy, selected in zip(row_policies, anchored_contacts):
                policy.validate(selected.detach().cpu().tolist())
            contact_indices = anchored_contacts
        # The source joint command is only an initialization; policy contact
        # selection and every joint remain mutable under the optimizer.
        canonical = torch.tensor(
            config.require("initialization.canonical_open_actuator"),
            device=actuator.device,
            dtype=actuator.dtype,
        )
        actuator = _freeze_unselected_finger_actuators(
            hand,
            actuator,
            contact_indices,
            canonical,
        )
        pose = torch.cat((target[:, :9], actuator), dim=1).detach().requires_grad_()
        hand.set_parameters(pose, contact_indices)
    if table_plane is not None:
        pose, _ = _project_pose_to_table_clearance(
            hand, pose, contact_indices, table_plane
        )
    initial_pose = pose.detach().clone()
    weights = config.require("optimization.weights")
    side_margin = float(config.require("optimization.side_margin"))
    normal_opposition_weight = float(
        config.require("optimization.normal_opposition_in_force_closure")
    )
    unselected_finger_margin = float(
        config.require("optimization.unselected_finger_opposite_flex.margin")
    )
    unselected_finger_scale = float(
        config.require("optimization.unselected_finger_opposite_flex.displacement_scale")
    )
    selected_contact_max_distance_m = float(
        config.require("generation.selected_contact_max_distance_m")
    )
    energy = cal_x2_mesh_energy(
        hand,
        object_model,
        active_sides,
        weights,
        side_margin=side_margin,
        normal_opposition_weight=normal_opposition_weight,
        unselected_finger_margin=unselected_finger_margin,
        unselected_finger_scale=unselected_finger_scale,
        table_plane=table_plane,
        selected_contact_max_distance_m=selected_contact_max_distance_m,
    )
    energy = _with_reachability_anchor(energy, hand, reachability_anchor)
    self_collision = hand.self_collision_diagnostics()
    initial_energy = _detach_mesh_energy(energy)

    feasibility_threshold = float(
        config.require("self_collision.feasibility_threshold")
    )
    diagnostic_threshold = float(self_collision.threshold)
    if diagnostic_threshold != feasibility_threshold:
        raise RuntimeError(
            "Self-collision diagnostics and optimizer feasibility thresholds disagree: "
            f"{diagnostic_threshold} != {feasibility_threshold}"
        )
    protection_enabled = bool(
        config.require("self_collision.feasibility_protection.enabled")
    )
    hard_threshold = float(
        config.require("self_collision.feasibility_protection.hard_threshold")
    )
    maximum_allowed_increase = float(
        config.require(
            "self_collision.feasibility_protection.maximum_allowed_increase"
        )
    )

    feasible_checkpoints = _CheckpointBank.empty(hand, energy, self_collision)
    fallback_checkpoints = _CheckpointBank.empty(hand, energy, self_collision)
    table_contact_checkpoints = (
        None
        if table_plane is None
        else _CheckpointBank.empty(hand, energy, self_collision)
    )
    initial_valid = _valid_checkpoint_rows(
        hand, energy, self_collision, row_policies, active_sides
    )
    fallback_checkpoints.update(
        _fallback_improvement_mask(
            fallback_checkpoints, initial_valid, energy, self_collision
        ),
        hand,
        energy,
        self_collision,
        step=0,
    )
    feasible_checkpoints.update(
        _feasible_improvement_mask(
            feasible_checkpoints, initial_valid, energy, self_collision
        ),
        hand,
        energy,
        self_collision,
        step=0,
    )
    if table_contact_checkpoints is not None:
        table_contact_checkpoints.update(
            _table_contact_improvement_mask(
                table_contact_checkpoints,
                initial_valid,
                energy,
                self_collision,
                requested_clearance_m=float(table_plane.clearance_m),
            ),
            hand,
            energy,
            self_collision,
            step=0,
        )
    initial_bidirectional_rows = initial_valid & self_collision.feasible.detach()
    initial_hand_object = _hand_object_penetration_diagnostics(
        hand,
        object_model,
        energy,
        initial_bidirectional_rows,
    )
    sparse_checkpoint_pool = _SparseCheckpointPool.empty(
        hand, capacity=BIDIRECTIONAL_CHECKPOINT_TOP_K
    )
    sparse_checkpoint_pool.update(
        initial_bidirectional_rows,
        hand,
        energy,
        initial_hand_object,
        step=0,
    )

    energy.total.sum().backward()
    optimizer = X2MeshAnnealing(
        hand, row_policies, active_sides, config, seed=seed + 7919
    )
    accepted_total = torch.zeros((), device=hand.device, dtype=torch.long)
    temperatures: list[float] = []
    nonfinite_proposal_rows = torch.zeros((), device=hand.device, dtype=torch.long)
    invalid_checkpoint_candidate_rows = (~initial_valid).sum().detach()
    feasibility_protection_rejections = torch.zeros(
        (), device=hand.device, dtype=torch.long
    )
    sparse_bidirectional_query_calls = int(
        bool(initial_bidirectional_rows.any().detach().cpu())
    )
    sparse_bidirectional_rows_evaluated = (
        initial_bidirectional_rows.sum().detach()
    )
    for _ in range(n_iterations):
        old_total = energy.total.detach().clone()
        old_self_collision_maximum = self_collision.maximum_penetration.detach().clone()
        old_pose, old_indices, old_grad, proposed_indices = optimizer.try_step()
        optimizer.zero_grad()
        if table_plane is not None:
            if hand.hand_pose is None:
                raise RuntimeError("annealing proposal did not materialize a hand pose")
            _project_pose_to_table_clearance(
                hand, hand.hand_pose, proposed_indices, table_plane
            )
        proposed_energy = cal_x2_mesh_energy(
            hand,
            object_model,
            active_sides,
            weights,
            side_margin=side_margin,
            normal_opposition_weight=normal_opposition_weight,
            unselected_finger_margin=unselected_finger_margin,
            unselected_finger_scale=unselected_finger_scale,
            table_plane=table_plane,
            selected_contact_max_distance_m=selected_contact_max_distance_m,
        )
        proposed_energy = _with_reachability_anchor(
            proposed_energy, hand, reachability_anchor
        )
        proposed_self_collision = hand.self_collision_diagnostics()
        proposed_valid = _valid_checkpoint_rows(
            hand,
            proposed_energy,
            proposed_self_collision,
            row_policies,
            active_sides,
        )
        invalid_checkpoint_candidate_rows += (~proposed_valid).sum().detach()
        # Checkpoints see every evaluated proposal, even proposals that the
        # Metropolis step subsequently rejects.
        fallback_checkpoints.update(
            _fallback_improvement_mask(
                fallback_checkpoints,
                proposed_valid,
                proposed_energy,
                proposed_self_collision,
            ),
            hand,
            proposed_energy,
            proposed_self_collision,
            step=optimizer.step,
        )
        feasible_checkpoints.update(
            _feasible_improvement_mask(
                feasible_checkpoints,
                proposed_valid,
                proposed_energy,
                proposed_self_collision,
            ),
            hand,
            proposed_energy,
            proposed_self_collision,
            step=optimizer.step,
        )
        if table_contact_checkpoints is not None:
            table_contact_checkpoints.update(
                _table_contact_improvement_mask(
                    table_contact_checkpoints,
                    proposed_valid,
                    proposed_energy,
                    proposed_self_collision,
                    requested_clearance_m=float(table_plane.clearance_m),
                ),
                hand,
                proposed_energy,
                proposed_self_collision,
                step=optimizer.step,
            )
        if optimizer.step % BIDIRECTIONAL_CHECKPOINT_PERIOD == 0:
            bidirectional_rows = (
                proposed_valid & proposed_self_collision.feasible.detach()
            )
            proposed_hand_object = _hand_object_penetration_diagnostics(
                hand,
                object_model,
                proposed_energy,
                bidirectional_rows,
            )
            sparse_checkpoint_pool.update(
                bidirectional_rows,
                hand,
                proposed_energy,
                proposed_hand_object,
                step=optimizer.step,
            )
            sparse_bidirectional_query_calls += int(
                bool(bidirectional_rows.any().detach().cpu())
            )
            sparse_bidirectional_rows_evaluated += (
                bidirectional_rows.sum().detach()
            )

        proposal_finite = _finite_candidate_rows(
            hand, proposed_energy, proposed_self_collision
        )
        nonfinite_proposal_rows += (~proposal_finite).sum().detach()
        proposal_allowed, serious_deterioration = _proposal_protection_masks(
            old_self_collision_maximum,
            proposed_self_collision.maximum_penetration,
            proposal_finite,
            enabled=protection_enabled,
            hard_threshold=hard_threshold,
            maximum_allowed_increase=maximum_allowed_increase,
        )
        feasibility_protection_rejections += serious_deterioration.sum().detach()
        accept, temperature = optimizer.accept_step(
            old_total,
            proposed_energy.total.detach(),
            old_pose,
            old_indices,
            old_grad,
            proposed_indices,
            proposal_allowed,
        )
        accepted_total += accept.sum().detach()
        temperatures.append(float(temperature))
        energy = cal_x2_mesh_energy(
            hand,
            object_model,
            active_sides,
            weights,
            side_margin=side_margin,
            normal_opposition_weight=normal_opposition_weight,
            unselected_finger_margin=unselected_finger_margin,
            unselected_finger_scale=unselected_finger_scale,
            table_plane=table_plane,
            selected_contact_max_distance_m=selected_contact_max_distance_m,
        )
        energy = _with_reachability_anchor(energy, hand, reachability_anchor)
        self_collision = hand.self_collision_diagnostics()
        current_finite = _finite_candidate_rows(hand, energy, self_collision)
        if not bool(current_finite.all().detach().cpu()):
            raise FloatingPointError(
                "X2 mesh optimization retained a NaN or infinite state after rollback"
            )
        optimizer.zero_grad()
        energy.total.sum().backward()

    current_valid = _valid_checkpoint_rows(
        hand, energy, self_collision, row_policies, active_sides
    )
    final_live_bidirectional_rows = (
        current_valid & self_collision.feasible.detach()
    )
    final_live_hand_object = _hand_object_penetration_diagnostics(
        hand,
        object_model,
        energy,
        final_live_bidirectional_rows,
    )
    sparse_checkpoint_pool.update(
        final_live_bidirectional_rows,
        hand,
        energy,
        final_live_hand_object,
        step=n_iterations,
    )
    sparse_bidirectional_query_calls += int(
        bool(final_live_bidirectional_rows.any().detach().cpu())
    )
    sparse_bidirectional_rows_evaluated += (
        final_live_bidirectional_rows.sum().detach()
    )

    if not bool(fallback_checkpoints.valid.all().detach().cpu()):
        missing = torch.nonzero(
            ~fallback_checkpoints.valid, as_tuple=False
        ).flatten().cpu().tolist()
        raise FloatingPointError(
            f"No finite, in-limit checkpoint exists for batch rows {missing}"
        )

    # Dense auditing is deliberately deferred until optimization finishes.
    # The sparse pool preserves both low-energy and low-penetration candidates
    # per row.  Audit every retained candidate and select the lowest-energy
    # strict <1 mm state (dense max/total break ties).  If none passes, retain
    # the candidate with the best dense max/total/energy tuple.  The best-energy
    # self-feasible bank is appended when it is not already present in the
    # sparse pool, so proposals between periodic sparse checkpoints are not
    # lost.
    batch_size = len(fallback_checkpoints.valid)
    selected_pose = fallback_checkpoints.pose.detach().clone()
    selected_contacts = fallback_checkpoints.contact_indices.detach().clone()
    selected_step = fallback_checkpoints.step.detach().clone()
    dense_feasible_selected = torch.zeros(
        batch_size, device=hand.device, dtype=torch.bool
    )
    dense_candidate_evaluated = torch.zeros_like(dense_feasible_selected)
    has_self_feasible_candidate = torch.zeros_like(dense_feasible_selected)
    best_dense_maximum = torch.full(
        (batch_size,), torch.inf, device=hand.device, dtype=hand.dtype
    )
    best_dense_total = torch.full_like(best_dense_maximum, torch.inf)
    best_dense_energy = torch.full_like(best_dense_maximum, torch.inf)
    best_dense_feasible_energy = torch.full_like(best_dense_maximum, torch.inf)
    best_dense_feasible_maximum = torch.full_like(best_dense_maximum, torch.inf)
    best_dense_feasible_total = torch.full_like(best_dense_maximum, torch.inf)
    best_dense_feasible_contact_realized = torch.zeros_like(
        dense_feasible_selected
    )
    best_dense_feasible_output_priority = torch.full(
        (batch_size,), -1, device=hand.device, dtype=torch.long
    )
    dense_bidirectional_query_calls = 0
    dense_bidirectional_rows_evaluated = 0

    def audit_dense_candidate(
        candidate_valid: torch.Tensor,
        candidate_pose: torch.Tensor,
        candidate_contacts: torch.Tensor,
        candidate_energy: torch.Tensor,
        candidate_step: torch.Tensor,
    ) -> None:
        nonlocal dense_bidirectional_query_calls
        nonlocal dense_bidirectional_rows_evaluated
        has_self_feasible_candidate.logical_or_(candidate_valid)
        evaluate_rows = candidate_valid
        if not bool(evaluate_rows.any().detach().cpu()):
            return

        # set_parameters is batch-shaped.  Invalid source rows are populated
        # from the always-valid fallback bank and excluded from the audit.
        materialized_pose = _select_rows(
            candidate_valid, candidate_pose, fallback_checkpoints.pose
        ).detach().clone()
        materialized_contacts = _select_rows(
            candidate_valid,
            candidate_contacts,
            fallback_checkpoints.contact_indices,
        ).detach().clone()
        hand.set_parameters(materialized_pose, materialized_contacts)
        candidate_diagnostics = _dense_hand_object_penetration_diagnostics(
            hand, object_model, evaluate_rows
        )
        dense_bidirectional_query_calls += 1
        dense_bidirectional_rows_evaluated += int(
            evaluate_rows.sum().detach().cpu()
        )
        dense_candidate_evaluated.logical_or_(evaluate_rows)

        finite = (
            torch.isfinite(candidate_diagnostics.maximum_penetration)
            & torch.isfinite(candidate_diagnostics.total_penetration)
        )
        nonfinite = evaluate_rows & ~finite
        if bool(nonfinite.any().detach().cpu()):
            rows = torch.nonzero(nonfinite, as_tuple=False).flatten().cpu().tolist()
            raise FloatingPointError(
                f"Dense hand-object audit is non-finite for batch rows {rows}"
            )

        candidate_feasible = evaluate_rows & candidate_diagnostics.feasible
        candidate_infeasible = evaluate_rows & ~candidate_diagnostics.feasible
        candidate_maximum = candidate_diagnostics.maximum_penetration
        candidate_total = candidate_diagnostics.total_penetration
        # ``selected_contact_realization`` is already a hard output contract.
        # Prefer a checkpoint satisfying it before comparing soft total
        # energy; otherwise a collision-feasible but unrealized keypoint tuple
        # can evict the only record that the caller is allowed to emit.
        if hand.contact_points is None:
            raise RuntimeError("dense checkpoint lacks selected contact points")
        candidate_contact_distance, _ = object_model.cal_distance(
            hand.contact_points
        )
        candidate_contact_realized = evaluate_rows & (
            candidate_contact_distance.detach().abs().amax(dim=-1)
            <= float(selected_contact_max_distance_m)
        )
        if table_plane is None:
            candidate_table_nonpenetrating = evaluate_rows
            candidate_requested_clearance = evaluate_rows
        else:
            candidate_table_clearance = (
                tabletop_plane_collision_mesh_minimum_clearance(hand, table_plane)
                .detach()
            )
            candidate_table_nonpenetrating = evaluate_rows & (
                candidate_table_clearance >= 0.0
            )
            candidate_requested_clearance = evaluate_rows & (
                candidate_table_clearance >= float(table_plane.clearance_m)
            )
        candidate_output_eligible = (
            candidate_contact_realized & candidate_table_nonpenetrating
        )
        # 0: not emit-eligible, 1: source-plane nonpenetrating, 2: also reaches
        # the requested generation margin. Dense hand-object feasibility is
        # still the outer class and cannot be traded against this preference.
        candidate_output_priority = (
            candidate_output_eligible.to(torch.long)
            + (
                candidate_output_eligible & candidate_requested_clearance
            ).to(torch.long)
        )
        feasible_better_energy = candidate_energy < best_dense_feasible_energy
        feasible_equal_energy = candidate_energy == best_dense_feasible_energy
        feasible_better_maximum = (
            candidate_maximum < best_dense_feasible_maximum
        )
        feasible_equal_maximum = (
            candidate_maximum == best_dense_feasible_maximum
        )
        feasible_better_total = candidate_total < best_dense_feasible_total
        better_output_class = (
            candidate_output_priority > best_dense_feasible_output_priority
        )
        same_output_class = (
            candidate_output_priority == best_dense_feasible_output_priority
        )
        better_feasible = candidate_feasible & (
            ~dense_feasible_selected
            | better_output_class
            | (
                same_output_class
                & (
                    feasible_better_energy
                    | (
                        feasible_equal_energy
                        & (
                            feasible_better_maximum
                            | (feasible_equal_maximum & feasible_better_total)
                        )
                    )
                )
            )
        )
        better_maximum = candidate_maximum < best_dense_maximum
        equal_maximum = candidate_maximum == best_dense_maximum
        better_total = candidate_total < best_dense_total
        equal_total = candidate_total == best_dense_total
        better_energy = candidate_energy < best_dense_energy
        better_fallback = candidate_infeasible & ~dense_feasible_selected & (
            better_maximum
            | (
                equal_maximum
                & (better_total | (equal_total & better_energy))
            )
        )
        select_candidate = better_feasible | better_fallback
        with torch.no_grad():
            selected_pose[select_candidate] = candidate_pose[select_candidate]
            selected_contacts[select_candidate] = candidate_contacts[
                select_candidate
            ]
            selected_step[select_candidate] = candidate_step[select_candidate]
            best_dense_maximum[better_fallback] = candidate_maximum[
                better_fallback
            ]
            best_dense_total[better_fallback] = candidate_total[better_fallback]
            best_dense_energy[better_fallback] = candidate_energy[
                better_fallback
            ]
            best_dense_feasible_energy[better_feasible] = candidate_energy[
                better_feasible
            ]
            best_dense_feasible_maximum[better_feasible] = candidate_maximum[
                better_feasible
            ]
            best_dense_feasible_total[better_feasible] = candidate_total[
                better_feasible
            ]
            best_dense_feasible_contact_realized[better_feasible] = (
                candidate_contact_realized[better_feasible]
            )
            best_dense_feasible_output_priority[better_feasible] = (
                candidate_output_priority[better_feasible]
            )
            dense_feasible_selected.logical_or_(candidate_feasible)

    if table_contact_checkpoints is not None:
        audit_dense_candidate(
            table_contact_checkpoints.valid,
            table_contact_checkpoints.pose,
            table_contact_checkpoints.contact_indices,
            table_contact_checkpoints.energy["total"],
            table_contact_checkpoints.step,
        )

    pool_matches_feasible = (
        sparse_checkpoint_pool.valid
        & (
            sparse_checkpoint_pool.pose
            == feasible_checkpoints.pose[:, None, :]
        ).all(dim=-1)
        & (
            sparse_checkpoint_pool.contact_indices
            == feasible_checkpoints.contact_indices[:, None, :]
        ).all(dim=-1)
    ).any(dim=1)
    unique_feasible_candidate = (
        feasible_checkpoints.valid & ~pool_matches_feasible
    )
    audit_dense_candidate(
        unique_feasible_candidate,
        feasible_checkpoints.pose,
        feasible_checkpoints.contact_indices,
        feasible_checkpoints.energy["total"],
        feasible_checkpoints.step,
    )

    for slot in range(sparse_checkpoint_pool.capacity):
        audit_dense_candidate(
            sparse_checkpoint_pool.valid[:, slot],
            sparse_checkpoint_pool.pose[:, slot],
            sparse_checkpoint_pool.contact_indices[:, slot],
            sparse_checkpoint_pool.energy_total[:, slot],
            sparse_checkpoint_pool.step[:, slot],
        )

    missing_dense_audit = (
        has_self_feasible_candidate & ~dense_candidate_evaluated
    )
    if bool(missing_dense_audit.any().detach().cpu()):
        rows = torch.nonzero(
            missing_dense_audit, as_tuple=False
        ).flatten().cpu().tolist()
        raise RuntimeError(
            f"Self-feasible checkpoints escaped dense auditing for rows {rows}"
        )

    use_bidirectional_feasible = dense_feasible_selected
    use_bidirectional_fallback = (
        has_self_feasible_candidate & ~use_bidirectional_feasible
    )
    use_fallback = ~(
        use_bidirectional_feasible | use_bidirectional_fallback
    )
    use_feasible = torch.zeros_like(use_fallback)
    restored_pose = selected_pose.detach().clone().requires_grad_(True)
    restored_contacts = selected_contacts.detach().clone()
    restored_step = selected_step.detach().clone()

    # Pose/contact are authoritative.  Rebuild all derived FK state, energy,
    # and collision diagnostics from the selected checkpoint.
    hand.set_parameters(restored_pose, restored_contacts)
    final_energy = cal_x2_mesh_energy(
        hand,
        object_model,
        active_sides,
        weights,
        side_margin=side_margin,
        normal_opposition_weight=normal_opposition_weight,
        unselected_finger_margin=unselected_finger_margin,
        unselected_finger_scale=unselected_finger_scale,
        table_plane=table_plane,
        selected_contact_max_distance_m=selected_contact_max_distance_m,
    )
    final_self_collision = hand.self_collision_diagnostics()
    restored_valid = _valid_checkpoint_rows(
        hand, final_energy, final_self_collision, row_policies, active_sides
    )
    if not bool(restored_valid.all().detach().cpu()):
        raise RuntimeError("Restored checkpoint failed its materialized validity audit")
    selected_self_feasible = (
        use_bidirectional_feasible | use_bidirectional_fallback
    )
    if bool(
        (
            selected_self_feasible & ~final_self_collision.feasible
        ).any().detach().cpu()
    ):
        raise RuntimeError("Dense-selected checkpoint lost self-collision feasibility")

    final_hand_object = _dense_hand_object_penetration_diagnostics(
        hand,
        object_model,
        restored_valid,
    )
    dense_bidirectional_query_calls += 1
    dense_bidirectional_rows_evaluated += int(
        restored_valid.sum().detach().cpu()
    )
    if bool(
        (
            use_bidirectional_feasible & ~final_hand_object.feasible
        ).any().detach().cpu()
    ):
        raise RuntimeError("A dense-feasible checkpoint failed the final dense audit")
    if bool(
        (
            use_bidirectional_fallback & final_hand_object.feasible
        ).any().detach().cpu()
    ):
        raise RuntimeError("A dense fallback unexpectedly passed the final dense audit")
    maximum_penetration = final_hand_object.maximum_penetration
    diagnostics = {
        "algorithm": "dexgraspnet_rmsprop_proposal_simulated_annealing",
        "iterations": n_iterations,
        "contact_closure_warmup_iterations": (
            optimizer.contact_closure_warmup_iterations
        ),
        "coordinate_step_limits": {
            "translation_m": float(optimizer.coordinate_step_size[0].detach().cpu()),
            "rotation_6d": float(optimizer.coordinate_step_size[3].detach().cpu()),
            "actuator_rad": float(optimizer.coordinate_step_size[9].detach().cpu()),
            "semantics": "BLOCKWISE_RMS_NORMALIZED_TRUST_REGION_V1",
        },
        "proposed_contact_changes": optimizer.proposed_contact_changes,
        "accepted_contact_changes": int(
            optimizer.accepted_contact_changes.detach().cpu()
        ),
        "steps_with_contact_resampling": optimizer.steps_with_contact_resampling,
        "mean_accepted_per_step": float(accepted_total.detach().cpu()) / n_iterations,
        "initial_temperature": temperatures[0] if temperatures else None,
        "final_temperature": temperatures[-1] if temperatures else None,
        "bidirectional_checkpoint_period": BIDIRECTIONAL_CHECKPOINT_PERIOD,
        "bidirectional_checkpoint_pool_capacity": BIDIRECTIONAL_CHECKPOINT_TOP_K,
        "bidirectional_penetration_threshold": HAND_OBJECT_PENETRATION_THRESHOLD,
        "dense_hand_surface_samples_per_set": (
            DENSE_HAND_SURFACE_SAMPLES_PER_SET
        ),
        "dense_hand_surface_samples_per_link": (
            3 * DENSE_HAND_SURFACE_SAMPLES_PER_SET
        ),
        "dense_hand_surface_point_count": (
            len(hand.backend.link_names)
            * 3
            * DENSE_HAND_SURFACE_SAMPLES_PER_SET
        ),
        "dense_object_surface_samples": DENSE_OBJECT_SURFACE_SAMPLES,
        "sparse_bidirectional_query_calls": sparse_bidirectional_query_calls,
        "sparse_bidirectional_rows_evaluated": int(
            sparse_bidirectional_rows_evaluated.detach().cpu()
        ),
        "dense_bidirectional_query_calls": dense_bidirectional_query_calls,
        "dense_bidirectional_rows_evaluated": (
            dense_bidirectional_rows_evaluated
        ),
        # Retain the original aggregate names until the pipeline revision is
        # formally bumped and downstream consumers migrate.
        "bidirectional_exact_query_calls": (
            sparse_bidirectional_query_calls + dense_bidirectional_query_calls
        ),
        "bidirectional_rows_evaluated": (
            int(sparse_bidirectional_rows_evaluated.detach().cpu())
            + dense_bidirectional_rows_evaluated
        ),
        "bidirectional_feasible_checkpoint_count": int(
            use_bidirectional_feasible.sum().detach().cpu()
        ),
        "bidirectional_fallback_checkpoint_count": int(
            use_bidirectional_fallback.sum().detach().cpu()
        ),
        "feasible_checkpoint_count": int(use_feasible.sum().detach().cpu()),
        "fallback_checkpoint_count": int(use_fallback.sum().detach().cpu()),
        "table_contact_checkpoint_count": int(
            0
            if table_contact_checkpoints is None
            else table_contact_checkpoints.valid.sum().detach().cpu()
        ),
        "nonfinite_proposal_rows": int(nonfinite_proposal_rows.detach().cpu()),
        "invalid_checkpoint_candidate_rows": int(
            invalid_checkpoint_candidate_rows.detach().cpu()
        ),
        "feasibility_protection_rejections": int(
            feasibility_protection_rejections.detach().cpu()
        ),
    }
    checkpoint_type = tuple(
        "bidirectional_feasible"
        if bool(bidirectional_feasible)
        else "bidirectional_fallback"
        if bool(bidirectional_fallback)
        else "fallback"
        for bidirectional_feasible, bidirectional_fallback in zip(
            use_bidirectional_feasible.detach().cpu().tolist(),
            use_bidirectional_fallback.detach().cpu().tolist(),
        )
    )
    return MeshBatchResult(
        initial_pose=initial_pose,
        final_pose=hand.hand_pose.detach().clone(),
        initial_energy=initial_energy,
        final_energy=_detach_mesh_energy(final_energy),
        contact_indices=hand.contact_point_indices.detach().clone(),
        active_sides=tuple(active_sides),
        optimizer_diagnostics=diagnostics,
        maximum_penetration=maximum_penetration.detach().clone(),
        hand_object_penetration=final_hand_object.detached(),
        self_collision_diagnostics=final_self_collision.detached(),
        checkpoint_type=checkpoint_type,
        restored_step=restored_step,
        actuator_positions=hand.actuator_positions.detach().clone(),
        joint_positions=hand.joint_positions.detach().clone(),
    )


def rotation_matrix_to_quaternion_wxyz(matrix: np.ndarray) -> list[float]:
    """Stable scalar-first quaternion conversion for JSON serialization."""

    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(matrix).as_quat()
    quaternion = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return [float(v) for v in quaternion]


def make_sample_records(
    hand: X2HandModel,
    object_model: MeshObjectModel,
    result: MeshBatchResult,
    candidates: Sequence[GenericContactCandidate],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    hand.set_parameters(result.final_pose.clone(), result.contact_indices.clone())
    if hand.contact_points is None:
        raise RuntimeError("Hand contacts have not been materialized")
    contact_distance, _ = object_model.cal_distance(hand.contact_points)
    contact_distance = contact_distance.detach().abs()
    contact_threshold = float(
        hand.config.require("generation.selected_contact_max_distance_m")
    )
    if not math.isfinite(contact_threshold) or contact_threshold < 0.0:
        raise ValueError(
            "generation.selected_contact_max_distance_m must be finite and non-negative"
        )
    contact_candidates_path = hand.config.configured_path(
        "contact_candidates.path", must_exist=True
    )
    contact_candidates_sha256 = hashlib.sha256(
        contact_candidates_path.read_bytes()
    ).hexdigest()
    records: list[dict[str, Any]] = []
    for index, side in enumerate(result.active_sides):
        if not bool(result.hand_object_penetration.evaluated[index].cpu()):
            raise RuntimeError(
                "Formal v5 records require a completed final dense hand-object audit"
            )
        actuator = result.actuator_positions[index].detach()
        joints = result.joint_positions[index].detach()
        rotation = hand.global_rotation[index].detach().cpu().numpy()
        selected = [candidates[int(v)] for v in result.contact_indices[index].cpu().tolist()]
        contacts = []
        for slot, candidate in enumerate(selected):
            contacts.append(
                {
                    **candidate.to_dict(),
                    "world_position": [float(v) for v in hand.contact_points[index, slot].detach().cpu()],
                    "world_surface_normal": [float(v) for v in hand.contact_normals[index, slot].detach().cpu()],
                    "object_surface_distance_m": float(
                        contact_distance[index, slot].cpu()
                    ),
                }
            )
        maximum_contact_distance = float(contact_distance[index].amax().cpu())
        contact_realized = maximum_contact_distance <= contact_threshold
        values = torch.cat(
            (
                result.final_pose[index],
                actuator,
                joints,
                result.final_energy.total[index : index + 1],
                result.self_collision_diagnostics.maximum_penetration[
                    index : index + 1
                ],
                result.self_collision_diagnostics.total_penetration[
                    index : index + 1
                ],
                result.hand_object_penetration.forward_total_penetration[
                    index : index + 1
                ],
                result.hand_object_penetration.forward_maximum_penetration[
                    index : index + 1
                ],
                result.hand_object_penetration.reverse_total_penetration[
                    index : index + 1
                ],
                result.hand_object_penetration.reverse_maximum_penetration[
                    index : index + 1
                ],
            )
        )
        finite = bool(torch.isfinite(values).all().detach().cpu())
        worst_pair = result.self_collision_diagnostics.worst_pair[index]
        optimization = copy.deepcopy(result.optimizer_diagnostics)
        optimization.update(
            {
                "restored_checkpoint": result.checkpoint_type[index],
                "feasible_checkpoint_found": (
                    bool(
                        result.self_collision_diagnostics.feasible[index].cpu()
                    )
                ),
                "self_collision_feasible_checkpoint_found": bool(
                    result.self_collision_diagnostics.feasible[index].cpu()
                ),
                "bidirectional_feasible_checkpoint_found": bool(
                    result.hand_object_penetration.feasible[index].cpu()
                ),
                "restored_step": int(result.restored_step[index].cpu()),
            }
        )
        records.append(
            {
                "schema_version": 1,
                "pipeline_revision": str(hand.config.require("pipeline_revision")),
                "sample_index": index,
                "active_side": side,
                "object": {
                    "mesh_path": str(object_model.mesh_path),
                    "scale": object_model.scale,
                    "watertight": True,
                },
                "provenance": {
                    "contact_candidates_sha256": contact_candidates_sha256,
                },
                "hand_pose": {
                    "translation": [float(v) for v in hand.global_translation[index].detach().cpu()],
                    "rotation_matrix": [[float(v) for v in row] for row in rotation],
                    "quaternion_wxyz": rotation_matrix_to_quaternion_wxyz(rotation),
                },
                "actuator_names": list(hand.actuator_names),
                "actuator": [float(v) for v in actuator.cpu()],
                "joint_names": list(hand.full_joint_names),
                "joint": [float(v) for v in joints.cpu()],
                "selected_contact_ids": [candidate.point_id for candidate in selected],
                "selected_contacts": contacts,
                "selected_contact_realization": {
                    "status": "PASS" if contact_realized else "UNMET",
                    "maximum_distance_m": maximum_contact_distance,
                    "threshold_m": contact_threshold,
                    "all_selected_contacts_within_threshold": contact_realized,
                    "frame": "MESH_OBJECT_FRAME",
                    "generation_admission_only": True,
                    "not_collision_or_physical_proof": True,
                },
                "energy": {
                    "initial_total": float(result.initial_energy.total[index].cpu()),
                    "total": float(result.final_energy.total[index].cpu()),
                    "terms": result.final_energy.detached_terms(index),
                    "mean_hand_object_normal_dot": float(
                        result.final_energy.normal_opposition[index].cpu()
                    ),
                    "normal_opposition_penalty": float(
                        result.final_energy.normal_opposition_penalty[index].cpu()
                    ),
                },
                "maximum_penetration": float(result.maximum_penetration[index].cpu()),
                "hand_object_penetration": {
                    "evaluation_mode": "dense_bidirectional",
                    "evaluated": bool(
                        result.hand_object_penetration.evaluated[index].cpu()
                    ),
                    "hand_surface_samples_per_set": (
                        DENSE_HAND_SURFACE_SAMPLES_PER_SET
                    ),
                    "hand_surface_samples_per_link": (
                        3 * DENSE_HAND_SURFACE_SAMPLES_PER_SET
                    ),
                    "hand_surface_point_count": (
                        len(hand.backend.link_names)
                        * 3
                        * DENSE_HAND_SURFACE_SAMPLES_PER_SET
                    ),
                    "object_surface_samples": DENSE_OBJECT_SURFACE_SAMPLES,
                    "forward_total_penetration": float(
                        result.hand_object_penetration.forward_total_penetration[
                            index
                        ].cpu()
                    ),
                    "forward_maximum_penetration": float(
                        result.hand_object_penetration.forward_maximum_penetration[
                            index
                        ].cpu()
                    ),
                    "reverse_total_penetration": float(
                        result.hand_object_penetration.reverse_total_penetration[
                            index
                        ].cpu()
                    ),
                    "reverse_maximum_penetration": float(
                        result.hand_object_penetration.reverse_maximum_penetration[
                            index
                        ].cpu()
                    ),
                    "total_penetration": float(
                        result.hand_object_penetration.total_penetration[index].cpu()
                    ),
                    "maximum_penetration": float(
                        result.hand_object_penetration.maximum_penetration[index].cpu()
                    ),
                    "feasible": bool(
                        result.hand_object_penetration.feasible[index].cpu()
                    ),
                    "threshold": float(
                        result.hand_object_penetration.threshold
                    ),
                },
                "self_collision": {
                    "maximum_penetration": float(
                        result.self_collision_diagnostics.maximum_penetration[
                            index
                        ].cpu()
                    ),
                    "total_penetration": float(
                        result.self_collision_diagnostics.total_penetration[
                            index
                        ].cpu()
                    ),
                    "worst_pair": list(worst_pair) if worst_pair is not None else None,
                    "feasible": bool(
                        result.self_collision_diagnostics.feasible[index].cpu()
                    ),
                    "threshold": float(
                        result.self_collision_diagnostics.threshold
                    ),
                },
                "seed": int(seed),
                "optimization": optimization,
                "finite": finite,
                "success": False,
                "simulation_success": False,
                "validation": {"status": "not_run", "backend": None},
            }
        )
    return records


__all__ = [
    "MeshBatchResult",
    "MeshEnergy",
    "ReachabilityPoseAnchor",
    "TabletopPlaneConstraint",
    "X2MeshAnnealing",
    "cal_unselected_opposite_flex_energy",
    "cal_x2_mesh_energy",
    "initialize_x2_convex_hull",
    "make_sample_records",
    "optimize_x2_mesh_batch",
    "tabletop_plane_collision_mesh_minimum_clearance",
    "tabletop_plane_minimum_clearance",
]
