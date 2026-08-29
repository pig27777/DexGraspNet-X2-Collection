#!/usr/bin/env python3
"""Generate side-conditioned single-object X2 grasps for a watertight mesh."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_N_ITERATIONS = 6000
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_generation.x2_mesh_generator import (
    DENSE_HAND_SURFACE_SAMPLES_PER_SET,
    DENSE_OBJECT_SURFACE_SAMPLES,
    HAND_OBJECT_PENETRATION_THRESHOLD,
    ReachabilityPoseAnchor,
    TabletopPlaneConstraint,
    make_sample_records,
    optimize_x2_mesh_batch,
    tabletop_plane_collision_mesh_minimum_clearance,
    tabletop_plane_minimum_clearance,
)


def _source_table_nonpenetrating(clearance_m: float) -> bool:
    """Source generation may reject plane crossing, not certify a margin."""

    return math.isfinite(float(clearance_m)) and float(clearance_m) >= 0.0
from grasp_generation.utils.mesh_object_model import MeshObjectModel
from grasp_generation.utils.x2_config import X2Config, load_x2_mesh_config
from grasp_generation.utils.x2_hand_model import X2HandModel
from grasp_generation.utils.x2_mesh_contacts import (
    FINGER_NAMES,
    GenericDexterousContactPolicy,
    load_generic_contact_candidates,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DexGraspNet-style generic mesh grasp generation for X2"
    )
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument("--side", choices=("front", "back", "both", "any"), default="any")
    parser.add_argument("--num-grasps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-contact", type=int, default=4)
    parser.add_argument(
        "--finger-count",
        type=int,
        choices=range(1, 6),
        help="require exactly this many distinct non-palm fingers in every sample",
    )
    parser.add_argument(
        "--finger-names",
        nargs="+",
        choices=FINGER_NAMES,
        help="require this exact non-palm finger set (implies its count)",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=DEFAULT_N_ITERATIONS,
        help=(
            "annealing iterations; defaults to the original DexGraspNet "
            f"order of magnitude ({DEFAULT_N_ITERATIONS})"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("data/x2_mesh_grasps"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--object-scale", type=float)
    parser.add_argument("--surface-samples", type=int)
    parser.add_argument(
        "--initialization-distance-lower-m",
        type=float,
        help="optional lower palm-side surface offset for contact-conditioned initialization",
    )
    parser.add_argument(
        "--initialization-distance-upper-m",
        type=float,
        help="optional upper palm-side surface offset for contact-conditioned initialization",
    )
    parser.add_argument(
        "--initialization-minimum-surface-z-m",
        type=float,
        help=(
            "optional object-frame lower bound for convex-hull initialization "
            "samples; use it to initialize table grasps on the exposed upper "
            "part of an upright object without changing the grasp objective"
        ),
    )
    parser.add_argument(
        "--table-plane-z-m",
        type=float,
        help=(
            "optional support-plane height in the mesh/object frame. When set, "
            "the optimizer penalizes sampled hand geometry below the requested "
            "clearance; exact tabletop admission remains downstream."
        ),
    )
    parser.add_argument(
        "--table-plane-normal",
        type=float,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=(0.0, 0.0, 1.0),
        help=(
            "support-plane normal in mesh/object coordinates; normalized by "
            "the generator and used only with --table-plane-z-m"
        ),
    )
    parser.add_argument(
        "--table-clearance-m",
        type=float,
        default=0.002,
        help="requested sampled hand/support-plane clearance when table conditioning is enabled",
    )
    parser.add_argument(
        "--table-clearance-weight",
        type=float,
        default=10000.0,
        help="positive generation-only support-plane penalty weight",
    )
    parser.add_argument("--freeze-thumb", action="store_true")
    parser.add_argument("--require-palm", action="store_true", help="require one authored palm keypoint in every selected contact set")
    parser.add_argument(
        "--prefer-distal-contacts",
        action="store_true",
        help=(
            "restrict non-palm source contacts to distal finger surfaces; "
            "use for palmar contact-realization generation"
        ),
    )
    parser.add_argument(
        "--require-contact-realization",
        action="store_true",
        help=(
            "emit only rows whose selected keypoints are within the configured "
            "object-surface distance; required for contact-conditioned static libraries"
        ),
    )
    parser.add_argument("--reachability-anchor-json", type=Path, help="JSON object with a finite 21D X2 root pose and optional non-negative translation_weight/rotation_weight")
    parser.add_argument(
        "--initial-raw-json",
        type=Path,
        help=(
            "reuse one generator raw pose, actuator vector, and selected contact "
            "tuple as a mutable initialization for constrained repair"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.num_grasps <= 0 or args.batch_size <= 0 or args.n_contact <= 0:
        parser.error("num-grasps, batch-size, and n-contact must be positive")
    if args.n_iterations <= 0:
        parser.error("n-iterations must be positive")
    if args.reachability_anchor_json is not None and args.initial_raw_json is not None:
        parser.error("reachability-anchor-json and initial-raw-json are mutually exclusive")
    if args.initial_raw_json is not None and args.side not in {"front", "back"}:
        parser.error("initial-raw-json requires an explicit front or back side")
    for name in (
        "initialization_distance_lower_m",
        "initialization_distance_upper_m",
    ):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0.0):
            parser.error(f"{name.replace('_', '-')} must be finite and non-negative")
    if (
        args.initialization_distance_lower_m is not None
        and args.initialization_distance_upper_m is not None
        and args.initialization_distance_lower_m >= args.initialization_distance_upper_m
    ):
        parser.error("initialization-distance-lower-m must be below initialization-distance-upper-m")
    if (
        args.initialization_minimum_surface_z_m is not None
        and not np.isfinite(args.initialization_minimum_surface_z_m)
    ):
        parser.error("initialization-minimum-surface-z-m must be finite")
    if args.table_plane_z_m is not None and not np.isfinite(args.table_plane_z_m):
        parser.error("table-plane-z-m must be finite")
    if not np.isfinite(np.asarray(args.table_plane_normal, dtype=np.float64)).all():
        parser.error("table-plane-normal must contain finite values")
    if np.linalg.norm(np.asarray(args.table_plane_normal, dtype=np.float64)) <= 0.0:
        parser.error("table-plane-normal must be non-zero")
    if not np.isfinite(args.table_clearance_m) or args.table_clearance_m < 0.0:
        parser.error("table-clearance-m must be finite and non-negative")
    if not np.isfinite(args.table_clearance_weight) or args.table_clearance_weight <= 0.0:
        parser.error("table-clearance-weight must be finite and positive")
    if args.finger_count is not None and args.finger_count > args.n_contact:
        parser.error("finger-count cannot exceed n-contact")
    if args.finger_names:
        if len(args.finger_names) != len(set(args.finger_names)):
            parser.error("finger-names must be unique")
        if args.finger_count is None:
            args.finger_count = len(args.finger_names)
        elif len(args.finger_names) != args.finger_count:
            parser.error("finger-names count must match finger-count")
    return args


def _with_cli_overrides(config: X2Config, args: argparse.Namespace) -> X2Config:
    data = copy.deepcopy(config.data)
    data["generation"]["n_contact"] = int(args.n_contact)
    if args.object_scale is not None:
        data["generation"]["object_scale"] = float(args.object_scale)
    if args.surface_samples is not None:
        data["generation"]["object_surface_samples"] = int(args.surface_samples)
    initialization_distance_lower_m = getattr(
        args, "initialization_distance_lower_m", None
    )
    initialization_distance_upper_m = getattr(
        args, "initialization_distance_upper_m", None
    )
    if initialization_distance_lower_m is not None:
        data["initialization"]["distance_lower"] = float(
            initialization_distance_lower_m
        )
    if initialization_distance_upper_m is not None:
        data["initialization"]["distance_upper"] = float(
            initialization_distance_upper_m
        )
    minimum_surface_z = getattr(
        args, "initialization_minimum_surface_z_m", None
    )
    if minimum_surface_z is not None:
        data["initialization"]["minimum_surface_z_m"] = float(
            minimum_surface_z
        )
    return X2Config(data=data, path=config.path, project_root=config.project_root)


def _requested_side_batches(
    mode: str, num_grasps: int, batch_size: int, rng: np.random.Generator
) -> list[tuple[str, ...]]:
    runs: list[tuple[str, ...]] = []
    if mode == "both":
        requested = (["front"] * num_grasps, ["back"] * num_grasps)
    elif mode in ("front", "back"):
        requested = ([mode] * num_grasps,)
    else:
        requested = ([str(v) for v in rng.choice(("front", "back"), size=num_grasps)],)
    for sides in requested:
        runs.extend(tuple(sides[start : start + batch_size]) for start in range(0, len(sides), batch_size))
    return runs


def _write_records(
    records: Sequence[dict[str, Any]],
    output_root: Path,
    counters: dict[str, int],
    *,
    overwrite: bool,
) -> list[Path]:
    written: list[Path] = []
    for record in records:
        side = str(record["active_side"])
        index = counters[side]
        counters[side] += 1
        directory = output_root / f"{side}_single" / "raw"
        directory.mkdir(parents=True, exist_ok=True)
        mesh_stem = Path(record["object"]["mesh_path"]).stem
        path = directory / f"{mesh_stem}_{side}_{index:06d}.json"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}; pass --overwrite")
        path.write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        written.append(path)
    return written


def _summarize(records: Sequence[dict[str, Any]], eligible_counts: dict[str, int]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["active_side"])].append(record)
    summary: dict[str, Any] = {
        "pipeline_revision": (
            str(records[0]["pipeline_revision"])
            if records
            else "x2_mesh_grasp_ownership_clean_v7"
        ),
        "simulation_run": False,
        "sides": {},
    }
    gate_records = [
        item["hand_object_penetration"]
        for item in records
        if isinstance(item.get("hand_object_penetration"), dict)
    ]
    if len(gate_records) != len(records):
        raise RuntimeError(
            "Every v5 generation record must contain dense hand-object diagnostics"
        )
    summary["dense_hand_object_gate"] = {
        "evaluation_mode": "dense_bidirectional",
        "hand_surface_samples_per_set": DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        "hand_surface_samples_per_link": 3
        * DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        "hand_surface_point_count": 17
        * 3
        * DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        "object_surface_samples": DENSE_OBJECT_SURFACE_SAMPLES,
        "threshold": HAND_OBJECT_PENETRATION_THRESHOLD,
        "strict_less_than": True,
        "sample_count": len(records),
        "evaluated_count": sum(
            bool(item.get("evaluated")) for item in gate_records
        ),
        "feasible_count": sum(
            bool(item.get("feasible")) for item in gate_records
        ),
    }
    for side in ("front", "back"):
        values = grouped.get(side, [])
        if not values:
            continue
        initial = np.asarray([item["energy"]["initial_total"] for item in values], dtype=np.float64)
        final = np.asarray([item["energy"]["total"] for item in values], dtype=np.float64)
        actuator = np.asarray([item["actuator"] for item in values], dtype=np.float64)
        regions = Counter(
            contact["region"] for item in values for contact in item["selected_contacts"]
        )
        self_collision = [item["self_collision"] for item in values]
        hand_object = [item["hand_object_penetration"] for item in values]
        summary["sides"][side] = {
            "initialized": len(values),
            "optimized": len(values),
            "eligible_contact_candidates": eligible_counts[side],
            "initial_energy_mean": float(initial.mean()),
            "final_energy_mean": float(final.mean()),
            "energy_change_mean": float((final - initial).mean()),
            "energy_decreased_count": int(np.sum(final < initial)),
            "selected_contact_regions": dict(sorted(regions.items())),
            "actuator_min": float(actuator.min()),
            "actuator_max": float(actuator.max()),
            "maximum_penetration": float(
                max(item["maximum_penetration"] for item in values)
            ),
            "self_collision_maximum_penetration": float(
                max(item["maximum_penetration"] for item in self_collision)
            ),
            "self_collision_total_penetration_mean": float(
                np.mean(
                    [item["total_penetration"] for item in self_collision],
                    dtype=np.float64,
                )
            ),
            "self_collision_feasible_count": int(
                sum(bool(item["feasible"]) for item in self_collision)
            ),
            "self_collision_threshold": float(self_collision[0]["threshold"]),
            "dense_hand_object_evaluated_count": int(
                sum(bool(item["evaluated"]) for item in hand_object)
            ),
            "dense_hand_object_feasible_count": int(
                sum(bool(item["feasible"]) for item in hand_object)
            ),
            "nan_or_inf": not all(item["finite"] for item in values),
            "accepted_contact_changes": int(
                sum(item["optimization"]["accepted_contact_changes"] for item in values)
                / len(values)
            ),
        }
    return summary


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    finger_count = getattr(args, "finger_count", None)
    config = _with_cli_overrides(load_x2_mesh_config(args.config), args)
    candidates = load_generic_contact_candidates(
        config.configured_path("contact_candidates.path", must_exist=True)
    )
    policies = {
        side: GenericDexterousContactPolicy(
            candidates,
            active_side=side,
            n_contact=args.n_contact,
            allow_thumb=bool(config.require("contact_candidates.allow_thumb")),
            target_finger_count=finger_count,
            required_finger_names=getattr(args, "finger_names", None),
            require_palm=bool(getattr(args, "require_palm", False)),
            prefer_distal_contacts=bool(
                getattr(args, "prefer_distal_contacts", False)
            ),
        )
        for side in ("front", "back")
    }
    device = torch.device(args.device)
    table_plane_z_m = getattr(args, "table_plane_z_m", None)
    table_clearance_m = float(getattr(args, "table_clearance_m", 0.002))
    table_clearance_weight = float(
        getattr(args, "table_clearance_weight", 10000.0)
    )
    table_plane_normal = tuple(
        float(value)
        for value in getattr(args, "table_plane_normal", (0.0, 0.0, 1.0))
    )
    table_plane = (
        None
        if table_plane_z_m is None
        else TabletopPlaneConstraint(
            normal=table_plane_normal,
            offset_m=float(table_plane_z_m),
            clearance_m=table_clearance_m,
            weight=table_clearance_weight,
        )
    )
    hand = X2HandModel(
        config,
        candidates,
        device=device,
        dtype=torch.float64,
        collision_samples_per_link=int(
            config.require("generation.hand_collision_samples_per_link")
        ),
        audit_collision_samples_per_link=int(
            config.require("generation.dense_hand_surface_samples_per_set")
        ),
        freeze_thumb=args.freeze_thumb,
    )
    reachability_anchor = None
    reachability_anchor_path = getattr(args, "reachability_anchor_json", None)
    initial_raw_path = getattr(args, "initial_raw_json", None)
    if reachability_anchor_path is not None:
        payload = json.loads(reachability_anchor_path.expanduser().resolve().read_text(encoding="utf-8"))
        pose = np.asarray(payload.get("pose"), dtype=np.float64)
        if pose.shape != (hand.POSE_DIMENSION,) or not np.isfinite(pose).all():
            raise ValueError("reachability anchor must provide one finite 21D pose")
        translation_weight = float(payload.get("translation_weight", 100.0))
        rotation_weight = float(payload.get("rotation_weight", 10.0))
        actuator = None
        if payload.get("actuator") is not None:
            values = np.asarray(payload["actuator"], dtype=np.float64)
            if values.shape != (12,) or not np.isfinite(values).all():
                raise ValueError("reachability anchor actuator must provide 12 finite values")
            actuator = torch.as_tensor(values, device=device, dtype=torch.float64).expand(args.batch_size, -1).clone()
        reachability_anchor = ReachabilityPoseAnchor(
            pose=torch.as_tensor(pose, device=device, dtype=torch.float64).expand(args.batch_size, -1).clone(),
            translation_weight=translation_weight,
            rotation_weight=rotation_weight,
            actuator=actuator,
        )
    elif initial_raw_path is not None:
        raw = json.loads(initial_raw_path.expanduser().resolve().read_text(encoding="utf-8"))
        if str(raw.get("active_side", "")).lower() != str(args.side).lower():
            raise ValueError("initial raw active_side must match the requested side")
        hand_pose = raw.get("hand_pose") or {}
        translation = np.asarray(hand_pose.get("translation"), dtype=np.float64)
        rotation = np.asarray(hand_pose.get("rotation_matrix"), dtype=np.float64)
        if translation.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("initial raw must provide a 3D translation and 3x3 rotation")
        if not np.isfinite(translation).all() or not np.isfinite(rotation).all():
            raise ValueError("initial raw hand pose must be finite")
        rotation6d = rotation[:, :2].T.reshape(6)
        raw_names = tuple(str(value) for value in raw.get("actuator_names", ()))
        raw_actuator = tuple(float(value) for value in raw.get("actuator", ()))
        if len(raw_names) != len(raw_actuator) or set(raw_names) != set(hand.actuator_names):
            raise ValueError("initial raw actuator names must match the configured hand")
        actuator_by_name = dict(zip(raw_names, raw_actuator))
        actuator_values = np.asarray(
            [actuator_by_name[name] for name in hand.actuator_names], dtype=np.float64
        )
        candidate_by_id = {
            candidate.point_id: index for index, candidate in enumerate(candidates)
        }
        selected_ids = tuple(str(value) for value in raw.get("selected_contact_ids", ()))
        if len(selected_ids) != int(args.n_contact) or any(
            point_id not in candidate_by_id for point_id in selected_ids
        ):
            raise ValueError("initial raw selected contacts must match n-contact and current candidates")
        pose_values = np.concatenate((translation, rotation6d, actuator_values))
        reachability_anchor_path = initial_raw_path
        reachability_anchor = ReachabilityPoseAnchor(
            pose=torch.as_tensor(
                pose_values, device=device, dtype=torch.float64
            ).expand(args.batch_size, -1).clone(),
            translation_weight=0.0,
            rotation_weight=0.0,
            actuator=torch.as_tensor(
                actuator_values, device=device, dtype=torch.float64
            ).expand(args.batch_size, -1).clone(),
            contact_indices=torch.as_tensor(
                [candidate_by_id[point_id] for point_id in selected_ids],
                device=device,
                dtype=torch.long,
            ).expand(args.batch_size, -1).clone(),
        )

    records: list[dict[str, Any]] = []
    table_conditioning_rejected_samples = 0
    table_conditioning_below_requested_margin_samples = 0
    contact_realization_rejected_samples = 0
    table_rejection_diagnostics: list[dict[str, Any]] = []
    table_rejection_records: list[dict[str, Any]] = []
    side_batches = _requested_side_batches(args.side, args.num_grasps, args.batch_size, rng)
    for batch_index, active_sides in enumerate(side_batches):
        object_model = MeshObjectModel(
            args.mesh_path,
            batch_size=len(active_sides),
            scale=float(config.require("generation.object_scale")),
            num_surface_samples=int(config.require("generation.object_surface_samples")),
            audit_surface_samples=int(
                config.require("generation.dense_object_surface_samples")
            ),
            device=device,
            dtype=torch.float64,
            seed=args.seed + batch_index,
        )
        result = optimize_x2_mesh_batch(
            hand,
            object_model,
            active_sides,
            policies,
            config,
            n_iterations=args.n_iterations,
            seed=args.seed + batch_index * 100003,
            rng=rng,
            table_plane=table_plane,
            reachability_anchor=(
                None if reachability_anchor is None else ReachabilityPoseAnchor(
                    pose=reachability_anchor.pose[: len(active_sides)],
                    translation_weight=reachability_anchor.translation_weight,
                    rotation_weight=reachability_anchor.rotation_weight,
                    actuator=(None if reachability_anchor.actuator is None else reachability_anchor.actuator[: len(active_sides)]),
                    contact_indices=(None if reachability_anchor.contact_indices is None else reachability_anchor.contact_indices[: len(active_sides)]),
                )
            ),
        )
        batch_records = make_sample_records(
            hand, object_model, result, candidates, seed=args.seed
        )
        table_clearances = (
            None
            if table_plane is None
            else tabletop_plane_minimum_clearance(
                hand, table_plane, dense_audit=True
            )
            .detach()
            .cpu()
            .tolist()
        )
        table_collision_mesh_clearances = (
            None
            if table_plane is None
            else tabletop_plane_collision_mesh_minimum_clearance(hand, table_plane)
            .detach()
            .cpu()
            .tolist()
        )
        for local_index, record in enumerate(batch_records):
            record["sample_index"] = len(records) + local_index
            if reachability_anchor is not None:
                # Preserve the provenance of the soft generation condition on
                # every raw row.  It deliberately does not upgrade the row to
                # an IK, motion, or tabletop-admitted candidate; those remain
                # downstream hard checks in sequential_arch_v1.
                record[
                    "initialization_seed"
                    if initial_raw_path is not None
                    else "reachability_anchor"
                ] = {
                    "status": (
                        "MUTABLE_RAW_POSE_ACTUATOR_CONTACT_INITIALIZATION_ONLY"
                        if initial_raw_path is not None
                        else "SOFT_GENERATION_PRIOR_ONLY"
                    ),
                    "source": str(reachability_anchor_path.expanduser().resolve()),
                    "translation_weight": float(
                        reachability_anchor.translation_weight
                    ),
                    "rotation_weight": float(reachability_anchor.rotation_weight),
                    "actuator_seed": reachability_anchor.actuator is not None,
                    "contact_seed": reachability_anchor.contact_indices is not None,
                    "not_ik_or_motion_proof": True,
                }
            participating = sorted(
                {
                    contact["finger_name"]
                    for contact in record["selected_contacts"]
                    if contact["finger_name"] in FINGER_NAMES
                },
                key=FINGER_NAMES.index,
            )
            record["finger_participation"] = {
                "target_count": finger_count or len(participating),
                "actual_count": len(participating),
                "finger_names": participating,
            }
            if finger_count is not None and len(participating) != finger_count:
                raise RuntimeError(
                    "Optimizer returned a contact selection outside the requested "
                    f"finger stratum: expected={finger_count}, actual={participating}"
                )
            contact_realization = record.get("selected_contact_realization")
            if (
                bool(getattr(args, "require_contact_realization", False))
                and (
                    not isinstance(contact_realization, dict)
                    or contact_realization.get("status") != "PASS"
                )
            ):
                contact_realization_rejected_samples += 1
                continue
            if table_plane is not None:
                collision_mesh_clearance = float(
                    table_collision_mesh_clearances[local_index]
                )
                requested_clearance_met = (
                    collision_mesh_clearance >= float(table_plane.clearance_m)
                )
                record["table_conditioning"] = {
                    "status": "GENERATION_SAMPLED_SUPPORT_PLANE_ONLY",
                    "frame": "MESH_OBJECT_FRAME",
                    "plane_normal": list(table_plane.normal),
                    "plane_offset_m": float(table_plane.offset_m),
                    "requested_clearance_m": float(table_plane.clearance_m),
                    "penalty_weight": float(table_plane.weight),
                    "dense_minimum_hand_plane_clearance_m": float(
                        table_clearances[local_index]
                    ),
                    "collision_mesh_vertex_minimum_hand_plane_clearance_m": float(
                        collision_mesh_clearance
                    ),
                    "source_plane_nonpenetrating": _source_table_nonpenetrating(
                        collision_mesh_clearance
                    ),
                    "requested_clearance_met": requested_clearance_met,
                    "not_exact_table_admission": True,
                }
                # A table-conditioned generator must not emit a row whose
                # calibrated collision mesh already crosses the support
                # plane.  This is a conservative generation admission only;
                # exact scene/table admission still runs in the sequential
                # pipeline before the row can become a static candidate.
                if not _source_table_nonpenetrating(collision_mesh_clearance):
                    table_conditioning_rejected_samples += 1
                    contact = record.get("selected_contact_realization") or {}
                    hand_object = record.get("hand_object_penetration") or {}
                    clearance = collision_mesh_clearance
                    table_rejection_diagnostics.append(
                        {
                            "batch_index": int(batch_index),
                            "local_index": int(local_index),
                            "active_side": record.get("active_side"),
                            "contact_status": contact.get("status"),
                            "maximum_contact_distance_m": contact.get(
                                "maximum_distance_m"
                            ),
                            "collision_mesh_minimum_clearance_m": clearance,
                            "clearance_deficit_m": float(
                                table_plane.clearance_m - clearance
                            ),
                            "dense_hand_object_maximum_penetration_m": (
                                hand_object.get("maximum_penetration")
                            ),
                            "dense_hand_object_feasible": hand_object.get(
                                "feasible"
                            ),
                        }
                    )
                    # Preserve the full optimized proposal for a bounded,
                    # downstream current-mount repair.  It remains explicitly
                    # rejected here and is written outside the admitted raw
                    # pool, so neither generation counts nor later tools can
                    # confuse it with a table-conditioned positive.
                    record["development_rejection"] = {
                        "status": "GENERATION_TABLE_REJECTED_DIAGNOSTIC_ONLY",
                        "reason": "SOURCE_SUPPORT_PLANE_PENETRATION",
                        "collision_mesh_minimum_clearance_m": clearance,
                        "required_source_nonpenetration_m": 0.0,
                        "planning_allowed": False,
                        "scientific_evidence_eligible": False,
                        "allowed_next_use": (
                            "BOUNDED_EXACT_TABLE_AND_CURRENT_CONTACT_REPAIR_ONLY"
                        ),
                    }
                    table_rejection_records.append(record)
                    continue
                if not requested_clearance_met:
                    table_conditioning_below_requested_margin_samples += 1
            records.append(record)

    counters = {"front": 0, "back": 0}
    paths = _write_records(
        records, args.output.expanduser().resolve(), counters, overwrite=args.overwrite
    )
    diagnostic_paths = _write_records(
        table_rejection_records,
        args.output.expanduser().resolve() / "diagnostic_table_rejected",
        {"front": 0, "back": 0},
        overwrite=args.overwrite,
    )
    summary = _summarize(
        records,
        {side: len(policy.eligible_indices) for side, policy in policies.items()},
    )
    summary.update(
        {
            "mesh_path": str(args.mesh_path.expanduser().resolve()),
            "side_mode": args.side,
            "num_output_samples": len(records),
            "finger_count": finger_count,
            "output_files": [str(path) for path in paths],
            "diagnostic_table_rejected_files": [
                str(path) for path in diagnostic_paths
            ],
            "dual_object_samples": 0,
            "valid_files_written": 0,
            "table_conditioning_rejected_sample_count": (
                int(table_conditioning_rejected_samples)
            ),
            "table_conditioning_below_requested_margin_sample_count": int(
                table_conditioning_below_requested_margin_samples
            ),
            "closest_table_rejections": sorted(
                table_rejection_diagnostics,
                key=lambda row: float(row["clearance_deficit_m"]),
            )[:8],
            "selected_contact_realization_rejected_sample_count": int(
                contact_realization_rejected_samples
            ),
            "selected_contact_realization": {
                "required": bool(
                    getattr(args, "require_contact_realization", False)
                ),
                "threshold_m": float(
                    config.require("generation.selected_contact_max_distance_m")
                ),
                "generation_admission_only": True,
                "not_collision_or_physical_proof": True,
            },
            "table_conditioning": (
                None
                if table_plane is None
                else {
                    "status": "GENERATION_SAMPLED_SUPPORT_PLANE_ONLY",
                    "frame": "MESH_OBJECT_FRAME",
                    "plane_normal": list(table_plane.normal),
                    "plane_offset_m": float(table_plane.offset_m),
                    "requested_clearance_m": float(table_plane.clearance_m),
                    "penalty_weight": float(table_plane.weight),
                    "not_exact_table_admission": True,
                }
            ),
        }
    )
    return records, summary


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _, summary = run(args)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
