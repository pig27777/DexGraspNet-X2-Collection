#!/usr/bin/env python3
"""Print a read-only capsule/hull self-collision audit for one X2 raw grasp.

The command never edits the input JSON.  The analytic result uses the same
explicit low-vertex collision hulls as the generator and PhysX asset; PhysX
articulation self-collision remains disabled and is not used as the oracle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HULL_MANIFEST = (
    PROJECT_ROOT / "x2_mujoco" / "payloads" / "x2_physx_collision_hulls.json"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_generation.utils.x2_config import load_x2_mesh_config  # noqa: E402
from grasp_generation.utils.x2_hand_model import X2HandModel  # noqa: E402
from grasp_generation.utils.x2_mesh_contacts import (  # noqa: E402
    load_generic_contact_candidates,
)
from grasp_generation.x2_isaac_validation import (  # noqa: E402
    X2ValidationError,
    load_raw_candidate,
)


def _rotation6d(rotation: np.ndarray) -> np.ndarray:
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise X2ValidationError("hand_pose.rotation_matrix must be finite 3x3")
    return rotation.T[:2].reshape(6)


def _local_bounds(points: np.ndarray) -> dict[str, list[float]]:
    return {
        "minimum": [float(value) for value in points.min(axis=0)],
        "maximum": [float(value) for value in points.max(axis=0)],
    }


def _load_link_geometry_audit(
    hand: X2HandModel,
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    """Read visual bounds and authored hull shrink provenance from the X2 asset."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise X2ValidationError(f"could not read collision-hull manifest {manifest_path}: {exc}") from exc
    links = manifest.get("links")
    if not isinstance(links, dict):
        raise X2ValidationError(f"{manifest_path}: links must be a JSON object")

    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:
        raise X2ValidationError("OpenUSD is required to audit visual mesh bounds") from exc
    stage = Usd.Stage.Open(str(hand.config.configured_path("robot.usd_path", must_exist=True)))
    if stage is None:
        raise X2ValidationError("OpenUSD could not open the configured X2 asset")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    report: dict[str, dict[str, Any]] = {}
    for link_name in hand.backend.link_names:
        collision = np.asarray(
            hand.backend.collision_meshes[link_name].vertices_local,
            dtype=np.float64,
        )
        provenance = links.get(link_name)
        if not isinstance(provenance, dict):
            raise X2ValidationError(f"{manifest_path}: missing link audit for {link_name}")
        source_path = provenance.get("source_prim_path")
        owner_path = provenance.get("owner_path")
        source = stage.GetPrimAtPath(str(source_path))
        owner = stage.GetPrimAtPath(str(owner_path))
        if not source.IsValid() or not owner.IsValid() or not source.IsA(UsdGeom.Mesh):
            raise X2ValidationError(
                f"configured X2 asset is missing visual source for {link_name}: {source_path}"
            )
        points = np.asarray(UsdGeom.Mesh(source).GetPointsAttr().Get(), dtype=np.float64)
        mesh_to_world = cache.GetLocalToWorldTransform(source)
        world_to_owner = cache.GetLocalToWorldTransform(owner).GetInverse()
        mesh_to_owner = np.asarray(mesh_to_world * world_to_owner, dtype=np.float64).T
        visual = points @ mesh_to_owner[:3, :3].T + mesh_to_owner[:3, 3]
        shrink = float(provenance.get("maximum_source_support_plane_violation"))
        if visual.ndim != 2 or visual.shape[1] != 3 or not np.isfinite(visual).all():
            raise X2ValidationError(f"visual mesh for {link_name} produced invalid local vertices")
        if not np.isfinite(shrink) or shrink < 0.0:
            raise X2ValidationError(f"invalid hull shrink audit for {link_name}")
        report[link_name] = {
            "visual_bounds_local_m": _local_bounds(visual),
            "collision_bounds_local_m": _local_bounds(collision),
            "maximum_support_plane_shrink_m": shrink,
            "visual_vertex_count": int(len(visual)),
            "collision_vertex_count": int(len(collision)),
        }
    return report


def build_report(
    raw_path: Path,
    *,
    config_path: Path | None = None,
    device: str = "cpu",
    hull_manifest: Path = DEFAULT_HULL_MANIFEST,
) -> dict[str, Any]:
    """Materialize one raw grasp and return its JSON-safe static collision audit."""

    candidate = load_raw_candidate(raw_path)
    config = load_x2_mesh_config(config_path)
    candidates = load_generic_contact_candidates(
        config.configured_path("contact_candidates.path", must_exist=True)
    )
    hand = X2HandModel(
        config,
        candidates,
        device=torch.device(device),
        dtype=torch.float64,
        collision_samples_per_link=int(
            config.require("generation.hand_collision_samples_per_link")
        ),
    )
    candidate_by_id = {value.point_id: index for index, value in enumerate(candidates)}
    try:
        contact_indices = [
            candidate_by_id[str(point_id)]
            for point_id in candidate.record["selected_contact_ids"]
        ]
    except KeyError as exc:
        raise X2ValidationError(f"raw grasp references unknown contact ID: {exc.args[0]}") from exc

    rotation = np.asarray(candidate.record["hand_pose"]["rotation_matrix"], dtype=np.float64)
    actuator = np.asarray(candidate.record["actuator"], dtype=np.float64)
    pose_np = np.concatenate((candidate.hand_translation, _rotation6d(rotation), actuator))
    pose = torch.tensor(pose_np, device=hand.device, dtype=hand.dtype).unsqueeze(0)
    contacts = torch.tensor(
        contact_indices, device=hand.device, dtype=torch.long
    ).unsqueeze(0)
    hand.set_parameters(pose, contacts)

    diagnostics = hand.self_collision_diagnostics()
    pair_names = tuple(tuple(names) for names in diagnostics.pair_names)
    pair_sum = diagnostics.pair_sum[0].detach().cpu().numpy()
    pair_max = diagnostics.pair_max[0].detach().cpu().numpy()
    pair_energy = diagnostics.pair_energy[0].detach().cpu().numpy()
    capsule_depth = (
        hand.backend.self_collision_depths(hand.joint_positions)[0]
        .detach()
        .cpu()
        .numpy()
    )
    capsule_by_pair = {
        frozenset(names): float(value)
        for names, value in zip(hand.backend.self_collision_proxy_pairs, capsule_depth)
    }
    capsule_weight = float(config.require("self_collision.capsule_weight"))
    hull_weight = float(config.require("self_collision.hull_weight"))
    capsule_total = float(np.maximum(capsule_depth, 0.0).sum())
    hull_total_energy = float(pair_energy.sum())
    pair_records = []
    for names, hull_sum, hull_max, contribution in zip(
        pair_names, pair_sum, pair_max, pair_energy
    ):
        capsule_signed_depth = capsule_by_pair.get(frozenset(names))
        capsule_contribution = (
            capsule_weight * max(capsule_signed_depth, 0.0)
            if capsule_signed_depth is not None
            else None
        )
        pair_records.append(
            {
                "links": list(names),
                "capsule_signed_depth_m": capsule_signed_depth,
                "capsule_signed_separation_m": (
                    -capsule_signed_depth if capsule_signed_depth is not None else None
                ),
                "capsule_energy_contribution": capsule_contribution,
                "hull_total_penetration_m": float(hull_sum),
                "hull_maximum_penetration_m": float(hull_max),
                "hull_pair_energy": float(contribution),
                "hull_weighted_energy_contribution": float(
                    hull_weight * contribution
                ),
            }
        )

    maximum = float(diagnostics.maximum[0].detach().cpu())
    total = float(diagnostics.total[0].detach().cpu())
    threshold = float(diagnostics.threshold)
    worst_pair = diagnostics.worst_pair_names(0)
    report = {
        "schema_version": 1,
        "diagnostic": "x2_sampled_convex_hull_self_collision",
        "source_raw": str(candidate.path),
        "source_sha256": candidate.source_sha256,
        "pipeline_revision": candidate.record.get("pipeline_revision"),
        "oracle": "deterministic_bidirectional_sampled_collision_hull",
        "physx_self_collision_enabled": False,
        "sampling": {
            "surface_samples_per_link_per_set": int(
                diagnostics.surface_samples_per_link
            ),
            "sets": ["vertices", "face_centroids", "face_interiors"],
        },
        "summary": {
            "maximum_penetration_m": maximum,
            "total_penetration_m": total,
            "worst_pair": list(worst_pair) if worst_pair is not None else None,
            "threshold_m": threshold,
            "feasible": maximum <= threshold,
            "capsule_total_penetration_m": capsule_total,
            "capsule_weighted_energy": capsule_weight * capsule_total,
            "hull_pair_energy_total": hull_total_energy,
            "hull_weighted_energy": hull_weight * hull_total_energy,
        },
        "pairs": pair_records,
        "links": _load_link_geometry_audit(
            hand, hull_manifest.expanduser().resolve()
        ),
    }
    # Fail here rather than allowing NaN/Inf to leak through stdout.
    json.dumps(report, allow_nan=False)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_json", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hull-manifest", type=Path, default=DEFAULT_HULL_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = build_report(
        args.raw_json,
        config_path=args.config,
        device=args.device,
        hull_manifest=args.hull_manifest,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
