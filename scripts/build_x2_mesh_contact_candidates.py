#!/usr/bin/env python3
"""Build side-conditioned X2 mesh contact candidates directly from authored USD markers."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_generation.utils.actuator_hand_model import ActuatorHandModel
from grasp_generation.utils.x2_config import load_x2_mesh_config
from grasp_generation.utils.x2_mesh_contacts import GenericContactCandidate


@dataclass(frozen=True)
class _AuthoredPoint:
    point_id: str
    link_name: str
    finger_name: str
    local_position: tuple[float, float, float]
    local_surface_normal: tuple[float, float, float]
    source_path: str


def _usd_column_matrix(cache: UsdGeom.XformCache, prim: Usd.Prim) -> np.ndarray:
    """Convert OpenUSD's row-vector local-to-world matrix to column convention."""

    return np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64).T.copy()


def _closest_rigid_ancestor(prim: Usd.Prim) -> Usd.Prim | None:
    current = prim
    while current and current.IsValid():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        current = current.GetParent()
    return None


def _marker_position_local(
    cache: UsdGeom.XformCache, marker: Usd.Prim, owner: Usd.Prim
) -> np.ndarray:
    marker_world = _usd_column_matrix(cache, marker)
    owner_world = _usd_column_matrix(cache, owner)
    point = (
        np.linalg.inv(owner_world)
        @ marker_world
        @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    )[:3]
    if point.shape != (3,) or not np.isfinite(point).all():
        raise RuntimeError(f"Non-finite marker transform at {marker.GetPath()}")
    return point


def _collision_trimeshes(hand: ActuatorHandModel) -> dict[str, trimesh.Trimesh]:
    return {
        link_name: trimesh.Trimesh(
            vertices=np.array(collision.vertices_local, copy=True),
            faces=np.array(collision.triangles, copy=True),
            process=False,
        )
        for link_name, collision in hand.collision_meshes.items()
    }


def _nearest_plane_normal(
    hand: ActuatorHandModel, link_name: str, point: np.ndarray
) -> np.ndarray:
    collision = hand.collision_meshes[link_name]
    values = collision.plane_normals_local @ point + collision.plane_offsets_local
    normal = np.asarray(
        collision.plane_normals_local[int(np.argmax(values))], dtype=np.float64
    )
    return normal / np.linalg.norm(normal)


def _project_nearest(
    meshes: dict[str, trimesh.Trimesh],
    hand: ActuatorHandModel,
    link_name: str,
    source: np.ndarray,
    *,
    tolerance: float,
    source_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    closest, distance, _ = trimesh.proximity.closest_point_naive(
        meshes[link_name], source[None, :]
    )
    projection_distance = float(distance[0])
    if not np.isfinite(projection_distance) or projection_distance > tolerance:
        raise RuntimeError(
            f"Keypoint {source_path} is {projection_distance:.6g} m from the "
            f"{link_name} collision hull; tolerance={tolerance:.6g} m"
        )
    projected = np.asarray(closest[0], dtype=np.float64)
    return projected, _nearest_plane_normal(hand, link_name, projected)


def _project_palm_face(
    hand: ActuatorHandModel,
    link_name: str,
    source: np.ndarray,
    direction: np.ndarray,
    *,
    tolerance: float,
    normal_min_dot: float,
    source_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect an authored palm marker with the calibrated outward hull face."""

    collision = hand.collision_meshes[link_name]
    normals = np.asarray(collision.plane_normals_local, dtype=np.float64)
    offsets = np.asarray(collision.plane_offsets_local, dtype=np.float64)
    signed = normals @ source + offsets
    if float(np.max(signed)) > 1.0e-7:
        raise RuntimeError(
            f"Palm keypoint {source_path} lies outside the {link_name} collision hull"
        )
    denominators = normals @ direction
    valid = denominators > 1.0e-10
    parameters = np.full(len(normals), np.inf, dtype=np.float64)
    parameters[valid] = -signed[valid] / denominators[valid]
    parameters[parameters < -1.0e-9] = np.inf
    plane_index = int(np.argmin(parameters))
    distance = float(parameters[plane_index])
    if not math.isfinite(distance) or distance > tolerance:
        raise RuntimeError(
            f"Palm keypoint {source_path} is {distance:.6g} m from its calibrated "
            f"load face; tolerance={tolerance:.6g} m"
        )
    normal = normals[plane_index]
    normal = normal / np.linalg.norm(normal)
    dot = float(normal @ direction)
    if dot < normal_min_dot:
        raise RuntimeError(
            f"Palm keypoint {source_path} selected a face with normal dot={dot:.6g}; "
            f"minimum={normal_min_dot:.6g}"
        )
    return source + distance * direction, normal


def _validate_marker(marker: Usd.Prim, expected_radius: float) -> None:
    if marker.HasAPI(UsdPhysics.CollisionAPI) or marker.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Contact marker unexpectedly has physics API: {marker.GetPath()}")
    radius = UsdGeom.Sphere(marker).GetRadiusAttr().Get()
    if radius is None or abs(float(radius) - expected_radius) > 1.0e-9:
        raise RuntimeError(
            f"Contact marker radius changed at {marker.GetPath()}: "
            f"{radius}; expected {expected_radius}"
        )


def _audit_finger_joint_clearance(
    stage: Usd.Stage, points: Iterable[_AuthoredPoint], minimum: float
) -> None:
    anchors: dict[str, list[np.ndarray]] = defaultdict(list)
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        for body_targets, position in (
            (list(joint.GetBody0Rel().GetTargets()), joint.GetLocalPos0Attr().Get()),
            (list(joint.GetBody1Rel().GetTargets()), joint.GetLocalPos1Attr().Get()),
        ):
            if len(body_targets) == 1 and position is not None:
                anchors[str(body_targets[0].name)].append(
                    np.asarray(position, dtype=np.float64)
                )
    for point in points:
        distances = [
            float(
                np.linalg.norm(
                    np.asarray(point.local_position, dtype=np.float64) - anchor
                )
            )
            for anchor in anchors.get(point.link_name, ())
        ]
        if not distances or min(distances) < minimum:
            raise RuntimeError(
                f"Finger keypoint {point.point_id} violates the {minimum:.6g} m "
                "joint-clearance requirement"
            )


def _authored_non_thumb_points(
    hand: ActuatorHandModel,
    config: Any,
    stage: Usd.Stage,
    cache: UsdGeom.XformCache,
    meshes: dict[str, trimesh.Trimesh],
) -> tuple[list[_AuthoredPoint], list[_AuthoredPoint], list[_AuthoredPoint]]:
    finger_links = config.require("robot.finger_links")
    link_to_finger = {
        str(link_name): str(finger_name)
        for finger_name, link_names in finger_links.items()
        for link_name in link_names
    }
    palm_name = str(config.require("robot.palm_link_name"))
    marker_radius = float(config.require("contact_candidates.marker_radius"))
    mirror_attribute = str(config.require("contact_candidates.palm_mirror_attribute"))
    finger_sources: list[tuple[str, str, np.ndarray, str]] = []
    palm_original: dict[str, tuple[np.ndarray, str]] = {}
    palm_mirrored: dict[str, tuple[np.ndarray, str]] = {}

    traversal = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
    for prim in traversal:
        if not prim.IsActive() or not prim.IsA(UsdGeom.Sphere):
            continue
        owner = _closest_rigid_ancestor(prim)
        if owner is None:
            continue
        owner_name = str(owner.GetName())
        if owner_name not in link_to_finger and owner_name != palm_name:
            continue
        _validate_marker(prim, marker_radius)
        point = _marker_position_local(cache, prim, owner)
        path = str(prim.GetPath())
        if owner_name in link_to_finger:
            finger_sources.append((link_to_finger[owner_name], owner_name, point, path))
            continue
        mirror_source = prim.GetAttribute(mirror_attribute).Get()
        if mirror_source:
            key = str(mirror_source)
            target = palm_mirrored
        else:
            key = str(prim.GetName())
            target = palm_original
        if key in target:
            raise RuntimeError(f"Duplicate palm marker identity {key!r}")
        target[key] = (point, path)

    expected_finger_count = int(
        config.require("contact_candidates.expected_finger_markers_per_finger")
    )
    finger_projection_tolerance = float(
        config.require("contact_candidates.finger_keypoint_projection_tolerance")
    )
    finger_sources.sort(key=lambda item: (item[0], item[1], item[3]))
    counter: Counter[str] = Counter()
    fingers: list[_AuthoredPoint] = []
    for finger_name, link_name, source, path in finger_sources:
        index = counter[finger_name]
        counter[finger_name] += 1
        projected, normal = _project_nearest(
            meshes,
            hand,
            link_name,
            source,
            tolerance=finger_projection_tolerance,
            source_path=path,
        )
        fingers.append(
            _AuthoredPoint(
                point_id=f"finger:{finger_name}:{index:03d}",
                link_name=link_name,
                finger_name=finger_name,
                local_position=tuple(float(value) for value in projected),
                local_surface_normal=tuple(float(value) for value in normal),
                source_path=path,
            )
        )
    if set(counter) != set(finger_links) or any(
        counter[name] != expected_finger_count for name in finger_links
    ):
        raise RuntimeError(
            f"Expected {expected_finger_count} markers for every non-thumb finger; "
            f"found {dict(counter)}"
        )
    _audit_finger_joint_clearance(
        stage,
        fingers,
        float(config.require("contact_candidates.finger_joint_clearance")),
    )

    expected_palm_count = int(
        config.require("contact_candidates.expected_palm_markers_per_side")
    )
    if (
        len(palm_original) != expected_palm_count
        or len(palm_mirrored) != expected_palm_count
        or set(palm_original) != set(palm_mirrored)
    ):
        raise RuntimeError(
            f"Expected matching {expected_palm_count}-point front/back palm groups; "
            f"found {len(palm_original)} and {len(palm_mirrored)}"
        )
    mirror_tolerance = float(config.require("contact_candidates.palm_mirror_tolerance"))
    palm_projection_tolerance = float(
        config.require("contact_candidates.palm_keypoint_projection_tolerance")
    )
    normal_min_dot = float(config.require("contact_candidates.palm_normal_min_dot"))
    palms: dict[str, list[_AuthoredPoint]] = {"front": [], "back": []}
    for index, source_name in enumerate(sorted(palm_original)):
        front_source, front_path = palm_original[source_name]
        back_source, back_path = palm_mirrored[source_name]
        expected_back = np.asarray(
            (front_source[0], -front_source[1], front_source[2]), dtype=np.float64
        )
        if float(np.linalg.norm(back_source - expected_back)) > mirror_tolerance:
            raise RuntimeError(f"Palm marker pair {source_name!r} is not an authored mirror")
        for side, source, path in (
            ("front", front_source, front_path),
            ("back", back_source, back_path),
        ):
            direction = np.asarray(
                config.require(f"palm_sides.{side}.normal_local"), dtype=np.float64
            )
            projected, normal = _project_palm_face(
                hand,
                palm_name,
                source,
                direction,
                tolerance=palm_projection_tolerance,
                normal_min_dot=normal_min_dot,
                source_path=path,
            )
            palms[side].append(
                _AuthoredPoint(
                    point_id=f"palm:{side}:{index:03d}",
                    link_name=palm_name,
                    finger_name="palm",
                    local_position=tuple(float(value) for value in projected),
                    local_surface_normal=tuple(float(value) for value in normal),
                    source_path=path,
                )
            )
    center_tolerance = float(config.require("contact_candidates.palm_center_tolerance"))
    for side in ("front", "back"):
        actual = np.mean(
            np.asarray([point.local_position for point in palms[side]], dtype=np.float64),
            axis=0,
        )
        configured = np.asarray(
            config.require(f"palm_sides.{side}.center_local"), dtype=np.float64
        )
        if float(np.linalg.norm(actual - configured)) > center_tolerance:
            raise RuntimeError(f"Configured {side} palm center does not match authored markers")
    return palms["front"], palms["back"], fingers


def _authored_thumb_candidates(
    hand: ActuatorHandModel,
    config: Any,
    stage: Usd.Stage,
    cache: UsdGeom.XformCache,
    meshes: dict[str, trimesh.Trimesh],
) -> list[GenericContactCandidate]:
    thumb_links = set(hand.thumb_links)
    marker_radius = float(config.require("contact_candidates.marker_radius"))
    projection_tolerance = float(
        config.require("contact_candidates.thumb_keypoint_projection_tolerance")
    )
    authored: list[tuple[str, Usd.Prim, Usd.Prim]] = []
    traversal = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
    for prim in traversal:
        if not prim.IsActive() or not prim.IsA(UsdGeom.Sphere):
            continue
        owner = _closest_rigid_ancestor(prim)
        if owner is not None and str(owner.GetName()) in thumb_links:
            authored.append((str(owner.GetName()), owner, prim))
    candidates: list[GenericContactCandidate] = []
    for link_name, owner, prim in sorted(authored, key=lambda item: str(item[2].GetPath())):
        _validate_marker(prim, marker_radius)
        source = _marker_position_local(cache, prim, owner)
        projected, normal = _project_nearest(
            meshes,
            hand,
            link_name,
            source,
            tolerance=projection_tolerance,
            source_path=str(prim.GetPath()),
        )
        candidates.append(
            GenericContactCandidate(
                point_id=f"thumb:{link_name}:{prim.GetName()}",
                link_name=link_name,
                finger_name="thumb",
                region="thumb",
                local_position=tuple(float(value) for value in projected),
                local_surface_normal=tuple(float(value) for value in normal),
                supported_sides=("front", "back"),
                source=f"authored_keypoint:{prim.GetPath()}",
            )
        )
    if len(candidates) != int(config.require("contact_candidates.expected_thumb_markers")):
        raise RuntimeError(f"Expected 17 authored thumb keypoints, found {len(candidates)}")
    return candidates


def _support_and_region(
    normal_in_palm: np.ndarray,
    front_normal: np.ndarray,
    threshold: float,
) -> tuple[tuple[str, ...], str]:
    dot = float(normal_in_palm @ front_normal)
    if dot >= threshold:
        return ("front",), "front_finger_surface"
    if dot <= -threshold:
        return ("back",), "back_finger_surface"
    return ("front", "back"), "shared_fingertip"


def _check_output(candidates: list[GenericContactCandidate], tolerance: float) -> None:
    ids = [candidate.point_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Generated contact point IDs are not unique")
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            if left.link_name != right.link_name:
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(left.local_position) - np.asarray(right.local_position)
                )
            )
            if distance <= tolerance:
                raise RuntimeError(
                    f"Duplicate contact candidates {left.point_id} and {right.point_id}"
                )


def build(config_path: str | Path | None = None) -> tuple[Path, Counter[str]]:
    config = load_x2_mesh_config(config_path, require_contact_candidates=False)
    hand = ActuatorHandModel(
        config,
        device="cpu",
        dtype=torch.float64,
        collision_samples_per_link=4,
    )
    stage = Usd.Stage.Open(str(hand.usd_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"OpenUSD could not open {hand.usd_path}")
    if abs(float(UsdGeom.GetStageMetersPerUnit(stage)) - 1.0) > 1.0e-12:
        raise RuntimeError("X2 keypoint USD must use meters")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    meshes = _collision_trimeshes(hand)
    front_palm, back_palm, fingers = _authored_non_thumb_points(
        hand, config, stage, cache, meshes
    )

    front_normal = np.asarray(
        config.require("palm_sides.front.normal_local"), dtype=np.float64
    )
    threshold = float(config.require("contact_candidates.normal_side_threshold"))
    zero = torch.zeros(16, dtype=torch.float64)
    rest_fk = {
        name: value.detach().cpu().numpy()
        for name, value in hand.forward_kinematics(zero).items()
    }
    shared_tip_ids: set[str] = set()
    for link_name in sorted({point.link_name for point in fingers if "distal" in point.link_name}):
        link_points = [point for point in fingers if point.link_name == link_name]
        cutoff = float(np.quantile([point.local_position[0] for point in link_points], 0.75))
        shared_tip_ids.update(
            point.point_id for point in link_points if point.local_position[0] >= cutoff
        )

    output: list[GenericContactCandidate] = []
    for side, points in (("front", front_palm), ("back", back_palm)):
        output.extend(
            GenericContactCandidate(
                point_id=point.point_id,
                link_name=point.link_name,
                finger_name=point.finger_name,
                region=f"{side}_palm",
                local_position=point.local_position,
                local_surface_normal=point.local_surface_normal,
                supported_sides=(side,),
                source=f"authored_keypoint:{point.source_path}",
            )
            for point in points
        )
    for point in fingers:
        if point.point_id in shared_tip_ids:
            sides, region = ("front", "back"), "shared_fingertip"
        else:
            local_normal = np.asarray(point.local_surface_normal, dtype=np.float64)
            normal_in_palm = rest_fk[point.link_name][:3, :3] @ local_normal
            sides, region = _support_and_region(normal_in_palm, front_normal, threshold)
        output.append(
            GenericContactCandidate(
                point_id=point.point_id,
                link_name=point.link_name,
                finger_name=point.finger_name,
                region=region,
                local_position=point.local_position,
                local_surface_normal=point.local_surface_normal,
                supported_sides=sides,
                source=f"authored_keypoint:{point.source_path}",
            )
        )
    output.extend(_authored_thumb_candidates(hand, config, stage, cache, meshes))
    _check_output(
        output, float(config.require("contact_candidates.duplicate_tolerance"))
    )

    counts = Counter(candidate.region for candidate in output)
    required = {
        "front_palm",
        "back_palm",
        "front_finger_surface",
        "back_finger_surface",
        "shared_fingertip",
        "thumb",
    }
    if not required <= set(counts):
        raise RuntimeError(
            f"Generated metadata omits regions: {sorted(required - set(counts))}"
        )
    path = config.configured_path("contact_candidates.path")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "pipeline_revision": str(config.require("pipeline_revision")),
        "coordinate_convention": (
            "local_position and local_surface_normal are in owning link frame"
        ),
        "candidate_count": len(output),
        "region_counts": dict(sorted(counts.items())),
        "source": {
            "keypoints": "x2_mujoco/x2_keypoints.usda",
            "collision_hulls": "x2_mujoco/payloads/x2_physx_collision_hulls.usda",
            "classification": (
                "authored side metadata plus rest-FK normals for non-tip finger surfaces"
            ),
            "thumb_candidates": (
                "authored USD Sphere keypoints projected to owning collision hull; "
                "all support front and back"
            ),
        },
        "candidates": [candidate.to_dict() for candidate in output],
    }
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    path, counts = build(args.config)
    print(json.dumps({"path": str(path), "region_counts": counts}, default=dict))


if __name__ == "__main__":
    main()
