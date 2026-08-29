#!/usr/bin/env python3
"""Build mode views and auditable two-object warm-start candidates.

The X2 generator calls the two contact surfaces ``front`` and ``back``.  For
this derived dataset the project convention is:

    front -> right mode
    back  -> left mode

Every source record remains a strictly audited, single-object PhysX pass.
Dual-object candidates are *not* declared valid: they combine disjoint finger
actuators and object poses in one hand frame, then explicitly require a future
joint static/PhysX validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from collections import deque
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_x2_valid_dataset import (  # noqa: E402
    FINGER_COUNTS,
    SIDES,
    ValidCandidate,
    ValidDatasetError,
    _atomic_json,
    _file_sha256,
    _strict_json,
    _valid_candidate,
    discover_attempt_valid,
)


MODE_BY_SIDE = {"front": "right", "back": "left"}
SIDE_BY_MODE = {value: key for key, value in MODE_BY_SIDE.items()}
FINGER_PREFIX = {
    "index": "rh_FF",
    "middle": "rh_MF",
    "ring": "rh_RF",
    "little": "rh_LF",
    "thumb": "rh_TH",
}
PROTOCOL_REVISION = "x2_right_left_dual_object_warm_start_v1"


def _object_pose_in_hand_frame(payload: dict[str, Any], source: Path) -> dict[str, Any]:
    hand_pose = payload.get("hand_pose")
    if not isinstance(hand_pose, dict):
        raise ValidDatasetError(f"{source}: hand_pose is missing")
    translation = hand_pose.get("translation")
    rotation = hand_pose.get("rotation_matrix")
    if (
        not isinstance(translation, list)
        or len(translation) != 3
        or not isinstance(rotation, list)
        or len(rotation) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in rotation)
    ):
        raise ValidDatasetError(f"{source}: hand_pose transform is malformed")
    try:
        t = [float(value) for value in translation]
        r = [[float(value) for value in row] for row in rotation]
    except (TypeError, ValueError) as exc:
        raise ValidDatasetError(f"{source}: hand_pose transform is non-numeric") from exc
    # The source stores hand-in-object T_O_H.  A shared hand frame needs T_H_O.
    inverse_rotation = [[r[column][row] for column in range(3)] for row in range(3)]
    inverse_translation = [
        -sum(inverse_rotation[row][column] * t[column] for column in range(3))
        for row in range(3)
    ]
    return {
        "translation": inverse_translation,
        "rotation_matrix": inverse_rotation,
        "frame": "shared_hand",
        "derived_from": "inverse(source.hand_pose)",
    }


def _actuator_map(payload: dict[str, Any], source: Path) -> dict[str, float]:
    return _named_vector(payload, source, "actuator_names", "actuator")


def _named_vector(
    payload: dict[str, Any],
    source: Path,
    names_key: str,
    values_key: str,
) -> dict[str, float]:
    names = payload.get(names_key)
    values = payload.get(values_key)
    if (
        not isinstance(names, list)
        or not isinstance(values, list)
        or len(names) != len(values)
        or any(not isinstance(name, str) for name in names)
    ):
        raise ValidDatasetError(
            f"{source}: {names_key}/{values_key} fields are malformed"
        )
    if len(names) != len(set(names)):
        raise ValidDatasetError(f"{source}: {names_key} contains duplicates")
    try:
        return {name: float(value) for name, value in zip(names, values)}
    except (TypeError, ValueError) as exc:
        raise ValidDatasetError(
            f"{source}: {values_key} contains non-numeric values"
        ) from exc


def _finger_for_actuator(name: str) -> str:
    matches = [
        finger for finger, prefix in FINGER_PREFIX.items() if name.startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValidDatasetError(f"Cannot assign actuator {name!r} to one X2 finger")
    return matches[0]


def _combined_named_state(
    right_payload: dict[str, Any],
    left_payload: dict[str, Any],
    right: ValidCandidate,
    left: ValidCandidate,
    *,
    names_key: str,
    values_key: str,
) -> tuple[list[str], list[float], dict[str, str]]:
    if (
        not right.finger_names.isdisjoint(left.finger_names)
        or right.finger_names | left.finger_names != set(FINGER_PREFIX)
    ):
        raise ValidDatasetError(
            "Combined candidate requires complementary, disjoint finger sets"
        )
    right_values = _named_vector(
        right_payload, right.path, names_key, values_key
    )
    left_values = _named_vector(left_payload, left.path, names_key, values_key)
    if list(right_values) != list(left_values):
        raise ValidDatasetError(
            f"{right.path} and {left.path}: {names_key} name/order mismatch"
        )
    owners: dict[str, str] = {}
    combined: list[float] = []
    for name in right_values:
        finger = _finger_for_actuator(name)
        if finger in right.finger_names:
            owners[finger] = "right"
            combined.append(right_values[name])
        elif finger in left.finger_names:
            owners[finger] = "left"
            combined.append(left_values[name])
        else:
            raise ValidDatasetError(
                f"Complementary candidate leaves actuator finger {finger!r} unowned"
            )
    if set(owners) != set(FINGER_PREFIX):
        raise ValidDatasetError(
            f"Combined {values_key} does not assign all five fingers"
        )
    return list(right_values), combined, owners


def _combined_actuators(
    right_payload: dict[str, Any],
    left_payload: dict[str, Any],
    right: ValidCandidate,
    left: ValidCandidate,
) -> tuple[list[str], list[float], dict[str, str]]:
    return _combined_named_state(
        right_payload,
        left_payload,
        right,
        left,
        names_key="actuator_names",
        values_key="actuator",
    )


def _combined_joints(
    right_payload: dict[str, Any],
    left_payload: dict[str, Any],
    right: ValidCandidate,
    left: ValidCandidate,
) -> tuple[list[str], list[float]]:
    names, values, _ = _combined_named_state(
        right_payload,
        left_payload,
        right,
        left,
        names_key="joint_names",
        values_key="joint",
    )
    return names, values


def _discover_sources(
    attempts_root: Path,
    *,
    include_incomplete_attempts: bool,
) -> tuple[
    dict[tuple[str, int], list[ValidCandidate]],
    list[dict[str, Any]],
]:
    if not include_incomplete_attempts:
        grouped = discover_attempt_valid(attempts_root)
    else:
        grouped = {
            (side, count): []
            for side in SIDES
            for count in FINGER_COUNTS
        }
        seen: set[Path] = set()
        for path in sorted(attempts_root.glob("attempt_*/**/valid/*.json")):
            candidate = _valid_candidate(path)
            if candidate.path in seen:
                raise ValidDatasetError(
                    f"validated source appears more than once: {candidate.path}"
                )
            seen.add(candidate.path)
            grouped[(candidate.side, candidate.finger_count)].append(candidate)
    source_attempts = [
        {
            "path": str(path.resolve()),
            "complete": (path / "complete.json").is_file(),
            "passed_route_count": sum(
                1 for _ in path.glob("**/valid/*.json")
            ),
        }
        for path in sorted(attempts_root.glob("attempt_*"))
        if path.is_dir()
    ]
    return grouped, source_attempts


def _round_robin_different_object_pairs(
    right_values: Iterable[ValidCandidate],
    left_values: Iterable[ValidCandidate],
    limit: int | None,
) -> list[tuple[ValidCandidate, ValidCandidate]]:
    """Return a maximum-cardinality, deterministic cross-object matching."""

    all_fingers = frozenset(FINGER_PREFIX)
    right_by_fingers: dict[frozenset[str], list[ValidCandidate]] = {}
    left_by_complement: dict[frozenset[str], list[ValidCandidate]] = {}
    for candidate in right_values:
        right_by_fingers.setdefault(candidate.finger_names, []).append(candidate)
    for candidate in left_values:
        complement = all_fingers - candidate.finger_names
        left_by_complement.setdefault(complement, []).append(candidate)

    def match_group(
        right_group: list[ValidCandidate],
        left_group: list[ValidCandidate],
    ) -> list[tuple[ValidCandidate, ValidCandidate]]:
        right_by_object: dict[str, deque[ValidCandidate]] = {}
        left_by_object: dict[str, deque[ValidCandidate]] = {}
        for candidate in sorted(
            right_group, key=lambda value: (value.object_id, str(value.path))
        ):
            right_by_object.setdefault(candidate.object_id, deque()).append(candidate)
        for candidate in sorted(
            left_group, key=lambda value: (value.object_id, str(value.path))
        ):
            left_by_object.setdefault(candidate.object_id, deque()).append(candidate)
        right_objects = sorted(right_by_object)
        left_objects = sorted(left_by_object)
        source = 0
        right_offset = 1
        left_offset = right_offset + len(right_objects)
        sink = left_offset + len(left_objects)
        graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

        def add_edge(start: int, end: int, capacity: int) -> list[int]:
            forward = [end, len(graph[end]), capacity, capacity]
            reverse = [start, len(graph[start]), 0, 0]
            graph[start].append(forward)
            graph[end].append(reverse)
            return forward

        for index, object_id in enumerate(right_objects):
            add_edge(
                source,
                right_offset + index,
                len(right_by_object[object_id]),
            )
        pair_edges: dict[tuple[str, str], list[int]] = {}
        infinite_capacity = len(right_group) + len(left_group)
        for right_index, right_object in enumerate(right_objects):
            for left_index, left_object in enumerate(left_objects):
                if right_object != left_object:
                    pair_edges[(right_object, left_object)] = add_edge(
                        right_offset + right_index,
                        left_offset + left_index,
                        infinite_capacity,
                    )
        for index, object_id in enumerate(left_objects):
            add_edge(
                left_offset + index,
                sink,
                len(left_by_object[object_id]),
            )

        while True:
            level = [-1] * len(graph)
            level[source] = 0
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for end, _, capacity, _ in graph[node]:
                    if capacity > 0 and level[end] < 0:
                        level[end] = level[node] + 1
                        queue.append(end)
            if level[sink] < 0:
                break
            cursor = [0] * len(graph)

            def send(node: int, flow: int) -> int:
                if node == sink:
                    return flow
                while cursor[node] < len(graph[node]):
                    edge = graph[node][cursor[node]]
                    end, reverse_index, capacity, _ = edge
                    if capacity > 0 and level[end] == level[node] + 1:
                        pushed = send(end, min(flow, capacity))
                        if pushed:
                            edge[2] -= pushed
                            graph[end][reverse_index][2] += pushed
                            return pushed
                    cursor[node] += 1
                return 0

            while send(source, infinite_capacity):
                pass

        result: list[tuple[ValidCandidate, ValidCandidate]] = []
        for right_object in right_objects:
            for left_object in left_objects:
                edge = pair_edges.get((right_object, left_object))
                if edge is None:
                    continue
                flow = edge[3] - edge[2]
                for _ in range(flow):
                    result.append(
                        (
                            right_by_object[right_object].popleft(),
                            left_by_object[left_object].popleft(),
                        )
                    )
        return result

    pairs: list[tuple[ValidCandidate, ValidCandidate]] = []
    for finger_names in sorted(
        right_by_fingers, key=lambda value: tuple(sorted(value))
    ):
        values = match_group(
            right_by_fingers[finger_names],
            left_by_complement.get(finger_names, []),
        )
        remaining = None if limit is None else limit - len(pairs)
        pairs.extend(values if remaining is None else values[:remaining])
        if limit is not None and len(pairs) == limit:
            return pairs
    return pairs


def _prepare_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise ValidDatasetError(
                f"Output already exists: {output_root}; pass --overwrite to replace it"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)


def build_dataset(
    attempts_root: Path,
    output_root: Path,
    *,
    pairs_per_combination: int | None = None,
    overwrite: bool = False,
    include_incomplete_attempts: bool = False,
) -> dict[str, Any]:
    grouped, source_attempts = _discover_sources(
        attempts_root,
        include_incomplete_attempts=include_incomplete_attempts,
    )
    _prepare_output(output_root, overwrite)
    records: list[dict[str, Any]] = []
    mode_counts: Counter[tuple[str, int]] = Counter()

    for side, mode in MODE_BY_SIDE.items():
        for finger_count in FINGER_COUNTS:
            directory = output_root / "single_object" / mode / f"f{finger_count}"
            directory.mkdir(parents=True, exist_ok=True)
            for index, candidate in enumerate(grouped[(side, finger_count)]):
                destination = directory / f"x2_{mode}_f{finger_count}_{index:06d}.json"
                os.link(candidate.path, destination)
                records.append(
                    {
                        "path": str(destination.resolve()),
                        "source": str(candidate.path),
                        "sha256": _file_sha256(destination),
                        "mode": mode,
                        "source_active_side": side,
                        "finger_count": finger_count,
                        "finger_names": sorted(candidate.finger_names),
                        "object_id": candidate.object_id,
                        "object_scale": candidate.object_scale,
                        "single_object_validation": "passed",
                    }
                )
                mode_counts[(mode, finger_count)] += 1

    dual_records: list[dict[str, Any]] = []
    dual_root = output_root / "dual_object_candidates"
    for right_count in (1, 2, 3, 4):
        left_count = 5 - right_count
        pairs = _round_robin_different_object_pairs(
            grouped[("front", right_count)],
            grouped[("back", left_count)],
            pairs_per_combination,
        )
        directory = dual_root / f"right_f{right_count}_left_f{left_count}"
        directory.mkdir(parents=True, exist_ok=True)
        for index, (right, left) in enumerate(pairs):
            right_payload = _strict_json(right.path)
            left_payload = _strict_json(left.path)
            names, actuator, owners = _combined_actuators(
                right_payload, left_payload, right, left
            )
            joint_names, joint = _combined_joints(
                right_payload, left_payload, right, left
            )
            candidate_id = (
                f"right_f{right_count}_left_f{left_count}_{index:06d}"
            )
            payload = {
                "schema_version": 1,
                "protocol_revision": PROTOCOL_REVISION,
                "candidate_id": candidate_id,
                "mode_mapping": dict(MODE_BY_SIDE),
                "hand": {
                    "frame": "shared_hand",
                    "actuator_names": names,
                    "actuator": actuator,
                    "joint_names": joint_names,
                    "joint": joint,
                    "finger_mode_owner": owners,
                    "state_source": (
                        "finger-owned raw input qpose from the two independently "
                        "PhysX-passed source records"
                    ),
                    "closing_actuator_target_saved": False,
                    "pose": {
                        "translation": [0.0, 0.0, 0.0],
                        "rotation_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                    },
                },
                "objects": [
                    {
                        "slot": "right",
                        "mode": "right",
                        "source_active_side": "front",
                        "object_id": right.object_id,
                        "mesh_path": right_payload["object"]["mesh_path"],
                        "scale": right.object_scale,
                        "finger_names": sorted(right.finger_names),
                        "pose_in_shared_hand_frame": _object_pose_in_hand_frame(
                            right_payload, right.path
                        ),
                        "source_valid": str(right.path),
                        "source_valid_sha256": _file_sha256(right.path),
                        "source_closing": right_payload["validation"]["preflight"][
                            "collision_aware_closing"
                        ],
                    },
                    {
                        "slot": "left",
                        "mode": "left",
                        "source_active_side": "back",
                        "object_id": left.object_id,
                        "mesh_path": left_payload["object"]["mesh_path"],
                        "scale": left.object_scale,
                        "finger_names": sorted(left.finger_names),
                        "pose_in_shared_hand_frame": _object_pose_in_hand_frame(
                            left_payload, left.path
                        ),
                        "source_valid": str(left.path),
                        "source_valid_sha256": _file_sha256(left.path),
                        "source_closing": left_payload["validation"]["preflight"][
                            "collision_aware_closing"
                        ],
                    },
                ],
                "composition_checks": {
                    "different_objects": right.object_id != left.object_id,
                    "disjoint_finger_sets": right.finger_names.isdisjoint(
                        left.finger_names
                    ),
                    "all_five_fingers_assigned": set(owners) == set(FINGER_PREFIX),
                },
                "dual_object_validation": {
                    "status": "not_run",
                    "single_object_passes_are_proven": True,
                    "required_before_use": [
                        "joint_hand_object_penetration",
                        "object_object_collision",
                        "shared_actuator_forward_kinematics",
                        "six_orientation_dual_object_physx",
                        "both_objects_retain_hand_contact",
                    ],
                },
            }
            destination = directory / f"{candidate_id}.json"
            _atomic_json(destination, payload)
            dual_records.append(
                {
                    "candidate_id": candidate_id,
                    "path": str(destination.resolve()),
                    "sha256": _file_sha256(destination),
                    "right_finger_count": right_count,
                    "left_finger_count": left_count,
                    "right_object_id": right.object_id,
                    "left_object_id": left.object_id,
                    "dual_object_validation": "not_run",
                }
            )

    manifest = {
        "schema_version": 1,
        "protocol_revision": PROTOCOL_REVISION,
        "mode_mapping": dict(MODE_BY_SIDE),
        "source_policy": (
            "strictly audited passed routes; incomplete attempts explicitly allowed"
            if include_incomplete_attempts
            else "strictly audited valid records from completed attempts only"
        ),
        "formal_source_completion_required": not include_incomplete_attempts,
        "formal_final": False,
        "attempts_root": str(attempts_root.resolve()),
        "source_attempts": source_attempts,
        "single_object_record_count": len(records),
        "single_object_counts": {
            mode: {
                f"f{finger_count}": mode_counts[(mode, finger_count)]
                for finger_count in FINGER_COUNTS
            }
            for mode in ("right", "left")
        },
        "dual_object_candidate_count": len(dual_records),
        "dual_object_status": "not_validated",
        "records": records,
        "dual_object_candidates": dual_records,
    }
    _atomic_json(output_root / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempts-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "x2_valid_5000" / "attempts",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "x2_dual_object",
    )
    parser.add_argument(
        "--pairs-per-combination",
        type=int,
        default=None,
        help="optional maximum for each right fK + left f(5-K) combination",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-incomplete-attempts",
        action="store_true",
        help=(
            "also use strict passed routes from incomplete attempts; output remains "
            "an interim, non-formal warm-start dataset"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.pairs_per_combination is not None and args.pairs_per_combination <= 0:
        raise SystemExit("--pairs-per-combination must be positive")
    try:
        manifest = build_dataset(
            args.attempts_root.expanduser().resolve(),
            args.output_root.expanduser().resolve(),
            pairs_per_combination=args.pairs_per_combination,
            overwrite=args.overwrite,
            include_incomplete_attempts=args.include_incomplete_attempts,
        )
    except ValidDatasetError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(
        f"single_object={manifest['single_object_record_count']} "
        f"dual_candidates={manifest['dual_object_candidate_count']} "
        f"output={args.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
