#!/usr/bin/env python3
"""Independently audit the completed formal X2 5000-valid dataset.

This command is intentionally read-only.  It treats the final manifest as a
claim, re-hashes every final record and attempt completion proof, and re-runs
the strict v6/v7 JSON audits from the collector before reporting success.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_x2_valid_dataset import (  # noqa: E402
    COLLECTION_PROTOCOL_REVISION,
    EXPECTED_ORIENTATIONS,
    FINGER_COUNTS,
    FORMAL_GENERAL_MESH_COUNT,
    FORMAL_PER_SIDE_FINGER_TARGET,
    FORMAL_TARGET_VALID,
    GENERATOR_PIPELINE_REVISION,
    REQUIRED_SIM_STEPS,
    SIDES,
    STRATIFIED_BATCH_SIZE,
    VALIDATION_BACKEND,
    VALIDATION_CRITERION,
    VALIDATION_PROTOCOL_REVISION,
    ValidDatasetError,
    _assert_completed_attempt,
    _file_sha256,
    _formal_general_mesh_catalog,
    _general_mesh_catalog,
    _strict_json,
    _valid_candidate,
    _verify_selection_manifest,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "x2_valid_5000"
DEFAULT_GENERAL_MESH_ROOT = PROJECT_ROOT / "data" / "meshdata"
PAIR_ID_PATTERN = re.compile(
    r"front_f(?P<front>[1-4])_back_f(?P<back>[1-4])_(?P<index>\d{6})"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidDatasetError(message)


def _resolved_within(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValidDatasetError(f"{label} escapes {root}: {resolved}") from exc
    return resolved


def _audit_manifest_header(manifest: Mapping[str, Any]) -> None:
    expected_side_counts = {
        side: {str(value): FORMAL_PER_SIDE_FINGER_TARGET for value in FINGER_COUNTS}
        for side in SIDES
    }
    _require(manifest.get("passed") is True, "manifest.passed is not true")
    _require(
        manifest.get("collection_protocol_revision")
        == COLLECTION_PROTOCOL_REVISION,
        "manifest collection protocol is stale",
    )
    _require(
        manifest.get("target_valid") == FORMAL_TARGET_VALID
        and manifest.get("valid_count") == FORMAL_TARGET_VALID,
        "manifest does not claim exactly 5000 valid records",
    )
    _require(
        manifest.get("per_side_finger_target") == FORMAL_PER_SIDE_FINGER_TARGET,
        "manifest per-side/finger target is not 500",
    )
    _require(
        manifest.get("side_finger_counts") == expected_side_counts,
        "manifest side/finger counts are not exactly balanced",
    )
    _require(
        manifest.get("paired_entry_count") == FORMAL_PER_SIDE_FINGER_TARGET * 4,
        "manifest paired entry count is not 2000",
    )
    _require(
        manifest.get("single_side_five_finger_entry_count")
        == FORMAL_PER_SIDE_FINGER_TARGET * len(SIDES),
        "manifest single-side f5 entry count is not 1000",
    )
    _require(
        manifest.get("required_general_object_count") == FORMAL_GENERAL_MESH_COUNT
        and manifest.get("covered_general_object_count")
        == FORMAL_GENERAL_MESH_COUNT,
        "manifest does not prove all 30 formal general meshes are covered",
    )
    generation = manifest.get("generation_protocol")
    _require(
        isinstance(generation, dict)
        and generation.get("stratified_batching") is True
        and generation.get("stratified_batch_size") == STRATIFIED_BATCH_SIZE
        and generation.get("rectangular_contact_count_partitions") == [4, 5],
        "manifest generation protocol is incomplete",
    )
    validation = manifest.get("validation_protocol")
    _require(
        isinstance(validation, dict)
        and validation.get("backend") == VALIDATION_BACKEND
        and validation.get("protocol_revision") == VALIDATION_PROTOCOL_REVISION
        and validation.get("criterion") == VALIDATION_CRITERION
        and validation.get("sim_steps") == REQUIRED_SIM_STEPS
        and validation.get("required_orientations") == list(EXPECTED_ORIENTATIONS),
        "manifest PhysX v7 protocol is incomplete",
    )


def _audit_completion_proofs(
    *, output_root: Path, proofs: Any
) -> list[Path]:
    _require(isinstance(proofs, list) and proofs, "completion proofs are missing")
    attempts_root = (output_root / "attempts").resolve()
    expected_paths = sorted(
        path.resolve() for path in attempts_root.glob("attempt_*/complete.json")
    )
    audited_paths: list[Path] = []
    for index, proof in enumerate(proofs):
        _require(isinstance(proof, dict), f"completion proof {index} is not an object")
        path_value = proof.get("path")
        _require(isinstance(path_value, str), f"completion proof {index} has no path")
        path = _resolved_within(
            Path(path_value), attempts_root, label=f"completion proof {index}"
        )
        _require(
            path.name == "complete.json" and path.parent.name.startswith("attempt_"),
            f"completion proof {index} is not an attempt marker",
        )
        _require(path.is_file(), f"completion proof is missing: {path}")
        _require(
            proof.get("sha256") == _file_sha256(path),
            f"completion proof hash is stale: {path}",
        )
        _assert_completed_attempt(path.parent)
        audited_paths.append(path)
    _require(
        sorted(audited_paths) == expected_paths
        and len(set(audited_paths)) == len(audited_paths),
        "manifest completion proof inventory differs from completed attempts",
    )
    return audited_paths


def _record_finger_names(record: Mapping[str, Any], *, label: str) -> set[str]:
    names = record.get("finger_names")
    _require(
        isinstance(names, list)
        and all(isinstance(value, str) for value in names)
        and len(names) == len(set(names)),
        f"{label}: finger_names are invalid",
    )
    return set(names)


def _audit_pairing(
    records: Sequence[Mapping[str, Any]],
    merged_entries: Any,
    *,
    per_side_finger_target: int,
) -> dict[str, int]:
    strata = Counter((record.get("side"), record.get("finger_count")) for record in records)
    expected = {
        (side, value): per_side_finger_target
        for side in SIDES
        for value in FINGER_COUNTS
    }
    _require(strata == Counter(expected), "derived record strata are not exactly balanced")

    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    five_records: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in records:
        side = record.get("side")
        count = record.get("finger_count")
        pair_id = record.get("pair_id")
        _require(side in SIDES and count in FINGER_COUNTS, "record side/finger count is invalid")
        if count == 5:
            _require(pair_id is None, "f5 record must have pair_id=null")
            path_value = record.get("path")
            _require(isinstance(path_value, str), "f5 record path is missing")
            match = re.search(r"_(\d{6})\.json$", path_value)
            _require(match is not None, f"f5 record index is invalid: {path_value}")
            key = (str(side), int(match.group(1)))
            _require(key not in five_records, f"duplicate f5 record index: {key}")
            five_records[key] = record
        else:
            _require(isinstance(pair_id, str), "f1--f4 record is missing pair_id")
            by_pair[pair_id].append(record)

    expected_pair_count = per_side_finger_target * 4
    _require(len(by_pair) == expected_pair_count, "derived pair count is incorrect")
    for pair_id, values in by_pair.items():
        match = PAIR_ID_PATTERN.fullmatch(pair_id)
        _require(match is not None, f"pair id format is invalid: {pair_id}")
        _require(len(values) == 2, f"pair {pair_id} does not contain two records")
        front = next((value for value in values if value.get("side") == "front"), None)
        back = next((value for value in values if value.get("side") == "back"), None)
        _require(front is not None and back is not None, f"pair {pair_id} lacks one side")
        front_count = front.get("finger_count")
        back_count = back.get("finger_count")
        _require(
            front_count == int(match.group("front"))
            and back_count == int(match.group("back"))
            and front_count + back_count == 5,
            f"pair {pair_id} finger counts are not complementary",
        )
        _require(
            front.get("object_id") == back.get("object_id"),
            f"pair {pair_id} uses different objects",
        )
        _require(
            _record_finger_names(front, label=pair_id).isdisjoint(
                _record_finger_names(back, label=pair_id)
            ),
            f"pair {pair_id} has overlapping front/back fingers",
        )

    _require(
        len(five_records) == per_side_finger_target * len(SIDES),
        "derived f5 single-side record count is incorrect",
    )
    _require(
        isinstance(merged_entries, list)
        and len(merged_entries) == expected_pair_count + len(five_records),
        "merged_entries length is incorrect",
    )
    pair_entries: dict[str, Mapping[str, Any]] = {}
    five_entries: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in merged_entries:
        _require(isinstance(entry, dict), "merged entry is not an object")
        pair_id = entry.get("pair_id")
        if pair_id is not None:
            _require(
                isinstance(pair_id, str) and pair_id not in pair_entries,
                "merged pair entry is duplicated or invalid",
            )
            pair_entries[pair_id] = entry
            continue
        side = next(
            (value for value in SIDES if f"{value}_finger_count" in entry), None
        )
        index = entry.get("index")
        _require(
            side is not None
            and entry.get(f"{side}_finger_count") == 5
            and isinstance(index, int),
            "merged f5 entry is invalid",
        )
        key = (side, index)
        _require(key not in five_entries, f"duplicate merged f5 entry: {key}")
        five_entries[key] = entry
    _require(set(pair_entries) == set(by_pair), "merged pair ids differ from records")
    _require(set(five_entries) == set(five_records), "merged f5 entries differ from records")

    for pair_id, values in by_pair.items():
        front = next(value for value in values if value.get("side") == "front")
        back = next(value for value in values if value.get("side") == "back")
        entry = pair_entries[pair_id]
        _require(
            entry.get("object_id") == front.get("object_id")
            and entry.get("front_finger_count") == front.get("finger_count")
            and entry.get("back_finger_count") == back.get("finger_count")
            and entry.get("front_finger_names")
            == sorted(_record_finger_names(front, label=pair_id))
            and entry.get("back_finger_names")
            == sorted(_record_finger_names(back, label=pair_id))
            and entry.get("disjoint") is True,
            f"merged entry does not reproduce pair {pair_id}",
        )
    for key, record in five_records.items():
        entry = five_entries[key]
        side, _ = key
        _require(
            entry.get("object_id") == record.get("object_id")
            and entry.get(f"{side}_finger_names")
            == sorted(_record_finger_names(record, label=f"f5 {key}")),
            f"merged f5 entry does not reproduce record {key}",
        )
    return {
        "paired_entry_count": len(by_pair),
        "single_side_five_finger_entry_count": len(five_records),
    }


def audit_dataset(*, output_root: Path, general_mesh_root: Path) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    general_mesh_root = general_mesh_root.expanduser().resolve()
    manifest_path = output_root / "manifest.json"
    _require(manifest_path.is_file(), f"final manifest is missing: {manifest_path}")
    manifest = _strict_json(manifest_path)
    _audit_manifest_header(manifest)

    official_catalog = _general_mesh_catalog(general_mesh_root)
    _verify_selection_manifest(general_mesh_root, official_catalog)
    formal_catalog = _formal_general_mesh_catalog(official_catalog)
    required_general_ids = {entry["object_id"] for entry in formal_catalog}
    _require(
        len(required_general_ids) == FORMAL_GENERAL_MESH_COUNT,
        "formal general mesh catalog is not exactly 30 objects",
    )
    completion_paths = _audit_completion_proofs(
        output_root=output_root,
        proofs=manifest.get("attempt_completion_proofs"),
    )

    records_value = manifest.get("records")
    _require(
        isinstance(records_value, list) and len(records_value) == FORMAL_TARGET_VALID,
        "manifest records list is not exactly 5000",
    )
    records: list[Mapping[str, Any]] = []
    final_root = (output_root / "final_valid").resolve()
    attempts_root = (output_root / "attempts").resolve()
    seen_paths: set[Path] = set()
    seen_sources: set[Path] = set()
    stratum_indices: dict[tuple[str, int], set[int]] = defaultdict(set)
    object_scales: dict[str, float] = {}
    for index, record in enumerate(records_value):
        _require(isinstance(record, dict), f"record {index} is not an object")
        path_value = record.get("path")
        source_value = record.get("source")
        _require(
            isinstance(path_value, str) and isinstance(source_value, str),
            f"record {index} path/source is missing",
        )
        path = _resolved_within(Path(path_value), final_root, label=f"record {index}")
        source = _resolved_within(
            Path(source_value), attempts_root, label=f"record {index} source"
        )
        _require(
            path.is_file() and source.is_file(),
            f"record {index} file/source is missing",
        )
        _require(path not in seen_paths, f"final path is duplicated: {path}")
        _require(source not in seen_sources, f"validated source is reused: {source}")
        seen_paths.add(path)
        seen_sources.add(source)
        _require(
            record.get("sha256") == _file_sha256(path),
            f"record hash is stale: {path}",
        )
        _require(
            os.path.samefile(path, source),
            f"final record is not its source hard link: {path}",
        )
        candidate = _valid_candidate(source)
        _require(
            record.get("side") == candidate.side
            and record.get("finger_count") == candidate.finger_count
            and record.get("finger_names") == sorted(candidate.finger_names)
            and record.get("object_id") == candidate.object_id
            and record.get("object_scale") == candidate.object_scale,
            f"manifest metadata differs from validated source: {source}",
        )
        expected_parent = final_root / candidate.side / f"f{candidate.finger_count}"
        _require(path.parent == expected_parent, f"record is in the wrong stratum: {path}")
        match = re.fullmatch(
            rf"x2_{candidate.side}_f{candidate.finger_count}_(\d{{6}})\.json",
            path.name,
        )
        _require(match is not None, f"final filename is invalid: {path.name}")
        stratum_indices[(candidate.side, candidate.finger_count)].add(
            int(match.group(1))
        )
        previous_scale = object_scales.setdefault(
            candidate.object_id, candidate.object_scale
        )
        _require(
            previous_scale == candidate.object_scale,
            f"object scale drift: {candidate.object_id}",
        )
        records.append(record)

    expected_indices = set(range(FORMAL_PER_SIDE_FINGER_TARGET))
    _require(
        all(
            stratum_indices[(side, value)] == expected_indices
            for side in SIDES
            for value in FINGER_COUNTS
        ),
        "one or more final strata do not use continuous indices 000000--000499",
    )
    final_files = {path.resolve() for path in final_root.glob("**/*.json")}
    _require(
        final_files == seen_paths,
        "final_valid contains missing or unmanifested JSON files",
    )

    pairing = _audit_pairing(
        records,
        manifest.get("merged_entries"),
        per_side_finger_target=FORMAL_PER_SIDE_FINGER_TARGET,
    )
    object_ids = set(object_scales)
    _require(
        required_general_ids <= object_ids,
        "derived final records miss a formal general mesh",
    )
    _require(
        manifest.get("object_ids") == sorted(object_ids)
        and manifest.get("object_count") == len(object_ids),
        "manifest object inventory differs from final records",
    )
    _require(
        manifest.get("object_scale_by_id")
        == {object_id: object_scales[object_id] for object_id in sorted(object_scales)},
        "manifest object scale inventory differs from final records",
    )
    _require(
        all(scale == 1.0 for scale in object_scales.values()),
        "formal final records must all use object scale 1.0",
    )

    return {
        "passed": True,
        "manifest": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "valid_count": len(records),
        "side_finger_counts": manifest["side_finger_counts"],
        **pairing,
        "object_count": len(object_ids),
        "required_general_object_count": len(required_general_ids),
        "covered_general_object_count": len(required_general_ids & object_ids),
        "attempt_completion_proof_count": len(completion_paths),
        "audited_record_sha256_count": len(seen_paths),
        "generator_pipeline_revision": GENERATOR_PIPELINE_REVISION,
        "validation_protocol_revision": VALIDATION_PROTOCOL_REVISION,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--general-mesh-root", type=Path, default=DEFAULT_GENERAL_MESH_ROOT
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = audit_dataset(
            output_root=args.output_root, general_mesh_root=args.general_mesh_root
        )
    except Exception as exc:
        print(
            json.dumps(
                {"passed": False, "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
                allow_nan=False,
            )
        )
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
