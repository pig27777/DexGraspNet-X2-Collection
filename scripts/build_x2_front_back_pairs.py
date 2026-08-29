#!/usr/bin/env python3
"""Build cross-object front/back pair records from PhysX-passed X2 grasps.

Each output JSON contains two independently validated input qposes.  It does
not blend their joints or claim that they form one simultaneous physical
scene.  The objects may differ; the complementary front/back finger sets must
be disjoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_x2_valid_dataset import (  # noqa: E402
    FINGER_COUNTS,
    PAIRING_PROTOCOL_REVISION,
    SIDES,
    ValidCandidate,
    ValidDatasetError,
    _atomic_json,
    _valid_candidate,
    pair_candidates,
)


DEFAULT_ATTEMPTS = (
    PROJECT_ROOT / "data" / "x2_valid_5000" / "attempts" / "attempt_0000",
    PROJECT_ROOT / "data" / "x2_valid_5000" / "attempts" / "attempt_0001",
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "x2_valid_5000" / "front_back_pairs_snapshot"
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _source_inventory(
    attempt_roots: Sequence[Path],
) -> tuple[
    dict[tuple[str, int], list[ValidCandidate]],
    list[dict[str, Any]],
]:
    grouped = {
        (side, finger_count): []
        for side in SIDES
        for finger_count in FINGER_COUNTS
    }
    attempts: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for attempt_root in attempt_roots:
        root = attempt_root.expanduser().resolve()
        paths = sorted(root.glob("**/valid/*.json"))
        for path in paths:
            candidate = _valid_candidate(path)
            if candidate.path in seen_paths:
                raise ValidDatasetError(
                    f"validated source appears more than once: {candidate.path}"
                )
            seen_paths.add(candidate.path)
            grouped[(candidate.side, candidate.finger_count)].append(candidate)
        attempts.append(
            {
                "path": str(root),
                "complete": (root / "complete.json").is_file(),
                "passed_source_count": len(paths),
            }
        )
    return grouped, attempts


def build_snapshot(
    *,
    attempt_roots: Sequence[Path],
    output_root: Path,
    limit_per_combination: int,
) -> dict[str, Any]:
    output = output_root.expanduser().resolve()
    if output.exists():
        raise ValidDatasetError(f"output already exists: {output}")
    grouped, attempts = _source_inventory(attempt_roots)
    pairs = pair_candidates(grouped, limit_per_combination)
    paired_samples: list[dict[str, Any]] = []
    combination_counts: dict[str, int] = {}
    used_sources: set[Path] = set()

    # Import here so the formal writer and snapshot writer share one exact
    # paired-record schema without making it part of this script's CLI surface.
    from scripts.collect_x2_valid_dataset import _write_paired_sample

    for front_count in (1, 2, 3, 4):
        back_count = 5 - front_count
        values = pairs[front_count]
        label = f"front_f{front_count}_back_f{back_count}"
        combination_counts[label] = len(values)
        for index, (front, back) in enumerate(values):
            if front.path in used_sources or back.path in used_sources:
                raise ValidDatasetError("a validated source was reused across pairs")
            used_sources.update((front.path, back.path))
            pair_id = f"{label}_{index:06d}"
            paired_samples.append(
                _write_paired_sample(
                    final_pairs_root=output,
                    pair_id=pair_id,
                    front=front,
                    back=back,
                )
            )

    manifest = {
        "schema_version": 1,
        "passed": True,
        "formal_final": False,
        "pairing_protocol_revision": PAIRING_PROTOCOL_REVISION,
        "composition_semantics": (
            "two_independently_physx_validated_grasps; "
            "front/back objects may differ"
        ),
        "limit_per_combination": limit_per_combination,
        "source_attempts": attempts,
        "passed_source_count": sum(
            len(values) for values in grouped.values()
        ),
        "combination_counts": combination_counts,
        "paired_sample_count": len(paired_samples),
        "validated_source_count_used": len(used_sources),
        "paired_samples": paired_samples,
    }
    _atomic_json(output / "pair_manifest.json", manifest)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempt-root",
        action="append",
        type=Path,
        dest="attempt_roots",
        help="Repeat for each attempt to include; defaults to attempts 0000 and 0001.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit-per-combination",
        type=_positive_int,
        default=500,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_snapshot(
        attempt_roots=tuple(args.attempt_roots or DEFAULT_ATTEMPTS),
        output_root=args.output_root,
        limit_per_combination=args.limit_per_combination,
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
