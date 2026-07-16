"""Tests for the restartable exact-valid X2 dataset collector."""

from __future__ import annotations

import argparse
import hashlib
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scripts.collect_x2_valid_dataset as collector
from scripts.collect_x2_valid_dataset import (
    FINGER_COUNTS,
    ValidCandidate,
    _finger_count,
    _planned_raw_count,
    _resume_incomplete_attempts,
    _attempt_completion_payload,
    _completed_attempt_roots,
    materialize_final,
    pair_candidates,
)


class X2ValidCollectorTest(unittest.TestCase):
    def test_formal_target_and_general_mesh_subset_are_hard_locked(self) -> None:
        self.assertEqual(collector.FORMAL_TARGET_VALID, 5000)
        self.assertEqual(collector.FORMAL_PER_SIDE_FINGER_TARGET, 500)
        self.assertEqual(collector.FORMAL_GENERAL_MESH_COUNT, 30)
        self.assertEqual(len(collector.FORMAL_GENERAL_MESH_IDS), 30)
        self.assertEqual(collector.FORMAL_GENERAL_MESH_IDS[0], "000")
        self.assertEqual(collector.FORMAL_GENERAL_MESH_IDS[-1], "087")
        defaults = collector._parse_args([])
        self.assertEqual(defaults.output_root.name, "x2_valid_5000")
        self.assertEqual(defaults.minimum_attempt_raw, 500)
        wrong = argparse.Namespace(target_valid=10000)
        with self.assertRaisesRegex(
            collector.ValidDatasetError, "exactly 5000 valid records"
        ):
            collector._run_locked(wrong)

    @staticmethod
    def _valid_record(
        finger_count: int,
        mesh_path: Path,
        side: str,
        names: list[str],
        raw_path: Path,
    ) -> dict:
        zero_dense_gate = {
            "evaluation_mode": "dense_bidirectional",
            "evaluated": True,
            "hand_surface_samples_per_set": (
                collector.DENSE_HAND_SURFACE_SAMPLES_PER_SET
            ),
            "hand_surface_samples_per_link": (
                collector.DENSE_HAND_SURFACE_SAMPLES_PER_LINK
            ),
            "hand_surface_point_count": collector.DENSE_HAND_SURFACE_POINT_COUNT,
            "object_surface_samples": collector.DENSE_OBJECT_SURFACE_SAMPLES,
            "forward_total_penetration": 0.0,
            "forward_maximum_penetration": 0.0,
            "reverse_total_penetration": 0.0,
            "reverse_maximum_penetration": 0.0,
            "total_penetration": 0.0,
            "maximum_penetration": 0.0,
            "feasible": True,
            "threshold": collector.FORMAL_RAW_PENETRATION_CAP,
        }
        positive_alphas = [
            float(value) for value in collector.FORMAL_CLOSING_ALPHAS
        ]
        return {
            "pipeline_revision": collector.GENERATOR_PIPELINE_REVISION,
            "active_side": side,
            "success": True,
            "simulation_success": True,
            "maximum_penetration": 0.0,
            "hand_object_penetration": zero_dense_gate,
            "validation": {
                "status": "passed",
                "backend": collector.VALIDATION_BACKEND,
                "protocol_revision": collector.VALIDATION_PROTOCOL_REVISION,
                "criterion": collector.VALIDATION_CRITERION,
                "source_raw": str(raw_path.resolve()),
                "source_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                "thresholds": {
                    "maximum_penetration_m": 0.001,
                    "maximum_object_displacement_m": 0.1,
                    "minimum_final_contact_force_n": 0.0,
                    "maximum_active_joint_error_rad": 0.1,
                    "maximum_newton_mimic_error_rad": 0.01,
                },
                "runtime": {
                    "simulation_steps": 100,
                    "substeps": collector.REQUIRED_SUBSTEPS,
                    "physics_step_count": 100 * collector.REQUIRED_SUBSTEPS,
                    "device": "cuda:0",
                    "actuator_drive": {
                        "stiffness_n_m_per_rad": collector.FORMAL_ACTUATOR_STIFFNESS,
                        "damping_n_m_s_per_rad": collector.FORMAL_ACTUATOR_DAMPING,
                        "armature_kg_m2": collector.FORMAL_ACTUATOR_ARMATURE,
                        "stiffness_source": "cli_override",
                        "damping_source": "cli_override",
                        "armature_source": "cli_override",
                        "effort_limit_source": "hand_usd",
                        "velocity_limit_source": "hand_usd",
                    },
                    "physx_solver": {
                        "solver_type": 1,
                        "external_forces_every_iteration": True,
                        "solve_articulation_contact_last": False,
                    },
                    "zero_gravity_preclose": {
                        "enabled": False,
                        "physics_step_count": 0,
                        "validation_physics_step_count_unchanged": (
                            100 * collector.REQUIRED_SUBSTEPS
                        ),
                        "displacement_reference": (
                            "raw_json_object_position_before_preclose"
                        ),
                    },
                    "contact_gradient_closing": {
                        "enabled": True,
                        "mode": "collision_aware_line_search",
                        "raw_penetration_cap_m": 0.001,
                        "target_penetration_cap_m": 0.0015,
                        "positive_alphas": positive_alphas,
                        "bidirectional_sampled_penetration": True,
                        "float32_quantized_and_rechecked": True,
                        "raw_json_remains_physx_initial_state": True,
                    },
                },
                "preflight": {
                    "raw_json_penetration_passed": True,
                    "penetration_passed": True,
                    "collision_aware_closing_raw_passed": True,
                    "self_collision_passed": True,
                    "hand_object_passed": True,
                    "collision_aware_closing": {
                        "enabled": True,
                        "raw_static_gate_passed": True,
                        "selected_target_safe": True,
                        "raw_penetration_passed": True,
                        "selected_penetration_passed": True,
                        "positive_alphas": positive_alphas,
                        "raw_penetration_cap_m": 0.001,
                        "target_penetration_cap_m": 0.0015,
                    },
                },
                "orientations": [
                    {
                        "name": name,
                        "passed": True,
                        "finite": True,
                        "hand_object_contact": True,
                        "gravity_vector_object_frame_m_s2": list(gravity),
                    }
                    for name, gravity in zip(
                        collector.EXPECTED_ORIENTATIONS,
                        collector.EXPECTED_GRAVITY_VECTORS,
                    )
                ],
                "passed_orientation_count": 6,
                "required_orientation_count": 6,
            },
            "object": {"mesh_path": str(mesh_path), "scale": 1.0},
            "finger_participation": {
                "target_count": finger_count,
                "actual_count": finger_count,
                "finger_names": names,
            },
        }

    def test_adaptive_raw_plan_is_conservative_and_has_floor(self) -> None:
        self.assertEqual(
            _planned_raw_count(
                deficit=100,
                raw_count=0,
                valid_count=0,
                minimum=250,
                oversample=1.25,
            ),
            250,
        )
        planned = _planned_raw_count(
            deficit=100,
            raw_count=1000,
            valid_count=200,
            minimum=10,
            oversample=1.25,
        )
        self.assertGreaterEqual(planned, 736)

    def test_pair_candidates_requires_same_object_and_disjoint_fingers(self) -> None:
        grouped: dict[tuple[str, int], list[ValidCandidate]] = {
            (side, value): []
            for side in ("front", "back")
            for value in FINGER_COUNTS
        }

        def candidate(
            label: str, side: str, count: int, names: list[str], object_id: str
        ) -> ValidCandidate:
            return ValidCandidate(
                path=Path(f"/{label}.json"),
                side=side,
                finger_count=count,
                finger_names=frozenset(names),
                object_id=object_id,
                object_scale=1.0,
            )

        grouped[("front", 1)] = [
            candidate("front_a", "front", 1, ["index"], "object_a"),
            candidate("front_b", "front", 1, ["thumb"], "object_b"),
        ]
        grouped[("back", 4)] = [
            # Same object but overlapping: this candidate must never be paired.
            candidate(
                "back_a_overlap",
                "back",
                4,
                ["index", "middle", "ring", "little"],
                "object_a",
            ),
            candidate(
                "back_a_complement",
                "back",
                4,
                ["middle", "ring", "little", "thumb"],
                "object_a",
            ),
            # Disjoint but a different object from front_a; object identity matters.
            candidate(
                "back_c",
                "back",
                4,
                ["middle", "ring", "little", "thumb"],
                "object_c",
            ),
            candidate(
                "back_b_complement",
                "back",
                4,
                ["index", "middle", "ring", "little"],
                "object_b",
            ),
        ]

        pairs = pair_candidates(grouped, per_side_finger_target=2)
        self.assertEqual(len(pairs[1]), 2)
        self.assertEqual(
            {front.object_id for front, _ in pairs[1]},
            {"object_a", "object_b"},
        )
        self.assertTrue(
            all(
                front.object_id == back.object_id
                and front.finger_names.isdisjoint(back.finger_names)
                for front, back in pairs[1]
            )
        )
        self.assertNotIn(
            Path("/back_a_overlap.json"),
            {back.path for _, back in pairs[1]},
        )

    def test_materialize_final_is_exact_and_balanced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "object" / "coacd" / "decomposed.obj"
            mesh.parent.mkdir(parents=True)
            mesh.write_text("mesh", encoding="utf-8")
            names = ["index", "middle", "ring", "little", "thumb"]
            grouped: dict[tuple[str, int], list[ValidCandidate]] = {
                (side, value): []
                for side in ("front", "back")
                for value in FINGER_COUNTS
            }
            for side in ("front", "back"):
                for finger_count in FINGER_COUNTS:
                    selected_names = (
                        names[:finger_count]
                        if side == "front"
                        else names[5 - finger_count :]
                    )
                    path = (
                        root
                        / "source"
                        / side
                        / "valid"
                        / f"{side}_f{finger_count}.json"
                    )
                    raw_path = path.parent.parent / "raw" / path.name
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_path.write_text(
                        json.dumps({"side": side, "finger_count": finger_count}),
                        encoding="utf-8",
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps(
                            self._valid_record(
                                finger_count,
                                mesh,
                                side,
                                selected_names,
                                raw_path,
                            )
                        ),
                        encoding="utf-8",
                    )
                    grouped[(side, finger_count)].append(
                        ValidCandidate(
                            path=path,
                            side=side,
                            finger_count=finger_count,
                            finger_names=frozenset(selected_names),
                            object_id="object",
                            object_scale=1.0,
                        )
                    )
                    self.assertEqual(
                        _finger_count(path, json.loads(path.read_text())), finger_count
                    )
            report = materialize_final(
                output_root=root / "output",
                grouped=grouped,
                per_side_finger_target=1,
            )
            self.assertEqual(report["valid_count"], 10)
            self.assertEqual(
                report["side_finger_counts"],
                {
                    side: {str(value): 1 for value in FINGER_COUNTS}
                    for side in ("front", "back")
                },
            )
            self.assertEqual(report["paired_entry_count"], 4)
            self.assertEqual(report["object_scale_by_id"], {"object": 1.0})
            self.assertTrue(
                report["generation_protocol"]["stratified_batching"]
            )
            self.assertTrue(
                all(record["object_scale"] == 1.0 for record in report["records"])
            )
            self.assertTrue(
                all(
                    set(entry["front_finger_names"]).isdisjoint(
                        entry["back_finger_names"]
                    )
                    for entry in report["merged_entries"]
                    if entry["pair_id"] is not None
                )
            )
            self.assertEqual(len(list((root / "output" / "final_valid").glob("**/*.json"))), 10)

    def test_incomplete_attempt_is_resumed_before_new_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempts = Path(directory)
            incomplete = attempts / "attempt_0003"
            incomplete.mkdir()
            (incomplete / "attempt.json").write_text(
                json.dumps(
                    {
                        "finger_counts": list(FINGER_COUNTS),
                        "complementary_side_fingers": True,
                        "raw_target": 1234,
                        "seed": 77,
                        "generation": {"finger_targets": {"1": 1234}},
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch(
                "scripts.collect_x2_valid_dataset._generate_attempt"
            ) as generate:
                _resume_incomplete_attempts(attempts, object())
            generate.assert_called_once_with(
                attempt_root=incomplete,
                finger_targets={1: 1234},
                seed=77,
                args=mock.ANY,
            )

    def test_only_audited_completion_marker_makes_attempt_countable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempts = Path(directory)
            attempt = attempts / "attempt_0000"
            attempt.mkdir()
            mesh = attempt / "object" / "coacd" / "decomposed.obj"
            mesh.parent.mkdir(parents=True)
            mesh.write_text("mesh", encoding="utf-8")
            mesh_sha256 = hashlib.sha256(mesh.read_bytes()).hexdigest()
            selection_manifest = attempt / "selection.json"
            selection_manifest.write_text("{}", encoding="utf-8")
            metadata = {
                "schema_version": 4,
                "collection_protocol_revision": collector.COLLECTION_PROTOCOL_REVISION,
                "raw_target": 10,
                "generation": {
                    "pipeline_revision": collector.GENERATOR_PIPELINE_REVISION,
                    "finger_counts": list(FINGER_COUNTS),
                    "finger_targets": {str(value): 2 for value in FINGER_COUNTS},
                    "n_iterations": 3,
                    "stratified_batching": True,
                    "stratified_batch_size": collector.STRATIFIED_BATCH_SIZE,
                    "primitive_object_scale": 1.0,
                },
                "validation": {
                    "backend": collector.VALIDATION_BACKEND,
                    "protocol_revision": collector.VALIDATION_PROTOCOL_REVISION,
                    "criterion": collector.VALIDATION_CRITERION,
                    "sim_steps": collector.REQUIRED_SIM_STEPS,
                    "substeps": collector.REQUIRED_SUBSTEPS,
                    "preclose_physics_steps": (
                        collector.REQUIRED_PRECLOSE_PHYSICS_STEPS
                    ),
                    "actuator_drive": {
                        "stiffness_n_m_per_rad": collector.FORMAL_ACTUATOR_STIFFNESS,
                        "damping_n_m_s_per_rad": collector.FORMAL_ACTUATOR_DAMPING,
                        "armature_kg_m2": collector.FORMAL_ACTUATOR_ARMATURE,
                    },
                    "physx_solver": {
                        "solver_type": 1,
                        "external_forces_every_iteration": True,
                        "solve_articulation_contact_last": False,
                    },
                },
                "objects": {
                    "primitive_ids": [],
                    "primitive_meshes": [],
                    "general_meshes": [
                        {
                            "kind": "general",
                            "object_id": "object",
                            "mesh_path": str(mesh.resolve()),
                            "sha256": mesh_sha256,
                            "object_scale": 1.0,
                        }
                    ],
                    "formal_general_mesh_ids": ["object"],
                    "general_selection_manifest": {
                        "path": str(selection_manifest.resolve()),
                        "sha256": hashlib.sha256(
                            selection_manifest.read_bytes()
                        ).hexdigest(),
                    },
                },
            }
            (attempt / "attempt.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with (attempt / "summary.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=("finger_count", "sample_count")
                )
                writer.writeheader()
                for side in ("front", "back"):
                    for finger_count in range(1, 6):
                        writer.writerow(
                            {"finger_count": finger_count, "sample_count": 1}
                        )
            generation_report_path = attempt / collector.GENERATION_SUMMARY_NAME
            generation_report = {
                "passed": True,
                "total_samples": 10,
                "total_finite_samples": 10,
                "instance_side_runs": 10,
                "summary_csv": str((attempt / "summary.csv").resolve()),
                "summary_json": str(generation_report_path.resolve()),
                "settings": {
                    "resume": True,
                    "stratified_batching": True,
                    "batch_size": collector.STRATIFIED_BATCH_SIZE,
                    "stratified_batch_size": collector.STRATIFIED_BATCH_SIZE,
                    "n_iterations": 3,
                    "finger_targets": {
                        str(value): 2 for value in FINGER_COUNTS
                    },
                    "finger_counts": list(FINGER_COUNTS),
                    "object_scale_by_id": {"object": 1.0},
                },
                "generator_calls": [
                    {
                        "instance_name": "object",
                        "side": side,
                        "finger_count": finger_count,
                        "object_scale": 1.0,
                        "stratified": True,
                    }
                    for side in ("front", "back")
                    for finger_count in FINGER_COUNTS
                ],
            }
            generation_report_path.write_text(
                json.dumps(generation_report), encoding="utf-8"
            )
            with (attempt / "validation_summary.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("candidate_count", "valid_count", "failed_count"),
                )
                writer.writeheader()
                writer.writerow(
                    {"candidate_count": 10, "valid_count": 1, "failed_count": 9}
                )
            for index in range(10):
                side = "front" if index < 5 else "back"
                name = f"sample_{index:02d}.json"
                raw = attempt / "general" / side / "raw" / name
                raw.parent.mkdir(parents=True, exist_ok=True)
                raw.write_text("{}", encoding="utf-8")
                status = "valid" if index == 0 else "failed"
                routed = raw.parent.parent / status / name
                routed.parent.mkdir(parents=True, exist_ok=True)
                routed.write_text("{}", encoding="utf-8")

            with mock.patch.object(collector, "PRIMITIVE_SPECS", ()), mock.patch.object(
                collector, "FORMAL_GENERAL_MESH_COUNT", 1
            ), mock.patch.object(
                collector, "FORMAL_GENERAL_MESH_IDS", ("object",)
            ):
                self.assertEqual(_completed_attempt_roots(attempts), [])
                marker = _attempt_completion_payload(attempt, metadata)
                (attempt / "complete.json").write_text(
                    json.dumps(marker), encoding="utf-8"
                )
                self.assertEqual(_completed_attempt_roots(attempts), [attempt])
                with (attempt / "validation_summary.csv").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write("10,1,9\n")
                with self.assertRaises(collector.ValidDatasetError):
                    _completed_attempt_roots(attempts)

    def test_official_general_mesh_scale_must_be_exactly_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "object" / "coacd" / "decomposed.obj"
            mesh.parent.mkdir(parents=True)
            mesh.write_text("mesh", encoding="utf-8")
            digest = hashlib.sha256(mesh.read_bytes()).hexdigest()
            manifest_path = root / "x2_general_mesh_manifest.json"

            def write_manifest(*, top_scale: float, row_scale: float) -> None:
                manifest_path.write_text(
                    json.dumps(
                        {
                            "passed": True,
                            "target_count": 1,
                            "selected_count": 1,
                            "license": "CC BY-NC 4.0 (DexGraspNet 2.0 assets)",
                            "object_scale": top_scale,
                            "meshes": [
                                {
                                    "object_id": "object",
                                    "sha256": digest,
                                    "object_scale": row_scale,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            with mock.patch.object(collector, "EXPECTED_GENERAL_MESHES", 1):
                write_manifest(top_scale=1.0, row_scale=0.5)
                with self.assertRaises(collector.ValidDatasetError):
                    collector._general_mesh_catalog(root)

                write_manifest(top_scale=1.0, row_scale=1.0)
                catalog = collector._general_mesh_catalog(root)
                collector._verify_selection_manifest(root, catalog)

                write_manifest(top_scale=0.5, row_scale=1.0)
                with self.assertRaises(collector.ValidDatasetError):
                    collector._verify_selection_manifest(root, catalog)

    def test_attempt_generator_command_enables_stratified_batching(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt_0000"
            commands: list[tuple[str, list[str]]] = []

            def fake_run(command: list[str], *, label: str) -> None:
                commands.append((label, command))
                if label == "complementary multi-finger generation":
                    (attempt / "summary.csv").write_text("summary\n", encoding="utf-8")
                    (attempt / collector.GENERATION_SUMMARY_NAME).write_text(
                        "{}", encoding="utf-8"
                    )
                else:
                    (attempt / "validation_summary.csv").write_text(
                        "summary\n", encoding="utf-8"
                    )

            args = argparse.Namespace(
                general_mesh_root=Path("general-meshes"),
                n_iterations=3,
                generation_device="cuda:0",
                jobs=2,
                validation_batch_size=64,
                sim_steps=100,
                validation_device="cuda:0",
            )
            with mock.patch.object(
                collector,
                "_attempt_metadata",
                return_value={"raw_target": 10},
            ), mock.patch.object(
                collector, "_run_checked", side_effect=fake_run
            ), mock.patch.object(
                collector, "_attempt_completion_payload", return_value={"passed": True}
            ), mock.patch.object(
                collector, "_assert_completed_attempt", return_value={}
            ):
                collector._generate_attempt(
                    attempt_root=attempt,
                    finger_targets={value: 2 for value in FINGER_COUNTS},
                    seed=7,
                    args=args,
                )

            generation_command = commands[0][1]
            self.assertIn("--resume", generation_command)
            self.assertIn("--stratified-batching", generation_command)
            start = generation_command.index("--general-mesh-ids") + 1
            end = generation_command.index("--side")
            self.assertEqual(
                tuple(generation_command[start:end]),
                collector.FORMAL_GENERAL_MESH_IDS,
            )


if __name__ == "__main__":
    unittest.main()
