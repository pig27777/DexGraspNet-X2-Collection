from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from grasp_generation.x2_isaac_validation import (
    CLOSING_LINE_SEARCH_ALPHAS,
    EXPECTED_ACTUATOR_NAMES,
    EXPECTED_JOINT_NAMES,
    FORMAL_ACTUATOR_ARMATURE,
    FORMAL_ACTUATOR_DAMPING,
    FORMAL_ACTUATOR_STIFFNESS,
    GRAVITY_TESTS_WXYZ,
    OrientationOutcome,
    PASSIVE_MIMIC_DRIVERS,
    PROTOCOL_REVISION,
    ValidationThresholds,
    X2ValidationError,
    compact_collision_aware_closing_batch_audit,
    discover_raw_candidates,
    evaluate_orientation,
    load_raw_candidate,
    make_object_centered_replay,
    make_validated_record,
    quaternion_matrix_wxyz,
    quaternion_wxyz_to_xyzw,
    select_collision_aware_closing,
    summarize_newton_mimic_errors,
    validation_output_path,
    write_validated_record,
)
from scripts.build_x2_primitive_dataset import selected_specs
from scripts.generate_x2_primitive_dataset import GeneralMeshSpec
import scripts.validate_x2_primitive_dataset as primitive_validator
from scripts.validate_x2_primitive_dataset import build_validator_command


def _record(mesh_path: Path, *, side: str = "front") -> dict:
    actuator = [0.01 * index for index in range(12)]
    actuator_by_name = dict(zip(EXPECTED_ACTUATOR_NAMES, actuator))
    joint = [
        actuator_by_name[name if name in actuator_by_name else PASSIVE_MIMIC_DRIVERS[name]]
        for name in EXPECTED_JOINT_NAMES
    ]
    return {
        "schema_version": 1,
        "pipeline_revision": "test",
        "sample_index": 0,
        "active_side": side,
        "object": {
            "mesh_path": str(mesh_path.resolve()),
            "scale": 1.0,
            "watertight": True,
        },
        "hand_pose": {
            "translation": [1.0, 0.0, 0.0],
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "actuator_names": list(EXPECTED_ACTUATOR_NAMES),
        "actuator": actuator,
        "joint_names": list(EXPECTED_JOINT_NAMES),
        "joint": joint,
        "selected_contact_ids": ["p0", "p1", "p2", "p3"],
        "selected_contacts": [
            {
                "point_id": f"p{index}",
                "link_name": f"link_{index}",
                "finger_name": "index",
                "region": "shared_fingertip",
                "source": f"authored_keypoint:test_{index}",
                "enabled": True,
                "local_position": [0.001 * index, 0.0, 0.0],
                "world_position": [0.001 * index, 0.0, 0.0],
                "local_surface_normal": [1.0, 0.0, 0.0],
                "world_surface_normal": [1.0, 0.0, 0.0],
                "supported_sides": ["front", "back"],
            }
            for index in range(4)
        ],
        "energy": {"initial_total": 10.0, "total": 1.0, "terms": {}},
        "maximum_penetration": 0.0005,
        "finite": True,
        "success": False,
        "simulation_success": False,
        "validation": {"status": "not_run", "backend": None},
    }


def _closing_audit(
    *,
    raw_maximum: float | None = 0.0005,
    selected_maximum: float | None = 0.0005,
    raw_cap: float = 0.001,
    target_cap: float = 0.0015,
    enabled: bool = True,
    raw_state_finite: bool = True,
    raw_limits_passed: bool = True,
    selected_state_finite: bool = True,
    selected_limits_passed: bool = True,
) -> dict:
    raw_penetration_passed = raw_maximum is not None and raw_maximum < raw_cap
    raw_gate = raw_state_finite and raw_limits_passed and raw_penetration_passed
    selected_penetration_passed = (
        selected_maximum is not None and selected_maximum < target_cap
    )
    selected_safe = (
        raw_gate
        and selected_state_finite
        and selected_limits_passed
        and selected_penetration_passed
    )
    selected_alpha = 1.0 if enabled and selected_safe else 0.0
    return {
        "enabled": enabled,
        "positive_alphas": (
            [float(value) for value in CLOSING_LINE_SEARCH_ALPHAS[:-1]]
            if enabled
            else []
        ),
        "bidirectional_sampled_penetration": True,
        "float32_quantized_and_rechecked": True,
        "raw_penetration_cap_m": raw_cap,
        "target_penetration_cap_m": target_cap,
        "selected_alpha": selected_alpha,
        "fell_back_to_raw": selected_alpha == 0.0,
        "raw_above_raw_cap": (
            raw_maximum is not None and raw_maximum >= raw_cap
        ),
        "raw_state_finite": raw_state_finite,
        "raw_actuator_limits_passed": raw_limits_passed,
        "raw_penetration_passed": raw_penetration_passed,
        "raw_static_gate_passed": raw_gate,
        "selected_state_finite": selected_state_finite,
        "selected_actuator_limits_passed": selected_limits_passed,
        "selected_penetration_passed": selected_penetration_passed,
        "selected_target_safe": selected_safe,
        "raw_maximum_penetration_m": raw_maximum,
        "selected_maximum_penetration_m": selected_maximum,
    }


def _formal_v6_runtime() -> dict:
    return {
        "device": "cuda:0",
        "simulation_steps": 100,
        "substeps": 2,
        "physics_step_count": 200,
        "actuator_drive": {
            "stiffness_n_m_per_rad": FORMAL_ACTUATOR_STIFFNESS,
            "damping_n_m_s_per_rad": FORMAL_ACTUATOR_DAMPING,
            "armature_kg_m2": FORMAL_ACTUATOR_ARMATURE,
            "stiffness_source": "cli_override",
            "damping_source": "cli_override",
            "armature_source": "cli_override",
        },
        "physx_solver": {
            "solver_type": 1,
            "external_forces_every_iteration": True,
            "solve_articulation_contact_last": False,
        },
        "contact_gradient_closing": {
            "enabled": True,
            "mode": "collision_aware_line_search",
            "raw_penetration_cap_m": 0.001,
            "target_penetration_cap_m": 0.0015,
            "positive_alphas": [
                float(value) for value in CLOSING_LINE_SEARCH_ALPHAS[:-1]
            ],
            "bidirectional_sampled_penetration": True,
            "float32_quantized_and_rechecked": True,
            "raw_json_remains_physx_initial_state": True,
        },
        "zero_gravity_preclose": {
            "enabled": False,
            "physics_step_count": 0,
            "duration_s": 0.0,
            "target_schedule": "linear_raw_to_final_actuator_target",
            "validation_physics_step_count_unchanged": 200,
            "displacement_reference": "raw_json_object_position_before_preclose",
        },
    }


def _add_v5_dense_gate(payload: dict, *, maximum: float = 0.0005) -> None:
    reverse = maximum * 0.5
    feasible = maximum < 0.001
    payload["pipeline_revision"] = "x2_mesh_grasp_unselected_finger_side_v5"
    payload["maximum_penetration"] = maximum
    payload["self_collision"] = {
        "maximum_penetration": 0.0001,
        "total_penetration": 0.0002,
        "worst_pair": None,
        "feasible": True,
        "threshold": 0.0005,
    }
    payload["hand_object_penetration"] = {
        "evaluation_mode": "dense_bidirectional",
        "evaluated": True,
        "hand_surface_samples_per_set": 256,
        "hand_surface_samples_per_link": 768,
        "hand_surface_point_count": 13056,
        "object_surface_samples": 8192,
        "forward_total_penetration": maximum,
        "forward_maximum_penetration": maximum,
        "reverse_total_penetration": reverse,
        "reverse_maximum_penetration": reverse,
        "total_penetration": maximum + reverse,
        "maximum_penetration": maximum,
        "feasible": feasible,
        "threshold": 0.001,
    }
    payload["optimization"] = {
        "restored_checkpoint": (
            "bidirectional_feasible" if feasible else "bidirectional_fallback"
        ),
        "restored_step": 10,
        "bidirectional_feasible_checkpoint_found": feasible,
        "dense_hand_surface_samples_per_set": 256,
        "dense_hand_surface_samples_per_link": 768,
        "dense_hand_surface_point_count": 13056,
        "dense_object_surface_samples": 8192,
        "dense_bidirectional_query_calls": 1,
        "dense_bidirectional_rows_evaluated": 1,
    }


class X2IsaacValidationTests(unittest.TestCase):
    def _write_raw(self, root: Path, mesh: Path, *, side: str = "front") -> Path:
        raw = root / "sphere" / side / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        path = raw / f"sphere_r020_{side}_000000.json"
        path.write_text(json.dumps(_record(mesh, side=side), allow_nan=False), encoding="utf-8")
        return path

    def test_candidate_schema_and_object_centered_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.obj"
            mesh.write_text("o mesh\n", encoding="utf-8")
            raw = self._write_raw(root, mesh)
            candidate = load_raw_candidate(raw)

            self.assertEqual(candidate.active_side, "front")
            self.assertEqual(candidate.source_bytes, raw.read_bytes())
            self.assertEqual(
                candidate.source_sha256, hashlib.sha256(raw.read_bytes()).hexdigest()
            )
            self.assertEqual(tuple(candidate.actuator_by_name), EXPECTED_ACTUATOR_NAMES)
            self.assertEqual(tuple(candidate.joint_by_name), EXPECTED_JOINT_NAMES)
            self.assertTrue(np.array_equal(quaternion_wxyz_to_xyzw([1, 0, 0, 0]), [0, 0, 0, 1]))

            replay = make_object_centered_replay(candidate)
            self.assertTrue(np.allclose(replay.hand_translation, [1.0, 0.0, 0.0]))
            self.assertTrue(np.allclose(replay.hand_quaternion_xyzw, [0.0, 0.0, 0.0, 1.0]))
            self.assertTrue(np.allclose(replay.object_translation, [0.0, 0.0, 0.0]))
            self.assertTrue(np.allclose(replay.object_quaternion_xyzw, [0.0, 0.0, 0.0, 1.0]))
            expected_gravity = np.asarray(
                [
                    [0.0, -9.81, 0.0],
                    [0.0, 9.81, 0.0],
                    [-9.81, 0.0, 0.0],
                    [9.81, 0.0, 0.0],
                    [0.0, 0.0, 9.81],
                    [0.0, 0.0, -9.81],
                ]
            )
            self.assertTrue(np.allclose(replay.gravity_vectors, expected_gravity, atol=1.0e-12))

    def test_loader_rejects_nan_duplicate_contacts_and_mimic_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.obj"
            mesh.write_text("o mesh\n", encoding="utf-8")
            raw = self._write_raw(root, mesh)

            payload = json.loads(raw.read_text(encoding="utf-8"))
            payload["maximum_penetration"] = float("nan")
            raw.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(X2ValidationError, "non-finite JSON"):
                load_raw_candidate(raw)

            payload = _record(mesh)
            payload["selected_contact_ids"] = ["p0", "p0"]
            raw.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(X2ValidationError, "must be unique"):
                load_raw_candidate(raw)

            payload = _record(mesh)
            payload["joint"][EXPECTED_JOINT_NAMES.index("rh_LFJ1")] += 0.1
            raw.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(X2ValidationError, "passive mimic mismatch"):
                load_raw_candidate(raw)

    def test_loader_audits_finger_participation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.obj"
            mesh.write_text("o mesh\n", encoding="utf-8")
            payload = _record(mesh)
            payload["finger_participation"] = {
                "target_count": 1,
                "actual_count": 1,
                "finger_names": ["index"],
            }
            path = root / "front_single" / "raw" / "sample.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
            load_raw_candidate(path)

            payload["finger_participation"]["target_count"] = 2
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                X2ValidationError, "counts do not match selected contacts"
            ):
                load_raw_candidate(path)

    def test_v4_requires_consistent_self_collision_and_gates_only_overall_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.obj"
            mesh.write_text("o mesh\n", encoding="utf-8")
            raw = self._write_raw(root, mesh)
            payload = json.loads(raw.read_text(encoding="utf-8"))
            payload["pipeline_revision"] = "x2_mesh_grasp_unselected_finger_side_v4"
            raw.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            with self.assertRaisesRegex(X2ValidationError, r"v4\+ self_collision diagnostics"):
                load_raw_candidate(raw)

            payload["self_collision"] = {
                "maximum_penetration": 0.0006,
                "total_penetration": 0.0012,
                "worst_pair": ["rh_thdistal", "rh_ffmiddle"],
                "feasible": False,
                "threshold": 0.0005,
            }
            raw.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            candidate = load_raw_candidate(raw)
            self.assertTrue(candidate.self_collision_gate_required)
            self.assertFalse(candidate.self_collision_feasible)
            self.assertFalse(candidate.hand_object_gate_required)
            self.assertIsNone(candidate.hand_object_feasible)
            outcomes = [
                OrientationOutcome(name, True, 0.01, 0.02, 0.1, 0.01, True)
                for name, _ in GRAVITY_TESTS_WXYZ
            ]
            result = make_validated_record(
                candidate,
                outcomes,
                ValidationThresholds(),
                collision_aware_closing=_closing_audit(),
            )
            self.assertTrue(result["simulation_success"])
            self.assertFalse(result["success"])
            self.assertIn(
                "self_collision_not_feasible",
                result["validation"]["failure_reasons"],
            )
            self.assertFalse(result["validation"]["preflight"]["self_collision_passed"])

            payload["self_collision"]["feasible"] = True
            raw.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            with self.assertRaisesRegex(X2ValidationError, "feasible disagrees"):
                load_raw_candidate(raw)

            payload["self_collision"]["maximum_penetration"] = 0.0005
            raw.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            feasible_candidate = load_raw_candidate(raw)
            feasible_result = make_validated_record(
                feasible_candidate,
                outcomes,
                ValidationThresholds(),
                collision_aware_closing=_closing_audit(),
            )
            self.assertTrue(feasible_result["simulation_success"])
            self.assertTrue(feasible_result["success"])

            payload.pop("self_collision")
            payload["pipeline_revision"] = "x2_mesh_grasp_unselected_finger_side_v5"
            raw.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            with self.assertRaisesRegex(X2ValidationError, r"v4\+ self_collision diagnostics"):
                load_raw_candidate(raw)

    def test_v5_dense_hand_object_gate_accepts_feasible_and_routes_boundary_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.obj"
            mesh.write_text("o mesh\n", encoding="utf-8")
            raw = self._write_raw(root, mesh)
            payload = json.loads(raw.read_text(encoding="utf-8"))
            _add_v5_dense_gate(payload, maximum=0.0008)
            raw.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            candidate = load_raw_candidate(raw)
            self.assertTrue(candidate.hand_object_gate_required)
            self.assertTrue(candidate.hand_object_feasible)
            outcomes = [
                OrientationOutcome(name, True, 0.01, 0.02, 0.1, 0.01, True)
                for name, _ in GRAVITY_TESTS_WXYZ
            ]
            passed = make_validated_record(
                candidate,
                outcomes,
                ValidationThresholds(),
                collision_aware_closing=_closing_audit(raw_maximum=0.0008),
            )
            self.assertTrue(passed["success"])
            self.assertTrue(
                passed["validation"]["preflight"]["hand_object_passed"]
            )

            _add_v5_dense_gate(payload, maximum=0.001)
            raw.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            boundary = load_raw_candidate(raw)
            self.assertTrue(boundary.hand_object_gate_required)
            self.assertFalse(boundary.hand_object_feasible)
            failed = make_validated_record(
                boundary,
                outcomes,
                ValidationThresholds(),
                collision_aware_closing=_closing_audit(
                    raw_maximum=0.001,
                    selected_maximum=0.001,
                ),
            )
            self.assertTrue(failed["simulation_success"])
            self.assertFalse(failed["success"])
            self.assertIn(
                "dense_hand_object_penetration_not_feasible",
                failed["validation"]["failure_reasons"],
            )
            self.assertFalse(
                failed["validation"]["preflight"]["hand_object_passed"]
            )

    def test_v5_dense_hand_object_gate_rejects_inconsistent_structure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.obj"
            mesh.write_text("o mesh\n", encoding="utf-8")
            raw = self._write_raw(root, mesh)
            baseline = json.loads(raw.read_text(encoding="utf-8"))
            _add_v5_dense_gate(baseline, maximum=0.0008)

            mutations = {
                "evaluated must be true": lambda payload: payload[
                    "hand_object_penetration"
                ].__setitem__("evaluated", False),
                "samples_per_set": lambda payload: payload[
                    "hand_object_penetration"
                ].__setitem__("hand_surface_samples_per_set", 255),
                "total_penetration is inconsistent": lambda payload: payload[
                    "hand_object_penetration"
                ].__setitem__("total_penetration", 0.5),
                "top-level maximum_penetration disagrees": lambda payload: payload.__setitem__(
                    "maximum_penetration", 0.0007
                ),
                "must restore bidirectional_feasible": lambda payload: payload[
                    "optimization"
                ].__setitem__("restored_checkpoint", "fallback"),
                "query_calls must be a positive integer": lambda payload: payload[
                    "optimization"
                ].__setitem__("dense_bidirectional_query_calls", 0),
            }
            for message, mutate in mutations.items():
                with self.subTest(message=message):
                    payload = json.loads(json.dumps(baseline))
                    mutate(payload)
                    raw.write_text(
                        json.dumps(payload, allow_nan=False), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(X2ValidationError, message):
                        load_raw_candidate(raw)

    def test_v3_without_self_collision_retains_legacy_validation_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.obj"
            mesh.write_text("o mesh\n", encoding="utf-8")
            raw = self._write_raw(root, mesh)
            payload = json.loads(raw.read_text(encoding="utf-8"))
            payload["pipeline_revision"] = "x2_mesh_grasp_unselected_finger_side_v3"
            raw.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            candidate = load_raw_candidate(raw)
            self.assertFalse(candidate.self_collision_gate_required)
            self.assertIsNone(candidate.self_collision_feasible)
            outcomes = [
                OrientationOutcome(name, True, 0.01, 0.02, 0.1, 0.01, True)
                for name, _ in GRAVITY_TESTS_WXYZ
            ]
            result = make_validated_record(
                candidate,
                outcomes,
                ValidationThresholds(),
                collision_aware_closing=_closing_audit(),
            )
            self.assertTrue(result["simulation_success"])
            self.assertTrue(result["success"])
            self.assertTrue(result["validation"]["preflight"]["self_collision_passed"])

    def test_loader_strictly_validates_contact_records_and_side_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.obj"
            mesh.write_text("o mesh\n", encoding="utf-8")
            raw = self._write_raw(root, mesh)

            mutations = {
                "count must match": lambda payload: payload["selected_contacts"].pop(),
                "point_id must match": lambda payload: payload["selected_contacts"][0].__setitem__(
                    "point_id", "wrong"
                ),
                "source must be a non-empty string": lambda payload: payload[
                    "selected_contacts"
                ][0].__setitem__("source", ""),
                "enabled must be true": lambda payload: payload["selected_contacts"][0].__setitem__(
                    "enabled", False
                ),
                "local_position must have shape": lambda payload: payload[
                    "selected_contacts"
                ][0].__setitem__("local_position", [0.0, 0.0]),
                "not near-unit": lambda payload: payload["selected_contacts"][0].__setitem__(
                    "world_surface_normal", [2.0, 0.0, 0.0]
                ),
                "unique front/back values": lambda payload: payload[
                    "selected_contacts"
                ][0].__setitem__("supported_sides", ["front", "other"]),
                "does not support active_side": lambda payload: payload[
                    "selected_contacts"
                ][0].__setitem__("supported_sides", ["back"]),
            }
            for expected_error, mutate in mutations.items():
                with self.subTest(expected_error=expected_error):
                    payload = _record(mesh)
                    mutate(payload)
                    raw.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
                    with self.assertRaisesRegex(X2ValidationError, expected_error):
                        load_raw_candidate(raw)

            payload = _record(mesh, side="back")
            raw.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            with self.assertRaisesRegex(X2ValidationError, "does not match active_side=back"):
                load_raw_candidate(raw)

    def test_object_centered_replay_keeps_json_hand_pose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.obj"
            mesh.write_text("o mesh\n", encoding="utf-8")
            raw = self._write_raw(root, mesh)
            payload = json.loads(raw.read_text(encoding="utf-8"))
            half = float(np.sqrt(0.5))
            payload["hand_pose"]["quaternion_wxyz"] = [half, 0.0, 0.0, half]
            payload["hand_pose"]["rotation_matrix"] = [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
            raw.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            replay = make_object_centered_replay(load_raw_candidate(raw))
            self.assertTrue(np.allclose(replay.hand_translation, [1.0, 0.0, 0.0], atol=1.0e-12))
            self.assertTrue(
                np.allclose(replay.hand_quaternion_xyzw, [0.0, 0.0, half, half], atol=1.0e-12)
            )
            self.assertTrue(np.allclose(replay.object_translation, 0.0, atol=1.0e-12))
            self.assertTrue(np.allclose(replay.gravity_vectors[0], [0.0, -9.81, 0.0], atol=1.0e-12))

    def test_object_centered_replay_matches_noncommuting_full_pose_transform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.obj"
            mesh.write_text("o mesh\n", encoding="utf-8")
            raw = self._write_raw(root, mesh)
            payload = json.loads(raw.read_text(encoding="utf-8"))
            quaternion = np.asarray([0.71, -0.21, 0.42, 0.52], dtype=np.float64)
            quaternion /= np.linalg.norm(quaternion)
            rotation = quaternion_matrix_wxyz(quaternion)
            translation = np.asarray([0.13, -0.27, 0.08], dtype=np.float64)
            payload["hand_pose"]["quaternion_wxyz"] = quaternion.tolist()
            payload["hand_pose"]["rotation_matrix"] = rotation.tolist()
            payload["hand_pose"]["translation"] = translation.tolist()
            raw.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

            candidate = load_raw_candidate(raw)
            replay = make_object_centered_replay(candidate)
            gravity_world = np.asarray([0.0, -9.81, 0.0])
            for direction_index, (_, test_quaternion) in enumerate(GRAVITY_TESTS_WXYZ):
                test_rotation = quaternion_matrix_wxyz(test_quaternion)
                hand_world = test_rotation @ rotation
                hand_translation_world = test_rotation @ translation
                relative_rotation = test_rotation.T @ hand_world
                relative_translation = test_rotation.T @ hand_translation_world
                self.assertTrue(np.allclose(relative_rotation, rotation, atol=1.0e-12))
                self.assertTrue(
                    np.allclose(relative_translation, replay.hand_translation, atol=1.0e-12)
                )
                expected_gravity = test_rotation.T @ gravity_world
                self.assertTrue(
                    np.allclose(
                        replay.gravity_vectors[direction_index], expected_gravity, atol=1.0e-12
                    )
                )

    def test_strict_hold_predicate_and_penetration_gate(self) -> None:
        thresholds = ValidationThresholds()
        passed = evaluate_orientation(
            name="identity",
            final_displacement=0.01,
            maximum_displacement=0.02,
            final_contact_force=0.1,
            maximum_active_joint_error=0.01,
            finite=True,
            thresholds=thresholds,
        )
        self.assertTrue(passed.passed)
        boundary = evaluate_orientation(
            name="identity",
            final_displacement=0.01,
            maximum_displacement=thresholds.retention_distance,
            final_contact_force=0.1,
            maximum_active_joint_error=0.01,
            finite=True,
            thresholds=thresholds,
            criterion="strict-hold",
        )
        self.assertFalse(boundary.passed)
        source_compatible = evaluate_orientation(
            name="identity",
            final_displacement=10.0,
            maximum_displacement=10.0,
            final_contact_force=0.1,
            maximum_active_joint_error=10.0,
            finite=True,
            thresholds=thresholds,
        )
        self.assertTrue(source_compatible.passed)
        explicit_contact = evaluate_orientation(
            name="identity",
            final_displacement=0.01,
            maximum_displacement=0.02,
            final_contact_force=0.0,
            maximum_active_joint_error=0.01,
            finite=True,
            thresholds=thresholds,
            hand_object_contact=True,
        )
        self.assertTrue(explicit_contact.passed)
        self.assertTrue(explicit_contact.as_dict()["hand_object_contact"])
        explicit_no_contact = evaluate_orientation(
            name="identity",
            final_displacement=0.01,
            maximum_displacement=0.02,
            final_contact_force=100.0,
            maximum_active_joint_error=0.01,
            finite=True,
            thresholds=thresholds,
            hand_object_contact=False,
        )
        self.assertFalse(explicit_no_contact.passed)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.obj"
            mesh.write_text("o mesh\n", encoding="utf-8")
            raw = self._write_raw(root, mesh)
            candidate = load_raw_candidate(raw)
            outcomes = [
                OrientationOutcome(
                    name=name,
                    passed=True,
                    final_displacement=0.01,
                    maximum_displacement=0.02,
                    final_contact_force=0.1,
                    maximum_active_joint_error=0.01,
                    finite=True,
                )
                for name, _ in GRAVITY_TESTS_WXYZ
            ]
            result = make_validated_record(
                candidate,
                outcomes,
                thresholds,
                collision_aware_closing=_closing_audit(),
            )
            self.assertEqual(
                result["validation"]["protocol_revision"],
                "x2_object_centered_dexgraspnet_six_orientation_v7",
            )
            self.assertEqual(result["validation"]["protocol_revision"], PROTOCOL_REVISION)
            self.assertTrue(result["success"])
            self.assertTrue(result["simulation_success"])
            self.assertEqual(result["validation"]["status"], "passed")
            self.assertEqual(result["validation"]["passed_orientation_count"], 6)

            candidate.record["maximum_penetration"] = thresholds.penetration
            object.__setattr__(candidate, "maximum_penetration", thresholds.penetration)
            result = make_validated_record(
                candidate,
                outcomes,
                thresholds,
                collision_aware_closing=_closing_audit(),
            )
            self.assertFalse(result["success"])
            self.assertTrue(result["simulation_success"])
            self.assertIn(
                "maximum_penetration_not_below_threshold",
                result["validation"]["failure_reasons"],
            )

    def test_nonfinite_simulation_is_json_safe_and_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.obj"
            mesh.write_text("o mesh\n", encoding="utf-8")
            candidate = load_raw_candidate(self._write_raw(root, mesh))
            thresholds = ValidationThresholds()
            outcomes = [
                evaluate_orientation(
                    name=name,
                    final_displacement=0.01,
                    maximum_displacement=0.02,
                    final_contact_force=0.1,
                    maximum_active_joint_error=0.01,
                    finite=True,
                    thresholds=thresholds,
                    hand_object_contact=True,
                )
                for name, _ in GRAVITY_TESTS_WXYZ
            ]
            outcomes[0] = evaluate_orientation(
                name="identity",
                final_displacement=float("nan"),
                maximum_displacement=float("inf"),
                final_contact_force=float("nan"),
                maximum_active_joint_error=float("inf"),
                finite=False,
                thresholds=thresholds,
                hand_object_contact=True,
            )
            result = make_validated_record(
                candidate,
                outcomes,
                thresholds,
                collision_aware_closing=_closing_audit(),
            )
            self.assertFalse(result["success"])
            self.assertFalse(result["simulation_success"])
            identity = result["validation"]["orientations"][0]
            self.assertIsNone(identity["final_displacement_m"])
            self.assertIsNone(identity["maximum_displacement_m"])
            self.assertIsNone(identity["final_contact_force_n"])
            self.assertIsNone(identity["maximum_active_joint_error_rad"])
            self.assertFalse(identity["finite"])
            self.assertIn(
                "nonfinite_simulation:identity",
                result["validation"]["failure_reasons"],
            )
            json.dumps(result, allow_nan=False)

    def test_newton_mimic_violation_fails_only_affected_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.obj"
            mesh.write_text("o mesh\n", encoding="utf-8")
            candidate = load_raw_candidate(self._write_raw(root, mesh))
            thresholds = ValidationThresholds(mimic_error=0.01)
            outcomes = [
                evaluate_orientation(
                    name=name,
                    final_displacement=0.01,
                    maximum_displacement=0.02,
                    final_contact_force=0.1,
                    maximum_active_joint_error=0.01,
                    maximum_newton_mimic_error=(0.012 if name == "identity" else 0.001),
                    finite=True,
                    thresholds=thresholds,
                    hand_object_contact=True,
                )
                for name, _ in GRAVITY_TESTS_WXYZ
            ]
            self.assertFalse(outcomes[0].passed)
            self.assertTrue(all(outcome.passed for outcome in outcomes[1:]))
            record = make_validated_record(
                candidate,
                outcomes,
                thresholds,
                collision_aware_closing=_closing_audit(),
            )
            self.assertFalse(record["simulation_success"])
            self.assertEqual(record["validation"]["passed_orientation_count"], 5)
            self.assertIn(
                "newton_mimic_tracking:identity",
                record["validation"]["failure_reasons"],
            )
            self.assertEqual(
                record["validation"]["orientations"][0][
                    "maximum_newton_mimic_error_rad"
                ],
                0.012,
            )
            json.dumps(record, allow_nan=False)

    def test_mimic_audit_separates_nonfinite_from_tracking_violations(self) -> None:
        audit = summarize_newton_mimic_errors(
            [float("inf")] * len(GRAVITY_TESTS_WXYZ)
            + [float("nan")] * len(GRAVITY_TESTS_WXYZ),
            threshold=0.01,
        )
        self.assertIsNone(audit["maximum_newton_mimic_error_rad"])
        self.assertEqual(audit["newton_mimic_finite_orientation_count"], 0)
        self.assertEqual(audit["newton_mimic_nonfinite_orientation_count"], 12)
        self.assertEqual(audit["newton_mimic_nonfinite_sample_count"], 2)
        self.assertEqual(audit["newton_mimic_violation_orientation_count"], 0)
        self.assertEqual(audit["newton_mimic_violation_sample_count"], 0)
        json.dumps(audit, allow_nan=False)

        audit = summarize_newton_mimic_errors(
            [0.001, 0.012, float("inf"), 0.004, 0.005, 0.006],
            threshold=0.01,
        )
        self.assertEqual(audit["maximum_newton_mimic_error_rad"], 0.012)
        self.assertEqual(audit["newton_mimic_finite_orientation_count"], 5)
        self.assertEqual(audit["newton_mimic_nonfinite_orientation_count"], 1)
        self.assertEqual(audit["newton_mimic_nonfinite_sample_count"], 1)
        self.assertEqual(audit["newton_mimic_violation_orientation_count"], 1)
        self.assertEqual(audit["newton_mimic_violation_sample_count"], 1)

    def test_collision_aware_closing_selects_each_row_independently(self) -> None:
        alphas = CLOSING_LINE_SEARCH_ALPHAS
        penetration = np.full((2, len(alphas)), 0.0002, dtype=np.float64)
        penetration[1, :3] = 0.002
        penetration[1, 3] = 0.0009
        finite = np.ones_like(penetration, dtype=bool)
        limits = np.ones_like(penetration, dtype=bool)
        selection = select_collision_aware_closing(
            penetration,
            finite,
            limits,
            raw_penetration_cap=0.001,
            target_penetration_cap=0.0015,
        )
        self.assertEqual(selection.selected_alphas.tolist(), [1.0, 0.125])
        self.assertEqual(selection.selected_indices.tolist(), [0, 3])
        self.assertTrue(selection.raw_static_gate_passed.all())

    def test_collision_aware_closing_strict_cap_and_failure_fallbacks(self) -> None:
        alphas = CLOSING_LINE_SEARCH_ALPHAS
        penetration = np.full((5, len(alphas)), 0.002, dtype=np.float64)
        penetration[:, -1] = 0.0005
        penetration[0, 0] = 0.001228  # Known-safe sphere target is admitted.
        penetration[1, :-1] = np.nan
        penetration[2, 0] = 0.002198  # Failed-sphere target is rejected.
        penetration[2, 1] = 0.0014
        penetration[3, :-1] = 0.0005
        penetration[4, -1] = 0.001  # Raw equality is strictly rejected.
        finite = np.isfinite(penetration)
        limits = np.ones_like(finite)
        limits[3, :-1] = False
        selection = select_collision_aware_closing(
            penetration,
            finite,
            limits,
            raw_penetration_cap=0.001,
            target_penetration_cap=0.0015,
        )
        self.assertEqual(
            selection.selected_alphas.tolist(), [1.0, 0.0, 0.5, 0.0, 0.0]
        )
        self.assertEqual(
            selection.raw_static_gate_passed.tolist(),
            [True, True, True, True, False],
        )

    def test_v6_sampled_raw_gate_strictly_controls_final_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.obj"
            mesh.write_text("o mesh\n", encoding="utf-8")
            candidate = load_raw_candidate(self._write_raw(root, mesh))
            thresholds = ValidationThresholds()
            outcomes = [
                OrientationOutcome(name, True, 0.01, 0.02, 0.1, 0.01, True)
                for name, _ in GRAVITY_TESTS_WXYZ
            ]
            sampled_failure = _closing_audit(
                raw_maximum=0.001,
                selected_maximum=0.001,
            )
            result = make_validated_record(
                candidate,
                outcomes,
                thresholds,
                collision_aware_closing=sampled_failure,
            )
            self.assertTrue(result["simulation_success"])
            self.assertFalse(result["success"])
            preflight = result["validation"]["preflight"]
            self.assertFalse(preflight["collision_aware_closing_raw_passed"])
            self.assertFalse(preflight["penetration_passed"])
            self.assertIn(
                "collision_aware_closing_raw_penetration_not_below_threshold",
                result["validation"]["failure_reasons"],
            )
            json.dumps(result, allow_nan=False)

            with self.assertRaisesRegex(X2ValidationError, "safe selected target"):
                make_validated_record(
                    candidate,
                    outcomes,
                    thresholds,
                    collision_aware_closing=_closing_audit(
                        raw_maximum=0.0005,
                        selected_maximum=0.002,
                    ),
                )

            no_force = make_validated_record(
                candidate,
                outcomes,
                thresholds,
                collision_aware_closing=_closing_audit(enabled=False),
            )
            self.assertTrue(no_force["success"])
            self.assertEqual(
                no_force["validation"]["preflight"]["collision_aware_closing"][
                    "selected_alpha"
                ],
                0.0,
            )

    def test_compact_closing_batch_audit_does_not_repeat_samples(self) -> None:
        first = _closing_audit(raw_maximum=0.0002, selected_maximum=0.0012)
        first["full_target_maximum_penetration_m"] = 0.0012
        second = _closing_audit(raw_maximum=0.001, selected_maximum=0.001)
        second["full_target_maximum_penetration_m"] = 0.0022
        audit = {
            "enabled": True,
            "mode": "collision_aware_line_search",
            "raw_penetration_cap_m": 0.001,
            "target_penetration_cap_m": 0.0015,
            "samples": [first, second],
        }
        compact = compact_collision_aware_closing_batch_audit(audit)
        self.assertNotIn("samples", compact)
        self.assertIn("samples", audit)
        aggregate = compact["batch_aggregate"]
        self.assertEqual(aggregate["sample_count"], 2)
        self.assertEqual(aggregate["raw_static_gate_failed_count"], 1)
        self.assertEqual(aggregate["selected_target_unsafe_count"], 1)
        self.assertEqual(aggregate["selected_alpha_counts"], {"1": 1, "0": 1})
        self.assertEqual(aggregate["maximum_raw_penetration_m"], 0.001)
        self.assertEqual(aggregate["maximum_selected_penetration_m"], 0.0012)
        json.dumps(compact, allow_nan=False)

    def test_atomic_routing_preserves_raw_and_removes_stale_opposite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.obj"
            mesh.write_text("o mesh\n", encoding="utf-8")
            raw = self._write_raw(root, mesh)
            raw_bytes = raw.read_bytes()
            candidate = load_raw_candidate(raw)
            thresholds = ValidationThresholds()
            outcomes = [
                OrientationOutcome(name, True, 0.01, 0.02, 0.1, 0.01, True)
                for name, _ in GRAVITY_TESTS_WXYZ
            ]
            passed_record = make_validated_record(
                candidate,
                outcomes,
                thresholds,
                collision_aware_closing=_closing_audit(),
            )
            valid_path = write_validated_record(candidate, passed_record)
            self.assertEqual(valid_path, validation_output_path(raw, True))
            self.assertTrue(valid_path.is_file())
            self.assertEqual(raw.read_bytes(), raw_bytes)

            failed_outcomes = list(outcomes)
            failed_outcomes[0] = OrientationOutcome("identity", False, 1.0, 1.0, 0.0, 0.01, True)
            failed_record = make_validated_record(
                candidate,
                failed_outcomes,
                thresholds,
                collision_aware_closing=_closing_audit(),
            )
            failed_path = write_validated_record(candidate, failed_record, overwrite=True)
            self.assertTrue(failed_path.is_file())
            self.assertFalse(valid_path.exists())
            self.assertEqual(raw.read_bytes(), raw_bytes)
            with self.assertRaisesRegex(X2ValidationError, "already exists"):
                write_validated_record(candidate, failed_record)

    def test_publish_rejects_raw_changed_after_load_and_uses_loaded_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.obj"
            mesh.write_text("o mesh\n", encoding="utf-8")
            raw = self._write_raw(root, mesh)
            candidate = load_raw_candidate(raw)
            outcomes = [
                OrientationOutcome(name, True, 0.01, 0.02, 0.1, 0.01, True)
                for name, _ in GRAVITY_TESTS_WXYZ
            ]
            record = make_validated_record(
                candidate,
                outcomes,
                ValidationThresholds(),
                collision_aware_closing=_closing_audit(),
            )
            self.assertEqual(record["validation"]["source_sha256"], candidate.source_sha256)

            raw.write_bytes(candidate.source_bytes + b"\n")
            with self.assertRaisesRegex(X2ValidationError, "changed since load"):
                write_validated_record(candidate, record)

    def test_discovery_filters_side_mesh_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_a = root / "a.obj"
            mesh_b = root / "b.obj"
            mesh_a.write_text("o a\n", encoding="utf-8")
            mesh_b.write_text("o b\n", encoding="utf-8")
            self._write_raw(root / "one", mesh_a, side="front")
            self._write_raw(root / "two", mesh_b, side="back")

            self.assertEqual(len(discover_raw_candidates(root, side="front")), 1)
            selected = discover_raw_candidates(root, mesh_path=mesh_b, side="both", limit=1)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].active_side, "back")

    def test_primitive_wrapper_uses_x2_validator_contract(self) -> None:
        spec = selected_specs(("sphere",))[0]
        args = SimpleNamespace(
            side="both",
            batch_size=32,
            sim_steps=100,
            substeps=2,
            preclose_physics_steps=0,
            closing_penetration_cap=0.0015,
            criterion="dexgraspnet-contact",
            device="cuda:0",
            limit_per_object=None,
            overwrite=False,
            resume=True,
            dry_run=False,
        )
        command = build_validator_command(
            spec=spec,
            mesh_root=Path("/meshes"),
            input_root=Path("/grasps"),
            summary_path=Path("/summary.json"),
            args=args,
        )
        self.assertTrue(command[1].endswith("scripts/validate_x2_mesh_grasps_physx.py"))
        self.assertEqual(command[command.index("--viz") + 1], "none")
        self.assertIn("--resume", command)
        self.assertIn("convex-hull", command)
        self.assertEqual(command[command.index("--batch-size") + 1], "32")
        self.assertEqual(command[command.index("--sim-steps") + 1], "100")
        self.assertEqual(command[command.index("--substeps") + 1], "2")
        self.assertEqual(
            command[command.index("--preclose-physics-steps") + 1], "0"
        )
        self.assertEqual(
            command[command.index("--closing-penetration-cap") + 1], "0.0015"
        )
        self.assertEqual(
            command[command.index("--actuator-stiffness") + 1], "1000.0"
        )
        self.assertEqual(
            command[command.index("--actuator-damping") + 1], "0.632455532"
        )
        self.assertEqual(
            command[command.index("--actuator-armature") + 1], "0.0001"
        )
        self.assertIn("--external-forces-every-iteration", command)
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")

    def test_general_mesh_wrapper_uses_convex_decomposition(self) -> None:
        spec = GeneralMeshSpec("object-a", Path("/objects/object-a/coacd/decomposed.obj"))
        args = SimpleNamespace(
            side="both",
            batch_size=32,
            sim_steps=100,
            substeps=2,
            preclose_physics_steps=0,
            closing_penetration_cap=0.0015,
            criterion="dexgraspnet-contact",
            device="cuda:0",
            limit_per_object=None,
            overwrite=True,
            resume=False,
            dry_run=False,
        )
        command = build_validator_command(
            spec=spec,
            mesh_root=Path("/meshes"),
            input_root=Path("/grasps"),
            summary_path=Path("/summary.json"),
            args=args,
        )
        self.assertEqual(
            command[command.index("--mesh-path") + 1], str(spec.path.resolve())
        )
        self.assertEqual(
            command[command.index("--collision-approximation") + 1],
            "convex-decomposition",
        )

    def test_primitive_wrapper_defaults_summaries_to_custom_input_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_root = Path(directory) / "custom_grasps"
            args = primitive_validator._parse_args(["--input-root", str(input_root)])
            summary_dir, summary_csv = primitive_validator._resolve_summary_paths(
                input_root.resolve(), args
            )
            self.assertEqual(summary_dir, input_root.resolve() / "validation_summaries")
            self.assertEqual(summary_csv, input_root.resolve() / "validation_summary.csv")

            explicit_dir = Path(directory) / "reports"
            explicit_csv = Path(directory) / "report.csv"
            args = primitive_validator._parse_args(
                [
                    "--input-root",
                    str(input_root),
                    "--summary-dir",
                    str(explicit_dir),
                    "--summary-csv",
                    str(explicit_csv),
                ]
            )
            summary_dir, summary_csv = primitive_validator._resolve_summary_paths(
                input_root.resolve(), args
            )
            self.assertEqual(summary_dir, explicit_dir.resolve())
            self.assertEqual(summary_csv, explicit_csv.resolve())

    def test_primitive_wrapper_resume_publishes_full_scanned_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "grasps"
            mesh_root = root / "meshes"
            summary_dir = root / "summaries"
            summary_csv = root / "summary.csv"
            spec = selected_specs(("sphere",))[0]
            mesh = mesh_root / spec.relative_path
            mesh.parent.mkdir(parents=True, exist_ok=True)
            mesh.write_text("o mesh\n", encoding="utf-8")

            front_raw = self._write_raw(input_root, mesh, side="front")
            back_raw = self._write_raw(input_root, mesh, side="back")
            thresholds = ValidationThresholds()
            passed_outcomes = [
                OrientationOutcome(name, True, 0.01, 0.02, 0.1, 0.01, True)
                for name, _ in GRAVITY_TESTS_WXYZ
            ]
            front_candidate = load_raw_candidate(front_raw)
            write_validated_record(
                front_candidate,
                make_validated_record(
                    front_candidate,
                    passed_outcomes,
                    thresholds,
                    collision_aware_closing=_closing_audit(),
                    runtime=_formal_v6_runtime(),
                ),
            )
            failed_outcomes = list(passed_outcomes)
            failed_outcomes[0] = OrientationOutcome(
                "identity", False, 0.2, 0.2, 0.0, 0.01, True
            )
            back_candidate = load_raw_candidate(back_raw)
            write_validated_record(
                back_candidate,
                make_validated_record(
                    back_candidate,
                    failed_outcomes,
                    thresholds,
                    collision_aware_closing=_closing_audit(),
                    runtime=_formal_v6_runtime(),
                ),
            )

            temporary_summaries: list[Path] = []

            def fake_run(command, **kwargs):
                del kwargs
                temporary = Path(command[command.index("--summary-json") + 1])
                temporary_summaries.append(temporary)
                self.assertNotEqual(temporary, summary_dir / f"{spec.instance_name}.json")
                temporary.write_text(
                    json.dumps(
                        {
                            "passed": True,
                            "candidate_count": 0,
                            "skipped_existing_count": 2,
                            "mesh_path": str(mesh.resolve()),
                            "object_scale": 1.0,
                        },
                        allow_nan=False,
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="")

            argv = [
                "--input-root",
                str(input_root),
                "--mesh-root",
                str(mesh_root),
                "--shapes",
                "sphere",
                "--side",
                "both",
                "--resume",
                "--summary-dir",
                str(summary_dir),
                "--summary-csv",
                str(summary_csv),
            ]
            with (
                patch.object(primitive_validator, "build_dataset"),
                patch.object(primitive_validator, "selected_specs", return_value=[spec]),
                patch.object(primitive_validator.subprocess, "run", side_effect=fake_run),
            ):
                self.assertEqual(primitive_validator.main(argv), 0)

            self.assertEqual(len(temporary_summaries), 1)
            self.assertFalse(temporary_summaries[0].exists())
            published = json.loads(
                (summary_dir / f"{spec.instance_name}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(published["processed_candidate_count"], 0)
            self.assertEqual(published["candidate_count"], 2)
            self.assertEqual(published["skipped_existing_count"], 2)
            self.assertEqual(published["valid_count"], 1)
            self.assertEqual(published["failed_count"], 1)
            self.assertEqual(published["pending_count"], 0)
            self.assertEqual(published["side_summary"]["front"]["passed"], 1)
            self.assertEqual(published["side_summary"]["back"]["failed"], 1)

            with summary_csv.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["candidate_count"], "2")
            self.assertEqual(rows[0]["valid_count"], "1")
            self.assertEqual(rows[0]["failed_count"], "1")
            self.assertEqual(rows[0]["front_total"], "1")
            self.assertEqual(rows[0]["back_total"], "1")

    def test_primitive_wrapper_resume_revalidates_stale_v5_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "grasps"
            mesh_root = root / "meshes"
            summary_dir = root / "summaries"
            summary_csv = root / "summary.csv"
            spec = selected_specs(("sphere",))[0]
            mesh = mesh_root / spec.relative_path
            mesh.parent.mkdir(parents=True, exist_ok=True)
            mesh.write_text("o mesh\n", encoding="utf-8")

            raw = self._write_raw(input_root, mesh, side="front")
            candidate = load_raw_candidate(raw)
            thresholds = ValidationThresholds()
            outcomes = [
                OrientationOutcome(name, True, 0.01, 0.02, 0.1, 0.01, True)
                for name, _ in GRAVITY_TESTS_WXYZ
            ]
            valid_path = write_validated_record(
                candidate,
                make_validated_record(
                    candidate,
                    outcomes,
                    thresholds,
                    collision_aware_closing=_closing_audit(),
                    runtime=_formal_v6_runtime(),
                ),
            )
            stale = json.loads(valid_path.read_text(encoding="utf-8"))
            stale["validation"]["protocol_revision"] = (
                "x2_object_centered_dexgraspnet_six_orientation_v5"
            )
            stale["validation"]["runtime"]["zero_gravity_preclose"].update(
                {"enabled": True, "physics_step_count": 30}
            )
            stale["validation"]["runtime"]["contact_gradient_closing"][
                "target_penetration_cap_m"
            ] = 0.001
            stale["validation"]["preflight"]["collision_aware_closing"][
                "target_penetration_cap_m"
            ] = 0.001
            valid_path.write_text(
                json.dumps(stale, allow_nan=False), encoding="utf-8"
            )

            def fake_run(command, **kwargs):
                del kwargs
                self.assertFalse(valid_path.exists())
                self.assertIn("--resume", command)
                self.assertEqual(
                    command[command.index("--preclose-physics-steps") + 1], "0"
                )
                self.assertEqual(
                    command[command.index("--closing-penetration-cap") + 1],
                    "0.0015",
                )
                write_validated_record(
                    candidate,
                    make_validated_record(
                        candidate,
                        outcomes,
                        thresholds,
                        collision_aware_closing=_closing_audit(),
                        runtime=_formal_v6_runtime(),
                    ),
                )
                temporary = Path(command[command.index("--summary-json") + 1])
                temporary.write_text(
                    json.dumps(
                        {
                            "passed": True,
                            "candidate_count": 1,
                            "skipped_existing_count": 0,
                            "valid_count": 1,
                            "failed_count": 0,
                            "mesh_path": str(mesh.resolve()),
                            "object_scale": 1.0,
                        },
                        allow_nan=False,
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="")

            argv = [
                "--input-root",
                str(input_root),
                "--mesh-root",
                str(mesh_root),
                "--shapes",
                "sphere",
                "--side",
                "front",
                "--resume",
                "--summary-dir",
                str(summary_dir),
                "--summary-csv",
                str(summary_csv),
            ]
            with (
                patch.object(primitive_validator, "build_dataset"),
                patch.object(primitive_validator, "selected_specs", return_value=[spec]),
                patch.object(primitive_validator.subprocess, "run", side_effect=fake_run),
            ):
                self.assertEqual(primitive_validator.main(argv), 0)

            routed = json.loads(valid_path.read_text(encoding="utf-8"))
            self.assertEqual(
                routed["validation"]["protocol_revision"], PROTOCOL_REVISION
            )
            self.assertFalse(
                routed["validation"]["runtime"]["zero_gravity_preclose"]["enabled"]
            )
            self.assertEqual(
                routed["validation"]["runtime"]["contact_gradient_closing"][
                    "target_penetration_cap_m"
                ],
                0.0015,
            )

    def test_primitive_wrapper_never_reuses_stale_published_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "grasps"
            mesh_root = root / "meshes"
            summary_dir = root / "summaries"
            summary_csv = root / "summary.csv"
            spec = selected_specs(("sphere",))[0]
            mesh = mesh_root / spec.relative_path
            mesh.parent.mkdir(parents=True, exist_ok=True)
            mesh.write_text("o mesh\n", encoding="utf-8")
            self._write_raw(input_root, mesh, side="front")
            summary_dir.mkdir(parents=True)
            published_path = summary_dir / f"{spec.instance_name}.json"
            stale_bytes = b'{"passed": true, "stale": true}\n'
            published_path.write_bytes(stale_bytes)

            def fake_run(command, **kwargs):
                del kwargs
                temporary = Path(command[command.index("--summary-json") + 1])
                self.assertNotEqual(temporary, published_path)
                # A zero exit without writing its unique summary must not fall back to the old file.
                return SimpleNamespace(returncode=0, stdout="")

            argv = [
                "--input-root",
                str(input_root),
                "--mesh-root",
                str(mesh_root),
                "--shapes",
                "sphere",
                "--side",
                "front",
                "--dry-run",
                "--summary-dir",
                str(summary_dir),
                "--summary-csv",
                str(summary_csv),
            ]
            with (
                patch.object(primitive_validator, "build_dataset"),
                patch.object(primitive_validator, "selected_specs", return_value=[spec]),
                patch.object(primitive_validator.subprocess, "run", side_effect=fake_run),
                self.assertRaisesRegex(
                    primitive_validator.PrimitiveValidationError,
                    "temporary validator summary",
                ),
            ):
                primitive_validator.main(argv)

            self.assertEqual(published_path.read_bytes(), stale_bytes)
            self.assertEqual(list(summary_dir.glob(f".{spec.instance_name}.*.summary.json")), [])

    def test_new_validator_does_not_import_official_shadowhand_validator(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "validate_x2_mesh_grasps_physx.py"
        ).read_text(encoding="utf-8")
        lowered = source.lower()
        self.assertNotIn("isaacgym", lowered)
        self.assertNotIn("utils.isaac_validator", lowered)
        self.assertNotIn("grasp_generation.scripts.validate_grasps", lowered)
        self.assertLess(source.index("AppLauncher(args)"), source.index("import isaaclab.sim as sim_utils", source.index("AppLauncher(args)")))
        self.assertIn("get_raw_contact_data", source)
        self.assertIn('filter_prim_paths_expr=[]', source)
        self.assertNotIn("contact_sensor.data.contact_pos_w", source)
        self.assertNotIn("contact_sensor.data.force_matrix_w", source)
        self.assertNotIn("contact_sensor.data.net_forces_w", source)
        self.assertIn("maximum_newton_mimic_error_rad", source)
        self.assertNotIn("PhysxMimicJointAPI.Apply", source)
        self.assertIn("robot.write_root_pose_to_sim_index(root_pose=hand_pose)", source)
        self.assertIn('"replay_frame": "object_centered_generator_frame"', source)
        self.assertNotIn("make_hand_fixed_replay", source)
        self.assertIn("gravity_vector_object_frame=", source)
        self.assertIn("hand.set_parameters(pose, contact_indices)", source)
        self.assertIn("range(args.sim_steps * args.substeps)", source)
        self.assertIn("range(args.preclose_physics_steps)", source)
        self.assertIn('default=0,\n        help=(\n            "Experimental zero-gravity setup', source)
        self.assertLess(
            source.index("forces=zero_forces"),
            source.index("for preclose_step in range(args.preclose_physics_steps)"),
        )
        self.assertIn("zero_forces = torch.zeros_like(forces)", source)
        self.assertIn("forces.shape[-1] != 3", source)
        self.assertIn("if math.isfinite(preclose_maximum_displacement)", source)
        self.assertIn(
            '"displacement_reference": "raw_json_object_position_before_preclose"',
            source,
        )
        self.assertIn("position=raw_joint_position", source)
        self.assertNotIn("position=initial_joint_position", source)
        self.assertIn('"initial_dof_state": "raw_json_joint_state"', source)
        self.assertIn('"actuator_target_state": (', source)
        self.assertIn('"body_mass_kg"', source)
        self.assertIn('"body_inertia_kg_m2_flat"', source)


if __name__ == "__main__":
    unittest.main()
