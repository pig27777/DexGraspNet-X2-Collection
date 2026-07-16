"""Regression tests for deterministic X2 primitive mesh/grasp datasets."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import trimesh

from scripts.build_x2_primitive_dataset import (
    PRIMITIVE_SPECS,
    SHAPES,
    audit_mesh,
    build_dataset,
)
from scripts.generate_x2_primitive_dataset import (
    BATCH_SIZE,
    CSV_FIELDS,
    DENSE_GATE_EVALUATION_MODE,
    DENSE_HAND_OBJECT_PENETRATION_THRESHOLD,
    DENSE_HAND_SURFACE_POINT_COUNT,
    DENSE_HAND_SURFACE_SAMPLES_PER_LINK,
    DENSE_HAND_SURFACE_SAMPLES_PER_SET,
    DENSE_OBJECT_SURFACE_SAMPLES,
    GENERATOR_PIPELINE_REVISION,
    N_CONTACT,
    OBJECT_SCALE,
    SURFACE_SAMPLES,
    PrimitiveGenerationError,
    discover_general_meshes,
    select_general_meshes,
    _parse_args,
    build_generator_command,
    plan_generation,
    run,
    summarize_records,
)


def _v5_dense_gate(maximum: float) -> dict[str, object]:
    reverse = maximum * 0.5
    return {
        "evaluation_mode": DENSE_GATE_EVALUATION_MODE,
        "evaluated": True,
        "hand_surface_samples_per_set": DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        "hand_surface_samples_per_link": DENSE_HAND_SURFACE_SAMPLES_PER_LINK,
        "hand_surface_point_count": DENSE_HAND_SURFACE_POINT_COUNT,
        "object_surface_samples": DENSE_OBJECT_SURFACE_SAMPLES,
        "forward_total_penetration": maximum,
        "forward_maximum_penetration": maximum,
        "reverse_total_penetration": reverse,
        "reverse_maximum_penetration": reverse,
        "total_penetration": maximum + reverse,
        "maximum_penetration": maximum,
        "feasible": maximum < DENSE_HAND_OBJECT_PENETRATION_THRESHOLD,
        "threshold": DENSE_HAND_OBJECT_PENETRATION_THRESHOLD,
    }


def _v5_dense_summary(sample_count: int, feasible_count: int) -> dict[str, object]:
    return {
        "evaluation_mode": DENSE_GATE_EVALUATION_MODE,
        "hand_surface_samples_per_set": DENSE_HAND_SURFACE_SAMPLES_PER_SET,
        "hand_surface_samples_per_link": DENSE_HAND_SURFACE_SAMPLES_PER_LINK,
        "hand_surface_point_count": DENSE_HAND_SURFACE_POINT_COUNT,
        "object_surface_samples": DENSE_OBJECT_SURFACE_SAMPLES,
        "threshold": DENSE_HAND_OBJECT_PENETRATION_THRESHOLD,
        "strict_less_than": True,
        "sample_count": sample_count,
        "evaluated_count": sample_count,
        "feasible_count": feasible_count,
    }


class X2PrimitiveDatasetTest(unittest.TestCase):
    def test_catalog_has_exact_twelve_named_instances(self) -> None:
        self.assertEqual(len(PRIMITIVE_SPECS), 12)
        self.assertEqual(
            {shape: sum(spec.shape == shape for spec in PRIMITIVE_SPECS) for shape in SHAPES},
            {"sphere": 3, "cylinder": 3, "cuboid": 3, "cube": 3},
        )
        names = [spec.instance_name for spec in PRIMITIVE_SPECS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            names,
            [
                "sphere_r020",
                "sphere_r030",
                "sphere_r040",
                "cylinder_r018_h100",
                "cylinder_r025_h100",
                "cylinder_r032_h080",
                "cuboid_x035_y055_z090",
                "cuboid_x045_y065_z110",
                "cuboid_x055_y075_z130",
                "cube_e040",
                "cube_e050",
                "cube_e060",
            ],
        )

    def test_mesh_build_is_deterministic_watertight_and_dimensionally_exact(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first)
            second_root = Path(second)
            first_reports = build_dataset(first_root, overwrite=True)
            second_reports = build_dataset(second_root, overwrite=True)
            self.assertEqual(len(first_reports), 12)
            self.assertEqual(len(second_reports), 12)
            for spec in PRIMITIVE_SPECS:
                first_path = first_root / spec.relative_path
                second_path = second_root / spec.relative_path
                self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
                mesh = trimesh.load(first_path, force="mesh", process=False)
                audit_mesh(mesh, spec, path=first_path)
                self.assertTrue(mesh.is_watertight)
                self.assertTrue(np.isfinite(np.asarray(mesh.vertices)).all())
                np.testing.assert_allclose(
                    mesh.extents, spec.expected_extents, rtol=0.0, atol=1.0e-9
                )
                if spec.shape == "cylinder":
                    radius, height = spec.dimensions
                    np.testing.assert_allclose(
                        mesh.bounds[:, 2], (-height / 2.0, height / 2.0),
                        rtol=0.0, atol=1.0e-9,
                    )
                    self.assertAlmostEqual(
                        float(np.linalg.vector_norm(mesh.vertices[:, :2], axis=1).max()),
                        radius,
                        places=9,
                    )

    def test_batch_defaults_and_subprocess_contract(self) -> None:
        args = _parse_args([])
        self.assertEqual(args.shapes, list(SHAPES))
        self.assertEqual(args.side, "both")
        self.assertEqual(args.num_grasps, 64)
        self.assertIsNone(args.target_total)
        self.assertEqual(args.n_iterations, 6000)
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.jobs, 1)
        spec = PRIMITIVE_SPECS[0]
        command = build_generator_command(
            spec=spec,
            side="back",
            mesh_path=Path("mesh.obj"),
            staging_output=Path("stage"),
            num_grasps=64,
            n_iterations=500,
            device="cuda:0",
            seed=7,
        )
        joined = " ".join(command)
        for expected in (
            "--side back",
            "--num-grasps 64",
            f"--batch-size {BATCH_SIZE}",
            f"--n-contact {N_CONTACT}",
            "--n-iterations 500",
            f"--surface-samples {SURFACE_SAMPLES}",
            f"--object-scale {OBJECT_SCALE}",
            "--seed 7",
            "--device cuda:0",
        ):
            self.assertIn(expected, joined)
        self.assertNotIn("--side both", joined)
        self.assertEqual(12 * 2 * 8, 192)
        self.assertEqual(12 * 2 * 64, 1536)

    def test_exact_five_thousand_plan_is_balanced_and_deterministic(self) -> None:
        tasks = plan_generation(
            PRIMITIVE_SPECS,
            ("front", "back"),
            num_grasps=64,
            target_total=5000,
        )
        counts = [task.num_grasps for task in tasks]
        self.assertEqual(len(tasks), 24)
        self.assertEqual(sum(counts), 5000)
        self.assertEqual(counts[:8], [209] * 8)
        self.assertEqual(counts[8:], [208] * 16)
        self.assertEqual(
            sum(task.num_grasps for task in tasks if task.side == "front"), 2500
        )
        self.assertEqual(
            sum(task.num_grasps for task in tasks if task.side == "back"), 2500
        )

    def test_general_mesh_discovery_and_combined_five_thousand_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("object-b", "object-a"):
                path = root / name / "coacd" / "decomposed.obj"
                path.parent.mkdir(parents=True)
                path.write_text("mesh", encoding="utf-8")
            specs = discover_general_meshes(root)
            self.assertEqual([spec.instance_name for spec in specs], ["object-a", "object-b"])
            tasks = plan_generation(
                (*PRIMITIVE_SPECS, *specs),
                ("front", "back"),
                num_grasps=64,
                target_total=5000,
            )
            self.assertEqual(len(tasks), 28)
            self.assertEqual(sum(task.num_grasps for task in tasks), 5000)
            self.assertLessEqual(
                max(task.num_grasps for task in tasks)
                - min(task.num_grasps for task in tasks),
                1,
            )

    def test_general_mesh_explicit_subset_is_ordered_and_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("002", "000", "001"):
                path = root / name / "coacd" / "decomposed.obj"
                path.parent.mkdir(parents=True)
                path.write_text("mesh", encoding="utf-8")
            specs = discover_general_meshes(root)
            selected = select_general_meshes(specs, ("002", "000"))
            self.assertEqual(
                [spec.instance_name for spec in selected], ["000", "002"]
            )
            with self.assertRaises(PrimitiveGenerationError):
                select_general_meshes(specs, ("missing",))
            with self.assertRaises(PrimitiveGenerationError):
                select_general_meshes(specs, ("000", "000"))

    def test_ten_thousand_plan_balances_all_five_finger_strata(self) -> None:
        tasks = plan_generation(
            PRIMITIVE_SPECS,
            ("front", "back"),
            num_grasps=64,
            target_total=10000,
            finger_counts=(1, 2, 3, 4, 5),
        )
        self.assertEqual(len(tasks), 120)
        self.assertEqual(sum(task.num_grasps for task in tasks), 10000)
        self.assertEqual({task.finger_count for task in tasks}, {1, 2, 3, 4, 5})
        for finger_count in range(1, 6):
            self.assertEqual(
                sum(
                    task.num_grasps
                    for task in tasks
                    if task.finger_count == finger_count
                ),
                2000,
            )
        command = build_generator_command(
            spec=PRIMITIVE_SPECS[0],
            side="front",
            mesh_path=Path("mesh.obj"),
            staging_output=Path("stage"),
            num_grasps=2,
            n_iterations=3,
            device="cuda",
            seed=0,
            finger_count=5,
        )
        self.assertEqual(command[command.index("--n-contact") + 1], "5")
        self.assertEqual(command[command.index("--finger-count") + 1], "5")

    def test_adaptive_plan_accepts_distinct_per_finger_totals(self) -> None:
        tasks = plan_generation(
            PRIMITIVE_SPECS,
            ("front", "back"),
            num_grasps=64,
            finger_counts=(1, 3, 5),
            finger_targets={1: 1000, 3: 1200, 5: 1400},
            complementary_side_fingers=True,
            seed=23,
        )
        self.assertEqual(len(tasks), 12 * 2 * 3)
        self.assertEqual(sum(task.num_grasps for task in tasks), 3600)
        self.assertEqual(
            {
                finger_count: sum(
                    task.num_grasps
                    for task in tasks
                    if task.finger_count == finger_count
                )
                for finger_count in (1, 3, 5)
            },
            {1: 1000, 3: 1200, 5: 1400},
        )
        parsed = _parse_args(
            [
                "--side",
                "both",
                "--finger-counts",
                "1",
                "3",
                "--finger-targets",
                "500",
                "700",
                "--complementary-side-fingers",
            ]
        )
        self.assertEqual(parsed.finger_targets, [500, 700])

    def test_complementary_side_masks_are_disjoint_per_object(self) -> None:
        tasks = plan_generation(
            PRIMITIVE_SPECS[:1],
            ("front", "back"),
            num_grasps=1,
            finger_counts=(1, 2, 3, 4, 5),
            complementary_side_fingers=True,
            seed=17,
        )
        by_key = {(task.side, task.finger_count): task for task in tasks}
        for front_count in (1, 2, 3, 4):
            front = by_key[("front", front_count)]
            back = by_key[("back", 5 - front_count)]
            self.assertEqual(len(front.finger_names), front_count)
            self.assertEqual(len(back.finger_names), 5 - front_count)
            self.assertTrue(set(front.finger_names).isdisjoint(back.finger_names))
            self.assertEqual(
                set(front.finger_names) | set(back.finger_names),
                {"index", "middle", "ring", "little", "thumb"},
            )

    def test_csv_summary_uses_strict_decrease_and_penetration_statistics(self) -> None:
        spec = PRIMITIVE_SPECS[0]
        records = [
            {
                "finite": True,
                "energy": {"initial_total": 10.0, "total": 8.0},
                "maximum_penetration": 0.003,
            },
            {
                "finite": False,
                "energy": {"initial_total": 7.0, "total": 7.0},
                "maximum_penetration": 0.001,
            },
            {
                "finite": True,
                "energy": {"initial_total": 5.0, "total": 6.0},
                "maximum_penetration": 0.002,
            },
        ]
        row = summarize_records(spec, "front", records)
        self.assertEqual(row["sample_count"], 3)
        self.assertEqual(row["finite_sample_count"], 2)
        self.assertEqual(row["energy_decreased_count"], 1)
        self.assertAlmostEqual(row["mean_initial_energy"], 22.0 / 3.0)
        self.assertAlmostEqual(row["mean_final_energy"], 7.0)
        self.assertAlmostEqual(row["maximum_penetration_mean"], 0.002)
        self.assertAlmostEqual(row["maximum_penetration_min"], 0.001)
        self.assertAlmostEqual(row["maximum_penetration_median"], 0.002)

    @staticmethod
    def _fake_generator(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        def value(flag: str) -> str:
            return command[command.index(flag) + 1]

        mesh_path = Path(value("--mesh-path")).resolve()
        side = value("--side")
        num_grasps = int(value("--num-grasps"))
        output = Path(value("--output"))
        raw = output / f"{side}_single" / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        feasible_count = 0
        for index in range(num_grasps):
            maximum_penetration = 0.001 * (index + 1)
            dense_gate = _v5_dense_gate(maximum_penetration)
            feasible_count += int(dense_gate["feasible"] is True)
            record = {
                "pipeline_revision": GENERATOR_PIPELINE_REVISION,
                "sample_index": index,
                "active_side": side,
                "object": {
                    "mesh_path": str(mesh_path),
                    "scale": float(value("--object-scale")),
                    "watertight": True,
                },
                "actuator": [0.0] * 12,
                "joint": [0.0] * 16,
                "selected_contact_ids": [f"contact-{slot}" for slot in range(4)],
                "energy": {
                    "initial_total": 10.0 + index,
                    "total": 5.0 + index,
                },
                "maximum_penetration": maximum_penetration,
                "hand_object_penetration": dense_gate,
                "seed": int(value("--seed")),
                "optimization": {"iterations": int(value("--n-iterations"))},
                "finite": index % 2 == 0,
                "success": False,
                "simulation_success": False,
                "validation": {"status": "not_run", "backend": None},
            }
            path = raw / f"{mesh_path.stem}_{side}_{index:06d}.json"
            path.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
        summary = {
            "pipeline_revision": GENERATOR_PIPELINE_REVISION,
            "side_mode": side,
            "num_output_samples": num_grasps,
            "dense_hand_object_gate": _v5_dense_summary(
                num_grasps, feasible_count
            ),
        }
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(summary), stderr=""
        )

    @staticmethod
    def _fake_stratified_generator(
        command: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        def value(flag: str) -> str:
            return command[command.index(flag) + 1]

        mesh_path = Path(value("--mesh-path")).resolve()
        object_scale = float(value("--object-scale"))
        seed = int(value("--seed"))
        n_iterations = int(value("--n-iterations"))
        batch_size = int(value("--batch-size"))
        plan = json.loads(Path(value("--plan")).read_text(encoding="utf-8"))
        groups = plan["groups"]
        feasible_count = 0
        for group in groups:
            side = group["side"]
            finger_count = group["finger_count"]
            finger_names = group["finger_names"]
            raw = Path(group["output"]) / f"{side}_single" / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            for index in range(group["num_grasps"]):
                maximum_penetration = 0.001 * (index + 1)
                dense_gate = _v5_dense_gate(maximum_penetration)
                feasible_count += int(dense_gate["feasible"] is True)
                record = {
                    "pipeline_revision": GENERATOR_PIPELINE_REVISION,
                    "sample_index": index,
                    "active_side": side,
                    "object": {
                        "mesh_path": str(mesh_path),
                        "scale": object_scale,
                        "watertight": True,
                    },
                    "actuator": [0.0] * 12,
                    "joint": [0.0] * 16,
                    "selected_contact_ids": [
                        f"contact-{slot}"
                        for slot in range(max(4, finger_count))
                    ],
                    "finger_participation": {
                        "target_count": finger_count,
                        "actual_count": finger_count,
                        "finger_names": finger_names,
                    },
                    "energy": {
                        "initial_total": 10.0 + index,
                        "total": 5.0 + index,
                    },
                    "maximum_penetration": maximum_penetration,
                    "hand_object_penetration": dense_gate,
                    "seed": seed,
                    "optimization": {"iterations": n_iterations},
                    "finite": True,
                    "success": False,
                    "simulation_success": False,
                    "validation": {"status": "not_run", "backend": None},
                }
                path = raw / f"{mesh_path.stem}_{side}_{index:06d}.json"
                path.write_text(
                    json.dumps(record, allow_nan=False), encoding="utf-8"
                )
        contact_partitions = {
            max(4, int(group["finger_count"])) for group in groups
        }
        summary = {
            "passed": True,
            "pipeline_revision": GENERATOR_PIPELINE_REVISION,
            "side_mode": "stratified",
            "group_count": len(groups),
            "num_output_samples": sum(group["num_grasps"] for group in groups),
            "object_scale": object_scale,
            "batch_count": len(contact_partitions),
            "batch_size": batch_size,
            "dense_hand_object_gate": _v5_dense_summary(
                sum(group["num_grasps"] for group in groups), feasible_count
            ),
        }
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(summary), stderr=""
        )

    def test_stratified_resume_batches_per_object_and_repairs_one_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_root = root / "primitive-meshes"
            general_root = root / "general-meshes"
            output_root = root / "grasps"
            general_mesh = (
                general_root / "object-x" / "coacd" / "decomposed.obj"
            )
            general_mesh.parent.mkdir(parents=True)
            general_mesh.write_text("mesh", encoding="utf-8")
            general_hash = hashlib.sha256(general_mesh.read_bytes()).hexdigest()
            (general_root / "x2_general_mesh_manifest.json").write_text(
                json.dumps(
                    {
                        "object_scale": 1.0,
                        "meshes": [
                            {
                                "object_id": "object-x",
                                "sha256": general_hash,
                                "object_scale": 1.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            common = [
                "--shapes",
                "sphere",
                "--include-general-meshes",
                "--general-mesh-root",
                str(general_root),
                "--side",
                "both",
                "--finger-counts",
                "1",
                "2",
                "3",
                "4",
                "5",
                "--finger-targets",
                "8",
                "8",
                "8",
                "8",
                "8",
                "--complementary-side-fingers",
                "--n-iterations",
                "3",
                "--seed",
                "11",
                "--mesh-root",
                str(mesh_root),
                "--output-root",
                str(output_root),
                "--resume",
                "--stratified-batching",
            ]
            initial_plans: list[dict] = []

            def initial_fake(
                command: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                initial_plans.append(
                    json.loads(
                        Path(command[command.index("--plan") + 1]).read_text(
                            encoding="utf-8"
                        )
                    )
                )
                return self._fake_stratified_generator(command, **kwargs)

            with mock.patch(
                "scripts.generate_x2_primitive_dataset.subprocess.run",
                side_effect=initial_fake,
            ) as initial_generator:
                report = run(_parse_args(common))

            self.assertEqual(initial_generator.call_count, 4)
            self.assertEqual([len(plan["groups"]) for plan in initial_plans], [10] * 4)
            self.assertEqual(report["generated_group_count"], 40)
            self.assertEqual(report["total_samples"], 40)
            self.assertEqual(report["total_finite_samples"], 40)
            self.assertTrue(report["settings"]["stratified_batching"])
            self.assertEqual(report["settings"]["stratified_batch_size"], 64)
            self.assertEqual(
                report["settings"]["generator_pipeline_revision"],
                GENERATOR_PIPELINE_REVISION,
            )
            self.assertEqual(
                report["settings"]["dense_hand_object_gate"][
                    "hand_surface_point_count"
                ],
                DENSE_HAND_SURFACE_POINT_COUNT,
            )
            self.assertEqual(report["dense_hand_object_gate"]["sample_count"], 40)
            self.assertEqual(
                report["dense_hand_object_gate"]["evaluated_count"], 40
            )
            self.assertIs(
                report["dense_hand_object_gate"]["all_samples_evaluated"], True
            )
            self.assertTrue(
                all(
                    call["pipeline_revision"] == GENERATOR_PIPELINE_REVISION
                    and call["dense_hand_object_gate"]["evaluated_count"]
                    == call["num_output_samples"]
                    for call in report["generator_calls"]
                )
            )
            self.assertEqual(
                report["settings"]["object_scale_by_id"]["object-x"], 1.0
            )
            persisted = json.loads(
                (output_root / "generation_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(persisted["total_samples"], 40)
            self.assertTrue(persisted["settings"]["stratified_batching"])
            self.assertIs(
                persisted["dense_hand_object_gate"]["all_samples_evaluated"],
                True,
            )

            preserved = (
                output_root
                / "sphere"
                / "front"
                / "raw"
                / "sphere_r040_f1_front_000000.json"
            )
            preserved_bytes = preserved.read_bytes()
            damaged = (
                output_root
                / "general"
                / "front"
                / "raw"
                / "object-x_f2_front_000000.json"
            )
            damaged.write_text("{}", encoding="utf-8")
            stale_valid = damaged.parent.parent / "valid" / damaged.name
            stale_valid.parent.mkdir(parents=True)
            stale_valid.write_text("stale", encoding="utf-8")
            repair_plans: list[dict] = []

            def repair_fake(
                command: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                repair_plans.append(
                    json.loads(
                        Path(command[command.index("--plan") + 1]).read_text(
                            encoding="utf-8"
                        )
                    )
                )
                return self._fake_stratified_generator(command, **kwargs)

            with mock.patch(
                "scripts.generate_x2_primitive_dataset.subprocess.run",
                side_effect=repair_fake,
            ) as repair_generator:
                repaired = run(_parse_args(common))

            self.assertEqual(repair_generator.call_count, 1)
            self.assertEqual(len(repair_plans), 1)
            self.assertEqual(len(repair_plans[0]["groups"]), 1)
            self.assertEqual(repair_plans[0]["groups"][0]["side"], "front")
            self.assertEqual(repair_plans[0]["groups"][0]["finger_count"], 2)
            self.assertEqual(repaired["reused_group_count"], 39)
            self.assertEqual(repaired["generated_group_count"], 1)
            self.assertEqual(repaired["total_samples"], 40)
            self.assertEqual(preserved.read_bytes(), preserved_bytes)
            self.assertEqual(json.loads(damaged.read_text())["object"]["scale"], 1.0)
            self.assertFalse(stale_valid.exists())

    def test_stratified_stage_rejects_an_incomplete_dense_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "grasps"
            args = _parse_args(
                [
                    "--shapes", "sphere",
                    "--side", "both",
                    "--finger-counts", "1",
                    "--finger-targets", "6",
                    "--complementary-side-fingers",
                    "--n-iterations", "3",
                    "--mesh-root", str(root / "meshes"),
                    "--output-root", str(output_root),
                    "--resume",
                    "--stratified-batching",
                ]
            )

            def incomplete_summary(
                command: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                completed = self._fake_stratified_generator(command, **kwargs)
                summary = json.loads(completed.stdout)
                summary["dense_hand_object_gate"]["evaluated_count"] -= 1
                return subprocess.CompletedProcess(
                    command, 0, stdout=json.dumps(summary), stderr=""
                )

            with mock.patch(
                "scripts.generate_x2_primitive_dataset.subprocess.run",
                side_effect=incomplete_summary,
            ):
                with self.assertRaisesRegex(
                    PrimitiveGenerationError, "every generated raw sample"
                ):
                    run(args)
            self.assertFalse(list(output_root.glob("**/raw/*.json")))
            self.assertFalse((output_root / "generation_summary.json").exists())

    def test_batch_routes_staging_outputs_and_overwrite_removes_only_stale_instance_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_root = root / "meshes"
            output_root = root / "grasps"
            args = _parse_args(
                [
                    "--shapes", "sphere",
                    "--side", "both",
                    "--num-grasps", "2",
                    "--n-iterations", "3",
                    "--device", "mock-device",
                    "--mesh-root", str(mesh_root),
                    "--output-root", str(output_root),
                ]
            )
            with mock.patch(
                "scripts.generate_x2_primitive_dataset.subprocess.run",
                side_effect=self._fake_generator,
            ) as patched:
                report = run(args)
            self.assertEqual(patched.call_count, 6)
            self.assertEqual(report["instance_side_runs"], 6)
            self.assertEqual(report["total_samples"], 12)
            self.assertEqual(report["total_finite_samples"], 6)
            self.assertFalse(list(output_root.glob("**/*_single")))
            self.assertFalse(list(output_root.glob("**/valid")))
            self.assertFalse(list(output_root.glob("**/failed")))
            for spec in PRIMITIVE_SPECS[:3]:
                for side in ("front", "back"):
                    paths = sorted(
                        (output_root / "sphere" / side / "raw").glob(
                            f"{spec.instance_name}_{side}_*.json"
                        )
                    )
                    self.assertEqual(
                        [path.name for path in paths],
                        [
                            f"{spec.instance_name}_{side}_000000.json",
                            f"{spec.instance_name}_{side}_000001.json",
                        ],
                    )
            with (output_root / "summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(tuple(rows[0]), CSV_FIELDS)
            self.assertEqual(len(rows), 6)

            with mock.patch(
                "scripts.generate_x2_primitive_dataset.subprocess.run"
            ) as should_not_run:
                with self.assertRaises(PrimitiveGenerationError):
                    run(args)
                should_not_run.assert_not_called()

            unrelated = output_root / "cube" / "front" / "raw" / "cube_e040_front_000999.json"
            unrelated.parent.mkdir(parents=True, exist_ok=True)
            unrelated.write_text("unrelated", encoding="utf-8")
            overwrite_args = _parse_args(
                [
                    "--shapes", "sphere",
                    "--side", "both",
                    "--num-grasps", "1",
                    "--n-iterations", "3",
                    "--jobs", "2",
                    "--mesh-root", str(mesh_root),
                    "--output-root", str(output_root),
                    "--overwrite",
                ]
            )
            with mock.patch(
                "scripts.generate_x2_primitive_dataset.subprocess.run",
                side_effect=self._fake_generator,
            ):
                overwrite_report = run(overwrite_args)
            self.assertEqual(overwrite_report["total_samples"], 6)
            self.assertTrue(unrelated.is_file())
            for spec in PRIMITIVE_SPECS[:3]:
                for side in ("front", "back"):
                    paths = list(
                        (output_root / "sphere" / side / "raw").glob(
                            f"{spec.instance_name}_{side}_*.json"
                        )
                    )
                    self.assertEqual(len(paths), 1)

    def test_concurrent_stage_failure_preserves_existing_outputs_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_root = root / "meshes"
            output_root = root / "grasps"
            existing = (
                output_root
                / "sphere"
                / "front"
                / "raw"
                / "sphere_r020_front_000000.json"
            )
            existing.parent.mkdir(parents=True)
            existing.write_text("existing-record", encoding="utf-8")
            summary = output_root / "summary.csv"
            summary.write_text("existing-summary\n", encoding="utf-8")
            args = _parse_args(
                [
                    "--shapes", "sphere",
                    "--side", "both",
                    "--num-grasps", "1",
                    "--n-iterations", "1",
                    "--jobs", "2",
                    "--mesh-root", str(mesh_root),
                    "--output-root", str(output_root),
                    "--overwrite",
                ]
            )

            def fail_one(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                mesh_path = command[command.index("--mesh-path") + 1]
                side = command[command.index("--side") + 1]
                if "sphere_r030.obj" in mesh_path and side == "front":
                    return subprocess.CompletedProcess(
                        command, 17, stdout="", stderr="intentional failure"
                    )
                return self._fake_generator(command, **kwargs)

            with mock.patch(
                "scripts.generate_x2_primitive_dataset.subprocess.run",
                side_effect=fail_one,
            ):
                with self.assertRaises(PrimitiveGenerationError):
                    run(args)
            self.assertEqual(existing.read_text(encoding="utf-8"), "existing-record")
            self.assertEqual(summary.read_text(encoding="utf-8"), "existing-summary\n")
            self.assertEqual(
                list(output_root.glob("*/*/raw/*.json")),
                [existing],
            )

    def test_resume_reuses_complete_groups_and_repairs_only_damaged_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_root = root / "meshes"
            output_root = root / "grasps"
            common = [
                "--shapes", "sphere",
                "--side", "both",
                "--num-grasps", "1",
                "--n-iterations", "3",
                "--seed", "11",
                "--mesh-root", str(mesh_root),
                "--output-root", str(output_root),
            ]
            with mock.patch(
                "scripts.generate_x2_primitive_dataset.subprocess.run",
                side_effect=self._fake_generator,
            ) as initial_generator:
                initial_report = run(_parse_args(common))
            self.assertEqual(initial_generator.call_count, 6)
            self.assertEqual(initial_report["generated_group_count"], 6)

            preserved = (
                output_root
                / "sphere"
                / "front"
                / "raw"
                / "sphere_r040_front_000000.json"
            )
            preserved_bytes = preserved.read_bytes()
            missing = (
                output_root
                / "sphere"
                / "front"
                / "raw"
                / "sphere_r020_front_000000.json"
            )
            missing.unlink()
            damaged = (
                output_root
                / "sphere"
                / "back"
                / "raw"
                / "sphere_r030_back_000000.json"
            )
            damaged.write_text("{}", encoding="utf-8")
            stale_valid = damaged.parent.parent / "valid" / damaged.name
            stale_valid.parent.mkdir(parents=True)
            stale_valid.write_text("stale", encoding="utf-8")

            with mock.patch(
                "scripts.generate_x2_primitive_dataset.subprocess.run",
                side_effect=self._fake_generator,
            ) as resumed_generator:
                report = run(_parse_args([*common, "--jobs", "2", "--resume"]))
            self.assertEqual(resumed_generator.call_count, 2)
            self.assertEqual(report["reused_group_count"], 4)
            self.assertEqual(report["generated_group_count"], 2)
            self.assertEqual(report["instance_side_runs"], 6)
            self.assertEqual(report["total_samples"], 6)
            self.assertEqual(preserved.read_bytes(), preserved_bytes)
            self.assertTrue(missing.is_file())
            self.assertEqual(json.loads(damaged.read_text())["seed"], 11)
            self.assertFalse(stale_valid.exists())
            with (output_root / "summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 6)
            self.assertFalse((output_root / "summary.csv.tmp").exists())

            with self.assertRaises(SystemExit):
                _parse_args(["--overwrite", "--resume"])

    def test_resume_repairs_every_stale_or_inconsistent_v5_dense_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "grasps"
            common = [
                "--shapes", "sphere",
                "--side", "both",
                "--num-grasps", "1",
                "--n-iterations", "3",
                "--seed", "11",
                "--mesh-root", str(root / "meshes"),
                "--output-root", str(output_root),
            ]
            with mock.patch(
                "scripts.generate_x2_primitive_dataset.subprocess.run",
                side_effect=self._fake_generator,
            ):
                run(_parse_args(common))

            target = (
                output_root
                / "sphere"
                / "front"
                / "raw"
                / "sphere_r020_front_000000.json"
            )

            def old_pipeline(payload: dict) -> None:
                payload["pipeline_revision"] = (
                    "x2_mesh_grasp_unselected_finger_side_v4"
                )

            def wrong_density(payload: dict) -> None:
                payload["hand_object_penetration"][
                    "hand_surface_samples_per_set"
                ] = 255

            def not_evaluated(payload: dict) -> None:
                payload["hand_object_penetration"]["evaluated"] = False

            def equality_marked_feasible(payload: dict) -> None:
                self.assertEqual(payload["maximum_penetration"], 0.001)
                payload["hand_object_penetration"]["feasible"] = True

            def wrong_top_level_maximum(payload: dict) -> None:
                payload["maximum_penetration"] = 0.0009

            def wrong_threshold(payload: dict) -> None:
                payload["hand_object_penetration"]["threshold"] = 0.0011

            for label, mutate in (
                ("v4 pipeline", old_pipeline),
                ("wrong dense density", wrong_density),
                ("unevaluated dense gate", not_evaluated),
                ("1 mm marked feasible", equality_marked_feasible),
                ("top maximum mismatch", wrong_top_level_maximum),
                ("wrong threshold", wrong_threshold),
            ):
                with self.subTest(label):
                    payload = json.loads(target.read_text(encoding="utf-8"))
                    mutate(payload)
                    target.write_text(
                        json.dumps(payload, allow_nan=False), encoding="utf-8"
                    )
                    with mock.patch(
                        "scripts.generate_x2_primitive_dataset.subprocess.run",
                        side_effect=self._fake_generator,
                    ) as generator:
                        report = run(_parse_args([*common, "--resume"]))
                    self.assertEqual(generator.call_count, 1)
                    self.assertEqual(report["reused_group_count"], 5)
                    self.assertEqual(report["generated_group_count"], 1)
                    repaired = json.loads(target.read_text(encoding="utf-8"))
                    self.assertEqual(
                        repaired["pipeline_revision"], GENERATOR_PIPELINE_REVISION
                    )
                    self.assertIs(
                        repaired["hand_object_penetration"]["evaluated"], True
                    )
                    self.assertFalse(
                        repaired["hand_object_penetration"]["feasible"]
                    )

    def test_resume_failure_checkpoints_success_and_cancels_unstarted_futures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "grasps"
            barrier = threading.Barrier(2)
            call_lock = threading.Lock()
            calls: list[list[str]] = []

            def fail_second(
                command: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                with call_lock:
                    ordinal = len(calls)
                    calls.append(command)
                if ordinal < 2:
                    barrier.wait(timeout=2.0)
                if ordinal == 1:
                    return subprocess.CompletedProcess(
                        command, 17, stdout="", stderr="intentional failure"
                    )
                if ordinal >= 2:
                    time.sleep(0.2)
                return self._fake_generator(command, **kwargs)

            args = _parse_args(
                [
                    "--shapes", "sphere",
                    "--side", "both",
                    "--num-grasps", "1",
                    "--n-iterations", "3",
                    "--jobs", "2",
                    "--mesh-root", str(root / "meshes"),
                    "--output-root", str(output_root),
                    "--resume",
                ]
            )
            output_root.mkdir(parents=True)
            (output_root / "summary.csv").write_text(
                "stale-complete-marker\n", encoding="utf-8"
            )
            with mock.patch(
                "scripts.generate_x2_primitive_dataset.subprocess.run",
                side_effect=fail_second,
            ):
                with self.assertRaises(PrimitiveGenerationError):
                    run(args)

            self.assertGreaterEqual(len(calls), 2)
            self.assertLess(len(calls), 6)
            self.assertTrue(list(output_root.glob("sphere/*/raw/*.json")))
            self.assertFalse((output_root / "summary.csv").exists())

    def test_primitive_scripts_do_not_import_generator_internals_or_simulators(self) -> None:
        forbidden_imports = ("grasp_generation", "pxr", "isaac")
        forbidden_symbols = (
            "optimize_x2_mesh_batch",
            "cal_x2_mesh_energy",
            "GenericDexterousContactPolicy",
            "X2HandModel",
            "ActuatorHandModel",
            "load_generic_contact_candidates",
        )
        for path in (
            Path("scripts/build_x2_primitive_dataset.py"),
            Path("scripts/generate_x2_primitive_dataset.py"),
        ):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            imports: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            self.assertFalse(
                [name for name in imports if name.startswith(forbidden_imports)]
            )
            self.assertFalse([symbol for symbol in forbidden_symbols if symbol in source])


if __name__ == "__main__":
    unittest.main()
