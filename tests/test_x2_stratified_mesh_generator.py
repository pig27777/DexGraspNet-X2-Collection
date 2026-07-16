"""Tests for the resident, row-policy X2 stratified mesh generator."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.generate_x2_mesh_grasps_stratified import (
    FINGER_NAMES,
    StratifiedGenerationError,
    _parse_args,
    load_plan,
    main,
    run,
    schedule_batches,
)


class X2StratifiedMeshGeneratorTest(unittest.TestCase):
    @staticmethod
    def _write_plan(root: Path, groups: list[dict]) -> Path:
        path = root / "plan.json"
        path.write_text(json.dumps({"groups": groups}), encoding="utf-8")
        return path

    def test_plan_is_strict_and_scheduler_round_robins_without_ragged_contacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._write_plan(
                root,
                [
                    {
                        "side": "front",
                        "finger_count": 1,
                        "finger_names": ["thumb"],
                        "num_grasps": 3,
                        "output": "front_f1",
                    },
                    {
                        "side": "back",
                        "finger_count": 4,
                        "finger_names": ["little", "ring", "middle", "index"],
                        "num_grasps": 2,
                        "output": "back_f4",
                    },
                    {
                        "side": "front",
                        "finger_count": 5,
                        "finger_names": list(reversed(FINGER_NAMES)),
                        "num_grasps": 2,
                        "output": "front_f5",
                    },
                    {
                        "side": "back",
                        "finger_count": 5,
                        "finger_names": list(FINGER_NAMES),
                        "num_grasps": 1,
                        "output": "back_f5",
                    },
                ],
            )
            groups = load_plan(plan)
            self.assertEqual(groups[1].finger_names, ("index", "middle", "ring", "little"))
            self.assertEqual(groups[2].finger_names, tuple(FINGER_NAMES))
            self.assertEqual(groups[0].output, (root / "front_f1").resolve())

            batches = schedule_batches(groups, batch_size=3)
            self.assertEqual(
                [(batch.contact_count, len(batch.rows)) for batch in batches],
                [(4, 3), (4, 2), (5, 3)],
            )
            self.assertEqual(
                [row.group.index for batch in batches[:2] for row in batch.rows],
                [0, 1, 0, 1, 0],
            )
            self.assertEqual(
                [row.group.index for row in batches[2].rows],
                [2, 3, 2],
            )
            for batch in batches:
                self.assertEqual(
                    {row.group.n_contact for row in batch.rows},
                    {batch.contact_count},
                )

            invalid = root / "invalid.json"
            invalid.write_text(
                json.dumps(
                    [
                        {
                            "side": "front",
                            "finger_count": 2,
                            "finger_names": ["thumb", "thumb"],
                            "num_grasps": 1,
                            "output": "bad",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(StratifiedGenerationError):
                load_plan(invalid)

    def test_run_reuses_resident_hand_and_routes_mixed_row_policies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "object.obj"
            mesh.write_text("mock mesh", encoding="utf-8")
            plan = self._write_plan(
                root,
                [
                    {
                        "side": "front",
                        "finger_count": 1,
                        "finger_names": ["thumb"],
                        "num_grasps": 2,
                        "output": "front_f1",
                    },
                    {
                        "side": "back",
                        "finger_count": 4,
                        "finger_names": ["index", "middle", "ring", "little"],
                        "num_grasps": 2,
                        "output": "back_f4",
                    },
                    {
                        "side": "front",
                        "finger_count": 5,
                        "finger_names": list(FINGER_NAMES),
                        "num_grasps": 2,
                        "output": "front_f5",
                    },
                ],
            )
            args = _parse_args(
                [
                    "--mesh-path", str(mesh),
                    "--plan", str(plan),
                    "--n-iterations", "3",
                    "--seed", "17",
                    "--device", "cpu",
                    "--object-scale", "0.1",
                    "--surface-samples", "32",
                    "--batch-size", "3",
                ]
            )

            object_calls: list[dict] = []
            optimizer_calls: list[dict] = []
            hand_calls: list[dict] = []

            class FakeObjectModel:
                def __init__(self, mesh_path: Path, **kwargs: object) -> None:
                    self.mesh_path = Path(mesh_path).resolve()
                    self.scale = float(kwargs["scale"])
                    object_calls.append(dict(kwargs))

            class FakeHand:
                def __init__(self, *_: object, **kwargs: object) -> None:
                    hand_calls.append(dict(kwargs))

            def fake_optimize(
                hand: object,
                object_model: object,
                active_sides: tuple[str, ...],
                policies: tuple[object, ...],
                config: object,
                **kwargs: object,
            ) -> SimpleNamespace:
                del hand, object_model, config
                optimizer_calls.append(
                    {
                        "active_sides": tuple(active_sides),
                        "policies": tuple(policies),
                        **kwargs,
                    }
                )
                return SimpleNamespace(
                    active_sides=tuple(active_sides), row_policies=tuple(policies)
                )

            def fake_records(
                hand: object,
                object_model: FakeObjectModel,
                result: SimpleNamespace,
                candidates: object,
                *,
                seed: int,
            ) -> list[dict]:
                del hand, candidates
                records = []
                for side, policy in zip(result.active_sides, result.row_policies):
                    records.append(
                        {
                            "schema_version": 1,
                            "pipeline_revision": (
                                "x2_mesh_grasp_unselected_finger_side_v6"
                            ),
                            "sample_index": 999,
                            "active_side": side,
                            "object": {
                                "mesh_path": str(object_model.mesh_path),
                                "scale": object_model.scale,
                                "watertight": True,
                            },
                            "selected_contacts": [
                                {"finger_name": name}
                                for name in policy.required_finger_names
                            ],
                            "selected_contact_ids": [
                                f"contact-{index}"
                                for index in range(policy.n_contact)
                            ],
                            "hand_object_penetration": {
                                "evaluated": True,
                                "feasible": True,
                            },
                            "seed": seed,
                            "finite": True,
                            "success": False,
                            "simulation_success": False,
                            "validation": {"status": "not_run", "backend": None},
                        }
                    )
                return records

            with (
                mock.patch(
                    "scripts.generate_x2_mesh_grasps_stratified.MeshObjectModel",
                    FakeObjectModel,
                ),
                mock.patch(
                    "scripts.generate_x2_mesh_grasps_stratified.X2HandModel",
                    FakeHand,
                ),
                mock.patch(
                    "scripts.generate_x2_mesh_grasps_stratified.optimize_x2_mesh_batch",
                    side_effect=fake_optimize,
                ),
                mock.patch(
                    "scripts.generate_x2_mesh_grasps_stratified.make_sample_records",
                    side_effect=fake_records,
                ),
            ):
                report = run(args)

            self.assertEqual(len(hand_calls), 1)
            self.assertEqual(len(object_calls), 3)
            self.assertEqual([call["batch_size"] for call in object_calls], [3, 1, 2])
            self.assertEqual(len(optimizer_calls), 3)
            self.assertEqual(
                [
                    {policy.n_contact for policy in call["policies"]}
                    for call in optimizer_calls
                ],
                [{4}, {4}, {5}],
            )
            self.assertEqual(
                [
                    tuple(policy.required_finger_names for policy in call["policies"])
                    for call in optimizer_calls
                ],
                [
                    (("thumb",), ("index", "middle", "ring", "little"), ("thumb",)),
                    (("index", "middle", "ring", "little"),),
                    (tuple(FINGER_NAMES), tuple(FINGER_NAMES)),
                ],
            )
            self.assertEqual(report["num_output_samples"], 6)
            self.assertEqual(
                report["pipeline_revision"],
                "x2_mesh_grasp_unselected_finger_side_v6",
            )
            self.assertEqual(
                report["dense_hand_object_gate"]["evaluated_count"], 6
            )
            self.assertEqual(
                [(batch["contact_count"], batch["sample_count"]) for batch in report["batches"]],
                [(4, 3), (4, 1), (5, 2)],
            )

            groups = load_plan(plan)
            for group in groups:
                paths = sorted(
                    (
                        group.output
                        / f"{group.side}_single"
                        / "raw"
                    ).glob("*.json")
                )
                self.assertEqual(len(paths), group.num_grasps)
                payloads = [json.loads(path.read_text()) for path in paths]
                self.assertEqual(
                    [payload["sample_index"] for payload in payloads],
                    list(range(group.num_grasps)),
                )
                for payload in payloads:
                    self.assertEqual(
                        payload["finger_participation"],
                        {
                            "target_count": group.finger_count,
                            "actual_count": group.finger_count,
                            "finger_names": list(group.finger_names),
                        },
                    )

            with self.assertRaises(StratifiedGenerationError):
                run(args)

    def test_main_prints_one_strict_json_summary(self) -> None:
        stdout = io.StringIO()
        with (
            mock.patch(
                "scripts.generate_x2_mesh_grasps_stratified.run",
                return_value={"passed": True, "value": 1.5},
            ),
            redirect_stdout(stdout),
        ):
            result = main(["--mesh-path", "mesh.obj", "--plan", "plan.json"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"passed": True, "value": 1.5})


if __name__ == "__main__":
    unittest.main()
