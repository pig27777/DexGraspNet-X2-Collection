from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pytest

from scripts.collect_x2_tabletop_physx_dataset import (
    FINGER_COUNTS,
    DEFAULT_FR5_HOME,
    DEFAULT_FR5_MOUNT,
    DEFAULT_FR5_VENDOR_ROOT,
    DEFAULT_X2_WORKSPACE_BOUNDS,
    FR5TableCollisionGate,
    GPUDevice,
    SIDES,
    _completion_status,
    _auto_gpu_execution_plan,
    _planned_rows_per_object,
    _pool_summary,
    _round_robin_select,
    _static_failure_reasons,
    _stratum_targets,
    _validator_command,
    build_catalog,
    _strict_json,
)


@pytest.fixture(scope="module")
def catalog():
    return build_catalog(0.09)


@pytest.fixture(scope="module")
def fr5_gate():
    return FR5TableCollisionGate(
        vendor_root=DEFAULT_FR5_VENDOR_ROOT,
        home_path=DEFAULT_FR5_HOME,
        mount_path=DEFAULT_FR5_MOUNT,
        x2_workspace_bounds_path=DEFAULT_X2_WORKSPACE_BOUNDS,
        object_table_xy_m=(0.0, 0.0),
        minimum_x2_root_table_distance_m=0.05,
        robot_table_clearance_m=0.005,
        ik_seed_count=8,
    )


def test_catalog_contains_12_primitives_and_30_normalized_general_meshes(catalog):
    primitives = [spec for spec in catalog if spec.kind == "primitive"]
    general = [spec for spec in catalog if spec.kind == "general"]
    assert len(catalog) == 42
    assert len(primitives) == 12
    assert len(general) == 30
    assert {spec.shape for spec in primitives} == {
        "sphere",
        "cylinder",
        "cuboid",
        "cube",
    }
    for spec in general:
        assert max(spec.physical_extents_m) == pytest.approx(0.09, abs=1.0e-12)
        assert spec.table_plane_z_m < max(spec.physical_extents_m)


def test_static_gate_accepts_only_complete_table_conditioned_record(catalog):
    spec = catalog[0]
    payload = {
        "finite": True,
        "active_side": "front",
        "finger_participation": {
            "target_count": 3,
            "actual_count": 3,
            "finger_names": ["index", "middle", "thumb"],
        },
        "object": {"mesh_path": str(spec.mesh_path), "scale": spec.scale},
        "selected_contact_realization": {"status": "PASS"},
        "table_conditioning": {
            "source_plane_nonpenetrating": True,
            "requested_clearance_met": True,
            "plane_offset_m": spec.table_plane_z_m,
        },
        "hand_object_penetration": {"feasible": True},
        "self_collision": {"feasible": True},
    }
    assert _static_failure_reasons(payload, spec=spec, finger_count=3) == []

    payload["selected_contact_realization"] = {"status": "FAIL"}
    payload["table_conditioning"]["requested_clearance_met"] = False
    payload["self_collision"] = {"feasible": False}
    assert _static_failure_reasons(payload, spec=spec, finger_count=3) == [
        "SELECTED_CONTACT_REALIZATION_FAILED",
        "SELF_COLLISION_GATE_FAILED",
        "TABLE_CLEARANCE_MARGIN_FAILED",
    ]


def test_adaptive_plan_is_bounded():
    deficits = {(side, finger): 1000 for side in SIDES for finger in FINGER_COUNTS}
    args = argparse.Namespace(
        expected_physx_pass_rate=0.1,
        minimum_raw_per_object_stratum=8,
        maximum_raw_per_object_stratum=32,
    )
    assert _planned_rows_per_object(deficits, object_count=42, args=args) == {
        finger: 32 for finger in FINGER_COUNTS
    }
    deficits = {(side, finger): 0 for side in SIDES for finger in FINGER_COUNTS}
    assert _planned_rows_per_object(deficits, object_count=42, args=args) == {}


def test_selected_finger_targets_are_exact_and_front_back_symmetric():
    args = argparse.Namespace(target_total=10000, finger_counts=[2, 3, 5])
    targets = _stratum_targets(args)
    assert targets == {
        ("front", 2): 1667,
        ("front", 3): 1667,
        ("front", 5): 1666,
        ("back", 2): 1667,
        ("back", 3): 1667,
        ("back", 5): 1666,
    }
    assert sum(targets.values()) == 10000


def test_validator_command_uses_mu1_and_safe_contact_optimization(catalog):
    args = argparse.Namespace(
        sim_steps=100,
        substeps=2,
        hand_friction=1.0,
        object_friction=1.0,
        closing_contact_threshold=0.003,
        closing_displacement=0.002,
        closing_gradient_scale=100.0,
        closing_penetration_cap=0.0015,
    )
    command = _validator_command(
        spec=catalog[0],
        object_root=Path("/tmp/object"),
        args=args,
        device="cuda:0",
        batch_size=32,
    )
    assert command[command.index("--hand-friction") + 1] == "1.0"
    assert command[command.index("--object-friction") + 1] == "1.0"
    assert command[command.index("--closing-displacement") + 1] == "0.002"
    assert command[command.index("--closing-penetration-cap") + 1] == "0.0015"
    assert command[command.index("--preclose-physics-steps") + 1] == "0"


def test_auto_gpu_plan_uses_two_workers_and_batch_32_on_5090_class_gpu():
    inventory = (
        GPUDevice(
            index=0,
            name="NVIDIA GeForce RTX 5090",
            total_memory_mb=32607,
            free_memory_mb=27000,
            utilization_percent=90,
        ),
    )
    plan = _auto_gpu_execution_plan(
        inventory,
        selected_indices=None,
        requested_generation_batch_size=32,
    )
    assert plan.generation_slots == ("cuda:0", "cuda:0")
    assert plan.validation_slots == ("cuda:0", "cuda:0")
    assert plan.generation_batch_size == 32
    assert plan.validation_batch_size == 32


def test_round_robin_selection_preserves_object_diversity(catalog):
    values = []
    for spec in catalog[:3]:
        for index in range(3):
            payload = {
                "active_side": "front",
                "finger_participation": {"actual_count": 2},
            }
            values.append((Path(f"/{spec.slug}_{index}.json"), payload, spec))
    selected = _round_robin_select(values, 5)
    assert [value[2].slug for value in selected[:3]] == sorted(
        spec.slug for spec in catalog[:3]
    )
    assert len({value[2].slug for value in selected}) == 3


def test_completion_requires_exact_strata_and_diversity(catalog):
    pool = []
    for side in SIDES:
        for finger in FINGER_COUNTS:
            for index in range(2):
                spec = catalog[(finger + index) % len(catalog)]
                payload = {
                    "active_side": side,
                    "finger_participation": {"actual_count": finger},
                }
                pool.append((Path(f"/{side}_{finger}_{index}.json"), payload, spec))
    summary = _pool_summary(pool)
    args = argparse.Namespace(
        target_total=20,
        minimum_object_coverage=1,
        minimum_general_coverage=0,
    )
    complete, deficits = _completion_status(summary, args)
    assert all(value == 0 for value in deficits.values())
    # The small synthetic pool does not cover all four primitive shape classes.
    assert complete is False


def test_fr5_gate_requires_one_collision_free_mounted_ik(catalog, fr5_gate):
    sample = Path(
        "artifacts/x2_table_static_success_20260828/raw/back/f1/"
        "cylinder_r025_h100_f1_back_000000.json"
    )
    assert sample.is_file()
    spec = next(value for value in catalog if value.object_id == "cylinder_r025_h100")
    result = fr5_gate.evaluate(_strict_json(sample), spec=spec, record_key=sample.name)
    assert result["status"] == "PASS"
    assert result["failure_reasons"] == []
    assert result["minimum_fr5_link_table_clearance_m"] >= 0.005
    assert result["achieved_x2_table_clearance_lower_bound_m"] >= 0.005
    assert result["achieved_x2_root_table_distance_m"] >= 0.05
    assert any(attempt["converged"] for attempt in result["ik_attempts"])


def test_fr5_gate_rejects_x2_root_too_close_to_table(catalog, fr5_gate):
    sample = Path(
        "artifacts/x2_table_static_success_20260828/raw/back/f1/"
        "cylinder_r025_h100_f1_back_000000.json"
    )
    payload = copy.deepcopy(_strict_json(sample))
    spec = next(value for value in catalog if value.object_id == "cylinder_r025_h100")
    payload["hand_pose"]["translation"][2] = spec.table_plane_z_m + 0.01
    result = fr5_gate.evaluate(payload, spec=spec, record_key="too_close")
    assert result["status"] == "FAIL"
    assert result["failure_reasons"] == ["X2_ROOT_TABLE_DISTANCE_FAILED"]
    assert result["ik_attempts"] == []


def test_fr5_gate_rejects_reachable_but_table_colliding_arm(catalog, fr5_gate):
    sample = Path(
        "artifacts/x2_table_static_success_20260828/raw/front/f2/"
        "cylinder_r025_h100_f2_front_000000.json"
    )
    spec = next(value for value in catalog if value.object_id == "cylinder_r025_h100")
    result = fr5_gate.evaluate(_strict_json(sample), spec=spec, record_key=sample.name)
    converged = [attempt for attempt in result["ik_attempts"] if attempt["converged"]]
    assert result["status"] == "FAIL"
    assert result["failure_reasons"] == ["FR5_TABLE_COLLISION"]
    assert converged
    assert all(attempt["fr5_table_collision_links"] for attempt in converged)
