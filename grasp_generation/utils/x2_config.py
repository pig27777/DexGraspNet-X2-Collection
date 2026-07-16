"""Strict configuration primitives for the X2 generic mesh generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_X2_MESH_CONFIG_PATH = PROJECT_ROOT / "configs" / "x2_mesh_grasp.yaml"


class X2ConfigurationError(RuntimeError):
    """Raised when X2 hand calibration or mesh-generator settings are invalid."""


@dataclass(frozen=True)
class X2Config:
    """Loaded YAML with strict lookup and project-relative path resolution."""

    data: dict[str, Any]
    path: Path
    project_root: Path = PROJECT_ROOT

    def require(self, dotted_key: str) -> Any:
        value: Any = self.data
        traversed: list[str] = []
        for key in dotted_key.split("."):
            traversed.append(key)
            if not isinstance(value, dict) or key not in value:
                raise X2ConfigurationError(
                    f"Missing required key '{dotted_key}' at "
                    f"'{'.'.join(traversed)}' in {self.path}"
                )
            value = value[key]
        if value is None or value == "":
            raise X2ConfigurationError(
                f"Required key '{dotted_key}' is empty in {self.path}"
            )
        return value

    def resolve_path(self, value: str | Path, *, must_exist: bool = False) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        path = path.resolve()
        if must_exist and not path.exists():
            raise X2ConfigurationError(f"Configured path does not exist: {path}")
        return path

    def configured_path(self, dotted_key: str, *, must_exist: bool = False) -> Path:
        return self.resolve_path(self.require(dotted_key), must_exist=must_exist)


def _numeric_vector(config: X2Config, key: str, length: int) -> tuple[float, ...]:
    value = config.require(key)
    if not isinstance(value, list) or len(value) != length:
        raise X2ConfigurationError(f"{key} must be a list of length {length}")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise X2ConfigurationError(f"{key} contains a non-numeric value") from exc
    if not all(item == item and abs(item) != float("inf") for item in result):
        raise X2ConfigurationError(f"{key} contains NaN or infinity")
    return result


def _validate_x2_mesh_config(
    config: X2Config, *, require_contact_candidates: bool = True
) -> None:
    if int(config.require("schema_version")) != 1:
        raise X2ConfigurationError("Only schema_version=1 is supported")
    if str(config.require("pipeline_revision")) != "x2_mesh_grasp_unselected_finger_side_v6":
        raise X2ConfigurationError(
            "pipeline_revision must be x2_mesh_grasp_unselected_finger_side_v6"
        )

    dense_hand_samples = config.require(
        "generation.dense_hand_surface_samples_per_set"
    )
    dense_object_samples = config.require(
        "generation.dense_object_surface_samples"
    )
    dense_threshold = float(
        config.require("generation.dense_hand_object_penetration_threshold")
    )
    if (
        not isinstance(dense_hand_samples, int)
        or isinstance(dense_hand_samples, bool)
        or dense_hand_samples != 256
        or not isinstance(dense_object_samples, int)
        or isinstance(dense_object_samples, bool)
        or dense_object_samples != 8192
        or dense_threshold != 0.001
    ):
        raise X2ConfigurationError(
            "Formal v6 requires exactly 256 dense hand samples per set, "
            "8192 dense object samples, and a 0.001 m penetration threshold"
        )

    config.configured_path("robot.usd_path", must_exist=True)
    config.configured_path(
        "contact_candidates.path", must_exist=require_contact_candidates
    )

    actuator_names = tuple(str(v) for v in config.require("robot.actuator_names"))
    thumb_names = tuple(str(v) for v in config.require("robot.thumb_actuators"))
    joint_names = tuple(str(v) for v in config.require("robot.full_joint_names"))
    if len(actuator_names) != 12 or len(set(actuator_names)) != 12:
        raise X2ConfigurationError("robot.actuator_names must contain 12 unique names")
    if len(thumb_names) != 4 or not set(thumb_names) <= set(actuator_names):
        raise X2ConfigurationError("robot.thumb_actuators must identify four actuators")
    if len(joint_names) != 16 or len(set(joint_names)) != 16:
        raise X2ConfigurationError("robot.full_joint_names must contain 16 unique names")

    active_names = tuple(str(v) for v in config.require("robot.active_non_thumb_actuators"))
    if tuple(actuator_names) != active_names + thumb_names:
        raise X2ConfigurationError(
            "active_non_thumb_actuators followed by thumb_actuators must equal actuator_names"
        )
    coupling = config.require("robot.passive_joint_coupling")
    if not isinstance(coupling, dict) or len(coupling) != 4:
        raise X2ConfigurationError("Exactly four passive follower joints are required")

    limits = config.require("robot.actuator_limits")
    if not isinstance(limits, dict) or set(limits) != set(actuator_names):
        raise X2ConfigurationError("actuator_limits must define all 12 actuators")
    numeric_limits: dict[str, tuple[float, float]] = {}
    for name in actuator_names:
        bounds = limits[name]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise X2ConfigurationError(f"Invalid actuator limits for {name}")
        lower, upper = float(bounds[0]), float(bounds[1])
        if not lower < upper:
            raise X2ConfigurationError(f"Invalid actuator limits for {name}")
        numeric_limits[name] = (lower, upper)

    fixed_thumb = config.require("robot.fixed_thumb_position")
    if not isinstance(fixed_thumb, dict) or set(fixed_thumb) != set(thumb_names):
        raise X2ConfigurationError("fixed_thumb_position must define the four thumb actuators")
    canonical = _numeric_vector(config, "initialization.canonical_open_actuator", 12)
    for name, value in zip(actuator_names, canonical):
        lower, upper = numeric_limits[name]
        if not lower <= value <= upper:
            raise X2ConfigurationError(f"Canonical open value for {name} is outside its limits")

    palm_path = str(config.require("robot.palm_link_path"))
    normals: dict[str, tuple[float, ...]] = {}
    for side in ("front", "back"):
        if str(config.require(f"palm_sides.{side}.frame_path")) != palm_path:
            raise X2ConfigurationError(f"{side}_grasp_frame must use the shared palm/root FK")
        _numeric_vector(config, f"palm_sides.{side}.center_local", 3)
        normal = _numeric_vector(config, f"palm_sides.{side}.normal_local", 3)
        if abs(sum(v * v for v in normal) - 1.0) > 1.0e-8:
            raise X2ConfigurationError(f"palm_sides.{side}.normal_local must be unit length")
        frame = config.require(f"palm_sides.{side}.grasp_frame")
        if not isinstance(frame, list) or len(frame) != 3 or any(
            not isinstance(row, list) or len(row) != 3 for row in frame
        ):
            raise X2ConfigurationError(f"palm_sides.{side}.grasp_frame must be 3x3")
        normals[side] = normal
    if sum(a * b for a, b in zip(normals["front"], normals["back"])) > -0.999:
        raise X2ConfigurationError("front/back palm normals must be antiparallel")

    if int(config.require("generation.n_contact")) <= 0:
        raise X2ConfigurationError("generation.n_contact must be positive")
    for key, expected in (
        ("contact_candidates.expected_finger_markers_per_finger", 47),
        ("contact_candidates.expected_palm_markers_per_side", 41),
        ("contact_candidates.expected_thumb_markers", 17),
    ):
        if int(config.require(key)) != expected:
            raise X2ConfigurationError(f"{key} must remain {expected}")
    for key in (
        "contact_candidates.marker_radius",
        "contact_candidates.finger_keypoint_projection_tolerance",
        "contact_candidates.palm_keypoint_projection_tolerance",
        "contact_candidates.thumb_keypoint_projection_tolerance",
        "contact_candidates.palm_normal_min_dot",
        "contact_candidates.palm_mirror_tolerance",
        "contact_candidates.palm_center_tolerance",
        "contact_candidates.finger_joint_clearance",
        "contact_candidates.duplicate_tolerance",
    ):
        value = float(config.require(key))
        if not value == value or abs(value) == float("inf") or value <= 0.0:
            raise X2ConfigurationError(f"{key} must be finite and positive")
    threshold = float(config.require("contact_candidates.normal_side_threshold"))
    if not 0.0 < threshold < 1.0:
        raise X2ConfigurationError("normal_side_threshold must lie in (0,1)")
    if not 0.0 <= float(config.require("optimization.switch_possibility")) <= 1.0:
        raise X2ConfigurationError("switch_possibility must lie in [0,1]")
    opposite_margin = float(
        config.require("optimization.unselected_finger_opposite_flex.margin")
    )
    opposite_scale = float(
        config.require(
            "optimization.unselected_finger_opposite_flex.displacement_scale"
        )
    )
    if not opposite_margin == opposite_margin or abs(opposite_margin) == float("inf"):
        raise X2ConfigurationError("unselected finger margin must be finite")
    if opposite_margin < 0.0:
        raise X2ConfigurationError("unselected finger margin must be non-negative")
    if not opposite_scale == opposite_scale or abs(opposite_scale) == float("inf"):
        raise X2ConfigurationError("unselected finger displacement scale must be finite")
    if opposite_scale <= 0.0:
        raise X2ConfigurationError("unselected finger displacement scale must be positive")

    self_collision = config.require("self_collision")
    if not isinstance(self_collision, dict):
        raise X2ConfigurationError("self_collision must be a mapping")
    for key in (
        "capsule_weight",
        "hull_weight",
        "broadphase_margin",
        "clearance_margin",
        "smoothness",
        "feasibility_threshold",
    ):
        value = float(config.require(f"self_collision.{key}"))
        if not value == value or abs(value) == float("inf") or value <= 0.0:
            raise X2ConfigurationError(f"self_collision.{key} must be finite and positive")
    sample_count = int(config.require("self_collision.surface_samples_per_link"))
    if sample_count <= 0:
        raise X2ConfigurationError(
            "self_collision.surface_samples_per_link must be positive"
        )
    overrides = config.require("self_collision.pair_weight_overrides")
    if not isinstance(overrides, dict) or set(overrides) != {"thumb_index"}:
        raise X2ConfigurationError(
            "self_collision.pair_weight_overrides must define only thumb_index"
        )
    thumb_index_weight = float(overrides["thumb_index"])
    if (
        not thumb_index_weight == thumb_index_weight
        or abs(thumb_index_weight) == float("inf")
        or thumb_index_weight <= 0.0
    ):
        raise X2ConfigurationError(
            "self_collision.pair_weight_overrides.thumb_index must be finite and positive"
        )
    protection = config.require("self_collision.feasibility_protection")
    if not isinstance(protection, dict) or not isinstance(protection.get("enabled"), bool):
        raise X2ConfigurationError(
            "self_collision.feasibility_protection.enabled must be boolean"
        )
    for key in ("hard_threshold", "maximum_allowed_increase"):
        value = float(config.require(f"self_collision.feasibility_protection.{key}"))
        if not value == value or abs(value) == float("inf") or value <= 0.0:
            raise X2ConfigurationError(
                f"self_collision.feasibility_protection.{key} must be finite and positive"
            )

    configured_links = {str(config.require("robot.palm_link_name"))}
    finger_links = config.require("robot.finger_links")
    if not isinstance(finger_links, dict):
        raise X2ConfigurationError("robot.finger_links must be a mapping")
    configured_links.update(
        str(link) for links in finger_links.values() for link in links
    )
    configured_links.update(str(link) for link in config.require("robot.thumb_links"))
    for key in (
        "robot.self_collision_proxy_exclusions",
        "robot.self_collision_hull_exclusions",
    ):
        exclusions = config.require(key)
        if not isinstance(exclusions, list):
            raise X2ConfigurationError(f"{key} must be a list")
        seen: set[frozenset[str]] = set()
        for entry in exclusions:
            if not isinstance(entry, dict):
                raise X2ConfigurationError(f"Every {key} entry must be a mapping")
            links = entry.get("links")
            reason = entry.get("reason")
            if (
                not isinstance(links, list)
                or len(links) != 2
                or len(set(str(link) for link in links)) != 2
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                raise X2ConfigurationError(f"Invalid {key} entry: {entry!r}")
            pair = frozenset(str(link) for link in links)
            if not pair <= configured_links or pair in seen:
                raise X2ConfigurationError(f"Unknown or duplicate {key} entry: {entry!r}")
            seen.add(pair)

    weights = config.require("optimization.weights")
    if (
        not isinstance(weights, dict)
        or "E_unselected_opposite_flex" not in weights
        or float(weights["E_unselected_opposite_flex"]) < 0.0
    ):
        raise X2ConfigurationError(
            "optimization.weights.E_unselected_opposite_flex must be non-negative"
        )


def load_x2_mesh_config(
    path: str | Path | None = None, *, require_contact_candidates: bool = True
) -> X2Config:
    """Load the hand-only X2 mesh grasp configuration."""

    config_path = Path(path).expanduser() if path is not None else DEFAULT_X2_MESH_CONFIG_PATH
    if not config_path.is_absolute():
        cwd_candidate = (Path.cwd() / config_path).resolve()
        config_path = cwd_candidate if cwd_candidate.is_file() else (PROJECT_ROOT / config_path).resolve()
    else:
        config_path = config_path.resolve()
    if not config_path.is_file():
        raise X2ConfigurationError(f"Configuration file does not exist: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise X2ConfigurationError(f"Cannot load {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise X2ConfigurationError("Configuration root must be a mapping")
    config = X2Config(data=data, path=config_path)
    _validate_x2_mesh_config(
        config, require_contact_candidates=require_contact_candidates
    )
    return config


__all__ = [
    "DEFAULT_X2_MESH_CONFIG_PATH",
    "PROJECT_ROOT",
    "X2Config",
    "X2ConfigurationError",
    "load_x2_mesh_config",
]
