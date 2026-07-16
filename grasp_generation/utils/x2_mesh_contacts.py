"""Side-conditioned contact candidates for generic X2 mesh grasping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


SIDES = ("front", "back")
FINGER_NAMES = ("index", "middle", "ring", "little", "thumb")


class GenericContactError(RuntimeError):
    """Raised when authored X2 mesh contact metadata is inconsistent."""


@dataclass(frozen=True)
class GenericContactCandidate:
    """One authored hand-surface candidate in its owning link frame."""

    point_id: str
    link_name: str
    finger_name: str
    region: str
    local_position: tuple[float, float, float]
    local_surface_normal: tuple[float, float, float]
    supported_sides: tuple[str, ...]
    source: str
    enabled: bool = True

    @property
    def position_local(self) -> tuple[float, float, float]:
        """Compatibility alias consumed by the shared differentiable FK."""

        return self.local_position

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenericContactCandidate":
        required = {
            "point_id",
            "link_name",
            "finger_name",
            "region",
            "local_position",
            "local_surface_normal",
            "supported_sides",
            "source",
        }
        missing = sorted(required - set(value))
        if missing:
            raise GenericContactError(f"Contact candidate is missing {missing}")
        try:
            position = tuple(float(v) for v in value["local_position"])
            normal = tuple(float(v) for v in value["local_surface_normal"])
        except (TypeError, ValueError) as exc:
            raise GenericContactError(f"Invalid position/normal in {value!r}") from exc
        if len(position) != 3 or len(normal) != 3 or not np.isfinite(position + normal).all():
            raise GenericContactError(f"Candidate position/normal must contain three finite values")
        normal_norm = float(np.linalg.norm(normal))
        if abs(normal_norm - 1.0) > 1.0e-5:
            raise GenericContactError(
                f"Candidate {value.get('point_id')} normal is not unit length: {normal_norm}"
            )
        sides = tuple(str(v) for v in value["supported_sides"])
        if not sides or len(set(sides)) != len(sides) or not set(sides) <= set(SIDES):
            raise GenericContactError(f"Invalid supported_sides for {value.get('point_id')}: {sides}")
        return cls(
            point_id=str(value["point_id"]),
            link_name=str(value["link_name"]),
            finger_name=str(value["finger_name"]),
            region=str(value["region"]),
            local_position=position,
            local_surface_normal=normal,
            supported_sides=sides,
            source=str(value["source"]),
            enabled=bool(value.get("enabled", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "link_name": self.link_name,
            "finger_name": self.finger_name,
            "region": self.region,
            "local_position": list(self.local_position),
            "local_surface_normal": list(self.local_surface_normal),
            "supported_sides": list(self.supported_sides),
            "source": self.source,
            "enabled": self.enabled,
        }


def load_generic_contact_candidates(path: str | Path) -> tuple[GenericContactCandidate, ...]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenericContactError(f"Cannot read contact candidates {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise GenericContactError(f"Unsupported contact candidate schema in {path}")
    raw = payload.get("candidates")
    if not isinstance(raw, list):
        raise GenericContactError(f"{path} must contain a candidates list")
    candidates = tuple(GenericContactCandidate.from_dict(item) for item in raw)
    if int(payload.get("candidate_count", -1)) != len(candidates):
        raise GenericContactError(f"candidate_count mismatch in {path}")
    ids = [candidate.point_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise GenericContactError("Contact point IDs must be globally unique")
    required_regions = {
        "front_palm",
        "back_palm",
        "front_finger_surface",
        "back_finger_surface",
        "shared_fingertip",
        "thumb",
    }
    present = {candidate.region for candidate in candidates if candidate.enabled}
    if not required_regions <= present:
        raise GenericContactError(
            f"Contact metadata omits required regions: {sorted(required_regions - present)}"
        )
    return candidates


class GenericDexterousContactPolicy:
    """Uniform unique sampling from candidates authored for one active palm side."""

    def __init__(
        self,
        candidates: Sequence[GenericContactCandidate],
        *,
        active_side: str,
        n_contact: int = 4,
        allow_thumb: bool = True,
        target_finger_count: int | None = None,
        required_finger_names: Sequence[str] | None = None,
    ) -> None:
        if active_side not in SIDES:
            raise GenericContactError(f"active_side must be front or back, got {active_side!r}")
        if not isinstance(n_contact, int) or n_contact <= 0:
            raise GenericContactError("n_contact must be a positive integer")
        self.candidates = tuple(candidates)
        self.active_side = active_side
        self.n_contact = n_contact
        self.allow_thumb = bool(allow_thumb)
        required = tuple(str(value) for value in (required_finger_names or ()))
        if (
            len(required) != len(set(required))
            or not set(required) <= set(FINGER_NAMES)
        ):
            raise GenericContactError(
                "required_finger_names must be unique known X2 finger names"
            )
        if required and target_finger_count is None:
            target_finger_count = len(required)
        if required and len(required) != target_finger_count:
            raise GenericContactError(
                "required_finger_names count must match target_finger_count"
            )
        if target_finger_count is not None and (
            not isinstance(target_finger_count, int)
            or target_finger_count < 1
            or target_finger_count > len(FINGER_NAMES)
        ):
            raise GenericContactError("target_finger_count must be in 1..5")
        if target_finger_count is not None and target_finger_count > n_contact:
            raise GenericContactError(
                "target_finger_count cannot exceed the selected contact count"
            )
        if target_finger_count == 5 and not self.allow_thumb:
            raise GenericContactError("A five-finger selection requires allow_thumb=true")
        self.target_finger_count = target_finger_count
        self.required_finger_names = required
        self.eligible_indices = tuple(
            index
            for index, candidate in enumerate(self.candidates)
            if candidate.enabled
            and active_side in candidate.supported_sides
            and (self.allow_thumb or candidate.finger_name != "thumb")
        )
        if len(self.eligible_indices) < n_contact:
            raise GenericContactError(
                f"Only {len(self.eligible_indices)} candidates support {active_side}; need {n_contact}"
            )
        available_fingers = self._finger_names(self.eligible_indices)
        if (
            self.target_finger_count is not None
            and len(available_fingers) < self.target_finger_count
        ):
            raise GenericContactError(
                f"Only {len(available_fingers)} fingers support {active_side}; "
                f"need {self.target_finger_count}"
            )

    def _finger_names(self, indices: Sequence[int]) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    self.candidates[int(index)].finger_name
                    for index in indices
                    if self.candidates[int(index)].finger_name in FINGER_NAMES
                },
                key=FINGER_NAMES.index,
            )
        )

    def sample(self, rng: np.random.Generator) -> tuple[int, ...]:
        if self.target_finger_count is None:
            chosen = rng.choice(
                self.eligible_indices, size=self.n_contact, replace=False
            )
            result = tuple(int(v) for v in chosen)
        else:
            available_fingers = self._finger_names(self.eligible_indices)
            selected_fingers = self.required_finger_names or tuple(
                str(value)
                for value in rng.choice(
                    available_fingers, size=self.target_finger_count, replace=False
                )
            )
            chosen = []
            for finger_name in selected_fingers:
                candidates = [
                    index
                    for index in self.eligible_indices
                    if self.candidates[index].finger_name == finger_name
                ]
                chosen.append(int(rng.choice(candidates)))
            allowed = [
                index
                for index in self.eligible_indices
                if index not in chosen
                and self.candidates[index].finger_name in (*selected_fingers, "palm")
            ]
            remaining = self.n_contact - len(chosen)
            if len(allowed) < remaining:
                raise GenericContactError(
                    "Not enough unique contacts for the requested finger stratum"
                )
            if remaining:
                chosen.extend(
                    int(value)
                    for value in rng.choice(allowed, size=remaining, replace=False)
                )
            rng.shuffle(chosen)
            result = tuple(chosen)
        self.validate(result)
        return result

    def resample_slot(
        self, indices: Sequence[int], slot: int, rng: np.random.Generator
    ) -> tuple[int, ...]:
        if slot < 0 or slot >= self.n_contact:
            raise GenericContactError(f"Contact slot {slot} is out of range")
        result = [int(v) for v in indices]
        used = {value for index, value in enumerate(result) if index != slot}
        available = []
        for value in self.eligible_indices:
            if value in used:
                continue
            proposed = tuple(
                value if index == slot else current
                for index, current in enumerate(result)
            )
            try:
                self.validate(proposed)
            except GenericContactError:
                continue
            available.append(value)
        if not available:
            return tuple(result)
        result[slot] = int(rng.choice(available))
        output = tuple(result)
        self.validate(output)
        return output

    def validate(self, indices: Sequence[int]) -> None:
        values = tuple(int(v) for v in indices)
        if len(values) != self.n_contact or len(values) != len(set(values)):
            raise GenericContactError("A contact selection must contain unique IDs")
        eligible = set(self.eligible_indices)
        if not set(values) <= eligible:
            raise GenericContactError(
                f"Selection contains a candidate unsupported by active_side={self.active_side}"
            )
        if self.target_finger_count is not None:
            actual_names = self._finger_names(values)
            actual = len(actual_names)
            if actual != self.target_finger_count:
                raise GenericContactError(
                    f"Selection uses {actual} fingers; expected {self.target_finger_count}"
                )
            if self.required_finger_names and set(actual_names) != set(
                self.required_finger_names
            ):
                raise GenericContactError(
                    f"Selection uses fingers {actual_names}; expected "
                    f"{self.required_finger_names}"
                )


def selected_candidates(
    candidates: Sequence[GenericContactCandidate], indices: Iterable[int]
) -> tuple[GenericContactCandidate, ...]:
    return tuple(candidates[int(index)] for index in indices)


__all__ = [
    "GenericContactCandidate",
    "GenericContactError",
    "GenericDexterousContactPolicy",
    "FINGER_NAMES",
    "SIDES",
    "load_generic_contact_candidates",
    "selected_candidates",
]
