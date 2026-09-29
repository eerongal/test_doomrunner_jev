from __future__ import annotations

from dataclasses import dataclass
from typing import Any


HOSTILE_CATEGORIES = {"monster", "player"}
TARGET_ALIGNMENT_TOLERANCE_DEGREES = 5.0
TARGET_CONTEXT_FIELDS = (
    "id",
    "name",
    "category",
    "distance",
    "relative_bearing_degrees",
)


def is_hostile(item: dict[str, Any]) -> bool:
    return str(item.get("category", "")).lower() in HOSTILE_CATEGORIES


@dataclass
class TargetLock:
    max_missed_decisions: int = 2
    _target_id: int | None = None
    _last_target: dict[str, Any] | None = None
    _missed_decisions: int = 0

    def reset(self) -> None:
        self._target_id = None
        self._last_target = None
        self._missed_decisions = 0

    def update(self, visible_objects: list[dict[str, Any]]) -> dict[str, Any] | None:
        hostiles = [item for item in visible_objects if is_hostile(item)]
        locked = next((item for item in hostiles if item.get("id") == self._target_id), None)
        if locked is not None:
            return self._remember(locked)
        if self._last_target is not None and self._missed_decisions < self.max_missed_decisions:
            self._missed_decisions += 1
            return {**self._last_target, "visible": False, "missed_decisions": self._missed_decisions}
        if not hostiles:
            self.reset()
            return None
        nearest = min(hostiles, key=lambda item: float(item.get("distance", float("inf"))))
        return self._remember(nearest)

    def _remember(self, target: dict[str, Any]) -> dict[str, Any]:
        self._target_id = int(target["id"])
        compact = {name: target[name] for name in TARGET_CONTEXT_FIELDS if name in target}
        compact["turn_hint"] = _turn_hint(float(target["relative_bearing_degrees"]))
        self._last_target = {**compact, "visible": True, "missed_decisions": 0}
        self._missed_decisions = 0
        return dict(self._last_target)


def _turn_hint(bearing: float) -> str:
    if bearing < -TARGET_ALIGNMENT_TOLERANCE_DEGREES:
        return "right"
    if bearing > TARGET_ALIGNMENT_TOLERANCE_DEGREES:
        return "left"
    return "aligned"
