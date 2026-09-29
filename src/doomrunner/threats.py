from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from doomrunner.targeting import is_hostile


THREAT_MEMORY_SECONDS = 1.5


@dataclass
class RecentThreatMemory:
    retention_seconds: float = THREAT_MEMORY_SECONDS
    _threats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def reset(self) -> None:
        self._threats.clear()

    def observe(self, observation: dict[str, Any], now: float) -> None:
        player = observation.get("player", {})
        for hostile in (item for item in observation.get("visible_objects", []) if is_hostile(item)):
            self._threats[str(hostile["id"])] = compact_threat(hostile, player, now)
        self._expire(now)

    def summary(self, now: float) -> list[dict[str, Any]]:
        self._expire(now)
        return [
            {
                "id": threat["id"],
                "name": threat["name"],
                "last_distance": threat["distance"],
                "last_relative_bearing_degrees": threat["relative_bearing_degrees"],
                "age_seconds": round(now - threat["seen_at"], 2),
            }
            for threat in sorted(self._threats.values(), key=lambda item: item["seen_at"], reverse=True)
        ]

    def target(self, object_id: str, observation: dict[str, Any], now: float) -> dict[str, Any] | None:
        self._expire(now)
        threat = self._threats.get(object_id)
        if threat is None:
            return None
        return {
            **threat,
            "visible": False,
            "relative_bearing_degrees": relative_bearing(threat, observation.get("player", {})),
        }

    def _expire(self, now: float) -> None:
        self._threats = {
            object_id: threat
            for object_id, threat in self._threats.items()
            if now - float(threat["seen_at"]) <= self.retention_seconds
        }


def compact_threat(hostile: dict[str, Any], player: dict[str, Any], now: float) -> dict[str, Any]:
    result = {
        name: hostile[name]
        for name in ("id", "name", "distance", "relative_bearing_degrees", "world_position")
        if name in hostile
    }
    result["seen_at"] = now
    position = player.get("position")
    if position is not None and hostile.get("world_position") is not None:
        result["world_bearing_degrees"] = world_bearing(position, hostile["world_position"])
    return result


def relative_bearing(threat: dict[str, Any], player: dict[str, Any]) -> float:
    if player.get("heading_degrees") is None or player.get("position") is None or threat.get("world_position") is None:
        return float(threat["relative_bearing_degrees"])
    return normalize_angle(world_bearing(player["position"], threat["world_position"]) - float(player["heading_degrees"]))


def world_bearing(position: list[float], target_position: list[float]) -> float:
    return math.degrees(math.atan2(float(target_position[1]) - float(position[1]), float(target_position[0]) - float(position[0])))


def normalize_angle(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0
