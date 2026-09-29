from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from doomrunner.actions import GameAction


BLOCKED_AFTER_SECONDS = 0.45
MIN_FORWARD_PROGRESS = 1.0
WALL_FOLLOWING_AFTER_SECONDS = 1.0
NEAR_WALL_MEDIAN_DEPTH = 18.0


@dataclass
class ProgressFeedback:
    _last_position: list[float] | None = None
    _last_observed_at: float | None = None
    _blocked_seconds: float = 0.0
    _wall_following_seconds: float = 0.0
    _wall_following_direction: str | None = None
    _near_wall_depth: float | None = None

    def reset(self) -> None:
        self._last_position = None
        self._last_observed_at = None
        self._blocked_seconds = 0.0
        self._wall_following_seconds = 0.0
        self._wall_following_direction = None
        self._near_wall_depth = None

    def observe(self, observation: dict[str, Any], applied_action: GameAction, now: float) -> None:
        position = observation.get("player", {}).get("position")
        if position is None:
            return
        distance = displacement(self._last_position, position)
        elapsed = elapsed_seconds(self._last_observed_at, now)
        if applied_action.movement == "forward" and distance is not None and distance < MIN_FORWARD_PROGRESS:
            self._blocked_seconds += elapsed
        elif distance is not None and distance >= MIN_FORWARD_PROGRESS:
            self._blocked_seconds = 0.0
        else:
            self._blocked_seconds = 0.0
        self._observe_wall_following(observation, applied_action, elapsed)
        self._last_position = [float(value) for value in position]
        self._last_observed_at = now

    def _observe_wall_following(self, observation: dict[str, Any], action: GameAction, elapsed: float) -> None:
        direction = turn_direction(action.turn)
        nearby_wall_depth = nearest_side_depth(observation.get("depth_grid"))
        if (
            direction is None
            or action.movement != "forward"
            or nearby_wall_depth is None
            or nearby_wall_depth > NEAR_WALL_MEDIAN_DEPTH
        ):
            self._wall_following_seconds = 0.0
            self._wall_following_direction = None
            self._near_wall_depth = None
            return
        if direction != self._wall_following_direction:
            self._wall_following_seconds = 0.0
        self._wall_following_seconds += elapsed
        self._wall_following_direction = direction
        self._near_wall_depth = nearby_wall_depth

    def summary(self) -> dict[str, Any] | None:
        if self._blocked_seconds < BLOCKED_AFTER_SECONDS:
            return self._wall_following_summary()
        return {
            "status": "blocked",
            "command": "forward",
            "blocked_seconds": round(self._blocked_seconds, 2),
            "obstruction_direction": "forward",
            "suggested_recovery": "backtrack_and_turn",
        }

    def _wall_following_summary(self) -> dict[str, Any] | None:
        if self._wall_following_seconds < WALL_FOLLOWING_AFTER_SECONDS:
            return None
        return {
            "status": "wall_following",
            "command": "forward",
            "turn_direction": self._wall_following_direction,
            "near_wall_depth": self._near_wall_depth,
            "wall_following_seconds": round(self._wall_following_seconds, 2),
            "suggested_recovery": "use an explicit hard turn or backtrack away from the nearby wall",
        }


def displacement(previous: list[float] | None, current: list[float]) -> float | None:
    if previous is None or len(previous) != len(current):
        return None
    return math.sqrt(sum((float(after) - float(before)) ** 2 for before, after in zip(previous, current)))


def elapsed_seconds(previous: float | None, current: float) -> float:
    return 0.0 if previous is None else max(0.0, current - previous)


def turn_direction(turn: str) -> str | None:
    if turn.endswith("_left"):
        return "left"
    if turn.endswith("_right"):
        return "right"
    return None


def side_depth(depth_grid: dict[str, Any] | None, direction: str | None) -> float | None:
    if depth_grid is None or direction is None:
        return None
    rows = depth_grid.get("values", [])
    if not rows:
        return None
    sampled_rows = rows[:-1] or rows
    columns = min(len(row) for row in sampled_rows)
    if columns == 0:
        return None
    midpoint = max(1, columns // 2)
    indexes = range(0, midpoint) if direction == "left" else range(midpoint, columns)
    values = [float(row[index]) for row in sampled_rows for index in indexes]
    if not values:
        return None
    values.sort()
    return values[len(values) // 2]


def nearest_side_depth(depth_grid: dict[str, Any] | None) -> float | None:
    depths = [side_depth(depth_grid, direction) for direction in ("left", "right")]
    available_depths = [depth for depth in depths if depth is not None]
    return min(available_depths) if available_depths else None
