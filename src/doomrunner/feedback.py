from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from doomrunner.actions import GameAction


AIM_TREND_EPSILON_DEGREES = 2.0
CIRCLING_MIN_TURN_STREAK = 4
CIRCLING_MIN_HEADING_CHANGE = 90.0
STUCK_MIN_PROGRESS_UNITS = 5.0
STUCK_MIN_DECISIONS = 2


@dataclass
class ControlFeedback:
    previous_action: GameAction = field(default_factory=GameAction)
    action_started_tic: int | None = None
    turn_streak: int = 0
    accumulated_heading_change: float = 0.0
    _turn_direction: str = "hold"
    _last_heading: float | None = None
    _last_target_id: int | None = None
    _last_target_bearing: float | None = None
    _action_at_last_snapshot: GameAction | None = None
    _last_position: tuple[float, float] | None = None
    _stuck_streak: int = 0

    def reset(self) -> None:
        self.previous_action = GameAction()
        self.action_started_tic = None
        self.turn_streak = 0
        self.accumulated_heading_change = 0.0
        self._turn_direction = "hold"
        self._last_heading = None
        self._last_target_id = None
        self._last_target_bearing = None
        self._action_at_last_snapshot = None
        self._last_position = None
        self._stuck_streak = 0

    def record_action(self, action: GameAction, observation: dict[str, Any] | None) -> None:
        direction = _turn_direction(action.turn)
        if direction == "hold":
            self.turn_streak = 0
            self.accumulated_heading_change = 0.0
        elif direction == self._turn_direction:
            self.turn_streak += 1
        else:
            self.turn_streak = 1
            self.accumulated_heading_change = 0.0
        self.previous_action = action
        self._turn_direction = direction
        if observation is None:
            return
        self.action_started_tic = int(observation["tic"])
        self._last_heading = float(observation["player"]["heading_degrees"])

    def snapshot(
        self,
        observation: dict[str, Any],
        target: dict[str, Any] | None,
    ) -> dict[str, Any]:
        tic = int(observation["tic"])
        heading = float(observation["player"]["heading_degrees"])
        self._record_heading_change(heading)
        target_feedback = self._target_feedback(target)
        held_tics = None if self.action_started_tic is None else max(0, tic - self.action_started_tic)
        evaluated_action = self._action_at_last_snapshot
        movement_feedback = self._movement_feedback(observation, evaluated_action)
        self._action_at_last_snapshot = self.previous_action
        return {
            "current_action": self.previous_action.to_dict(),
            "current_action_held_tics": held_tics,
            "evaluated_action": None if evaluated_action is None else evaluated_action.to_dict(),
            "turn_streak": self.turn_streak,
            "heading_change_during_streak": round(self.accumulated_heading_change, 1),
            "possible_circling": (
                self.turn_streak >= CIRCLING_MIN_TURN_STREAK
                and abs(self.accumulated_heading_change) >= CIRCLING_MIN_HEADING_CHANGE
            ),
            **movement_feedback,
            **target_feedback,
        }

    def _record_heading_change(self, heading: float) -> None:
        if self._last_heading is not None and self._turn_direction != "hold":
            self.accumulated_heading_change += _angle_delta(heading, self._last_heading)
        self._last_heading = heading

    def _target_feedback(self, target: dict[str, Any] | None) -> dict[str, Any]:
        if target is None:
            self._last_target_id = None
            self._last_target_bearing = None
            return {"aim_trend": "no_target", "target_bearing_change": None}
        target_id = int(target["id"])
        bearing = float(target["relative_bearing_degrees"])
        visible = bool(target.get("visible", True))
        if not visible:
            return {"aim_trend": "target_temporarily_lost", "target_bearing_change": None}
        trend, change = self._compare_bearing(target_id, bearing)
        self._last_target_id = target_id
        self._last_target_bearing = bearing
        return {"aim_trend": trend, "target_bearing_change": change}

    def _compare_bearing(self, target_id: int, bearing: float) -> tuple[str, float | None]:
        if target_id != self._last_target_id or self._last_target_bearing is None:
            return "new_target", None
        change = round(abs(bearing) - abs(self._last_target_bearing), 1)
        if change < -AIM_TREND_EPSILON_DEGREES:
            return "improving", change
        if change > AIM_TREND_EPSILON_DEGREES:
            return "worsening", change
        return "stable", change

    def _movement_feedback(
        self,
        observation: dict[str, Any],
        evaluated_action: GameAction | None,
    ) -> dict[str, Any]:
        position = observation.get("player", {}).get("position")
        if not position or len(position) < 2:
            return {"distance_moved": None, "stuck_streak": self._stuck_streak, "possible_stuck": False}
        current = (float(position[0]), float(position[1]))
        distance = None if self._last_position is None else math.dist(current, self._last_position)
        self._last_position = current
        expected_movement = evaluated_action is not None and evaluated_action.movement != "stop"
        if expected_movement and distance is not None and distance < STUCK_MIN_PROGRESS_UNITS:
            self._stuck_streak += 1
        if distance is not None and distance >= STUCK_MIN_PROGRESS_UNITS:
            self._stuck_streak = 0
        return {
            "distance_moved": None if distance is None else round(distance, 1),
            "stuck_streak": self._stuck_streak,
            "possible_stuck": self._stuck_streak >= STUCK_MIN_DECISIONS,
        }


def _turn_direction(turn: str) -> str:
    if turn.endswith("left"):
        return "left"
    if turn.endswith("right"):
        return "right"
    return "hold"


def _angle_delta(current: float, previous: float) -> float:
    return (current - previous + 180.0) % 360.0 - 180.0
