from __future__ import annotations

import math
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from doomrunner.actions import GameAction
from doomrunner.targeting import is_hostile


RAW_OBJECT_FIELDS = (
    "id",
    "name",
    "category",
    "distance",
    "relative_bearing_degrees",
    "screen_box",
)
PLAYER_DELTA_FIELDS = ("health", "armor", "weapon_ammo")
COMBAT_DELTA_FIELDS = ("kills", "hits", "hits_taken", "damage_dealt", "damage_taken")
LONGITUDINAL_AXIS = {"backward": -1, "stop": 0, "forward": 1}
LATERAL_AXIS = {"left": -1, "stop": 0, "right": 1}
YAW_AXIS = {"hard_left": -2, "soft_left": -1, "hold": 0, "soft_right": 1, "hard_right": 2}
WEAPON_STEP = {"previous": -1, "keep": 0, "next": 1}
COMPACT_PLAYER_FIELDS = ("health", "armor", "weapon", "weapon_ammo", "attack_ready", "heading_degrees")
COMPACT_COMBAT_FIELDS = ("kills", "hits", "hits_taken", "damage_dealt", "damage_taken", "health_change")
COMPACT_OBJECT_FIELDS = ("id", "name", "category", "distance", "relative_bearing_degrees", "screen_box")
DEPTH_SECTOR_NAMES = ("left", "center", "right")


@dataclass
class RawObservationHistory:
    max_frames: int
    _frames: deque[dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self._frames = deque(maxlen=self.max_frames)

    def reset(self) -> None:
        self._frames.clear()

    def build(self, observation: dict[str, Any], action: GameAction) -> dict[str, Any]:
        self._frames.append(
            {
                "observation": compact_raw_observation(observation),
                "control_state": encode_control_state(action),
            }
        )
        frames = list(self._frames)
        transitions = [observation_transition(older, newer) for older, newer in zip(frames, frames[1:])]
        return {
            "observation_mode": "raw_temporal_deltas",
            "transition_order": "oldest_to_newest",
            "recent_transitions": transitions,
            "current": deepcopy(frames[-1]),
        }


@dataclass
class CompactObservationHistory:
    _previous_observation: dict[str, Any] | None = None

    def reset(self) -> None:
        self._previous_observation = None

    def build(self, observation: dict[str, Any], action: GameAction) -> dict[str, Any]:
        result = compact_measurements(observation, self._previous_observation, encode_control_state(action))
        self._previous_observation = compact_raw_observation(observation)
        return result


def compact_measurements(
    observation: dict[str, Any],
    previous: dict[str, Any] | None,
    control_applied: dict[str, int] | None,
) -> dict[str, Any]:
    player = observation.get("player", {})
    objects = observation.get("visible_objects", [])
    return {
        "observation_mode": "compact_measurements",
        "tic": observation.get("tic"),
        "player": {
            **{name: deepcopy(player[name]) for name in COMPACT_PLAYER_FIELDS if name in player},
            "speed": vector_magnitude(player.get("velocity")),
        },
        "combat": {
            name: deepcopy(observation["combat"][name])
            for name in COMPACT_COMBAT_FIELDS
            if name in observation.get("combat", {})
        },
        "motion_since_previous_decision": compact_motion(previous, observation, control_applied),
        "visible_hostiles": [compact_object(item) for item in objects if is_hostile(item)],
        "visible_resources_and_objects": [compact_object(item) for item in objects if not is_hostile(item)],
        "depth_sectors": summarize_depth_sectors(observation.get("depth_grid")),
    }


def compact_object(item: dict[str, Any]) -> dict[str, Any]:
    return {name: deepcopy(item[name]) for name in COMPACT_OBJECT_FIELDS if name in item}


def compact_motion(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    control_applied: dict[str, int] | None,
) -> dict[str, Any] | None:
    if previous is None:
        return None
    before_player = previous.get("player", {})
    after_player = current.get("player", {})
    position_change = vector_delta(before_player.get("position"), after_player.get("position"))
    result: dict[str, Any] = {
        "elapsed_tics": int(current["tic"]) - int(previous["tic"]),
        "control_applied": deepcopy(control_applied),
        "position_delta": position_change,
        "distance_moved": vector_magnitude(position_change),
    }
    if "heading_degrees" in before_player and "heading_degrees" in after_player:
        result["heading_delta_degrees"] = round(
            angle_delta(after_player["heading_degrees"], before_player["heading_degrees"]),
            1,
        )
    result["combat_delta"] = numeric_deltas(
        previous.get("combat", {}),
        current.get("combat", {}),
        COMBAT_DELTA_FIELDS,
    )
    return result


def summarize_depth_sectors(depth_grid: dict[str, Any] | None) -> dict[str, Any] | None:
    if not depth_grid:
        return None
    rows = depth_grid.get("values", [])
    columns = int(depth_grid.get("columns", 0))
    if not rows or columns <= 0:
        return None
    sampled_rows = rows[:-1] or rows
    sectors = {
        name: depth_statistics(
            [value for row in sampled_rows for value in row[start:end]],
        )
        for name, (start, end) in zip(DEPTH_SECTOR_NAMES, sector_ranges(columns))
    }
    return {
        "encoding": depth_grid.get("encoding", "vizdoom_uint8"),
        "note": "raw depth signal sampled across upper and middle screen rows; compare sectors relatively",
        **sectors,
    }


def sector_ranges(columns: int) -> tuple[tuple[int, int], ...]:
    first = max(1, columns // 3)
    second = max(first + 1, 2 * columns // 3)
    return ((0, first), (first, min(second, columns)), (min(second, columns), columns))


def depth_statistics(values: list[Any]) -> dict[str, float] | None:
    if not values:
        return None
    numeric = [float(value) for value in values]
    return {
        "minimum": round(min(numeric), 1),
        "median": round(float(median(numeric)), 1),
        "maximum": round(max(numeric), 1),
    }


def vector_magnitude(value: Any) -> float | None:
    if value is None:
        return None
    return round(math.sqrt(sum(float(component) ** 2 for component in value)), 1)


def compact_raw_observation(observation: dict[str, Any]) -> dict[str, Any]:
    result = {
        name: deepcopy(observation[name])
        for name in ("tic", "episode_time_tics", "player", "combat")
        if name in observation
    }
    result["visible_objects"] = [
        {name: item[name] for name in RAW_OBJECT_FIELDS if name in item}
        for item in observation.get("visible_objects", [])
    ]
    if observation.get("depth_grid") is not None:
        result["depth_grid"] = compact_depth_grid(observation["depth_grid"])
    return result


def compact_depth_grid(depth_grid: dict[str, Any]) -> dict[str, Any]:
    return {
        "shape": [depth_grid["rows"], depth_grid["columns"]],
        "values": [value for row in depth_grid["values"] for value in row],
    }


def observation_transition(older: dict[str, Any], newer: dict[str, Any]) -> dict[str, Any]:
    before = older["observation"]
    after = newer["observation"]
    return {
        "elapsed_tics": int(after["tic"]) - int(before["tic"]),
        "control_applied": deepcopy(older["control_state"]),
        "player_delta": player_delta(before.get("player", {}), after.get("player", {})),
        "combat_delta": numeric_deltas(before.get("combat", {}), after.get("combat", {}), COMBAT_DELTA_FIELDS),
        "object_changes": object_changes(before.get("visible_objects", []), after.get("visible_objects", [])),
        "depth_delta": depth_delta(before.get("depth_grid"), after.get("depth_grid")),
    }


def encode_control_state(action: GameAction) -> dict[str, int]:
    return {
        "longitudinal_axis": LONGITUDINAL_AXIS[action.movement],
        "lateral_axis": LATERAL_AXIS[action.strafe],
        "yaw_axis": YAW_AXIS[action.turn],
        "trigger": int(action.fire),
        "interact": int(action.use),
        "weapon_step": WEAPON_STEP.get(action.weapon, 0),
        "weapon_select": weapon_selection(action.weapon),
    }


def weapon_selection(action: str) -> int:
    if not action.startswith("select_"):
        return 0
    try:
        return int(action.removeprefix("select_"))
    except ValueError:
        return 0


def player_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result = numeric_deltas(before, after, PLAYER_DELTA_FIELDS)
    result["position"] = vector_delta(before.get("position"), after.get("position"))
    result["velocity"] = vector_delta(before.get("velocity"), after.get("velocity"))
    if "heading_degrees" in before and "heading_degrees" in after:
        result["heading_degrees"] = round(angle_delta(after["heading_degrees"], before["heading_degrees"]), 1)
    if before.get("weapon") != after.get("weapon"):
        result["weapon"] = {"from": before.get("weapon"), "to": after.get("weapon")}
    return result


def numeric_deltas(before: dict[str, Any], after: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {
        name: round(float(after[name]) - float(before[name]), 1)
        for name in fields
        if name in before and name in after and after[name] != before[name]
    }


def vector_delta(before: Any, after: Any) -> list[float] | None:
    if before is None or after is None or len(before) != len(after):
        return None
    return [round(float(current) - float(previous), 1) for previous, current in zip(before, after)]


def object_changes(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    previous_by_id = {item["id"]: item for item in before}
    current_by_id = {item["id"]: item for item in after}
    tracked = []
    for object_id in previous_by_id.keys() & current_by_id.keys():
        previous = previous_by_id[object_id]
        current = current_by_id[object_id]
        tracked.append(
            {
                "id": object_id,
                "distance": round(float(current["distance"]) - float(previous["distance"]), 1),
                "relative_bearing_degrees": round(
                    angle_delta(current["relative_bearing_degrees"], previous["relative_bearing_degrees"]),
                    1,
                ),
            }
        )
    return {
        "appeared": [deepcopy(current_by_id[object_id]) for object_id in current_by_id.keys() - previous_by_id.keys()],
        "disappeared_ids": list(previous_by_id.keys() - current_by_id.keys()),
        "tracked_deltas": tracked,
    }


def depth_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> list[int] | None:
    if before is None or after is None or before.get("shape") != after.get("shape"):
        return None
    return [int(current) - int(previous) for previous, current in zip(before["values"], after["values"])]


def angle_delta(current: float, previous: float) -> float:
    return (float(current) - float(previous) + 180.0) % 360.0 - 180.0
