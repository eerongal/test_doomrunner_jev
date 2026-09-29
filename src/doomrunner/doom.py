from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from doomrunner.actions import GameAction
from doomrunner.config import DoomConfig
from doomrunner.targeting import is_hostile

LOGGER = logging.getLogger(__name__)


class DoomSession:
    TICS_PER_SECOND = 35.0
    USE_PULSE_INTERVAL_TICS = 6

    def __init__(self, config: DoomConfig, max_visible_objects: int, always_run: bool) -> None:
        import vizdoom as vzd

        self.vzd = vzd
        self.config = config
        self.max_visible_objects = max_visible_objects
        self.always_run = always_run
        self.game = vzd.DoomGame()
        self.buttons = [
            vzd.Button.MOVE_FORWARD,
            vzd.Button.MOVE_BACKWARD,
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.TURN_LEFT_RIGHT_DELTA,
            vzd.Button.ATTACK,
            vzd.Button.USE,
            vzd.Button.SELECT_NEXT_WEAPON,
            vzd.Button.SELECT_PREV_WEAPON,
            vzd.Button.SPEED,
            *[getattr(vzd.Button, f"SELECT_WEAPON{number}") for number in range(1, 10)],
        ]
        self.variables = [
            vzd.GameVariable.HEALTH,
            vzd.GameVariable.ARMOR,
            vzd.GameVariable.SELECTED_WEAPON,
            vzd.GameVariable.SELECTED_WEAPON_AMMO,
            vzd.GameVariable.ATTACK_READY,
            vzd.GameVariable.KILLCOUNT,
            vzd.GameVariable.HITCOUNT,
            vzd.GameVariable.HITS_TAKEN,
            vzd.GameVariable.DAMAGECOUNT,
            vzd.GameVariable.DAMAGE_TAKEN,
            vzd.GameVariable.POSITION_X,
            vzd.GameVariable.POSITION_Y,
            vzd.GameVariable.POSITION_Z,
            vzd.GameVariable.ANGLE,
            vzd.GameVariable.PITCH,
            vzd.GameVariable.VELOCITY_X,
            vzd.GameVariable.VELOCITY_Y,
            *[getattr(vzd.GameVariable, f"AMMO{number}") for number in range(1, 5)],
            *[getattr(vzd.GameVariable, f"WEAPON{number}") for number in range(1, 10)],
        ]
        self._last_health: float | None = None
        self._last_kills: float | None = None
        self._buffer_errors: set[str] = set()
        self._state_unavailable_reported = False
        self._screen_buffer: Any | None = None
        self._use_pulse_countdown = 0

    def start(self) -> None:
        scenario = Path(self.vzd.scenarios_path) / f"{self.config.scenario}.cfg"
        if not scenario.is_file():
            raise ValueError(f"ViZDoom scenario does not exist: {scenario}")
        self.game.load_config(str(scenario))
        self.game.set_doom_game_path(str(self.config.wad_path))
        self.game.set_doom_map(self.config.map)
        self.game.set_doom_skill(self.config.skill)
        self.game.set_window_visible(self.config.visible and not self.config.presentation_enabled)
        self.game.set_screen_resolution(getattr(self.vzd.ScreenResolution, self.config.screen_resolution))
        self.game.set_screen_format(self.vzd.ScreenFormat.RGB24)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_automap_buffer_enabled(False)
        self.game.set_objects_info_enabled(False)
        self.game.set_available_buttons(self.buttons)
        self.game.set_button_max_value(
            self.vzd.Button.TURN_LEFT_RIGHT_DELTA,
            self.config.hard_turn_degrees_per_tic,
        )
        self.game.set_available_game_variables(self.variables)
        self.game.set_episode_timeout(self.config.episode_timeout_tics)
        mode = self.vzd.Mode.ASYNC_PLAYER if self.config.async_mode else self.vzd.Mode.PLAYER
        self.game.set_mode(mode)
        self.game.init()
        LOGGER.info("ViZDoom started scenario=%s wad=%s mode=%s", self.config.scenario, self.config.wad_path, mode.name)

    def close(self) -> None:
        self.game.close()
        LOGGER.info("ViZDoom stopped")

    def is_finished(self) -> bool:
        return self.game.is_episode_finished()

    def new_episode(self) -> None:
        self.game.new_episode()
        self._last_health = None
        self._last_kills = None
        self._screen_buffer = None
        self._use_pulse_countdown = 0

    def tick(self, action: GameAction, pulse: bool) -> None:
        self.game.set_action(self.action_vector(action, pulse, self._next_use_pulse(action)))
        self.game.advance_action(1)

    def action_vector(self, action: GameAction, pulse: bool, use_pulse: bool | None = None) -> list[float]:
        action.validate()
        use = pulse if use_pulse is None else use_pulse
        return [
            float(action.movement == "forward"),
            float(action.movement == "backward"),
            float(action.strafe == "left"),
            float(action.strafe == "right"),
            self._turn_delta(action.turn),
            float(action.fire),
            float(action.use and use),
            float(action.weapon == "next" and pulse),
            float(action.weapon == "previous" and pulse),
            float(self.always_run),
            *[float(action.weapon == f"select_{number}" and pulse) for number in range(1, 10)],
        ]

    def _next_use_pulse(self, action: GameAction) -> bool:
        if not action.use:
            self._use_pulse_countdown = 0
            return False
        if self._use_pulse_countdown > 0:
            self._use_pulse_countdown -= 1
            return False
        self._use_pulse_countdown = self.USE_PULSE_INTERVAL_TICS - 1
        return True

    def _turn_delta(self, turn: str) -> float:
        rates = {
            "hard_left": -self.config.hard_turn_degrees_per_tic,
            "soft_left": -self.config.soft_turn_degrees_per_tic,
            "hold": 0.0,
            "soft_right": self.config.soft_turn_degrees_per_tic,
            "hard_right": self.config.hard_turn_degrees_per_tic,
        }
        return rates[turn]

    def observe(self, include_depth: bool = False) -> dict[str, Any] | None:
        try:
            state = self.game.get_state()
        except Exception as error:
            if not self._state_unavailable_reported:
                LOGGER.warning("ViZDoom state unavailable for one or more tics: %s", error)
                self._state_unavailable_reported = True
            return None
        if state is None:
            return None
        self._state_unavailable_reported = False
        self._screen_buffer = self._state_buffer(state, "screen_buffer")
        values = {variable.name.lower(): self.game.get_game_variable(variable) for variable in self.variables}
        health = values["health"]
        kills = values["killcount"]
        observation = {
            "state_number": state.number,
            "tic": state.tic,
            "episode_time_tics": self.game.get_episode_time(),
            "player": {
                "health": round(health),
                "armor": round(values["armor"]),
                "weapon": round(values["selected_weapon"]),
                "weapon_ammo": round(values["selected_weapon_ammo"]),
                "owned_weapons": [number for number in range(1, 10) if values[f"weapon{number}"] > 0],
                "ammo": {
                    "bullets": round(values["ammo1"]),
                    "shells": round(values["ammo2"]),
                    "rockets": round(values["ammo3"]),
                    "cells": round(values["ammo4"]),
                },
                "attack_ready": bool(values["attack_ready"]),
                "position": [round(values["position_x"], 1), round(values["position_y"], 1), round(values["position_z"], 1)],
                "heading_degrees": round(values["angle"], 1),
                "pitch_degrees": round(values["pitch"], 1),
                "velocity": [round(values["velocity_x"], 1), round(values["velocity_y"], 1)],
            },
            "combat": {
                "kills": round(kills),
                "hits": round(values["hitcount"]),
                "hits_taken": round(values["hits_taken"]),
                "damage_dealt": round(values["damagecount"]),
                "damage_taken": round(values["damage_taken"]),
                "health_change": None if self._last_health is None else round(health - self._last_health),
                "kills_change": None if self._last_kills is None else round(kills - self._last_kills),
            },
            "visible_objects": self._visible_objects(self._state_buffer(state, "labels"), values),
            "visible_object_counts": self._visible_object_counts(self._state_buffer(state, "labels")),
        }
        if include_depth:
            observation["depth_grid"] = self._sample_depth_grid(self._state_buffer(state, "depth_buffer"))
        self._last_health = health
        self._last_kills = kills
        return observation

    def screen_buffer(self) -> Any | None:
        return self._screen_buffer

    def _state_buffer(self, state: Any, name: str) -> Any:
        try:
            return getattr(state, name)
        except Exception as error:
            if name not in self._buffer_errors:
                LOGGER.warning("ViZDoom %s unavailable for one or more tics: %s", name, error)
                self._buffer_errors.add(name)
            return None

    def _sample_depth_grid(self, depth_buffer: Any) -> dict[str, Any] | None:
        if depth_buffer is None:
            return None
        height, width = depth_buffer.shape[:2]
        rows = self.config.depth_grid_rows
        columns = self.config.depth_grid_columns
        y_indexes = [min(int((row + 0.5) * height / rows), height - 1) for row in range(rows)]
        x_indexes = [min(int((column + 0.5) * width / columns), width - 1) for column in range(columns)]
        values = [[int(depth_buffer[y, x]) for x in x_indexes] for y in y_indexes]
        return {
            "encoding": "vizdoom_uint8",
            "rows": rows,
            "columns": columns,
            "values": values,
        }

    def _visible_objects(self, labels: Any, values: dict[str, float]) -> list[dict[str, Any]]:
        if not labels:
            return []
        px, py = values["position_x"], values["position_y"]
        heading = values["angle"]
        objects = [
            self._label_to_object(label, px, py, heading)
            for label in labels
            if label.object_category.lower() != "self"
        ]
        objects.sort(key=lambda item: (not is_hostile(item), item["distance"]))
        return objects[: self.max_visible_objects]

    @staticmethod
    def _visible_object_counts(labels: Any) -> list[dict[str, Any]]:
        counts: dict[tuple[str, str], int] = {}
        for label in labels or []:
            if label.object_category.lower() == "self":
                continue
            key = label.object_name, label.object_category
            counts[key] = counts.get(key, 0) + 1
        return [
            {"name": name, "category": category, "count": count}
            for (name, category), count in sorted(counts.items())
        ]

    @staticmethod
    def _label_to_object(label: Any, px: float, py: float, heading: float) -> dict[str, Any]:
        dx = label.object_position_x - px
        dy = label.object_position_y - py
        bearing = math.degrees(math.atan2(dy, dx)) % 360
        relative = (bearing - heading + 180) % 360 - 180
        return {
            "id": label.object_id,
            "name": label.object_name,
            "category": label.object_category,
            "distance": round(math.hypot(dx, dy), 1),
            "relative_bearing_degrees": round(relative, 1),
            "screen_box": [label.x, label.y, label.width, label.height],
            "world_position": [round(label.object_position_x, 1), round(label.object_position_y, 1), round(label.object_position_z, 1)],
        }
