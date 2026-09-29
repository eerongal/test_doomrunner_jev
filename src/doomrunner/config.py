from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DoomConfig:
    wad_path: Path
    scenario: str = "deathmatch"
    map: str = "map01"
    visible: bool = True
    async_mode: bool = True
    screen_resolution: str = "RES_640X480"
    presentation_enabled: bool = False
    episode_timeout_tics: int = 4200
    skill: int = 3
    soft_turn_degrees_per_tic: float = 1.0
    hard_turn_degrees_per_tic: float = 3.0
    depth_grid_rows: int = 4
    depth_grid_columns: int = 8


@dataclass(frozen=True)
class ServerConfig:
    base_url: str = "http://127.0.0.1:8096"
    planner_provider: str = "local"
    planner_base_url: str | None = None
    decision_model: str = "LFM2.5-8B"
    decision_prompt_profile: str = "detailed"
    planner_model: str = "LFM2.5-8B"
    planner_reasoning_effort: str = "none"
    planner_json_mode: bool = True
    planner_max_tokens: int = 768
    planner_timeout_seconds: float = 30.0
    planner_deadline_seconds: float | None = None
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class ControllerConfig:
    decision_pacing: str = "max"
    decision_hz: float = 4.0
    planner_enabled: bool = True
    planner_interval_seconds: float = 10.0
    planner_calm_seconds: float = 3.0
    planner_min_health_for_background_work: int = 60
    planner_damage_threshold: int = 15
    planner_critical_health: int = 30
    survival_retreat_health: int = 35
    idle_hold_recovery: bool = False
    campaign_navigation: bool = False
    planner_blocked_progress_cooldown_seconds: float = 10.0
    memory_enabled: bool = False
    max_visible_objects: int = 8
    always_run: bool = True
    observation_mode: str = "raw"
    raw_history_length: int = 3
    semantic_control_lease_seconds: float = 2.0
    serialize_inference_requests: bool = True


@dataclass(frozen=True)
class TelemetryConfig:
    directory: Path = Path("logs")


@dataclass(frozen=True)
class AppConfig:
    doom: DoomConfig
    server: ServerConfig
    controller: ControllerConfig
    telemetry: TelemetryConfig

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        root = path.resolve().parent.parent
        doom_data = dict(raw.get("doom", {}))
        telemetry_data = dict(raw.get("telemetry", {}))
        doom_data["wad_path"] = _resolve_path(root, doom_data.get("wad_path", ""))
        telemetry_data["directory"] = _resolve_path(root, telemetry_data.get("directory", "logs"))
        config = cls(
            doom=DoomConfig(**doom_data),
            server=ServerConfig(**raw.get("server", {})),
            controller=ControllerConfig(**raw.get("controller", {})),
            telemetry=TelemetryConfig(**telemetry_data),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.doom.wad_path.is_file():
            raise ValueError(f"Doom WAD does not exist: {self.doom.wad_path}")
        if self.controller.decision_pacing not in {"max", "fixed"}:
            raise ValueError("controller.decision_pacing must be 'max' or 'fixed'")
        if self.controller.decision_pacing == "fixed" and self.controller.decision_hz <= 0:
            raise ValueError("controller.decision_hz must be greater than zero")
        if self.controller.planner_interval_seconds <= 0:
            raise ValueError("controller.planner_interval_seconds must be greater than zero")
        if self.controller.planner_calm_seconds <= 0:
            raise ValueError("controller.planner_calm_seconds must be greater than zero")
        if not 0 <= self.controller.planner_min_health_for_background_work <= 100:
            raise ValueError("controller.planner_min_health_for_background_work must be between 0 and 100")
        if self.controller.planner_damage_threshold <= 0:
            raise ValueError("controller.planner_damage_threshold must be greater than zero")
        if self.controller.planner_critical_health <= 0:
            raise ValueError("controller.planner_critical_health must be greater than zero")
        if not 1 <= self.controller.survival_retreat_health <= 100:
            raise ValueError("controller.survival_retreat_health must be between 1 and 100")
        if self.controller.planner_blocked_progress_cooldown_seconds <= 0:
            raise ValueError("controller.planner_blocked_progress_cooldown_seconds must be greater than zero")
        if self.controller.observation_mode not in {"raw", "compact", "semantic", "assisted"}:
            raise ValueError("controller.observation_mode must be 'raw', 'compact', 'semantic', or 'assisted'")
        if self.controller.raw_history_length <= 0:
            raise ValueError("controller.raw_history_length must be greater than zero")
        if self.controller.semantic_control_lease_seconds <= 0:
            raise ValueError("controller.semantic_control_lease_seconds must be greater than zero")
        if not 1 <= self.doom.skill <= 5:
            raise ValueError("doom.skill must be between 1 and 5")
        if self.doom.soft_turn_degrees_per_tic <= 0:
            raise ValueError("doom.soft_turn_degrees_per_tic must be greater than zero")
        if self.doom.hard_turn_degrees_per_tic < self.doom.soft_turn_degrees_per_tic:
            raise ValueError("doom.hard_turn_degrees_per_tic must be at least the soft turn rate")
        if self.doom.depth_grid_rows <= 0 or self.doom.depth_grid_columns <= 0:
            raise ValueError("doom depth grid dimensions must be greater than zero")
        if not self.server.base_url.startswith(("http://", "https://")):
            raise ValueError("server.base_url must be an HTTP URL")
        if self.server.planner_base_url and not self.server.planner_base_url.startswith(("http://", "https://")):
            raise ValueError("server.planner_base_url must be an HTTP URL when configured")
        if self.server.planner_provider not in {"local", "openrouter"}:
            raise ValueError("server.planner_provider must be 'local' or 'openrouter'")
        if self.server.planner_provider == "openrouter" and self.server.planner_base_url:
            raise ValueError("server.planner_base_url is only supported for the local planner provider")
        if self.server.decision_prompt_profile not in {"detailed", "compact"}:
            raise ValueError("server.decision_prompt_profile must be 'detailed' or 'compact'")
        if self.server.planner_reasoning_effort not in {"none", "low", "medium", "high"}:
            raise ValueError("server.planner_reasoning_effort must be 'none', 'low', 'medium', or 'high'")
        if not isinstance(self.server.planner_json_mode, bool):
            raise ValueError("server.planner_json_mode must be a boolean")
        if self.server.planner_max_tokens <= 0:
            raise ValueError("server.planner_max_tokens must be greater than zero")
        if self.server.planner_timeout_seconds <= 0:
            raise ValueError("server.planner_timeout_seconds must be greater than zero")
        if self.server.planner_deadline_seconds is not None and self.server.planner_deadline_seconds <= 0:
            raise ValueError("server.planner_deadline_seconds must be greater than zero when configured")


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path
