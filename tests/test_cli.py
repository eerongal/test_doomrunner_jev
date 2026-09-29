import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from doomrunner.cli import apply_overrides, run
from doomrunner.config import AppConfig, ControllerConfig, DoomConfig, ServerConfig, TelemetryConfig


class CliOverrideTests(unittest.TestCase):
    def test_overrides_level_one_model_and_disables_planner(self) -> None:
        config = AppConfig(
            doom=DoomConfig(wad_path=Path("doom2.wad")),
            server=ServerConfig(),
            controller=ControllerConfig(),
            telemetry=TelemetryConfig(),
        )
        args = Namespace(
            decision_model="Qwen3.5-9B",
            planner_model=None,
            planner_reasoning_effort="low",
            no_planner=True,
            observation_mode="assisted",
        )

        result = apply_overrides(config, args)

        self.assertEqual("Qwen3.5-9B", result.server.decision_model)
        self.assertEqual(config.server.planner_model, result.server.planner_model)
        self.assertEqual("low", result.server.planner_reasoning_effort)
        self.assertFalse(result.controller.planner_enabled)
        self.assertEqual("assisted", result.controller.observation_mode)

    def test_normal_command_runs_requested_episodes(self) -> None:
        args = Namespace(
            config=Path("config.json"),
            decision_model=None,
            planner_model=None,
            planner_reasoning_effort=None,
            no_planner=False,
            observation_mode=None,
            mode="jev",
            check=False,
            episodes=3,
        )
        config = Mock()
        controller = Mock()
        controller.run = AsyncMock()
        with patch("doomrunner.cli.AppConfig.load", return_value=config), patch(
            "doomrunner.cli.apply_overrides", return_value=config
        ), patch("doomrunner.cli.DoomController", return_value=controller):
            import asyncio

            asyncio.run(run(args))

        controller.run.assert_awaited_once_with(3)
