import json
import tempfile
import unittest
from pathlib import Path

from doomrunner.config import AppConfig


class ConfigTests(unittest.TestCase):
    def test_loads_valid_config_and_resolves_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "config"
            config_dir.mkdir()
            wad = root / "doom2.wad"
            wad.write_bytes(b"test")
            path = config_dir / "doomrunner.json"
            path.write_text(json.dumps({"doom": {"wad_path": "doom2.wad"}}), encoding="utf-8")

            config = AppConfig.load(path)

            self.assertEqual(wad, config.doom.wad_path)
            self.assertEqual(root / "logs", config.telemetry.directory)
            self.assertEqual("raw", config.controller.observation_mode)
            self.assertFalse(config.controller.idle_hold_recovery)

    def test_rejects_missing_wad(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "config"
            config_dir.mkdir()
            path = config_dir / "doomrunner.json"
            path.write_text(json.dumps({"doom": {"wad_path": "missing.wad"}}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not exist"):
                AppConfig.load(path)

    def test_rejects_unknown_decision_pacing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "config"
            config_dir.mkdir()
            wad = root / "doom2.wad"
            wad.write_bytes(b"test")
            path = config_dir / "doomrunner.json"
            path.write_text(
                json.dumps({"doom": {"wad_path": "doom2.wad"}, "controller": {"decision_pacing": "turbo"}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "decision_pacing"):
                AppConfig.load(path)

    def test_rejects_invalid_planner_background_health_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "config"
            config_dir.mkdir()
            wad = root / "doom2.wad"
            wad.write_bytes(b"test")
            path = config_dir / "doomrunner.json"
            path.write_text(
                json.dumps(
                    {
                        "doom": {"wad_path": "doom2.wad"},
                        "controller": {"planner_min_health_for_background_work": 101},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "planner_min_health_for_background_work"):
                AppConfig.load(path)

    def test_rejects_invalid_survival_retreat_health_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "config"
            config_dir.mkdir()
            wad = root / "doom2.wad"
            wad.write_bytes(b"test")
            path = config_dir / "doomrunner.json"
            path.write_text(
                json.dumps({"doom": {"wad_path": "doom2.wad"}, "controller": {"survival_retreat_health": 0}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "survival_retreat_health"):
                AppConfig.load(path)

    def test_rejects_unknown_planner_reasoning_effort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "config"
            config_dir.mkdir()
            wad = root / "doom2.wad"
            wad.write_bytes(b"test")
            path = config_dir / "doomrunner.json"
            path.write_text(
                json.dumps(
                    {
                        "doom": {"wad_path": "doom2.wad"},
                        "server": {"planner_reasoning_effort": "extreme"},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "planner_reasoning_effort"):
                AppConfig.load(path)

    def test_rejects_non_boolean_planner_json_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "config"
            config_dir.mkdir()
            wad = root / "doom2.wad"
            wad.write_bytes(b"test")
            path = config_dir / "doomrunner.json"
            path.write_text(
                json.dumps({"doom": {"wad_path": "doom2.wad"}, "server": {"planner_json_mode": "false"}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "planner_json_mode"):
                AppConfig.load(path)

    def test_rejects_nonpositive_planner_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "config"
            config_dir.mkdir()
            wad = root / "doom2.wad"
            wad.write_bytes(b"test")
            path = config_dir / "doomrunner.json"
            path.write_text(
                json.dumps({"doom": {"wad_path": "doom2.wad"}, "server": {"planner_deadline_seconds": 0}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "planner_deadline_seconds"):
                AppConfig.load(path)

    def test_rejects_unknown_planner_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "config"
            config_dir.mkdir()
            wad = root / "doom2.wad"
            wad.write_bytes(b"test")
            path = config_dir / "doomrunner.json"
            path.write_text(
                json.dumps({"doom": {"wad_path": "doom2.wad"}, "server": {"planner_provider": "invalid"}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "planner_provider"):
                AppConfig.load(path)

    def test_rejects_unknown_decision_prompt_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "config"
            config_dir.mkdir()
            wad = root / "doom2.wad"
            wad.write_bytes(b"test")
            path = config_dir / "doomrunner.json"
            path.write_text(
                json.dumps(
                    {
                        "doom": {"wad_path": "doom2.wad"},
                        "server": {"decision_prompt_profile": "minimal"},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "decision_prompt_profile"):
                AppConfig.load(path)

    def test_accepts_compact_observation_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "config"
            config_dir.mkdir()
            wad = root / "doom2.wad"
            wad.write_bytes(b"test")
            path = config_dir / "doomrunner.json"
            path.write_text(
                json.dumps({"doom": {"wad_path": "doom2.wad"}, "controller": {"observation_mode": "compact"}}),
                encoding="utf-8",
            )

            config = AppConfig.load(path)

            self.assertEqual("compact", config.controller.observation_mode)

    def test_accepts_semantic_observation_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "config"
            config_dir.mkdir()
            wad = root / "doom2.wad"
            wad.write_bytes(b"test")
            path = config_dir / "doomrunner.json"
            path.write_text(
                json.dumps({"doom": {"wad_path": "doom2.wad"}, "controller": {"observation_mode": "semantic"}}),
                encoding="utf-8",
            )

            config = AppConfig.load(path)

            self.assertEqual("semantic", config.controller.observation_mode)


if __name__ == "__main__":
    unittest.main()
