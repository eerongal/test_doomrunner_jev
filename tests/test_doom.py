import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from doomrunner.actions import GameAction
from doomrunner.config import DoomConfig
from doomrunner.doom import DoomSession


class DoomActionVectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = DoomSession(DoomConfig(wad_path=Path("doom2.wad")), 8, True)
        self.action = GameAction(
            movement="forward",
            strafe="left",
            turn="hard_right",
            fire=True,
            use=True,
            weapon="next",
        )

    def test_combines_continuous_and_impulse_controls(self) -> None:
        vector = self.session.action_vector(self.action, pulse=True)

        self.assertEqual([1.0, 0.0, 1.0, 0.0], vector[:4])
        self.assertEqual(3.0, vector[4])
        self.assertEqual(1.0, vector[5])
        self.assertEqual(1.0, vector[6])
        self.assertEqual(1.0, vector[7])

    def test_clears_use_and_weapon_impulses_but_holds_continuous_controls(self) -> None:
        vector = self.session.action_vector(self.action, pulse=False)

        self.assertEqual([1.0, 0.0, 1.0, 0.0], vector[:4])
        self.assertEqual(3.0, vector[4])
        self.assertEqual(1.0, vector[5])
        self.assertEqual(0.0, vector[6])
        self.assertEqual(0.0, vector[7])

    def test_allows_a_dedicated_use_pulse(self) -> None:
        vector = self.session.action_vector(self.action, pulse=False, use_pulse=True)

        self.assertEqual(1.0, vector[6])
        self.assertEqual(0.0, vector[7])

    def test_uses_vizdoom_turn_direction_convention(self) -> None:
        left = self.session.action_vector(GameAction(turn="soft_left"), pulse=False)
        right = self.session.action_vector(GameAction(turn="soft_right"), pulse=False)

        self.assertEqual(-1.0, left[4])
        self.assertEqual(1.0, right[4])

    def test_tick_advances_async_game_state(self) -> None:
        self.session.game = Mock()

        self.session.tick(self.action, pulse=True)

        self.session.game.set_action.assert_called_once()
        self.session.game.advance_action.assert_called_once_with(1)

    def test_visible_object_budget_prioritizes_hostiles(self) -> None:
        labels = [
            self._label(1, "Stimpack", "Health", 10),
            self._label(2, "Clip", "Ammo", 20),
            self._label(3, "Demon", "Monster", 100),
        ]
        self.session.max_visible_objects = 2

        objects = self.session._visible_objects(labels, {"position_x": 0, "position_y": 0, "angle": 0})

        self.assertEqual("Demon", objects[0]["name"])
        self.assertEqual("Stimpack", objects[1]["name"])

    def test_summarizes_every_visible_label_without_the_object_budget(self) -> None:
        labels = [
            self._label(1, "Stimpack", "Health", 10),
            self._label(2, "Stimpack", "Health", 20),
            self._label(3, "Demon", "Monster", 100),
        ]

        counts = self.session._visible_object_counts(labels)

        self.assertEqual(
            [
                {"name": "Demon", "category": "Monster", "count": 1},
                {"name": "Stimpack", "category": "Health", "count": 2},
            ],
            counts,
        )

    def test_samples_depth_buffer_into_configured_grid(self) -> None:
        self.session.config = DoomConfig(
            wad_path=Path("doom2.wad"),
            depth_grid_rows=2,
            depth_grid_columns=4,
        )
        depth = np.arange(32, dtype=np.uint8).reshape(4, 8)

        result = self.session._sample_depth_grid(depth)

        self.assertEqual(2, result["rows"])
        self.assertEqual(4, result["columns"])
        self.assertEqual([[9, 11, 13, 15], [25, 27, 29, 31]], result["values"])

    def test_tolerates_a_temporarily_unavailable_vizdoom_buffer(self) -> None:
        class NullBuffer:
            @property
            def labels(self) -> None:
                raise RuntimeError("cannot create a pybind11::array_t from a nullptr")

        result = self.session._state_buffer(NullBuffer(), "labels")

        self.assertIsNone(result)
        self.assertIn("labels", self.session._buffer_errors)

    def test_tolerates_a_temporarily_unavailable_vizdoom_state(self) -> None:
        self.session.game = Mock()
        self.session.game.get_state.side_effect = RuntimeError("cannot create a pybind11::array_t from a nullptr")

        result = self.session.observe()

        self.assertIsNone(result)
        self.assertTrue(self.session._state_unavailable_reported)

    @staticmethod
    def _label(object_id: int, name: str, category: str, x: float) -> Mock:
        return Mock(
            object_id=object_id,
            object_name=name,
            object_category=category,
            object_position_x=x,
            object_position_y=0,
            object_position_z=0,
            x=0,
            y=0,
            width=1,
            height=1,
        )


if __name__ == "__main__":
    unittest.main()
