import unittest

from doomrunner.actions import GameAction
from doomrunner.observation import CompactObservationHistory, RawObservationHistory, encode_control_state


def observation(tic: int) -> dict:
    return {
        "state_number": tic,
        "tic": tic,
        "episode_time_tics": tic,
        "player": {"position": [tic, 0, 0], "velocity": [1, 0]},
        "combat": {"kills": 0},
        "visible_objects": [
            {
                "id": 1,
                "name": "Demon",
                "category": "Monster",
                "distance": 100,
                "relative_bearing_degrees": 10,
                "screen_box": [1, 2, 3, 4],
                "world_position": [100, 100, 0],
            },
            {
                "id": 2,
                "name": "Medikit",
                "category": "Health",
                "distance": 40,
                "relative_bearing_degrees": -15,
                "screen_box": [4, 5, 6, 7],
                "world_position": [20, 20, 0],
            },
        ],
        "depth_grid": {
            "encoding": "vizdoom_uint8",
            "rows": 2,
            "columns": 6,
            "values": [[10, 20, 30, 40, 50, 60], [70, 80, 90, 100, 110, 120]],
        },
        "tactical_target": {"id": 1},
        "control_feedback": {"possible_stuck": True},
    }


class RawObservationHistoryTests(unittest.TestCase):
    def test_keeps_bounded_oldest_to_newest_sensor_history(self) -> None:
        history = RawObservationHistory(max_frames=3)
        history.build(observation(1), GameAction(movement="forward"))
        history.build(observation(2), GameAction(turn="soft_left"))

        result = history.build(observation(3), GameAction())

        self.assertEqual(3, result["current"]["observation"]["tic"])
        self.assertEqual(2, len(result["recent_transitions"]))
        self.assertEqual(1, result["recent_transitions"][0]["control_applied"]["longitudinal_axis"])
        self.assertEqual(-1, result["recent_transitions"][1]["control_applied"]["yaw_axis"])
        self.assertEqual(0, result["current"]["control_state"]["longitudinal_axis"])
        self.assertEqual([1.0, 0.0, 0.0], result["recent_transitions"][0]["player_delta"]["position"])

    def test_excludes_assisted_conclusions_and_world_coordinates(self) -> None:
        result = RawObservationHistory(max_frames=1).build(observation(1), GameAction())
        raw = result["current"]["observation"]

        self.assertNotIn("tactical_target", raw)
        self.assertNotIn("control_feedback", raw)
        self.assertNotIn("world_position", raw["visible_objects"][0])
        self.assertIn("depth_grid", raw)
        self.assertEqual([2, 6], raw["depth_grid"]["shape"])
        self.assertEqual([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], raw["depth_grid"]["values"])


class CompactObservationHistoryTests(unittest.TestCase):
    def test_encodes_direct_weapon_selection(self) -> None:
        control = encode_control_state(GameAction(weapon="select_4"))

        self.assertEqual(0, control["weapon_step"])
        self.assertEqual(4, control["weapon_select"])

    def test_reports_neutral_grouped_measurements(self) -> None:
        history = CompactObservationHistory()

        result = history.build(observation(1), GameAction(movement="forward"))

        self.assertEqual("compact_measurements", result["observation_mode"])
        self.assertEqual([1], [item["id"] for item in result["visible_hostiles"]])
        self.assertEqual([2], [item["id"] for item in result["visible_resources_and_objects"]])
        self.assertNotIn("world_position", result["visible_hostiles"][0])
        self.assertIsNone(result["motion_since_previous_decision"])
        self.assertEqual(15.0, result["depth_sectors"]["left"]["median"])
        self.assertEqual(35.0, result["depth_sectors"]["center"]["median"])
        self.assertEqual(55.0, result["depth_sectors"]["right"]["median"])

    def test_pairs_prior_control_with_measured_motion(self) -> None:
        history = CompactObservationHistory()
        history.build(observation(1), GameAction())

        result = history.build(observation(4), GameAction(movement="forward", turn="soft_right"))
        motion = result["motion_since_previous_decision"]

        self.assertEqual(3, motion["elapsed_tics"])
        self.assertEqual(1, motion["control_applied"]["longitudinal_axis"])
        self.assertEqual(1, motion["control_applied"]["yaw_axis"])
        self.assertEqual([3.0, 0.0, 0.0], motion["position_delta"])
        self.assertEqual(3.0, motion["distance_moved"])

    def test_does_not_add_assisted_conclusions(self) -> None:
        result = CompactObservationHistory().build(observation(1), GameAction())
        serialized_keys = str(result).lower()

        self.assertNotIn("stuck", serialized_keys)
        self.assertNotIn("tactical_target", serialized_keys)
        self.assertNotIn("turn_hint", serialized_keys)


if __name__ == "__main__":
    unittest.main()
