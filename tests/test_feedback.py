import unittest

from doomrunner.actions import GameAction
from doomrunner.feedback import ControlFeedback


def observation(tic: int, heading: float) -> dict:
    return {"tic": tic, "player": {"heading_degrees": heading, "position": [0, 0, 0]}}


def target(bearing: float) -> dict:
    return {"id": 7, "relative_bearing_degrees": bearing, "visible": True}


class ControlFeedbackTests(unittest.TestCase):
    def test_reports_when_aim_is_improving(self) -> None:
        feedback = ControlFeedback()
        action = GameAction(turn="soft_left")
        feedback.record_action(action, observation(1, 0))
        feedback.snapshot(observation(1, 0), target(30))

        result = feedback.snapshot(observation(2, 2), target(20))

        self.assertEqual("improving", result["aim_trend"])
        self.assertEqual(-10.0, result["target_bearing_change"])
        self.assertEqual(action.to_dict(), result["evaluated_action"])

    def test_detects_sustained_circling(self) -> None:
        feedback = ControlFeedback()
        action = GameAction(turn="hard_right")
        feedback.record_action(action, observation(1, 0))
        feedback.record_action(action, observation(2, 0))
        feedback.record_action(action, observation(3, 0))
        feedback.record_action(action, observation(4, 0))

        result = feedback.snapshot(observation(20, 100), None)

        self.assertTrue(result["possible_circling"])

    def test_wraps_heading_change_at_zero_degrees(self) -> None:
        feedback = ControlFeedback()
        feedback.record_action(GameAction(turn="soft_right"), observation(1, 359))

        result = feedback.snapshot(observation(2, 1), None)

        self.assertEqual(2.0, result["heading_change_during_streak"])

    def test_detects_repeated_movement_without_progress(self) -> None:
        feedback = ControlFeedback()
        action = GameAction(movement="forward")
        feedback.record_action(action, observation(1, 0))
        feedback.snapshot(observation(1, 0), None)
        feedback.record_action(action, observation(2, 0))
        feedback.snapshot(observation(2, 0), None)
        feedback.record_action(action, observation(3, 0))

        result = feedback.snapshot(observation(3, 0), None)

        self.assertTrue(result["possible_stuck"])
        self.assertEqual(2, result["stuck_streak"])

    def test_keeps_stuck_state_until_position_recovers(self) -> None:
        feedback = ControlFeedback()
        moving = GameAction(movement="forward")
        feedback.record_action(moving, observation(1, 0))
        feedback.snapshot(observation(1, 0), None)
        feedback.record_action(moving, observation(2, 0))
        feedback.snapshot(observation(2, 0), None)
        feedback.record_action(moving, observation(3, 0))
        feedback.snapshot(observation(3, 0), None)
        feedback.record_action(GameAction(), observation(4, 0))

        result = feedback.snapshot(observation(4, 0), None)

        self.assertTrue(result["possible_stuck"])
        self.assertGreaterEqual(result["stuck_streak"], 2)
