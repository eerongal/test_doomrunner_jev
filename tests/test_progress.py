import unittest

from doomrunner.actions import GameAction
from doomrunner.progress import ProgressFeedback


def observation(x: float) -> dict:
    return {"player": {"position": [x, 0, 0]}}


def wall_observation(x: float) -> dict:
    return {
        "player": {"position": [x, 0, 0]},
        "depth_grid": {"values": [[50, 50, 8, 8], [50, 50, 8, 8], [10, 10, 10, 10]]},
    }


class ProgressFeedbackTests(unittest.TestCase):
    def test_reports_blocked_after_sustained_failed_forward_motion(self) -> None:
        feedback = ProgressFeedback()
        feedback.observe(observation(0), GameAction(movement="forward"), now=10.0)
        feedback.observe(observation(0), GameAction(movement="forward"), now=10.5)

        result = feedback.summary()

        self.assertEqual("blocked", result["status"])
        self.assertEqual("backtrack_and_turn", result["suggested_recovery"])

    def test_clears_blocked_status_when_forward_progress_returns(self) -> None:
        feedback = ProgressFeedback()
        feedback.observe(observation(0), GameAction(movement="forward"), now=10.0)
        feedback.observe(observation(0), GameAction(movement="forward"), now=10.5)
        feedback.observe(observation(5), GameAction(movement="forward"), now=10.6)

        self.assertIsNone(feedback.summary())

    def test_reports_wall_following_while_forward_progress_continues(self) -> None:
        feedback = ProgressFeedback()
        action = GameAction(movement="forward", turn="soft_right")
        feedback.observe(wall_observation(0), action, now=10.0)
        feedback.observe(wall_observation(20), action, now=10.5)
        feedback.observe(wall_observation(40), action, now=11.0)

        result = feedback.summary()

        self.assertEqual("wall_following", result["status"])
        self.assertEqual("right", result["turn_direction"])

    def test_reports_wall_following_when_turning_away_from_the_nearby_wall(self) -> None:
        feedback = ProgressFeedback()
        action = GameAction(movement="forward", turn="soft_left")
        feedback.observe(wall_observation(0), action, now=10.0)
        feedback.observe(wall_observation(20), action, now=10.5)
        feedback.observe(wall_observation(40), action, now=11.0)

        result = feedback.summary()

        self.assertEqual("wall_following", result["status"])
        self.assertEqual("left", result["turn_direction"])
        self.assertEqual(8.0, result["near_wall_depth"])


if __name__ == "__main__":
    unittest.main()
