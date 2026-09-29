import unittest

from doomrunner.actions import GameAction, decision_schema


class GameActionTests(unittest.TestCase):
    def test_default_action_stands_still(self) -> None:
        action = GameAction()

        self.assertEqual("stop", action.movement)
        self.assertEqual("stop", action.strafe)
        self.assertEqual("hold", action.turn)
        self.assertFalse(action.fire)

    def test_accepts_simultaneous_channels(self) -> None:
        action = GameAction.from_decision(
            {
                "movement": "forward",
                "strafe": "left",
                "turn": "hard_right",
                "fire": True,
                "use": False,
                "weapon": "keep",
            }
        )

        self.assertEqual("forward", action.movement)
        self.assertEqual("left", action.strafe)
        self.assertEqual("hard_right", action.turn)
        self.assertTrue(action.fire)

    def test_rejects_unknown_choice(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid movement"):
            GameAction.from_decision({"movement": "teleport"})

    def test_mobility_policy_changes_movement_question_without_mutating_base_schema(self) -> None:
        advance = decision_schema("advance")
        hold = decision_schema("hold_position")

        self.assertIn("forward movement", advance["movement"]["description"])
        self.assertIn("holding position", hold["movement"]["description"])
        self.assertNotEqual(advance["movement"]["description"], hold["movement"]["description"])

    def test_rejects_non_boolean_fire(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be booleans"):
            GameAction.from_decision({"fire": "true"})


if __name__ == "__main__":
    unittest.main()
