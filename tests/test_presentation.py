import unittest

import numpy as np

from doomrunner.inference import Plan
from doomrunner.presentation import action_matrix, frame_to_rgb, objective_display
from doomrunner.semantic import SemanticIntent


class PresentationTests(unittest.TestCase):
    def test_highlights_the_current_semantic_choices(self) -> None:
        intent = SemanticIntent(
            mode="fight",
            face="hostile:7",
            move="engage",
            dodge="left",
            trigger="fire",
            use="no_use",
            weapon="keep",
        )
        rows = {row.label: row for row in action_matrix(intent)}

        self.assertEqual("fight", rows["MODE"].selected)
        self.assertEqual("hostile:7", rows["FACE"].selected)
        self.assertEqual("engage", rows["MOVE"].selected)
        self.assertEqual("fire", rows["FIRE REQUEST"].selected)

    def test_shows_a_motor_use_override(self) -> None:
        rows = {row.label: row for row in action_matrix(SemanticIntent(), {"use": True})}

        self.assertEqual("use", rows["USE (MOTOR)"].selected)

    def test_converts_channel_first_vizdoom_frames_to_rgb(self) -> None:
        frame = np.zeros((3, 2, 4), dtype=np.uint8)
        frame[0, :, :] = 10
        frame[1, :, :] = 20
        frame[2, :, :] = 30

        rgb = frame_to_rgb(frame)

        self.assertEqual((2, 4, 3), rgb.shape)
        self.assertEqual([10, 20, 30], rgb[0, 0].tolist())

    def test_shows_a_safety_override_instead_of_the_stale_plan_objective(self) -> None:
        display = objective_display(Plan(), "CRITICAL HEALTH (20 HP): retreat from visible threats.")

        self.assertIn("SAFETY OVERRIDE", display.header)
        self.assertIn("CRITICAL HEALTH", display.primary)
        self.assertIn("SAVED L2 PLAN", display.secondary)


if __name__ == "__main__":
    unittest.main()
