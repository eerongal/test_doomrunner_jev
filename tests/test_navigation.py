import unittest

from doomrunner.navigation import SeenSpaceNavigator


def state(position: list[float], heading: float, values: list[list[int]]) -> dict:
    return {
        "player": {"position": position, "heading_degrees": heading},
        "depth_grid": {"rows": 2, "columns": 6, "values": values},
    }


class SeenSpaceNavigatorTests(unittest.TestCase):
    def test_prefers_open_candidate_that_has_not_been_visited(self) -> None:
        navigator = SeenSpaceNavigator()
        navigator.observe(state([40, 30, 0], 0, [[48, 48, 20, 20, 48, 48], [0, 0, 0, 0, 0, 0]]))
        navigator.observe(state([0, 0, 0], 0, [[48, 48, 20, 20, 48, 48], [0, 0, 0, 0, 0, 0]]))

        guide = navigator.guide()

        self.assertIsNotNone(guide)
        self.assertEqual("right", guide.direction)

    def test_is_unavailable_without_depth_or_player_pose(self) -> None:
        navigator = SeenSpaceNavigator()
        navigator.observe({"player": {}})

        self.assertEqual({"status": "unavailable"}, navigator.summary())


if __name__ == "__main__":
    unittest.main()
