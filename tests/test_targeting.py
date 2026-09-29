import unittest

from doomrunner.targeting import TargetLock


def hostile(object_id: int, distance: float, bearing: float = 0.0) -> dict:
    return {
        "id": object_id,
        "name": "Demon",
        "category": "Monster",
        "distance": distance,
        "relative_bearing_degrees": bearing,
    }


class TargetLockTests(unittest.TestCase):
    def test_keeps_visible_target_when_another_becomes_nearer(self) -> None:
        lock = TargetLock()

        first = lock.update([hostile(1, 100), hostile(2, 200)])
        second = lock.update([hostile(1, 150), hostile(2, 50)])

        self.assertEqual(1, first["id"])
        self.assertEqual(1, second["id"])

    def test_provides_unambiguous_turn_hint(self) -> None:
        lock = TargetLock()

        right = lock.update([hostile(1, 100, -20)])
        lock.reset()
        aligned = lock.update([hostile(2, 100, 3)])
        lock.reset()
        left = lock.update([hostile(3, 100, 20)])

        self.assertEqual("right", right["turn_hint"])
        self.assertEqual("aligned", aligned["turn_hint"])
        self.assertEqual("left", left["turn_hint"])

    def test_retains_lost_target_briefly(self) -> None:
        lock = TargetLock(max_missed_decisions=1)
        lock.update([hostile(1, 100, 10)])

        missing = lock.update([hostile(2, 50)])
        replacement = lock.update([hostile(2, 50)])

        self.assertEqual(1, missing["id"])
        self.assertFalse(missing["visible"])
        self.assertEqual(2, replacement["id"])
        self.assertTrue(replacement["visible"])
