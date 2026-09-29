import unittest

from doomrunner.threats import RecentThreatMemory


def observation(heading: float = 0.0) -> dict:
    return {
        "player": {"position": [0, 0, 0], "heading_degrees": heading},
        "visible_objects": [
            {
                "id": 7,
                "name": "Demon",
                "category": "Monster",
                "distance": 100,
                "relative_bearing_degrees": 0,
                "world_position": [100, 0, 0],
            }
        ],
    }


class RecentThreatMemoryTests(unittest.TestCase):
    def test_retains_directly_seen_threat_for_a_short_interval(self) -> None:
        memory = RecentThreatMemory(retention_seconds=1.5)
        memory.observe(observation(), now=10.0)

        target = memory.target("7", {"player": {"position": [0, 0, 0], "heading_degrees": 20}}, now=11.0)

        self.assertIsNotNone(target)
        self.assertFalse(target["visible"])
        self.assertEqual(-20.0, target["relative_bearing_degrees"])

    def test_expires_old_threats(self) -> None:
        memory = RecentThreatMemory(retention_seconds=1.5)
        memory.observe(observation(), now=10.0)

        self.assertEqual([], memory.summary(now=11.6))


if __name__ == "__main__":
    unittest.main()
