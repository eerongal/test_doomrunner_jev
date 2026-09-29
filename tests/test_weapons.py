import unittest

from doomrunner.weapons import recommended_weapon_action


class WeaponPolicyTests(unittest.TestCase):
    def test_prefers_loaded_high_power_weapon_at_range(self) -> None:
        observation = {"player": {"weapon": 2, "owned_weapons": [1, 2, 3, 4], "ammo": {"bullets": 30, "shells": 8}}}

        self.assertEqual("select_4", recommended_weapon_action(observation, {"distance": 500}))

    def test_avoids_rocket_launcher_at_close_range(self) -> None:
        observation = {"player": {"weapon": 2, "owned_weapons": [2, 5, 9], "ammo": {"bullets": 30, "rockets": 10, "shells": 8}}}

        self.assertEqual("select_9", recommended_weapon_action(observation, {"distance": 80}))

    def test_does_not_select_melee_against_a_distant_target(self) -> None:
        observation = {"player": {"weapon": 2, "owned_weapons": [1, 2], "ammo": {"bullets": 0}}}

        self.assertEqual("keep", recommended_weapon_action(observation, {"distance": 500}))
