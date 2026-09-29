import unittest

from doomrunner.actions import GameAction
from doomrunner.semantic import DODGE_SKILLS, SemanticIntent, SemanticMotor, semantic_context, semantic_schema


def observation(bearing: float = 20.0, include_hostile: bool = True) -> dict:
    visible = []
    if include_hostile:
        visible.append(
            {
                "id": 7,
                "name": "Demon",
                "category": "Monster",
                "distance": 120,
                "relative_bearing_degrees": bearing,
            }
        )
    return {
        "player": {"attack_ready": True},
        "visible_objects": visible,
        "depth_grid": {
            "encoding": "vizdoom_uint8",
            "rows": 2,
            "columns": 6,
            "values": [[30, 30, 5, 5, 15, 15], [10, 10, 10, 10, 10, 10]],
        },
    }


def enclosed_observation() -> dict:
    result = observation()
    result["depth_grid"]["values"] = [[0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]]
    return result


class SemanticSchemaTests(unittest.TestCase):
    def test_offers_exact_visible_hostiles_as_face_choices(self) -> None:
        compact = {
            "observation_mode": "compact_measurements",
            "visible_hostiles": [{"id": 7, "name": "Demon", "relative_bearing_degrees": 20}],
        }

        context = semantic_context(compact)
        schema = semantic_schema(context, "advance")

        self.assertEqual("semantic_skills", context["observation_mode"])
        self.assertEqual("hostile:7", context["visible_hostiles"][0]["face_choice"])
        self.assertIn("hostile:7", schema["face"]["choices"])

    def test_rejects_a_target_that_was_not_offered(self) -> None:
        with self.assertRaisesRegex(ValueError, "semantic face"):
            SemanticIntent.from_decision(
                {"face": "hostile:99"},
                ("hold", "hostile:7"),
                ("fight", "hold"),
                DODGE_SKILLS,
            )

    def test_offers_recent_threat_as_a_face_choice(self) -> None:
        context = semantic_context(
            {"visible_hostiles": []},
            recent_threats=[{"id": 7, "name": "Demon", "age_seconds": 0.2}],
        )

        self.assertIn("recent_hostile:7", semantic_schema(context, "advance")["face"]["choices"])

    def test_live_hostile_requires_a_combat_or_hold_mode(self) -> None:
        context = semantic_context({"visible_hostiles": [{"id": 7, "name": "Demon"}]})

        choices = semantic_schema(context, "advance")["immediate_mode"]["choices"]

        self.assertIn("fight", choices)
        self.assertNotIn("evade", choices)
        self.assertNotIn("explore", choices)
        self.assertEqual(["hostile:7"], semantic_schema(context, "advance")["face"]["choices"])

    def test_campaign_navigation_requires_exploration_and_forward_progress_without_hostiles(self) -> None:
        context = semantic_context({}, campaign_navigation=True)
        schema = semantic_schema(context, "advance")

        self.assertEqual("campaign_navigation", context["mission_profile"])
        self.assertEqual(["explore"], schema["immediate_mode"]["choices"])
        self.assertEqual(["advance"], schema["move"]["choices"])

    def test_campaign_navigation_restores_combat_choices_for_visible_hostiles(self) -> None:
        context = semantic_context({"visible_hostiles": [{"id": 7, "name": "Demon"}]}, campaign_navigation=True)
        schema = semantic_schema(context, "advance")

        self.assertIn("fight", schema["immediate_mode"]["choices"])
        self.assertEqual(["advance", "retreat", "engage", "hold"], schema["move"]["choices"])

    def test_campaign_navigation_probes_a_blocked_surface_with_use(self) -> None:
        context = semantic_context(
            {},
            progress_feedback={"status": "blocked", "obstruction_direction": "forward"},
            campaign_navigation=True,
        )

        self.assertEqual(["use"], semantic_schema(context, "advance")["use"]["choices"])

    def test_campaign_navigation_does_not_use_without_a_blocked_surface(self) -> None:
        context = semantic_context({}, campaign_navigation=True)

        self.assertEqual(["no_use"], semantic_schema(context, "advance")["use"]["choices"])

    def test_close_hostile_allows_evasion(self) -> None:
        context = semantic_context({"visible_hostiles": [{"id": 7, "name": "Demon", "distance": 100}]})

        choices = semantic_schema(context, "advance")["immediate_mode"]["choices"]

        self.assertIn("evade", choices)

    def test_weapon_switching_is_unavailable_while_ammunition_remains(self) -> None:
        context = semantic_context({"player": {"weapon_ammo": 12}})

        choices = semantic_schema(context, "advance")["weapon"]["choices"]

        self.assertEqual(["keep"], choices)

    def test_weapon_switching_is_available_when_current_weapon_is_empty(self) -> None:
        context = semantic_context({"player": {"weapon_ammo": 0}})

        choices = semantic_schema(context, "advance")["weapon"]["choices"]

        self.assertEqual(["keep", "next", "previous"], choices)

    def test_recent_damage_requires_a_dodge_direction(self) -> None:
        context = semantic_context(
            {
                "visible_hostiles": [{"id": 7, "name": "Demon"}],
                "combat": {"health_change": -5},
            }
        )

        schema = semantic_schema(context, "advance")

        self.assertEqual({"active": True, "reason": "recent_damage"}, context["combat_pressure"])
        self.assertEqual(["left", "right"], schema["dodge"]["choices"])

    def test_visible_explosive_requires_a_dodge_direction(self) -> None:
        context = semantic_context(
            {
                "visible_hostiles": [{"id": 7, "name": "Demon"}],
                "visible_resources_and_objects": [{"category": "explosive", "distance": 48}],
            }
        )

        schema = semantic_schema(context, "advance")

        self.assertEqual("visible_explosive", context["combat_pressure"]["reason"])
        self.assertEqual(["left", "right"], schema["dodge"]["choices"])

    def test_close_hostile_requires_a_dodge_direction(self) -> None:
        context = semantic_context(
            {
                "visible_hostiles": [{"id": 7, "name": "Demon", "distance": 120}],
            }
        )

        schema = semantic_schema(context, "advance")

        self.assertEqual("close_hostile", context["combat_pressure"]["reason"])
        self.assertEqual(["left", "right"], schema["dodge"]["choices"])


class SemanticMotorTests(unittest.TestCase):
    def test_tracks_selected_target_and_only_fires_when_aligned(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        intent = SemanticIntent(face="hostile:7", move="advance", trigger="fire")
        motor.commit(intent, now=10.0)

        turning = motor.action(observation(bearing=20), now=10.1)
        aligned = motor.action(observation(bearing=2), now=10.2)

        self.assertEqual("hard_left", turning.turn)
        self.assertFalse(turning.fire)
        self.assertEqual("hold", aligned.turn)
        self.assertTrue(aligned.fire)
        self.assertEqual("forward", aligned.movement)

    def test_withholds_fire_when_selected_target_leaves_view(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(face="hostile:7", trigger="fire"), now=10.0)

        action = motor.action(observation(include_hostile=False), now=10.1)

        self.assertFalse(action.fire)
        self.assertEqual("hold", action.turn)

    def test_combines_longitudinal_movement_and_dodge(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(move="advance", dodge="right"), now=10.0)

        action = motor.action(observation(), now=10.1)

        self.assertEqual("forward", action.movement)
        self.assertEqual("right", action.strafe)

    def test_engage_maintains_range_from_selected_hostile(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(face="hostile:7", move="engage", dodge="left"), now=10.0)

        close_state = observation(bearing=0)
        close_state["visible_objects"][0]["distance"] = 50
        close = motor.action(close_state, now=10.1)
        medium_state = observation(bearing=0)
        medium_state["visible_objects"][0]["distance"] = 200
        medium = motor.action(medium_state, now=10.2)
        far_state = observation(bearing=0)
        far_state["visible_objects"][0]["distance"] = 300
        far = motor.action(far_state, now=10.3)

        self.assertEqual("backward", close.movement)
        self.assertEqual("stop", medium.movement)
        self.assertEqual("forward", far.movement)
        self.assertEqual("left", medium.strafe)

    def test_fight_mode_uses_engagement_movement_and_fire_permission(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(mode="fight", face="hostile:7"), now=10.0)
        close_state = observation(bearing=0)
        close_state["visible_objects"][0]["distance"] = 50

        action = motor.action(close_state, now=10.1)

        self.assertEqual("backward", action.movement)
        self.assertTrue(action.fire)

    def test_critical_health_breaks_contact_with_a_visible_hostile(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0, survival_retreat_health=35)
        motor.commit(SemanticIntent(mode="fight", face="hostile:7", dodge="left"), now=10.0)
        state = observation(bearing=0)
        state["player"]["health"] = 30

        action = motor.action(state, now=10.1)

        self.assertEqual("backward", action.movement)
        self.assertEqual("right", action.strafe)
        self.assertTrue(action.fire)

    def test_survival_retreat_releases_when_the_hostile_leaves_view(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0, survival_retreat_health=35)
        motor.commit(SemanticIntent(mode="fight", face="hostile:7"), now=10.0)
        state = observation(bearing=0)
        state["player"]["health"] = 30
        motor.action(state, now=10.1)

        released = motor.action(observation(include_hostile=False), now=10.2)

        self.assertEqual("stop", released.movement)

    def test_requested_fire_can_target_an_aligned_hostile_in_a_crowd(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(mode="fight", face="hostile:7", trigger="fire"), now=10.0)
        crowded_state = observation(bearing=20)
        crowded_state["visible_objects"].append(
            {
                "id": 8,
                "name": "Demon",
                "category": "Monster",
                "distance": 150,
                "relative_bearing_degrees": 8,
            }
        )

        action = motor.action(crowded_state, now=10.1)

        self.assertTrue(action.fire)

    def test_hold_fire_does_not_target_an_aligned_secondary_hostile(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(mode="fight", face="hostile:7", trigger="hold_fire"), now=10.0)
        crowded_state = observation(bearing=20)
        crowded_state["visible_objects"].append(
            {
                "id": 8,
                "name": "Demon",
                "category": "Monster",
                "distance": 150,
                "relative_bearing_degrees": 2,
            }
        )

        action = motor.action(crowded_state, now=10.1)

        self.assertFalse(action.fire)

    def test_fight_holds_forward_movement_until_a_far_target_is_centered(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(mode="fight", face="hostile:7", dodge="left"), now=10.0)
        off_axis = observation(bearing=20)
        off_axis["visible_objects"][0]["distance"] = 300
        centered = observation(bearing=0)
        centered["visible_objects"][0]["distance"] = 300

        turning = motor.action(off_axis, now=10.1)
        advancing = motor.action(centered, now=10.2)

        self.assertEqual("stop", turning.movement)
        self.assertEqual("stop", turning.strafe)
        self.assertEqual("hard_left", turning.turn)
        self.assertEqual("forward", advancing.movement)
        self.assertEqual("left", advancing.strafe)

    def test_fight_backpedals_and_keeps_model_selected_strafe_when_a_close_hostile_is_off_axis(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(mode="fight", face="hostile:7", dodge="left"), now=10.0)
        off_axis = observation(bearing=30)
        off_axis["visible_objects"][0]["distance"] = 120

        action = motor.action(off_axis, now=10.1)

        self.assertEqual("backward", action.movement)
        self.assertEqual("left", action.strafe)
        self.assertEqual("hard_left", action.turn)

    def test_exploration_backpedals_and_fires_at_a_centered_new_hostile(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(mode="explore", move="advance"), now=10.0)
        motor.observe(observation(bearing=0), now=10.05, applied_action=GameAction(movement="forward"))

        action = motor.action(observation(bearing=0), now=10.1)

        self.assertEqual("backward", action.movement)
        self.assertEqual("right", action.strafe)
        self.assertEqual("hold", action.turn)
        self.assertTrue(action.fire)

    def test_exploration_turns_toward_but_does_not_fire_at_an_off_axis_hostile(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(mode="explore", move="advance"), now=10.0)
        motor.observe(observation(bearing=-20), now=10.05, applied_action=GameAction(movement="forward"))

        action = motor.action(observation(bearing=-20), now=10.1)

        self.assertEqual("backward", action.movement)
        self.assertEqual("left", action.strafe)
        self.assertEqual("hard_right", action.turn)
        self.assertFalse(action.fire)

    def test_fight_mode_tracks_a_just_lost_selected_target_without_firing(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        seen = observation(bearing=0)
        seen["player"].update({"position": [0, 0, 0], "heading_degrees": 0})
        seen["visible_objects"][0]["world_position"] = [100, 0, 0]
        motor.observe(seen, now=10.0, applied_action=GameAction())
        motor.commit(SemanticIntent(mode="fight", face="hostile:7"), now=10.1)
        lost = observation(include_hostile=False)
        lost["player"].update({"position": [0, 0, 0], "heading_degrees": 30})

        action = motor.action(lost, now=10.2)

        self.assertEqual("hard_right", action.turn)
        self.assertEqual("stop", action.movement)
        self.assertFalse(action.fire)

    def test_engage_holds_when_the_selected_hostile_leaves_view(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(face="hostile:7", move="engage"), now=10.0)

        action = motor.action(observation(include_hostile=False), now=10.1)

        self.assertEqual("stop", action.movement)

    def test_clear_space_steers_toward_stronger_depth_signal(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(face="clear_space", move="advance"), now=10.0)

        action = motor.action(observation(), now=10.1)

        self.assertEqual("soft_left", action.turn)

    def test_explore_frontier_prefers_an_open_less_visited_side(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(face="explore_frontier", move="advance"), now=10.0)
        state = observation(include_hostile=False)
        state["player"].update({"position": [0, 0, 0], "heading_degrees": 0})
        state["depth_grid"]["values"] = [[12, 12, 12, 12, 48, 48], [0, 0, 0, 0, 0, 0]]

        motor.observe(state, now=10.0, applied_action=GameAction())
        action = motor.action(state, now=10.1)

        self.assertEqual("soft_right", action.turn)
        self.assertEqual("available", motor.navigation_summary()["status"])

    def test_recent_threat_turns_without_firing_or_advancing(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        seen = observation(bearing=0)
        seen["player"].update({"position": [0, 0, 0], "heading_degrees": 0})
        seen["visible_objects"][0]["world_position"] = [100, 0, 0]
        motor.observe(seen, now=10.0, applied_action=GameAction())
        motor.commit(SemanticIntent(face="recent_hostile:7", move="engage", trigger="fire"), now=10.1)
        lost = observation(include_hostile=False)
        lost["player"].update({"position": [0, 0, 0], "heading_degrees": 30})

        action = motor.action(lost, now=10.2)

        self.assertEqual("hard_right", action.turn)
        self.assertEqual("stop", action.movement)
        self.assertFalse(action.fire)

    def test_backtrack_and_turn_runs_a_bounded_recovery(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(face="backtrack_and_turn"), now=10.0)

        recovery = motor.action(observation(include_hostile=False), now=10.1)
        released = motor.action(observation(include_hostile=False), now=11.3)

        self.assertEqual("backward", recovery.movement)
        self.assertTrue(recovery.turn.startswith("hard_"))
        self.assertEqual("stop", released.movement)

    def test_executes_a_short_level_two_navigation_directive(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.apply_navigation_directive("turn_hard_right", duration_seconds=1, now=10.0)

        action = motor.action(observation(include_hostile=False), now=10.5)

        self.assertEqual("forward", action.movement)
        self.assertEqual("hard_right", action.turn)

    def test_does_not_apply_navigation_directive_over_a_visible_hostile(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(mode="fight", face="hostile:7"), now=10.0)
        motor.apply_navigation_directive("turn_hard_right", duration_seconds=1, now=10.0)

        action = motor.action(observation(bearing=20), now=10.5)

        self.assertEqual("hard_left", action.turn)

    def test_exposes_blocked_progress_feedback(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        state = observation(include_hostile=False)
        state["player"]["position"] = [0, 0, 0]
        motor.observe(state, now=10.0, applied_action=GameAction(movement="forward"))
        motor.observe(state, now=10.5, applied_action=GameAction(movement="forward"))

        self.assertEqual("blocked", motor.progress_feedback()["status"])

    def test_blocked_forward_progress_triggers_a_short_recovery_without_hostiles(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        state = observation(include_hostile=False)
        state["player"]["position"] = [0, 0, 0]
        motor.observe(state, now=10.0, applied_action=GameAction(movement="forward"))
        motor.observe(state, now=10.5, applied_action=GameAction(movement="forward"))

        action = motor.action(state, now=10.6)

        self.assertEqual("backward", action.movement)
        self.assertIn(action.strafe, {"left", "right"})
        self.assertEqual(f"hard_{action.strafe}", action.turn)

    def test_campaign_mode_tests_a_blocked_surface_before_recovery(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0, campaign_navigation=True)
        state = observation(include_hostile=False)
        state["player"]["position"] = [0, 0, 0]
        motor.observe(state, now=10.0, applied_action=GameAction(movement="forward"))
        motor.observe(state, now=10.5, applied_action=GameAction(movement="forward"))

        action = motor.action(state, now=10.6)

        self.assertEqual("forward", action.movement)
        self.assertTrue(action.use)

    def test_campaign_mode_does_not_use_while_unobstructed(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0, campaign_navigation=True)
        motor.commit(SemanticIntent(mode="explore", move="advance"), now=10.0)

        action = motor.action(observation(include_hostile=False), now=10.1)

        self.assertFalse(action.use)

    def test_blocked_recovery_does_not_override_visible_combat(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        blocked_state = observation(include_hostile=False)
        blocked_state["player"]["position"] = [0, 0, 0]
        motor.observe(blocked_state, now=10.0, applied_action=GameAction(movement="forward"))
        motor.observe(blocked_state, now=10.5, applied_action=GameAction(movement="forward"))
        motor.commit(SemanticIntent(mode="fight", face="hostile:7"), now=10.5)
        combat_state = observation(bearing=0)
        combat_state["visible_objects"][0]["distance"] = 300

        action = motor.action(combat_state, now=10.6)

        self.assertEqual("forward", action.movement)
        self.assertTrue(action.fire)

    def test_clear_space_escapes_an_enclosed_corner(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(face="clear_space", move="advance"), now=10.0)

        action = motor.action(enclosed_observation(), now=10.1)

        self.assertEqual("backward", action.movement)
        self.assertEqual("right", action.strafe)
        self.assertEqual("hard_right", action.turn)

    def test_repeated_corner_escape_performs_a_full_dead_end_reversal(self) -> None:
        motor = SemanticMotor(lease_seconds=5.0)
        motor.commit(SemanticIntent(face="clear_space", move="advance"), now=10.0)
        motor.action(enclosed_observation(), now=10.1)
        second = motor.action(enclosed_observation(), now=11.0)

        self.assertEqual("dead_end_reversal", motor.escape_mode)
        self.assertEqual("backward", second.movement)
        self.assertEqual("stop", second.strafe)
        self.assertEqual("hard_left", second.turn)

    def test_new_clear_space_intent_does_not_cancel_an_active_escape(self) -> None:
        motor = SemanticMotor(lease_seconds=5.0)
        motor.commit(SemanticIntent(face="clear_space", move="advance"), now=10.0)
        motor.action(enclosed_observation(), now=10.1)
        motor.commit(SemanticIntent(face="clear_space", move="advance"), now=10.2)

        action = motor.action(enclosed_observation(), now=10.3)

        self.assertTrue(motor.is_escaping(10.3))
        self.assertEqual("backward", action.movement)

    def test_clear_space_does_not_escape_when_one_side_is_open(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(face="clear_space", move="advance"), now=10.0)

        action = motor.action(observation(), now=10.1)

        self.assertEqual("forward", action.movement)

    def test_corner_escape_does_not_override_a_non_clear_space_intent(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(face="hold", move="advance"), now=10.0)

        action = motor.action(enclosed_observation(), now=10.1)

        self.assertEqual("forward", action.movement)
        self.assertEqual("hold", action.turn)

    def test_releases_controls_after_lease_expires(self) -> None:
        motor = SemanticMotor(lease_seconds=2.0)
        motor.commit(SemanticIntent(move="advance", dodge="left"), now=10.0)

        action = motor.action(observation(), now=12.1)

        self.assertEqual("stop", action.movement)
        self.assertEqual("stop", action.strafe)


if __name__ == "__main__":
    unittest.main()
