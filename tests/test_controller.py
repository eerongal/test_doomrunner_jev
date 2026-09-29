import unittest
from types import SimpleNamespace

from doomrunner.controller import (
    DoomController,
    decision_application_snapshot,
    fight_target_is_stale,
    fight_target_requires_requery,
    heuristic_action,
    is_short_recovery_plan,
    noncombat_intent_requires_requery,
    planner_combat_active,
    planner_event,
    planner_health_is_safe,
    normalize_idle_hold_plan,
    planner_observation,
)
from doomrunner.inference import Plan
from doomrunner.semantic import SemanticIntent


class PlannerEventTests(unittest.TestCase):
    def test_requests_plan_after_heavy_damage(self) -> None:
        observation = {
            "player": {"health": 70},
            "combat": {"health_change": -20},
        }

        self.assertEqual("heavy_damage", planner_event(observation, 15, 30))

    def test_critical_health_takes_priority(self) -> None:
        observation = {
            "player": {"health": 25},
            "combat": {"health_change": -5},
        }

        self.assertEqual("critical_health", planner_event(observation, 15, 30))

    def test_ignores_small_or_absent_health_changes(self) -> None:
        small_hit = {"player": {"health": 80}, "combat": {"health_change": -5}}
        unchanged = {"player": {"health": 80}, "combat": {"health_change": 0}}

        self.assertIsNone(planner_event(small_hit, 15, 30))
        self.assertIsNone(planner_event(unchanged, 15, 30))

    def test_planner_failure_backoff_suppresses_event_retries(self) -> None:
        controller = object.__new__(DoomController)
        controller.mode = "jev"
        controller.config = SimpleNamespace(
            controller=SimpleNamespace(
                planner_enabled=True,
                planner_damage_threshold=15,
                planner_critical_health=30,
                serialize_inference_requests=True,
                observation_mode="raw",
            )
        )
        controller._planner_task = None
        controller._decision_task = None
        controller._planner_retry_at = 20.0
        controller._next_plan_at = 0.0
        controller._has_requested_plan = True
        observation = {"player": {"health": 20}, "combat": {"health_change": -20}}

        self.assertIsNone(controller._planner_trigger(observation, 19.0))
        self.assertEqual("critical_health", controller._planner_trigger(observation, 20.0))

    def test_serialized_planner_waits_for_level_one_request(self) -> None:
        controller = object.__new__(DoomController)
        controller.mode = "jev"
        controller.config = SimpleNamespace(
            controller=SimpleNamespace(
                planner_enabled=True,
                planner_damage_threshold=15,
                planner_critical_health=30,
                serialize_inference_requests=True,
                observation_mode="raw",
            )
        )
        controller._planner_task = None
        controller._decision_task = object()
        controller._planner_retry_at = 0.0
        controller._next_plan_at = 0.0
        controller._next_navigation_recovery_plan_at = float("inf")
        controller._has_requested_plan = False
        observation = {"player": {"health": 100}, "combat": {"health_change": 0}}

        self.assertIsNone(controller._planner_trigger(observation, 1.0))

    def test_initial_plan_starts_despite_active_combat_and_low_health(self) -> None:
        controller = object.__new__(DoomController)
        controller.mode = "jev"
        controller.config = SimpleNamespace(
            controller=SimpleNamespace(
                planner_enabled=True,
                planner_damage_threshold=15,
                planner_critical_health=30,
                planner_min_health_for_background_work=60,
                serialize_inference_requests=False,
                observation_mode="semantic",
            )
        )
        controller._planner_task = None
        controller._decision_task = None
        controller._planner_retry_at = 0.0
        controller._next_plan_at = 0.0
        controller._next_navigation_recovery_plan_at = float("inf")
        controller._has_requested_plan = False
        controller._has_accepted_decision = True
        controller._planner_calm_after = 60.0
        controller._pending_planner_trigger = None
        controller.telemetry = SimpleNamespace(write=lambda *args, **kwargs: None)
        observation = {"player": {"health": 20}, "combat": {"health_change": -20}}

        self.assertEqual("episode_start", controller._planner_trigger(observation, 1.0))

    def test_blocked_progress_triggers_a_rate_limited_level_two_plan(self) -> None:
        controller = object.__new__(DoomController)
        controller.mode = "jev"
        controller.config = SimpleNamespace(
            controller=SimpleNamespace(
                planner_enabled=True,
                planner_damage_threshold=15,
                planner_critical_health=30,
                serialize_inference_requests=False,
                observation_mode="semantic",
            )
        )
        controller._planner_task = None
        controller._decision_task = None
        controller._planner_retry_at = 0.0
        controller._next_navigation_recovery_plan_at = 10.0
        controller._next_plan_at = 30.0
        controller._has_requested_plan = True
        controller.semantic_motor = SimpleNamespace(progress_feedback=lambda: {"status": "blocked"})
        observation = {"player": {"health": 100}, "combat": {"health_change": 0}}

        self.assertIsNone(controller._planner_trigger(observation, 9.0))
        self.assertEqual("blocked_progress", controller._planner_trigger(observation, 10.0))

    def test_wall_following_triggers_a_navigation_replan(self) -> None:
        controller = object.__new__(DoomController)
        controller.mode = "jev"
        controller.config = SimpleNamespace(
            controller=SimpleNamespace(
                planner_enabled=True,
                planner_damage_threshold=15,
                planner_critical_health=30,
                serialize_inference_requests=False,
                observation_mode="semantic",
            )
        )
        controller._planner_task = None
        controller._decision_task = None
        controller._planner_retry_at = 0.0
        controller._next_navigation_recovery_plan_at = 0.0
        controller._next_plan_at = 30.0
        controller._has_requested_plan = True
        controller.semantic_motor = SimpleNamespace(progress_feedback=lambda: {"status": "wall_following"})
        observation = {"player": {"health": 100}, "combat": {"health_change": 0}}

        self.assertEqual("wall_following", controller._planner_trigger(observation, 1.0))

    def test_defers_the_highest_priority_plan_until_combat_is_quiet(self) -> None:
        controller = object.__new__(DoomController)
        controller.mode = "jev"
        controller.config = SimpleNamespace(
            controller=SimpleNamespace(
                planner_enabled=True,
                planner_damage_threshold=15,
                planner_critical_health=30,
                serialize_inference_requests=False,
                observation_mode="raw",
            )
        )
        controller._planner_task = None
        controller._decision_task = None
        controller._planner_retry_at = 0.0
        controller._next_plan_at = 0.0
        controller._has_requested_plan = True
        controller._planner_calm_after = 20.0
        controller._pending_planner_trigger = None
        controller.telemetry = SimpleNamespace(write=lambda *args, **kwargs: None)
        observation = {"player": {"health": 20}, "combat": {"health_change": -20}}

        self.assertIsNone(controller._planner_trigger(observation, 10.0))
        self.assertEqual("critical_health", controller._pending_planner_trigger)
        self.assertEqual("critical_health", controller._planner_trigger(observation, 20.0))

    def test_dedicated_planner_runs_during_combat_and_low_health(self) -> None:
        controller = object.__new__(DoomController)
        controller.mode = "jev"
        controller.config = SimpleNamespace(
            server=SimpleNamespace(planner_base_url="http://127.0.0.1:8097"),
            controller=SimpleNamespace(
                planner_enabled=True,
                planner_damage_threshold=15,
                planner_critical_health=30,
                planner_min_health_for_background_work=60,
                serialize_inference_requests=True,
                observation_mode="raw",
            ),
        )
        controller._planner_task = None
        controller._decision_task = object()
        controller._planner_retry_at = 0.0
        controller._next_plan_at = 0.0
        controller._has_requested_plan = True
        controller._planner_calm_after = 20.0
        controller._pending_planner_trigger = None
        observation = {"player": {"health": 20}, "combat": {"health_change": -20}}

        self.assertEqual("critical_health", controller._planner_trigger(observation, 10.0))

    def test_combat_window_tracks_visible_hostiles_and_fresh_damage(self) -> None:
        hostile = {"visible_objects": [{"category": "Monster"}], "combat": {"health_change": 0}}
        damaged = {"visible_objects": [], "combat": {"health_change": -1}}
        quiet = {"visible_objects": [], "combat": {"health_change": 0}}
        initial = {"visible_objects": [], "combat": {"health_change": None}}

        self.assertTrue(planner_combat_active(hostile))
        self.assertTrue(planner_combat_active(damaged))
        self.assertFalse(planner_combat_active(quiet))
        self.assertFalse(planner_combat_active(initial))

    def test_defers_pending_plans_below_the_background_health_threshold(self) -> None:
        controller = object.__new__(DoomController)
        controller.mode = "jev"
        controller.config = SimpleNamespace(
            controller=SimpleNamespace(
                planner_enabled=True,
                planner_damage_threshold=15,
                planner_critical_health=30,
                planner_min_health_for_background_work=60,
                serialize_inference_requests=False,
                observation_mode="raw",
            )
        )
        controller._planner_task = None
        controller._decision_task = None
        controller._planner_retry_at = 0.0
        controller._next_plan_at = 0.0
        controller._has_requested_plan = False
        controller._has_accepted_decision = True
        controller._planner_calm_after = float("-inf")
        controller._pending_planner_trigger = None
        controller.telemetry = SimpleNamespace(write=lambda *args, **kwargs: None)
        low_health = {"player": {"health": 40}, "combat": {"health_change": 0}}
        recovered = {"player": {"health": 60}, "combat": {"health_change": 0}}

        self.assertFalse(planner_health_is_safe(low_health, 60))
        self.assertEqual("episode_start", controller._planner_trigger(low_health, 1.0))
        controller._has_requested_plan = True
        self.assertIsNone(controller._planner_trigger(low_health, 2.0))
        self.assertEqual("periodic", controller._pending_planner_trigger)
        self.assertEqual("periodic", controller._planner_trigger(recovered, 3.0))

    def test_planner_snapshot_includes_semantic_navigation_feedback(self) -> None:
        controller = object.__new__(DoomController)
        controller.config = SimpleNamespace(controller=SimpleNamespace(observation_mode="semantic"))
        controller.semantic_motor = SimpleNamespace(
            progress_feedback=lambda: {"status": "blocked"},
            navigation_summary=lambda: {"preferred_direction": "left"},
        )
        observation = {"player": {"health": 100}}

        snapshot = controller._planner_snapshot(observation)

        self.assertEqual({"player": {"health": 100}}, observation)
        self.assertEqual("blocked", snapshot["controller_feedback"]["movement_feedback"]["status"])
        self.assertEqual("left", snapshot["controller_feedback"]["local_navigation"]["preferred_direction"])

    def test_campaign_planner_snapshot_declares_level_progress_mission(self) -> None:
        controller = object.__new__(DoomController)
        controller.config = SimpleNamespace(
            controller=SimpleNamespace(observation_mode="raw", campaign_navigation=True)
        )

        snapshot = controller._planner_snapshot({"player": {"health": 100}})

        self.assertEqual("campaign_navigation", snapshot["mission_profile"])

    def test_planner_snapshot_uses_aggregate_object_counts(self) -> None:
        observation = {
            "player": {"health": 100},
            "visible_objects": [{"id": 1, "name": "Demon", "category": "Monster"}],
            "visible_object_counts": [{"name": "Stimpack", "category": "Health", "count": 20}],
        }
        controller = object.__new__(DoomController)
        controller.config = SimpleNamespace(controller=SimpleNamespace(observation_mode="raw"))

        snapshot = controller._planner_snapshot(observation)

        self.assertEqual([{"name": "Stimpack", "category": "Health", "count": 20}], snapshot["visible_object_counts"])
        self.assertNotIn("visible_objects", snapshot)

    def test_planner_snapshot_excludes_decorative_gore(self) -> None:
        observation = {
            "visible_object_counts": [
                {"name": "DeadMarine", "category": "Gore", "count": 1},
                {"name": "ArmorBonus", "category": "Armor", "count": 2},
            ]
        }

        snapshot = planner_observation(observation)

        self.assertEqual([{"name": "ArmorBonus", "category": "Armor", "count": 2}], snapshot["visible_object_counts"])

    def test_replaces_an_unjustified_idle_hold_with_exploration(self) -> None:
        plan = Plan.from_dict({"mobility_policy": "hold_position", "immediate_goal": "Wait for enemies."})
        observation = {
            "player": {"health": 100},
            "combat": {"health_change": 0},
            "visible_objects": [],
            "controller_feedback": {"movement_feedback": {"status": "wall_following", "turn_direction": "right"}},
        }

        normalized = normalize_idle_hold_plan(plan, observation)

        self.assertEqual("advance", normalized.mobility_policy)
        self.assertEqual("backtrack_right", normalized.navigation_directive)
        self.assertIn("do not wait", normalized.immediate_goal)

    def test_preserves_a_recovery_directive_when_replacing_an_idle_hold(self) -> None:
        plan = Plan.from_dict(
            {
                "mobility_policy": "hold_position",
                "navigation_directive": "turn_hard_right",
                "navigation_directive_seconds": 2,
            }
        )

        normalized = normalize_idle_hold_plan(plan, {"player": {"health": 100}, "visible_objects": []})

        self.assertEqual("advance", normalized.mobility_policy)
        self.assertEqual("turn_hard_right", normalized.navigation_directive)
        self.assertEqual(2, normalized.navigation_directive_seconds)

    def test_identifies_a_short_recovery_plan(self) -> None:
        plan = Plan.from_dict(
            {
                "immediate_goal": "Recover from blocked movement",
                "navigation_directive": "backtrack_left",
                "navigation_directive_seconds": 2,
            }
        )

        self.assertTrue(is_short_recovery_plan(plan))

    def test_retains_a_hold_plan_when_health_is_low(self) -> None:
        plan = Plan.from_dict({"mobility_policy": "hold_position"})

        normalized = normalize_idle_hold_plan(plan, {"player": {"health": 30}, "visible_objects": []})

        self.assertEqual(plan, normalized)


class HeuristicAimingTests(unittest.TestCase):
    def test_turns_left_for_positive_relative_bearing(self) -> None:
        observation = {
            "visible_objects": [
                {"id": 1, "category": "Monster", "relative_bearing_degrees": 30},
            ]
        }

        self.assertEqual("hard_left", heuristic_action(observation).turn)

    def test_turns_right_for_negative_relative_bearing(self) -> None:
        observation = {
            "visible_objects": [
                {"id": 1, "category": "Monster", "relative_bearing_degrees": -30},
            ]
        }

        self.assertEqual("hard_right", heuristic_action(observation).turn)


class StaleFightTargetTests(unittest.TestCase):
    def test_retains_a_recent_track_when_its_selected_target_is_gone(self) -> None:
        intent = SemanticIntent(mode="fight", face="hostile:7")

        self.assertTrue(fight_target_is_stale(intent, {"visible_objects": []}))
        self.assertFalse(fight_target_requires_requery(intent, {"visible_objects": []}))

    def test_keeps_fight_when_its_selected_target_is_visible(self) -> None:
        intent = SemanticIntent(mode="fight", face="hostile:7")
        observation = {"visible_objects": [{"id": 7, "category": "Monster"}]}

        self.assertFalse(fight_target_is_stale(intent, observation))

    def test_requeries_when_a_different_hostile_is_visible(self) -> None:
        intent = SemanticIntent(mode="fight", face="hostile:7")
        observation = {"visible_objects": [{"id": 8, "category": "Monster"}]}

        self.assertTrue(fight_target_requires_requery(intent, observation))

    def test_requeries_a_late_exploration_intent_when_a_hostile_is_visible(self) -> None:
        intent = SemanticIntent(mode="explore")
        observation = {"visible_objects": [{"id": 8, "category": "Monster"}]}

        self.assertTrue(noncombat_intent_requires_requery(intent, observation))

    def test_keeps_a_combat_intent_when_a_hostile_is_visible(self) -> None:
        intent = SemanticIntent(mode="fight", face="hostile:8")
        observation = {"visible_objects": [{"id": 8, "category": "Monster"}]}

        self.assertFalse(noncombat_intent_requires_requery(intent, observation))


class DecisionTelemetryTests(unittest.TestCase):
    def test_captures_live_hostiles_when_a_decision_is_applied(self) -> None:
        observation = {
            "tic": 42,
            "player": {"health": 88, "heading_degrees": 90.0},
            "visible_objects": [
                {"id": 7, "name": "Demon", "category": "Monster", "distance": 95.2, "relative_bearing_degrees": -4.0},
                {"id": 8, "name": "HealthBonus", "category": "Item", "distance": 20},
            ],
        }

        snapshot = decision_application_snapshot(observation)

        self.assertEqual(42, snapshot["tic"])
        self.assertEqual(88, snapshot["player"]["health"])
        self.assertEqual([{"id": 7, "name": "Demon", "distance": 95.2, "relative_bearing_degrees": -4.0}], snapshot["visible_hostiles"])


if __name__ == "__main__":
    unittest.main()
