import os
import unittest
from unittest.mock import MagicMock

import httpx

from doomrunner.config import ServerConfig
from doomrunner.inference import LlamaClient, Plan, _extract_json_object, message_text, raise_for_status_with_body, validate_plan_text


class PlanTests(unittest.TestCase):
    def test_defaults_to_exploratory_bootstrap_plan(self) -> None:
        plan = Plan.from_dict({})

        self.assertEqual("advance", plan.mobility_policy)

    def test_accepts_hold_position_policy(self) -> None:
        plan = Plan.from_dict({"mobility_policy": "hold_position"})

        self.assertEqual("hold_position", plan.mobility_policy)
        self.assertIn('"mobility_policy":"hold_position"', plan.prompt_text())

    def test_replaces_unknown_mobility_policy(self) -> None:
        plan = Plan.from_dict({"mobility_policy": "teleport"})

        self.assertEqual("free", plan.mobility_policy)

    def test_replaces_null_plan_text_with_a_safe_default(self) -> None:
        plan = Plan.from_dict({"goal": None, "immediate_goal": None})

        self.assertNotEqual("None", plan.goal)
        self.assertNotEqual("None", plan.immediate_goal)

    def test_retains_layered_goals_in_the_level_one_plan(self) -> None:
        plan = Plan.from_dict(
            {
                "immediate_goal": "Turn away from the wall and backtrack into open space.",
                "near_term_goal": "Reach the arena center.",
                "long_term_goal": "Survive and clear threats.",
            }
        )

        prompt = plan.prompt_text()

        self.assertIn('"immediate_goal":"Turn away from the wall and backtrack into open space."', prompt)
        self.assertIn('"near_term_goal":"Reach the arena center."', prompt)
        self.assertIn('"long_term_goal":"Survive and clear threats."', prompt)

    def test_compact_level_one_plan_keeps_only_immediate_controls(self) -> None:
        plan = Plan.from_dict(
            {
                "immediate_goal": "Turn away from the wall.",
                "near_term_goal": "Reach the arena center.",
                "long_term_goal": "Survive and clear threats.",
                "navigation_directive": "backtrack_left",
            }
        )

        prompt = plan.level_one_prompt_text()

        self.assertIn('"immediate_goal":"Turn away from the wall."', prompt)
        self.assertIn('"navigation_directive":"backtrack_left"', prompt)
        self.assertNotIn("near_term_goal", prompt)
        self.assertNotIn("long_term_goal", prompt)

    def test_accepts_a_bounded_navigation_directive(self) -> None:
        plan = Plan.from_dict({"navigation_directive": "turn_hard_right", "navigation_directive_seconds": 8})

        self.assertEqual("turn_hard_right", plan.navigation_directive)
        self.assertEqual(3, plan.navigation_directive_seconds)


class PlannerRequestTests(unittest.TestCase):
    def test_disables_thinking_and_enables_json_object_grammar(self) -> None:
        client = LlamaClient(ServerConfig())
        response = MagicMock()
        response.json.return_value = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"goal":"survive","immediate_goal":"fight visible threats",'
                            '"near_term_goal":"reach open space","long_term_goal":"clear the arena",'
                            '"mobility_policy":"free","priorities":[],'
                            '"avoid":[],"tactical_rule":"","expires_after_seconds":30}'
                        )
                    },
                }
            ]
        }
        client.http = MagicMock()
        client.planner_http = client.http
        client.http.post.return_value = response

        client.plan({}, Plan(), [], "episode_start")

        request = client.http.post.call_args.kwargs["json"]
        self.assertEqual(30.0, client.http.post.call_args.kwargs["timeout"])
        self.assertEqual({"enable_thinking": False}, request["chat_template_kwargs"])
        self.assertEqual("none", request["reasoning_effort"])
        self.assertEqual("auto", request["reasoning_format"])
        self.assertEqual(768, request["max_tokens"])
        self.assertEqual({"type": "json_object"}, request["response_format"])
        self.assertIn("Do not reuse the current_plan wording", request["messages"][0]["content"])
        self.assertIn("gore, and decorations are never enemies", request["messages"][0]["content"])
        self.assertIn("do not establish a defensive perimeter", request["messages"][0]["content"])
        self.assertIn("test for a door or switch", request["messages"][0]["content"])

    def test_extracts_json_from_a_fenced_planner_response(self) -> None:
        result = _extract_json_object('Plan:\n```json\n{"mobility_policy":"advance"}\n```')

        self.assertEqual("advance", result["mobility_policy"])

    def test_extracts_text_from_openrouter_content_parts(self) -> None:
        result = message_text([{"type": "text", "text": "{\"goal\":\"survive\"}"}])

        self.assertEqual('{"goal":"survive"}', result)

    def test_treats_null_openrouter_content_as_empty(self) -> None:
        self.assertEqual("", message_text(None))

    def test_routes_planning_to_a_dedicated_planner_server(self) -> None:
        client = LlamaClient(ServerConfig(planner_base_url="http://127.0.0.1:8097"))
        response = MagicMock()
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"goal":"survive","immediate_goal":"fight visible threats",'
                            '"near_term_goal":"reach open space","long_term_goal":"clear the arena"}'
                        )
                    }
                }
            ]
        }
        client.http = MagicMock()
        client.planner_http = MagicMock()
        client.planner_http.post.return_value = response

        client.plan({}, Plan(), [], "episode_start")

        client.http.post.assert_not_called()
        client.planner_http.post.assert_called_once()

    def test_routes_openrouter_planning_without_local_only_parameters(self) -> None:
        previous_key = os.environ.get("OPEN_ROUTER_KEY")
        os.environ["OPEN_ROUTER_KEY"] = "test-key"
        try:
            client = LlamaClient(ServerConfig(planner_provider="openrouter", planner_model="stealth/space-bunny-alpha"))
        finally:
            if previous_key is None:
                del os.environ["OPEN_ROUTER_KEY"]
            else:
                os.environ["OPEN_ROUTER_KEY"] = previous_key
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": '{"goal":"survive","immediate_goal":"fight visible threats"}'}}]
        }
        client.planner_http = MagicMock()
        client.planner_http.post.return_value = response

        client.plan({}, Plan(), [], "episode_start")

        request = client.planner_http.post.call_args.kwargs["json"]
        self.assertEqual("chat/completions", client.planner_http.post.call_args.args[0])
        self.assertEqual("stealth/space-bunny-alpha", request["model"])
        self.assertEqual({"type": "json_object"}, request["response_format"])
        self.assertEqual("none", request["reasoning_effort"])
        self.assertNotIn("chat_template_kwargs", request)
        self.assertNotIn("current_plan", request["messages"][1]["content"])
        self.assertIn("navigation_directive must be a non-none", request["messages"][0]["content"])

    def test_omits_openrouter_json_mode_when_the_model_does_not_support_it(self) -> None:
        previous_key = os.environ.get("OPEN_ROUTER_KEY")
        os.environ["OPEN_ROUTER_KEY"] = "test-key"
        try:
            client = LlamaClient(ServerConfig(planner_provider="openrouter", planner_json_mode=False))
        finally:
            if previous_key is None:
                del os.environ["OPEN_ROUTER_KEY"]
            else:
                os.environ["OPEN_ROUTER_KEY"] = previous_key
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": '{"goal":"survive","immediate_goal":"fight visible threats"}'}}]
        }
        client.planner_http = MagicMock()
        client.planner_http.post.return_value = response

        client.plan({}, Plan(), [], "episode_start")

        request = client.planner_http.post.call_args.kwargs["json"]
        self.assertNotIn("response_format", request)
        self.assertNotIn("near_term_goal", request["messages"][0]["content"])

    def test_rejects_empty_required_planner_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "immediate_goal"):
            validate_plan_text({"goal": "survive", "immediate_goal": None})


class SemanticDecisionRequestTests(unittest.TestCase):
    def test_builds_dynamic_semantic_schema_and_parses_intent(self) -> None:
        client = LlamaClient(ServerConfig())
        response = MagicMock()
        response.json.return_value = {
            "results": [
                {
                    "decision": {
                        "immediate_mode": "fight",
                        "face": "hostile:7",
                        "move": "advance",
                        "dodge": "carry_on",
                        "trigger": "fire",
                        "use": "no_use",
                        "weapon": "keep",
                    },
                    "fields": {},
                }
            ]
        }
        client.http = MagicMock()
        client.http.post.return_value = response
        context = {"visible_hostiles": [{"id": 7, "name": "Demon"}]}

        result = client.decide(context, Plan(), "semantic")

        request = client.http.post.call_args.kwargs["json"]
        self.assertIn("hostile:7", request["schema"]["face"]["choices"])
        self.assertEqual("hostile:7", result.intent.face)
        self.assertEqual("fire", result.intent.trigger)

    def test_compact_profile_uses_a_minimal_schema_and_plan(self) -> None:
        client = LlamaClient(ServerConfig(decision_prompt_profile="compact"))
        response = MagicMock()
        response.json.return_value = {
            "results": [
                {
                    "decision": {
                        "immediate_mode": "fight",
                        "face": "hostile:7",
                        "move": "advance",
                        "dodge": "carry_on",
                        "trigger": "fire",
                        "use": "no_use",
                        "weapon": "keep",
                    },
                    "fields": {},
                }
            ]
        }
        client.http = MagicMock()
        client.http.post.return_value = response
        context = {"visible_hostiles": [{"id": 7, "name": "Demon"}]}
        plan = Plan.from_dict({"near_term_goal": "Longer route detail."})

        client.decide(context, plan, "semantic")

        request = client.http.post.call_args.kwargs["json"]
        self.assertEqual("Target or viewing skill.", request["schema"]["face"]["description"])
        self.assertNotIn("near_term_goal", request["instructions"])


class HttpErrorTests(unittest.TestCase):
    def test_includes_server_response_body(self) -> None:
        response = MagicMock()
        response.text = '{"error":"bad allocation"}'
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=MagicMock(),
        )

        with self.assertRaisesRegex(RuntimeError, "bad allocation"):
            raise_for_status_with_body(response)


if __name__ == "__main__":
    unittest.main()
