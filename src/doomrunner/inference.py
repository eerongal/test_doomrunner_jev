from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

from doomrunner.actions import ASSISTED_INSTRUCTIONS, COMPACT_INSTRUCTIONS, RAW_INSTRUCTIONS, GameAction, decision_schema
from doomrunner.config import ServerConfig
from doomrunner.semantic import (
    COMPACT_SEMANTIC_INSTRUCTIONS,
    SEMANTIC_INSTRUCTIONS,
    SemanticIntent,
    semantic_face_choices,
    semantic_dodge_choices,
    semantic_mode_choices,
    semantic_schema,
    semantic_weapon_choices,
)

LOGGER = logging.getLogger(__name__)
MOBILITY_POLICIES = ("free", "hold_position", "seek_cover", "advance", "retreat")
NAVIGATION_DIRECTIVES = ("none", "turn_hard_left", "turn_hard_right", "backtrack_left", "backtrack_right")
PLAN_TEXT_LIMIT = 240
MAX_NAVIGATION_DIRECTIVE_SECONDS = 3
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENVIRONMENT_VARIABLE = "OPEN_ROUTER_KEY"


@dataclass(frozen=True)
class DecisionResult:
    action: GameAction
    confidence: dict[str, float]
    timings: dict[str, Any]
    usage: dict[str, Any]
    intent: SemanticIntent | None = None


@dataclass(frozen=True)
class Plan:
    goal: str = "Explore the area while surviving and defeating immediate threats."
    immediate_goal: str = "Respond to visible threats and current obstructions."
    near_term_goal: str = "Move toward an open position that supports survival and combat."
    long_term_goal: str = "Survive the episode while defeating threats."
    mobility_policy: str = "advance"
    navigation_directive: str = "none"
    navigation_directive_seconds: int = 0
    priorities: tuple[str, ...] = ()
    avoid: tuple[str, ...] = ()
    tactical_rule: str = ""
    expires_after_seconds: int = 30

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Plan":
        mobility_policy = str(value.get("mobility_policy", cls.mobility_policy))
        if mobility_policy not in MOBILITY_POLICIES:
            mobility_policy = "free"
        navigation_directive = str(value.get("navigation_directive", cls.navigation_directive))
        if navigation_directive not in NAVIGATION_DIRECTIVES:
            navigation_directive = "none"
        directive_seconds = navigation_directive_seconds(value.get("navigation_directive_seconds", 0))
        if navigation_directive == "none":
            directive_seconds = 0
        return cls(
            goal=plan_text(value.get("goal"), cls.goal),
            immediate_goal=plan_text(value.get("immediate_goal"), cls.immediate_goal),
            near_term_goal=plan_text(value.get("near_term_goal"), cls.near_term_goal),
            long_term_goal=plan_text(value.get("long_term_goal"), cls.long_term_goal),
            mobility_policy=mobility_policy,
            navigation_directive=navigation_directive,
            navigation_directive_seconds=directive_seconds,
            priorities=tuple(plan_text(item, "") for item in value.get("priorities", []))[:5],
            avoid=tuple(plan_text(item, "") for item in value.get("avoid", []))[:5],
            tactical_rule=plan_text(value.get("tactical_rule"), ""),
            expires_after_seconds=max(5, min(300, int(value.get("expires_after_seconds", 30)))),
        )

    def prompt_text(self) -> str:
        return json.dumps(
            {
                "goal": self.goal,
                "immediate_goal": self.immediate_goal,
                "near_term_goal": self.near_term_goal,
                "long_term_goal": self.long_term_goal,
                "mobility_policy": self.mobility_policy,
                "navigation_directive": self.navigation_directive,
                "navigation_directive_seconds": self.navigation_directive_seconds,
                "priorities": self.priorities,
                "avoid": self.avoid,
                "tactical_rule": self.tactical_rule,
            },
            separators=(",", ":"),
        )

    def level_one_prompt_text(self) -> str:
        return json.dumps(
            {
                "immediate_goal": self.immediate_goal,
                "mobility_policy": self.mobility_policy,
                "navigation_directive": self.navigation_directive,
                "navigation_directive_seconds": self.navigation_directive_seconds,
                "tactical_rule": self.tactical_rule,
            },
            separators=(",", ":"),
        )


def plan_text(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text[:PLAN_TEXT_LIMIT] if text else default


def navigation_directive_seconds(value: Any) -> int:
    try:
        return max(0, min(MAX_NAVIGATION_DIRECTIVE_SECONDS, int(value)))
    except (TypeError, ValueError):
        return 0


class LlamaClient:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.http = httpx.Client(base_url=config.base_url.rstrip("/"), timeout=config.timeout_seconds)
        self._planner_uses_decision_server = config.planner_provider == "local" and not config.planner_base_url
        self.planner_http = self._create_planner_client()

    def _create_planner_client(self) -> httpx.Client:
        if self._planner_uses_decision_server:
            return self.http
        if self.config.planner_provider == "openrouter":
            api_key = os.environ.get(OPENROUTER_API_KEY_ENVIRONMENT_VARIABLE)
            if not api_key:
                raise ValueError(f"{OPENROUTER_API_KEY_ENVIRONMENT_VARIABLE} is required for the OpenRouter planner")
            return httpx.Client(
                base_url=OPENROUTER_BASE_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=self.config.planner_timeout_seconds,
            )
        return httpx.Client(base_url=self.config.planner_base_url.rstrip("/"), timeout=self.config.planner_timeout_seconds)

    def close(self) -> None:
        self.http.close()
        if not self._planner_uses_decision_server:
            self.planner_http.close()

    def list_models(self) -> list[dict[str, Any]]:
        response = self.http.get("/v1/models")
        response.raise_for_status()
        return response.json().get("data", [])

    def list_planner_models(self) -> list[dict[str, Any]]:
        response = self.planner_http.get("/v1/models")
        response.raise_for_status()
        return response.json().get("data", [])

    def decide(self, observation: dict[str, Any], plan: Plan, observation_mode: str = "assisted") -> DecisionResult:
        compact_prompt = self.config.decision_prompt_profile == "compact"
        instructions_by_mode = {
            "assisted": ASSISTED_INSTRUCTIONS,
            "compact": COMPACT_INSTRUCTIONS,
            "raw": RAW_INSTRUCTIONS,
            "semantic": COMPACT_SEMANTIC_INSTRUCTIONS if compact_prompt else SEMANTIC_INSTRUCTIONS,
        }
        base_instructions = instructions_by_mode[observation_mode]
        plan_text = plan.level_one_prompt_text() if compact_prompt else plan.prompt_text()
        instructions = f"{base_instructions}\nCurrent higher-level plan: {plan_text}"
        schema = (
            semantic_schema(observation, plan.mobility_policy, compact=compact_prompt)
            if observation_mode == "semantic"
            else decision_schema(plan.mobility_policy)
        )
        response = self.http.post(
            "/v1/decision",
            json={
                "model": self.config.decision_model,
                "instructions": instructions,
                "contexts": [json.dumps(observation, separators=(",", ":"))],
                "schema": schema,
                "mode": "auto",
                "cache_prompt": True,
            },
        )
        raise_for_status_with_body(response)
        payload = response.json()
        item = payload["results"][0]
        confidence = {name: float(field["probability"]) for name, field in item.get("fields", {}).items()}
        intent = None
        action = GameAction()
        if observation_mode == "semantic":
            intent = SemanticIntent.from_decision(
                item["decision"],
                semantic_face_choices(observation),
                semantic_mode_choices(observation),
                semantic_dodge_choices(observation),
                semantic_weapon_choices(observation),
            )
        else:
            action = GameAction.from_decision(item["decision"])
        return DecisionResult(
            action=action,
            confidence=confidence,
            timings=payload.get("timings", {}),
            usage=payload.get("usage", {}),
            intent=intent,
        )

    def plan(self, snapshot: dict[str, Any], current: Plan, memory: list[str], trigger: str) -> Plan:
        request = {
            "model": self.config.planner_model,
            "messages": [
                {
                    "role": "system",
                    "content": planner_instructions(self.config.planner_provider),
                },
                {
                    "role": "user",
                    "content": json.dumps(planner_context(self.config.planner_provider, trigger, snapshot, current, memory)),
                },
            ],
            "temperature": 0,
            "max_tokens": self.config.planner_max_tokens,
        }
        if self.config.planner_provider == "local":
            uses_reasoning = self.config.planner_reasoning_effort != "none"
            request["reasoning_effort"] = self.config.planner_reasoning_effort
            request["reasoning_format"] = "auto"
            request["chat_template_kwargs"] = {"enable_thinking": uses_reasoning}
            if not uses_reasoning:
                request["response_format"] = {"type": "json_object"}
        elif self.config.planner_provider == "openrouter":
            request["reasoning_effort"] = self.config.planner_reasoning_effort
            if self.config.planner_json_mode:
                request["response_format"] = {"type": "json_object"}
        planner_path = "chat/completions" if self.config.planner_provider == "openrouter" else "/v1/chat/completions"
        response = self.planner_http.post(
            planner_path,
            json=request,
            timeout=self.config.planner_timeout_seconds,
        )
        raise_for_status_with_body(response)
        choice = response.json()["choices"][0]
        message = choice["message"]
        content = message_text(message.get("content"))
        if not content:
            reasoning = message_text(message.get("reasoning_content")) or message_text(message.get("reasoning"))
            if reasoning:
                LOGGER.warning("Planner response had no content; using its reasoning field")
                content = reasoning
        if not content:
            raise ValueError(
                "planner returned no usable content "
                f"(finish_reason={choice.get('finish_reason')}, content_type={type(message.get('content')).__name__})"
            )
        LOGGER.info("Planner returned a new goal trigger=%s", trigger)
        plan_data = _extract_json_object(content)
        validate_plan_text(plan_data, self.config.planner_provider)
        return Plan.from_dict(plan_data)


def planner_instructions(provider: str) -> str:
    if provider == "openrouter":
        return (
            "Return JSON only for a Doom agent. Level 1 handles controls. Required non-empty strings: goal and "
            "immediate_goal. Also return mobility_policy (free, hold_position, seek_cover, advance, retreat), "
            "navigation_directive (none, turn_hard_left, turn_hard_right, backtrack_left, backtrack_right), "
            "navigation_directive_seconds (0-3), tactical_rule, and expires_after_seconds (5-300). Keep goal, "
            "immediate_goal, and tactical_rule under 16 words each. The goal must name a concrete observation. If "
            "movement_feedback is blocked or wall_following and no hostile is visible, navigation_directive must be a "
            "non-none backtrack or hard turn for 2-3 seconds; immediate_goal must describe exploration after that "
            "one-shot recovery. Prefer advance or free "
            "when safe. Campaign navigation means explore, collect useful items, and progress; never wait for an "
            "encounter. Treat corpses, gore, and decorations as non-hostile. Return no Markdown."
        )
    return (
        "Return one JSON tactical plan for a Doom agent; Level 1 handles controls. Every required text field "
        "must be a non-empty string, never null. Required: goal, immediate_goal, near_term_goal, "
        "long_term_goal, mobility_policy (free, hold_position, seek_cover, advance, retreat), "
        "navigation_directive (none, turn_hard_left, turn_hard_right, backtrack_left, backtrack_right), "
        "navigation_directive_seconds (0-3), priorities (up to 5 strings), avoid (up to 5 strings), "
        "tactical_rule, and expires_after_seconds (5-300). Prefer advance or free when safe; use "
        "hold_position only for a concrete tactical reason. For blocked_progress or wall_following without a "
        "live hostile, choose a non-none navigation directive for 1-3 seconds. Base every goal on the current "
        "observation, not the current_plan: name a concrete fact such as health, visible item counts, threats, "
        "damage, or navigation feedback. Do not reuse the current_plan wording. When health is low and health "
        "items are visible, prioritize recovering health; when health is safe and armor is visible, prioritize armor "
        "before exploration. When mission_profile is campaign_navigation, prioritize exploring new areas, collecting "
        "useful items, and level progress; do not establish a defensive perimeter or wait for the first encounter. "
        "When campaign movement_feedback reports blocked forward progress without a live hostile, first use the "
        "nearby surface to test for a door or switch before choosing a wall-recovery directive. "
        "Corpses, gore, and decorations are never enemies or objectives. Return JSON only, no Markdown."
    )


def planner_context(
    provider: str,
    trigger: str,
    snapshot: dict[str, Any],
    current: Plan,
    memory: list[str],
) -> dict[str, Any]:
    context = {"trigger": trigger, "observation": snapshot}
    if provider == "openrouter":
        return context
    return {**context, "current_plan": current.prompt_text(), "memory": memory}


def validate_plan_text(value: dict[str, Any], provider: str = "local") -> None:
    required_fields = ("goal", "immediate_goal") if provider == "openrouter" else ("goal", "immediate_goal", "near_term_goal", "long_term_goal")
    missing_fields = [field for field in required_fields if not str(value.get(field) or "").strip()]
    if missing_fields:
        raise ValueError(f"planner response omitted required plan text: {', '.join(missing_fields)}")


def message_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    return "".join(str(item.get("text", "")) for item in value if isinstance(item, dict)).strip()


def _extract_json_object(content: str) -> dict[str, Any]:
    object_start = content.find("{")
    if object_start < 0:
        raise ValueError("planner response did not contain a JSON object")
    value, _ = json.JSONDecoder().raw_decode(content[object_start:])
    if not isinstance(value, dict):
        raise ValueError("planner response JSON must be an object")
    return value


def raise_for_status_with_body(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        body = response.text.strip()
        detail = body[:1000] if body else "<empty response>"
        raise RuntimeError(f"{error}; server response: {detail}") from error
