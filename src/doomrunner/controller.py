from __future__ import annotations

import asyncio
import logging
import time
from copy import deepcopy
from dataclasses import replace
from typing import Any

from doomrunner.actions import GameAction
from doomrunner.config import AppConfig
from doomrunner.doom import DoomSession
from doomrunner.feedback import ControlFeedback
from doomrunner.inference import DecisionResult, LlamaClient, Plan
from doomrunner.observation import CompactObservationHistory, RawObservationHistory
from doomrunner.presentation import PresentationDisplay
from doomrunner.semantic import SemanticIntent, SemanticMotor, selected_hostile, semantic_context
from doomrunner.targeting import TargetLock, is_hostile
from doomrunner.telemetry import TelemetryWriter

LOGGER = logging.getLogger(__name__)
PLANNER_FAILURE_RETRY_SECONDS = 5.0
PLANNER_IGNORED_OBJECT_CATEGORIES = frozenset({"corpse", "decoration", "gore"})
IDLE_HOLD_MINIMUM_HEALTH = 50
RECOVERY_DIRECTIVE_SECONDS = 2
PLANNER_TRIGGER_PRIORITIES = {
    "episode_start": 0,
    "periodic": 0,
    "blocked_progress": 1,
    "wall_following": 1,
    "heavy_damage": 2,
    "critical_health": 3,
}


class DoomController:
    def __init__(self, config: AppConfig, mode: str = "jev") -> None:
        self.config = config
        self.mode = mode
        self.doom = DoomSession(
            config.doom,
            config.controller.max_visible_objects,
            config.controller.always_run,
        )
        self.client = LlamaClient(config.server)
        self.telemetry = TelemetryWriter(config.telemetry.directory)
        self.presentation = PresentationDisplay(config.doom.presentation_enabled)
        self.action = GameAction()
        self.plan = Plan()
        self.memory: list[str] = []
        self._pulse = True
        self._decision_task: asyncio.Task[DecisionResult] | None = None
        self._planner_task: asyncio.Task[Plan] | None = None
        self._last_observation: dict[str, Any] | None = None
        self._accepted_decisions = 0
        self._last_decision_latency_ms: float | None = None
        self._next_plan_at = float("inf")
        self._planner_started_at: float | None = None
        self._planner_retry_at = float("-inf")
        self._next_navigation_recovery_plan_at = float("-inf")
        self._has_requested_plan = False
        self._has_accepted_decision = False
        self._planner_calm_after = float("-inf")
        self._pending_planner_trigger: str | None = None
        self.target_lock = TargetLock()
        self.feedback = ControlFeedback()
        self.raw_history = RawObservationHistory(config.controller.raw_history_length)
        self.compact_history = CompactObservationHistory()
        self.semantic_motor = SemanticMotor(
            config.controller.semantic_control_lease_seconds,
            config.controller.survival_retreat_health,
            config.controller.campaign_navigation,
        )
        self._semantic_escape_active = False

    async def run(self, episodes: int) -> None:
        self.doom.start()
        self.telemetry.write("run_started", mode=self.mode, config=self.config)
        try:
            for episode in range(1, episodes + 1):
                if episode > 1:
                    self.doom.new_episode()
                await self._run_episode(episode)
        finally:
            self._cancel_tasks()
            self.presentation.close()
            self.client.close()
            self.doom.close()
            self.telemetry.write("run_stopped")

    async def check(self) -> None:
        errors: list[str] = []
        server_ready = False
        observation: dict[str, Any] | None = None
        try:
            await asyncio.to_thread(self._check_server)
            server_ready = True
        except Exception as error:
            errors.append(f"llama-server: {error}")
            LOGGER.error("llama-server check failed: %s", error)
        try:
            observation = self._check_doom()
        except Exception as error:
            errors.append(f"ViZDoom: {error}")
            LOGGER.error("ViZDoom check failed: %s", error)
        if server_ready and observation is not None:
            try:
                context = self._decision_context(observation)
                result = await asyncio.to_thread(
                    self.client.decide,
                    context,
                    self.plan,
                    self.config.controller.observation_mode,
                )
                decision = result.intent.to_dict() if result.intent is not None else result.action.to_dict()
                LOGGER.info("JEV smoke decision=%s total_ms=%s", decision, result.timings.get("total_ms"))
            except Exception as error:
                errors.append(f"JEV: {error}")
                LOGGER.error("JEV smoke decision failed: %s", error)
        if server_ready and observation is not None and self.config.controller.planner_enabled:
            try:
                plan = await asyncio.to_thread(self.client.plan, observation, self.plan, self.memory, "preflight")
                LOGGER.info("Level 2 preflight passed goal=%s", plan.goal)
            except Exception as error:
                errors.append(f"Level 2 planner: {error}")
                LOGGER.error("Level 2 preflight failed: %s", error)
        self.client.close()
        if errors:
            raise RuntimeError("; ".join(errors))
        LOGGER.info("All configuration checks passed")

    def _check_server(self) -> None:
        models = self.client.list_models()
        ids = [model.get("id") for model in models]
        LOGGER.info("llama-server reachable models=%s", ids)
        if self.config.server.decision_model not in ids:
            raise ValueError(f"decision model '{self.config.server.decision_model}' not reported by llama-server")
        if not self._uses_dedicated_planner_server():
            return
        if self.config.server.planner_provider == "openrouter":
            LOGGER.info("OpenRouter planner configured model=%s", self.config.server.planner_model)
            return
        planner_models = self.client.list_planner_models()
        planner_ids = [model.get("id") for model in planner_models]
        LOGGER.info("planner server reachable models=%s", planner_ids)
        if self.config.server.planner_model not in planner_ids:
            raise ValueError(f"planner model '{self.config.server.planner_model}' not reported by planner server")

    def _check_doom(self) -> dict[str, Any]:
        self.doom.start()
        try:
            initial = self.doom.observe(include_depth=self._uses_depth_observations)
            self.doom.tick(GameAction(), pulse=False)
            observation = self.doom.observe(include_depth=self._uses_depth_observations)
        finally:
            self.doom.close()
        if initial is None or observation is None:
            raise RuntimeError("ViZDoom started but returned no initial observation")
        if observation["tic"] <= initial["tic"]:
            raise RuntimeError(f"ViZDoom state did not advance beyond tic {initial['tic']}")
        LOGGER.info("ViZDoom advanced from tic=%s to tic=%s", initial["tic"], observation["tic"])
        return observation

    async def _run_episode(self, episode: int) -> None:
        fixed_pacing = self.config.controller.decision_pacing == "fixed"
        interval = 1.0 / self.config.controller.decision_hz if fixed_pacing else 0.0
        next_decision = time.monotonic()
        episode_started = time.monotonic()
        next_metrics = episode_started + 5.0
        start_tic: int | None = None
        start_decisions = self._accepted_decisions
        self.action = GameAction()
        self._pulse = True
        self.target_lock.reset()
        self.feedback.reset()
        self.raw_history.reset()
        self.compact_history.reset()
        self.semantic_motor.reset()
        self._semantic_escape_active = False
        self._next_plan_at = episode_started
        self._planner_retry_at = episode_started
        self._next_navigation_recovery_plan_at = episode_started
        self._has_requested_plan = False
        self._has_accepted_decision = False
        self._last_decision_latency_ms = None
        self._planner_calm_after = episode_started
        self._pending_planner_trigger = None
        LOGGER.info("Episode %d started", episode)
        self.telemetry.write("episode_started", episode=episode)
        while not self.doom.is_finished():
            loop_started = time.monotonic()
            if self._uses_semantic_observations:
                self._clear_expired_recovery_directive(loop_started)
                self.action = self.semantic_motor.action(self._last_observation, loop_started)
                self._report_semantic_escape(loop_started)
            self.doom.tick(self.action, self._pulse)
            self._pulse = False
            if self.doom.is_finished():
                break
            observation = self.doom.observe(include_depth=self._uses_depth_observations)
            if observation is not None:
                self._last_observation = observation
                if self._uses_semantic_observations:
                    self.semantic_motor.observe(observation, loop_started, self.action)
                self._update_planner_calm_window(observation, loop_started)
                start_tic = observation["tic"] if start_tic is None else start_tic
            decision_accepted = await self._accept_completed_tasks(episode)
            now = time.monotonic()
            planner_blocks_decision = self._planner_blocks_decision() and self._planner_task is not None
            if (
                observation is not None
                and now >= next_decision
                and self._decision_task is None
                and not decision_accepted
                and not planner_blocks_decision
            ):
                self._start_decision(observation)
                next_decision = now + interval if fixed_pacing else now
            if observation is not None:
                trigger = self._planner_trigger(observation, now)
                if trigger is not None:
                    self._start_planner(observation, trigger, now)
            if observation is not None and start_tic is not None and now >= next_metrics:
                self._report_runtime_metrics(episode_started, start_tic, start_decisions, observation)
                next_metrics = now + 5.0
            self._render_presentation(observation)
            elapsed = time.monotonic() - loop_started
            await self._pace_game_loop(elapsed)
        LOGGER.info("Episode %d finished", episode)
        if self._last_observation is not None and start_tic is not None:
            self._report_runtime_metrics(episode_started, start_tic, start_decisions, self._last_observation)
        self.telemetry.write("episode_finished", episode=episode, observation=self._last_observation)

    def _start_decision(self, observation: dict[str, Any]) -> None:
        if self.mode == "heuristic":
            self.action = heuristic_action(observation)
            self._pulse = True
            self.telemetry.write("heuristic_decision", observation=observation, action=self.action.to_dict())
            return
        context = self._decision_context(observation)
        self._decision_task = asyncio.create_task(
            asyncio.to_thread(
                self.client.decide,
                context,
                self.plan,
                self.config.controller.observation_mode,
            )
        )
        self.telemetry.write("decision_requested", observation=context, plan=self.plan)

    async def _accept_completed_tasks(self, episode: int) -> bool:
        decision_accepted = False
        if self._decision_task is not None and self._decision_task.done():
            task, self._decision_task = self._decision_task, None
            try:
                result = task.result()
                intent = result.intent
                if self._uses_semantic_observations:
                    if intent is None:
                        raise ValueError("semantic decision did not return an intent")
                    if fight_target_requires_requery(intent, self._last_observation):
                        LOGGER.info("Discarded stale Level 1 fight target=%s because another hostile is visible", intent.face)
                        self.telemetry.write("decision_discarded_stale_target", episode=episode, intent=intent)
                        return decision_accepted
                    if noncombat_intent_requires_requery(intent, self._last_observation):
                        LOGGER.info("Discarded Level 1 noncombat intent=%s because hostiles are visible", intent.mode)
                        self.telemetry.write("decision_discarded_noncombat_intent", episode=episode, intent=intent)
                        return decision_accepted
                    if fight_target_is_stale(intent, self._last_observation):
                        LOGGER.info("Retained lost Level 1 fight target=%s as a recent threat track", intent.face)
                        self.telemetry.write("decision_retained_recent_threat", episode=episode, intent=intent)
                    self.semantic_motor.commit(intent, time.monotonic())
                    self.action = self.semantic_motor.action(self._last_observation, time.monotonic())
                else:
                    self.action = result.action
                self._pulse = True
                self._accepted_decisions += 1
                self._has_accepted_decision = True
                self._last_decision_latency_ms = float(result.timings.get("total_ms", 0)) or None
                if self._uses_assisted_observations:
                    self.feedback.record_action(result.action, self._last_observation)
                decision_accepted = True
                self.telemetry.write(
                    "decision_accepted",
                    episode=episode,
                    action=self.action.to_dict(),
                    intent=intent,
                    confidence=result.confidence,
                    timings=result.timings,
                    usage=result.usage,
                    application_snapshot=decision_application_snapshot(self._last_observation),
                )
            except Exception as error:
                LOGGER.warning("Level 1 decision failed: %s", error)
                self.telemetry.write("decision_failed", error=str(error))
        if self._planner_task is None or not self._planner_task.done():
            return decision_accepted
        task, self._planner_task = self._planner_task, None
        planner_latency_ms = elapsed_milliseconds(self._planner_started_at)
        self._planner_started_at = None
        try:
            returned_plan = task.result()
            self.plan = (
                normalize_idle_hold_plan(returned_plan, self._last_observation or {})
                if self.config.controller.idle_hold_recovery
                else returned_plan
            )
            if self.plan != returned_plan:
                LOGGER.warning("Replaced an unjustified Level 2 hold_position plan with exploration recovery")
                self.telemetry.write("plan_normalized_idle_hold", original_plan=returned_plan, plan=self.plan)
            refresh_seconds = min(
                self.config.controller.planner_interval_seconds,
                self.plan.expires_after_seconds,
            )
            self._next_plan_at = time.monotonic() + refresh_seconds
            if self.config.controller.memory_enabled:
                self.memory = [self.plan.prompt_text()]
            self._apply_navigation_directive()
            LOGGER.info(
                "Level 2 plan updated goal=%s mobility=%s directive=%s latency_ms=%.0f refresh_in=%.1fs",
                self.plan.goal,
                self.plan.mobility_policy,
                self.plan.navigation_directive,
                planner_latency_ms,
                refresh_seconds,
            )
            self.telemetry.write("plan_updated", plan=self.plan, latency_ms=planner_latency_ms)
        except Exception as error:
            self._planner_retry_at = time.monotonic() + PLANNER_FAILURE_RETRY_SECONDS
            LOGGER.warning(
                "Level 2 planning failed; retrying in %.1fs: %s",
                PLANNER_FAILURE_RETRY_SECONDS,
                error,
            )
            self.telemetry.write("plan_failed", error=str(error))
        return decision_accepted

    def _decision_context(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._uses_raw_observations:
            return self.raw_history.build(observation, self.action)
        if self._uses_compact_observations:
            return self.compact_history.build(observation, self.action)
        if self._uses_semantic_observations:
            compact = self.compact_history.build(observation, self.action)
            return semantic_context(
                compact,
                self.semantic_motor.navigation_summary(),
                self.semantic_motor.recent_threats(time.monotonic()),
                self.semantic_motor.progress_feedback(),
                getattr(self.config.controller, "campaign_navigation", False),
            )
        context = dict(observation)
        target = self.target_lock.update(observation.get("visible_objects", []))
        context["tactical_target"] = target
        context["control_feedback"] = self.feedback.snapshot(observation, target)
        return context

    @property
    def _uses_raw_observations(self) -> bool:
        return self.config.controller.observation_mode == "raw"

    @property
    def _uses_compact_observations(self) -> bool:
        return self.config.controller.observation_mode == "compact"

    @property
    def _uses_assisted_observations(self) -> bool:
        return self.config.controller.observation_mode == "assisted"

    @property
    def _uses_semantic_observations(self) -> bool:
        return self.config.controller.observation_mode == "semantic"

    @property
    def _uses_depth_observations(self) -> bool:
        return self._uses_raw_observations or self._uses_compact_observations or self._uses_semantic_observations

    def _start_planner(self, observation: dict[str, Any], trigger: str, now: float) -> None:
        snapshot = self._planner_snapshot(observation)
        self._planner_task = asyncio.create_task(
            self._request_plan(snapshot, trigger)
        )
        self._planner_started_at = now
        self._has_requested_plan = True
        if self._pending_planner_trigger == trigger:
            self._pending_planner_trigger = None
        self._next_plan_at = now + self.config.controller.planner_interval_seconds
        if trigger in {"blocked_progress", "wall_following"}:
            self._next_navigation_recovery_plan_at = now + self.config.controller.planner_blocked_progress_cooldown_seconds
        LOGGER.info("Level 2 planning started trigger=%s", trigger)
        self.telemetry.write("plan_requested", trigger=trigger, observation=snapshot, current_plan=self.plan)

    async def _request_plan(self, snapshot: dict[str, Any], trigger: str) -> Plan:
        request = asyncio.to_thread(self.client.plan, snapshot, self.plan, self.memory, trigger)
        deadline = self.config.server.planner_deadline_seconds
        if deadline is None:
            return await request
        try:
            return await asyncio.wait_for(request, timeout=deadline)
        except asyncio.TimeoutError as error:
            raise RuntimeError(f"Level 2 planner exceeded its {deadline:.1f}s deadline") from error

    def _planner_trigger(self, observation: dict[str, Any], now: float) -> str | None:
        if self.mode != "jev" or not self.config.controller.planner_enabled or self._planner_task is not None:
            return None
        if self._planner_blocks_decision() and self._decision_task is not None:
            return None
        if now < self._planner_retry_at:
            return None
        if not self._has_requested_plan:
            if not self._has_accepted_decision:
                return None
            return "episode_start"
        trigger = getattr(self, "_pending_planner_trigger", None)
        trigger = higher_priority_planner_trigger(trigger, self._new_planner_trigger(observation, now))
        if trigger is None:
            return None
        if self._planner_shares_decision_server():
            if not planner_health_is_safe(
                observation,
                getattr(self.config.controller, "planner_min_health_for_background_work", 0),
            ):
                self._defer_planner(trigger, reason="low_health", observation=observation)
                return None
            if now < getattr(self, "_planner_calm_after", float("-inf")):
                self._defer_planner(trigger)
                return None
        return trigger

    def _uses_dedicated_planner_server(self) -> bool:
        server = getattr(self.config, "server", None)
        return bool(getattr(server, "planner_base_url", None)) or getattr(server, "planner_provider", "local") != "local"

    def _planner_shares_decision_server(self) -> bool:
        return not self._uses_dedicated_planner_server()

    def _planner_blocks_decision(self) -> bool:
        return self.config.controller.serialize_inference_requests and self._planner_shares_decision_server()

    def _new_planner_trigger(self, observation: dict[str, Any], now: float) -> str | None:
        navigation_event = self._navigation_recovery_event(now)
        if navigation_event is not None:
            return navigation_event
        event = planner_event(
            observation,
            self.config.controller.planner_damage_threshold,
            self.config.controller.planner_critical_health,
        )
        if event is not None:
            return event
        if now < self._next_plan_at:
            return None
        return "periodic" if self._has_requested_plan else "episode_start"

    def _update_planner_calm_window(self, observation: dict[str, Any], now: float) -> None:
        if not planner_combat_active(observation):
            return
        self._planner_calm_after = now + self.config.controller.planner_calm_seconds

    def _defer_planner(
        self,
        trigger: str,
        reason: str = "combat",
        observation: dict[str, Any] | None = None,
    ) -> None:
        pending = getattr(self, "_pending_planner_trigger", None)
        selected = higher_priority_planner_trigger(pending, trigger)
        if selected == pending:
            return
        self._pending_planner_trigger = selected
        if reason == "low_health":
            threshold = self.config.controller.planner_min_health_for_background_work
            health = planner_health(observation or {})
            LOGGER.info(
                "Level 2 planning deferred trigger=%s until health is at least %s (current=%s)",
                selected,
                threshold,
                health,
            )
            self.telemetry.write(
                "plan_deferred",
                trigger=selected,
                reason=reason,
                health=health,
                min_health=threshold,
            )
            return
        LOGGER.info("Level 2 planning deferred trigger=%s until combat is quiet", selected)
        self.telemetry.write("plan_deferred", trigger=selected, reason=reason, calm_after=self._planner_calm_after)

    def _navigation_recovery_event(self, now: float) -> str | None:
        if not self._uses_semantic_observations or now < self._next_navigation_recovery_plan_at:
            return None
        feedback = self.semantic_motor.progress_feedback()
        if feedback is None:
            return None
        status = feedback.get("status")
        if status == "blocked":
            return "blocked_progress"
        return "wall_following" if status == "wall_following" else None

    def _apply_navigation_directive(self) -> None:
        if not self._uses_semantic_observations:
            return
        now = time.monotonic()
        self.semantic_motor.apply_navigation_directive(
            self.plan.navigation_directive,
            self.plan.navigation_directive_seconds,
            now,
        )
        directive = self.semantic_motor.navigation_directive_summary(now)
        if directive is None:
            return
        LOGGER.info("Level 2 navigation directive applied=%s", directive)
        self.telemetry.write("navigation_directive_applied", **directive)

    def _planner_snapshot(self, observation: dict[str, Any]) -> dict[str, Any]:
        snapshot = planner_observation(observation)
        if getattr(self.config.controller, "campaign_navigation", False):
            snapshot["mission_profile"] = "campaign_navigation"
        if not self._uses_semantic_observations:
            return snapshot
        snapshot["controller_feedback"] = {
            "movement_feedback": self.semantic_motor.progress_feedback(),
            "local_navigation": self.semantic_motor.navigation_summary(),
        }
        return snapshot

    def _render_presentation(self, observation: dict[str, Any] | None) -> None:
        if observation is None:
            return
        self.presentation.render(
            self.doom.screen_buffer(),
            observation,
            self.plan,
            self.semantic_motor.intent if self._uses_semantic_observations else None,
            self.action.to_dict(),
            self._last_decision_latency_ms,
            self._planner_status(),
            self._active_control_override(observation),
        )

    def _active_control_override(self, observation: dict[str, Any]) -> str | None:
        if not self._uses_semantic_observations:
            return None
        now = time.monotonic()
        if self.action.use:
            return "BLOCKED SURFACE: pressing use once to test for a door or switch."
        if self.semantic_motor.blocked_recovery_active(now):
            return "BLOCKED MOVEMENT: reversing and turning briefly before resuming the current objective."
        if not self.semantic_motor.survival_retreat_active:
            return None
        health = observation.get("player", {}).get("health", "--")
        return f"CRITICAL HEALTH ({health} HP): break contact, backpedal, and strafe away from the nearest visible threat."

    def _clear_expired_recovery_directive(self, now: float) -> None:
        if not is_short_recovery_plan(self.plan):
            return
        if self.semantic_motor.navigation_directive_summary(now) is not None:
            return
        self.plan = replace(
            self.plan,
            immediate_goal="Explore forward and test blocked surfaces with use before taking a detour.",
            mobility_policy="free",
            navigation_directive="none",
            navigation_directive_seconds=0,
        )
        LOGGER.info("Level 2 recovery directive completed; restored exploration objective")
        self.telemetry.write("plan_recovery_directive_completed", plan=self.plan)

    def _planner_status(self) -> str:
        if not self.config.controller.planner_enabled:
            return "OFF"
        if not self._has_requested_plan and not self._has_accepted_decision:
            return "WAITING FOR L1"
        if self._planner_task is not None:
            return "THINKING"
        if self._pending_planner_trigger is not None:
            return "DEFERRED"
        return "READY"

    async def _pace_game_loop(self, elapsed: float) -> None:
        if self.config.doom.async_mode:
            await asyncio.sleep(0)
            return
        await asyncio.sleep(max(0.0, 1.0 / DoomSession.TICS_PER_SECOND - elapsed))

    def _report_runtime_metrics(
        self,
        episode_started: float,
        start_tic: int,
        start_decisions: int,
        observation: dict[str, Any],
    ) -> None:
        wall_seconds = max(time.monotonic() - episode_started, 1e-9)
        game_seconds = max(0.0, (observation["tic"] - start_tic) / DoomSession.TICS_PER_SECOND)
        metrics = {
            "wall_seconds": round(wall_seconds, 3),
            "game_seconds": round(game_seconds, 3),
            "real_time_factor": round(game_seconds / wall_seconds, 3),
            "decision_hz": round((self._accepted_decisions - start_decisions) / wall_seconds, 3),
        }
        LOGGER.info(
            "Runtime game=%.1fs wall=%.1fs real_time_factor=%.2f decision_hz=%.2f",
            metrics["game_seconds"],
            metrics["wall_seconds"],
            metrics["real_time_factor"],
            metrics["decision_hz"],
        )
        self.telemetry.write("runtime_metrics", **metrics)

    def _report_semantic_escape(self, now: float) -> None:
        active = self.semantic_motor.is_escaping(now)
        if active == self._semantic_escape_active:
            return
        self._semantic_escape_active = active
        mode = self.semantic_motor.escape_mode
        event = f"semantic_{mode}_{'started' if active else 'ended'}"
        LOGGER.info("Semantic escape %s mode=%s direction=%s", "started" if active else "ended", mode, self.semantic_motor.escape_direction)
        self.telemetry.write(event, mode=mode, direction=self.semantic_motor.escape_direction)

    def _cancel_tasks(self) -> None:
        for task in (self._decision_task, self._planner_task):
            if task is not None:
                task.cancel()


def heuristic_action(observation: dict[str, Any]) -> GameAction:
    objects = observation.get("visible_objects", [])
    target = next((item for item in objects if is_hostile(item)), None)
    if target is None:
        return GameAction(turn="soft_right")
    bearing = float(target["relative_bearing_degrees"])
    if bearing < -15:
        return GameAction(turn="hard_right")
    if bearing > 15:
        return GameAction(turn="hard_left")
    if bearing < -5:
        return GameAction(turn="soft_right")
    if bearing > 5:
        return GameAction(turn="soft_left")
    ready = bool(observation.get("player", {}).get("attack_ready"))
    return GameAction(fire=ready)


def fight_target_is_stale(intent: SemanticIntent, observation: dict[str, Any] | None) -> bool:
    return intent.mode == "fight" and (observation is None or selected_hostile(intent.face, observation) is None)


def fight_target_requires_requery(intent: SemanticIntent, observation: dict[str, Any] | None) -> bool:
    return fight_target_is_stale(intent, observation) and any(
        is_hostile(item) for item in (observation or {}).get("visible_objects", [])
    )


def noncombat_intent_requires_requery(intent: SemanticIntent, observation: dict[str, Any] | None) -> bool:
    if intent.mode not in {"explore", "hold"}:
        return False
    return any(is_hostile(item) for item in (observation or {}).get("visible_objects", []))


def decision_application_snapshot(observation: dict[str, Any] | None) -> dict[str, Any]:
    if observation is None:
        return {"available": False}
    player = observation.get("player", {})
    return {
        "available": True,
        "tic": observation.get("tic"),
        "player": {
            name: player[name]
            for name in ("health", "armor", "heading_degrees", "weapon", "weapon_ammo")
            if name in player
        },
        "visible_hostiles": [
            {
                name: item[name]
                for name in ("id", "name", "distance", "relative_bearing_degrees")
                if name in item
            }
            for item in observation.get("visible_objects", [])
            if is_hostile(item)
        ],
    }


def planner_event(
    observation: dict[str, Any],
    damage_threshold: int,
    critical_health: int,
) -> str | None:
    player = observation.get("player", {})
    combat = observation.get("combat", {})
    health = int(player.get("health", 100))
    health_change = combat.get("health_change")
    if health_change is None or health_change >= 0:
        return None
    if health <= critical_health:
        return "critical_health"
    if -health_change >= damage_threshold:
        return "heavy_damage"
    return None


def planner_observation(observation: dict[str, Any]) -> dict[str, Any]:
    player = observation.get("player", {})
    combat = observation.get("combat", {})
    return {
        "player": {
            name: player[name]
            for name in ("health", "armor", "weapon", "weapon_ammo", "attack_ready")
            if name in player
        },
        "combat": {
            name: combat[name]
            for name in ("kills", "hits", "hits_taken", "damage_dealt", "damage_taken", "health_change")
            if name in combat
        },
        "visible_object_counts": planner_object_counts(observation.get("visible_object_counts", [])),
    }


def planner_object_counts(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in objects
        if str(item.get("category", "")).lower() not in PLANNER_IGNORED_OBJECT_CATEGORIES
    ]


def normalize_idle_hold_plan(plan: Plan, observation: dict[str, Any]) -> Plan:
    if plan.mobility_policy != "hold_position" or not planner_is_unjustifiably_idle(observation):
        return plan
    directive = plan.navigation_directive if plan.navigation_directive != "none" else recovery_directive(observation)
    directive_seconds = plan.navigation_directive_seconds if plan.navigation_directive != "none" else RECOVERY_DIRECTIVE_SECONDS
    return replace(
        plan,
        goal="Explore open space and collect useful resources before the next encounter.",
        immediate_goal="Leave the current position and explore toward open space; do not wait for enemies.",
        near_term_goal="Follow the local open-space direction and break away from any wall contact.",
        mobility_policy="advance",
        navigation_directive=directive,
        navigation_directive_seconds=directive_seconds if directive != "none" else 0,
        priorities=("explore open space", "collect useful resources", "avoid wall contact"),
        tactical_rule="No live threat is visible and health is safe, so continue moving instead of holding position.",
    )


def is_short_recovery_plan(plan: Plan) -> bool:
    text = plan.immediate_goal.lower()
    return plan.navigation_directive != "none" and any(word in text for word in ("recover", "blocked", "wall"))


def planner_is_unjustifiably_idle(observation: dict[str, Any]) -> bool:
    if planner_health(observation) < IDLE_HOLD_MINIMUM_HEALTH:
        return False
    if float(observation.get("combat", {}).get("health_change") or 0) < 0:
        return False
    return not any(is_hostile(item) for item in observation.get("visible_objects", []))


def recovery_directive(observation: dict[str, Any]) -> str:
    feedback = observation.get("controller_feedback", {}).get("movement_feedback") or {}
    if feedback.get("status") not in {"blocked", "wall_following"}:
        return "none"
    direction = feedback.get("turn_direction")
    return f"backtrack_{direction}" if direction in {"left", "right"} else "backtrack_left"


def planner_combat_active(observation: dict[str, Any]) -> bool:
    if any(is_hostile(item) for item in observation.get("visible_objects", [])):
        return True
    return float(observation.get("combat", {}).get("health_change") or 0) < 0


def planner_health_is_safe(observation: dict[str, Any], minimum_health: int) -> bool:
    return planner_health(observation) >= minimum_health


def planner_health(observation: dict[str, Any]) -> int:
    return int(observation.get("player", {}).get("health", 100))


def elapsed_milliseconds(started_at: float | None) -> float:
    return 0.0 if started_at is None else (time.monotonic() - started_at) * 1000


def higher_priority_planner_trigger(current: str | None, candidate: str | None) -> str | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    if PLANNER_TRIGGER_PRIORITIES[candidate] > PLANNER_TRIGGER_PRIORITIES[current]:
        return candidate
    return current
