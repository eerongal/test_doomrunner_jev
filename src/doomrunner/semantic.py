from __future__ import annotations

import logging
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Any

from doomrunner.navigation import FrontierGuide, SeenSpaceNavigator
from doomrunner.progress import ProgressFeedback
from doomrunner.targeting import is_hostile
from doomrunner.threats import RecentThreatMemory
from doomrunner.weapons import recommended_weapon_action

from doomrunner.actions import GameAction, WEAPON_CHOICES


FACE_SKILLS = ("clear_space", "explore_frontier", "backtrack_and_turn", "hold", "scan_left", "scan_right")
IMMEDIATE_MODES = ("fight", "evade", "explore", "hold")
MOVE_SKILLS = ("advance", "retreat", "engage", "hold")
DODGE_SKILLS = ("carry_on", "left", "right")
TRIGGER_SKILLS = ("hold_fire", "fire")
USE_SKILLS = ("no_use", "use")
TARGET_PREFIX = "hostile:"
RECENT_TARGET_PREFIX = "recent_hostile:"
MAX_COMBAT_RESOURCES = 2
SOFT_TURN_THRESHOLD_DEGREES = 4.0
HARD_TURN_THRESHOLD_DEGREES = 18.0
FIRE_ALIGNMENT_DEGREES = 5.0
REQUESTED_FIRE_ALIGNMENT_DEGREES = 12.0
ENGAGEMENT_ADVANCE_ALIGNMENT_DEGREES = 12.0
NEAR_DEPTH_SIGNAL = 12.0
SIDE_DEPTH_ADVANTAGE = 3.0
CORNER_DEPTH_SIGNAL = 3.0
CORNER_ESCAPE_SECONDS = 0.75
DEAD_END_ESCAPE_SECONDS = 1.8
DEAD_END_REPETITION_WINDOW_SECONDS = 4.0
DEAD_END_ESCAPE_COUNT = 2
BACKTRACK_TURN_SECONDS = 1.2
BLOCKED_RECOVERY_SECONDS = 1.2
BLOCKED_SURFACE_PROBE_COOLDOWN_SECONDS = 2.0
BLOCKED_SURFACE_WAIT_SECONDS = 1.2
CLOSE_ENGAGEMENT_DISTANCE = 160.0
FAR_ENGAGEMENT_DISTANCE = 256.0
DEFAULT_SURVIVAL_RETREAT_HEALTH = 35
NAVIGATION_DIRECTIVE_ACTIONS = {
    "turn_hard_left": GameAction(movement="forward", turn="hard_left"),
    "turn_hard_right": GameAction(movement="forward", turn="hard_right"),
    "backtrack_left": GameAction(movement="backward", strafe="left", turn="hard_left"),
    "backtrack_right": GameAction(movement="backward", strafe="right", turn="hard_right"),
}

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticIntent:
    mode: str = "explore"
    face: str = "hold"
    move: str = "hold"
    dodge: str = "carry_on"
    trigger: str = "hold_fire"
    use: str = "no_use"
    weapon: str = "keep"

    @classmethod
    def from_decision(
        cls,
        value: dict[str, Any],
        face_choices: tuple[str, ...],
        mode_choices: tuple[str, ...],
        dodge_choices: tuple[str, ...],
        weapon_choices: tuple[str, ...] = WEAPON_CHOICES,
    ) -> "SemanticIntent":
        intent = cls(
            mode=str(value.get("immediate_mode", "hold")),
            face=str(value.get("face", "hold")),
            move=str(value.get("move", "hold")),
            dodge=str(value.get("dodge", "carry_on")),
            trigger=str(value.get("trigger", "hold_fire")),
            use=str(value.get("use", "no_use")),
            weapon=str(value.get("weapon", "keep")),
        )
        intent.validate(face_choices, mode_choices, dodge_choices, weapon_choices)
        return intent

    def validate(
        self,
        face_choices: tuple[str, ...],
        mode_choices: tuple[str, ...],
        dodge_choices: tuple[str, ...],
        weapon_choices: tuple[str, ...] = WEAPON_CHOICES,
    ) -> None:
        require_choice("immediate_mode", self.mode, mode_choices)
        require_choice("face", self.face, face_choices)
        require_choice("move", self.move, MOVE_SKILLS)
        require_choice("dodge", self.dodge, dodge_choices)
        require_choice("trigger", self.trigger, TRIGGER_SKILLS)
        require_choice("use", self.use, USE_SKILLS)
        require_choice("weapon", self.weapon, weapon_choices)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class SemanticMotor:
    def __init__(
        self,
        lease_seconds: float,
        survival_retreat_health: int = DEFAULT_SURVIVAL_RETREAT_HEALTH,
        campaign_navigation: bool = False,
    ) -> None:
        self.lease_seconds = lease_seconds
        self.survival_retreat_health = survival_retreat_health
        self.campaign_navigation = campaign_navigation
        self.intent: SemanticIntent | None = None
        self.committed_at = float("-inf")
        self._escape_until = float("-inf")
        self._escape_direction = "left"
        self._escape_mode = "none"
        self._escape_starts: deque[float] = deque()
        self._navigation_directive = "none"
        self._navigation_directive_until = float("-inf")
        self._blocked_recovery_until = float("-inf")
        self._blocked_recovery_direction = "left"
        self._blocked_surface_probe_available_at = float("-inf")
        self._blocked_surface_wait_until = float("-inf")
        self._exploration_hostile_seen = False
        self._survival_retreat_active = False
        self.navigator = SeenSpaceNavigator()
        self.threat_memory = RecentThreatMemory()
        self.progress = ProgressFeedback()

    def reset(self) -> None:
        self.intent = None
        self.committed_at = float("-inf")
        self._escape_until = float("-inf")
        self._escape_mode = "none"
        self._escape_starts.clear()
        self._navigation_directive = "none"
        self._navigation_directive_until = float("-inf")
        self._blocked_recovery_until = float("-inf")
        self._blocked_recovery_direction = "left"
        self._blocked_surface_probe_available_at = float("-inf")
        self._blocked_surface_wait_until = float("-inf")
        self._exploration_hostile_seen = False
        self._survival_retreat_active = False
        self.navigator.reset()
        self.threat_memory.reset()
        self.progress.reset()

    def observe(self, observation: dict[str, Any], now: float, applied_action: GameAction) -> None:
        self.navigator.observe(observation)
        self.threat_memory.observe(observation, now)
        self.progress.observe(observation, applied_action, now)
        self._exploration_hostile_seen = self._exploration_intent_has_new_hostile(observation)

    def navigation_summary(self) -> dict[str, Any]:
        return self.navigator.summary()

    def recent_threats(self, now: float) -> list[dict[str, Any]]:
        return self.threat_memory.summary(now)

    def progress_feedback(self) -> dict[str, Any] | None:
        return self.progress.summary()

    def apply_navigation_directive(self, directive: str, duration_seconds: int, now: float) -> None:
        self._navigation_directive = directive if directive in NAVIGATION_DIRECTIVE_ACTIONS else "none"
        self._navigation_directive_until = now + duration_seconds if self._navigation_directive != "none" else float("-inf")

    def navigation_directive_summary(self, now: float) -> dict[str, Any] | None:
        if self._navigation_directive_until <= now:
            return None
        return {
            "directive": self._navigation_directive,
            "remaining_seconds": round(self._navigation_directive_until - now, 2),
        }

    def commit(self, intent: SemanticIntent, now: float) -> None:
        self.intent = intent
        self.committed_at = now
        self._exploration_hostile_seen = False
        if not self._uses_clear_space_escape():
            self._escape_until = float("-inf")
            self._escape_mode = "none"

    def action(self, observation: dict[str, Any] | None, now: float) -> GameAction:
        if observation is None:
            return GameAction()
        return self._action(observation, now)

    def _action(self, observation: dict[str, Any], now: float) -> GameAction:
        survival_action = self._survival_retreat_action(observation)
        if survival_action is not None:
            return self._apply_weapon_policy(survival_action, observation, closest_visible_hostile(observation))
        directive = self._navigation_directive_action(observation, now)
        if directive is not None:
            return directive
        blocked_surface_probe = self._blocked_surface_probe_action(observation, now)
        if blocked_surface_probe is not None:
            return blocked_surface_probe
        blocked_recovery = self._blocked_recovery_action(observation, now)
        if blocked_recovery is not None:
            return blocked_recovery
        if self.intent is None or now - self.committed_at > self.lease_seconds:
            return GameAction()
        combat_interrupt = self._exploration_combat_interrupt(observation)
        if combat_interrupt is not None:
            return combat_interrupt
        recovery = self._backtrack_and_turn_action(now)
        if recovery is not None:
            return recovery
        escape = self._corner_escape_action(observation, now)
        if escape is not None:
            return escape
        target = self._selected_target(observation, now)
        mode_action = self._mode_action(target, observation)
        if mode_action is not None:
            return self._apply_weapon_policy(mode_action, observation, target)
        return self._apply_weapon_policy(GameAction(
            movement=self._movement(target, observation),
            strafe={"left": "left", "right": "right", "carry_on": "stop"}[self.intent.dodge],
            turn=self._turn(target, observation),
            fire=self.intent.trigger == "fire" and target_is_visible_and_aligned(target),
            use=self.intent.use == "use",
            weapon=self.intent.weapon,
        ), observation, target)

    def _survival_retreat_action(self, observation: dict[str, Any]) -> GameAction | None:
        target = closest_visible_hostile(observation)
        active = target is not None and player_health(observation) <= self.survival_retreat_health
        self._report_survival_retreat(active, player_health(observation))
        if not active:
            return None
        return GameAction(
            movement="backward",
            strafe=evade_strafe_direction(target),
            turn=turn_toward_bearing(float(target["relative_bearing_degrees"])),
            fire=target_is_visible_and_aligned(target),
            weapon="keep" if self.intent is None else self.intent.weapon,
        )

    def _report_survival_retreat(self, active: bool, health: int) -> None:
        if active == self._survival_retreat_active:
            return
        self._survival_retreat_active = active
        status = "started" if active else "ended"
        LOGGER.info("Emergency survival retreat %s health=%s", status, health)

    def _apply_weapon_policy(self, action: GameAction, observation: dict[str, Any], target: dict[str, Any] | None) -> GameAction:
        weapon = recommended_weapon_action(observation, target)
        return action if weapon == "keep" else replace(action, weapon=weapon)

    def _navigation_directive_action(self, observation: dict[str, Any], now: float) -> GameAction | None:
        if self._navigation_directive_until <= now:
            return None
        if any(is_hostile(item) for item in observation.get("visible_objects", [])):
            return None
        return NAVIGATION_DIRECTIVE_ACTIONS[self._navigation_directive]

    def _exploration_combat_interrupt(self, observation: dict[str, Any]) -> GameAction | None:
        if not self._exploration_hostile_seen:
            return None
        target = closest_visible_hostile(observation)
        if target is None:
            return None
        return GameAction(
            movement="backward",
            strafe=evade_strafe_direction(target),
            turn=turn_toward_bearing(float(target["relative_bearing_degrees"])),
            fire=target_is_visible_and_aligned(target),
            weapon=self.intent.weapon,
        )

    def _exploration_intent_has_new_hostile(self, observation: dict[str, Any]) -> bool:
        if self.intent is None or self.intent.mode != "explore":
            return False
        if selected_hostile_id(self.intent.face) is not None or selected_recent_threat_id(self.intent.face) is not None:
            return False
        return closest_visible_hostile(observation) is not None

    def is_escaping(self, now: float) -> bool:
        return self._escape_until > now

    @property
    def escape_direction(self) -> str:
        return self._escape_direction

    @property
    def escape_mode(self) -> str:
        return self._escape_mode

    @property
    def survival_retreat_active(self) -> bool:
        return self._survival_retreat_active

    def blocked_recovery_active(self, now: float) -> bool:
        return self._blocked_recovery_until > now

    def _corner_escape_action(self, observation: dict[str, Any], now: float) -> GameAction | None:
        if not self._uses_clear_space_escape():
            self._escape_until = float("-inf")
            self._escape_mode = "none"
            return None
        if self._escape_until > now:
            return self._escape_action()
        if not depth_is_enclosed(observation.get("depth_grid")):
            self._escape_mode = "none"
            return None
        self._record_escape_start(now)
        self._escape_direction = opposite_direction(self._escape_direction)
        duration = DEAD_END_ESCAPE_SECONDS if self._is_repeated_dead_end() else CORNER_ESCAPE_SECONDS
        self._escape_mode = "dead_end_reversal" if duration == DEAD_END_ESCAPE_SECONDS else "corner_escape"
        self._escape_until = now + duration
        return self._escape_action()

    def _backtrack_and_turn_action(self, now: float) -> GameAction | None:
        if self.intent is None or self.intent.face != "backtrack_and_turn":
            return None
        if now - self.committed_at > BACKTRACK_TURN_SECONDS:
            return GameAction()
        direction = recovery_direction(self.navigator.guide(), self._escape_direction)
        return GameAction(
            movement="backward",
            turn=f"hard_{direction}",
            use=self.intent.use == "use",
            weapon=self.intent.weapon,
        )

    def _blocked_recovery_action(self, observation: dict[str, Any], now: float) -> GameAction | None:
        if any(is_hostile(item) for item in observation.get("visible_objects", [])):
            return None
        if self._blocked_recovery_until <= now:
            feedback = self.progress_feedback() or {}
            if feedback.get("status") != "blocked":
                return None
            self._blocked_recovery_direction = recovery_direction(self.navigator.guide(), self._escape_direction)
            self._blocked_recovery_until = now + BLOCKED_RECOVERY_SECONDS
            LOGGER.info("Blocked movement recovery started direction=%s", self._blocked_recovery_direction)
        return GameAction(
            movement="backward",
            strafe=self._blocked_recovery_direction,
            turn=f"hard_{self._blocked_recovery_direction}",
        )

    def _blocked_surface_probe_action(self, observation: dict[str, Any], now: float) -> GameAction | None:
        if not self.campaign_navigation:
            return None
        if any(is_hostile(item) for item in observation.get("visible_objects", [])):
            return None
        if self._blocked_surface_wait_until > now:
            return GameAction(movement="forward")
        if now < self._blocked_surface_probe_available_at:
            return None
        feedback = self.progress_feedback() or {}
        if feedback.get("status") != "blocked":
            return None
        self._blocked_surface_probe_available_at = now + BLOCKED_SURFACE_PROBE_COOLDOWN_SECONDS
        self._blocked_surface_wait_until = now + BLOCKED_SURFACE_WAIT_SECONDS
        LOGGER.info("Blocked campaign surface probe started; waiting for a possible door")
        return GameAction(movement="forward", use=True)

    def _uses_clear_space_escape(self) -> bool:
        return self.intent is not None and self.intent.mode == "explore" and self.intent.face == "clear_space" and self.intent.move == "advance"

    def _escape_action(self) -> GameAction:
        if self._escape_mode == "dead_end_reversal":
            return GameAction(
                movement="backward",
                turn=f"hard_{self._escape_direction}",
                use=self.intent is not None and self.intent.use == "use",
                weapon="keep" if self.intent is None else self.intent.weapon,
            )
        return GameAction(
            movement="backward",
            strafe=self._escape_direction,
            turn=f"hard_{self._escape_direction}",
            use=self.intent is not None and self.intent.use == "use",
            weapon="keep" if self.intent is None else self.intent.weapon,
        )

    def _record_escape_start(self, now: float) -> None:
        self._escape_starts.append(now)
        earliest = now - DEAD_END_REPETITION_WINDOW_SECONDS
        while self._escape_starts and self._escape_starts[0] < earliest:
            self._escape_starts.popleft()

    def _is_repeated_dead_end(self) -> bool:
        return len(self._escape_starts) >= DEAD_END_ESCAPE_COUNT

    def _turn(self, target: dict[str, Any] | None, observation: dict[str, Any]) -> str:
        if target is not None:
            return turn_toward_bearing(float(target["relative_bearing_degrees"]))
        if self.intent is None:
            return "hold"
        if self.intent.face == "scan_left":
            return "soft_left"
        if self.intent.face == "scan_right":
            return "soft_right"
        if self.intent.face in {"clear_space", "explore_frontier"}:
            return turn_toward_frontier(self.navigator.guide(), observation.get("depth_grid"))
        return "hold"

    def _mode_action(self, target: dict[str, Any] | None, observation: dict[str, Any]) -> GameAction | None:
        if self.intent is None or self.intent.mode == "explore":
            return None
        if self.intent.mode == "hold":
            return GameAction(turn=self._turn(target, observation), weapon=self.intent.weapon)
        if self.intent.mode == "evade":
            return GameAction(
                movement="backward",
                strafe={"left": "left", "right": "right", "carry_on": "stop"}[self.intent.dodge],
                turn=self._turn(target, observation),
                fire=self._requested_fire_is_safe(observation),
                weapon=self.intent.weapon,
            )
        return GameAction(
            movement=self._engagement_movement(target, observation),
            strafe=self._fight_strafe(target, observation),
            turn=self._turn(target, observation),
            fire=target_is_visible_and_aligned(target) or self._requested_fire_is_safe(observation),
            weapon=self.intent.weapon,
        )

    def _requested_fire_is_safe(self, observation: dict[str, Any]) -> bool:
        if self.intent is None or self.intent.trigger != "fire":
            return False
        return any(
            hostile_is_aligned(hostile, REQUESTED_FIRE_ALIGNMENT_DEGREES)
            for hostile in observation.get("visible_objects", [])
            if is_hostile(hostile)
        )

    def _fight_strafe(self, target: dict[str, Any] | None, observation: dict[str, Any]) -> str:
        if target is None or not target.get("visible", True):
            return "stop"
        if not needs_off_axis_evasion(target, observation):
            if abs(float(target["relative_bearing_degrees"])) > ENGAGEMENT_ADVANCE_ALIGNMENT_DEGREES:
                return "stop"
        return {"left": "left", "right": "right", "carry_on": "stop"}[self.intent.dodge]

    def _movement(self, target: dict[str, Any] | None, observation: dict[str, Any]) -> str:
        if self.intent is None:
            return "stop"
        if self.intent.move != "engage":
            return {"advance": "forward", "retreat": "backward", "hold": "stop"}[self.intent.move]
        return self._engagement_movement(target, observation)

    def _engagement_movement(self, target: dict[str, Any] | None, observation: dict[str, Any]) -> str:
        if target is None or not target.get("visible", True):
            return "stop"
        if abs(float(target["relative_bearing_degrees"])) > ENGAGEMENT_ADVANCE_ALIGNMENT_DEGREES:
            return "backward" if needs_off_axis_evasion(target, observation) else "stop"
        distance = float(target.get("distance", FAR_ENGAGEMENT_DISTANCE))
        if distance < CLOSE_ENGAGEMENT_DISTANCE:
            return "backward"
        if distance > FAR_ENGAGEMENT_DISTANCE:
            return "forward"
        return "stop"

    def _selected_target(self, observation: dict[str, Any], now: float) -> dict[str, Any] | None:
        face = self.intent.face if self.intent is not None else ""
        target = selected_hostile(face, observation)
        if target is not None:
            return {**target, "visible": True}
        object_id = selected_recent_threat_id(face)
        if object_id is None and self.intent is not None and self.intent.mode == "fight":
            object_id = selected_hostile_id(face)
        return None if object_id is None else self.threat_memory.target(object_id, observation, now)


def semantic_context(
    compact_context: dict[str, Any],
    navigation: dict[str, Any] | None = None,
    recent_threats: list[dict[str, Any]] | None = None,
    progress_feedback: dict[str, Any] | None = None,
    campaign_navigation: bool = False,
) -> dict[str, Any]:
    context = deepcopy(compact_context)
    context["observation_mode"] = "semantic_skills"
    if campaign_navigation:
        context["mission_profile"] = "campaign_navigation"
    context["combat_pressure"] = combat_pressure(context)
    for hostile in context.get("visible_hostiles", []):
        hostile["face_choice"] = target_choice(hostile["id"])
        hostile["engagement_range"] = engagement_range(hostile.get("distance"))
    if navigation is not None:
        context["local_navigation"] = navigation
    visible_ids = {str(hostile["id"]) for hostile in context.get("visible_hostiles", [])}
    retained_threats = [threat for threat in recent_threats or [] if str(threat["id"]) not in visible_ids]
    if retained_threats:
        context["recent_threats"] = retained_threats
    if progress_feedback is not None:
        context["movement_feedback"] = progress_feedback
    if context.get("visible_hostiles"):
        context["visible_resources_and_objects"] = context.get("visible_resources_and_objects", [])[:MAX_COMBAT_RESOURCES]
        context["combat_priority"] = "Visible hostiles are immediate threats: choose a hostile face choice before exploration."
    return context


def semantic_schema(
    context: dict[str, Any],
    mobility_policy: str,
    compact: bool = False,
) -> dict[str, dict[str, Any]]:
    face_choices = semantic_face_choices(context)
    mode_choices = semantic_mode_choices(context)
    dodge_choices = semantic_dodge_choices(context)
    move_choices = semantic_move_choices(context)
    use_choices = semantic_use_choices(context)
    if compact:
        return {
            "immediate_mode": {
                "type": "enum",
                "choices": list(mode_choices),
                "description": "Immediate priority.",
            },
            "face": {
                "type": "enum",
                "choices": list(face_choices),
                "description": "Target or viewing skill.",
            },
            "move": {
                "type": "enum",
                "choices": list(move_choices),
                "description": "Forward or backward intent.",
            },
            "dodge": {
                "type": "enum",
                "choices": list(dodge_choices),
                "description": "Lateral movement intent.",
            },
            "trigger": {
                "type": "enum",
                "choices": list(TRIGGER_SKILLS),
                "description": "Fire permission.",
            },
            "use": {
                "type": "enum",
                "choices": list(use_choices),
                "description": "Interact impulse.",
            },
            "weapon": {
                "type": "enum",
                "choices": list(semantic_weapon_choices(context)),
                "description": "Weapon-change impulse.",
            },
        }
    movement_guidance = {
        "advance": "Prefer advance when there is no immediate reason to hold or retreat.",
        "retreat": "Prefer retreat while the higher-level plan calls for withdrawal.",
        "hold_position": "Prefer hold unless movement is needed to survive.",
        "seek_cover": "Choose retreat or advance according to the observed geometry rather than remaining exposed.",
        "free": "Advance when safe; hold only for a concrete tactical reason.",
    }.get(mobility_policy, "Advance when safe; hold only for a concrete tactical reason.")
    return {
        "immediate_mode": {
            "type": "enum",
            "choices": list(mode_choices),
            "description": (
                "Choose the immediate priority. fight tracks the chosen live hostile, maintains range, and fires when "
                "aligned. evade retreats while facing the chosen threat. explore is available only when no live hostile "
                "is visible. hold intentionally stops movement."
            ),
        },
        "face": {
            "type": "enum",
            "choices": list(face_choices),
            "description": (
                "Choose what the motor should continuously face. A hostile choice tracks that exact currently visible "
                "hostile. A recent_hostile choice turns toward a directly observed hostile's last known direction but "
                "never authorizes firing. clear_space follows locally open depth; explore_frontier prefers a less-visited "
                "open direction; backtrack_and_turn is a bounded recovery for reported blocked forward progress; scans "
                "deliberately inspect one side."
            ),
        },
        "move": {
            "type": "enum",
            "choices": list(move_choices),
            "description": (
                "Choose longitudinal movement independently of facing and dodge. engage maintains range from the exact "
                "selected hostile: it approaches a far target, holds a medium range, and backs away at close range. "
                f"{movement_guidance}"
            ),
        },
        "dodge": {
            "type": "enum",
            "choices": list(dodge_choices),
            "description": (
                "Choose simultaneous lateral movement. Active combat pressure requires a left or right dodge; otherwise "
                "carry_on is allowed when there is no immediate attack to avoid."
            ),
        },
        "trigger": {
            "type": "enum",
            "choices": list(TRIGGER_SKILLS),
            "description": (
                "Grant fire permission only while facing a currently visible hostile. The motor suppresses shots until "
                "that chosen hostile is aligned."
            ),
        },
        "use": {
            "type": "enum",
            "choices": list(use_choices),
            "description": "Pulse use only when a nearby door or switch should be activated.",
        },
        "weapon": {
            "type": "enum",
            "choices": list(semantic_weapon_choices(context)),
            "description": "Keep the current weapon unless ammunition, range, or safety gives a concrete reason to change.",
        },
    }


def semantic_face_choices(context: dict[str, Any]) -> tuple[str, ...]:
    targets = tuple(target_choice(item["id"]) for item in context.get("visible_hostiles", []))
    recent_targets = tuple(recent_target_choice(item["id"]) for item in context.get("recent_threats", []))
    if targets:
        return targets
    return tuple(dict.fromkeys(FACE_SKILLS + targets + recent_targets))


def semantic_mode_choices(context: dict[str, Any]) -> tuple[str, ...]:
    if not context.get("visible_hostiles"):
        if context.get("mission_profile") == "campaign_navigation":
            return "explore",
        return "explore", "hold"
    if evade_is_available(context):
        return "fight", "evade", "hold"
    return "fight", "hold"


def semantic_move_choices(context: dict[str, Any]) -> tuple[str, ...]:
    if context.get("mission_profile") == "campaign_navigation" and not context.get("visible_hostiles"):
        return "advance",
    return MOVE_SKILLS


def semantic_use_choices(context: dict[str, Any]) -> tuple[str, ...]:
    if context.get("mission_profile") != "campaign_navigation":
        return USE_SKILLS
    feedback = context.get("movement_feedback") or {}
    if feedback.get("status") == "blocked" and not context.get("visible_hostiles"):
        return "use",
    return "no_use",


def evade_is_available(context: dict[str, Any]) -> bool:
    if context.get("combat_pressure", {}).get("active"):
        return True
    return any(
        float(hostile.get("distance", float("inf"))) <= CLOSE_ENGAGEMENT_DISTANCE
        for hostile in context.get("visible_hostiles", [])
    )


def semantic_dodge_choices(context: dict[str, Any]) -> tuple[str, ...]:
    return ("left", "right") if context.get("combat_pressure", {}).get("active") else DODGE_SKILLS


def semantic_weapon_choices(context: dict[str, Any]) -> tuple[str, ...]:
    ammunition = context.get("player", {}).get("weapon_ammo")
    try:
        return WEAPON_CHOICES if int(ammunition) <= 0 else ("keep",)
    except (TypeError, ValueError):
        return "keep",


def combat_pressure(context: dict[str, Any]) -> dict[str, Any]:
    combat = context.get("combat", {})
    movement = context.get("motion_since_previous_decision") or {}
    deltas = movement.get("combat_delta", {})
    took_damage = float(combat.get("health_change") or 0) < 0 or float(deltas.get("hits_taken") or 0) > 0
    projectile = next(
        (
            item
            for item in context.get("visible_resources_and_objects", [])
            if str(item.get("category", "")).lower() == "explosive"
        ),
        None,
    )
    if projectile is not None:
        return {"active": True, "reason": "visible_explosive", "distance": projectile.get("distance")}
    close_hostile = next(
        (
            hostile
            for hostile in context.get("visible_hostiles", [])
            if engagement_range(hostile.get("distance")) == "close"
        ),
        None,
    )
    if close_hostile is not None:
        return {"active": True, "reason": "close_hostile", "distance": close_hostile.get("distance")}
    return {"active": took_damage, "reason": "recent_damage" if took_damage else "none"}


def target_choice(object_id: Any) -> str:
    return f"{TARGET_PREFIX}{object_id}"


def recent_target_choice(object_id: Any) -> str:
    return f"{RECENT_TARGET_PREFIX}{object_id}"


def engagement_range(distance: Any) -> str:
    if distance is None:
        return "unknown"
    value = float(distance)
    if value < CLOSE_ENGAGEMENT_DISTANCE:
        return "close"
    if value > FAR_ENGAGEMENT_DISTANCE:
        return "far"
    return "medium"


def selected_hostile(face: str, observation: dict[str, Any]) -> dict[str, Any] | None:
    object_id = selected_hostile_id(face)
    if object_id is None:
        return None
    return next(
        (
            item
            for item in observation.get("visible_objects", [])
            if str(item.get("id")) == object_id and str(item.get("category", "")).lower() in {"monster", "player"}
        ),
        None,
    )


def closest_visible_hostile(observation: dict[str, Any]) -> dict[str, Any] | None:
    hostiles = [item for item in observation.get("visible_objects", []) if is_hostile(item)]
    if not hostiles:
        return None
    return min(hostiles, key=lambda item: abs(float(item["relative_bearing_degrees"])))


def player_health(observation: dict[str, Any]) -> int:
    try:
        return int(observation.get("player", {}).get("health", 100))
    except (TypeError, ValueError):
        return 100


def selected_hostile_id(face: str) -> str | None:
    return face[len(TARGET_PREFIX) :] if face.startswith(TARGET_PREFIX) else None


def selected_recent_threat_id(face: str) -> str | None:
    return face[len(RECENT_TARGET_PREFIX) :] if face.startswith(RECENT_TARGET_PREFIX) else None


def target_is_visible_and_aligned(target: dict[str, Any] | None) -> bool:
    return target is not None and bool(target.get("visible", True)) and hostile_is_aligned(target, FIRE_ALIGNMENT_DEGREES)


def hostile_is_aligned(hostile: dict[str, Any], alignment_degrees: float) -> bool:
    return abs(float(hostile.get("relative_bearing_degrees", float("inf")))) <= alignment_degrees


def turn_toward_bearing(bearing: float) -> str:
    magnitude = abs(bearing)
    if magnitude <= SOFT_TURN_THRESHOLD_DEGREES:
        return "hold"
    strength = "hard" if magnitude >= HARD_TURN_THRESHOLD_DEGREES else "soft"
    direction = "left" if bearing > 0 else "right"
    return f"{strength}_{direction}"


def evade_strafe_direction(target: dict[str, Any]) -> str:
    return "right" if float(target["relative_bearing_degrees"]) >= 0 else "left"


def needs_off_axis_evasion(target: dict[str, Any], observation: dict[str, Any]) -> bool:
    if abs(float(target["relative_bearing_degrees"])) <= ENGAGEMENT_ADVANCE_ALIGNMENT_DEGREES:
        return False
    return nearest_hostile_distance(observation) <= FAR_ENGAGEMENT_DISTANCE


def nearest_hostile_distance(observation: dict[str, Any]) -> float:
    distances = [
        float(item.get("distance", float("inf")))
        for item in observation.get("visible_objects", [])
        if is_hostile(item)
    ]
    return min(distances, default=float("inf"))


def turn_toward_clear_space(depth_grid: dict[str, Any] | None) -> str:
    medians = depth_sector_medians(depth_grid)
    if medians is None:
        return "soft_left"
    left, center, right = medians
    if center > NEAR_DEPTH_SIGNAL and center + SIDE_DEPTH_ADVANTAGE >= max(left, right):
        return "hold"
    return "soft_left" if left >= right else "soft_right"


def turn_toward_frontier(guide: FrontierGuide | None, depth_grid: dict[str, Any] | None) -> str:
    if guide is None or guide.direction == "center":
        return turn_toward_clear_space(depth_grid)
    return f"soft_{guide.direction}"


def recovery_direction(guide: FrontierGuide | None, fallback: str) -> str:
    if guide is None or guide.direction == "center":
        return fallback
    return guide.direction


def depth_is_enclosed(depth_grid: dict[str, Any] | None) -> bool:
    medians = depth_sector_medians(depth_grid)
    return medians is not None and max(medians) <= CORNER_DEPTH_SIGNAL


def opposite_direction(direction: str) -> str:
    return "right" if direction == "left" else "left"


def depth_sector_medians(depth_grid: dict[str, Any] | None) -> tuple[float, float, float] | None:
    if not depth_grid or not depth_grid.get("values"):
        return None
    rows = depth_grid["values"][:-1] or depth_grid["values"]
    columns = int(depth_grid.get("columns", len(rows[0])))
    if columns < 3:
        return None
    boundaries = (0, columns // 3, 2 * columns // 3, columns)
    medians = []
    for start, end in zip(boundaries, boundaries[1:]):
        values = sorted(float(value) for row in rows for value in row[start:end])
        middle = len(values) // 2
        medians.append(values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2)
    return medians[0], medians[1], medians[2]


def require_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"invalid semantic {name} '{value}', expected one of {choices}")


SEMANTIC_INSTRUCTIONS = """Choose a short-lived semantic control intent for the Doom player.
First choose IMMEDIATE_MODE. When a live hostile is visible, choose fight or evade before exploration: fight retains the selected hostile as the immediate priority, turns to keep it centered, holds forward and strafe movement until it is centered enough to pursue, retreats from close threats, and fires immediately once aligned. Prefer a close, already-centered hostile over one at the edge of view when several are visible. Do not abandon a visible selected hostile to explore or run past it. Evade backs away while keeping the selected threat in view. When no live hostile is visible, choose explore or hold. If COMBAT_PRESSURE is active, choose the required left or right DODGE direction to avoid continuing on a predictable line. All returned channels execute simultaneously. FACE selects the target or viewing skill; MOVE controls exploration movement; DODGE independently adds lateral movement; USE and WEAPON are impulses.
While an explore intent is waiting for its next decision, a newly visible hostile interrupts exploration: the motor stops, turns toward the most centered visible hostile, and fires only if it is aligned. This reflex never fires at an unseen target and does not override a fight or evade intent.
Choose only options offered in the schema. Hostile face choices refer to exact objects in visible_hostiles. recent_hostile choices come only from a hostile directly observed in the last short interval; use one to reacquire a lost threat, never to fire. When movement_feedback reports blocked forward progress, choose backtrack_and_turn to perform a brief retreat and hard turn before reassessing. clear_space follows local depth guidance. explore_frontier uses a small memory of recently occupied cells to favor a locally open direction that has been visited less often; it is not a complete map route. When clear_space is advancing and depth samples indicate an enclosed corner, the motor performs a short reverse-and-strafe exit attempt. Scanning observes one side without choosing a target. Standing still remains valid when tactically useful.
The motor executes the intent in real time for a bounded lease while the next decision is computed. Choose an intent that is safe to hold briefly and follows the current higher-level plan."""


COMPACT_SEMANTIC_INSTRUCTIONS = """Choose a short-lived Doom control intent. All channels execute together.
When a live hostile is visible, choose fight and select its hostile face choice; do not explore or run past it. Fight turns toward the target and fires when aligned. Evade is only available during close-range or active combat pressure, and retreats while keeping the target in view. Prefer a close or centered hostile. With no hostile, explore or hold.
Choose left or right dodge during combat pressure. When movement feedback reports blocked progress, use backtrack_and_turn. clear_space follows open depth; explore_frontier follows less-visited open space. Weapon changes are only available when the current weapon is empty. Choose only schema options. Standing still is only for a concrete tactical reason."""
