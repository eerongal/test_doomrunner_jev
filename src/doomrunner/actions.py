from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any


MOVEMENT_CHOICES = ("forward", "backward", "stop")
STRAFE_CHOICES = ("left", "right", "stop")
TURN_CHOICES = ("hard_left", "soft_left", "hold", "soft_right", "hard_right")
WEAPON_CHOICES = ("keep", "next", "previous")
WEAPON_SELECT_CHOICES = tuple(f"select_{number}" for number in range(1, 10))
WEAPON_ACTION_CHOICES = WEAPON_CHOICES + WEAPON_SELECT_CHOICES


@dataclass(frozen=True)
class GameAction:
    movement: str = "stop"
    strafe: str = "stop"
    turn: str = "hold"
    fire: bool = False
    use: bool = False
    weapon: str = "keep"

    @classmethod
    def from_decision(cls, value: dict[str, Any]) -> "GameAction":
        action = cls(
            movement=value.get("movement", "stop"),
            strafe=value.get("strafe", "stop"),
            turn=value.get("turn", "hold"),
            fire=value.get("fire", False),
            use=value.get("use", False),
            weapon=value.get("weapon", "keep"),
        )
        action.validate()
        return action

    def validate(self) -> None:
        _require_choice("movement", self.movement, MOVEMENT_CHOICES)
        _require_choice("strafe", self.strafe, STRAFE_CHOICES)
        _require_choice("turn", self.turn, TURN_CHOICES)
        _require_choice("weapon", self.weapon, WEAPON_ACTION_CHOICES)
        if type(self.fire) is not bool or type(self.use) is not bool:
            raise ValueError("fire and use must be booleans")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"invalid {name} '{value}', expected one of {choices}")


DECISION_SCHEMA: dict[str, dict[str, Any]] = {
    "movement": {
        "type": "enum",
        "choices": list(MOVEMENT_CHOICES),
        "description": "Choose forward or backward movement relative to current heading. Stop when movement is unsafe or unnecessary.",
    },
    "strafe": {
        "type": "enum",
        "choices": list(STRAFE_CHOICES),
        "description": "Choose lateral movement independently of forward movement. Strafe to evade attacks while fighting.",
    },
    "turn": {
        "type": "enum",
        "choices": list(TURN_CHOICES),
        "description": "Choose a horizontal turn rate held until the next decision. Positive target bearing requires left; negative requires right. When exploring without a target, scan with a soft turn instead of holding indefinitely.",
    },
    "fire": {
        "type": "boolean",
        "description": "Fire when a visible hostile is aligned closely enough and attack_ready is true.",
    },
    "use": {
        "type": "boolean",
        "description": "Activate a nearby door or switch only when interaction is useful.",
    },
    "weapon": {
        "type": "enum",
        "choices": list(WEAPON_CHOICES),
        "description": "Keep the current weapon unless it has no ammunition or is unsuitable.",
    },
}


ASSISTED_INSTRUCTIONS = """Control the Doom player to survive and defeat hostile monsters.
All control fields apply together. Movement and strafing are independent and may form a diagonal.
Choosing movement=stop and strafe=stop is valid. Hold position when the higher-level mobility policy requests it unless an immediate dodge is needed to survive.
When mobility_policy is free and no tactical_target is visible, actively explore by moving and scanning. Do not remain idle facing the same direction; standing still is reserved for a tactical reason or an explicit hold_position plan.
Turn is a continuous rate held until the next decision. A positive relative bearing is to the left and a negative bearing is to the right.
Focus on tactical_target while it is visible. Match the turn direction to its turn_hint; when turn_hint is aligned, choose hold or only a soft correction. If the target is temporarily lost, use its last bearing briefly instead of immediately choosing another distant target.
Use control_feedback to judge evaluated_action, which caused the reported aim_trend. current_action is the newer action being applied now. If aim_trend is worsening, do not repeat the evaluated turn direction. If possible_circling is true, break the turn streak.
If possible_stuck is true, do not repeat the blocked movement. Turn and strafe, reverse, or otherwise change the approach until distance_moved recovers.
Fire only when a hostile is aligned and the weapon is ready. Do not waste ammunition.
Use is a one-tic impulse for nearby doors and switches. Weapon changes are one-time impulses.
Prefer survival over chasing a distant target. Continue the current plan unless the observation makes it unsafe."""


RAW_INSTRUCTIONS = """Control the Doom player to survive and defeat hostile monsters.
All control fields apply together. Movement, strafing, turning, firing, use, and weapon selection are independent.
You receive one complete current sensor observation plus recent factual transitions ordered from oldest to newest. Each transition reports numerical sensor changes and the control applied during that interval.
Historical controls are causal evidence, not recommendations to repeat. Their encoding is: longitudinal_axis -1=backward, 0=stationary, 1=forward; lateral_axis -1=left, 0=stationary, 1=right; yaw_axis -2=hard left, -1=soft left, 0=hold, 1=soft right, 2=hard right; trigger and interact are 0 or 1; weapon_step -1=previous, 0=keep, 1=next.
Infer the meaning of those changes yourself. Decide whether the player is blocked, circling, aligned with a target, in danger, or should remain still.
Depth samples are raw ViZDoom unsigned byte values arranged from top to bottom and left to right; they are sensory input rather than map-unit distances.
Turn is a continuous rate held until the next decision. Positive relative object bearing is to the left and negative bearing is to the right.
Use and weapon changes are one-tic impulses. Prefer survival while making progress toward defeating threats."""


COMPACT_INSTRUCTIONS = """Control the Doom player to survive and defeat hostile monsters.
All control fields apply together. Movement, strafing, turning, firing, use, and weapon selection are independent.
You receive a compact set of current measurements. No target, obstruction, aim, or preferred action has been selected for you.
motion_since_previous_decision reports the measured result of the encoded control_applied during the preceding interval. Use the control and resulting distance, position, and heading changes together to assess its effect.
The control encoding is: longitudinal_axis -1=backward, 0=stationary, 1=forward; lateral_axis -1=left, 0=stationary, 1=right; yaw_axis -2=hard left, -1=soft left, 0=hold, 1=soft right, 2=hard right; trigger and interact are 0 or 1; weapon_step -1=previous, 0=keep, 1=next.
visible_hostiles and visible_resources_and_objects contain every currently reported object in their respective groups, without selecting a preferred one.
Depth sectors contain summarized raw ViZDoom depth signals, not map-unit distances. Compare the left, center, and right measurements as relative sensory evidence.
Turn is a continuous rate held until the next decision. Positive relative object bearing is to the left and negative bearing is to the right.
Use and weapon changes are one-tic impulses. Fire only when current measurements support doing so. Prefer survival while making progress toward the current plan."""


BASE_INSTRUCTIONS = ASSISTED_INSTRUCTIONS


def decision_schema(mobility_policy: str) -> dict[str, dict[str, Any]]:
    schema = deepcopy(DECISION_SCHEMA)
    policy_guidance = {
        "advance": "The plan requires forward movement; choose forward unless current evidence shows it is immediately blocked or unsafe.",
        "retreat": "The plan requires withdrawal; choose backward unless current evidence shows it is immediately blocked or unsafe.",
        "hold_position": "The plan requires holding position; choose stop unless movement is needed for an immediate dodge.",
        "seek_cover": "Move to reduce exposure; do not choose stop unless remaining stationary currently provides cover.",
        "free": "Actively explore when safe; prefer forward over stop when no immediate threat or obstruction is evident.",
    }
    guidance = policy_guidance.get(mobility_policy, policy_guidance["free"])
    schema["movement"]["description"] = f"{schema['movement']['description']} {guidance}"
    schema["fire"]["description"] = (
        "Fire now only when a hostile is currently visible, sufficiently aligned, and attack_ready is true. "
        "A historical trigger value is not a recommendation."
    )
    return schema
