from __future__ import annotations

from typing import Any


WEAPON_AMMO = {1: None, 2: "bullets", 3: "shells", 4: "bullets", 5: "rockets", 6: "cells", 7: "cells", 8: None, 9: "shells"}
WEAPON_POWER = {1: 5, 2: 20, 3: 60, 4: 75, 5: 80, 6: 90, 7: 100, 8: 35, 9: 85}
CLOSE_RANGE = 160.0
ROCKET_SAFE_RANGE = 240.0


def recommended_weapon_action(observation: dict[str, Any], target: dict[str, Any] | None) -> str:
    if target is None:
        return "keep"
    player = observation.get("player", {})
    selected = int(player.get("weapon", 0))
    preferred = preferred_weapon(player, float(target.get("distance", float("inf"))))
    return "keep" if preferred is None or preferred == selected else f"select_{preferred}"


def preferred_weapon(player: dict[str, Any], distance: float) -> int | None:
    owned = {int(number) for number in player.get("owned_weapons", [])}
    candidates = [number for number in owned if weapon_is_viable(number, player, distance)]
    if not candidates:
        return None
    return max(candidates, key=lambda number: weapon_score(number, distance))


def weapon_has_ammo(weapon: int, player: dict[str, Any]) -> bool:
    ammo_type = WEAPON_AMMO.get(weapon)
    return ammo_type is None or int(player.get("ammo", {}).get(ammo_type, 0)) > 0


def weapon_is_viable(weapon: int, player: dict[str, Any], distance: float) -> bool:
    if weapon in {1, 8} and distance > CLOSE_RANGE:
        return False
    return weapon_has_ammo(weapon, player)


def weapon_score(weapon: int, distance: float) -> int:
    if weapon == 5 and distance < ROCKET_SAFE_RANGE:
        return 0
    if weapon == 9 and distance <= CLOSE_RANGE:
        return WEAPON_POWER[weapon] + 20
    if weapon == 4 and distance > CLOSE_RANGE:
        return WEAPON_POWER[weapon] + 15
    return WEAPON_POWER.get(weapon, 0)
