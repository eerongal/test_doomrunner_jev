from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

CELL_SIZE = 32.0
MIN_FRONTIER_DEPTH = 8.0
MAX_FRONTIER_DISTANCE = 192.0
SECTOR_ANGLES = {"left": 35.0, "center": 0.0, "right": -35.0}


@dataclass(frozen=True)
class FrontierGuide:
    direction: str
    depth: float
    visits: int


class SeenSpaceNavigator:
    """A small local memory of occupied space, derived only from player motion and depth."""

    def __init__(self) -> None:
        self._visits: Counter[tuple[int, int]] = Counter()
        self._guide: FrontierGuide | None = None

    def reset(self) -> None:
        self._visits.clear()
        self._guide = None

    def observe(self, observation: dict[str, Any]) -> None:
        player = observation.get("player", {})
        position = player.get("position")
        heading = player.get("heading_degrees")
        medians = depth_sector_medians(observation.get("depth_grid"))
        if position is None or heading is None or medians is None:
            self._guide = None
            return
        self._visits[cell_for(position)] += 1
        self._guide = self._best_frontier(position, float(heading), medians)

    def guide(self) -> FrontierGuide | None:
        return self._guide

    def summary(self) -> dict[str, Any]:
        guide = self._guide
        if guide is None:
            return {"status": "unavailable"}
        return {
            "status": "available",
            "preferred_direction": guide.direction,
            "depth_signal": round(guide.depth, 1),
            "visited_candidate_cells": guide.visits,
            "note": "A low visit count means locally seen-space memory favors that open direction.",
        }

    def _best_frontier(
        self,
        position: list[float],
        heading: float,
        medians: tuple[float, float, float],
    ) -> FrontierGuide | None:
        candidates = []
        for direction, depth in zip(("left", "center", "right"), medians):
            if depth < MIN_FRONTIER_DEPTH:
                continue
            visits = self._visits[cell_for(project(position, heading + SECTOR_ANGLES[direction], depth))]
            candidates.append((visits, -depth, direction, depth))
        if not candidates:
            return None
        visits, _, direction, depth = min(candidates)
        return FrontierGuide(direction=direction, depth=depth, visits=visits)


def cell_for(position: list[float]) -> tuple[int, int]:
    return math.floor(float(position[0]) / CELL_SIZE), math.floor(float(position[1]) / CELL_SIZE)


def project(position: list[float], heading_degrees: float, depth: float) -> list[float]:
    distance = min(float(depth), MAX_FRONTIER_DISTANCE)
    radians = math.radians(heading_degrees)
    return [float(position[0]) + math.cos(radians) * distance, float(position[1]) + math.sin(radians) * distance]


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
