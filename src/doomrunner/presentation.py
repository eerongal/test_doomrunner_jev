from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from doomrunner.inference import Plan
from doomrunner.semantic import DODGE_SKILLS, FACE_SKILLS, IMMEDIATE_MODES, MOVE_SKILLS, TRIGGER_SKILLS, USE_SKILLS, WEAPON_CHOICES, SemanticIntent

LOGGER = logging.getLogger(__name__)
PANEL_WIDTH = 300
OBJECTIVE_HEIGHT = 144
BACKGROUND = "#111827"
PANEL_BACKGROUND = "#172033"
INACTIVE = "#273449"
INACTIVE_TEXT = "#93a4bc"
ACTIVE = "#13d8c8"
ACTIVE_TEXT = "#041615"
TEXT = "#e7edf7"
MUTED_TEXT = "#aab8cc"


@dataclass(frozen=True)
class ActionRow:
    label: str
    options: tuple[str, ...]
    selected: str | None


@dataclass(frozen=True)
class ObjectiveDisplay:
    header: str
    primary: str
    secondary: str


def action_matrix(intent: SemanticIntent | None, action: dict[str, Any] | None = None) -> tuple[ActionRow, ...]:
    selected = intent or SemanticIntent(mode="hold", face="hold", move="hold")
    use = "use" if action and action.get("use") else selected.use
    return (
        ActionRow("MODE", IMMEDIATE_MODES, selected.mode if intent else None),
        ActionRow("FACE", FACE_SKILLS + (selected.face,), selected.face if intent else None),
        ActionRow("MOVE", MOVE_SKILLS, selected.move if intent else None),
        ActionRow("DODGE", DODGE_SKILLS, selected.dodge if intent else None),
        ActionRow("FIRE REQUEST", TRIGGER_SKILLS, selected.trigger if intent else None),
        ActionRow("USE (MOTOR)", USE_SKILLS, use if intent else None),
        ActionRow("WEAPON", WEAPON_CHOICES, selected.weapon if intent else None),
    )


class PresentationDisplay:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._root: Any | None = None
        self._canvas: Any | None = None
        self._photo: Any | None = None
        self._closed = False
        self._frame_unavailable_reported = False

    def render(
        self,
        frame: Any | None,
        observation: dict[str, Any] | None,
        plan: Plan,
        intent: SemanticIntent | None,
        action: dict[str, Any],
        decision_latency_ms: float | None,
        planner_status: str,
        control_override: str | None,
    ) -> None:
        if not self.enabled or self._closed:
            return
        if frame is None:
            if not self._frame_unavailable_reported:
                LOGGER.warning("Presentation display is waiting for a ViZDoom screen buffer")
                self._frame_unavailable_reported = True
            return
        try:
            self._open(frame)
            self._draw(frame, observation or {}, plan, intent, action, decision_latency_ms, planner_status, control_override)
        except Exception as error:
            LOGGER.error("Presentation display disabled: %s", error, exc_info=True)
            self._closed = True

    def close(self) -> None:
        if self._root is None:
            return
        try:
            self._root.destroy()
        except Exception:
            pass
        self._root = None
        self._canvas = None
        self._photo = None

    def _open(self, frame: Any) -> None:
        if self._root is not None:
            return
        import tkinter as tk

        width, height = frame_dimensions(frame)
        self._root = tk.Tk()
        self._root.title("Doomrunner")
        self._root.configure(background=BACKGROUND)
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._close_window)
        self._canvas = tk.Canvas(
            self._root,
            width=PANEL_WIDTH + width,
            height=height + OBJECTIVE_HEIGHT,
            bg=BACKGROUND,
            highlightthickness=0,
        )
        self._canvas.pack()
        LOGGER.info("Presentation display opened frame=%sx%s", width, height)

    def _draw(
        self,
        frame: Any,
        observation: dict[str, Any],
        plan: Plan,
        intent: SemanticIntent | None,
        action: dict[str, Any],
        decision_latency_ms: float | None,
        planner_status: str,
        control_override: str | None,
    ) -> None:
        from PIL import Image, ImageTk

        if self._canvas is None or self._root is None:
            return
        rgb = frame_to_rgb(frame)
        height, width = rgb.shape[:2]
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._canvas.delete("all")
        self._canvas.create_image(PANEL_WIDTH, 0, anchor="nw", image=self._photo)
        self._draw_action_panel(observation, intent, action, decision_latency_ms, planner_status, height)
        self._draw_objective_panel(plan, control_override, width, height)
        self._root.update_idletasks()
        self._root.update()

    def _draw_action_panel(
        self,
        observation: dict[str, Any],
        intent: SemanticIntent | None,
        action: dict[str, Any],
        decision_latency_ms: float | None,
        planner_status: str,
        game_height: int,
    ) -> None:
        if self._canvas is None:
            return
        player = observation.get("player", {})
        combat = observation.get("combat", {})
        self._canvas.create_rectangle(0, 0, PANEL_WIDTH, game_height, fill=PANEL_BACKGROUND, outline="")
        self._canvas.create_text(16, 16, anchor="nw", text="DOOMRUNNER // L1", fill=ACTIVE, font=("Segoe UI", 12, "bold"))
        latency = "--" if decision_latency_ms is None else f"{decision_latency_ms:.0f} ms"
        stats = (
            f"HP {player.get('health', '--')}   ARM {player.get('armor', '--')}\n"
            f"AMMO {player.get('weapon_ammo', '--')}   KILLS {combat.get('kills', '--')}\n"
            f"L1 {latency}   L2 {planner_status}\n"
            f"GAME TRIGGER {'ON' if action.get('fire', False) else 'OFF'}"
        )
        self._canvas.create_text(16, 42, anchor="nw", text=stats, fill=TEXT, font=("Segoe UI", 9))
        y = 115
        for row in action_matrix(intent, action):
            self._canvas.create_text(16, y, anchor="nw", text=row.label, fill=MUTED_TEXT, font=("Segoe UI", 8, "bold"))
            y += 15
            y = self._draw_options(row, y)
            y += 8
        motor = (
            f"MOTOR  {action.get('movement', 'stop')} / {action.get('strafe', 'stop')} / "
            f"{action.get('turn', 'hold')}  FIRE={'ON' if action.get('fire') else 'OFF'} "
            f"USE={'ON' if action.get('use') else 'OFF'}"
        )
        self._canvas.create_text(16, game_height - 28, anchor="nw", text=motor, fill=MUTED_TEXT, font=("Segoe UI", 8))

    def _draw_options(self, row: ActionRow, y: int) -> int:
        if self._canvas is None:
            return y
        x = 16
        for option in dict.fromkeys(row.options):
            label = compact_option(option)
            width = max(44, len(label) * 7 + 16)
            if x + width > PANEL_WIDTH - 12:
                x = 16
                y += 25
            active = option == row.selected
            self._canvas.create_rectangle(
                x,
                y,
                x + width,
                y + 20,
                fill=ACTIVE if active else INACTIVE,
                outline="",
            )
            self._canvas.create_text(
                x + width / 2,
                y + 10,
                text=label,
                fill=ACTIVE_TEXT if active else INACTIVE_TEXT,
                font=("Segoe UI", 8, "bold" if active else "normal"),
            )
            x += width + 5
        return y + 20

    def _draw_objective_panel(
        self,
        plan: Plan,
        control_override: str | None,
        game_width: int,
        game_height: int,
    ) -> None:
        if self._canvas is None:
            return
        total_width = PANEL_WIDTH + game_width
        top = game_height
        self._canvas.create_rectangle(0, top, total_width, top + OBJECTIVE_HEIGHT, fill="#0b1220", outline="")
        display = objective_display(plan, control_override)
        self._canvas.create_text(16, top + 14, anchor="nw", text=display.header, fill=ACTIVE, font=("Segoe UI", 10, "bold"))
        self._canvas.create_text(
            16,
            top + 38,
            anchor="nw",
            text=truncate(display.primary, 150),
            width=total_width - 32,
            fill=TEXT,
            font=("Segoe UI", 12, "bold"),
        )
        self._canvas.create_text(
            16,
            top + 83,
            anchor="nw",
            text=truncate(display.secondary, 160),
            width=total_width - 32,
            fill=MUTED_TEXT,
            font=("Segoe UI", 9),
        )

    def _close_window(self) -> None:
        self._closed = True
        self.close()


def frame_dimensions(frame: Any) -> tuple[int, int]:
    rgb = frame_to_rgb(frame)
    height, width = rgb.shape[:2]
    return width, height


def frame_to_rgb(frame: Any) -> Any:
    if getattr(frame, "ndim", 0) != 3:
        raise ValueError("ViZDoom screen buffer must have three dimensions")
    if frame.shape[0] in {3, 4}:
        frame = frame.transpose(1, 2, 0)
    if frame.shape[-1] < 3:
        raise ValueError("ViZDoom screen buffer must contain RGB channels")
    return frame[:, :, :3].copy()


def compact_option(option: str) -> str:
    return option.replace("_", " ").upper()


def objective_display(plan: Plan, control_override: str | None) -> ObjectiveDisplay:
    if control_override:
        return ObjectiveDisplay(
            "ACTIVE SAFETY OVERRIDE // NOT AN L1 DECISION",
            control_override,
            f"SAVED L2 PLAN: {plan.goal}",
        )
    return ObjectiveDisplay(
        "CURRENT OBJECTIVE PASSED TO L1",
        plan.immediate_goal,
        f"PLAN: {plan.goal}",
    )


def truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else f"{value[: limit - 1].rstrip()}…"
