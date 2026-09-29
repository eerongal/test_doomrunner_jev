from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from doomrunner.config import AppConfig
from doomrunner.controller import DoomController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the two-level Doom controller")
    parser.add_argument("--config", type=Path, default=Path("config/doomrunner.json"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--mode", choices=("jev", "heuristic"), default="jev")
    parser.add_argument("--decision-model", help="override the configured Level 1 model ID")
    parser.add_argument("--planner-model", help="override the configured Level 2 model ID")
    parser.add_argument(
        "--planner-reasoning-effort",
        choices=("none", "low", "medium", "high"),
        help="override Level 2 reasoning; use none for non-thinking models",
    )
    parser.add_argument("--no-planner", action="store_true", help="disable Level 2 for an isolated Level 1 run")
    parser.add_argument(
        "--observation-mode",
        choices=("raw", "compact", "semantic", "assisted"),
        help="override the Level 1 observation mode",
    )
    parser.add_argument("--check", action="store_true", help="validate ViZDoom and llama-server connectivity, then exit")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


async def run(args: argparse.Namespace) -> None:
    load_dotenv()
    config = apply_overrides(AppConfig.load(args.config), args)
    controller = DoomController(config, mode=args.mode)
    if args.check:
        await controller.check()
        return
    await controller.run(args.episodes)


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    server = replace(
        config.server,
        decision_model=args.decision_model or config.server.decision_model,
        planner_model=args.planner_model or config.server.planner_model,
        planner_reasoning_effort=args.planner_reasoning_effort or config.server.planner_reasoning_effort,
    )
    controller = replace(
        config.controller,
        planner_enabled=False if args.no_planner else config.controller.planner_enabled,
        observation_mode=args.observation_mode or config.controller.observation_mode,
    )
    return replace(config, server=server, controller=controller)


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.log_level != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.episodes <= 0:
        raise SystemExit("--episodes must be greater than zero")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Stopped by user")
    except Exception as error:
        logging.getLogger(__name__).error("Doomrunner stopped unexpectedly: %s", error, exc_info=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
