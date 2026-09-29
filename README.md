# Doomrunner

Doomrunner is a Python ViZDoom harness for testing a two-level LLM controller in real time.

- Level 1 is a local, fast JEV-style decision request to llama-server's `/v1/decision` endpoint. It selects simultaneous movement, strafing, turn, fire, use, and weapon actions.
- Level 2 is a slower tactical planner that refreshes the objective, mobility policy, and short navigation directives through `/v1/chat/completions`.
- Doom continues in real time while either level is computing. A slow plan never pauses the game.

The local inference server is the CUDA-enabled fork at [thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp).

## What is included

The repository contains the Doomrunner source, tests, configuration profiles, and helper scripts. It intentionally does not contain proprietary WADs, model files, telemetry, virtual environments, or API credentials.

Ignored local-only files include:

- `.env` - OpenRouter key
- `wads/` and `*.wad` - Doom game data
- `logs/` - runtime telemetry
- `.venv/` and Python cache files

## Requirements

- Windows with Python 3.10 or later
- ViZDoom, installed through the project dependencies
- A Doom II or Ultimate Doom WAD that you own
- A built local [llama.cpp fork](https://github.com/thecodacus/llama.cpp) server with the `/v1/decision` endpoint enabled
- One local Level 1 model configured in `models.ini`

## Install

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

If your Microsoft Store Python installation cannot start a virtual environment, install Python from python.org or select another working `py` interpreter and recreate `.venv`.

## Start local inference

Start the modified llama.cpp server separately. The project profiles expect it on port `8096`.

```powershell
C:\development\aistuff\llama.cpp\build-cuda\bin\Release\llama-server.exe `
  --models-preset C:\development\aistuff\llama.cpp\models.ini `
  --models-max 2 `
  --port 8096
```

For any model used by Level 1, set `decision-seqs` to at least `3` in `models.ini`. A planner-only model may use `decision-seqs=0`.

`--models-max 2` permits a local Level 1 and local Level 2 model pair. Use `--models-max 1` for isolated Level 1 testing or when VRAM is tight.

## Configuration profiles

All profiles are under `config/`. Change `doom.wad_path` to a local WAD location when needed.

| Profile | Scenario | Level 1 | Level 2 |
| --- | --- | --- | --- |
| `doomrunner.json` | ViZDoom deathmatch | Qwen3.5-4B | Qwen3.5-2B, local |
| `doomrunner-combat.json` | `defend_the_center` arena | Qwen3.5-4B semantic control | Qwen3.5-2B, local |
| `doomrunner-e1m1.json` | Ultimate Doom E1M1 | Qwen3.5-4B semantic control | Qwen3.5-2B, local |
| `doomrunner-e1m1-2b-walk.json` | Ultimate Doom E1M1 | Qwen3.5-2B semantic control | Qwen3.5-2B, local |
| `doomrunner-e1m1-openrouter.json` | Ultimate Doom E1M1 | Qwen3.5-2B semantic control | `stealth/space-bunny-alpha` via OpenRouter |
| `doomrunner-e1m1-nemotron.json` | Ultimate Doom E1M1 | Qwen3.5-2B semantic control | `nvidia/nemotron-3.5-lightning:free` via OpenRouter |

The E1M1 profiles expect an ignored `wads/doomu.wad`. The arena and default profiles currently use the local Doom II WAD path recorded in their JSON files.

## Run

Validate the active configuration and service connections first:

```powershell
doomrunner --config config/doomrunner-combat.json --check
```

Run the semantic arena profile:

```powershell
doomrunner --config config/doomrunner-combat.json --mode jev --episodes 1
```

Run E1M1 with local planning:

```powershell
doomrunner --config config/doomrunner-e1m1.json --mode jev --episodes 1
```

Run E1M1 using the smaller local model for both levels and walking speed:

```powershell
doomrunner --config config/doomrunner-e1m1-2b-walk.json --mode jev --episodes 1
```

Run Level 1 only:

```powershell
doomrunner --config config/doomrunner-combat.json --mode jev --no-planner --episodes 1
```

Run the heuristic baseline without inference:

```powershell
doomrunner --config config/doomrunner-combat.json --mode heuristic --episodes 1
```

## OpenRouter Level 2

Create an ignored `.env` file in the repository root containing your key:

```text
OPEN_ROUTER_KEY=your-key-here
```

Then use one of the OpenRouter profiles:

```powershell
doomrunner --config config/doomrunner-e1m1-openrouter.json --check
doomrunner --config config/doomrunner-e1m1-openrouter.json --mode jev --episodes 1
```

The Nemotron free-profile includes a 15-second planner deadline. This prevents very late remote plans from being applied, but free-provider queueing or provider degradation can still cause Level 2 retries. Level 1 continues to control Doom during those retries.

## Control modes

`observation_mode` is selected per profile or with `--observation-mode`.

- `raw`: model returns direct controls from current measurements and short factual history.
- `compact`: smaller neutral sensor summary.
- `assisted`: direct controls with derived targeting and movement feedback.
- `semantic`: model chooses a bounded intent; the local motor tracks the selected hostile, controls aiming and firing alignment, applies close-range evasions, and breaks failed forward movement.

Campaign profiles use semantic control. When forward movement is blocked without a visible hostile, the local motor presses `use` once to test a nearby door or switch, waits briefly for a possible door to rise, then applies a bounded recovery if movement still cannot proceed. This behavior is disabled in arena profiles.

## Presentation and telemetry

Set `doom.presentation_enabled` to `true` to show the Doomrunner overlay. It displays the current objective, saved Level 2 plan, active control override, player state, Level 1 latency, planner status, model intent, and the actual motor controls.

Runtime telemetry is written as JSON Lines in ignored `logs/` files. Accepted decisions include the applied action and a snapshot of the current player state and visible hostiles.

## Tests

```powershell
python -m pytest
```

The project uses an editable install, so source changes under `src/` apply to later runs without reinstalling the package.
