"""Every scene the robot can run in, in one place. A scenario is plugged in by adding ONE file here:

    scenarios/<name>.py           a procedural scenario: `build(difficulty, seed) -> core.Scenario`
                                  (pure data, no Isaac Sim imports; see laboratory.py)
    scenarios/scenes/<name>.json  a fixed scene: {"name", "description", "objects": [{"name", "shape", "xy", ...}]}

Nothing else changes: the registry finds it (available()), agent/settings.py accepts SCENARIO = "<name>",
./launch/sim.sh and ./launch/agent.sh start it, and sim/run_scenario.py builds it. The agent never imports
scenarios: it only sees the cameras, joints and measured objects.

    from scenarios import available, generate
    available()                                          # {"default": "frozen", "laboratory": "procedural", ...}
    sc = generate("laboratory", "medium", seed=42)      # -> core.Scenario (deterministic for (name, difficulty, seed))
    sc.metadata()                                        # ground truth (evaluation only, never given to the agent)

Registered now:
    default         default.py      the frozen cube scene of the manipulation baseline (referenced, not copied)
    laboratory      laboratory.py   procedural sample-handling bench, easy / medium / hard x seed
    ap_occlusion    scenes/ap_occlusion.json    hidden objects (active perception)
    ap_categories   scenes/ap_categories.json   new object categories (active perception)

Infrastructure (not scenarios): core.py (data model, geometry), parts.py (reusable fixtures and objects),
build.py (Isaac side: install / populate), checks.py (offline design checks:
python -m scenarios.checks --scenario laboratory --difficulty easy --seed 42 --out design.json).

Every scenario shares the work cell of the frozen base scene (robot, table camera on its stand, microphone prop);
config/perception.py and frozen control/robot.py describe that cell.
Simulator:  ./launch/sim.sh --scenario <name> [--difficulty D --seed N]
Validation: ./launch/evaluate.sh validate-scenarios --scenario laboratory
Docs:       docs/scenarios/README.md, docs/scenarios/LABORATORY.md
"""
from __future__ import annotations

import pkgutil
from importlib import import_module
from pathlib import Path

from scenarios.core import DIFFICULTIES, Scenario

HERE = Path(__file__).resolve().parent
SCENES = HERE / "scenes"
INFRASTRUCTURE = {"core", "parts", "build", "checks", "default"}


def _procedural() -> dict[str, str]:
    """Modules in scenarios/ that define build(difficulty, seed): name -> module path."""
    out = {}
    for m in pkgutil.iter_modules([str(HERE)]):
        if m.ispkg or m.name in INFRASTRUCTURE or m.name.startswith("_"):
            continue
        if hasattr(import_module(f"scenarios.{m.name}"), "build"):
            out[m.name] = f"scenarios.{m.name}"
    return out


SCENARIOS = _procedural()  # procedural scenarios (scenario library): name -> module


def available() -> dict[str, str]:
    """Every selectable scenario: name -> kind ("frozen" | "procedural" | "json")."""
    return {"default": "frozen", **{n: "procedural" for n in sorted(SCENARIOS)},
            **{p.stem: "json" for p in sorted(SCENES.glob("*.json"))}}


def kind(name: str) -> str | None:
    return available().get(name)


def scene_file(name: str) -> Path:
    """The JSON file of a fixed scene."""
    return SCENES / f"{name}.json"


def generate(name: str, difficulty: str = "easy", seed: int = 0) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; library: {sorted(SCENARIOS)}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    return import_module(SCENARIOS[name]).build(difficulty, int(seed))
