"""Scenario library: procedural Isaac Sim workstations as pure data, plus the small JSON test scenes.

    from scenarios import generate, SCENARIOS
    sc = generate("laboratory", "medium", seed=42)     # -> core.Scenario (deterministic for (name, difficulty, seed))
    sc.metadata()                                       # ground truth (evaluation only, never given to the agent)

Where things are:
    core.py         data model and geometry (Scenario, Obj, Fixture, Placer, penetration, metadata); no Isaac
    parts.py        reusable procedural fixtures and objects (rack, tray, plinth, shelf, capped container, ...)
    laboratory.py   the laboratory scenario: build(difficulty, seed), difficulty CONFIG table
    build.py        Isaac side: install() before the frozen scene setup, populate() after it (import only once
                    the SimulationApp exists)
    checks.py       offline design checks (spawn, visibility, grasp / placement / viewpoint feasibility):
                    python -m scenarios.checks --scenario laboratory --difficulty easy --seed 42 --out design.json
    scenes/*.json   small active-perception test scenes (ap_occlusion, ap_categories)

Simulator:  .venv/bin/python -m sim.run_scenario --scenario laboratory --difficulty medium --seed 42 [--headless]
Validation: .venv/bin/python evaluation/validate_scenarios.py --scenario laboratory
Docs:       docs/scenarios/README.md, docs/scenarios/LABORATORY.md
"""
from __future__ import annotations

from importlib import import_module

from scenarios.core import DIFFICULTIES, Scenario

SCENARIOS = {"laboratory": "scenarios.laboratory"}


def generate(name: str, difficulty: str = "easy", seed: int = 0) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; library: {sorted(SCENARIOS)}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    return import_module(SCENARIOS[name]).build(difficulty, int(seed))
