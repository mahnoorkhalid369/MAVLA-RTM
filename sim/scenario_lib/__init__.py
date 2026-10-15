"""Multi-scenario benchmark library (procedural Isaac Sim workstations, pure-data definitions).

    from sim.scenario_lib import generate, SCENARIOS
    sc = generate("laboratory", "medium", seed=42)     # -> core.Scenario (deterministic for (name, difficulty, seed))
    sc.metadata()                                       # ground truth (evaluation only, never given to the agent)

Simulator: python -m sim.run_scenario --scenario laboratory --difficulty medium --seed 42 [--headless]
Validation: .venv/bin/python evaluation/validate_scenarios.py --scenario laboratory
"""

from __future__ import annotations

from importlib import import_module

from sim.scenario_lib.core import DIFFICULTIES, Scenario

SCENARIOS = {"laboratory": "sim.scenario_lib.laboratory"}


def generate(name: str, difficulty: str = "easy", seed: int = 0) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; library: {sorted(SCENARIOS)}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    return import_module(SCENARIOS[name]).build(difficulty, int(seed))
