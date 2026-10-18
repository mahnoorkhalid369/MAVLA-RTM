"""Repository structure and configuration: copied constants still equal their frozen sources, paths point at real
files, the capability registry is consistent, prompts are unchanged, every current module imports, active code
uses no removed import path, and no harness replaces production functions at runtime.

The agent-environment checks are skipped where pydantic / langgraph are missing (system Python)."""

import ast
import hashlib
import importlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ACTIVE = ["agent", "perception", "config", "scenarios", "sim", "evaluation", "tests", "launch"]


def py_files(dirs=ACTIVE):
    for d in dirs:
        yield from (p for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts)


# ------------------------------------------------------------------------------------------ configuration
def test_copied_microphone_position_equals_the_frozen_scene():
    from config.perception import FIXTURES, MIC_POSITION
    from sim.scene_objects import MIC_POSITION as FROZEN
    assert MIC_POSITION == FROZEN
    mic = {f.name: f for f in FIXTURES}["microphone_base"]
    assert mic.centre[:2] == FROZEN[:2]


def test_table_camera_fixtures_follow_the_frozen_camera_pose():
    from config.perception import FIXTURES
    from control.robot import TABLE_CAMERA_POSITION
    pole = {f.name: f for f in FIXTURES}["camera_stand_pole"]
    assert pole.centre[:2] == TABLE_CAMERA_POSITION[:2] and pole.half[2] * 2 == TABLE_CAMERA_POSITION[2]


def test_reach_ring_and_tolerance_have_one_source():
    from config.agent import REACH_MAX_M, REACH_MIN_M, REACH_TOLERANCE_DEG
    from perception.viewpoints import REACH_TOL_DEG
    from scenarios.core import REACH_MAX, REACH_MIN
    assert (REACH_MIN, REACH_MAX) == (REACH_MIN_M, REACH_MAX_M) == (0.45, 0.90)
    assert REACH_TOL_DEG == REACH_TOLERANCE_DEG == 1.5


def test_tolerances_and_budgets_keep_the_baseline_values():
    from config.agent import Budgets, Tolerances
    t = Tolerances()
    assert (t.xy_m, t.z_m, t.tilt_deg, t.contact_deg, t.min_support_width_m) == (0.010, 0.005, 5.0, 1.5, 0.030)
    assert Budgets().retries == {"perception": 2, "planning": 2, "grasp": 1, "execution": 0, "placement": 1,
                                 "understanding": 0}


# ------------------------------------------------------------------------------------------------- paths
def test_model_and_scenario_paths_are_inside_the_repository():
    from config.models import YOLO_WEIGHTS
    from scenarios.build import ROOT as BUILD_ROOT, TEXTURES
    assert BUILD_ROOT == ROOT and TEXTURES == ROOT / "out" / "scenario_textures"
    assert YOLO_WEIGHTS == ROOT / "out/seg/runs/yolo11s_seg_r1/weights/best.pt"


@pytest.mark.parametrize("scene", ["ap_occlusion", "ap_categories"])
def test_json_scenes_exist_where_run_scenario_reads_them(scene):
    spec = json.loads((ROOT / "scenarios" / "scenes" / f"{scene}.json").read_text())
    assert spec["objects"] and all("shape" in o and "xy" in o for o in spec["objects"])
    assert '(ROOT / "scenarios" / "scenes" / f"{args.scenario}.json")' in (ROOT / "sim" / "run_scenario.py").read_text()


def test_required_untracked_assets_are_present():
    """Models and asset tables are not in git (large / downloaded); this reports which are missing on this machine."""
    from config.models import YOLO_WEIGHTS
    required = {"YOLO weights": YOLO_WEIGHTS, "Gemma 4 E2B": ROOT / "models" / "gemma-4-E2B-it" / "config.json",
                "YCB asset table (JSON scenes)": ROOT / "data" / "assets" / "asset_dims.json"}
    missing = [k for k, p in required.items() if not p.exists()]
    if missing:
        pytest.skip(f"not on this machine: {missing} (see docs/validation/FINAL_MAIN_READINESS.md, assets)")


def test_library_scenario_registry_resolves():
    import scenarios
    assert scenarios.SCENARIOS == {"laboratory": "scenarios.laboratory"}
    sc = scenarios.generate("laboratory", "easy", 42)
    assert sc.id == "laboratory" and sc.metadata()["objects"]


# --------------------------------------------------------------------------- agent: registry and prompts
PROMPT_SHA256_16 = {  # identical to the prompts before the 2026-10-17 restructuring (commit aa9a6a4)
    "ANSWER_PROMPT": "89876bc14da608e2",
    "COMPARE_PROMPT": "d9c0c00cf26b401e",
    "CONFIRM_PROMPT": "60d867c0f1984edf",
    "DECOMPOSE_PROMPT": "8fbef0fcba9bbf48",
    "DESCRIBE_ONE_PROMPT": "30e7f4389ba0ab9c",
    "DESCRIBE_PROMPT": "c76f8ba28b9ad381",
    "PARSE_SYSTEM": "35cb5a94bc3508a9",
    "PLAN_SYSTEM": "0f14440da0535f1c",
    "REQUEST_SYSTEM": "be95de3b6adede69",
    "RESOLVE_PROMPT": "8195f33d0eec2a60",
    "SHORTLIST_PROMPT": "32ae760793cc8575",
}


def test_prompts_are_unchanged():
    """A prompt change changes the experiment: it must be deliberate, with this table updated in the same commit."""
    pytest.importorskip("pydantic")
    import agent.vlm_roles as v
    names = sorted(n for n in dir(v) if n.endswith(("_PROMPT", "_SYSTEM")))
    assert names == sorted(PROMPT_SHA256_16)
    assert {n: hashlib.sha256(getattr(v, n).encode()).hexdigest()[:16] for n in names} == PROMPT_SHA256_16


def test_capability_registry_planner_schema_and_supervisor_agree(tmp_path):
    pytest.importorskip("pydantic")
    pytest.importorskip("langgraph")
    from typing import get_args

    from agent.capabilities import EFFECT, INTERNAL, SPECS
    from agent.supervisor import GoalSupervisor
    from agent.vlm_roles import PlanStep
    from tests.support.fakes import GoalTools
    assert set(get_args(PlanStep.model_fields["do"].annotation)) == set(SPECS)  # the planner can name exactly these
    sup = GoalSupervisor(GoalTools([]), lambda *a, **k: None, tmp_path, None)
    # every capability has an executor; internal operations (verify) are delegated but never plannable
    assert {c for w in sup.reg.workers.values() for c in w.operations} == set(SPECS) | set(INTERNAL)
    assert not set(INTERNAL) & set(get_args(PlanStep.model_fields["do"].annotation))
    assert sup.reg.worker_of("verify") == "verification" and INTERNAL["verify"].effect == "none"
    assert {n for n, s in SPECS.items() if s.effect == "object_change"} == {"pick", "place", "pick_place"}
    assert {n for n, s in SPECS.items() if s.safety == "motion"} == {"pick", "place", "pick_place", "return_home",
                                                                     "inspect", "explore"}
    assert set(EFFECT) <= set(SPECS)


CURRENT_MODULES = [
    "agent.run", "agent.session", "agent.supervisor", "agent.planning", "agent.controls", "agent.task_state",
    "agent.workers", "agent.authority", "agent.goals", "agent.pick_place", "agent.ask", "agent.eventlog",
    "agent.events", "agent.live", "agent.tools", "agent.capabilities", "agent.vlm_roles",
    "perception.detector", "perception.geometry", "perception.somark", "perception.world_model",
    "perception.viewpoints", "config.perception", "config.agent", "config.models",
    "scenarios", "scenarios.core", "scenarios.parts", "scenarios.laboratory", "scenarios.checks",
]


@pytest.mark.parametrize("name", CURRENT_MODULES)
def test_current_module_imports_without_simulator_or_ros(name):
    if name.startswith("agent."):
        pytest.importorskip("pydantic")
        pytest.importorskip("langgraph")
    importlib.import_module(name)


# --------------------------------------------------------------------------------------- code hygiene
REMOVED = re.compile(r"\b(agent\.(detector|geometry|somark|world|exploration|config|world_state)|sim\.scenario_lib)\b"
                     r"|\bagent/(detector|geometry|somark|world|exploration|config|world_state)\.py\b"
                     r"|\bsim/scenario_lib\b|\bsim/scenarios/")


def test_no_removed_import_path_in_active_code():
    hits = []
    for p in list(py_files()) + list((ROOT / "launch").glob("*.sh")):
        if p.name == "test_structure.py":
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if REMOVED.search(line):
                hits.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not hits, "\n".join(hits)


def test_no_harness_reassigns_production_functions():
    """Experiment controls are explicit (agent/controls.py): no module attribute of agent / perception code is
    reassigned at runtime in evaluation/ or experiments/ (tests use pytest's monkeypatch, which is undone)."""
    hits = []
    for p in py_files(["evaluation", "experiments", "agent", "perception", "scenarios"]):
        tree = ast.parse(p.read_text())
        modules = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
                   if a.name.split(".")[0] in ("agent", "perception", "config")}
        modules |= {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
                    and n.module.split(".")[0] in ("agent", "perception", "config") for a in n.names}
        for n in ast.walk(tree):
            if isinstance(n, (ast.Assign, ast.AugAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    if reassigns_module_attribute(t, modules):
                        hits.append(f"{p.relative_to(ROOT)}:{n.lineno}")
    assert not hits, hits


def reassigns_module_attribute(target, modules) -> bool:
    """`mod.x = ...`, `mod.Cls.x = ...`, `mod.TABLE[k] = ...`, `mod.a, mod.b = ...` with mod an imported module."""
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(reassigns_module_attribute(e, modules) for e in target.elts)
    if not isinstance(target, (ast.Attribute, ast.Subscript)):
        return False
    node = target.value
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return isinstance(node, ast.Name) and node.id in modules
