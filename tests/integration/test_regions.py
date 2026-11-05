"""Calibrated fixed regions (config/layout.py, 2026-11-04): the work cell's flat markings as scene entities with their
own ids (R<n>), never world objects. Gemma reads which region the words mean; Python accepts ids, computes geometry
(placement spots, containment) and keeps the id canonical through plan, request, result and verification.
No simulator, no GPU; the placement tests use the real kinematics and collision checks."""

import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pydantic")

import scenarios  # noqa: E402
from agent.goals import GoalEvaluator  # noqa: E402
from agent.tools import Tools  # noqa: E402
from agent.vlm_roles import PlanStep, RecordedVLM, Roles, SubPlan  # noqa: E402
from config import layout  # noqa: E402
from control.robot import ARM_HOME  # noqa: E402
from scenarios import workcell  # noqa: E402
from tests.support.fakes import GoalTools, geom  # noqa: E402
from tests.support.runners import plan, run_supervisor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def cell(monkeypatch):
    """The work cell's layout, as the launcher exports it (WORKCELL_LAYOUT)."""
    monkeypatch.setattr(layout, "REGIONS", layout.load(scenarios.layout("workcell")))


# ------------------------------------------------------------------------------------------- A. layout
def test_the_layout_comes_from_the_scenario_geometry():
    r1, r2 = layout.REGIONS["R1"], layout.REGIONS["R2"]
    assert (r1.name, r1.frame, r1.kind) == ("pickup area", "robot", "planar_region")
    assert r1.centre[:2] == workcell.PICKUP["centre"] and r1.size == workcell.PICKUP["size"] and r1.centre[2] == 0.0
    c = workcell.MAT["cell"]
    assert r2.name == "arrangement mat" and r2.centre[:2] == workcell.MAT["centre"]
    assert r2.size == (3 * c + workcell.MAT["border"],) * 2
    cells = [r for r in layout.REGIONS.values() if r.name.startswith("arrangement mat cell")]
    assert [r.id for r in cells] == [f"R{k}" for k in range(3, 12)] and all(r.size == (c, c) for r in cells)
    assert set(layout.REGIONS) == {f"R{k}" for k in range(1, 12)}  # trays are perceived objects, not layout regions


def test_the_launcher_exports_the_layout_and_other_scenarios_have_none():
    import shlex
    from agent import settings
    line = next(x for x in settings.shell().splitlines() if "WORKCELL_LAYOUT=" in x)
    assert shlex.split(line) == ["export", f"WORKCELL_LAYOUT={scenarios.layout(settings.SCENARIO)}"]  # as bash evaluates it
    assert scenarios.layout("laboratory") == "" and layout.load("") == {}


def test_a_region_is_not_a_world_object(tmp_path):
    t = GoalTools([plan(("pick_place", {"object": "a", "relation": "in", "target": "R1"}))])
    t.perceive("table")
    assert all(isinstance(k, int) for k in t.world.objects) and "R1" not in t.world.objects


# ------------------------------------------------------------------------------------ B. Gemma's context
def test_the_master_sees_the_regions_beside_the_objects(tmp_path):
    t = RegionTools([plan(("observe", {}), success=[{"pred": "seen", "args": []}])])
    r, ev = run_supervisor(tmp_path, t, instruction="look")
    ctx = t.plan_contexts[0]
    assert "R1: pickup area" in ctx and "R2: arrangement mat" in ctx and "refer to a region only by its id" in ctx
    planning = next(e for e in ev if e["kind"] == "planning")
    assert planning["regions"][:2] == ["R1 pickup area", "R2 arrangement mat"]


def test_understanding_the_request_gets_no_regions_and_planning_gets_their_names_only(tmp_path):
    """2026-11-04: with the regions before anything was measured, Gemma asked "in the pickup area or on the mat?".
    Understanding reads the user's language only; planning and decisions get each region's id and name, no geometry."""
    from agent.vlm_roles import Clarification
    seen = []
    t = RegionTools([plan(("observe", {}), success=[{"pred": "holding", "args": ["a"]}])])  # unmet: Gemma decides
    t.roles.clarify = lambda request, conversation, context="": (seen.append(context), Clarification(status="clear", request=request))[1]
    t.decisions = [{"action": "stop", "reason": "enough"}]
    run_supervisor(tmp_path, t, instruction="look")
    assert all("R1" not in c and "pickup area" not in c and "arrangement mat" not in c for c in seen)  # (skipped: nothing measured)
    for ctx in (t.plan_contexts[0], t.decide_calls[0]["context"]):
        assert "R1: pickup area\nR2: arrangement mat\n" in ctx
        region_lines = ctx[ctx.index("Fixed regions"):]
        assert not re.search(r"\bmm\b|centre|x [+-]\d|where objects|rubber", region_lines)  # no geometry, no purpose


def test_a_decomposition_gets_the_regions_and_the_objects_by_their_own_ids(tmp_path):
    from agent.capabilities.planning import run_decompose
    from types import SimpleNamespace
    seen = {}

    def decompose(goal, facts, ids, context="", menu="", task_preds=()):
        seen.update(facts=facts, ids=ids)
        return SubPlan.model_validate({"steps": [{"worker": "manipulation", "do": "pick_place", "object": f"#{ids[0]}",
                                                  "relation": "in", "target": "R1"}],
                                       "goal": [{"pred": "in", "args": [f"#{ids[0]}", "R1"]}], "reason": "r"})
    t = GoalTools([])
    obs = t.perceive("table")
    t.roles.decompose, t.describe = decompose, lambda obs, marks=None: None
    ctx = SimpleNamespace(tools=t, obs=obs, log=lambda *a, **k: None, state=None)
    r = run_decompose(ctx, {"goal": "g"})
    assert r.outcome == "completed" and r.data["steps"][0][1]["reference"]["text"] == "R1"
    assert re.search(r"^#\d+: ", seen["facts"], re.M) and "R1: pickup area" in seen["facts"]
    assert all(isinstance(i, int) for i in seen["ids"])


# ------------------------------------------------------------------------------------ C. no language lookup
def test_no_region_name_is_compared_in_agent_code():
    """The deterministic layer accepts ids; which region the words mean is Gemma's output. No production module maps a
    region's name (or any part of it) to an id: the names appear only in the scenario's layout and in Gemma's context."""
    names = {r.name for r in layout.REGIONS.values()} | {r.name.split()[0] for r in layout.REGIONS.values()}
    names |= {" ".join(r.name.split()[:2]) for r in layout.REGIONS.values()}  # "pickup area", "pickup", "arrangement mat", ...
    words = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names))) + r")\b", re.I)
    for f in [*(ROOT / "agent").rglob("*.py"), *(ROOT / "config").glob("*.py")]:
        if f.name == "settings.py":  # the user's settings file: a comment describes the scene to choose
            continue
        src = f.read_text()
        code = "\n".join(line.split("#")[0] for line in src.splitlines() if not line.strip().startswith(("#", '"""')))
        strings = re.findall(r"[\"']([^\"'\n]*)[\"']", code)  # string literals: where a lookup table would live
        hits = [s for s in strings if words.search(s)]
        assert not hits, f"{f.name}: {hits}"


# ---------------------------------------------------------------------------- D. valid region references
def _step(**raw):
    return PlanStep.model_validate({"worker": "manipulation", **raw})


def test_a_region_id_is_a_valid_destination_and_stays_its_canonical_id():
    for t in ("R1", "r1", " R1 "):
        assert _step(do="place", relation="in", target=t).args["reference"]["text"] == "R1"
    assert _step(do="pick_place", object="a", relation="in", target="R2").args["reference"]["text"] == "R2"
    assert _step(do="place", relation="in", target="green tray").args["reference"]["text"] == "green tray"  # words: an object


def test_an_unknown_region_id_is_invalid_with_the_ids_there_are():
    with pytest.raises(ValueError, match=r"R999 is not a region of this work cell \(its regions: R1, R2, R3"):
        _ = _step(do="place", relation="in", target="R999").args
    with pytest.raises(ValueError, match="R1 is a fixed region of the work cell, not an object"):
        _ = _step(do="pick", object="R1").args


def test_the_planner_retries_with_the_region_ids_when_it_names_an_unknown_one():
    bad = ('{"goal": "g", "success": [{"pred": "in", "args": ["the block", "R999"]}], "steps": [{"worker": "manipulation", '
           '"do": "pick_place", "object": "the block", "relation": "in", "target": "R999"}], "reason": "r"}')
    vlm = RecordedVLM([bad, bad.replace("R999", "R1")])
    p = Roles(vlm).plan_goal("put the block in the area", "menu")
    assert p.steps[0].args["reference"]["text"] == "R1" and p.success[0].args == ["the block", "R1"]
    assert "R999 is not a region of this work cell (its regions: R1, R2" in vlm.calls[1]


def test_a_region_cannot_be_the_object_of_a_condition():
    bad = ('{"goal": "g", "success": [{"pred": "holding", "args": ["R1"]}], "steps": [{"worker": "perception", '
           '"do": "observe"}], "reason": "r"}')
    vlm = RecordedVLM([bad, bad.replace('"holding", "args": ["R1"]', '"seen", "args": []')])
    Roles(vlm).plan_goal("x", "menu")
    assert "R1 is a fixed region, not an object" in vlm.calls[1]


def test_a_decomposition_may_name_regions_and_the_retry_lists_both_kinds_of_id():
    ok = ('{"steps": [{"worker": "manipulation", "do": "pick_place", "object": "#9", "relation": "in", "target": "R1"}], '
          '"goal": [{"pred": "in", "args": ["#9", "R1"]}], "reason": "r"}')
    vlm = RecordedVLM([ok.replace('"target": "R1"', '"target": "R77"'), ok])
    sp = Roles(vlm).decompose("g", "#9: block", [9])
    assert sp.steps[0].args["reference"]["text"] == "R1"
    assert "use only these #ids: #9; these region ids: R1, R2" in vlm.calls[1]


# --------------------------------------------------------------------- E, F. placement inside a region
def _tools(objects):
    t = Tools(bridge=None, detector=None, roles=None, executor=None, log=lambda *a, **k: None, run_dir=Path("."))
    return t, [t.world.observe(g, None) for g in objects]


def _inside(t, xy, region, held_radius):
    uv = region.local(xy)
    return all(abs(uv[i]) <= region.size[i] / 2 - held_radius - t.REGION_EDGE_M + 1e-9 for i in (0, 1))


def test_an_empty_region_takes_the_object_at_its_centre():
    r1 = layout.REGIONS["R1"]
    t, (a,) = _tools([geom(0.0, 0.62, 0.04)])  # a block on the mat, to go to the pickup area
    p = t.plan_pick_place(a, "in", None, "robot", np.array(ARM_HOME, float), region="R1")
    assert p.reference_region == "R1" and p.reference_id is None and p.relation == "in"
    assert np.allclose(p.target_centre[:2], r1.centre[:2], atol=1e-9)
    radius = float(np.hypot(p.held.width, p.held.length)) / 2
    assert _inside(t, p.target_centre[:2], r1, radius) and 0.38 <= np.hypot(*p.target_centre[:2]) <= 0.90
    assert abs(p.target_top - p.held.height) < 1e-9  # resting on the table (the region's surface)


def test_an_occupied_centre_is_not_used_blindly():
    r1 = layout.REGIONS["R1"]
    t, (a, occupant) = _tools([geom(0.0, 0.62, 0.04), geom(*r1.centre[:2], 0.05, fp=(0.05, 0.05))])
    p = t.plan_pick_place(a, "in", None, "robot", np.array(ARM_HOME, float), region="R1")
    xy, oc = np.asarray(p.target_centre[:2]), np.asarray(r1.centre[:2])
    assert np.hypot(*(xy - oc)) >= 0.04 / 2 + 0.05 / 2 + t.rel.gap_m - 1e-9  # clear of the object at the centre
    assert _inside(t, xy, r1, float(np.hypot(p.held.width, p.held.length)) / 2)


def test_a_full_region_is_reported_not_forced():
    from control.motion import PlanningError
    cell = layout.REGIONS["R3"]  # one 120 mm mat cell
    t, (a, _) = _tools([geom(0.47, 0.44, 0.04), geom(*cell.centre[:2], 0.05, fp=(0.06, 0.06))])
    with pytest.raises(PlanningError, match="no free spot inside R3") as exc:
        t.plan_pick_place(a, "in", None, "robot", np.array(ARM_HOME, float), region="R3")
    assert t.deterministic(str(exc.value))  # re-observing cannot change it


def test_a_relation_a_region_does_not_compute_is_not_substituted():
    from control.motion import PlanningError
    t, (a,) = _tools([geom(0.0, 0.62, 0.04)])
    with pytest.raises(PlanningError, match="fixed planar region: placing 'next_to' it is not computed"):
        t.plan_pick_place(a, "next_to", None, "robot", np.array(ARM_HOME, float), region="R1")


# ----------------------------------------------------------------------------------- G. verification
def test_in_a_region_is_measured_on_the_whole_footprint():
    r1 = layout.REGIONS["R1"]
    cx, cy = r1.centre[:2]
    edge = cx + r1.size[0] / 2
    t, (inside, outside, overhang, lifted) = _tools([geom(cx, cy, 0.05), geom(0.0, 0.62, 0.05),
                                                     geom(edge - 0.01, cy, 0.05),  # centre inside, 10 mm overhang
                                                     geom(cx, cy - 0.1, 0.09, support=0.04)])  # on top of something
    ev = GoalEvaluator(t, None)
    res = {o: ev.evaluate({"pred": "in", "args": [f"#{o}", "R1"]}, {}) for o in (inside, outside, overhang, lifted)}
    assert res[inside].satisfied is True and "mm inside the outline of R1 (pickup area)" in res[inside].evidence
    assert res[outside].satisfied is False and res[overhang].satisfied is False and res[lifted].satisfied is False
    assert res[inside].confidence == "measured"
    assert ev.evaluate({"pred": "on", "args": [f"#{inside}", "R1"]}, {}).satisfied is None  # not computed, not guessed


# --------------------------------------------------------------------------------- H. canonical identity
class RegionTools(GoalTools):
    """GoalTools whose planner and placement check follow a fixed-region destination (the real in_region geometry)."""
    in_region = Tools.in_region
    REGION_INSIDE_TOL_M, REGION_REST_TOL_M = Tools.REGION_INSIDE_TOL_M, Tools.REGION_REST_TOL_M

    def plan_pick_place(self, oid, rel, rid, frame, q, region=None):
        p = super().plan_pick_place(oid, rel, rid, frame, q)
        self.regions_planned = getattr(self, "regions_planned", []) + [region]
        if region is not None:
            r = layout.region(region)
            p.target_centre, p.target_top = [r.centre[0], r.centre[1], 0.02], 0.04
        p.reference_region = region
        return p

    def verify_placement(self, plan):
        v = super().verify_placement(plan)
        v.update(object_id=plan.object_id, relation=plan.relation, reference_id=plan.reference_id,
                 reference_region=getattr(plan, "reference_region", None))
        return v


def test_the_region_id_stays_canonical_from_plan_to_verification(tmp_path):
    t = RegionTools([plan(("pick_place", {"object": "a", "relation": "in", "target": "R1"}),
                          success=[{"pred": "in", "args": ["a", "R1"]}])])
    r, ev = run_supervisor(tmp_path, t, instruction="put a in the pickup area")
    assert r["status"] == "COMPLETED", r["message"]
    assert t.regions_planned == ["R1"]  # the worker planned into R1, from its calibrated geometry
    step = r["plan"][0]
    assert step["args"]["reference"]["text"] == "R1"  # the plan
    assert any(e["kind"] == "region_reference" and e["region"] == "R1" for e in ev)
    deleg = next(e for e in ev if e["kind"] == "delegate" and e.get("capability") == "pick_place")
    assert "R1" in deleg["object_refs"]  # the worker request
    loc = next(e for e in ev if e["kind"] == "localized")
    assert loc["reference_region"] == "R1" and loc["reference_id"] is None  # no object matched for it
    assert not [e for e in ev if e["kind"] == "match" and e.get("text") == "R1"]  # never resolved as words
    placed = [v for v in r.get("verified", []) if v.get("check") == "placement"] or \
        [e for e in ev if e["kind"] == "verification" and e.get("check") == "placement"]
    rows = next(e for e in ev if e["kind"] == "goal_check")["results"]
    assert [x["satisfied"] for x in rows] == [True] and rows[0]["condition"]["args"] == ["a", "R1"]
    assert "R1" not in r.get("bindings", {})  # an id needs no binding: it is canonical already


def test_a_relation_the_region_cannot_take_moves_nothing(tmp_path):
    t = RegionTools([plan(("pick_place", {"object": "a", "relation": "on", "target": "R1"}),
                          success=[{"pred": "on", "args": ["a", "R1"]}])])
    t.decisions = [{"action": "stop", "reason": "cannot"}]
    r, ev = run_supervisor(tmp_path, t, instruction="put a on the area")
    assert t.executed == [] and r["status"] != "COMPLETED"
    assert r["plan"][0]["outcome"] == "unsupported"
    consult = next(e for e in ev if e["kind"] == "consult")  # the master (Gemma) gets the reason and decides
    assert "fixed planar region: placing 'on' it is not computed" in consult["reason"]


# ------------------------------------------------------------------------- I. object placement unchanged
def test_placing_on_a_perceived_object_is_unchanged(tmp_path):
    t = RegionTools([plan(("pick_place", {"object": "a", "relation": "on", "target": "b"}))])
    r, ev = run_supervisor(tmp_path, t, instruction="put a on b")
    assert r["status"] == "COMPLETED" and t.regions_planned == [None]
    loc = next(e for e in ev if e["kind"] == "localized")
    assert loc.get("reference_region") is None and loc["reference_id"] is not None
    assert not any(e["kind"] == "region_reference" for e in ev)
