"""Supervisor-orchestrated architecture: worker registry and contracts, goal conditions over measured state,
compositional manipulation across instructions, decomposition, uncertainty-driven inspection, safety refusals.
Mocked tools (no sim, no GPU)."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langgraph")

from agent import capabilities as caps  # noqa: E402
from agent.goals import GoalEvaluator, derive_conditions  # noqa: E402
from agent.vlm_roles import RecordedVLM, Roles, SceneAnswer, SubPlan  # noqa: E402
from agent.workers import WorkerRegistry  # noqa: E402
from agent.world_state import TaskState  # noqa: E402
from tests.test_supervisor_goal import GoalTools, plan, run  # noqa: E402

PICK_A = ("pick", {"object": "a"})


def session(plans, **kw):
    t = GoalTools(plans, **kw)
    t.bridge = SimpleNamespace(q=t.bridge.q, finger=lambda: 0.0)
    t.OPEN_FINGER_DEG = 2.0
    return t


def run_in(tmp_path, name, t, **kw):
    d = tmp_path / name
    d.mkdir()
    return run(d, t, **kw)


# ------------------------------------------------------------------------------------------- registry
def test_registry_routes_capabilities_to_workers_and_wraps_results():
    log = []
    reg = WorkerRegistry(lambda kind, **kw: log.append((kind, kw)),
                         {"locate": ("perception", lambda c, a, r: caps.StepResult("completed", "found", {"bindings": {"x": 4}})),
                          "pick": ("manipulation", lambda c, a, r: 1 / 0)}, caps.SPECS)
    assert reg.worker_of("pick") == "manipulation" and reg.snapshot()["manipulation"]["can_move_robot"]
    assert not reg.snapshot()["perception"]["can_move_robot"]
    ok = reg.dispatch(reg.request("g", "locate", {"target": "x"}), SimpleNamespace())
    assert (ok.task_id, ok.status, ok.data["bindings"]) == ("t1", "completed", {"x": 4})
    bad = reg.dispatch(reg.request("g", "pick", {}), SimpleNamespace())  # a crash is a failure, never a success
    assert bad.status == "failed" and "ZeroDivisionError" in bad.failure_reason
    assert reg.snapshot()["manipulation"]["status"] == "failed"
    assert [k for k, _ in log] == ["delegate", "worker_result"] * 2


# ---------------------------------------------------------------------------------------------- goals
def test_derived_conditions_describe_the_final_state_not_the_steps():
    pp = {"object": {"text": "a"}, "relation": "next_to", "reference": {"text": "b"}}
    assert derive_conditions([("pick", {"object": {"text": "a"}})]) == [{"pred": "holding", "args": ["a"]}]
    # picked and later placed: holding(a) is not part of the final state
    conds = derive_conditions([("pick", {"object": {"text": "a"}}), ("place", {**pp, "object": {"text": "a"}})])
    assert conds == [{"pred": "next_to", "args": ["a", "b"]}, {"pred": "not_holding", "args": []}]


def test_unknown_predicates_and_unbound_objects_are_never_true():
    t = session([])
    ev = GoalEvaluator(t, TaskState("x"))
    assert ev.evaluate({"pred": "sliced", "args": ["banana"]}, {}).satisfied is None
    assert ev.evaluate({"pred": "on", "args": ["a", "b"]}, {}).satisfied is None
    t.perceive("table")
    r = ev.evaluate({"pred": "on", "args": ["#1", "#2"]}, {})  # "#id" binds directly; 1 is beside 2, not on it
    assert r.satisfied is False and r.confidence == "measured" and r.bound == {"#1": 1, "#2": 2}


# ------------------------------------------------------------------------- brown-block regression (CP1)
def test_planner_rejects_a_placeholder_destination_and_uses_pick():
    """Reproduced failure: 'pick the dark brown block' -> pick_place with target 'null'."""
    bad = json.dumps({"goal": "g", "steps": [{"do": "pick_place", "object": "dark brown block", "relation": "on", "target": "null"}]})
    good = json.dumps({"goal": "g", "steps": [{"do": "pick", "object": "dark brown block"}]})
    vlm = RecordedVLM([bad, good])
    p = Roles(vlm).plan_goal("Pick up the dark brown block.", "menu")
    assert [s.capability for s in p.steps] == ["pick"] and "pick_place" in vlm.calls[1]
    assert p.steps[0].args == {"object": {"text": "dark brown block", "relation": None, "anchor": None}}


def test_pick_only_ends_holding_without_placement_or_home(tmp_path):
    t = session([plan(PICK_A)])
    r, ev = run_in(tmp_path, "1", t, instruction="Pick up a.")
    assert r["status"] == "COMPLETED" and r["holding"] == 1 and t.home_calls == 0
    assert "placement" not in [v["check"] for v in r["verifications"]]
    assert r["goal_check"] == [{"condition": {"pred": "holding", "args": ["a"], "source": "derived"}, "satisfied": True,
                                "confidence": "measured", "evidence": r["goal_check"][0]["evidence"], "bound": {"a": 1}}]
    assert [e["worker"] for e in ev if e["kind"] == "delegate"] == ["manipulation"]


def test_follow_up_pronoun_binds_to_the_held_object_and_places_it(tmp_path):
    t = session([plan(PICK_A), plan(("place", {"object": "it", "relation": "next_to", "target": "c"}))])
    run_in(tmp_path, "1", t, instruction="Pick up a.")
    r, ev = run_in(tmp_path, "2", t, instruction="Put it beside c.")
    assert r["status"] == "COMPLETED" and r["holding"] is None and t.home_calls == 1
    assert [(e["text"], e["world_id"]) for e in ev if e["kind"] == "binding"] == [("it", 1)]
    assert len(t.planned) == 2  # one grasp plan, one place-held plan: the object was not regrasped
    assert {c["pred"] for c in r["conditions"]} == {"next_to", "not_holding"}
    assert all(g["satisfied"] for g in r["goal_check"])


def test_placing_far_from_the_reference_is_not_success(tmp_path):
    """Negative: the held object ends 680 mm from c (wrong spot): measured goal check says no."""
    t = session([plan(PICK_A), plan(("place", {"object": "it", "relation": "next_to", "target": "c"}))], misplace=True)
    run_in(tmp_path, "1", t, instruction="Pick up a.")
    r, _ = run_in(tmp_path, "2", t, instruction="Put it beside c.")
    assert r["status"] == "FAILED" and "goal is not met" in r["message"] and "did not report success" in r["message"]
    nt = next(g for g in r["goal_check"] if g["condition"]["pred"] == "next_to")
    assert nt["satisfied"] is False and nt["confidence"] == "measured" and "680 mm apart" in nt["evidence"]


def test_pick_place_of_the_held_object_becomes_place(tmp_path):
    t = session([plan(PICK_A), plan(("pick_place", {"object": "it", "relation": "on", "target": "b"}))])
    run_in(tmp_path, "1", t)
    r, ev = run_in(tmp_path, "2", t)
    assert r["status"] == "COMPLETED" and r["plan"][-1]["capability"] == "place"
    assert any(e["kind"] == "plan_adjusted" for e in ev)
    assert [g["condition"]["pred"] for g in r["goal_check"]] == ["on", "not_holding"] and all(g["satisfied"] for g in r["goal_check"])


# ------------------------------------------------------------------------------------- safety refusals
def test_picking_while_holding_another_object_is_refused_and_not_replanned(tmp_path):
    t = session([plan(PICK_A), plan(("pick", {"object": "c"})), plan(("pick", {"object": "c"}))])
    run_in(tmp_path, "1", t)
    n_exec = len(t.executed)
    r, _ = run_in(tmp_path, "2", t)
    assert r["status"] == "FAILED" and r["outcome"] == "unsafe" and r["replans"] == 0 and len(t.executed) == n_exec


def test_place_with_an_empty_gripper_is_unsupported(tmp_path):
    t = session([plan(("place", {"object": "a", "relation": "on", "target": "b"}))])
    r, _ = run_in(tmp_path, "1", t)
    assert r["status"] == "UNSUPPORTED" and t.executed == []


def test_pronoun_with_nothing_to_refer_to_asks_for_the_object(tmp_path):
    t = session([plan(("place", {"object": "it", "relation": "on", "target": "b"}))])
    r, _ = run_in(tmp_path, "1", t)
    assert r["status"] == "CANCELLED" and r["outcome"] == "insufficient_info" and t.executed == []


def test_failure_classes_map_to_replannable_or_final_outcomes():
    from agent.capabilities import _failure_outcome as f
    final = {"infeasible", "unsafe", "cancelled", "unsupported"}
    assert f("infeasible", "carry above target: out of reach") == "infeasible"
    assert f("planning", "lift: collision, held with tube_3") == "failed"  # may change with new observations
    assert f("home", "HOME not reached: max joint error 2.1 deg") == "unsafe"  # no new motion from an unverified pose
    assert f("execution", "x: safety layer rejected the trajectory: joint limit") == "unsafe"
    assert f("execution", "x: did not reach the goal: 3.00 deg off (controller timeout)") == "failed"
    assert f("asked", "no answer within 120 s") == "cancelled"
    assert f("perception", "nothing matches 'zz'") == "insufficient_info"
    assert {f(c, "r") for c in ("grasp", "placement", "understanding", "perception", None)} == {"failed"}
    assert not {f(c, "r") for c in ("planning", "grasp", "placement")} & final


def test_nondeterministic_planning_failure_is_still_replanned(tmp_path, monkeypatch):
    """Useful replanning is kept: a collision-type planning error (not deterministic) gets one replan."""
    from control.motion import PlanningError
    t = session([plan(PICK_A), plan(PICK_A)])

    def collides(*a, **k):
        raise PlanningError("descend to grasp: collision, pads with tube_3 (0.4 mm)")
    monkeypatch.setattr(t, "plan_pick", collides)
    r, ev = run_in(tmp_path, "1", t)
    assert r["status"] == "FAILED" and r["replans"] == 1 and len(t.plan_calls) == 2 and t.executed == []
    assert [e["accepted"] for e in ev if e["kind"] == "replan"] == [False]  # same step, unchanged situation


def test_deterministic_infeasibility_is_not_replanned(tmp_path, monkeypatch):
    from control.motion import PlanningError
    t = session([plan(PICK_A), plan(PICK_A)])

    def unreachable(*a, **k):
        raise PlanningError("grasp approach: [302, 148, 206] mm is out of reach")
    monkeypatch.setattr(t, "plan_pick", unreachable)
    r, _ = run_in(tmp_path, "1", t)
    assert r["status"] == "FAILED" and r["outcome"] == "infeasible" and len(t.plan_calls) == 1 and t.executed == []
    assert r["holding"] is None


# ------------------------------------------------------------------------------------- decomposition
def _decomposer(t, subplan):
    calls = []

    def decompose(goal, facts, ids, context=""):
        calls.append((goal, facts, ids))
        return subplan
    t.roles.decompose = decompose
    return calls


def test_decomposition_inserts_steps_and_its_conditions_decide_success(tmp_path):
    t = session([plan(("observe", {}), ("decompose", {"goal": "stack a on b"}))])
    sp = SubPlan.model_validate({"steps": [{"do": "pick_place", "object": "#1", "relation": "on", "target": "#2"}],
                                 "goal": [{"pred": "on", "args": ["#1", "#2"]}], "reason": "one move"})
    calls = _decomposer(t, sp)
    r, ev = run_in(tmp_path, "1", t, instruction="stack a on b")
    assert r["status"] == "COMPLETED" and calls and "#1:" in calls[0][1]
    assert [s["capability"] for s in r["plan"]] == ["observe", "decompose", "pick_place"]
    assert r["plan"][2]["origin"] == "decomposition" and r["conditions"][0]["source"] == "explicit"
    assert r["goal_check"][0]["satisfied"] is True and t.home_calls == 1


def test_decomposition_whose_final_state_is_not_reached_is_not_success(tmp_path):
    t = session([plan(("observe", {}), ("decompose", {"goal": "g"}))])
    _decomposer(t, SubPlan.model_validate({"steps": [{"do": "pick_place", "object": "#1", "relation": "on", "target": "#2"}],
                                           "goal": [{"pred": "on", "args": ["#3", "#2"]}], "reason": "wrong object"}))
    r, _ = run_in(tmp_path, "1", t)
    assert r["status"] == "FAILED" and "goal is not met" in r["message"] and "did not report success" in r["message"]


# ------------------------------------------------------------------------------- uncertainty-driven inspection
def test_uncertain_answer_triggers_a_closer_look_then_answers_again(tmp_path, monkeypatch):
    inspected = []
    monkeypatch.setattr(caps, "run_inspect", lambda ctx, args: (inspected.append(args["target"]),
                                                                caps.StepResult("completed", "close look"))[1])
    answers = [SceneAnswer(answer="object 1?", object_ids=[1], certain=False, check_closer=[1, 3]),
               SceneAnswer(answer="object 3", object_ids=[3])]
    t = session([plan(("observe", {}), ("answer", {"question": "which small one is round?"}))])
    t.roles.answer = lambda q, f, ids: answers.pop(0)
    r, ev = run_in(tmp_path, "1", t)
    assert r["status"] == "COMPLETED" and inspected == ["#%d" % t.ids[1], "#%d" % t.ids[3]]
    assert [s["capability"] for s in r["plan"]] == ["observe", "answer", "inspect", "inspect", "answer"]
    assert r["plan"][1]["status"] == "replaced" and "object 3" in r["message"] and "not independently verified" in r["message"]


def test_gemma_calls_and_tokens_are_counted():
    vlm = RecordedVLM(['{"goal": "g", "steps": [{"do": "observe"}]}'])
    vlm.last_tokens = {"prompt": 10, "answer": 3, "images": 0}
    roles = Roles(vlm)
    roles.plan_goal("look", "menu")
    assert roles.calls == 1 and roles.tokens == {"prompt": 10, "answer": 3, "images": 0}
