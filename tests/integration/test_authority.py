"""Authorization boundary (agent/authority.py + WorkerRegistry.dispatch), location constraints on targets, and goal
verification of questions. Mocked execution only (no sim, no robot)."""


import numpy as np
import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langgraph")

from agent import capabilities as caps  # noqa: E402
from agent.authority import classify, from_classification  # noqa: E402
from agent.goals import GoalEvaluator  # noqa: E402
from agent.vlm_roles import RecordedVLM, Roles, SceneAnswer, SubPlan, VLMFailure  # noqa: E402
from agent.task_state import TaskState  # noqa: E402
from tests.support.fakes import GoalTools, geom  # noqa: E402
from tests.support.runners import plan, run_supervisor as run  # noqa: E402

QUESTION = "Which objects are tubes and which are vials?"


def question_tools(plans, **kw):
    t = GoalTools(plans, **kw)
    t.request_kind = "question"
    return t


# ----------------------------------------------------------------------------------- authority decision
def test_object_change_needs_an_action_with_verbatim_evidence():
    assert from_classification("Put the a on the b.", "action", "Put the a on the b").allows("object_change")
    assert not from_classification(QUESTION, "question", QUESTION).allows("object_change")
    forged = from_classification(QUESTION, "action", "pick up the tube and place it on the vial")  # words not in the request
    assert not forged.allows("object_change") and "verbatim" in forged.note
    for r in (from_classification(QUESTION, "question", ""), from_classification("x", "description", "")):
        assert r.allows("none") and r.allows("sensing")  # observation and camera inspection stay available


def test_classifier_failure_means_least_privilege():
    class Broken:
        def classify_request(self, text):
            raise VLMFailure("request: invalid answer after one retry")
    assert not classify("Put the a on the b.", Broken()).allows("object_change")
    assert not classify("Put the a on the b.", object()).allows("object_change")  # no classifier at all


def test_request_classifier_sees_only_the_request_and_is_validated():
    vlm = RecordedVLM(['{"kind": "action", "evidence": ""}', '{"kind": "action", "evidence": "put the a on the b"}'])
    r = Roles(vlm).classify_request("Please put the a on the b.")
    assert r.kind == "action" and "evidence" in vlm.calls[1] and "Request: Please put the a on the b." in vlm.calls[0]


# --------------------------------------------------------------- information requests cannot manipulate
def test_question_decomposed_into_pick_and_place_moves_nothing(tmp_path):
    """The live failure (current-system test #4): a tube/vial question decomposed into pick #17 + place on #8."""
    t = question_tools([plan(("observe", {}), ("decompose", {"goal": "identify which objects are tubes and which are vials"}))])
    t.roles.decompose = lambda goal, facts, ids, context="", menu="": SubPlan.model_validate(
        {"steps": [{"do": "pick", "object": "#1"}, {"do": "place", "relation": "on", "target": "#2"}],
         "goal": [{"pred": "on", "args": ["#1", "#2"]}], "reason": "group them"})
    t.decisions = [{"action": "replan", "steps": [{"do": "answer", "question": QUESTION}]}]  # the question is unanswered
    r, ev = run(tmp_path, t, instruction=QUESTION)
    assert t.executed == [] and t.parks == 0  # no grasp, no carry, not even the parking move for them
    assert {x["capability"] for x in r["refused"]} == {"pick", "place"}
    assert any(e["kind"] == "authorization" and e.get("allowed") is False for e in ev)
    assert [s["capability"] for s in r["plan"]] == ["observe", "decompose", "answer"]  # the question is still answered
    assert r["status"] == "COMPLETED" and "did not pick, place" in r["message"]
    assert [c["pred"] for c in r["conditions"]] == ["answered", "seen"]  # the decomposition's on(#1, #2) was not adopted


def test_planner_naming_a_manipulation_for_a_question_is_refused_before_execution(tmp_path):
    t = question_tools([plan(("pick_place", {"object": "a", "relation": "on", "target": "b"}), ("answer", {"question": QUESTION}))])
    t.decisions = [{"action": "continue", "reason": "answer without moving anything"}]  # Gemma is told about the refusal
    r, ev = run(tmp_path, t, instruction=QUESTION)
    assert t.executed == [] and r["plan"][0]["status"] == "refused" and r["plan"][1]["status"] == "done"
    blocked = [e for e in ev if e["kind"] == "worker_result" and e.get("outcome") == "unauthorized"]
    assert len(blocked) == 1 and not [e for e in ev if e["kind"] == "skill"]  # the skill never started


def test_explicit_manipulation_request_keeps_its_capabilities(tmp_path):
    t = GoalTools([plan(("pick_place", {"object": "a", "relation": "on", "target": "b"}))])  # request_kind "action"
    r, _ = run(tmp_path, t, instruction="Put the a on the b.")
    assert r["status"] == "COMPLETED" and r["refused"] == [] and "approach" in t.executed and t.home_calls == 1
    assert list(r["authority"]["grants"]) == ["none", "sensing", "object_change"]


def test_camera_inspection_stays_available_for_information_requests(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(caps, "run_inspect", lambda ctx, args: (seen.append(args["target"]),
                                                                caps.StepResult("completed", "close look"))[1])
    t = question_tools([plan(("observe", {}), ("inspect", {"object": "a"}), ("answer", {"question": "what is a?"}))])
    t.request_kind = "description"
    r, _ = run(tmp_path, t, instruction="Look closely at a and tell me what it is.")
    assert seen == ["a"] and r["refused"] == []  # inspection authorized and run (the stub adds no wrist observation)
    assert not any(g.get("outcome") == "unauthorized" for g in r["plan"])


# --------------------------------------------------------------------------- location constraints
def test_in_relation_is_measured_containment():
    from agent.tools import Tools
    t = Tools(bridge=None, detector=None, roles=None, executor=None, log=lambda *a, **k: None, run_dir=".")
    tray = geom(0.64, -0.10, 0.018, fp=(0.30, 0.19))
    tray["extents"], tray["yaw_deg"] = [0.30, 0.19], 0.0
    vial_in = geom(0.73, -0.10, 0.064, support=0.004)
    shelf_bottle = geom(-0.32, 1.29, 0.85, support=0.70)
    assert t.relation_holds(vial_in, tray, "in")
    assert not t.relation_holds(shelf_bottle, tray, "in")


def test_wrong_location_target_is_rejected_even_if_it_looks_right(tmp_path):
    """The live failure (#3): a white cylinder elsewhere passed resolve AND confirm. With the location constraint it
    is measured as not being in the anchor, and the crop check is not even asked."""
    step = ("inspect", {"object": "a", "object_relation": "in", "object_anchor": "b"})
    t = GoalTools([plan(step), plan(step)])  # the replan repeats it: rejected, unchanged situation
    r, ev = run(tmp_path, t, instruction="Inspect the a in the b.")
    assert r["status"] == "FAILED" and "is not in 'b'" in r["message"] and t.confirmed == [] and t.executed == []
    chk = [e for e in ev if e["kind"] == "location_check"]
    assert len(chk) == 1 and chk[0]["holds"] is False


# ------------------------------------------------------------------------------ question goals
def test_a_scene_report_does_not_answer_a_question():
    st = TaskState(QUESTION)
    st.facts = {"objects": [{"mark": 1}]}  # a report exists
    r = GoalEvaluator(None, st).evaluate({"pred": "answered", "args": [QUESTION]}, {})
    assert r.satisfied is False and "scene report is not an answer" in r.evidence


def test_an_uncertain_answer_does_not_complete_a_question(tmp_path):
    t = question_tools([plan(("observe", {}), ("describe_scene", {}))],
                       answer=SceneAnswer(answer="probably 1 is a tube", object_ids=[1], certain=False))
    t.decisions = [{"action": "replan", "steps": [{"do": "answer", "question": QUESTION}]}]  # a report is not an answer
    r, _ = run(tmp_path, t, instruction=QUESTION)
    assert [s["capability"] for s in r["plan"]] == ["observe", "describe_scene", "answer"]
    assert r["status"] == "FAILED" and "not certain" in r["message"] and "did not report success" in r["message"]


def test_an_unanswerable_question_is_never_completed(tmp_path):
    t = question_tools([plan(("observe", {}), ("describe_scene", {}))])
    t.roles.answer = lambda q, f, ids: (_ for _ in ()).throw(VLMFailure("answer: invalid answer after one retry"))
    t.plans.append(plan(("observe", {}), ("describe_scene", {})))  # the replan offers only a report again
    r, _ = run(tmp_path, t, instruction=QUESTION)
    assert r["status"] != "COMPLETED"


def test_a_certain_answer_completes_a_question(tmp_path):
    t = question_tools([plan(("observe", {}), ("answer", {"question": QUESTION}))])
    r, _ = run(tmp_path, t, instruction=QUESTION)
    assert r["status"] == "COMPLETED" and r["goal_check"][0]["condition"]["pred"] == "answered"
    assert np.all([g["satisfied"] for g in r["goal_check"]])
