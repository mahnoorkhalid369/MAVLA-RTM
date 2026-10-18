"""The closed, Gemma-driven loop (agent/supervisor.py, agent/workers.py): structured delegation contracts, sub-goal
verification by the verification worker, Gemma deciding the next step (continue / replan / ask_user / finish / stop)
whenever a result needs a decision, the deterministic guards around it (budget, repeated step, authorization, safety),
shared task state and the handoff trace. Deterministic mocked workers and scripted Gemma decisions (tests only: the
live workflow uses the real workers and Gemma); no simulator, no GPU, no motion, no scenario knowledge."""

from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langgraph")

import json  # noqa: E402

from agent.ask import ScriptedResponder  # noqa: E402
from agent.controls import ExecutionControls  # noqa: E402
from agent.events import format_event, format_trace  # noqa: E402
from agent.vlm_roles import RecordedVLM, Roles, SceneAnswer, VLMFailure  # noqa: E402
from tests.support.fakes import GoalTools  # noqa: E402
from tests.support.runners import plan, run_supervisor as run  # noqa: E402


def run_with(tmp_path, t, controls, instruction="do it"):
    """run_supervisor with explicit ExecutionControls (the Gemma decision budget)."""
    from agent.eventlog import EventLog
    from agent.supervisor import GoalSupervisor
    log = EventLog(tmp_path / "events.jsonl")
    r = GoalSupervisor(t, log, tmp_path, ScriptedResponder(["cancel"]), controls=controls).run(instruction)
    log.close()
    return r, [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]

PICK_A = ("pick", {"object": "a"})
PLACE_IT = ("place", {"object": "it", "relation": "next_to", "target": "c"})


def tools(plans, **kw):
    t = GoalTools(plans, **kw)
    t.bridge = SimpleNamespace(q=t.bridge.q, finger=lambda: 0.0)
    t.OPEN_FINGER_DEG = 2.0
    return t


def of(ev, kind):
    return [e for e in ev if e["kind"] == kind]


# ------------------------------------------------------------------------------------------- contracts
def test_every_delegation_and_result_carries_the_structured_contract(tmp_path):
    t = tools([plan(PICK_A)])
    r, ev = run(tmp_path, t, instruction="Pick up a.")
    assert r["status"] == "COMPLETED"
    for d in of(ev, "delegate"):
        assert d["sender"] == "supervisor" and d["receiver"] == d["worker"] and d["goal"] == "Pick up a."
        assert d["objective"] and isinstance(d["context"], dict) and "held_object" in d["context"]
        assert d["constraints"]["allowed_effects"] and "effect" in d["constraints"] and isinstance(d["expected"], list)
    pick = next(d for d in of(ev, "delegate") if d["capability"] == "pick")
    assert pick["expected"] == [{"pred": "holding", "args": ["a"]}] and pick["constraints"]["effect"] == "object_change"
    for res in of(ev, "worker_result"):
        assert res["sender"] == res["worker"] and res["receiver"] == "supervisor"
        assert {"status", "outcome", "summary", "evidence", "uncertainty", "needs", "tools", "failure_reason"} <= set(res)


def test_workers_report_and_only_the_supervisor_concludes(tmp_path):
    t = tools([plan(PICK_A)])
    r, ev = run(tmp_path, t, instruction="Pick up a.")
    assert [e["worker"] for e in of(ev, "delegate")] == ["manipulation", "verification", "verification"]
    assert all(res["status"] in ("completed", "failed", "blocked", "insufficient_info", "cancelled") for res in of(ev, "worker_result"))
    assert [e["status"] for e in of(ev, "report")] == ["COMPLETED"]  # the supervisor's own conclusion, once
    assert r["subgoal_checks"][0]["outcome"] == "completed" and r["goal_check"][0]["satisfied"] is True


# ------------------------------------------------------------------------------------- feedback loop
def test_routine_success_uses_one_gemma_plan_and_no_extra_reasoning(tmp_path):
    t = tools([plan(PICK_A)])
    r, ev = run(tmp_path, t, instruction="Pick up a.")
    assert r["status"] == "COMPLETED" and len(t.plan_calls) == 1 and not of(ev, "consult") and r["consults"] == []
    assert t.decide_calls == []
    assert [e["decision"] for e in of(ev, "review")] == ["continue"]


def test_a_failed_subgoal_returns_to_gemma_with_the_evidence_and_progress(tmp_path):
    """The worker says done; the verification worker measures the opposite; Gemma decides again from that."""
    t = tools([plan(PICK_A, PLACE_IT), plan(PICK_A, PLACE_IT)], misplace=True)
    plan_goal = t.roles.plan_goal

    def planner(text, menu, context=""):  # after the first plan, placements land where they are asked to
        p = plan_goal(text, menu, context)
        t.misplace = len(t.plan_calls) < 2
        return p
    t.roles.plan_goal = planner
    r, ev = run(tmp_path, t, instruction="Put a beside c.")
    first_place = next(s for s in r["plan"] if s["capability"] == "place")
    assert first_place["status"] == "replaced" and first_place["outcome"] == "failed"
    assert "sub-goal is not met" in r["subgoal_checks"][1]["summary"] or r["subgoal_checks"][1]["outcome"] == "failed"
    first = t.decide_calls[0]  # what Gemma decided from: the measured evidence and the progress
    assert "sub-goal is not met" in first["event"] and "centres 680 mm apart" in first["event"]
    assert "Progress so far" in first["progress"] and "sub-goal is not met" in first["progress"]
    assert r["consults"][0]["action"] == "replan" and r["consults"][0]["accepted"]
    assert r["status"] == "COMPLETED" and any(c["condition"]["pred"] == "next_to" and c["satisfied"] for c in r["goal_check"])
    assert of(ev, "consult") and any("re-planned" in e["reason"] for e in of(ev, "plan_update"))


def test_a_goal_not_shown_after_every_step_asks_gemma_for_the_remaining_steps(tmp_path):
    answers = [SceneAnswer(answer="maybe two", object_ids=[1], certain=False),
               SceneAnswer(answer="two", object_ids=[1, 2])]
    t = tools([plan(("observe", {}), ("answer", {"question": "how many?"})),
               plan(("answer", {"question": "how many objects are there?"}))])
    t.request_kind = "question"
    t.roles.answer = lambda q, f, ids: answers.pop(0)
    r, ev = run(tmp_path, t, instruction="How many objects are there?")
    # the uncertain answer goes to Gemma first (it re-plans the answer); the goal check then passes
    assert len(t.plan_calls) == 2 and "not certain" in t.decide_calls[0]["event"]
    assert "Progress so far" in t.decide_calls[0]["progress"]
    assert [c["action"] for c in r["consults"]] == ["replan"] and r["consults"][0]["steps"] == ["answer"]
    assert r["status"] == "COMPLETED" and r["replans"] == 1 and "not independently verified" in r["message"]
    assert any(e["decision"] == "ask Gemma" for e in of(ev, "review"))


def test_the_loop_stops_when_the_budget_is_used(tmp_path):
    answers = [SceneAnswer(answer="maybe", object_ids=[1], certain=False)] * 3
    t = tools([plan(("observe", {}), ("answer", {"question": "q?"}))])
    t.request_kind = "question"
    t.roles.answer = lambda q, f, ids: answers.pop(0)
    t.decisions = [{"action": "replan", "steps": [{"do": "answer", "question": "q again?"}]}] * 3
    r, _ = run_with(tmp_path, t, ExecutionControls(max_replans=1), instruction="Which one?")
    assert r["status"] == "FAILED" and r["replans"] == 1 and len(t.decide_calls) == 1  # one decision, then stop
    assert "budget 1 used" in r["message"] and "did not report success" in r["message"]


def test_a_zero_budget_never_asks_gemma_again(tmp_path):
    t = tools([plan(PICK_A)], placement=(False,), contact=(False,))
    r, _ = run_with(tmp_path, t, ExecutionControls(max_replans=0), instruction="Pick up a.")
    assert r["status"] == "FAILED" and t.decide_calls == [] and r["replans"] == 0


def test_unsupported_goals_are_reported_as_such(tmp_path):
    t = tools([plan(reason="no capability can do that")])
    r, _ = run(tmp_path, t, instruction="Sing a song.")
    assert r["status"] == "UNSUPPORTED" and "no capability can do that" in r["message"] and t.executed == []


# ------------------------------------------------------------------------------- Gemma's decisions, one by one
UNCERTAIN = SceneAnswer(answer="maybe #1", object_ids=[1], certain=False)


def uncertain_then(*steps):
    """A plan whose first answer is uncertain (so Gemma must decide), followed by the given steps."""
    t = tools([plan(("observe", {}), ("answer", {"question": "q?"}), *steps)])
    t.request_kind = "question"
    answers = [UNCERTAIN, SceneAnswer(answer="#1", object_ids=[1])]
    t.roles.answer = lambda q, f, ids: answers.pop(0)
    return t


def test_gemma_continue_keeps_the_remaining_plan(tmp_path):
    t = uncertain_then(("answer", {"question": "q, once more?"}))
    t.decisions = [{"action": "continue", "reason": "the next step settles it"}]
    r, ev = run(tmp_path, t, instruction="Which one?")
    assert [s["capability"] for s in r["plan"]] == ["observe", "answer", "answer"] and r["status"] == "COMPLETED"
    assert [e["action"] for e in of(ev, "gemma_decision")] == ["continue"] and len(t.plan_calls) == 1


def test_gemma_can_ask_the_user_and_decides_again_from_the_answer(tmp_path):
    t = uncertain_then()
    t.decisions = [{"action": "ask_user", "question": "Do you mean the left one?"},
                   {"action": "replan", "steps": [{"do": "answer", "question": "q, the left one?"}]}]
    r, ev = run(tmp_path, t, ScriptedResponder(["yes, the left one"]), instruction="Which one?")
    assert r["questions"] == [{"question": "Do you mean the left one?", "answer": "yes, the left one", "asked_by": "gemma"}]
    assert "'yes, the left one'" in t.decide_calls[1]["event"] and r["status"] == "COMPLETED"
    line = format_event({"wall": "t", **of(ev, "gemma_decision")[0]})
    assert line.component == "GEMMA" and "ASK_USER" in line.text


def test_no_answer_to_gemmas_question_cancels(tmp_path):
    t = uncertain_then()
    t.decisions = [{"action": "ask_user", "question": "Which one do you mean?"}]
    r, _ = run(tmp_path, t, ScriptedResponder(["cancel"]), instruction="Which one?")
    assert r["status"] == "CANCELLED" and "No answer" in r["message"] and len(t.decide_calls) == 1


def test_gemma_finish_skips_the_rest_and_the_goal_is_still_measured(tmp_path):
    t = uncertain_then(("return_home", {}))
    t.decisions = [{"action": "finish", "reason": "answered well enough"}]
    r, _ = run(tmp_path, t, instruction="Which one?")
    assert r["plan"][-1]["status"] == "skipped" and r["goal_check"]  # finish = go to the measured goal check
    assert r["status"] != "COMPLETED"  # the only answer was uncertain: Gemma's "finish" does not make it a success


def test_gemma_stop_ends_with_its_reason(tmp_path):
    t = tools([plan(PICK_A)], contact=(False,))
    t.decisions = [{"action": "stop", "reason": "the object cannot be grasped"}]
    r, _ = run(tmp_path, t, instruction="Pick up a.")
    assert r["status"] == "FAILED" and "Gemma stopped: the object cannot be grasped" in r["message"]


def test_an_invalid_gemma_decision_is_a_clear_failure(tmp_path):
    t = tools([plan(PICK_A)], contact=(False,))
    t.decisions = [VLMFailure("decide: invalid answer after one retry")]
    r, _ = run(tmp_path, t, instruction="Pick up a.")
    assert r["status"] == "FAILED" and "Gemma could not decide" in r["message"]
    assert r["consults"][0]["accepted"] is False


def test_continue_without_remaining_steps_is_rejected_by_validation(tmp_path):
    t = tools([plan(PICK_A)], contact=(False,))
    t.decisions = [{"action": "continue"}]  # nothing is pending: schema-level check refuses it
    r, _ = run(tmp_path, t, instruction="Pick up a.")
    assert r["status"] == "FAILED" and "no steps remain" in r["message"]


def test_safety_and_authorization_stops_do_not_ask_gemma(tmp_path):
    t = tools([plan(("pick_place", {"object": "a", "relation": "on", "target": "b"}))], exec_fail="approach")
    r, _ = run(tmp_path, t, instruction="Put a on b.")
    assert r["status"] == "FAILED" and r["outcome"] == "unsafe" and t.decide_calls == []
    t = tools([plan(PICK_A, ("observe", {}))])
    t.request_kind = "description"  # a description request does not authorize picking
    (tmp_path / "q").mkdir()
    r, _ = run(tmp_path / "q", t, instruction="Describe a.")
    assert r["plan"][0]["status"] == "refused" and t.executed == [] and t.decide_calls == []


def test_roles_decide_validates_and_retries_with_the_error():
    vlm = RecordedVLM([json.dumps({"action": "replan", "steps": []}),
                       json.dumps({"action": "replan", "steps": [{"do": "locate", "object": "a"}], "reason": "r"})])
    d = Roles(vlm).decide("find a", "Progress so far:", "locate failed", "menu", has_pending=False)
    assert d.action == "replan" and d.steps[0].capability == "locate" and "replan needs the remaining steps" in vlm.calls[1]
    with pytest.raises(VLMFailure):
        Roles(RecordedVLM([json.dumps({"action": "dance"})] * 2)).decide("g", "", "e", "menu", has_pending=True)


# ---------------------------------------------------------------------------------------- shared state
def test_the_ledger_records_robot_state_sensors_capabilities_and_uncertainty(tmp_path):
    answers = [SceneAnswer(answer="maybe", object_ids=[1], certain=False), SceneAnswer(answer="one", object_ids=[1])]
    t = tools([plan(("observe", {}), ("answer", {"question": "q?"})), plan(("answer", {"question": "which one exactly?"}))])
    t.request_kind = "question"
    t.roles.answer = lambda q, f, ids: answers.pop(0)
    r, _ = run(tmp_path, t, instruction="Which one?")
    import json
    st = json.loads((tmp_path / "task_state.json").read_text())
    assert {"held_object", "motion_count"} <= set(st["robot"]) and "answer" in st["capabilities"]
    assert any("model-generated" in " ".join(u["items"]) or "not certain" in " ".join(u["items"]) for u in st["uncertainties"])
    assert st["consults"] and st["plan"] and st["hypotheses"].get("answers")


def test_assumptions_are_kept_apart_from_measurements(tmp_path):
    t = tools([plan(PICK_A), plan(PLACE_IT)])
    run(tmp_path / "1" if (tmp_path / "1").mkdir() is None else tmp_path, t, instruction="Pick up a.")
    (tmp_path / "2").mkdir()
    r, _ = run(tmp_path / "2", t, instruction="Put it beside c.")
    assert r["assumptions"] and "'it' means #" in r["assumptions"][0]["assumed"]


# --------------------------------------------------------------------------------------------- trace
def test_the_trace_shows_decisions_reviews_plan_updates_and_consultations():
    ev = lambda kind, **f: {"wall": "10:00:00", "kind": kind, **f}  # noqa: E731
    d = format_trace(ev("decision", index=2, capability="locate", worker="perception", reason="next pending step"))
    assert "decide: step 2 locate -> PERCEPTION" in d.text
    rv = format_trace(ev("review", index=2, capability="locate", outcome="insufficient_info", decision="explore, then locate again",
                         reason="not visible"))
    assert "review step 2 locate (insufficient_info): explore, then locate again" in rv.text
    pu = format_trace(ev("plan_update", reason="subtask", plan=[{"index": 3, "capability": "explore", "status": "pending"}]))
    assert "3. explore [pending]" in pu.text
    c = format_trace(ev("consult", reason="goal not met: x", replans=1, budget=1))
    assert "-> GEMMA" in c.text
    wr = format_trace(ev("worker_result", worker="perception", task_id="t1", status="insufficient_info", outcome="insufficient_info",
                         summary="not visible", failure_reason="not visible", metrics={"seconds": 1, "gemma_calls": 2},
                         confidence="none", tools=["perceive", "gemma:resolve"], uncertainty=["the evidence does not settle it"],
                         needs="explore"))
    assert "tools: perceive, gemma:resolve" in wr.text and "needs: explore" in wr.text and "uncertain:" in wr.text
