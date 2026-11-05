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
from agent import capabilities as caps  # noqa: E402
from agent.events import format_event, format_trace  # noqa: E402
from agent.vlm_roles import RecordedVLM, Roles, SceneAnswer, VLMFailure  # noqa: E402
from tests.support.fakes import GoalTools  # noqa: E402
from tests.support.runners import plan, problems, run_supervisor as run  # noqa: E402


def run_with(tmp_path, t, controls, instruction="do it"):
    """run_supervisor with explicit ExecutionControls (the Gemma decision budget)."""
    from agent.eventlog import EventLog
    from agent.supervisor import GoalSupervisor
    log = EventLog(tmp_path / "events.jsonl")
    r = GoalSupervisor(t, log, tmp_path, ScriptedResponder(["cancel"]), controls=controls).run(instruction)
    log.close()
    return r, [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]

PICK_A = ("pick", {"object": "a"})
PLACE_IT = ("place", {"object": "#1", "relation": "next_to", "target": "c"})  # the held object, by its id


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
    # manipulation does the pick; verification checks the subtask, the original goal so far, and the final goal
    assert [e["worker"] for e in of(ev, "delegate")] == ["manipulation", "verification", "verification", "verification"]
    assert all(res["status"] in ("completed", "failed", "blocked", "insufficient_info", "cancelled") for res in of(ev, "worker_result"))
    assert [e["status"] for e in of(ev, "report")] == ["COMPLETED"]  # the supervisor's own conclusion, once
    assert r["subgoal_checks"][0]["outcome"] == "completed" and r["goal_check"][0]["satisfied"] is True


# ------------------------------------------------------------------------------------- feedback loop
def test_a_subtask_that_verifies_the_whole_goal_completes_without_extra_reasoning(tmp_path):
    t = tools([plan(PICK_A)])
    r, ev = run(tmp_path, t, instruction="Pick up a.")
    assert r["status"] == "COMPLETED" and len(t.plan_calls) == 1 and t.decide_calls == [] and r["consults"] == []
    assert [e["decision"] for e in of(ev, "review")] == ["original goal verified", "task verified"]


def test_a_failed_subgoal_returns_to_gemma_with_the_evidence_and_the_task_memory(tmp_path):
    """The worker says done; the verification worker measures the opposite; Gemma decides again from that."""
    # the re-plan places whatever is held (no object named: the fake world re-measures the moved object as a new id)
    goal = [{"pred": "next_to", "args": ["a", "c"]}, {"pred": "not_holding"}]  # the user's words, as Gemma states them
    t = tools([plan(PICK_A, PLACE_IT, success=goal), plan(PICK_A, ("place", {"relation": "next_to", "target": "c"}))],
              misplace=True)
    plan_goal = t.roles.plan_goal

    def planner(text, menu, context="", already_met=None):  # after the first plan, placements land where they are asked to
        p = plan_goal(text, menu, context)
        t.misplace = len(t.plan_calls) < 2
        return p
    t.roles.plan_goal = planner
    r, ev = run(tmp_path, t, instruction="Put a beside c.")
    first_place = next(s for s in r["plan"] if s["capability"] == "place")
    assert first_place["status"] == "replaced" and first_place["outcome"] == "failed"
    assert r["subgoal_checks"][1]["outcome"] == "failed"
    first = problems(t)[0]  # what Gemma decided from: the measured evidence and the task memory
    assert "sub-goal is not met" in first["event"] and "centres 680 mm apart" in first["event"]
    assert "Original goal (fixed)" in first["progress"] and "Still to achieve: next_to(a, c)" in first["progress"]
    assert [c["action"] for c in r["consults"]] == ["continue", "replan", "continue"] and r["consults"][1]["accepted"]
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
    # observe went as planned (continue); the uncertain answer leaves answered(...) unmet: Gemma re-plans the answer
    assert len(t.plan_calls) == 2 and "not certain" in problems(t)[0]["event"]
    assert "answered(How many objects are there?)=NOT met" in problems(t)[0]["progress"]
    assert [c["action"] for c in r["consults"]] == ["continue", "replan"] and r["consults"][1]["steps"] == ["answer"]
    assert r["status"] == "COMPLETED" and r["replans"] == 1 and "not independently verified" in r["message"]
    assert any(e["decision"] == "ask Gemma" for e in of(ev, "review"))


def test_the_loop_stops_when_the_budget_is_used(tmp_path):
    answers = [SceneAnswer(answer="maybe", object_ids=[1], certain=False)] * 3
    t = tools([plan(("observe", {}), ("answer", {"question": "q?"}))])
    t.request_kind = "question"
    t.roles.answer = lambda q, f, ids: answers.pop(0)
    t.decisions = [{"action": "replan", "steps": [{"do": "answer", "question": "q again?"}]}] * 3
    r, _ = run_with(tmp_path, t, ExecutionControls(max_replans=1), instruction="Which one?")
    assert r["status"] == "FAILED" and r["replans"] == 1 and len(problems(t)) == 2  # one change accepted, the next refused
    assert "budget 1 used" in r["message"] and "did not report success" in r["message"]


def test_a_zero_budget_allows_no_change_of_course(tmp_path):
    t = tools([plan(PICK_A)], placement=(False,), contact=(False,))
    t.decisions = [{"action": "replan", "steps": [{"do": "observe"}, {"do": "pick", "object": "a"}]}]
    r, _ = run_with(tmp_path, t, ExecutionControls(max_replans=0), instruction="Pick up a.")
    assert r["status"] == "FAILED" and r["replans"] == 0 and "budget 0 used" in r["message"]


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
    assert [e["action"] for e in of(ev, "gemma_decision")] == ["continue", "continue"] and len(t.plan_calls) == 1


def test_gemma_can_ask_the_user_and_decides_again_from_the_answer(tmp_path):
    t = uncertain_then()
    t.decisions = [{"action": "ask_user", "question": "Do you mean the left one?"},
                   {"action": "replan", "steps": [{"do": "answer", "question": "q, the left one?"}]}]
    r, ev = run(tmp_path, t, ScriptedResponder(["yes, the left one"]), instruction="Which one?")
    assert r["questions"] == [{"question": "Do you mean the left one?", "answer": "yes, the left one", "asked_by": "gemma"}]
    assert any("'yes, the left one'" in c["event"] for c in t.decide_calls) and r["status"] == "COMPLETED"
    line = format_event({"wall": "t", **[e for e in of(ev, "gemma_decision") if e["action"] == "ask_user"][0]})
    assert line.component == "GEMMA" and "ASK_USER" in line.text


def test_no_answer_to_gemmas_question_cancels(tmp_path):
    t = uncertain_then()
    t.decisions = [{"action": "ask_user", "question": "Which one do you mean?"}]
    r, _ = run(tmp_path, t, ScriptedResponder(["cancel"]), instruction="Which one?")
    assert r["status"] == "CANCELLED" and "No answer" in r["message"] and len(problems(t)) == 1


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


def test_safety_stops_do_not_ask_gemma_and_refusals_are_reported_to_it(tmp_path):
    t = tools([plan(("pick_place", {"object": "a", "relation": "on", "target": "b"}))], exec_fail="approach")
    r, _ = run(tmp_path, t, instruction="Put a on b.")
    assert r["status"] == "FAILED" and r["outcome"] == "unsafe" and t.decide_calls == []
    t = tools([plan(PICK_A, ("observe", {}))])
    t.request_kind = "description"  # a description request does not authorize picking
    t.decisions = [{"action": "continue", "reason": "go on without the refused step"}]
    (tmp_path / "q").mkdir()
    r, _ = run(tmp_path / "q", t, instruction="Describe a.")
    assert r["plan"][0]["status"] == "refused" and t.executed == [] and "not authorized" in problems(t)[0]["event"]


def test_roles_decide_validates_and_retries_with_the_error():
    vlm = RecordedVLM([json.dumps({"action": "replan", "steps": []}),
                       json.dumps({"action": "replan", "steps": [{"worker": "perception", "do": "locate", "object": "a"}],
                                   "reason": "r"})])
    d = Roles(vlm).decide("find a", "Original goal (fixed): find a", "locate failed", "menu", has_pending=False)
    assert d.action == "replan" and d.steps[0].capability == "locate" and "replan needs the remaining steps" in vlm.calls[1]
    with pytest.raises(VLMFailure):
        Roles(RecordedVLM([json.dumps({"action": "dance"})] * 2)).decide("g", "", "e", "menu", has_pending=True)


# ------------------------------------------------------------------- task memory and ownership of the goal
def test_1_a_worker_succeeds_and_gemma_moves_on_to_the_next_needed_worker(tmp_path):
    """Perception finds the object (subtask achieved), the original goal is not reached yet, and Gemma hands the next
    subtask to the manipulation worker."""
    t = tools([plan(("locate", {"object": "a"}), success=[{"pred": "on", "args": ["a", "b"]}])])
    t.decisions = [{"action": "replan", "steps": [{"worker": "manipulation", "do": "pick_place", "object": "a",
                                                   "relation": "on", "target": "b"}], "reason": "a is found: move it"}]
    r, ev = run(tmp_path, t, instruction="Put a on b.")
    first = t.decide_calls[0]
    assert "perception worker: locate" in first["event"] and "subtask achieved" in first["event"]
    assert "Still to achieve: on(a, b)" in first["progress"]
    assert [s["worker"] for s in r["plan"]] == ["perception", "manipulation"]
    assert r["status"] == "COMPLETED" and r["goal_check"][0]["condition"]["pred"] == "on" and r["goal_check"][0]["satisfied"]


def test_2_uncertainty_makes_gemma_ask_for_useful_new_evidence(tmp_path, monkeypatch):
    inspected = []

    def inspect(ctx, args):  # a stand-in inspection: records the wrist observation a real one adds to the world model
        inspected.append(args["target"])
        ctx.tools.world.objects[int(args["target"][1:])].observations.append({"camera": "wrist", "frame_id": "w", "mark": 1})
        return caps.StepResult("completed", "close look from the wrist camera")
    monkeypatch.setattr(caps, "run_inspect", inspect)
    answers = [SceneAnswer(answer="maybe #1", object_ids=[1], certain=False, check_closer=[1]),
               SceneAnswer(answer="#1", object_ids=[1])]
    t = tools([plan(("observe", {}), ("answer", {"question": "which is round?"}))])
    t.request_kind = "question"
    t.roles.answer = lambda q, f, ids: answers.pop(0)
    t.decisions = [lambda: {"action": "replan", "steps": [{"worker": "inspection", "do": "inspect", "object": "#1"},
                                                          {"worker": "perception", "do": "answer", "question": "which is round?"}]}]
    r, _ = run(tmp_path, t, instruction="Which is round?")
    asked = problems(t)[0]
    assert "uncertain" in asked["event"] and "closer look could settle" in asked["event"]
    assert inspected == ["#1"] and r["status"] == "COMPLETED"
    assert [x["worker"] for x in r["exchanges"]] == ["perception", "perception", "inspection", "perception"]


def test_3_a_failed_worker_is_not_repeated_with_unchanged_evidence(tmp_path):
    t = tools([plan(PICK_A)], contact=(False,))
    t.decisions = [{"action": "replan", "steps": [{"do": "pick", "object": "a"}]},  # identical, nothing changed
                   {"action": "replan", "steps": [{"do": "observe"}, {"do": "pick", "object": "a"}]},  # new evidence first
                   {"action": "stop", "reason": "still no grasp"}]
    r, ev = run(tmp_path, t, instruction="Pick up a.")
    assert [e["accepted"] for e in of(ev, "replan")] == [False, True]
    assert "repeated a failed subtask although nothing changed" in problems(t)[1]["event"]
    assert r["failures"] and "Failed before: manipulation pick" in problems(t)[1]["progress"]


def test_4_a_successful_worker_does_not_complete_an_unfinished_task(tmp_path):
    """'pick the second block' planned as an observation only: the observation succeeds (tool and subtask), the
    original goal (holding the block) is not met, so the task is not completed."""
    t = tools([plan(("observe", {}), success=[{"pred": "holding", "args": ["the second block"]}])])
    t.decisions = [{"action": "stop", "reason": "I only looked"}]
    r, ev = run(tmp_path, t, instruction="Pick the second block.")
    obs = r["exchanges"][0]
    assert (obs["tool"], obs["subtask"]) == ("completed", "done (no measurable effect)")
    assert r["status"] != "COMPLETED" and any("holding(the second block)" in u for u in r["unmet"])
    assert r["conditions"] == [{"pred": "holding", "args": ["the second block"], "source": "gemma"}]


def test_5_replanning_keeps_the_original_goal(tmp_path):
    t = tools([plan(PICK_A, success=[{"pred": "on", "args": ["a", "b"]}])], contact=(False,))
    t.decisions = [{"action": "replan", "steps": [{"do": "observe"}], "reason": "look again"},
                   {"action": "finish", "reason": "done"}, {"action": "stop", "reason": "cannot"}]
    r, ev = run(tmp_path, t, instruction="Put a on b.")
    assert r["conditions"] == r["original_goal"]["criteria"] == [{"pred": "on", "args": ["a", "b"], "source": "gemma"}]
    assert r["status"] != "COMPLETED"  # a re-plan to an observation and a "finish" do not change or drop the goal
    assert [s["capability"] for s in r["plan"]] == ["pick", "observe"]


def test_6_completed_only_with_measured_evidence_for_the_original_goal(tmp_path):
    bad = tools([plan(("pick_place", {"object": "a", "relation": "next_to", "target": "c"}))], misplace=True)
    bad.decisions = [{"action": "finish", "reason": "the worker said done"}, {"action": "stop", "reason": "-"}]
    r, _ = run(tmp_path, bad, instruction="Put a beside c.")  # the skill reports done; the placement is measured wrong
    assert (r["exchanges"][0]["tool"], r["exchanges"][0]["subtask"]) == ("completed", "not achieved")
    assert r["status"] != "COMPLETED"  # tool success is not task success
    good = tools([plan(("pick_place", {"object": "a", "relation": "on", "target": "b"}))])
    (tmp_path / "g").mkdir()
    r, _ = run(tmp_path / "g", good, instruction="Put a on b.")
    assert r["status"] == "COMPLETED" and all(g["satisfied"] and g["confidence"] == "measured" for g in r["goal_check"])


def test_each_instruction_gets_a_fresh_task_memory(tmp_path):
    t = tools([plan(PICK_A), plan(("place", {"object": "#1", "relation": "next_to", "target": "c"}))])
    (tmp_path / "1").mkdir()
    (tmp_path / "2").mkdir()
    r1, _ = run(tmp_path / "1", t, instruction="Pick up a.")
    r2, _ = run(tmp_path / "2", t, instruction="Put it beside c.")
    import json as _json
    st2 = _json.loads((tmp_path / "2" / "task_state.json").read_text())
    assert st2["finalized"] and st2["instruction"] == "Put it beside c." and [x["operation"] for x in st2["exchanges"]] == ["place"]
    assert r1["original_goal"]["instruction"] == "Pick up a." and r2["original_goal"]["instruction"] == "Put it beside c."


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


def test_criteria_in_the_prompts_notation_and_nested_shapes_are_read_structurally():
    """Measured with the real Gemma (2026-10-20): it writes criteria as "on(a, b)" strings or in "pred", and nests a
    condition's own keys under "args". JSON / notation shape only; free text stays an unknown predicate."""
    from agent.vlm_roles import GoalPlan
    p = GoalPlan.model_validate({"success": ["on(red cube, green cube)", {"pred": 'holding("a")'}, "at_home()",
                                            {"pred": "viewing()", "args": {"direction": [-1, 0, 0], "frame": "robot"}},
                                            {"pred": "viewing", "args": [0, 0, -1]}, "the robot reports"]})
    got = [c.model_dump(exclude_none=True) for c in p.success]
    assert got[:3] == [{"pred": "on", "args": ["red cube", "green cube"]}, {"pred": "holding", "args": ["a"]},
                       {"pred": "at_home", "args": []}]
    assert got[3] == {"pred": "viewing", "args": [], "direction": [-1, 0, 0], "frame": "robot"}
    assert got[4]["direction"] == [0, 0, -1] and got[5]["pred"] == "the robot reports"


def test_continue_keeps_the_plan_and_gemma_is_told_its_extra_steps_were_not_used(tmp_path):
    """Live 2026-10-20: Gemma answered continue + new steps. continue keeps the planned steps; the extra ones are not
    used, and the next event says so (use replan to change the steps)."""
    t = tools([plan(("observe", {}), ("pick", {"object": "a"}), success=[{"pred": "on", "args": ["a", "b"]}])])
    t.continue_on_success = False
    t.decisions = [{"action": "continue", "steps": [{"do": "place", "object": "#1", "relation": "on", "target": "b"}]},
                   {"action": "stop", "reason": "-"}]
    r, ev = run(tmp_path, t, instruction="Put a on b.")
    assert [s["capability"] for s in r["plan"]] == ["observe", "pick"]  # the plan was kept
    assert "were not used; use replan" in t.decide_calls[1]["event"]


def test_words_bound_to_the_held_object_in_this_task_name_it(tmp_path):
    """Live 2026-10-20: 'red block' was bound to #6 by the pick; a pick_place of 'red block' is then the held object
    (a precondition fact, no motion; Gemma decides), not 'another object' (an unsafe stop)."""
    t = tools([plan(("pick", {"object": "a"}), ("pick_place", {"object": "a", "relation": "on", "target": "b"}),
                    success=[{"pred": "on", "args": ["a", "b"]}])])
    t.decisions = [{"action": "replan", "steps": [{"do": "place", "object": "#1", "relation": "on", "target": "b"}]}]
    r, ev = run(tmp_path, t, instruction="Put a on b.")
    pp = r["exchanges"][1]
    assert (pp["operation"], pp["outcome"]) == ("pick_place", "infeasible") and "already in the gripper" in pp["happened"]
    assert r["status"] == "COMPLETED" and [x["worker"] for x in r["exchanges"]] == ["manipulation"] * 3


def test_a_worker_named_by_its_menu_label_is_the_same_worker():
    """Live 2026-10-20: Gemma wrote "manipulation worker" (the menu's label); a different worker is still refused."""
    from agent.capabilities import check_worker
    check_worker("pick_place", "manipulation worker")
    check_worker("pick_place", "Manipulation")
    with pytest.raises(ValueError):
        check_worker("pick_place", "inspection worker")


def test_after_a_failure_gemma_decides_from_a_menu_that_marks_the_failed_operation(tmp_path):
    """Live 2026-10-21: decompose failed, Gemma chose decompose again (three times). The failure is now written next to
    the operation in the menu Gemma decides from (nothing is blocked; the decision stays Gemma's)."""
    t = tools([plan(PICK_A, PLACE_IT)], misplace=True)
    run(tmp_path, t, instruction="Put a beside c.")
    first = problems(t)[0]
    place = next(x for x in first["menu"].splitlines() if x.strip().startswith("- place"))
    assert "FAILED in this task: 1 time(s)" in place
    assert "FAILED" not in next(x for x in first["menu"].splitlines() if x.strip().startswith("- pick"))
