"""Goal supervisor (agent/supervisor.py), capabilities and planner validation, with mocked tools (no sim, no GPU)."""

import json

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langgraph")

from tests.support.fakes import GoalTools  # noqa: E402,F401
from tests.support.runners import plan, problems, run_supervisor as run  # noqa: E402,F401

from agent.vlm_roles import RecordedVLM, Roles, SceneAnswer, VLMFailure  # noqa: E402

PP = {"object": "a", "relation": "on", "target": "b"}
PP2 = {**PP, "object": "c"}








def test_pick_and_place_goal_delegates_to_the_validated_skill(tmp_path):
    t = GoalTools([plan(("pick_place", PP))])
    r, ev = run(tmp_path, t)
    assert r["status"] == "COMPLETED" and len(t.plan_calls) == 1  # one planner call replaces the old parse call
    assert t.home_calls == 1 and "back at HOME" in r["message"]
    assert [v["check"] for v in r["verifications"]] == ["contact", "contact_after_carry", "placement", "home"]
    goal_states = [e["state"] for e in ev if e["kind"] == "state" and e.get("level") == "goal"]
    assert goal_states[:4] == ["INTERPRETING", "DISPATCHING", "DELEGATING", "REVIEWING"] and goal_states[-1] == "COMPLETED"
    st = json.loads((tmp_path / "task_state.json").read_text())
    assert st["status"] == "completed" and st["plan"][0]["outcome"] == "completed" and st["verified"]


def test_observation_only_goal_moves_nothing_and_reports_grounded_objects(tmp_path):
    t = GoalTools([plan(("observe", {}), ("describe_scene", {}))])
    r, ev = run(tmp_path, t, instruction="Take a look at your surroundings.")
    assert r["status"] == "COMPLETED" and t.executed == [] and t.parks == 0 and t.perceived == 1
    assert "I see 3 objects" in r["message"] and "Gemma's descriptions, not verified" in r["message"]
    assert "x +550 mm" in r["message"]  # measured position of object 1
    st = json.loads((tmp_path / "task_state.json").read_text())
    assert len(st["facts"]["objects"]) == 3 and st["hypotheses"]["descriptions"]  # facts and hypotheses kept apart
    assert any(e["kind"] == "scene_report" for e in ev)


def test_scene_question_observes_automatically_and_marks_the_answer_unverified(tmp_path):
    t = GoalTools([plan(("answer", {"question": "how many cubes?"}))])  # no observe step: added automatically
    r, ev = run(tmp_path, t)
    assert r["status"] == "COMPLETED" and t.perceived == 1 and "not verified" in r["message"]
    assert any(e["kind"] == "step" and e.get("event") == "auto_observe" for e in ev)


def test_empty_plan_is_reported_as_unsupported_with_the_capabilities(tmp_path):
    t = GoalTools([plan(reason="I cannot sing")])
    r, _ = run(tmp_path, t)
    assert r["status"] == "UNSUPPORTED" and "I cannot sing" in r["message"] and "Operations implemented now:" in r["message"]
    assert t.executed == []


@pytest.fixture
def explore_stub(monkeypatch):
    """Exploration internals are tested elsewhere; here it returns a scripted result."""
    from agent import capabilities as caps
    calls = []

    def fake(ctx, args, finds=False):
        calls.append(args)
        if finds:
            ctx.found_by_wrist = args["target"]
        return caps.StepResult("completed", f"looked into 1 hidden region(s); {'found it' if finds else 'nothing new'}")
    monkeypatch.setattr(caps, "run_explore", fake)
    return calls


def test_not_visible_target_goes_to_gemma_which_decides_to_explore(tmp_path, explore_stub):
    """No keyword / needs rule adds the exploration: Gemma decides it from the worker's result."""
    t = GoalTools([plan(("observe", {}), ("locate", {"object": "zz"}))], ambiguous={"zz": []})
    t.decisions = [{"action": "replan", "steps": [{"do": "explore", "object": "zz"}, {"do": "locate", "object": "zz"}],
                    "reason": "not visible: look into hidden regions"},
                   {"action": "stop", "reason": "still not visible after exploring"}]
    r, ev = run(tmp_path, t)
    assert [s["capability"] for s in r["plan"]] == ["observe", "locate", "explore", "locate"]
    assert [s["status"] for s in r["plan"]][:3] == ["done", "replaced", "done"]
    assert len(explore_stub) == 1 and "insufficient_info" in problems(t)[0]["event"]
    assert "suggests" not in problems(t)[0]["event"]  # Python names no next operation (removed 2026-10-20)
    assert r["status"] == "FAILED" and r["outcome"] == "insufficient_info"  # exploring is not finding
    assert [e["action"] for e in ev if e["kind"] == "gemma_decision" and e["action"] != "continue"] == ["replan", "stop"]


def test_found_after_exploration_completes_the_goal(tmp_path, monkeypatch):
    from agent import capabilities as caps
    monkeypatch.setattr(caps, "run_explore", lambda ctx, args: caps.StepResult("completed", "looked behind object 2"))
    real_locate, n = caps.run_locate, []

    def locate(ctx, args):
        n.append(1)
        if len(n) == 1:
            return caps.StepResult("insufficient_info", "I do not see 'zz' from the current view")
        return caps.StepResult("completed", "'zz' found with the wrist camera at x +100 mm",
                               {"mark": 9, "camera": "wrist", "bindings": {"zz": 1}})
    monkeypatch.setattr(caps, "run_locate", locate)
    t = GoalTools([plan(("observe", {}), ("locate", {"object": "zz"}))])
    t.decisions = [{"action": "replan", "steps": [{"do": "explore", "object": "zz"}, {"do": "locate", "object": "zz"}]}]
    r, _ = run(tmp_path, t)
    assert r["status"] == "COMPLETED" and r["message"].startswith("looked behind") is False and "found with the wrist camera" in r["message"]


def test_multi_step_plan_returns_home_once_at_the_end(tmp_path):
    t = GoalTools([plan(("pick_place", PP), ("pick_place", PP2))])
    r, ev = run(tmp_path, t)
    assert r["status"] == "COMPLETED" and t.home_calls == 1
    starts = [e for e in ev if e["kind"] == "skill" and e.get("event") == "start"]
    assert [s["final_home"] for s in starts] == [False, True]


def test_observe_then_manipulate_reuses_the_observation_without_parking_first(tmp_path):
    t = GoalTools([plan(("observe", {}), ("pick_place", PP))])
    r, ev = run(tmp_path, t)
    assert r["status"] == "COMPLETED"
    assert not any("park J1" in str(e) for e in ev)  # no pre-observation park (only the placement check's view pose)
    assert any(e["kind"] == "tool" and e.get("tool") == "observation_reused" for e in ev)
    assert t.perceived == 1  # the skill did not capture again (verification uses its own stub)


def test_ambiguity_cancel_skips_the_rest(tmp_path):
    t = GoalTools([plan(("pick_place", PP), ("pick_place", PP2))], ambiguous={"a": [1, 3]})
    r, _ = run(tmp_path, t)
    assert r["status"] == "CANCELLED" and r["plan"][1]["status"] == "skipped" and t.executed == []


def test_unsafe_failure_is_never_replanned(tmp_path):
    t = GoalTools([plan(("pick_place", PP)), plan(("pick_place", PP))], exec_fail="approach")
    r, _ = run(tmp_path, t)
    assert r["status"] == "FAILED" and r["outcome"] == "unsafe" and len(t.plan_calls) == 1 and r["replans"] == 0


def test_a_repeated_failed_step_is_sent_back_to_gemma(tmp_path):
    """Gemma re-plans the identical failed step although nothing changed: rejected (deterministic guard), Gemma is told
    why and decides again (here: stop). Each decision counts against the budget."""
    t = GoalTools([plan(("pick_place", PP))], placement=(False,))
    t.decisions = [{"action": "replan", "steps": [{"do": "pick_place", **PP}]}, {"action": "stop", "reason": "cannot place it"}]
    r, ev = run(tmp_path, t)
    assert r["status"] == "FAILED" and r["replans"] == 2 and "Gemma stopped: cannot place it" in r["message"]
    assert [e["accepted"] for e in ev if e["kind"] == "replan"] == [False]
    assert "repeated a failed subtask" in problems(t)[1]["event"]


def test_a_replanned_alternative_must_still_satisfy_the_original_goal(tmp_path):
    """The goal is answered(q): a re-plan to a scene report does not complete it; a re-plan that answers does."""
    t = GoalTools([plan(("answer", {"question": "q"})), plan(("observe", {}), ("describe_scene", {}))])
    t.roles.answer = lambda q, f, ids: (_ for _ in ()).throw(VLMFailure("bad"))  # the first answer fails
    r, _ = run(tmp_path, t)
    assert r["status"] != "COMPLETED" and r["conditions"][0]["pred"] == "answered"
    calls = []

    def answer(q, f, ids):
        calls.append(q)
        if len(calls) == 1:
            raise VLMFailure("bad")
        return SceneAnswer(answer="two", object_ids=[1])
    t = GoalTools([plan(("answer", {"question": "q"})), plan(("observe", {}), ("answer", {"question": "q"}))])
    t.roles.answer = answer
    (tmp_path / "2").mkdir()
    r, _ = run(tmp_path / "2", t)
    assert r["status"] == "COMPLETED" and r["replans"] == 1
    assert [s["status"] for s in r["plan"]] == ["replaced", "done", "done"]


def test_planner_failure_is_a_clear_failure(tmp_path):
    t = GoalTools([VLMFailure("plan: invalid answer after one retry")])
    r, _ = run(tmp_path, t)
    assert r["status"] == "FAILED" and "could not turn the instruction into steps" in r["message"]


# --------------------------------------------------------------------------- planner validation (real Roles)
def test_planner_keeps_unknown_operations_and_checks_known_operations_arguments():
    """The operation name is open: an operation nobody implements is kept (reported unavailable when reached, never
    replaced); an implemented operation's arguments are checked by its own interface (one retry with the error)."""
    good = json.dumps({"goal": "g", "success": [{"pred": "on", "args": ["a", "b"]}],
                       "steps": [{"worker": "manipulation", "do": "pick_place", "object": "a", "relation": "on", "target": "b"}]})
    for bad in ('{"steps": [{"do": "pick_place", "object": "a"}]}',
                '{"steps": [{"do": "answer"}]}',
                '{"steps": [{"do": "locate"}]}',
                '{"steps": [{"do": "pick_place", "object": "a", "relation": "inside", "target": "b"}]}',
                '{"steps": [{"do": "look", "direction": "left"}]}',
                '{"steps": [{"do": ""}]}'):
        vlm = RecordedVLM([bad, good])
        p = Roles(vlm).plan_goal("x", "menu")
        assert p.steps[0].capability == "pick_place" and len(vlm.calls) == 2  # one retry with the error
        assert p.steps[0].args == {"object": {"text": "a", "relation": None, "anchor": None}, "relation": "on",
                                   "reference": {"text": "b", "relation": None, "anchor": None}, "frame": "robot"}
    p = Roles(RecordedVLM(['{"goal": "g", "success": [{"pred": "at_home"}], "steps": [{"do": "move_hand", "offset": [-0.1, 0, 0], '
                           '"frame": "robot"}]}'])).plan_goal("x", "m")
    assert p.steps[0].capability == "move_hand" and p.steps[0].args == {"offset": [-0.1, 0, 0], "frame": "robot"}


def test_an_operation_nobody_implements_is_reported_unavailable_and_never_replaced(tmp_path):
    t = GoalTools([plan(("move_hand", {"offset": [-0.1, 0, 0], "frame": "robot"}))])
    t.decisions = [{"action": "stop", "reason": "I cannot move my hand freely"}]
    r, ev = run(tmp_path, t, instruction="Move your hand to the left.")
    assert r["status"] == "UNSUPPORTED" and t.executed == [] and "'move_hand' is not implemented" in r["message"]
    assert [s["capability"] for s in r["plan"]] == ["move_hand"]  # nothing substituted
    assert "not implemented" in problems(t)[0]["event"]


def test_locate_finds_confirms(tmp_path):
    t = GoalTools([plan(("observe", {}), ("locate", {"object": "a"}))])
    r, ev = run(tmp_path, t)
    assert r["status"] == "COMPLETED" and "is object 1" in r["message"] and t.executed == []


def test_planner_menu_lists_only_registry_capabilities():
    from agent.capabilities import SPECS, planner_menu
    menu = planner_menu()
    assert all(f"- {n}:" in menu for n in SPECS) and all(sp.implemented for sp in SPECS.values())
    assert {s.safety for s in SPECS.values()} == {"read_only", "motion"}


def test_gemma_plan_is_executed_as_written_and_a_missing_answer_goes_back_to_gemma(tmp_path):
    """No completion rules rewrite Gemma's plan (behaviour change 2026-10-18). A question whose plan never answers
    fails the goal check (answered(...) comes from the request kind), and Gemma adds the missing step."""
    t = GoalTools([plan(("observe", {}))])
    r, ev = run(tmp_path, t, instruction="Take a look at your surroundings.")
    assert r["status"] == "COMPLETED" and [s["capability"] for s in r["plan"]] == ["observe"] and t.decide_calls == []
    t = GoalTools([plan(("observe", {})), plan(("answer", {"question": "how many cubes are there?"}))])
    t.request_kind = "question"
    (tmp_path / "q").mkdir()
    r, _ = run(tmp_path / "q", t, instruction="How many cubes are there?")
    assert [s["capability"] for s in r["plan"]] == ["observe", "answer"] and "not verified" in r["message"]
    assert "answered(How many cubes are there?)=NOT met" in t.decide_calls[0]["progress"] and r["status"] == "COMPLETED"


def test_gemma_listing_several_candidates_is_ambiguous_not_inferred_invisible(tmp_path, explore_stub):
    """Behaviour change (2026-10-20): Python no longer overrides Gemma's ambiguous answer with a word-overlap test
    ("no description contains the words -> not visible"). Gemma's verdict stands and Gemma decides what next."""
    t = GoalTools([plan(("observe", {}), ("locate", {"object": "banana"}))], ambiguous={"banana": [1, 2, 3]})
    t.decisions = [{"action": "stop", "reason": "none of them looks like a banana"}]
    r, _ = run(tmp_path, t)
    assert r["status"] == "CANCELLED" and r["outcome"] == "ambiguous" and "several objects could be 'banana'" in r["message"]
    assert "several objects could be 'banana'" in problems(t)[0]["event"]


def test_locate_with_no_candidate_from_gemma_reports_not_seen(tmp_path):
    t = GoalTools([plan(("observe", {}), ("locate", {"object": "banana"}))], ambiguous={"banana": []})
    t.decisions = [{"action": "stop", "reason": "not here"}]
    r, _ = run(tmp_path, t)
    assert r["status"] == "FAILED" and r["outcome"] == "insufficient_info" and "do not see 'banana'" in r["message"]


def test_result_comes_before_the_observation_summary(tmp_path):
    t = GoalTools([plan(("observe", {}), ("describe_scene", {}))])
    r, _ = run(tmp_path, t)
    assert r["message"].startswith("I see 3 objects") and "workspace points from the camera)" in r["message"]


def test_replan_that_drops_the_objective_cannot_complete_the_goal(tmp_path, explore_stub):
    """Regression (live 18:48): 'find the banana' -> locate failed -> a replan [observe] must not complete the goal:
    located(banana) is part of the ORIGINAL goal (fixed at interpretation), so the check fails and the loop stops."""
    t = GoalTools([plan(("observe", {}), ("locate", {"object": "banana"})), plan(("observe", {}))],
                  ambiguous={"banana": [1, 2, 3]})
    r, ev = run(tmp_path, t)
    assert r["status"] != "COMPLETED" and any("located(banana)" in u for u in r["unmet"])
    assert any(c["condition"]["pred"] == "located" and not c["satisfied"] for c in r["goal_check"])
    # 2026-10-26: observing again into an unchanged scene is refused as a repeat (it looped live), not run
    assert [s["capability"] for s in r["plan"]].count("observe") == 1
    assert any(e["kind"] == "replan" and not e["accepted"] for e in ev)


def test_arguments_flat_or_nested_are_the_same_step():
    """JSON shape only (measured 2026-10-20: Gemma nested the arguments under "arguments" after the prompt cleanup;
    live 2026-10-25: under "argument", so every decompose step lost its goal)."""
    from agent.vlm_roles import PlanStep
    flat = PlanStep.model_validate({"do": "locate", "object": "the <A>"})
    for key in ("arguments", "argument", "args"):
        assert PlanStep.model_validate({"do": "locate", key: {"object": "the <A>"}}).args == flat.args


def test_a_finish_with_gemmas_own_reply_answers_the_question(tmp_path):
    """live 2026-11-04: "how many cubes are on the table" was planned as only observe; Gemma chose finish three times
    with no way to say the count. A finish now carries Gemma's reply: it is the answer, marked not verified."""
    t = GoalTools([plan(("observe", {}))])
    t.request_kind = "question"
    t.decisions = [{"action": "finish", "answer": "There are 4 cubes on the table.", "reason": "counted"}]
    r, _ = run(tmp_path, t, instruction="How many cubes are on the table?")
    assert r["status"] == "COMPLETED" and r["message"].startswith("There are 4 cubes on the table.")
    assert "not independently verified" in r["message"]


def test_the_answer_line_is_read_from_gemmas_decision():
    vlm = RecordedVLM(["ACTION: finish\nANSWER: There are 4 cubes:\n#3, #5, #10 and #12.\nREASON: counted"])
    d = Roles(vlm).decide("how many cubes", "", "", "", has_pending=False)
    assert d.action == "finish" and d.answer == "There are 4 cubes: #3, #5, #10 and #12."
