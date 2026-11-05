"""Robustness fixes of 2026-10-26 (docs/ROBUSTNESS_FIXES_2026-10-26.md): every instruction ends (completed, asked or
stopped) whatever Gemma answers, and none of the live failures of that day can repeat. Scripted Gemma, no sim, no GPU."""

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langgraph")

from agent import capabilities as caps  # noqa: E402
from agent.capabilities.specs import ADAPT, PLACE_RELATIONS  # noqa: E402
from agent.supervisor import MAX_CONSULTS  # noqa: E402
from agent.vlm_roles import Condition, RecordedVLM, Roles, SubPlan  # noqa: E402
from tests.integration.test_orchestration import _decomposer, run_in, session  # noqa: E402
from tests.support.fakes import GoalTools  # noqa: E402
from tests.support.runners import plan, run_supervisor as run  # noqa: E402

PICK_A = ("pick", {"object": "a"})


class Answers:
    """A person at the terminal: one answer per question (None = no answer)."""

    def __init__(self, *answers):
        self.answers, self.asked = list(answers), []

    def ask(self, q):
        self.asked.append(q)
        return self.answers.pop(0) if self.answers else None

    def __call__(self, q):  # skill questions: no answer (not used here)
        from agent.ask import Answer
        return Answer("cancel", None, "")


# ------------------------------------------------------------------------------------------- 1. no loops
def test_a_gemma_that_always_observes_again_cannot_loop(tmp_path):
    """Live 2026-10-26 (place in the red tray): three "observe" replans in a row, nothing moved, budget used up."""
    t = GoalTools([plan(("observe", {}), ("locate", {"object": "zz"}))], ambiguous={"zz": []})
    t.decisions = [{"action": "replan", "steps": [{"do": "observe"}]} for _ in range(20)]
    r, ev = run(tmp_path, t)
    assert r["status"] != "COMPLETED"
    assert [s["capability"] for s in r["plan"]].count("observe") == 1  # never observed again into an unchanged scene
    assert sum(e["kind"] == "gemma_decision" for e in ev) <= MAX_CONSULTS


def test_a_gemma_that_keeps_replanning_without_progress_is_stopped_by_a_question(tmp_path):
    """Two changes of course with nothing changed between them: the master asks the user instead of a third."""
    t = GoalTools([plan(("observe", {}), ("locate", {"object": "zz"}))], ambiguous={"zz": []})
    t.decisions = [{"action": "replan", "steps": [{"do": "locate", "object": f"zz{k}"}]} for k in range(20)]
    person = Answers(None)
    r, ev = run(tmp_path, t, responder=person)
    assert person.asked and "not making progress" in person.asked[0]
    assert r["status"] == "FAILED" and "no answer to my question" in r["message"]
    assert sum(e["kind"] == "replan" and e["accepted"] for e in ev) == 1  # only the first change of course ran


def test_a_decomposition_cannot_contain_decompose():
    """Live 2026-10-26: every decomposition returned another decompose (steps 1 -> 4 -> 5 -> 7)."""
    bad = '{"steps": [{"worker": "planning", "do": "decompose", "goal": "g"}], "goal": [], "reason": "r"}'
    good = ('{"steps": [{"worker": "manipulation", "do": "pick_place", "object": "#1", "relation": "on", "target": "#2"}], '
            '"goal": [{"pred": "on", "args": ["#1", "#2"]}], "reason": "r"}')
    vlm = RecordedVLM([bad, good])
    p = Roles(vlm).decompose("g", "facts", [1, 2])
    assert p.steps[0].capability == "pick_place" and len(vlm.calls) == 2
    assert "cannot contain decompose" in vlm.calls[1]


def test_a_nested_decompose_that_gets_through_is_dropped_and_free_text_conditions_are_not_added(tmp_path):
    t = session([plan(("observe", {}), ("decompose", {"goal": "stack a on b"}), success=[{"pred": "on", "args": ["a", "b"]}])])
    sp = SubPlan.model_validate({"steps": [{"worker": "planning", "do": "decompose", "goal": "stack a on b"},
                                           {"worker": "manipulation", "do": "pick_place", "object": "#1", "relation": "on",
                                            "target": "#2"}],
                                 "goal": [{"pred": "on", "args": ["#1", "#2"]}, {"pred": "on", "args": ["#1", "#2"]},
                                          {"pred": "all of them are stacked", "args": []}], "reason": "r"})
    _decomposer(t, sp)
    r, ev = run_in(tmp_path, "1", t, instruction="stack a on b")
    assert [s["capability"] for s in r["plan"]].count("decompose") == 1
    assert r["status"] == "COMPLETED"
    assert [c["pred"] for c in r["conditions"]] == ["on", "on"]  # original + the decomposition's one, no duplicate
    assert any(e["kind"] == "plan_rejected" for e in ev)
    assert {e["reason"] for e in ev if e["kind"] == "goal_conditions_skipped"} == {"already a goal condition", "no evaluator"}


def test_continue_with_new_steps_when_none_remain_is_a_replan(tmp_path):
    """Live 2026-10-26: "continue" + steps with nothing planned was invalid twice and ended the task."""
    from agent.vlm_roles import Decision, check_decision
    d = Decision.model_validate({"action": "continue", "steps": [{"worker": "perception", "do": "observe"}]})
    check_decision(d, has_pending=False)  # valid: the supervisor takes it as a replan
    t = GoalTools([plan(("locate", {"object": "zz"}))], ambiguous={"zz": []})
    t.decisions = [d, {"action": "stop", "reason": "enough"}]
    r, ev = run(tmp_path, t)
    assert [s["capability"] for s in r["plan"]] == ["locate", "observe"]
    assert any(e["kind"] == "replan" and "taken as a replan" in (e.get("reason") or "") for e in ev)


# ----------------------------------------------------------------------------- 2. Gemma's answers read reliably
def test_the_first_json_object_is_read_whatever_follows_it():
    from agent.vlm_roles.roles import _json
    assert _json('```json\n{"action": "stop"}\n```\n{"action": "replan"}') == {"action": "stop"}  # "Extra data" before
    assert _json('Here: {"id": 4, "c": [1, 2]} hope it helps') == {"id": 4, "c": [1, 2]}


def test_a_bare_number_is_a_measured_object_id():
    """Live 2026-10-26: pick_place '4' while holding #4 was refused as unsafe (not recognised as the held object)."""
    assert ADAPT["pick_place"]({"object": "4", "relation": "in", "target": "15"})["object"]["text"] == "#4"
    assert ADAPT["place"]({"relation": "in", "target": "15"})["reference"]["text"] == "#15"
    assert Condition.model_validate("on(4, red tray)").args == ["#4", "red tray"]


# --------------------------------------------------------------------------------- 3. the held object
def test_place_needs_no_object_and_its_menu_has_no_template_slot():
    """Live 2026-10-26: Gemma copied "#<held id>" from the menu; place puts down the held object anyway."""
    menu = caps.planner_menu(caps.available())
    place = next(line for line in menu.splitlines() if line.strip().startswith("- place"))
    assert "<held" not in place and "relation=in" in place
    assert ADAPT["place"]({"relation": "in", "target": "the tray"})["object"]["text"] is None


def test_pick_place_of_the_held_object_names_the_operation_that_works(tmp_path):
    t = session([plan(PICK_A), plan(("pick_place", {"object": "#1", "relation": "in", "target": "b"}))])
    run_in(tmp_path, "1", t)
    t.decisions = [{"action": "stop", "reason": "r"}]
    r, ev = run_in(tmp_path, "2", t)
    happened = r["exchanges"][0]["happened"]
    assert "already in the gripper" in happened and "place relation=in target=b" in happened


def test_a_task_that_stops_while_holding_says_so(tmp_path):
    """Live 2026-10-26: four tasks in a row ended with a block in the gripper and no word about it."""
    t = session([plan(PICK_A), plan(("locate", {"object": "zz"}))], ambiguous={"zz": []})
    run_in(tmp_path, "1", t)
    t.decisions = [{"action": "stop", "reason": "cannot find it"}]
    r, _ = run_in(tmp_path, "2", t)
    assert r["status"] == "FAILED" and r["holding"] is not None
    assert f"still holding #{r['holding']}" in r["message"] and "where to put it" in r["message"]


# -------------------------------------------------------------------------------------- 4. "in" a container
def test_in_is_a_placement_relation_and_a_measurable_goal():
    from agent.goals import PREDICATES
    assert "in" in PLACE_RELATIONS and "in" in PREDICATES
    assert ADAPT["place"]({"relation": "in", "target": "#3"})["relation"] == "in"


def test_a_gemma_that_keeps_repeating_a_failed_step_is_stopped_by_a_question(tmp_path):
    """Live 2026-10-26: decompose failed, then Gemma asked for the same decompose four times; each was refused as a
    repeat until the budget ended the task without asking anyone. Now the second refusal in a row asks the user."""
    t = GoalTools([plan(("observe", {}), ("locate", {"object": "zz"}))], ambiguous={"zz": []})
    t.decisions = [{"action": "replan", "steps": [{"do": "locate", "object": "zz"}]} for _ in range(10)]
    person = Answers(None)
    r, ev = run(tmp_path, t, responder=person)
    assert person.asked and "not making progress" in person.asked[0]
    assert r["status"] == "FAILED" and sum(e["kind"] == "replan" and not e["accepted"] for e in ev) == 2


def test_the_users_own_words_in_template_brackets_are_their_words():
    """Live 2026-10-26: on(<yellow block>, <blue tray>) was rejected twice as a template placeholder."""
    good = ('{"goal": "g", "success": [{"pred": "in", "args": ["<yellow block>", "<blue tray>"]}], '
            '"steps": [{"worker": "manipulation", "do": "pick_place", "object": "<yellow block>", "relation": "in", '
            '"target": "<blue tray>"}], "reason": "r"}')
    vlm = RecordedVLM([good])
    p = Roles(vlm).plan_goal("put the yellow block in the blue tray", "menu")
    assert len(vlm.calls) == 1 and p.success[0].args == ["yellow block", "blue tray"]
    assert p.steps[0].args["object"]["text"] == "yellow block"
    bad = '{"goal": "g", "success": [{"pred": "holding", "args": ["<object_id>"]}], "steps": [{"do": "pick", "object": "<object_id>"}]}'
    vlm = RecordedVLM([bad, bad])
    with pytest.raises(Exception, match="placeholder"):
        Roles(vlm).plan_goal("put the yellow block in the blue tray", "menu")


def test_a_decomposition_is_not_offered_decompose(monkeypatch):
    """The cause of the nested decompositions: decompose was in the menu Gemma decomposes with."""
    import agent.capabilities.planning as planning
    seen = {}

    class R:
        def decompose(self, goal, facts, ids, context="", menu="", task_preds=()):
            seen["menu"] = menu
            return SubPlan(steps=[], goal=[], reason="nothing")
    from types import SimpleNamespace
    ctx = SimpleNamespace(tools=SimpleNamespace(roles=R(), held_object=None, world=SimpleNamespace(objects={})),
                          obs=SimpleNamespace(object_ids={}), log=lambda *a, **k: None)
    monkeypatch.setattr(planning, "_ensure_described", lambda c: None)
    planning.run_decompose(ctx, {"goal": "g"})
    assert "- decompose" not in seen["menu"] and "- pick_place" in seen["menu"]


def test_a_decomposition_is_offered_only_operations_that_advance_the_task(monkeypatch):
    """Live 2026-11-04 (six runs): offered observe, the decomposition planned only observe although the objects are
    already measured when it runs. Its menu has no sensing operation; the ones that change the task stay."""
    import re as _re
    import agent.capabilities.planning as planning
    seen = {}

    class R:
        def decompose(self, goal, facts, ids, context="", menu="", task_preds=()):
            seen["menu"] = menu
            return SubPlan(steps=[], goal=[], reason="nothing")
    from types import SimpleNamespace
    ctx = SimpleNamespace(tools=SimpleNamespace(roles=R(), held_object=None, world=SimpleNamespace(objects={})),
                          obs=SimpleNamespace(object_ids={}), log=lambda *a, **k: None)
    monkeypatch.setattr(planning, "_ensure_described", lambda c: None)
    planning.run_decompose(ctx, {"goal": "g"})
    offered = set(_re.findall(r"^  - (\w+):", seen["menu"], _re.M))
    assert offered.isdisjoint({"observe", "describe_scene", "locate", "inspect", "look", "answer", "explore", "decompose"})
    assert {"pick", "place", "pick_place", "return_home"} <= offered


def test_a_destination_is_never_the_held_object(tmp_path):
    """Live 2026-10-26: the held red cube, seen lifted in the gripper, was a candidate for "red tray"; the place step
    asked the user to choose between the cube and the tray. The held object is now excluded as a destination."""
    import inspect
    from agent import pick_place
    src = inspect.getsource(pick_place.PickPlaceWorkflow._resolve_one)
    assert "self.tools.held_object" in src and 'role == "reference"' in src
    t = session([plan(PICK_A), plan(("place", {"relation": "on", "target": "b"}))])
    run_in(tmp_path, "1", t)
    seen = []
    real = t.resolve

    def resolve(obs, text, exclude=(), choose=True):
        seen.append((text, set(exclude)))
        return real(obs, text, exclude=exclude) if exclude else real(obs, text)
    t.resolve = resolve
    run_in(tmp_path, "2", t)
    held_marks = [ex for text, ex in seen if text == "b"]
    assert held_marks and all(ex for ex in held_marks)  # the held object's mark was excluded every time


def test_a_plan_that_places_in_must_not_state_on_as_its_success():
    """Live 2026-10-26: plan "place relation=in red tray", success "on(red block, red tray)": the block landed in the
    tray (placement confirmed) and the goal check said, correctly, "not on top". Gemma now gets the plan back."""
    bad = ('{"goal": "g", "success": [{"pred": "on", "args": ["red block", "red tray"]}], "steps": ['
           '{"worker": "manipulation", "do": "pick", "object": "red block"}, '
           '{"worker": "manipulation", "do": "place", "relation": "in", "target": "red tray"}], "reason": "r"}')
    good = bad.replace('"pred": "on"', '"pred": "in"')
    vlm = RecordedVLM([bad, good])
    p = Roles(vlm).plan_goal("pick the red block and place it in the red tray", "menu")
    assert p.success[0].pred == "in" and len(vlm.calls) == 2
    # live 2026-10-26 (eighth and ninth checks): told only to make them agree, Gemma changed the STEP to "on" (twice);
    # the retry now quotes the request and names the criterion that matches the steps
    assert 'the user asked: "pick the red block and place it in the red tray"' in vlm.calls[1]
    assert "write the criterion as in(red block, red tray)" in vlm.calls[1]


# --------------------------------------------------------------- 5. understanding before planning (2026-10-26)
def test_gemma_asks_until_the_request_is_clear_then_plans_with_the_conversation(tmp_path):
    """The master converses before it plans: Gemma asks in its own words, the user answers in free text, and the plan
    is made from the request together with the conversation."""
    from agent.vlm_roles import Clarification
    t = GoalTools([plan(("observe", {}))])
    t.perceive("table")  # objects measured: there are candidates to ask about (nothing measured: no question)
    turns = [Clarification(status="ask", question="Which block do you mean?"),
             Clarification(status="clear", request="put the block that is in my gripper in the red tray")]
    seen = []

    def clarify(request, conversation, context=""):
        seen.append((request, list(conversation), context))
        return turns.pop(0)
    t.roles.clarify = clarify
    person = Answers("the one you are holding")
    r, ev = run(tmp_path, t, responder=person, instruction="place the block in red tray")
    assert person.asked == ["Which block do you mean?"]
    assert seen[1][1] == [("Which block do you mean?", "the one you are holding")]  # Gemma saw the answer
    planned = t.plan_calls[0]
    assert "the one you are holding" in planned and "put the block that is in my gripper in the red tray" in planned
    assert r["original_goal"]["conversation"] == [["Which block do you mean?", "the one you are holding"]]


def test_the_conversation_is_bounded_and_stop_cancels(tmp_path):
    from agent.supervisor import MAX_CLARIFY
    from agent.vlm_roles import Clarification
    t = GoalTools([plan(("observe", {}))])
    t.perceive("table")  # objects measured: there are candidates to ask about
    t.roles.clarify = lambda request, conversation, context="": Clarification(status="ask", question=f"q{len(conversation)}?")
    person = Answers(*["a"] * 10)
    run(tmp_path, t, responder=person)
    assert len(person.asked) == MAX_CLARIFY  # then it plans from what it knows
    t2 = GoalTools([plan(("observe", {}))])
    t2.perceive("table")
    t2.roles.clarify = lambda request, conversation, context="": Clarification(status="ask", question="which?")
    second = tmp_path / "second"
    second.mkdir()
    r, _ = run(second, t2, responder=Answers("stop"))
    assert r["status"] == "CANCELLED" and t2.plan_calls == []


def test_a_new_request_in_an_answer_becomes_the_goal(tmp_path):
    """Live 2026-10-26: answering "place the block in red tray" to "not making progress" was bent into the old goal
    ("pick the white block") and the task re-observed until its budget ran out."""
    t = GoalTools([plan(("locate", {"object": "zz"})), plan(("observe", {}))], ambiguous={"zz": []})
    t.decisions = [{"action": "new_goal", "request": "place the block in red tray", "reason": "the user asks for this now"}]
    r, ev = run(tmp_path, t, instruction="pick the white block")
    assert any(e["kind"] == "new_goal" and e["request"] == "place the block in red tray" for e in ev)
    assert t.plan_calls[-1].startswith("place the block in red tray")  # understood and planned afresh
    assert r["plan"][0]["status"] in ("failed", "replaced", "skipped") and r["plan"][-1]["capability"] == "observe"


def test_an_answer_in_words_ends_the_skill_question_and_reaches_the_supervisor():
    from agent.ask import Answer
    from agent.pick_place import PickPlaceWorkflow
    wf = PickPlaceWorkflow.__new__(PickPlaceWorkflow)
    out = wf._cancelled(Answer("text", raw="you already have block in your grasp"), "Which one?")
    assert out["next"] == "FAILED" and out["failure"]["cls"] == "answered"
    assert "you already have block in your grasp" in out["failure"]["reason"]


def test_a_plan_that_puts_the_object_down_must_not_state_holding_as_its_success():
    """Live 2026-10-26: success "holding(yellow block); in(yellow block, green tray)" for pick -> place. The block was
    placed and verified in the tray, and the task failed on holding (criteria are checked on the final state)."""
    bad = ('{"goal": "g", "success": [{"pred": "holding", "args": ["yellow block"]}, {"pred": "in", "args": ["yellow block", '
           '"green tray"]}], "steps": [{"worker": "manipulation", "do": "pick", "object": "yellow block"}, '
           '{"worker": "manipulation", "do": "place", "relation": "in", "target": "green tray"}], "reason": "r"}')
    good = bad.replace('{"pred": "holding", "args": ["yellow block"]}, ', "")
    vlm = RecordedVLM([bad, good])
    p = Roles(vlm).plan_goal("pick the yellow block and place it in the green tray", "menu")
    assert [c.pred for c in p.success] == ["in"] and len(vlm.calls) == 2
    assert "remove that criterion" in vlm.calls[1] and "holding(yellow block)" in vlm.calls[1]


def test_a_pick_alone_keeps_holding_as_its_success():
    pick = ('{"goal": "g", "success": [{"pred": "holding", "args": ["yellow block"]}], "steps": ['
            '{"worker": "manipulation", "do": "pick", "object": "yellow block"}], "reason": "r"}')
    vlm = RecordedVLM([pick])
    assert Roles(vlm).plan_goal("pick the yellow block", "menu").success[0].pred == "holding" and len(vlm.calls) == 1


def test_a_replan_that_repeats_the_planned_steps_is_not_a_change_of_course(tmp_path):
    """Live 2026-10-26: Gemma "replanned" place-in-tray over the pending identical place: 1 of 3 changes used for nothing."""
    t = GoalTools([plan(("observe", {}), ("locate", {"object": "a"}))])
    t.continue_on_success = False
    t.decisions = [{"action": "replan", "steps": [{"do": "locate", "object": "a"}]}, {"action": "stop", "reason": "enough"}]
    r, ev = run(tmp_path, t)
    assert any(e["kind"] == "replan" and "same steps as planned" in e.get("reason", "") for e in ev)
    assert [s["capability"] for s in r["plan"]] == ["observe", "locate"]  # the planned step ran; nothing added
    assert r["replans"] <= 1  # only the final stop counted
