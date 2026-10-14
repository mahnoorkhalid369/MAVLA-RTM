"""Goal supervisor (agent/supervisor.py), capabilities and planner validation, with mocked tools (no sim, no GPU)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langgraph")

from agent.ask import ScriptedResponder  # noqa: E402
from agent.eventlog import EventLog  # noqa: E402
from agent.supervisor import GoalSupervisor  # noqa: E402
from agent.vlm_roles import GoalPlan, RecordedVLM, Roles, SceneAnswer, VLMFailure  # noqa: E402
from tests.test_agent import StubTools  # noqa: E402

PP = {"object": "a", "relation": "on", "target": "b"}
PP2 = {**PP, "object": "c"}


def plan(*steps, reason=""):
    return GoalPlan(goal="g", steps=[{"do": c, **a} for c, a in steps], success="s", reason=reason)


class GoalTools(StubTools):
    """StubTools + what the goal supervisor needs; plans are scripted."""

    def __init__(self, plans, answer=None, **kw):
        super().__init__(**kw)
        self.plans, self.plan_calls, self.perceived, self.parks = list(plans), [], 0, 0
        self.motion_count, self.view_cleared_at = 0, -1
        tools = self

        def plan_goal(text, menu):
            tools.plan_calls.append(text)
            p = tools.plans.pop(0)
            if isinstance(p, Exception):
                raise p
            return p
        self.roles = SimpleNamespace(plan_goal=plan_goal, answer=lambda q, facts, ids: answer or SceneAnswer(answer="two", object_ids=[1]))

    def execute(self, steps, label):
        self.motion_count += 1
        return super().execute(steps, label)

    def park_for_view(self, xy):
        self.parks += 1
        self.view_cleared_at = self.motion_count
        return {"ok": True}

    def perceive(self, camera):
        self.perceived += 1
        o = super().perceive(camera)
        o.camera, o.scores, o.excluded = "table", {k: 0.9 for k in o.marks}, []
        o.taken_at, o.motion_count = 0.0, self.motion_count
        o.cleared_view = self.view_cleared_at == self.motion_count
        return o

    def is_fresh(self, obs, need_clear_view=False):
        return obs is not None and obs.motion_count == self.motion_count and (obs.cleared_view or not need_clear_view)

    def view_blockage(self, q=None):
        return 1, 36

    def obstacles(self):
        return []

    def lexical_candidates(self, obs, text):
        return []

    def resolve(self, obs, text):
        from agent.vlm_roles import Resolution
        if text in self.ambiguous:
            return Resolution(id=None, ambiguous=True, candidates=self.ambiguous[text])
        return Resolution(id={"a": 1, "b": 2, "c": 3}.get(text, 2))


def run(tmp_path, tools, responder=None, instruction="do it"):
    log = EventLog(tmp_path / "events.jsonl")
    r = GoalSupervisor(tools, log, tmp_path, responder or ScriptedResponder(["cancel"])).run(instruction)
    log.close()
    return r, [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]


def test_pick_and_place_goal_delegates_to_the_validated_skill(tmp_path):
    t = GoalTools([plan(("pick_place", PP))])
    r, ev = run(tmp_path, t)
    assert r["status"] == "COMPLETED" and len(t.plan_calls) == 1  # one planner call replaces the old parse call
    assert t.home_calls == 1 and "back at HOME" in r["message"]
    assert [v["check"] for v in r["verifications"]] == ["contact", "contact_after_carry", "placement", "home"]
    goal_states = [e["state"] for e in ev if e["kind"] == "state" and e.get("level") == "goal"]
    assert goal_states[:3] == ["INTERPRETING", "DISPATCHING", "MANIPULATING"] and goal_states[-1] == "COMPLETED"
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
    assert r["status"] == "UNSUPPORTED" and "I cannot sing" in r["message"] and "describe the table" in r["message"]
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


def test_not_visible_target_triggers_one_exploration_then_locate_again(tmp_path, explore_stub):
    t = GoalTools([plan(("observe", {}), ("locate", {"object": "zz"})), plan(("observe", {}), ("locate", {"object": "zz"}))],
                  ambiguous={"zz": []})
    r, ev = run(tmp_path, t)
    assert [s["capability"] for s in r["plan"]] == ["observe", "locate", "explore", "locate"]
    assert [s["status"] for s in r["plan"]][:3] == ["done", "replaced", "done"]
    assert len(explore_stub) == 1 and any(e["kind"] == "subtask" for e in ev)
    assert r["status"] == "FAILED" and r["outcome"] == "insufficient_info"  # exploring is not finding


def test_found_after_exploration_completes_the_goal(tmp_path, monkeypatch):
    from agent import capabilities as caps
    monkeypatch.setattr(caps, "run_explore", lambda ctx, args: caps.StepResult("completed", "looked behind object 2"))
    real_locate, n = caps.run_locate, []

    def locate(ctx, args):
        n.append(1)
        if len(n) == 1:
            return caps.StepResult("insufficient_info", "I do not see 'zz' from the current view")
        return caps.StepResult("completed", "'zz' found with the wrist camera at x +100 mm", {"mark": 9, "camera": "wrist"})
    monkeypatch.setattr(caps, "run_locate", locate)
    t = GoalTools([plan(("observe", {}), ("locate", {"object": "zz"}))])
    r, _ = run(tmp_path, t)
    assert r["status"] == "COMPLETED" and r["message"].startswith("looked behind") is False and "found with the wrist camera" in r["message"]


def test_multi_step_plan_returns_home_once_at_the_end(tmp_path):
    t = GoalTools([plan(("pick_place", PP), ("pick_place", PP2))])
    r, ev = run(tmp_path, t)
    assert r["status"] == "COMPLETED" and t.home_calls == 1
    starts = [e for e in ev if e["kind"] == "skill" and e.get("event") == "start"]
    assert [s["final_home"] for s in starts] == [False, True]


def test_observe_then_manipulate_reuses_the_cleared_view_observation(tmp_path):
    t = GoalTools([plan(("observe", {}), ("pick_place", PP))])
    r, ev = run(tmp_path, t)
    assert r["status"] == "COMPLETED" and t.parks >= 1
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


def test_recoverable_failure_replans_once_and_detects_repetition(tmp_path):
    t = GoalTools([plan(("pick_place", PP)), plan(("pick_place", PP))], placement=(False,))
    r, ev = run(tmp_path, t)
    assert r["status"] == "FAILED" and r["replans"] == 1 and len(t.plan_calls) == 2
    assert "Replanning offered nothing different" in r["message"]
    assert [e["accepted"] for e in ev if e["kind"] == "replan"] == [False]


def test_replanned_alternative_can_complete_the_goal(tmp_path):
    t = GoalTools([plan(("answer", {"question": "q"})), plan(("observe", {}), ("describe_scene", {}))])
    t.roles.answer = lambda q, f, ids: (_ for _ in ()).throw(VLMFailure("bad"))  # first plan fails
    r, _ = run(tmp_path, t)
    assert r["status"] == "COMPLETED" and r["replans"] == 1
    assert [s["status"] for s in r["plan"]] == ["replaced", "done", "done"]


def test_planner_failure_is_a_clear_failure(tmp_path):
    t = GoalTools([VLMFailure("plan: invalid answer after one retry")])
    r, _ = run(tmp_path, t)
    assert r["status"] == "FAILED" and "could not turn the instruction into steps" in r["message"]


# --------------------------------------------------------------------------- planner validation (real Roles)
def test_planner_rejects_unknown_capabilities_and_bad_arguments():
    good = json.dumps({"goal": "g", "steps": [{"do": "pick_place", "object": "a", "relation": "on", "target": "b"}]})
    for bad in ('{"steps": [{"do": "dance"}]}',
                '{"steps": [{"do": "pick_place", "object": "a"}]}',
                '{"steps": [{"do": "answer"}]}',
                '{"steps": [{"do": "locate"}]}',
                json.dumps({"steps": [{"do": "pick_place", "object": "a", "relation": "on", "target": "b",
                                       "object_relation": "on", "object_anchor": "b"}]})):
        vlm = RecordedVLM([bad, good])
        p = Roles(vlm).plan_goal("x", "menu")
        assert p.steps[0].capability == "pick_place" and len(vlm.calls) == 2  # one retry with the error
        assert p.steps[0].args == {"object": {"text": "a", "relation": None, "anchor": None}, "relation": "on",
                                   "reference": {"text": "b", "relation": None, "anchor": None}, "frame": "robot"}
    with pytest.raises(VLMFailure):
        Roles(RecordedVLM(['{"steps": [{"do": "rm -rf"}]}'] * 2)).plan_goal("x", "menu")


def test_locate_finds_confirms(tmp_path):
    t = GoalTools([plan(("observe", {}), ("locate", {"object": "a"}))])
    r, ev = run(tmp_path, t)
    assert r["status"] == "COMPLETED" and "is object 1" in r["message"] and t.executed == []


def test_planner_menu_lists_only_registry_capabilities():
    from agent.capabilities import SPECS, planner_menu
    menu = planner_menu()
    assert all(f"- {n}:" in menu for n in SPECS) and all(sp.implemented for sp in SPECS.values())
    assert {s.safety for s in SPECS.values()} == {"read_only", "motion"}


def test_plan_completion_rules_are_generic_and_logged(tmp_path):
    t = GoalTools([plan(("observe", {}))])  # "take a look": observation nobody uses -> reported
    r, ev = run(tmp_path, t, instruction="Take a look at your surroundings.")
    assert r["status"] == "COMPLETED" and [s["capability"] for s in r["plan"]] == ["observe", "describe_scene"]
    assert any(e["kind"] == "plan_completed" for e in ev)
    t = GoalTools([plan(("observe", {}))])
    (tmp_path / "q").mkdir()
    r, _ = run(tmp_path / "q", t, instruction="How many cubes are there?")  # a question gets an answer step
    assert [s["capability"] for s in r["plan"]] == ["observe", "answer"] and "not verified" in r["message"]


def test_locate_reports_not_visible_when_no_description_matches(tmp_path, explore_stub):
    t = GoalTools([plan(("observe", {}), ("locate", {"object": "banana"})), plan(("observe", {}), ("locate", {"object": "banana"}))],
                  ambiguous={"banana": [1, 2, 3]})  # Gemma lists everything when nothing fits
    r, _ = run(tmp_path, t)
    assert r["status"] == "FAILED" and r["outcome"] == "insufficient_info" and "do not see 'banana'" in r["message"]


def test_result_comes_before_the_observation_summary(tmp_path):
    t = GoalTools([plan(("observe", {}), ("describe_scene", {}))])
    r, _ = run(tmp_path, t)
    assert r["message"].startswith("I see 3 objects") and r["message"].rstrip().endswith("workspace points from the camera)")


def test_counting_answer_must_match_the_cited_objects():
    vlm = RecordedVLM(['{"answer": "11", "object_ids": [1, 2, 3]}', '{"answer": "3", "object_ids": [1, 2, 3]}'])
    a = Roles(vlm).answer("how many?", "facts", [1, 2, 3, 4])
    assert a.answer == "3" and "count again" in vlm.calls[1]


def test_replan_that_drops_the_objective_is_rejected(tmp_path, explore_stub):
    """Regression (live 18:48): 'find the banana' -> locate failed (also after exploring) -> a replan [observe]
    must not complete the goal."""
    t = GoalTools([plan(("observe", {}), ("locate", {"object": "banana"})), plan(("observe", {}))],
                  ambiguous={"banana": [1, 2, 3]})
    r, ev = run(tmp_path, t)
    assert r["status"] == "FAILED" and "do not see 'banana'" in r["message"]
    assert [e["reason"] for e in ev if e["kind"] == "replan"] == ["the new plan no longer pursues the failed step (locate)"]


def test_explore_at_the_end_of_a_plan_gets_a_locate(tmp_path, explore_stub):
    t = GoalTools([plan(("explore", {"object": "thing", "near": "b"})), plan(("observe", {}))], ambiguous={"thing": []})
    r, ev = run(tmp_path, t)
    assert [s["capability"] for s in r["plan"]] == ["explore", "locate"] and r["status"] != "COMPLETED"
    assert any(e["kind"] == "plan_completed" for e in ev)
