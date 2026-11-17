"""A goal already true before anything runs (live 2026-11-04, out/demo/2026-11-04/live_144519/01): "pick the yellow
block" was planned with SUCCESS not_holding(); the gripper was empty, so the goal check passed after an observation
and the task reported COMPLETED. A request for a change now gets its plan back on the one retry when every success
condition is already measurably true, and an action whose goal was true before it started is never reported as
COMPLETED unless a step changed an object. Which condition to write stays Gemma's."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pydantic")

from agent.vlm_roles import RecordedVLM, Roles  # noqa: E402
from tests.support.fakes import GoalTools  # noqa: E402
from tests.support.runners import plan, run_supervisor  # noqa: E402

NOT_HOLDING = "GOAL: the block is grasped\nSUCCESS: not_holding()\nSTEP: perception | observe\nREASON: look first"
HOLDING = "GOAL: the block is grasped\nSUCCESS: holding(yellow block)\nSTEP: perception | observe\nREASON: look first"


def _tools(plans=(), kind="action"):
    t = GoalTools(list(plans))
    t.bridge = SimpleNamespace(q=lambda: np.zeros(6), finger=lambda: 0.0)  # an open, empty gripper (measured)
    t.request_kind = kind
    return t


def _gemma(t, answers):
    vlm = RecordedVLM(answers)
    t.roles.plan_goal = Roles(vlm).plan_goal  # the real planner role, its checks and its one retry
    return vlm


def test_an_action_whose_goal_is_already_true_is_sent_back_and_gemma_restates_it(tmp_path):
    t = _tools()
    vlm = _gemma(t, [NOT_HOLDING, HOLDING])
    r, _ = run_supervisor(tmp_path, t, instruction="pick the yellow block")
    assert "Your SUCCESS conditions are already true before the requested action is performed." in vlm.calls[1]
    assert "State what measurable conditions must be true after the request is completed." in vlm.calls[1]
    assert "holding(" not in vlm.calls[1].split("\n\n")[-1]  # the feedback names no condition: Gemma chooses
    assert [(c["pred"], c["args"]) for c in r["conditions"]] == [("holding", ["yellow block"])]


def test_a_goal_that_cannot_be_decided_yet_is_allowed(tmp_path):
    t = _tools()
    vlm = _gemma(t, [HOLDING])
    r, _ = run_supervisor(tmp_path, t, instruction="pick the yellow block")
    assert len(vlm.calls) == 1  # nothing measured yet: holding(yellow block) is unknown, not true
    assert [c["pred"] for c in r["conditions"]] == ["holding"]


def test_a_check_request_already_true_is_not_rejected_and_completes(tmp_path):
    t = _tools([plan(("observe", {}), success=[{"pred": "not_holding", "args": []}])], kind="description")
    r, _ = run_supervisor(tmp_path, t, instruction="check that your gripper is empty")
    assert r["status"] == "COMPLETED" and [c["pred"] for c in r["conditions"]] == ["not_holding"]


def test_an_action_goal_true_from_the_start_with_nothing_changed_is_not_completed(tmp_path):
    # the scripted planner skips the plan check (as if it had got through): the final check still holds
    t = _tools([plan(("observe", {}), success=[{"pred": "not_holding", "args": []}])])
    r, _ = run_supervisor(tmp_path, t, instruction="pick the yellow block")
    assert r["status"] == "FAILED" and t.executed == []
    assert "were already true before I did anything, and no step changed an object" in r["message"]
    assert "I did not report success." in r["message"]
