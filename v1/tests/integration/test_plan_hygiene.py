"""Gemma's plans and decisions (agent/vlm_roles/roles.py, agent/capabilities/specs.py): a template slot copied from a
prompt is rejected with a reason (one retry), and an operation that already failed in this task is marked in the menu
Gemma decides from. Live 2026-10-21: "#<held id>", "<object_id>", "<purple block>" in plans and criteria, and
decompose chosen again after each failed decompose. Scripted Gemma answers (no sim, no GPU)."""

import json

import pytest

pytest.importorskip("pydantic")

from agent.capabilities.specs import planner_menu  # noqa: E402
from agent.vlm_roles import RecordedVLM, Roles  # noqa: E402
from agent.vlm_roles.roles import check_placeholders  # noqa: E402


@pytest.mark.parametrize("value", ["#<held id>", "<object_id>", "<purple block>", {"object": {"text": "<x>"}}])
def test_template_slots_are_rejected(value):
    with pytest.raises(ValueError, match="template placeholder"):
        check_placeholders(value)


@pytest.mark.parametrize("value", ["purple block", "#4", "the block on your right", {"relation": "on", "target": "red block"}])
def test_real_words_and_ids_pass(value):
    check_placeholders(value)


def test_a_plan_with_a_slot_gets_one_retry_with_the_reason():
    bad = {"goal": "g", "success": [{"pred": "on", "args": ["#4", "<purple block>"]}],
           "steps": [{"worker": "manipulation", "do": "place", "object": "#<held id>", "relation": "on", "target": "<purple block>"}]}
    good = {"goal": "g", "success": [{"pred": "on", "args": ["#4", "other purple block"]}],
            "steps": [{"worker": "manipulation", "do": "place", "object": "#4", "relation": "on", "target": "other purple block"}]}
    vlm = RecordedVLM([json.dumps(bad), json.dumps(good)])
    p = Roles(vlm).plan_goal("place it on other purple block", planner_menu())
    assert p.steps[0].args["object"]["text"] == "#4" and "template placeholder" in vlm.calls[1]


def test_a_slot_in_the_success_criteria_alone_is_rejected():
    bad = {"goal": "g", "success": [{"pred": "holding", "args": ["#<object_id>"]}],
           "steps": [{"worker": "manipulation", "do": "pick", "object": "purple block"}]}
    good = dict(bad, success=[{"pred": "holding", "args": ["purple block"]}])
    vlm = RecordedVLM([json.dumps(bad), json.dumps(good)])
    Roles(vlm).plan_goal("pick purple block", planner_menu())
    assert "success criterion holding" in vlm.calls[1]


def test_the_menu_marks_an_operation_that_failed_in_this_task():
    menu = planner_menu(failed={"decompose": "2 time(s), last: invalid answer after one retry"})
    line = next(x for x in menu.splitlines() if x.strip().startswith("- decompose"))
    assert "FAILED in this task: 2 time(s)" in line and "prefer another operation, ask_user or stop" in line
    assert "FAILED" not in next(x for x in menu.splitlines() if x.strip().startswith("- pick"))
    assert "FAILED" not in planner_menu()
