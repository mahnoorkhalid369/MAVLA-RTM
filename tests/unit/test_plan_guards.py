"""Structural guards on Gemma's plans (live 2026-11-04): malformed conditions, observe-only decompositions, unknown
#ids. They say what is wrong; which operation to use stays Gemma's choice."""

import pytest

pytest.importorskip("pydantic")

from agent.goals import GoalEvaluator  # noqa: E402
from agent.vlm_roles import RecordedVLM, Roles, VLMFailure  # noqa: E402

PICK_PLACE = '{"steps": [{"worker": "manipulation", "do": "pick_place", "object": "#9", "relation": "in", "target": "#1"}], "goal": [{"pred": "in", "args": ["#9", "#1"]}], "reason": "r"}'
OBSERVE_ONLY = '{"steps": [{"worker": "perception", "do": "observe"}], "goal": [{"pred": "in", "args": ["#9", "#1"]}], "reason": "look first"}'


def _decompose(answers, **kw):
    vlm = RecordedVLM(answers)
    return Roles(vlm).decompose("put a block in the tray", "#1: tray\n#9: block", [1, 9], **kw), vlm


def test_an_observe_only_decomposition_is_rejected_and_gemma_chooses_again():
    sp, vlm = _decompose([OBSERVE_ONLY, PICK_PLACE])
    assert [st.do for st in sp.steps] == ["pick_place"]
    assert "none of them can change an object" in vlm.calls[1] and "pick_place" in vlm.calls[1]


def test_the_task_criteria_count_when_the_decomposition_states_none():
    bare = '{"steps": [{"worker": "perception", "do": "observe"}], "goal": [], "reason": "r"}'
    with pytest.raises(VLMFailure):
        _decompose([bare, bare], task_preds=["holding", "in"])
    sp, _ = _decompose([bare])  # a goal with no object-state condition may only sense
    assert [st.do for st in sp.steps] == ["observe"]


def test_a_condition_without_its_arguments_is_rejected():
    bad = PICK_PLACE.replace('"goal": [', '"goal": [{"pred": "holding", "args": []}, ')
    sp, vlm = _decompose([bad, PICK_PLACE])
    assert "holding() is malformed" in vlm.calls[1] and len(sp.goal) == 1


def test_an_unknown_reference_is_answered_with_the_ids_there_are():
    bad = PICK_PLACE.replace('"#9", "relation"', '"#<id_of_block>", "relation"')
    _, vlm = _decompose([bad, PICK_PLACE])
    assert "use only these #ids: #1, #9" in vlm.calls[1]


def test_a_malformed_condition_is_unknown_not_a_crash():
    r = GoalEvaluator(tools=None, state=None).evaluate({"pred": "holding", "args": []}, {})
    assert r.satisfied is None and "malformed" in r.evidence

