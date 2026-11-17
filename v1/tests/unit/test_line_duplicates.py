"""Labels that occur at most once in a line-protocol answer (live 2026-11-04, out/demo/2026-11-04/live_150302/01):
Gemma thought aloud and wrote "ACTION: finish ... Wait ... ACTION: replan"; the parser joined them into "finish
replan" and the retry got only the schema's list of actions. A second ACTION, QUESTION, REQUEST or GOAL line is now
rejected by name; nothing is joined and no value is chosen in Python."""

import pytest

pytest.importorskip("pydantic")

from agent.vlm_roles import RecordedVLM, Roles, VLMFailure  # noqa: E402
from agent.vlm_roles.roles import parse_lines  # noqa: E402

# the live answer, cut down: a first decision, reasoning, then a second decision
THOUGHT_ALOUD = """ACTION: finish
REASON: the observation is done, so I will assume the next step is to execute the pick.

Wait, the goal is not reached yet. I must execute the pick.

ACTION: continue
STEP: manipulation | pick | object=#6"""
FINAL = "ACTION: continue\nSTEP: manipulation | pick | object=#6\nREASON: #6 is the object to pick"
FEEDBACK = ("line 6: a second ACTION line was written. An answer has exactly one ACTION. Give only your final decision "
            "and do not write corrections or reasoning between the format lines")


def test_two_action_lines_are_rejected_with_the_duplicate_label_feedback():
    with pytest.raises(ValueError) as exc:
        parse_lines("decide", THOUGHT_ALOUD)
    assert str(exc.value) == FEEDBACK


def test_the_one_retry_recovers_with_a_single_action():
    vlm = RecordedVLM([THOUGHT_ALOUD, FINAL])
    d = Roles(vlm).decide("g", "p", "e", "menu", has_pending=True)
    assert FEEDBACK in vlm.calls[1]  # the retry says exactly what is wrong
    assert d.action == "continue" and [(s.do, s.args["object"]["text"]) for s in d.steps] == [("pick", "#6")]


def test_a_second_action_on_the_retry_fails_the_decision():
    with pytest.raises(VLMFailure, match="a second ACTION line was written"):
        Roles(RecordedVLM([THOUGHT_ALOUD, THOUGHT_ALOUD])).decide("g", "p", "e", "menu", has_pending=True)


@pytest.mark.parametrize("role, answer, label", [
    ("decide", "ACTION: ask_user\nQUESTION: which one?\nQUESTION: the left or the right one?\nREASON: r", "QUESTION"),
    ("decide", "ACTION: new_goal\nREQUEST: put it down\nREQUEST: put it in the tray\nREASON: r", "REQUEST"),
    ("plan", "GOAL: holding the block\nGOAL: the block is held\nSUCCESS: holding(block)\nSTEP: perception | observe", "GOAL"),
])
def test_duplicate_question_request_and_goal_are_rejected(role, answer, label):
    with pytest.raises(ValueError, match=f"a second {label} line was written. An answer has exactly one {label}."):
        parse_lines(role, answer)


def test_repeated_labels_that_may_repeat_are_unchanged():
    _, out = parse_lines("decide", "ACTION: replan\nSTEP: manipulation | pick | object=#6\n"
                                   "STEP: manipulation | return_home\nREASON: r")
    assert [s["do"] for s in out["steps"]] == ["pick", "return_home"]
    _, out = parse_lines("plan", "GOAL: g\nSUCCESS: holding(#6)\nSUCCESS: at_home()\nSTEP: perception | observe")
    assert [c["pred"] for c in out["success"]] == ["holding", "at_home"]
