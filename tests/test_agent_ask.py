"""Clarification questions (agent/ask.py + PickPlaceWorkflow): regression tests for the bug where the
interactive demo answered its own question with a scripted "cancel" (2026-10-14).

Needs pydantic + langgraph (.venv-seg); skipped where they are missing.
"""

import io
import json
import os

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langgraph")

from agent.ask import Answer, Candidate, LineReader, Question, ScriptedResponder, TerminalResponder, parse_answer  # noqa: E402
from tests.test_agent import StubTools, run  # noqa: E402

Q = Question("choose", "Which one?", "object", [
    Candidate(1, "yellow cube", (550, 150), 0.57, (40, 40, 40)),
    Candidate(3, "gold yellow cube", (0, 850), 0.85, (50, 50, 40)),
    Candidate(7, "olive cube", (0, 430), 0.43, (45, 45, 45))])


def terminal(text: str, timeout_s: float = 5.0) -> tuple[TerminalResponder, io.StringIO]:
    out = io.StringIO()
    return TerminalResponder(LineReader(io.StringIO(text)), out, timeout_s), out


# ------------------------------------------------------------------------ parsing a person's answer
@pytest.mark.parametrize("text, mark", [("3", 3), ("#7", 7), ("object 7", 7), ("I mean 1", 1), ("the gold one", 3),
                                        ("olive", 7), ("nearest", 7), ("the farthest one", 3), ("far yellow", 3)])
def test_valid_answers_select_one_candidate(text, mark):
    a, why = parse_answer(text, Q)
    assert a is not None and a.kind == "choice" and a.mark == mark, why


@pytest.mark.parametrize("text, reason", [("9", "not one of"), ("1 or 3", "several numbers"), ("yellow", "fits several"),
                                          ("purple", "no candidate"), ("", "empty"), ("hmm", "no candidate")])
def test_invalid_answers_are_explained(text, reason):
    a, why = parse_answer(text, Q)
    assert a is None and reason in why


@pytest.mark.parametrize("text", ["cancel", "Stop please", "abort", "quit"])
def test_only_explicit_words_cancel(text):
    assert parse_answer(text, Q)[0].kind == "cancel"


# ------------------------------------------------------------------------- the terminal responder
def test_terminal_shows_ids_descriptions_and_positions_then_waits_for_input():
    resp, out = terminal("3\n")
    a = resp(Q)
    shown = out.getvalue()
    assert a.kind == "choice" and a.mark == 3 and a.source == "terminal"
    for c in Q.candidates:
        assert f"{c.mark}: {c.description}" in shown and f"x {c.centre_mm[0]:+.0f} mm" in shown
    assert "answer>" in shown


def test_terminal_repeats_the_question_after_an_invalid_answer():
    resp, out = terminal("9\nyellow\nthe gold one\n")
    a = resp(Q)
    assert a.mark == 3 and [r["input"] for r in a.attempts] == ["9", "yellow"]
    assert out.getvalue().count("Please answer again") == 2


def test_terminal_eof_cancels_by_policy():
    resp, _ = terminal("")
    assert resp(Q).kind == "eof"


def test_terminal_timeout_cancels_by_policy():
    r, w = os.pipe()  # nothing is ever written
    try:
        resp = TerminalResponder(LineReader(os.fdopen(r)), io.StringIO(), timeout_s=0.2)
        assert resp(Q).kind == "timeout"
    finally:
        os.close(w)


def test_scripted_answers_are_marked_scripted():
    a = ScriptedResponder(["cancel"])(Q)
    assert a.kind == "cancel" and a.source == "scripted"


# ---------------------------------------------------------------------- end to end in the supervisor
def test_supervisor_needs_an_explicit_responder(tmp_path):
    from agent.eventlog import EventLog
    from agent.pick_place import PickPlaceWorkflow
    with pytest.raises(TypeError):
        PickPlaceWorkflow(StubTools(), EventLog(tmp_path / "e.jsonl"), tmp_path)  # no silent default answer


def test_object_choice_continues_the_same_task(tmp_path):
    t = StubTools(ambiguous={"a": [1, 3]})
    resp, out = terminal("3\n")
    r, ev = run(tmp_path, t, responder=resp)
    assert r["status"] == "COMPLETED"
    assert t.planned[0][0] == t.ids[3]  # the user's object is the one moved
    assert (3, "a") not in t.confirmed  # the person's choice is not re-judged by the VLM
    asks = [e for e in ev if e["kind"] == "ask_user"]
    assert len(asks) == 1 and asks[0]["source"] == "terminal" and asks[0]["mark"] == 3 and asks[0]["role"] == "object"
    states = [e["state"] for e in ev if e["kind"] == "state"]
    assert states.count("OBSERVING") == 1  # not restarted


def test_destination_choice_continues_the_same_task(tmp_path):
    t = StubTools(ambiguous={"b": [2, 3]})
    resp, _ = terminal("2\n")
    r, ev = run(tmp_path, t, responder=resp)
    assert r["status"] == "COMPLETED"
    assert t.planned[0][1] == t.ids[2]
    assert [e["role"] for e in ev if e["kind"] == "ask_user"] == ["reference"]


def test_both_object_and_destination_ambiguous(tmp_path):
    t = StubTools(ambiguous={"a": [1, 3], "b": [2, 3]})
    resp, _ = terminal("nearest\n2\n")
    r, ev = run(tmp_path, t, responder=resp)
    assert r["status"] == "COMPLETED"
    assert [(e["role"], e["mark"]) for e in ev if e["kind"] == "ask_user"] == [("object", 1), ("reference", 2)]


def test_invalid_then_valid_selection(tmp_path):
    t = StubTools(ambiguous={"a": [1, 3]})
    resp, _ = terminal("5\nthe gold one\n")
    r, ev = run(tmp_path, t, responder=resp)
    ask = [e for e in ev if e["kind"] == "ask_user"][0]
    assert r["status"] == "COMPLETED" and ask["mark"] == 3 and ask["rejected"][0]["input"] == "5"


def test_explicit_cancel(tmp_path):
    resp, _ = terminal("cancel\n")
    r, _ = run(tmp_path, StubTools(ambiguous={"a": [1, 3]}), responder=resp)
    assert r["status"] == "CANCELLED" and "you answered 'cancel'" in r["failure"]["reason"]
    assert r["failure"]["answer_source"] == "terminal"


def test_eof_while_asking_cancels(tmp_path):
    resp, _ = terminal("")
    r, _ = run(tmp_path, StubTools(ambiguous={"b": [2, 3]}), responder=resp)
    assert r["status"] == "CANCELLED" and "end of input" in r["failure"]["reason"]


def test_question_log_is_in_the_report(tmp_path):
    resp, _ = terminal("1\n")
    r, _ = run(tmp_path, StubTools(ambiguous={"a": [1, 3]}), responder=resp)
    assert r["questions"] and r["questions"][0]["answer"] == "choice" and r["questions"][0]["raw"] == "1"
    json.dumps(r, default=str)
