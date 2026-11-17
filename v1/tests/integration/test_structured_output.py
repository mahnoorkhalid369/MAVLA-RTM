"""Gemma's structured answers and the task's authority (live 2026-11-04, out/demo/2026-11-04/live_052445):
  - a plan that was sensible but not valid JSON ("{" closed with "]") got only "Expecting ',' delimiter: line 6
    column 3" on its retry and failed twice the same way. The retry now says where and what is structurally wrong;
    nothing in the answer is repaired by Python, and every stage (raw, parser input, parsed, normalized) is logged.
  - the clarification answer "do you see any white block in arrangement mat?" reclassified "pick the white block" as a
    description and took away the permission to pick. Authority now comes from the original instruction only."""

import json

import pytest

pytest.importorskip("pydantic")

from agent.ask import ScriptedResponder  # noqa: E402
from agent.eventlog import EventLog  # noqa: E402
from agent.supervisor import GoalSupervisor  # noqa: E402
from agent.vlm_roles import Clarification, RecordedVLM, RequestKind, Roles, VLMFailure  # noqa: E402
from tests.support.fakes import GoalTools  # noqa: E402
from tests.support.runners import plan  # noqa: E402

# Gemma's first answer to "place the tall pink block in green tray" (live_052445/01), verbatim
MALFORMED = '''```json
{
  "goal": "The tall pink block from the pickup area is placed in the green tray.",
  "success": [
    {
      "pred": "on(<tall pink block>, <green tray>)"
  ],
  "steps": [
    {
      "worker": "perception worker",
      "do": "observe",
      "args": {}
    }
  ],
  "reason": "observe first"
}
```'''
FIXED = MALFORMED.replace('"pred": "on(<tall pink block>, <green tray>)"\n  ],', '"pred": "on(<tall pink block>, <green tray>)"\n    }\n  ],')


def _plan(answers):
    events = []
    vlm = RecordedVLM(answers)
    roles = Roles(vlm, log=lambda kind, **e: events.append({"kind": kind, **e}))
    return roles, vlm, events


def test_malformed_json_gets_structural_feedback_and_every_stage_is_logged():
    roles, vlm, events = _plan([MALFORMED, FIXED])
    p = roles.plan_goal("place the tall pink block in green tray", "menu")
    bad, good = [e for e in events if e["kind"] == "vlm"]
    # the raw answer and what the parser read, as they were
    assert bad["answer"] == MALFORMED and bad["parser_input"].startswith("{\n  \"goal\"") and bad["valid"] is False
    pe = bad["parse_error"]
    assert (pe["message"], pe["line"], pe["column"]) == ("Expecting ',' delimiter", 6, 3) and pe["offset"] == 160
    assert ">>   ]," in pe["context"] and '"pred": "on(<tall pink block>, <green tray>)"' in pe["context"]
    assert pe["mismatch"] == ('A JSON object opened with "{" at line 4 column 5 must be closed with "}", but "]" '
                              'closes it at line 6 column 3.')
    # the retry says where and what, structurally, and asks for the same answer
    retry = vlm.calls[1]
    assert "Your previous response was not valid JSON." in retry
    assert "Parser error: Expecting ',' delimiter at line 6 column 3." in retry and ">>   ]," in retry
    assert 'must be closed with "}"' in retry and "correcting the JSON structure only" in retry
    # the second answer parses; its content is Gemma's, unchanged
    assert good["valid"] is True and good["parsed"] == json.loads(FIXED.strip("`json\n"))
    assert good["normalized"]["steps"][0]["do"] == "observe"
    assert [st.do for st in p.steps] == ["observe"] and p.goal == "The tall pink block from the pickup area is placed in the green tray."
    # the user's words Gemma wrote in template brackets lose the brackets (unwrap_user_words, unchanged); nothing else
    assert [(c.pred, c.args) for c in p.success] == [("on", ["tall pink block", "green tray"])]


def test_malformed_json_is_never_repaired_by_python():
    roles, _, events = _plan([MALFORMED, MALFORMED])
    with pytest.raises(VLMFailure, match="not valid JSON: Expecting ',' delimiter at line 6 column 3"):
        roles.plan_goal("x", "menu")
    assert all(e["valid"] is False and "parsed" not in e for e in events)


def test_a_steps_array_closed_like_an_object_is_located_too():
    # live_052445/02: '{ "worker": ..., "args": {} ],' inside "steps"
    raw = '{"goal": "g", "success": [], "steps": [\n  {"worker": "perception", "do": "observe", "args": {}\n  ],\n "reason": "r"}'
    roles, vlm, events = _plan([raw, raw])
    with pytest.raises(VLMFailure):
        roles.plan_goal("x", "menu")
    assert 'A JSON object opened with "{" at line 2 column 3 must be closed with "}"' in vlm.calls[1]


def test_a_schema_error_keeps_its_field_message():
    roles, vlm, _ = _plan(['{"goal": "g", "steps": [{"worker": "perception"}], "success": []}',
                           '{"goal": "g", "steps": [], "success": [], "reason": "r"}'])
    roles.plan_goal("x", "menu")
    assert "Your previous answer was not valid (steps.0.do:" in vlm.calls[1]


def test_the_terminal_shows_the_raw_output_and_the_parse_error():
    from agent.events import format_event
    roles, _, events = _plan([MALFORMED, FIXED])
    roles.plan_goal("place the tall pink block in green tray", "menu")
    line = format_event({**events[0], "wall": "00:00:00"})
    assert line.detail[0] == "GEMMA RAW OUTPUT attempt 1:" and any("GEMMA JSON PARSE ERROR: Expecting ',' delimiter, line 6 "
                                                                  "column 3 (character 160)" in d for d in line.detail)


# ----------------------------------------------------------------------------------------------- authority
class _Asking(GoalTools):
    """Gemma asks one question before planning; the request classifier (Gemma's role) reads a question as a description
    and anything else as an action."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.classified = []
        self.roles.clarify = lambda request, conversation, context="": (
            Clarification(status="ask", question="Which white block do you mean?") if not conversation
            else Clarification(status="clear", request=request))
        self.roles.classify_request = lambda text: (self.classified.append(text), RequestKind(
            kind="description" if text.strip().endswith("?") else "action", evidence=text))[1]


def _run(tmp_path, tools, instruction, answer):
    log = EventLog(tmp_path / "events.jsonl")
    r = GoalSupervisor(tools, log, tmp_path, ScriptedResponder([answer])).run(instruction)
    log.close()
    return r


def test_a_clarification_answer_phrased_as_a_question_keeps_the_permission_to_pick(tmp_path):
    t = _Asking([plan(("pick", {"object": "a"}))])
    r = _run(tmp_path, t, "pick the white block", "do you see any white block in arrangement mat?")
    assert t.classified == ["pick the white block"]  # the original instruction, not the conversation
    assert r["authority"]["kind"] == "action" and "object_change" in r["authority"]["grants"]
    assert not r["refused"] and r["status"] == "COMPLETED" and t.held_object is not None


def test_a_sensing_only_instruction_stays_sensing_only(tmp_path):
    t = _Asking([plan(("pick", {"object": "a"}))])
    r = _run(tmp_path, t, "do you see a white block?", "pick it")
    assert t.classified == ["do you see a white block?"]
    assert r["authority"]["kind"] == "description" and "object_change" not in r["authority"]["grants"]
    assert [x["capability"] for x in r["refused"]] == ["pick"] and t.held_object is None
