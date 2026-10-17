"""Session lifecycle (agent/session.Session): consecutive instructions of one agent process share the world model,
referents and a VERIFIED held-object state; each instruction keeps its own ledger, conditions and thread.
Mocked robot / perception (no sim, no GPU): these tests show state handling, not physical grasp reliability."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langgraph")

from agent.ask import ScriptedResponder  # noqa: E402
from agent.session import Session  # noqa: E402
from agent.vlm_roles import RecordedVLM  # noqa: E402
from tests.support.fakes import GoalTools  # noqa: E402

PICK = '{"goal": "hold it", "steps": [{"do": "pick", "object": "%s"}]}'
PLACE = '{"goal": "placed", "steps": [{"do": "place", "object": "it", "relation": "%s", "target": "%s"}]}'
LOOK = '{"goal": "report", "steps": [{"do": "observe"}, {"do": "describe_scene"}]}'
# each instruction first gets the request classification (agent/authority.py), then the plan
REQ_PICK = '{"kind": "action", "evidence": "Pick up"}'
REQ_PUT = '{"kind": "action", "evidence": "Put it"}'
REQ_LOOK = '{"kind": "description", "evidence": "Take a look"}'
REQ_DANCE = '{"kind": "action", "evidence": "Do a dance"}'
NAMES = {"a": 1, "sample A1": 1, "c": 3, "processing rack": 2}


class SessionTools(GoalTools):
    """Constructed by Session exactly like agent.tools.Tools; resolves the lab-style names used below."""
    instances: list = []

    def __init__(self, bridge, detector, roles, executor, log, run_dir, live=None, memory=None):
        super().__init__([])
        self.roles, self.log, self.run_dir, self.bridge = roles, log, run_dir, bridge
        SessionTools.instances.append(self)

    def resolve(self, obs, text):
        from agent.vlm_roles import Resolution
        if text in self.ambiguous:
            return Resolution(id=None, ambiguous=True, candidates=self.ambiguous[text])
        return Resolution(id=NAMES.get(text, 2))


@pytest.fixture
def new_session(monkeypatch, tmp_path):
    import agent.tools
    monkeypatch.setattr(agent.tools, "Tools", SessionTools)
    SessionTools.instances = []
    made = []

    def make(answers):
        def finger():  # mock gripper: closed on the object while the stub holds one, open otherwise
            return 0.4 if SessionTools.instances and SessionTools.instances[-1].held_object is not None else 0.0
        bridge = SimpleNamespace(q=lambda: np.zeros(6), finger=finger)
        s = Session(bridge, None, RecordedVLM(list(answers)), None, ScriptedResponder(["cancel"]))
        made.append(s)
        return s

    def run(s, text):
        out = tmp_path / f"{s.id}_{len(made)}_{s.count + 1:02d}"
        r = s.run(text, out)
        return r, [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    return make, run


def test_two_instructions_in_one_session_share_the_referent(new_session):
    make, run = new_session
    s = make([REQ_PICK, PICK % "a", REQ_PUT, PLACE % ("next_to", "c")])
    r1, _ = run(s, "Pick up a.")
    r2, ev = run(s, "Put it beside c.")
    assert (r1["status"], r1["holding"]) == ("COMPLETED", 1)
    assert r2["status"] == "COMPLETED" and r2["holding"] is None
    assert len(SessionTools.instances) == 1  # one Tools for the session
    assert [(e["text"], e["world_id"]) for e in ev if e["kind"] == "binding"] == [("it", 1)]
    # re-checked, not assumed: at the start of the instruction and again right before the place motion
    assert [e["held"] for e in ev if e["kind"] == "session_state"] == ["verified", "verified"]
    assert r2["session"] == {"id": s.id, "instruction_index": 2}


def test_pick_sample_a1_then_put_it_in_the_processing_rack(new_session):
    make, run = new_session
    s = make([REQ_PICK, PICK % "sample A1", REQ_PUT, PLACE % ("on", "processing rack")])
    run(s, "Pick up sample A1.")
    r, ev = run(s, "Put it in the processing rack.")
    assert r["status"] == "COMPLETED" and r["plan"][0]["capability"] == "place"
    assert [(e["text"], e["world_id"]) for e in ev if e["kind"] == "binding"] == [("it", 1)]
    assert {g["condition"]["pred"]: g["satisfied"] for g in r["goal_check"]} == {"on": True, "not_holding": True}
    assert r["goal_check"][0]["bound"] == {"#1": 1, "processing rack": 2}


def test_a_new_session_does_not_inherit_held_state(new_session):
    make, run = new_session
    run(make([REQ_PICK, PICK % "a"]), "Pick up a.")
    fresh = make([REQ_PUT, PLACE % ("next_to", "c")])
    r, ev = run(fresh, "Put it beside c.")
    assert len(SessionTools.instances) == 2 and SessionTools.instances[1].held_object is None
    assert r["status"] == "CANCELLED" and r["outcome"] == "insufficient_info" and SessionTools.instances[1].executed == []
    assert [e["held"] for e in ev if e["kind"] == "session_state"] == ["none"]


def test_failed_and_cancelled_instructions_do_not_corrupt_the_session(new_session):
    make, run = new_session
    bad = '{"steps": [{"do": "dance"}]}'  # rejected twice -> the instruction fails at interpretation
    s = make([REQ_PICK, PICK % "a", REQ_DANCE, bad, bad, REQ_PUT, PLACE % ("next_to", "zz"), REQ_PUT, PLACE % ("next_to", "c")])
    run(s, "Pick up a.")
    r_fail, _ = run(s, "Do a dance.")
    t = SessionTools.instances[0]
    t.ambiguous = {"zz": [2, 3]}
    n = len(t.executed)
    r_cancel, _ = run(s, "Put it beside zz.")  # clarification answered "cancel"
    assert r_fail["status"] == "FAILED" and r_cancel["status"] == "CANCELLED" and len(t.executed) == n
    assert t.held_object == 1  # still held, still the referent
    r, ev = run(s, "Put it beside c.")
    assert r["status"] == "COMPLETED" and [(e["text"], e["world_id"]) for e in ev if e["kind"] == "binding"] == [("it", 1)]


def test_goal_conditions_do_not_leak_into_the_next_instruction(new_session):
    make, run = new_session
    s = make([REQ_PICK, PICK % "a", REQ_LOOK, LOOK])
    r1, _ = run(s, "Pick up a.")
    r2, _ = run(s, "Take a look at the table.")
    assert [c["pred"] for c in r1["conditions"]] == ["holding"]
    assert [c["pred"] for c in r2["conditions"]] == ["reported"]
    assert [p["capability"] for p in r2["plan"]] == ["observe", "describe_scene"]


def test_a_dropped_object_is_not_trusted_as_held(new_session):
    """Remembered 'held' but the fingers are open at the next instruction: the claim is dropped, the object's pose
    is unknown, and no place motion runs."""
    make, run = new_session
    s = make([REQ_PICK, PICK % "a", REQ_PUT, PLACE % ("next_to", "c")])
    run(s, "Pick up a.")
    t = SessionTools.instances[0]
    t.contact, n = [False], len(t.executed)
    r, ev = run(s, "Put it beside c.")
    assert [e["held"] for e in ev if e["kind"] == "session_state"] == ["invalidated"]
    assert t.held_object is None and t.world.objects[1].status == "missing"
    assert r["status"] == "CANCELLED" and len(t.executed) == n  # asked, nothing moved
    assert "no longer in the gripper" in r["message"] and "Nothing was moved" in r["message"]


def test_losing_the_object_during_the_instruction_blocks_the_place(new_session):
    make, run = new_session
    s = make([REQ_PICK, PICK % "a", REQ_PUT, PLACE % ("next_to", "c")])
    run(s, "Pick up a.")
    t = SessionTools.instances[0]
    t.contact, n = [True, False], len(t.executed)  # verified at the start, gone by the place step
    r, _ = run(s, "Put it beside c.")
    assert r["status"] == "CANCELLED" and "no longer hold #1" in r["message"] and "Nothing was moved" in r["message"]
    assert len(t.executed) == n and t.held_object is None
