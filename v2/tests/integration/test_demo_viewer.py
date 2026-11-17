"""The dashboard's log lines (agent/events.py): one readable line per event worth showing, nothing for the rest."""

from agent.events import format_event


def ev(kind, **f):
    return {"kind": kind, "wall": "10:00:00", **f}


def test_the_loop_messages_and_the_skill_progress_are_shown():
    assert str(format_event(ev("say", who="master", text="plan: perception: find it"))).endswith("plan: perception: find it")
    assert format_event(ev("say", who="result", text="COMPLETED: holding #12")).level == "success"
    assert format_event(ev("say", who="manipulation", text="-> failed: grasp: no contact")).level == "error"
    assert format_event(ev("state", state="GRASPING")).text == "GRASPING"
    assert format_event(ev("verification", check="placement", **{"pass": False})).level == "error"
    assert format_event(ev("master", call="plan", answer="...")) is None
