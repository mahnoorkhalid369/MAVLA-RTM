"""Clear-view parking in the pick-and-place skill (agent/pick_place.py): the arm observes from its current pose; the
existing clear-view park (Tools.park_for_view) is only a fallback when the required target / destination could not be
localized and the arm blocks part of the camera's view. Generic workspace occlusion alone never moves the arm.
Stub tools (no sim, no GPU, no motion)."""

import json

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langgraph")

from agent.ask import ScriptedResponder  # noqa: E402
from agent.eventlog import EventLog  # noqa: E402
from agent.pick_place import PickPlaceWorkflow  # noqa: E402
from tests.support.fakes import StubTools  # noqa: E402

PICK = {"object": {"text": "a"}, "relation": None, "reference": None, "frame": "robot"}
PLACE = {"object": {"text": "#1"}, "relation": "on", "reference": {"text": "b"}, "frame": "robot"}


class ViewTools(StubTools):
    """StubTools whose camera view is partly blocked by the arm (1/36 points, like the live report) and whose
    resolution can miss a reference on the first observation (it is not visible from the current pose)."""

    def __init__(self, unseen_first=(), blocked=1, **kw):
        super().__init__(**kw)
        self.parks, self.perceived, self.blocked = [], 0, blocked
        self.unseen_first, self.missed = set(unseen_first), set()

    def view_blockage(self, q=None):
        return self.blocked, 36

    def park_for_view(self, target_xy=None):
        self.parks.append(target_xy)
        return {"ok": True}

    def perceive(self, camera):
        self.perceived += 1
        return super().perceive(camera)

    def resolve(self, obs, text, exclude=(), choose=True):
        from agent.vlm_roles import Resolution
        if text in self.unseen_first and text not in self.missed:
            self.missed.add(text)
            return Resolution(id=None, ambiguous=False, candidates=[], reason="not in this view")
        return super().resolve(obs, text, exclude)


def run(tmp_path, tools, mode, task):
    log = EventLog(tmp_path / "events.jsonl")
    wf = PickPlaceWorkflow(tools, log, tmp_path, ScriptedResponder(["cancel"]), final_home=False, mode=mode)
    r = wf.run("test", task)
    log.close()
    ev = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    return r, ev


def pre_placement_parks(tools):
    """Parks before the placement check (the placement check passes a target point; the pre-observation park does not)."""
    return [p for p in tools.parks if p is None]


def states(ev):
    return [e["state"] for e in ev if e["kind"] == "state"]


def hold_object_1(t):
    t.perceive("table")  # the world knows the objects
    t.held_object, t.held_geom = 1, t.world.objects[1].geometry
    t.world.holding = 1


def test_1_pick_target_visible_no_park(tmp_path):
    t = ViewTools()
    r, ev = run(tmp_path, t, "pick", PICK)
    assert pre_placement_parks(t) == [] and t.perceived == 1
    assert states(ev)[:3] == ["OBSERVING", "LOCALIZING", "PLANNING_GRASP"]  # manipulation continues from this pose


def test_2_place_destination_visible_no_park(tmp_path):
    t = ViewTools()
    hold_object_1(t)
    n = t.perceived
    r, ev = run(tmp_path, t, "place", PLACE)
    assert pre_placement_parks(t) == [] and t.perceived - n == 1
    assert states(ev)[:3] == ["OBSERVING", "LOCALIZING", "PLANNING_GRASP"]  # placement planning continues


def test_3_pick_target_not_visible_parks_reobserves_and_retries(tmp_path):
    t = ViewTools(unseen_first={"a"})
    r, ev = run(tmp_path, t, "pick", PICK)
    assert states(ev)[:6] == ["OBSERVING", "LOCALIZING", "RECOVERING", "OBSERVING", "LOCALIZING", "PLANNING_GRASP"]
    assert pre_placement_parks(t) == [None] and t.perceived == 2  # the existing clear-view park, then observe again
    decision = next(e for e in ev if e.get("tool") == "clear_view_decision")
    assert decision["park"] is True and decision["view_blocked"] == "1/36"


def test_4_place_destination_not_visible_parks_reobserves_and_retries(tmp_path):
    t = ViewTools(unseen_first={"b"})
    hold_object_1(t)
    n = t.perceived
    r, ev = run(tmp_path, t, "place", PLACE)
    assert states(ev)[:6] == ["OBSERVING", "LOCALIZING", "RECOVERING", "OBSERVING", "LOCALIZING", "PLANNING_GRASP"]
    assert pre_placement_parks(t) == [None] and t.perceived - n == 2


def test_an_unblocked_view_is_not_parked_for_even_after_a_miss(tmp_path):
    """Parking cannot reveal anything when the arm blocks nothing: the existing retry observes again in place."""
    t = ViewTools(unseen_first={"a"}, blocked=0)
    r, ev = run(tmp_path, t, "pick", PICK)
    assert pre_placement_parks(t) == [] and t.perceived == 2


class RejectingTools(ViewTools):
    """The target is found, but Gemma's crop check rejects the first candidate (live 2026-10-21: 'maroon tall
    block' -> brown cube #1, crop check 'no')."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.rejections = 1

    def confirm(self, obs, mark, text):
        from agent.vlm_roles import Confirmation
        self.confirmed.append((mark, text))
        if self.rejections:
            self.rejections -= 1
            return Confirmation(match="no", reason="not that one")
        return Confirmation(match="yes")


def test_a_rejected_candidate_is_not_a_visibility_problem_no_park(tmp_path):
    t = RejectingTools()
    r, ev = run(tmp_path, t, "pick", PICK)
    assert states(ev)[:6] == ["OBSERVING", "LOCALIZING", "RECOVERING", "OBSERVING", "LOCALIZING", "PLANNING_GRASP"]
    assert pre_placement_parks(t) == [] and not any(e.get("tool") == "clear_view_decision" for e in ev)


class RememberingTools(RejectingTools):
    """Records what resolve was told to leave out."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.excluded = []

    def resolve(self, obs, text, exclude=(), choose=True):
        from agent.vlm_roles import Resolution
        self.excluded.append(set(exclude))
        return Resolution(id=3 if 1 in set(exclude) else 1)


def test_a_rejected_candidate_is_left_out_of_the_next_attempt(tmp_path):
    """Live 2026-10-21: the crop check rejected the brown cube for 'maroon tall block' and the next attempt picked it
    again. A rejected object is left out of later attempts in the same task."""
    t = RememberingTools()
    r, ev = run(tmp_path, t, "pick", PICK)
    assert t.excluded[:2] == [set(), {1}] and t.confirmed[:2] == [(1, "a"), (3, "a")]
