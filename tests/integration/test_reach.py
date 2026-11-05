"""Out-of-reach grasps (agent/tools/manipulation.py grasp_reach, agent/pick_place.py): when no gripper orientation the
arm can take picks the object up, the skill says so in one clear sentence. Real kinematics; stub tools for the skill."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langgraph")

from agent.ask import ScriptedResponder  # noqa: E402
from agent.eventlog import EventLog  # noqa: E402
from agent.pick_place import PickPlaceWorkflow  # noqa: E402
from agent.tools.manipulation import ManipulationTools  # noqa: E402
from control.kinematics import Kinematics  # noqa: E402
from control.motion import PlanningError  # noqa: E402
from tests.support.fakes import StubTools  # noqa: E402


def reach_of(x, y):
    t = ManipulationTools.__new__(ManipulationTools)
    t.kin, t.log = Kinematics(), (lambda *a, **k: None)
    t.world = SimpleNamespace(objects={1: SimpleNamespace(geometry={"centre": [x, y, 0.0125], "top_z": 0.025})})
    return t.grasp_reach(1)


def test_a_block_in_front_is_graspable_from_above():
    assert reach_of(0.0, 0.6)["from_above"] is True


def test_the_live_block_behind_the_base_is_reachable_only_tilted():
    r = reach_of(0.002, -0.621)  # live 2026-10-21: "pick the white block at the back"
    assert r["from_above"] is False and r["any_orientation"] is True and r["tilt_deg"] > 15


def test_a_block_far_away_is_outside_the_range_in_any_orientation():
    r = reach_of(0.0, 1.6)
    assert (r["from_above"], r["any_orientation"]) == (False, False)


class FarTools(StubTools):
    def __init__(self, reach):
        super().__init__()
        self.reach = reach

    def plan_pick(self, oid, q):
        raise PlanningError("move above object: [2, -621, 171] mm is out of reach")

    def grasp_reach(self, oid):
        return self.reach


def run_pick(tmp_path, tools):
    log = EventLog(tmp_path / "events.jsonl")
    r = PickPlaceWorkflow(tools, log, tmp_path, ScriptedResponder(["cancel"]), final_home=False, mode="pick").run(
        "pick a", {"object": {"text": "a"}, "relation": None, "reference": None, "frame": "robot"})
    log.close()
    return r


@pytest.mark.parametrize("reach,start,expect", [
    ({"from_above": False, "any_orientation": True, "tilt_deg": 29, "distance_m": 0.62, "bearing_deg": 180,
      "centre": [0.002, -0.621, 0.013]}, "Neither a top-down nor a tilted grasp (up to 40 deg) can pick up object #",
     "no grasp with the pads on two of its faces passed"),
    ({"from_above": False, "any_orientation": True, "tilt_deg": 52, "distance_m": 0.9, "bearing_deg": 180,
      "centre": [0.002, -0.9, 0.013]}, "No possible gripper orientation can pick up object #",
     "only tilted 52 deg from vertical, beyond the 40 deg limit for tilted grasps"),
    ({"from_above": False, "any_orientation": False, "tilt_deg": None, "distance_m": 1.6, "bearing_deg": 0,
      "centre": [0.0, 1.6, 0.013]}, "No possible gripper orientation can pick up object #", "it is outside the arm's range"),
])
def test_no_possible_orientation_is_stated_cleanly(tmp_path, reach, start, expect):
    r = run_pick(tmp_path, FarTools(reach))
    reason = r["failure"]["reason"]
    assert r["status"] == "FAILED" and r["failure"]["cls"] == "infeasible"
    assert reason.startswith(start) and expect in reason


def test_a_grasp_reachable_from_above_keeps_the_planners_message(tmp_path):
    r = run_pick(tmp_path, FarTools({"from_above": True, "any_orientation": True, "tilt_deg": 0, "distance_m": 0.6,
                                     "bearing_deg": 0, "centre": [0.0, 0.6, 0.0]}))
    assert "out of reach" in r["failure"]["reason"] and "No possible" not in r["failure"]["reason"]


def test_resolve_gives_gemma_the_measured_height_size_position_and_camera_colour():
    """Live 2026-10-21: 'maroon tall block' was matched on words only ('Red Cube' for the 71 mm maroon block); the
    shared resolve tool now adds what the depth camera measured to every candidate."""
    from agent.tools.resolution import ResolutionTools
    from agent.vlm_roles import Resolution
    seen = {}

    class T(ResolutionTools):
        def pool(self, obs, text):
            return list(obs.marks)

        def describe(self, obs, marks=None):
            pass

        def log(self, *a, **k):
            pass
    t = T()
    from agent.vlm_roles import ReferenceSpec
    t.roles = SimpleNamespace(resolve=lambda som, descs, text, ids, allowed=None: (seen.update(d=descs), Resolution(id=8))[1],
                              parse_reference=lambda text: ReferenceSpec())  # no split: the description path
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[:2, :2] = (128, 26, 38)
    mask = np.zeros((4, 4), bool)
    mask[:2, :2] = True
    obs = SimpleNamespace(marks={8: mask}, rgb=rgb, som=None, descriptions={8: {"name": "Cube", "colour": "Red"}},
                          measured={8: {"height": 0.071, "footprint": [0.039, 0.039], "centre": [0.282, 0.797, 0.035]}})
    assert t.resolve(obs, "maroon tall block").id == 8
    m = seen["d"][0]["measured"]
    assert "71 mm tall" in m and "footprint 39 x 39 mm" in m and "x +282, y +797 mm" in m and "rgb(128, 26, 38)" in m


def test_the_resolve_prompt_carries_the_measured_facts():
    import json as _json
    from agent.vlm_roles import RecordedVLM, Roles
    vlm = RecordedVLM([_json.dumps({"id": 8, "ambiguous": False, "candidates": [8], "reason": "tallest dark red"})])
    Roles(vlm).resolve(np.zeros((4, 4, 3), np.uint8), [{"id": 8, "name": "Cube", "colour": "Red", "measured": "71 mm tall"}],
                       "maroon tall block", [8])
    assert "8: Cube (Red, , ); measured: 71 mm tall" in vlm.calls[0] and "reliable" in vlm.calls[0]
