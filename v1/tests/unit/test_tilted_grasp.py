"""Bounded tilted grasp (control/skills.py tilted_pick, agent/tools/manipulation.py plan_pick): a physical fallback for
the same delegated pick when a top-down grasp is not possible. Top-down stays the default; tilt is about the closing
axis (pads stay on the same vertical faces), at most SkillParams.max_tilt_deg, with every safety check kept."""

from types import SimpleNamespace

import numpy as np
import pytest

from control.collision import CollisionChecker
from control.kinematics import Kinematics
from control.motion import PlanningError
from control.robot import ARM_HOME, fingertip_below_tcp
from control.skills import Held, SkillParams, SkillPlanner, grip_rotation


@pytest.mark.parametrize("yaw", [0.0, 37.0, -90.0])
@pytest.mark.parametrize("tilt", [10.0, -30.0])
def test_tilt_is_about_the_closing_axis_so_the_pads_stay_on_the_same_faces(yaw, tilt):
    R0, R = grip_rotation(yaw), grip_rotation(yaw, tilt)
    assert np.allclose(R[:, 0], R0[:, 0])  # closing axis unchanged (and horizontal)
    assert abs(R[2, 0]) < 1e-9
    tilt_from_vertical = np.degrees(np.arccos(-R[2, 2]))  # tool z points into the object: down at tilt 0
    assert tilt_from_vertical == pytest.approx(abs(tilt))


def test_zero_tilt_is_exactly_the_top_down_grasp():
    assert np.array_equal(grip_rotation(25.0, 0.0), grip_rotation(25.0))
    sp = SkillPlanner(Kinematics())
    assert sp._grasp_z(0.06, 0.06, 0.05, 0.0) == sp._grasp_z(0.06, 0.06, 0.05)


def test_a_tilted_grasp_keeps_the_pad_corners_above_the_table():
    p = SkillParams()
    sp = SkillPlanner(Kinematics(), p)
    width, height = 0.05, 0.03  # a low block: the clearance, not the grip depth, sets the grasp height
    for tilt in (10.0, 20.0, 30.0):
        z, _ = sp._grasp_z(height, height, width, tilt)
        t = np.radians(tilt)
        lowest = z - fingertip_below_tcp(width) * np.cos(t) - p.pad_half_width * np.sin(t)
        assert lowest >= p.tip_clearance - 1e-9


def test_the_held_box_stays_upright_in_the_world_when_held_tilted():
    kin = Kinematics()
    for tilt in (0.0, 20.0, -30.0):
        R = grip_rotation(40.0, tilt)
        q, ok = kin.ik_best(np.array([0.0, 0.55, 0.05]), R, ARM_HOME, extra_seeds=[ARM_HOME])
        assert ok
        held = Held(height=0.06, grip_depth=0.025, yaw_deg=40.0, width=0.05, length=0.05, tilt_deg=tilt)
        T_box, _ = held.collision_body()
        T_tcp = kin.fk(q)
        T_tool0 = T_tcp.copy()
        T_tool0[:3, 3] = T_tcp[:3, 3] - T_tcp[:3, 2] * 0.1337  # tcp is 133.7 mm along tool z from tool0
        W = T_tool0 @ T_box
        assert np.degrees(np.arccos(abs(W[2, 2]))) < 0.5  # box axis vertical
        # the TCP is grip_depth below the object's top, on its axis
        top = W[:3, 3] + W[:3, 2] * (np.sign(W[2, 2]) * 0.03)
        assert np.allclose(top[:2], T_tcp[:2, 3], atol=1e-6) and top[2] - T_tcp[2, 3] == pytest.approx(0.025, abs=1e-6)


def test_a_top_down_reachable_object_gets_the_top_down_plan(tmp_path):
    t, log = _tools([0.0, 0.58])
    plan = t.plan_pick(1, np.array(ARM_HOME, float))
    assert plan.grasp["method"] == "top_down" and plan.held.tilt_deg == 0.0
    sel = [e for e in log if e.get("tool") == "grasp_select"]
    assert sel and sel[-1]["selected"] == "top_down" and sel[-1]["tilted"] == []


def test_an_object_at_the_far_edge_falls_back_to_a_bounded_tilted_grasp_of_the_same_object():
    t, log = _tools([0.0, 0.95])  # top-down cannot reach 0.95 m straight ahead; a 20 deg lean can
    plan = t.plan_pick(1, np.array(ARM_HOME, float))
    assert plan.object_id == 1 and plan.relation == "hold" and plan.place == []  # same pick, nothing added
    assert plan.grasp["method"] == "tilted" and 0 < abs(plan.grasp["tilt_deg"]) <= SkillParams().max_tilt_deg
    assert plan.held.tilt_deg == plan.grasp["tilt_deg"]
    assert [s.label for s in plan.approach[:1]][0].startswith("descend to grasp (tilted")
    sel = [e for e in log if e.get("tool") == "grasp_select"][-1]
    assert sel["selected"] == "tilted" and sel["top_down"]["ok"] is False and "out of reach" in sel["top_down"]["reason"]
    assert any(c["ok"] for c in sel["tilted"]) and all({"yaw_deg", "tilt_deg", "ok", "reason"} <= set(c) for c in sel["tilted"])
    # the smallest tilt that works wins: no candidate with a smaller tilt passed
    assert all(not c["ok"] for c in sel["tilted"] if abs(c["tilt_deg"]) < abs(plan.grasp["tilt_deg"]))


def test_the_live_block_behind_the_base_is_picked_tilted_at_40_deg():
    """Live 2026-10-25 (object #13 at x +2, y -621 mm, 0.62 m straight behind the base): J1 stops at +-170, so the arm
    never faces it; square to its faces the hand needs 40 deg, the bound since 2026-10-25."""
    t, log = _tools([0.002, -0.621])
    plan = t.plan_pick(1, np.array(ARM_HOME, float))
    assert plan.grasp["method"] == "tilted" and abs(plan.grasp["tilt_deg"]) == 40.0


def test_a_block_far_behind_the_base_is_infeasible_and_the_result_goes_back_unchanged():
    t, log = _tools([0.002, -0.9])  # straight behind, beyond the measured 0.74 m (control/workspace.py)
    with pytest.raises(PlanningError) as e:
        t.plan_pick(1, np.array(ARM_HOME, float))
    msg = str(e.value)
    assert "out of reach" in msg and "no tilted grasp up to 40 deg passed the checks" in msg
    from agent.tools.manipulation import ManipulationTools
    assert ManipulationTools.deterministic(msg)  # reported to the master as infeasible, not retried here
    sel = [x for x in log if x.get("tool") == "grasp_select"][-1]
    assert sel["selected"] is None and sel["tilted"] and not any(c["ok"] for c in sel["tilted"])
    assert max(abs(c["tilt_deg"]) for c in sel["tilted"]) == SkillParams().max_tilt_deg == 40.0


def test_pick_place_stays_top_down_only():
    t, log = _tools([0.0, 0.95])
    with pytest.raises(PlanningError, match="out of reach"):
        t.plan_pick_place(1, "next_to", 1, "robot", np.array(ARM_HOME, float))
    assert not [e for e in log if e.get("tool") == "grasp_select"]


def test_placement_refuses_a_tilted_hold_instead_of_planning_it_as_top_down():
    from agent.tools.manipulation import ManipulationTools
    t = ManipulationTools.__new__(ManipulationTools)
    t.held, t.held_object = Held(height=0.05, grip_depth=0.02, width=0.05, tilt_deg=20.0), 1
    with pytest.raises(PlanningError, match="held with a tilted grasp"):
        t.plan_place_held("on", 2, "robot", np.array(ARM_HOME, float))
    assert ManipulationTools.deterministic("the object is held with a tilted grasp (+20 deg); placement ...")


# ----------------------------------------------------------------------------------- helpers
def _tools(xy, size=0.05):
    from agent.tools.manipulation import ManipulationTools
    log = []
    t = ManipulationTools.__new__(ManipulationTools)
    t.kin = Kinematics()
    t.log = lambda kind, **k: log.append({"kind": kind, **k})
    geom = {"centre": [xy[0], xy[1], size / 2], "top_z": size, "height": size, "support_z": 0.0, "tilt_deg": 0.5,
            "grasp_width": size, "footprint": [size, size], "grasp_yaw_deg": 0.0, "symmetry_deg": 90.0,
            "extents": [size, size, size]}
    half = np.array([size / 2] * 3)
    obj = SimpleNamespace(id=1, status="on_surface", geometry=geom,
                          box=lambda: __import__("control.collision", fromlist=["Box"]).Box(
                              "obj_1", np.array(geom["centre"]), np.eye(3), half))
    t.world = SimpleNamespace(objects={1: obj})
    return t, log


def test_grasp_numbers_are_plain_floats_for_the_checkpointed_skill_state():
    """Live 2026-10-26: a numpy.float64 grip depth (from cos(tilt), even at 0 deg) made LangGraph's checkpointer fail
    ("not msgpack serializable") right after a successful lift; the pick was reported failed."""
    t, _ = _tools([0.0, 0.58])
    p = t.plan_pick(1, np.array(ARM_HOME, float))
    for v in [p.held.grip_depth, p.held.height, *p.grasp.values()]:
        assert type(v) in (float, str), (v, type(v))
    ormsgpack = pytest.importorskip("ormsgpack")  # LangGraph's serializer (agent environment); the types above suffice
    ormsgpack.packb({"grip_depth": p.held.grip_depth, "grasp": p.grasp})


def test_a_neighbour_in_the_way_is_not_dodged_by_tilting(monkeypatch):
    """Live 2026-10-26: top-down collided with a neighbour by 0.7 mm; the block was taken tilted and could then not be
    placed. The tilted grasp is for reach only: other top-down failures are reported as they are."""
    from agent.tools.manipulation import ManipulationTools
    t, log = _tools([0.0, 0.58])

    def collides(*a, **k):
        raise PlanningError("descend to grasp: collision, pads with obj_6 (0.7 mm)")
    monkeypatch.setattr(ManipulationTools, "_plan_grasp", collides)
    with pytest.raises(PlanningError, match="collision"):
        t.plan_pick(1, np.array(ARM_HOME, float))
    assert not [e for e in log if e.get("tool") == "grasp_select"]  # no tilted attempt
