"""Grasp skills (control/skills.py), plan validation (vlm/plan.py) and world memory, no simulator."""

import numpy as np
import pytest

from control.kinematics import Kinematics
from control.motion import PlanningError
from control.robot import ARM_HOME
from control.skills import SkillPlanner, grip_rotation
from perception.world_memory import WorldMemory, matches
from vlm.plan import PlanError, parse_plan

KIN = Kinematics()
SKILLS = SkillPlanner(KIN)


@pytest.mark.parametrize("yaw", [-45.0, 0.0, 30.0, 90.0])
def test_hover_points_straight_down_at_the_requested_yaw(yaw):
    top = np.array([0.0, 0.58, 0.05])
    _, q, used = SKILLS.hover(ARM_HOME, top, yaw)
    T = KIN.fk(q)
    assert np.linalg.norm(T[:3, 3] - (top + [0, 0, SKILLS.p.approach])) < 1e-3
    assert np.degrees(np.arccos(-T[2, 2])) < 0.1  # tool axis vertical
    assert (used - yaw) % 180.0 == pytest.approx(0.0, abs=1e-6)  # same grip (parallel gripper: 180 deg symmetric)
    assert np.allclose(T[:3, :3], grip_rotation(used), atol=1e-3)


def test_too_wide_object_is_refused():
    with pytest.raises(PlanningError):
        SKILLS.pick(ARM_HOME, np.array([0.0, 0.58, 0.05]), width=0.10)


def test_pick_then_place_plans_and_keeps_yaw():
    steps, q, held = SKILLS.pick(ARM_HOME, np.array([0.0, 0.58, 0.06]), yaw_deg=20.0)
    assert held.height == pytest.approx(0.06) and held.yaw_deg % 180 == pytest.approx(20.0)
    more, q = SKILLS.place(q, held, np.array([0.15, 0.50, 0.05]))
    assert [s.label for s in more][-2:] == ["open gripper", "retreat"]


def test_plan_parsing_rules():
    plan = parse_plan('{"reply": "ok", "steps": [{"action": "pick", "object": "red cube"}, '
                      '{"action": "place", "target": "green cube"}, {"action": "home"}]}')
    assert [s.action for s in plan.steps] == ["pick", "place", "home"]
    with pytest.raises(PlanError):
        parse_plan('{"steps": [{"action": "place", "target": "green cube"}]}')  # place without pick
    with pytest.raises(PlanError):
        parse_plan('{"steps": [{"action": "dance"}]}')  # not a robot skill
    with pytest.raises(PlanError):
        parse_plan("I will pick the red block.")  # not JSON


def test_world_memory_merges_sightings_and_separates_stacks():
    m = WorldMemory()
    a = m.observe("red cube", [0.0, 0.58, 0.06])
    assert m.observe("red block", [0.004, 0.583, 0.061]).id == a.id  # same object again
    g = m.observe("green cube", [0.15, 0.50, 0.05])
    assert m.observe("lime cube", [0.15, 0.50, 0.09]).id != g.id  # stacked on top: a different object
    m.pick(a.id)
    m.place(a.id, [0.30, 0.40, 0.06])
    assert m.get(a.id).moved_by_robot and m.find("red cube").id == a.id
    assert matches("yellow cube", "gold cube") and not matches("red cube", "green cube")


def test_stacked_object_uses_its_own_height():
    """Lime cube (40 mm) on the blue cube (50 mm): the grasp and the release height use 40 mm, and the
    closing fingertips stay above the blue cube's top."""
    from control.robot import fingertip_below_tcp
    top = np.array([-0.15, 0.50, 0.09])
    steps, q, held = SKILLS.pick(ARM_HOME, top, width=0.04, height=0.04)
    assert held.height == pytest.approx(0.04)
    tcp_z = KIN.fk(steps[2].trajectory.positions[-1])[2, 3]  # end of "descend to grasp"
    assert tcp_z - fingertip_below_tcp(0.04) >= 0.05 + SKILLS.p.tip_clearance - 1e-6


def test_small_cube_fingertips_clear_the_floor_when_closed():
    from control.robot import fingertip_below_tcp
    steps, q, held = SKILLS.pick(ARM_HOME, np.array([0.0, 0.6, 0.035]), width=0.035)  # orange cube, 35 mm
    tcp_z = KIN.fk(steps[2].trajectory.positions[-1])[2, 3]
    assert tcp_z - fingertip_below_tcp(0.035) >= SKILLS.p.tip_clearance - 1e-6
