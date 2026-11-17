"""Real CPU kinematics and collision planning, without robot commands or model inference."""

from pathlib import Path

import numpy as np
import pytest

from agent.tools import Tools
from control.robot import ARM_HOME
from control.skills import SkillParams
from tests.support.fakes import geom


@pytest.fixture(scope="module")
def transfer():
    t = Tools(None, None, None, None, lambda *a, **k: None, Path("/tmp"))
    t.world.observe(geom(0., .55, .06, fp=(.04, .04)), None)
    t.world.observe(geom(.20, .55, .03, fp=(.14, .14)), None)
    return t, t.plan_pick_place(1, "on", 2, "robot", np.array(ARM_HOME))


def test_transfer_passes_payload_through_checked_carry_waypoint_and_plans_empty_home(transfer):
    _, plan = transfer
    assert plan.home and plan.home[-1].trajectory.positions[-1] == pytest.approx(ARM_HOME)
    carry = plan.place[0]
    assert "carrying waypoint" in carry.label
    assert np.any(np.all(np.isclose(carry.trajectory.positions, ARM_HOME), axis=1))
    assert plan.held.body_size == pytest.approx((.04, .04, .06))


def test_final_lowering_obeys_reduced_tcp_speed(transfer):
    t, plan = transfer
    trajectory = plan.place[1].trajectory
    points = np.array([t.kin.fk(q)[:3, 3] for q in trajectory.positions])
    speed = np.linalg.norm(np.diff(points, axis=0), axis=1) / np.diff(trajectory.times)
    assert speed.max() <= SkillParams().place_speed * 1.02


def test_offcentre_body_is_placed_at_support_centre_for_either_symmetric_yaw():
    from control.kinematics import Kinematics
    from control.skills import SkillPlanner, Held, grip_rotation
    kin = Kinematics()
    held = Held(.04, .02, yaw_deg=180., width=.03, length=.08, body_offset=(.008, -.006))
    support = np.array([.12, .55, .0])
    steps, _ = SkillPlanner(kin).place(np.array(ARM_HOME), held, support)
    T = kin.fk(steps[1].trajectory.positions[-1])
    body_centre = T[:2, 3] + T[:2, :2] @ np.array(held.body_offset)
    assert body_centre == pytest.approx(support[:2], abs=.002)
