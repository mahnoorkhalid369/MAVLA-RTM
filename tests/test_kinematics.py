"""Kinematics and motion planning (control/), no simulator: FK/IK consistency, limits, straight lines.

Units: metres and radians inside the code; tolerances below say which.
Run: /usr/bin/python3 -m pytest tests/
"""

import numpy as np
import pytest

from control.kinematics import R_DOWN, Kinematics
from control.motion import PlanningError, joint_move, linear_move, max_path_deviation
from control.robot import ARM_HOME, ARM_SPEED

KIN = Kinematics()
RNG = np.random.default_rng(42)


def rotation_angle_deg(R1, R2) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1))))


def test_six_revolute_joints_with_limits():
    assert len(KIN.lower) == 6 and np.all(KIN.lower < KIN.upper)


def test_fk_is_a_rigid_transform():
    T = KIN.fk(ARM_HOME)
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)


def test_ik_recovers_fk_pose_for_random_configurations():
    """IK of FK(q) must reach the same TCP pose (not necessarily the same q: several IK solutions exist)."""
    failures = 0
    for _ in range(40):
        q = RNG.uniform(KIN.lower * 0.6, KIN.upper * 0.6)
        T = KIN.fk(q)
        q_ik, ok = KIN.ik_best(T[:3, 3], T[:3, :3], ARM_HOME)
        if not ok:
            failures += 1
            continue
        T_ik = KIN.fk(q_ik)
        assert np.linalg.norm(T_ik[:3, 3] - T[:3, 3]) < 1e-3  # 1 mm
        assert rotation_angle_deg(T_ik[:3, :3], T[:3, :3]) < 0.1  # deg
        assert np.all(q_ik >= KIN.lower - 1e-9) and np.all(q_ik <= KIN.upper + 1e-9)
    assert failures <= 4, f"IK failed on {failures}/40 reachable poses"


def test_unreachable_target_is_reported():
    _, ok = KIN.ik_best(np.array([0.0, 1.6, 0.2]), R_DOWN, ARM_HOME)  # 1.6 m: beyond the ~0.9 m reach
    assert not ok


def test_joint_move_rejects_goal_outside_limits():
    goal = ARM_HOME.copy()
    goal[1] = KIN.upper[1] + 0.2
    with pytest.raises(PlanningError):
        joint_move(KIN, ARM_HOME, goal, ARM_SPEED)


def test_joint_move_starts_and_ends_at_rest_on_target():
    goal = ARM_HOME + np.radians([20, -10, 15, 0, 10, 30])
    traj = joint_move(KIN, ARM_HOME, goal, ARM_SPEED)
    assert np.allclose(traj.positions[0], ARM_HOME) and np.allclose(traj.positions[-1], goal)
    speed = np.abs(np.diff(traj.positions, axis=0)) / np.diff(traj.times)[:, None]
    assert speed.max() <= ARM_SPEED * 1.05  # never faster than the commanded joint speed


def test_linear_move_stays_on_the_straight_line():
    start, _ = KIN.ik_best(np.array([0.0, 0.58, 0.18]), R_DOWN, ARM_HOME)
    p0, p1 = KIN.fk(start)[:3, 3], np.array([0.0, 0.58, 0.035])
    traj = linear_move(KIN, start, p1, R_DOWN)
    assert np.linalg.norm(KIN.fk(traj.positions[-1])[:3, 3] - p1) < 1e-3
    assert max_path_deviation(KIN, traj, p0, p1) < 1e-3  # whole swept path within 1 mm of the line
