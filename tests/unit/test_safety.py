"""Joint-limit enforcement (safety/joint_limits.py), independent of the planner."""

import numpy as np

from control.kinematics import Kinematics
from control.motion import joint_move
from control.robot import ARM_HOME, ARM_JOINTS, ARM_SPEED
from safety.joint_limits import JointLimits, VelocityMonitor, check_trajectory, clamp_command

LIMITS = JointLimits.from_urdf()
KIN = Kinematics()


def test_limits_are_the_manual_values():
    assert np.allclose(np.degrees(LIMITS.max_velocity), [166.2, 122.5, 141.2, 268.7, 296.8, 293.2], atol=0.05)


def test_planner_trajectory_is_accepted():
    traj = joint_move(KIN, ARM_HOME, ARM_HOME + np.radians([40, -20, 30, 60, 20, 90]), ARM_SPEED)
    assert check_trajectory(LIMITS, ARM_JOINTS, traj.times, traj.positions, current=ARM_HOME) == []


def test_time_compressed_trajectory_is_rejected():
    """The same path played 8x faster: J2 would need ~2x its 122.5 deg/s limit."""
    traj = joint_move(KIN, ARM_HOME, ARM_HOME + np.radians([0, -60, 0, 0, 0, 0]), ARM_SPEED)
    problems = check_trajectory(LIMITS, ARM_JOINTS, traj.times / 8.0, traj.positions, current=ARM_HOME)
    assert any("j2_joint" in p and "exceeds" in p for p in problems)


def test_out_of_range_and_jump_are_rejected():
    pts = np.array([ARM_HOME, ARM_HOME + np.radians([0, 0, 0, 0, 0, 0])])
    pts[1][1] = np.radians(70)  # J2 limit is +65
    assert any("outside" in p for p in check_trajectory(LIMITS, ARM_JOINTS, [0.0, 10.0], pts, current=ARM_HOME))
    far = ARM_HOME + np.radians([10, 0, 0, 0, 0, 0])
    assert any("starts" in p for p in check_trajectory(LIMITS, ARM_JOINTS, [0.0, 10.0], [far, far], current=ARM_HOME))


def test_bad_timestamps_are_rejected():
    assert check_trajectory(LIMITS, ARM_JOINTS, [0.0, 0.0], [ARM_HOME, ARM_HOME], current=ARM_HOME)


def test_direct_command_is_clamped():
    goal, speed, notes = clamp_command(LIMITS, "j2_joint", np.radians(80), np.radians(500))
    assert np.isclose(goal, LIMITS.upper[1]) and np.isclose(speed, LIMITS.max_velocity[1]) and len(notes) == 2
    assert clamp_command(LIMITS, "finger_joint", 0.5, 3.0) == (0.5, 3.0, [])  # gripper: not an arm joint


def test_velocity_monitor_reports_an_excursion_once():
    mon = VelocityMonitor(LIMITS)
    fast = np.zeros(6)
    fast[0] = LIMITS.max_velocity[0] * 1.3
    assert mon.update(ARM_JOINTS, np.zeros(6), 0.0) == []
    assert len(mon.update(ARM_JOINTS, fast, 0.1)) == 1  # starts
    assert mon.update(ARM_JOINTS, fast, 0.2) == []  # same excursion: not repeated
    end = mon.update(ARM_JOINTS, np.zeros(6), 0.3)
    assert len(end) == 1 and "130 %" in end[0] and mon.events == 1
