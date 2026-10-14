"""Coordinated motion planning for the VT6L arm: joint-space and straight-line (Cartesian) moves.

Plain numpy on top of control/kinematics.py (FK + damped least-squares IK); no ROS, no Isaac Sim.
Both planners return a Trajectory sampled every DT seconds that the simulator plays back.

* joint_move: all joints start and stop together with a smooth (minimum-jerk-like) profile;
  the duration is set by the joint that has the furthest to go at its speed limit.
* linear_move: the TCP follows a straight line (and turns smoothly if the orientation changes);
  IK is solved at every sample, warm-started from the previous one, and the plan is rejected
  if a point is unreachable, hits a joint limit, or needs a joint flip (near a singularity).

    kin = Kinematics()
    traj = linear_move(kin, q_now, target_xyz, target_R, speed=0.05)
    traj.times, traj.positions   # (n,), (n, 6)

Run `python3 -m control.motion` for a self-test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control.kinematics import Kinematics

DT = 1.0 / 30.0  # trajectory sample period (s); the simulator interpolates between samples
MAX_JOINT_STEP = np.deg2rad(4.0)  # largest allowed joint change between samples of a line
PEAK_FACTOR = 1.875  # peak / average speed of the quintic profile below


class PlanningError(RuntimeError):
    """The requested motion cannot be executed (unreachable, joint limit, singularity)."""


@dataclass
class Trajectory:
    times: np.ndarray  # (n,) seconds from start
    positions: np.ndarray  # (n, dof) joint angles (rad)
    description: str = ""

    @property
    def duration(self) -> float:
        return float(self.times[-1])


def _profile(s: np.ndarray) -> np.ndarray:
    """Quintic time scaling 0 -> 1 with zero velocity and acceleration at both ends."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _samples(duration: float) -> tuple[np.ndarray, np.ndarray]:
    n = max(int(np.ceil(duration / DT)), 1)
    times = np.linspace(0.0, n * DT, n + 1)
    return times, _profile(times / times[-1])


def _check_limits(kin: Kinematics, q: np.ndarray, where: str) -> None:
    if np.any(q < kin.lower - 1e-6) or np.any(q > kin.upper + 1e-6):
        bad = [kin.joint_names[i].split("_")[0].upper() for i in np.where((q < kin.lower - 1e-6) | (q > kin.upper + 1e-6))[0]]
        raise PlanningError(f"{where} needs {', '.join(bad)} beyond its limit")


def joint_move(kin: Kinematics, q_start: np.ndarray, q_goal: np.ndarray, max_speed: float | np.ndarray) -> Trajectory:
    """Synchronised joint-space move: every joint arrives at the same time."""
    q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
    _check_limits(kin, q_goal, "the goal")
    delta = q_goal - q_start
    speeds = np.broadcast_to(np.asarray(max_speed, float), delta.shape)
    duration = max(float(np.max(np.abs(delta) / speeds)) * PEAK_FACTOR, 0.2)
    times, s = _samples(duration)
    positions = q_start + s[:, None] * delta
    lead = kin.joint_names[int(np.argmax(np.abs(delta) / speeds))].split("_")[0].upper()
    return Trajectory(times, positions, f"joint move, {times[-1]:.1f} s (paced by {lead})")


def _slerp_matrix(R0: np.ndarray, R1: np.ndarray, s: float) -> np.ndarray:
    """Rotation a fraction s of the way from R0 to R1 (about a fixed axis)."""
    R_rel = R0.T @ R1
    angle = np.arccos(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return R0
    axis = np.array([R_rel[2, 1] - R_rel[1, 2], R_rel[0, 2] - R_rel[2, 0], R_rel[1, 0] - R_rel[0, 1]])
    axis /= 2.0 * np.sin(angle)
    a, (x, y, z) = s * angle, axis
    c, si, C = np.cos(a), np.sin(a), 1.0 - np.cos(a)
    R_s = np.array(
        [
            [c + x * x * C, x * y * C - z * si, x * z * C + y * si],
            [y * x * C + z * si, c + y * y * C, y * z * C - x * si],
            [z * x * C - y * si, z * y * C + x * si, c + z * z * C],
        ]
    )
    return R0 @ R_s


def linear_move(
    kin: Kinematics,
    q_start: np.ndarray,
    p_goal: np.ndarray,
    R_goal: np.ndarray | None = None,
    speed: float = 0.05,
    angular_speed: float = np.deg2rad(30.0),
) -> Trajectory:
    """Straight-line TCP move to p_goal (m, base frame); R_goal None keeps the current orientation."""
    q_start = np.asarray(q_start, float)
    T0 = kin.fk(q_start)
    p0, R0 = T0[:3, 3], T0[:3, :3]
    R1 = R0 if R_goal is None else np.asarray(R_goal, float)
    p1 = np.asarray(p_goal, float)
    distance = float(np.linalg.norm(p1 - p0))
    turn = float(np.arccos(np.clip((np.trace(R0.T @ R1) - 1.0) / 2.0, -1.0, 1.0)))
    duration = max(distance / speed, turn / angular_speed) * PEAK_FACTOR
    if duration < 1e-6:
        return Trajectory(np.array([0.0]), q_start[None, :], "already there")

    times, s = _samples(max(duration, 0.2))
    positions = np.empty((len(times), len(q_start)))
    q = q_start.copy()
    for i, si in enumerate(s):
        p, R = p0 + si * (p1 - p0), _slerp_matrix(R0, R1, si)
        q_next, ok = kin.ik(p, R, q, iterations=60, tol=1e-6)
        if not ok:
            raise PlanningError(
                f"no IK solution {si * distance * 1000:.0f} mm along the line: out of reach with this tool "
                "orientation (try 'goto ... down', or a shorter move)"
            )
        if np.max(np.abs(q_next - q)) > MAX_JOINT_STEP:
            raise PlanningError(
                f"the line passes near a singularity {si * distance * 1000:.0f} mm along (a joint would have to flip)"
            )
        q = q_next
        positions[i] = q
    _check_limits(kin, positions[-1], "the goal")
    return Trajectory(times, positions, f"straight line {distance * 1000:.0f} mm, {times[-1]:.1f} s")


def max_path_deviation(kin: Kinematics, traj: Trajectory, p0: np.ndarray, p1: np.ndarray) -> float:
    """Largest distance (m) of the planned TCP path from the straight segment p0 -> p1."""
    d = p1 - p0
    length = np.linalg.norm(d)
    worst = 0.0
    for q in traj.positions:
        p = kin.fk(q)[:3, 3]
        if length < 1e-9:
            dev = np.linalg.norm(p - p0)
        else:
            t = np.clip(np.dot(p - p0, d) / length**2, 0.0, 1.0)
            dev = np.linalg.norm(p - (p0 + t * d))
        worst = max(worst, float(dev))
    return worst


if __name__ == "__main__":
    from control.robot import ARM_HOME, ARM_SPEED

    kin = Kinematics()
    q = ARM_HOME.copy()
    p_home = kin.fk(q)[:3, 3]
    print(f"TCP at home: {np.round(p_home * 1000, 1).tolist()} mm")

    traj = joint_move(kin, q, np.deg2rad([30, -20, 20, 0, -60, 0]), ARM_SPEED)
    print(f"joint move: {traj.description}; all joints reach the goal at t = {traj.duration:.2f} s")

    for label, goal in [("down 300 mm", p_home + [0, 0, -0.3]), ("left 200 mm", p_home + [-0.2, 0, 0]),
                        ("to (300, 400, 200) mm", np.array([0.3, 0.4, 0.2]))]:
        t = linear_move(kin, q, goal, speed=0.05)
        reached = kin.fk(t.positions[-1])[:3, 3]
        print(f"line {label}: {t.description}; end error {np.linalg.norm(reached - goal) * 1000:.3f} mm; "
              f"max deviation from line {max_path_deviation(kin, t, p_home, goal) * 1000:.3f} mm")
    try:
        linear_move(kin, q, np.array([0.0, 1.5, 0.5]))
    except PlanningError as e:
        print(f"unreachable goal rejected: {e}")
