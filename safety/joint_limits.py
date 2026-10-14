"""Joint-limit enforcement that does not trust the planner (numpy only, no ROS / Isaac Sim).

Two layers, used by the simulator's command interface (sim/ros_control.py):
  1. check_trajectory(): a command validator. A joint trajectory is rejected before execution if any
     point is outside the joint range, any segment is faster than the joint's maximum speed, the times
     are not increasing, or it starts away from where the arm is (a jump = infinite speed).
  2. VelocityMonitor: a runtime monitor comparing MEASURED joint speeds with the limits; it reports
     each excursion once (when it starts) with the peak value when it ends.

Limits come from the URDF (Epson VT6L manual: ranges and max. operating speeds, see assets/vt6l/).
In simulation this is the only speed enforcement (PhysX does not apply the imported speed limits).
On a real robot the Epson controller's own limits stay authoritative; this layer is an extra check.
Units: rad, rad/s, s.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class JointLimits:
    names: list[str]
    lower: np.ndarray
    upper: np.ndarray
    max_velocity: np.ndarray

    @classmethod
    def from_urdf(cls, names: list[str] | None = None) -> "JointLimits":
        from control.kinematics import Kinematics
        from control.robot import ARM_JOINTS

        kin = Kinematics(tip="j6_link")
        assert kin.joint_names == ARM_JOINTS
        if names is not None and list(names) != ARM_JOINTS:
            raise ValueError(f"limits are defined for {ARM_JOINTS}")
        return cls(list(ARM_JOINTS), kin.lower.copy(), kin.upper.copy(), kin.max_velocity.copy())

    def index(self, name: str) -> int | None:
        return self.names.index(name) if name in self.names else None


def check_trajectory(limits: JointLimits, joint_names: list[str], times, points, current=None,
                     speed_tolerance: float = 0.02, max_start_jump: float = np.radians(2.0)) -> list[str]:
    """Problems with a trajectory ([] = acceptable). Joints not in `limits` (e.g. the gripper) are not checked.

    times: (n,) seconds from start; points: (n, len(joint_names)) rad; current: positions now (same order)."""
    times = np.asarray(times, float)
    points = np.atleast_2d(np.asarray(points, float))
    problems: list[str] = []
    if len(times) != len(points) or len(times) == 0:
        return ["trajectory has no points or mismatched times / points"]
    if times[0] < 0 or np.any(np.diff(times) <= 0):
        problems.append("times must start at >= 0 and increase strictly")
        return problems
    for c, name in enumerate(joint_names):
        j = limits.index(name)
        if j is None:
            continue
        q = points[:, c]
        lo, hi, vmax = limits.lower[j], limits.upper[j], limits.max_velocity[j]
        if q.min() < lo - 1e-6 or q.max() > hi + 1e-6:
            bad = q.min() if q.min() < lo else q.max()
            problems.append(f"{name}: {np.degrees(bad):.1f} deg is outside [{np.degrees(lo):.0f}, {np.degrees(hi):.0f}]")
        path_q, path_t = q, times
        if current is not None:
            jump = abs(q[0] - current[c])
            if times[0] == 0.0 and jump > max_start_jump:
                problems.append(f"{name}: starts {np.degrees(jump):.1f} deg away from the current position")
            elif times[0] > 0.0:  # first segment runs from the current position
                path_q, path_t = np.concatenate([[current[c]], q]), np.concatenate([[0.0], times])
        if len(path_q) > 1:
            speed = np.abs(np.diff(path_q)) / np.diff(path_t)
            k = int(np.argmax(speed))
            if speed[k] > vmax * (1 + speed_tolerance):
                problems.append(f"{name}: {np.degrees(speed[k]):.0f} deg/s at t = {path_t[k]:.2f} s exceeds "
                                f"the {np.degrees(vmax):.1f} deg/s limit")
    return problems


def clamp_command(limits: JointLimits, name: str, goal: float, speed: float) -> tuple[float, float, list[str]]:
    """Direct joint command: (goal inside the range, speed <= limit, notes on anything changed)."""
    j = limits.index(name)
    if j is None:
        return goal, speed, []
    notes = []
    g = float(np.clip(goal, limits.lower[j], limits.upper[j]))
    if g != goal:
        notes.append(f"{name}: goal {np.degrees(goal):.1f} deg clamped to {np.degrees(g):.1f}")
    s = float(min(speed, limits.max_velocity[j]))
    if s < speed:
        notes.append(f"{name}: speed {np.degrees(speed):.0f} deg/s clamped to {np.degrees(s):.1f}")
    return g, s, notes


@dataclass
class VelocityMonitor:
    limits: JointLimits
    tolerance: float = 0.05  # measured speeds are noisy: report above limit * (1 + tolerance)
    peak: dict = field(default_factory=dict)  # joint -> peak ratio of the excursion in progress
    events: int = 0

    def update(self, joint_names: list[str], velocities, t: float) -> list[str]:
        """Feed one /joint_states sample; returns messages for excursions that start or end now."""
        out = []
        for name, v in zip(joint_names, velocities):
            j = self.limits.index(name)
            if j is None:
                continue
            ratio = abs(float(v)) / self.limits.max_velocity[j]
            over = ratio > 1 + self.tolerance
            if over and name not in self.peak:
                self.peak[name] = ratio
                self.events += 1
                out.append(f"t={t:.2f} s {name}: measured {np.degrees(abs(v)):.0f} deg/s, limit "
                           f"{np.degrees(self.limits.max_velocity[j]):.1f} deg/s")
            elif over:
                self.peak[name] = max(self.peak[name], ratio)
            elif name in self.peak:
                out.append(f"t={t:.2f} s {name}: back within limit (peak {self.peak.pop(name) * 100:.0f} % of the limit)")
        return out
