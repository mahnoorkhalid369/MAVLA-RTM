"""Joint-limit enforcement that does not trust the planner (numpy only, no ROS / Isaac Sim).

Two layers, used by the simulator's command interface (sim/ros_control.py):
  1. check_trajectory(): a command validator. A joint trajectory is rejected before execution if any
     point is outside the joint range, any segment is faster than the joint's maximum speed, the times
     are not increasing, or it starts away from where the arm is (a jump = infinite speed).
     It is also rejected if any point puts J1/J2/J3 in a combination the Epson manual forbids
     (forbidden_combination(), VT series manual Rev.6, 5.3 p.88), on every point, not only the target.
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


# Epson VT6L manual (VT series Rev.6) 5.3 "Restriction of Manipulator Operation by Joint Angle Combination", p.88:
# J2/J3 combinations (deg) in which the arms would interfere with each other; the controller stops the robot there.
# Our joint angles use Epson's zero position and directions (assets/vt6l/vt6l.urdf), so the chart applies directly.
FORBIDDEN_J2_J3_FRONT = [(-160.0, -51.0), (-160.0, 30.0), (-120.0, -20.0), (-110.0, -20.0), (-85.0, -51.0)]  # |J1| <= 110
FORBIDDEN_J2_J3_BACK = [(-160.0, -51.0), (-160.0, 190.0), (-120.0, 190.0), (-120.0, 0.0), (-60.0, -40.0), (65.0, -40.0),
                        (65.0, -51.0)]  # 110 < |J1| <= 170
FORBIDDEN_J1_SPLIT_DEG = 110.0
# The manual warns the inside/outside decision may differ near the boundary (calculation error): we stay this far out,
# and within this margin of |J1| = 110 deg both charts apply.
FORBIDDEN_MARGIN_DEG = 2.0


def _inside(x: float, y: float, poly) -> bool:
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _distance(x: float, y: float, poly) -> float:
    d = np.inf
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        a, b, p = np.array([x1, y1]), np.array([x2, y2]), np.array([x, y])
        t = np.clip((p - a) @ (b - a) / max((b - a) @ (b - a), 1e-12), 0.0, 1.0)
        d = min(d, float(np.linalg.norm(p - (a + t * (b - a)))))
    return d


def forbidden_combination(q, margin_deg: float = FORBIDDEN_MARGIN_DEG) -> str | None:
    """None if J1/J2/J3 (rad, q[0:3]) are allowed; else why (Epson manual 5.3), including margin_deg around the zones."""
    j1, j2, j3 = (float(np.degrees(v)) for v in np.asarray(q, float)[:3])
    charts = []
    if abs(j1) <= FORBIDDEN_J1_SPLIT_DEG + margin_deg:
        charts.append(("|J1| <= 110", FORBIDDEN_J2_J3_FRONT))
    if abs(j1) > FORBIDDEN_J1_SPLIT_DEG - margin_deg:
        charts.append(("|J1| > 110", FORBIDDEN_J2_J3_BACK))
    for label, poly in charts:
        if _inside(j2, j3, poly) or _distance(j2, j3, poly) < margin_deg:
            return (f"J1 {j1:.0f}, J2 {j2:.0f}, J3 {j3:.0f} deg is in (or within {margin_deg:.0f} deg of) a J2/J3 "
                    f"combination the Epson manual forbids for {label} (the arms would interfere)")
    return None


def forbidden_on_path(positions) -> str | None:
    """The first forbidden J1/J2/J3 combination along a joint path (n x >=3, rad), or None."""
    for k, q in enumerate(np.atleast_2d(np.asarray(positions, float))):
        why = forbidden_combination(q)
        if why:
            return f"point {k}: {why}"
    return None


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
    arm = [limits.names[i] for i in range(3)]  # J1, J2, J3
    if all(n in joint_names for n in arm):
        cols = [list(joint_names).index(n) for n in arm]
        why = forbidden_on_path(points[:, cols])
        if why:
            problems.append(why)
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
