"""Robot execution over ROS 2 (the spec's "Execution agent"): runs planned steps on the simulated
VT6L and reports what actually happened. The same interface can later be backed by an Epson
controller instead of the simulator.

    ex = RosExecutor()                      # creates its own node
    result = ex.execute(steps)              # MoveStep / GripperStep from control/skills.py
    if not result.ok: print(result.failed_step, result.reason)

Uses the simulator's interface only: /joint_trajectory and /joint_command out; /joint_states,
/clock and /safety/events in. A trajectory the simulator's safety layer rejects fails the step.
Optional: ground_truth() reads /sim/ground_truth/objects (simulation only, for evaluation).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from control.robot import ARM_JOINTS, GRIPPER_JOINT, GRIPPER_SPEED
from control.skills import GripperStep, MoveStep


@dataclass
class ExecutionResult:
    ok: bool
    failed_step: str = ""
    reason: str = ""
    log: list[str] = field(default_factory=list)  # one line per executed step
    steps: list[dict] = field(default_factory=list)  # per step: label, planned_s, max_tracking_deg, settled_deg


class RosExecutor:
    def __init__(self, node=None, reach_tolerance_deg: float = 1.5, timeout_factor: float = 3.0) -> None:
        if not rclpy.ok():
            rclpy.init()
        self.node = node or rclpy.create_node("vt6l_executor")
        self.reach_tol = np.radians(reach_tolerance_deg)  # a step fails if it ends further off than this
        self.settle_tol = np.radians(0.1)  # wait this close before the next step
        self.timeout_factor = timeout_factor
        self.joints: dict[str, float] = {}
        self.sim_time = 0.0
        self.safety_events: list[str] = []
        self.truth: dict | None = None
        self._truth_subscription = None
        self.node.create_subscription(JointState, "joint_states", lambda m: self.joints.update(zip(m.name, m.position)), 10)
        self.node.create_subscription(Clock, "clock", self._on_clock, qos_profile_sensor_data)
        self.node.create_subscription(String, "safety/events", lambda m: self.safety_events.append(m.data), 10)
        self.traj_pub = self.node.create_publisher(JointTrajectory, "joint_trajectory", 10)
        self.cmd_pub = self.node.create_publisher(JointState, "joint_command", 10)

    def _on_clock(self, msg: Clock) -> None:
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    # ---------------------------------------------------------------- state
    def spin(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def wait_ready(self, timeout: float = 30.0, need_truth: bool = False) -> bool:
        if need_truth:
            self._subscribe_truth()
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.spin(0.1)
            if all(j in self.joints for j in ARM_JOINTS + [GRIPPER_JOINT]) and self.sim_time > 0 and (self.truth or not need_truth):
                return True
        return False

    def q(self) -> np.ndarray:
        return np.array([self.joints[j] for j in ARM_JOINTS])

    def gripper(self) -> float:
        return float(self.joints[GRIPPER_JOINT])

    def wait_sim(self, seconds: float) -> None:
        """Wait for `seconds` of SIMULATION time (the sim may run slower or faster than real time)."""
        start, deadline = self.sim_time, time.monotonic() + max(30.0, 20 * seconds)
        while self.sim_time - start < seconds and time.monotonic() < deadline:
            self.spin(0.02)

    def ground_truth(self, fresh_after: float | None = None) -> dict:
        """Latest simulator ground truth {name: {"position", "quat_wxyz"}} (simulation / evaluation only)."""
        self._subscribe_truth()
        t0 = self.sim_time if fresh_after is None else fresh_after
        end = time.monotonic() + 5.0
        while time.monotonic() < end and (self.truth is None or self.truth["sim_time"] < t0):
            self.spin(0.05)
        return self.truth["objects"] if self.truth else {}

    def _subscribe_truth(self):
        """Explicit evaluation opt-in. Normal agent execution never subscribes to this topic."""
        if self._truth_subscription is None:
            self._truth_subscription = self.node.create_subscription(
                String, "sim/ground_truth/objects", lambda m: setattr(self, "truth", json.loads(m.data)), 10)

    # ------------------------------------------------------------ commands
    def set_gripper(self, position: float) -> None:
        self.cmd_pub.publish(JointState(name=[GRIPPER_JOINT], position=[float(position)], velocity=[float(GRIPPER_SPEED)]))

    def send_trajectory(self, step: MoveStep) -> None:
        msg = JointTrajectory()
        msg.joint_names = list(ARM_JOINTS)
        for t, q in zip(step.trajectory.times, step.trajectory.positions):
            sec = int(t)
            msg.points.append(JointTrajectoryPoint(positions=[float(v) for v in q],
                                                   time_from_start=Duration(sec=sec, nanosec=int((t - sec) * 1e9))))
        self.traj_pub.publish(msg)

    def execute(self, steps: list) -> ExecutionResult:
        log: list[str] = []
        steps_out: list[dict] = []
        for s in steps:
            wall_start, sim_start = time.monotonic(), self.sim_time
            if isinstance(s, GripperStep):
                self.set_gripper(s.position)
                self.wait_sim(s.settle)
                # then until the fingers reach the target or stop moving (stalled on an object), max 3 s
                deadline, last = self.sim_time + 3.0, self.gripper()
                wall_deadline = time.monotonic() + 60.0
                while (abs(self.gripper() - s.position) > np.radians(1.0) and self.sim_time < deadline
                       and time.monotonic() < wall_deadline):
                    self.wait_sim(0.2)
                    if abs(self.gripper() - last) < np.radians(0.2):
                        break
                    last = self.gripper()
                log.append(f"{s.label}: finger_joint {np.degrees(self.gripper()):.1f} deg (target {np.degrees(s.position):.0f})")
                completed = self.sim_time-sim_start >= s.settle
                steps_out.append(self._timing(s.label, s.settle, wall_start, sim_start, kind="gripper", ok=completed))
                if not completed:
                    return ExecutionResult(False, s.label, "simulation clock stalled during gripper settling", log, steps_out)
                continue
            n_events = len(self.safety_events)
            t0 = time.monotonic()
            self.send_trajectory(s)
            # while it runs: largest gap between the measured joints and the plan at the same sim time
            start, track = self.sim_time, 0.0
            traj = s.trajectory
            while self.sim_time - start < traj.duration and time.monotonic() - t0 < 30 + 20 * traj.duration:
                self.spin(0.02)
                t = self.sim_time - start
                planned = np.array([np.interp(t, traj.times, traj.positions[:, j]) for j in range(traj.positions.shape[1])])
                track = max(track, float(np.max(np.abs(self.q() - planned))))
            goal = s.trajectory.positions[-1]
            motion_wall, motion_sim = time.monotonic(), self.sim_time
            deadline_sim = self.sim_time + max(2.0, s.trajectory.duration * (self.timeout_factor - 1))
            deadline_wall = time.monotonic() + max(30.0, 20 * (deadline_sim - self.sim_time))
            while (np.max(np.abs(self.q() - goal)) > self.reach_tol and self.sim_time < deadline_sim
                   and time.monotonic() < deadline_wall):
                self.spin(0.02)
            # settle: until within settle_tol, or the arm has STOPPED moving (a steady offset, e.g. J5
            # while the simulated gripper squeezes an object), max 3 s of sim time; so the next step
            # (e.g. closing the gripper) starts from where the arm really comes to rest
            settle_end, last = self.sim_time + 3.0, self.q()
            settle_wall = time.monotonic() + 60.0
            while (np.max(np.abs(self.q() - goal)) > self.settle_tol and self.sim_time < settle_end
                   and time.monotonic() < settle_wall):
                self.wait_sim(0.1)
                if np.max(np.abs(self.q() - last)) < np.radians(0.005):
                    break
                last = self.q()
            rejected = [e for e in self.safety_events[n_events:] if "REJECTED" in e]
            per_joint = np.degrees(self.q() - goal)
            err = float(np.max(np.abs(per_joint)))
            worst = f"J{int(np.argmax(np.abs(per_joint))) + 1}"
            steps_out.append({**self._timing(s.label, traj.duration, wall_start, sim_start, kind="move"),
                              "motion_wall_s": motion_wall-wall_start, "motion_sim_s": motion_sim-sim_start,
                              "settling_wall_s": time.monotonic()-motion_wall,
                              "settling_sim_s": self.sim_time-motion_sim,
                              "max_tracking_deg": round(float(np.degrees(track)), 3), "settled_deg": round(err, 3),
                              "settled_per_joint_deg": [round(float(v), 3) for v in per_joint],
                              "ok": not rejected and err <= np.degrees(self.reach_tol)
                                    and motion_sim-sim_start >= traj.duration})
            if rejected:
                return ExecutionResult(False, s.label, "safety layer rejected the trajectory: " + rejected[0], log, steps_out)
            if motion_sim-sim_start < traj.duration:
                return ExecutionResult(False, s.label, "simulation clock stalled before trajectory completion", log, steps_out)
            if err > np.degrees(self.reach_tol):
                return ExecutionResult(False, s.label, f"did not reach the goal: {err:.2f} deg off (controller timeout)", log, steps_out)
            log.append(f"{s.label}: settled {err:.3f} deg from the goal ({worst}), max tracking error {np.degrees(track):.2f} deg "
                       f"({s.trajectory.duration:.1f} s planned, {time.monotonic() - t0:.1f} s wall)")
        return ExecutionResult(True, log=log, steps=steps_out)

    def _timing(self, label, planned, wall_start, sim_start, **fields):
        wall_end = time.monotonic()
        sim_elapsed = self.sim_time - sim_start
        return {"label": label, "planned_s": float(planned), "sim_start_s": sim_start,
                "sim_end_s": self.sim_time, "sim_elapsed_s": sim_elapsed,
                "start_s": wall_start, "end_s": wall_end, "wall_s": wall_end-wall_start,
                "real_time_factor": sim_elapsed/(wall_end-wall_start) if wall_end > wall_start else None, **fields}
