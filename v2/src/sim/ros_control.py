"""ROS 2 command interface for the live simulation.

Runs inside the Isaac Sim process (rclpy from the sourced ROS 2 Jazzy install).

    /joint_trajectory  trajectory_msgs/JointTrajectory  in   planned, time-stamped joint path
                                                              (coordinated moves, straight lines);
                                                              played back against simulation time
    /joint_command     sensor_msgs/JointState           in   direct targets (rad); optional velocity
                                                              per joint = max speed (rad/s); ramps
                                                              there and cancels any running trajectory
    /joint_states      sensor_msgs/JointState           out  measured positions / velocities
    /safety/events     std_msgs/String                  out  rejected trajectories, clamped commands,
                                                              measured speeds over the limit
    /sim/ground_truth/objects  std_msgs/String (JSON)   out  SIMULATOR GROUND TRUTH object poses, for
                                                              evaluation and known-pose baselines only

With `limits` (safety/joint_limits.py), commands are checked here, independently of the planner:
trajectories outside the joint ranges or faster than the joint speed limits are rejected (not
executed), direct commands are clamped, and measured speeds are monitored.
"""

from __future__ import annotations

import numpy as np

try:
    import rclpy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from trajectory_msgs.msg import JointTrajectory
except ImportError as exc:  # pragma: no cover - environment problem, explained to the user
    raise ImportError("rclpy not found: run `source /opt/ros/jazzy/setup.bash` before starting the sim") from exc


from safety.joint_limits import JointLimits, VelocityMonitor, check_trajectory, clamp_command  # noqa: E402


class JointCommandInterface:
    def __init__(self, joint_names: list[str], initial: np.ndarray, default_speeds: np.ndarray,
                 limits: JointLimits | None = None) -> None:
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("vt6l_sim")
        self.names = list(joint_names)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.target = np.array(initial, dtype=float)  # what the drives are told now
        self.goal = self.target.copy()  # where each joint is ramping to (direct commands)
        self.default_speeds = np.array(default_speeds, dtype=float)
        self.speeds = self.default_speeds.copy()
        self.trajectory: tuple[list[int], np.ndarray, np.ndarray] | None = None  # (joints, times, points)
        self.trajectory_time = 0.0
        self.node.create_subscription(JointState, "joint_command", self._on_command, 10)
        self.node.create_subscription(JointTrajectory, "joint_trajectory", self._on_trajectory, 10)
        self.state_pub = self.node.create_publisher(JointState, "joint_states", 10)
        self.safety_pub = self.node.create_publisher(String, "safety/events", 10)
        self.limits = limits
        self.monitor = VelocityMonitor(limits) if limits is not None else None
        self.measured = np.array(initial, dtype=float)

    def _safety(self, text: str) -> None:
        print(f"[safety] {text}", flush=True)
        self.safety_pub.publish(String(data=text))

    def _on_command(self, msg: JointState) -> None:
        if self.trajectory is not None and any(n in self.index and self.index[n] in self.trajectory[0] for n in msg.name):
            self.goal[:] = self.target  # cancel: hold where the trajectory had got to
            self.trajectory = None
        for i, name in enumerate(msg.name):
            if name not in self.index or i >= len(msg.position):
                continue
            j = self.index[name]
            has_speed = i < len(msg.velocity) and msg.velocity[i] > 0
            goal, speed = msg.position[i], (msg.velocity[i] if has_speed else self.default_speeds[j])
            if self.limits is not None:
                goal, speed, notes = clamp_command(self.limits, name, goal, speed)
                for note in notes:
                    self._safety(f"command clamped: {note}")
            self.goal[j], self.speeds[j] = goal, speed

    def _on_trajectory(self, msg: JointTrajectory) -> None:
        joints = [self.index[n] for n in msg.joint_names if n in self.index]
        columns = [k for k, n in enumerate(msg.joint_names) if n in self.index]
        if not joints or not msg.points:
            return
        times = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points])
        points = np.array([[p.positions[k] for k in columns] for p in msg.points])
        if self.limits is not None:
            names = [self.names[j] for j in joints]
            problems = check_trajectory(self.limits, names, times, points, current=self.measured[joints])
            if problems:
                self._safety("trajectory REJECTED (not executed): " + "; ".join(problems))
                return
        self.trajectory, self.trajectory_time = (joints, times, points), 0.0
        self.goal[joints] = points[-1]

    def step(self, dt: float) -> np.ndarray:
        """Process incoming messages and return the drive targets for this physics step."""
        rclpy.spin_once(self.node, timeout_sec=0.0)
        if self.trajectory is not None:
            joints, times, points = self.trajectory
            self.trajectory_time += dt
            for c, j in enumerate(joints):
                self.target[j] = np.interp(self.trajectory_time, times, points[:, c])
            if self.trajectory_time >= times[-1]:
                self.trajectory = None
            others = [j for j in range(len(self.names)) if j not in joints]
        else:
            others = list(range(len(self.names)))
        # joints not in a trajectory ramp towards their direct-command goal
        o = np.array(others, dtype=int)
        self.target[o] += np.clip(self.goal[o] - self.target[o], -self.speeds[o] * dt, self.speeds[o] * dt)
        return self.target

    def publish_state(self, positions: np.ndarray, velocities: np.ndarray, sim_time: float | None = None) -> None:
        self.measured = np.array(positions, dtype=float)
        if self.monitor is not None:
            t = self.node.get_clock().now().nanoseconds * 1e-9
            for event in self.monitor.update(self.names, velocities, t):
                self._safety(f"speed monitor: {event}")
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        if sim_time is not None:
            msg.header.stamp.sec = int(sim_time)
            msg.header.stamp.nanosec = int((sim_time - int(sim_time)) * 1e9)
        msg.name = self.names
        msg.position = [float(v) for v in positions]
        msg.velocity = [float(v) for v in velocities]
        self.state_pub.publish(msg)

    def close(self) -> None:
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class CameraPosePublisher:
    """Optical camera extrinsics from the current USD transform, on the simulation clock. No object poses."""

    def __init__(self, node, cameras):
        from geometry_msgs.msg import PoseStamped
        from isaacsim.core.experimental.prims import XformPrim
        self.message = PoseStamped
        self.cameras = [(node.create_publisher(PoseStamped, f"{prefix}/optical_pose", 10), path)
                        for prefix, path in cameras]
        self.view = XformPrim([path for _, path in self.cameras])

    def publish(self, sim_time):
        from isaacsim.core.experimental.utils.backend import use_backend
        try:
            with use_backend('usdrt'):
                positions, orientations = self.view.get_world_poses()
        except RuntimeError as exc:
            if 'Fabric Scene Delegate' not in str(exc):
                raise
            positions, orientations = self.view.get_world_poses()
        positions, orientations = positions.numpy(), orientations.numpy()
        for i, (publisher, path) in enumerate(self.cameras):
            xyz, q = positions[i].tolist(), orientations[i].tolist()  # Isaac quaternion order is wxyz
            msg = self.message()
            msg.header.frame_id = "world"
            msg.header.stamp.sec = int(sim_time)
            msg.header.stamp.nanosec = int((sim_time - int(sim_time)) * 1e9)
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = xyz
            msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z = q
            publisher.publish(msg)


class GroundTruthPublisher:
    """Simulator ground truth of the test objects, as JSON on /sim/ground_truth/objects:
    {"sim_time": s, "objects": {name: {"position": [x, y, z] m, "quat_wxyz": [w, x, y, z]}}}.
    For evaluation and for the known-pose baseline experiment; the perception pipeline must never read it."""

    def __init__(self, node, names: list[str]) -> None:
        import json

        self._json = json
        self.names = list(names)
        self.pub = node.create_publisher(String, "sim/ground_truth/objects", 10)

    def publish(self, sim_time: float, positions: np.ndarray, orientations: np.ndarray) -> None:
        objects = {n: {"position": [round(float(v), 5) for v in p], "quat_wxyz": [round(float(v), 5) for v in o]}
                   for n, p, o in zip(self.names, positions, orientations)}
        self.pub.publish(String(data=self._json.dumps({"sim_time": round(sim_time, 4), "objects": objects})))
