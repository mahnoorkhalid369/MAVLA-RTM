"""ROS 2 bridge for the agent: one node on a background MultiThreadedExecutor (cameras, joint states,
clock); the agent's tools ask for snapshots and wait on them with timeouts.

The robot's motions run through `execution/ros_executor.RosExecutor`, which keeps its own node and
spins it itself during `execute` (it is a frozen, validated file): two nodes, each spun by exactly
one executor, so there is no shared-spin deadlock. Ground truth is NOT subscribed here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    camera: str
    rgb: np.ndarray
    depth: np.ndarray
    K: tuple[float, float, float, float]
    stamp: float
    q: np.ndarray  # arm joints when the frame was taken
    finger: float


class RosBridge:
    def __init__(self) -> None:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import CameraInfo, Image, JointState

        from control.robot import ARM_JOINTS, GRIPPER_JOINT

        if not rclpy.ok():
            rclpy.init()
        self.arm_joints, self.gripper_joint = ARM_JOINTS, GRIPPER_JOINT
        self.node = rclpy.create_node("vt6l_agent_bridge")
        self.lock = threading.Lock()
        self.latest: dict = {}
        self.joints: dict[str, float] = {}
        self.sim_time = 0.0
        for cam, prefix in (("table", "table_camera"), ("wrist", "camera")):
            self.node.create_subscription(Image, f"{prefix}/color/image_raw", lambda m, c=cam: self._img(c, "rgb", m), qos_profile_sensor_data)
            self.node.create_subscription(Image, f"{prefix}/depth/image_rect_raw", lambda m, c=cam: self._img(c, "depth", m), qos_profile_sensor_data)
            self.node.create_subscription(CameraInfo, f"{prefix}/color/camera_info", lambda m, c=cam: self._info(c, m), qos_profile_sensor_data)
        self.node.create_subscription(JointState, "joint_states", self._joints, 10)
        self.node.create_subscription(Clock, "clock", self._clock, qos_profile_sensor_data)
        self.executor = MultiThreadedExecutor(num_threads=3)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------ callbacks
    def _img(self, cam: str, kind: str, m) -> None:
        if kind == "rgb":
            a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[..., :3].copy()
        else:
            a = np.frombuffer(m.data, np.float32 if m.encoding == "32FC1" else np.uint16).reshape(m.height, m.width).astype(np.float32)
            if m.encoding == "16UC1":
                a /= 1000.0
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        with self.lock:
            self.latest[(cam, kind)] = (a, stamp)

    def _info(self, cam: str, m) -> None:
        with self.lock:
            self.latest[(cam, "K")] = (m.k[0], m.k[4], m.k[2], m.k[5])

    def _joints(self, m) -> None:
        with self.lock:
            self.joints.update(zip(m.name, m.position))

    def _clock(self, m) -> None:
        self.sim_time = m.clock.sec + m.clock.nanosec * 1e-9

    # --------------------------------------------------------------------- queries
    def ready(self, timeout: float = 120.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self.lock:
                ok = all(j in self.joints for j in self.arm_joints) and ("table", "rgb") in self.latest and ("table", "K") in self.latest
            if ok and self.sim_time > 0:
                return True
            time.sleep(0.2)
        return False

    def q(self) -> np.ndarray:
        with self.lock:
            return np.array([self.joints[j] for j in self.arm_joints])

    def finger(self) -> float:
        with self.lock:
            return float(self.joints.get(self.gripper_joint, 0.0))

    def wait_still(self, tol_deg: float = 0.05, window_s: float = 0.3, timeout_s: float = 6.0) -> None:
        start, last, since = time.monotonic(), None, None
        while time.monotonic() - start < timeout_s:
            q = self.q()
            if last is not None and np.max(np.abs(q - last)) < np.radians(tol_deg):
                since = since or time.monotonic()
                if time.monotonic() - since >= window_s:
                    return
            else:
                since = None
            last = q
            time.sleep(0.05)

    def snapshot(self, camera: str = "table", timeout: float = 10.0) -> Frame:
        """A frame (RGB + depth) rendered AFTER this call, with the arm still."""
        self.wait_still()
        t_req = self.sim_time
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self.lock:
                rgb, depth, K = self.latest.get((camera, "rgb")), self.latest.get((camera, "depth")), self.latest.get((camera, "K"))
            if rgb and depth and K and rgb[1] > t_req + 0.05 and depth[1] > t_req + 0.05 and abs(rgb[1] - depth[1]) < 0.1:
                return Frame(camera, rgb[0], depth[0], K, rgb[1], self.q(), self.finger())
            time.sleep(0.05)
        raise TimeoutError(f"no fresh {camera} frame (camera rendering not running?)")

    def close(self) -> None:
        self.executor.shutdown()
        self.node.destroy_node()
