"""ROS 2 bridge for the agent: one node on a background MultiThreadedExecutor (cameras, joint states,
clock); the agent's tools ask for snapshots and wait on them with timeouts.

The robot's motions run through `execution/ros_executor.RosExecutor`, which keeps its own node and
spins it itself during `execute`: two nodes, each spun by exactly
one executor, so there is no shared-spin deadlock. Ground truth is NOT subscribed here.
"""

from __future__ import annotations

import threading
import time

from agent.timing import timed

import numpy as np
from perception.capture import CaptureBuffer, Frame, ObservationError, optical_pose


class RosBridge:
    def __init__(self) -> None:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import CameraInfo, Image, JointState
        from geometry_msgs.msg import PoseStamped

        from control.robot import ARM_JOINTS, GRIPPER_JOINT

        if not rclpy.ok():
            rclpy.init()
        self.arm_joints, self.gripper_joint = ARM_JOINTS, GRIPPER_JOINT
        self.node = rclpy.create_node("vt6l_agent_bridge")
        self.lock = threading.Lock()
        self.latest: dict = {}
        self.capture = CaptureBuffer()
        self.joints: dict[str, float] = {}
        self.sim_time = 0.0
        self.epoch = 0
        for cam, prefix in (("table", "table_camera"), ("wrist", "camera")):
            self.node.create_subscription(Image, f"{prefix}/color/image_raw", lambda m, c=cam: self._img(c, "rgb", m), qos_profile_sensor_data)
            self.node.create_subscription(Image, f"{prefix}/depth/image_rect_raw", lambda m, c=cam: self._img(c, "depth", m), qos_profile_sensor_data)
            for kind, topic in (("rgb", "color"), ("depth", "depth")):
                self.node.create_subscription(CameraInfo, f"{prefix}/{topic}/camera_info",
                                              lambda m, c=cam, k=kind: self._info(c, k, m), qos_profile_sensor_data)
            self.node.create_subscription(PoseStamped, f"{prefix}/optical_pose",
                                          lambda m, c=cam: self._pose(c, m), qos_profile_sensor_data)
        self.node.create_subscription(JointState, "joint_states", self._joints, 10)
        self.node.create_subscription(Clock, "clock", self._clock, qos_profile_sensor_data)
        self.executor = MultiThreadedExecutor(num_threads=3)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------ callbacks
    def _img(self, cam: str, kind: str, m) -> None:
        if kind == "rgb":
            if m.encoding.lower() not in ('rgb8', 'rgba8', 'bgr8', 'bgra8'):
                raise ObservationError(f'unsupported colour encoding {m.encoding}')
            channels = 4 if m.encoding.lower() in ("rgba8", "bgra8") else 3
            rows = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)
            a = rows[:, :m.width * channels].reshape(m.height, m.width, channels)[..., :3].copy()
            if m.encoding.lower().startswith("bgr"):
                a = a[..., ::-1].copy()
        else:
            if m.encoding not in ("32FC1", "16UC1"):
                raise ObservationError(f"unsupported depth encoding {m.encoding}")
            dtype = np.dtype(np.float32 if m.encoding == "32FC1" else np.uint16).newbyteorder(">" if m.is_bigendian else "<")
            rows = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)
            a = rows[:, :m.width * dtype.itemsize].copy().view(dtype).reshape(m.height, m.width).astype(np.float32)
            if m.encoding == "16UC1":
                a /= 1000.0
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        with self.lock:
            self.latest[(cam, kind)] = (a, stamp)
            self.capture.images[cam, kind].append((a, stamp, m.header.frame_id))

    def _info(self, cam: str, kind: str, m) -> None:
        with self.lock:
            K = (m.k[0], m.k[4], m.k[2], m.k[5])
            self.latest[(cam, "K")] = K
            stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            self.capture.intrinsics[cam, kind].append((K, (m.height, m.width), m.header.frame_id, stamp))

    def _pose(self, cam, m):
        if m.header.frame_id != "world":
            return
        p, q = m.pose.position, m.pose.orientation
        pose = optical_pose([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        with self.lock:
            self.capture.poses[cam].append((stamp, pose))

    def _joints(self, m) -> None:
        with self.lock:
            self.joints.update(zip(m.name, m.position))
            if all(j in self.joints for j in self.arm_joints):
                stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                self.capture.joints.append((stamp, np.array([self.joints[j] for j in self.arm_joints]),
                                            float(self.joints.get(self.gripper_joint, 0))))

    def _clock(self, m) -> None:
        stamp = m.clock.sec + m.clock.nanosec * 1e-9
        with self.lock:
            if stamp < self.sim_time - .01:
                self.capture = CaptureBuffer()
                self.latest.clear()
                self.epoch += 1
            self.sim_time = stamp

    # --------------------------------------------------------------------- queries
    def ready(self, timeout: float = 120.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self.lock:
                ok = (all(j in self.joints for j in self.arm_joints) and self.capture.poses["table"]
                      and ("table", "depth") in self.latest and ("table", "rgb") in self.latest)
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

    @timed("settling")
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
        raise TimeoutError("arm did not settle before RGB-D capture")

    @timed("capture")
    def snapshot(self, camera: str = "table", timeout: float = 10.0) -> Frame:
        """A frame (RGB + depth) rendered AFTER this call, with the arm still."""
        self.wait_still()
        t_req = self.sim_time
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self.lock:
                frame = self.capture.select(camera, t_req + .05)
            if frame is not None:
                frame.epoch = self.epoch
                return frame
            time.sleep(0.05)
        raise ObservationError(f"no synchronized {camera} RGB-D/calibration/optical-pose/joint bundle; check sensor publishers")

    def close(self) -> None:
        self.executor.shutdown()
        self.node.destroy_node()
