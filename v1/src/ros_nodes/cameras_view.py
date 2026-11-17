#!/usr/bin/env python3
"""ROS 2 node: both cameras in ONE window, a 2x2 grid.

    table RGB | table depth   (on a stand, eye-to-hand)
    wrist RGB | wrist depth   (on J5, eye-in-hand)

Runs with the system Python and ROS 2 Jazzy (not a venv):
    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 ros_nodes/cameras_view.py

Window keys: q / Esc close, s save a snapshot to out/snapshots/.
Drawing helpers are shared with ros_nodes/camera_viewer.py (single camera, mouse depth readout).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_viewer import SNAPSHOT_DIR, RateMeter, colorize_depth, label  # noqa: E402

WINDOW = "VT6L cameras"  # ASCII only (Qt backend)
PANE_W, PANE_H = 640, 360
# camera name, topic prefix, depth colour range (m) or None for colour only
CAMERAS = (("table", "table_camera", (1.2, 3.2)), ("wrist", "camera", (0.1, 1.5)))


class CamerasView(Node):
    def __init__(self) -> None:
        super().__init__("cameras_view")
        self.bridge = CvBridge()
        self.frames = {name: {"rgb": None, "depth": None, "rate": RateMeter()} for name, *_ in CAMERAS}
        for name, prefix, depth_range in CAMERAS:
            self.create_subscription(Image, f"{prefix}/color/image_raw",
                                     lambda m, n=name: self.on_rgb(n, m), qos_profile_sensor_data)
            if depth_range:
                self.create_subscription(Image, f"{prefix}/depth/image_rect_raw",
                                         lambda m, n=name: self.on_depth(n, m), qos_profile_sensor_data)

    def on_rgb(self, name: str, msg: Image) -> None:
        self.frames[name]["rgb"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.frames[name]["rate"].tick()

    def on_depth(self, name: str, msg: Image) -> None:
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32)
        self.frames[name]["depth"] = depth / 1000.0 if msg.encoding == "16UC1" else depth

    def compose(self) -> np.ndarray:
        panes = []
        for name, _, depth_range in CAMERAS:
            f = self.frames[name]
            panes.append(self.pane(f["rgb"], f"{name}  {f['rate'].hz:4.1f} Hz", f"{name}: waiting for images..."))
            if depth_range:
                d_min, d_max = depth_range
                depth = None if f["depth"] is None else colorize_depth(f["depth"], d_min, d_max)
                panes.append(self.pane(depth, f"{name} depth  {d_min:.1f}-{d_max:.1f} m (near = red)", ""))
        return np.vstack([np.hstack(panes[:2]), np.hstack(panes[2:4])])

    @staticmethod
    def pane(image: np.ndarray | None, text: str, waiting: str) -> np.ndarray:
        if image is None:
            out = np.zeros((PANE_H, PANE_W, 3), np.uint8)
            label(out, waiting, (10, 22))
            return out
        out = cv2.resize(image, (PANE_W, PANE_H), interpolation=cv2.INTER_AREA)
        label(out, text, (10, 22))
        return out


def main() -> None:
    rclpy.init(args=sys.argv[1:])
    view = CamerasView()
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while rclpy.ok():
            rclpy.spin_once(view, timeout_sec=0.03)
            frame = view.compose()
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s"):
                SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                path = SNAPSHOT_DIR / time.strftime("cameras_%Y%m%d_%H%M%S.png")
                cv2.imwrite(str(path), frame)
                view.get_logger().info(f"saved {path}")
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        view.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
