#!/usr/bin/env python3
"""ROS 2 node: live view of the wrist camera's RGB and depth streams side by side.

Runs with the system Python and ROS 2 Jazzy (not the Isaac Sim venv):

    source /opt/ros/jazzy/setup.bash
    python3 ros_nodes/camera_viewer.py                 # live window
    python3 ros_nodes/camera_viewer.py --depth-max 2   # colour scale 0.1-2.0 m
    python3 ros_nodes/camera_viewer.py --snapshot out/view.png   # save one frame and exit

Window keys: q / Esc quit, s save a snapshot to out/snapshots/.
Hover the mouse over either half to read the depth at that pixel.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

WINDOW = "D435 wrist camera - RGB and depth"  # ASCII only: the Qt backend fails on other characters
PANE_WIDTH = 640
SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "out" / "snapshots"


class RateMeter:
    def __init__(self, window_s: float = 2.0) -> None:
        self.stamps: deque[float] = deque()
        self.window_s = window_s

    def tick(self) -> None:
        now = time.monotonic()
        self.stamps.append(now)
        while self.stamps and now - self.stamps[0] > self.window_s:
            self.stamps.popleft()

    @property
    def hz(self) -> float:
        if len(self.stamps) < 2:
            return 0.0
        return (len(self.stamps) - 1) / (self.stamps[-1] - self.stamps[0])


class CameraViewer(Node):
    def __init__(self, color_topic: str, depth_topic: str) -> None:
        super().__init__("camera_viewer")
        self.bridge = CvBridge()
        self.color: np.ndarray | None = None  # BGR
        self.depth: np.ndarray | None = None  # metres, float32
        self.color_rate, self.depth_rate = RateMeter(), RateMeter()
        self.create_subscription(Image, color_topic, self.on_color, qos_profile_sensor_data)
        self.create_subscription(Image, depth_topic, self.on_depth, qos_profile_sensor_data)
        self.get_logger().info(f"subscribed to {color_topic} and {depth_topic}")

    def on_color(self, msg: Image) -> None:
        self.color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.color_rate.tick()

    def on_depth(self, msg: Image) -> None:
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32)
        if msg.encoding == "16UC1":  # RealSense hardware convention: millimetres
            depth = depth / 1000.0
        self.depth = depth
        self.depth_rate.tick()


def colorize_depth(depth: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    scaled = np.clip((np.nan_to_num(depth) - d_min) / (d_max - d_min), 0.0, 1.0)
    image = cv2.applyColorMap((255 - scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)  # near = red
    image[~valid] = 0
    return image


def label(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.5) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def color_bar(width: int, d_min: float, d_max: float) -> np.ndarray:
    ramp = np.linspace(255, 0, width).astype(np.uint8)[None, :].repeat(14, axis=0)
    bar = np.zeros((34, width, 3), np.uint8)
    bar[:14] = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = int(frac * (width - 1))
        value = d_min + frac * (d_max - d_min)
        text = f"{value:.2f} m"
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        cv2.putText(bar, text, (min(max(x - size // 2, 0), width - size), 29), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    return bar


def compose(viewer: CameraViewer, d_min: float, d_max: float, mouse: tuple[int, int] | None) -> np.ndarray:
    h, w = viewer.color.shape[:2]
    scale = PANE_WIDTH / w
    pane_h = int(h * scale)
    rgb = cv2.resize(viewer.color, (PANE_WIDTH, pane_h), interpolation=cv2.INTER_AREA)
    depth_img = cv2.resize(colorize_depth(viewer.depth, d_min, d_max), (PANE_WIDTH, pane_h), interpolation=cv2.INTER_NEAREST)

    valid = viewer.depth[np.isfinite(viewer.depth) & (viewer.depth > 0)]
    label(rgb, f"RGB {w}x{h}  {viewer.color_rate.hz:4.1f} Hz", (10, 22))
    stats = f"{valid.min():.2f}-{valid.max():.2f} m" if valid.size else "no returns"
    label(depth_img, f"Depth {viewer.depth.shape[1]}x{viewer.depth.shape[0]}  {viewer.depth_rate.hz:4.1f} Hz  {stats}", (10, 22))

    if mouse is not None:
        mx, my = mouse
        px = mx % PANE_WIDTH  # same pixel on either half
        if 0 <= my < pane_h:
            dx, dy = int(px / scale), int(my / scale)
            d = viewer.depth[min(dy, viewer.depth.shape[0] - 1), min(dx, viewer.depth.shape[1] - 1)]
            text = f"{d:.3f} m" if np.isfinite(d) and d > 0 else "no depth"
            for pane in (rgb, depth_img):
                cv2.drawMarker(pane, (px, my), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
                label(pane, text, (min(px + 10, PANE_WIDTH - 90), max(my - 10, 40)))

    canvas = np.hstack([rgb, depth_img])
    bar = np.zeros((34, canvas.shape[1], 3), np.uint8)
    bar[:, PANE_WIDTH:] = color_bar(PANE_WIDTH, d_min, d_max)
    label(bar, "depth colour scale (near = red, far = blue, black = no return)", (10, 22), 0.45)
    bar[:, PANE_WIDTH:] = np.maximum(bar[:, PANE_WIDTH:], color_bar(PANE_WIDTH, d_min, d_max))
    return np.vstack([canvas, bar])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--color-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/depth/image_rect_raw")
    parser.add_argument("--depth-min", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=1.5)
    parser.add_argument("--title", default=WINDOW, help="window name (ASCII)")
    parser.add_argument("--snapshot", type=Path, default=None, help="save one composite frame and exit")
    args, ros_args = parser.parse_known_args()
    window = args.title

    rclpy.init(args=ros_args)
    viewer = CameraViewer(args.color_topic, args.depth_topic)
    mouse: list[tuple[int, int] | None] = [None]
    if args.snapshot is None:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, lambda event, x, y, *_: mouse.__setitem__(0, (x, y)))
    try:
        while rclpy.ok():
            rclpy.spin_once(viewer, timeout_sec=0.02)
            if viewer.color is None or viewer.depth is None:
                continue
            frame = compose(viewer, args.depth_min, args.depth_max, mouse[0])
            if args.snapshot is not None:
                if viewer.color_rate.hz > 0 and viewer.depth_rate.hz > 0:
                    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(args.snapshot), frame)
                    viewer.get_logger().info(f"saved {args.snapshot}")
                    break
                continue
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s"):
                SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                path = SNAPSHOT_DIR / time.strftime("camera_%Y%m%d_%H%M%S.png")
                cv2.imwrite(str(path), frame)
                viewer.get_logger().info(f"saved {path}")
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        viewer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
