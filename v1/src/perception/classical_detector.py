"""Classical RGB-D object detection for the table camera (no learned model): depth geometry + colour.

Pipeline (numpy only):
  1. back-project every depth pixel to the world (intrinsics + the camera's extrinsic pose)
  2. keep points 1-25 cm above the floor, away from the robot base and the camera stand
  3. split them into touching groups (5 mm grid)
  4. "peel" each group from the top: the highest top face is one object; remove its column above the
     next level and repeat, so a stack gives one object per block
  5. per object: top-face centre (min-area rectangle: robust to partial occlusion), yaw, footprint,
     top height and measured colour
  6. identify each against the OBJECT CATALOGUE (sim/scene_objects.py: names, colours, sizes, like
     CAD models in a real cell) by colour name and footprint; never against simulator ground truth.

    det = ClassicalDetector()
    found = det.detect(depth, rgb, K, T_world_camera)        # [Detection]
    poses = det.identify(found)                              # {name: (centre xyz, yaw deg)}

TablePoseEstimator(ex) grabs one table-camera frame through ROS, checks it (dimensions, timestamps,
intrinsics, extrinsics against TF) and returns the identified poses: the camera pose source for
misc/experiments/known_pose_pick_place.py (Baseline B, archived).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception.grounding import _blobs, colour_name, footprint_yaw
from perception.world_memory import _with_family

MIN_Z, MAX_Z = 0.010, 0.25  # m above the floor
MIN_POINTS = 25
LAYER = 0.012  # m: points this close below the highest point form the top face
EXCLUDE = [((0.0, 0.0), 0.22), ((1.5, 0.25), 0.18)]  # (xy, radius): robot base, camera stand foot


@dataclass
class Detection:
    centre_xy: np.ndarray  # top-face centre (m, world)
    top_z: float
    yaw_deg: float | None  # None: round footprint
    footprint: tuple[float, float]  # length >= width (m)
    colour: str
    rgb: tuple[int, int, int]
    points: int
    extents: tuple[float, float] = (0.0, 0.0)  # top face along the yaw axis x and y (m)


def _rect_centre(xy: np.ndarray, yaw: float | None) -> np.ndarray:
    """Centre of the min-area rectangle (or of the extent for round shapes): unlike the median, it is not
    pulled towards the visible part when one side of the top face is hidden."""
    a = np.radians(yaw or 0.0)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    local = xy @ R
    mid = (np.percentile(local, 2, axis=0) + np.percentile(local, 98, axis=0)) / 2
    return mid @ R.T


class ClassicalDetector:
    def __init__(self, catalogue: dict | None = None) -> None:
        if catalogue is None:
            from sim.scene_objects import OBJECTS
            catalogue = OBJECTS
        self.catalogue = catalogue

    def world_points(self, depth: np.ndarray, rgb: np.ndarray, K, T: np.ndarray):
        h, w = depth.shape
        fx, fy, cx, cy = K
        v, u = np.mgrid[0:h, 0:w]
        z = depth.astype(np.float64)
        ok = np.isfinite(z) & (z > 0.05) & (z < 6.0)
        cam = np.stack([(u[ok] - cx) / fx * z[ok], (v[ok] - cy) / fy * z[ok], z[ok]], axis=1)
        world = cam @ T[:3, :3].T + T[:3, 3]
        return world, rgb[ok][:, :3]

    def detect(self, depth: np.ndarray, rgb: np.ndarray, K, T: np.ndarray) -> list[Detection]:
        pts, col = self.world_points(depth, rgb, K, T)
        keep = (pts[:, 2] > MIN_Z) & (pts[:, 2] < MAX_Z)
        for xy, rad in EXCLUDE:
            keep &= np.hypot(pts[:, 0] - xy[0], pts[:, 1] - xy[1]) > rad
        pts, col = pts[keep], col[keep]
        out: list[Detection] = []
        if len(pts) < MIN_POINTS:
            return out
        for group in _blobs(pts[:, :2]):
            if len(group) >= MIN_POINTS:
                out += self._peel(pts[group], col[group])
        return out

    def _peel(self, pts: np.ndarray, col: np.ndarray) -> list[Detection]:
        found = []
        for _ in range(4):  # at most 4 stacked blocks
            if len(pts) < MIN_POINTS:
                break
            top = float(np.percentile(pts[:, 2], 99))
            face = pts[:, 2] > top - LAYER
            if face.sum() < MIN_POINTS:
                break
            yaw, fp = footprint_yaw(pts[face, :2])
            centre = _rect_centre(pts[face, :2], yaw)
            if yaw is None:  # round: a disc around the centre
                inside = np.hypot(*(pts[:, :2] - centre).T) <= max(fp) / 2 + 0.003
                extents = (max(fp), max(fp))
            else:  # this object's column: its top-face rectangle (+3 mm) seen from above
                a = np.radians(yaw)
                local = (pts[:, :2] - centre) @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                half = self._half_axes(pts[face, :2], centre, yaw)
                inside = np.all(np.abs(local) <= half + 0.003, axis=1)
                extents = (2 * half[0], 2 * half[1])
            rgb_med = np.median(col[inside].astype(float), axis=0)
            found.append(Detection(centre, top, yaw, fp, colour_name(rgb_med), tuple(int(v) for v in rgb_med),
                                   int(inside.sum()), extents))
            pts, col = pts[~inside], col[~inside]  # what is left: e.g. the visible rim of the block below
        return found

    @staticmethod
    def _half_axes(face_xy: np.ndarray, centre: np.ndarray, yaw: float) -> np.ndarray:
        a = np.radians(yaw)
        local = (face_xy - centre) @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return (np.percentile(local, 98, axis=0) - np.percentile(local, 2, axis=0)) / 2

    def identify(self, detections: list[Detection], size_tol: float = 0.012) -> dict[str, tuple[np.ndarray, float]]:
        """{catalogue name: (centre xyz, yaw deg)}: each detection to the best catalogue entry with a
        compatible colour (same name or close family) and footprint; greedy, best matches first."""
        candidates = []
        for k, d in enumerate(detections):
            for name, o in self.catalogue.items():
                words = set(name.split("_"))
                if not (words & _with_family({d.colour})):
                    continue
                size = sorted(o.size[:2], reverse=True)
                err = max(abs(d.footprint[0] - size[0]), abs(d.footprint[1] - size[1]))
                if err > size_tol or d.top_z < o.height - 0.008:
                    continue
                exact = d.colour in words
                candidates.append((0 if exact else 1, err, k, name))
        poses, used_d, used_n = {}, set(), set()
        for _, _, k, name in sorted(candidates):
            if k in used_d or name in used_n:
                continue
            d, o = detections[k], self.catalogue[name]
            centre = np.array([d.centre_xy[0], d.centre_xy[1], d.top_z - o.height / 2])
            yaw = 0.0 if d.yaw_deg is None else d.yaw_deg
            if o.shape == "box" and abs(o.size[0] - o.size[1]) >= 0.008:
                # rectangle: report the yaw of the catalogue's local x axis (the measured axes may be swapped)
                long_is_x = o.size[0] >= o.size[1]
                if (d.extents[0] >= d.extents[1]) != long_is_x:
                    yaw = yaw + 90.0 if yaw <= 0 else yaw - 90.0
            poses[name] = (centre, yaw)
            used_d.add(k)
            used_n.add(name)
        return poses


class TablePoseEstimator:
    """Camera pose source for the pick-and-place experiments (Baseline B). Callable: est(ex) -> poses."""

    def __init__(self, tf_tolerance_mm: float = 1.0) -> None:
        self.detector = ClassicalDetector()
        self.tf_tol = tf_tolerance_mm / 1000
        self.last_checks: dict = {}
        self.last_detections: list[Detection] = []
        self._subs = None

    def _subscribe(self, node) -> None:
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self.msgs: dict = {}
        node.create_subscription(Image, "table_camera/color/image_raw", lambda m: self.msgs.__setitem__("rgb", m), qos_profile_sensor_data)
        node.create_subscription(Image, "table_camera/depth/image_rect_raw", lambda m: self.msgs.__setitem__("depth", m), qos_profile_sensor_data)
        node.create_subscription(CameraInfo, "table_camera/color/camera_info", lambda m: self.msgs.__setitem__("info", m), qos_profile_sensor_data)
        from tf2_ros import Buffer, TransformListener
        self.tf = Buffer()
        self._tfl = TransformListener(self.tf, node)
        self._subs = True

    def frame(self, ex, timeout: float = 10.0):
        """(rgb, depth, K, T, checks) from one fresh, checked table-camera frame."""
        from control.robot import table_camera_pose

        if self._subs is None:
            self._subscribe(ex.node)
        self.msgs.clear()
        t_req = ex.sim_time
        import time as _t
        end = _t.monotonic() + timeout
        while _t.monotonic() < end and not all(k in self.msgs for k in ("rgb", "depth", "info")):
            ex.spin(0.05)
        if not all(k in self.msgs for k in ("rgb", "depth", "info")):
            raise RuntimeError("no table-camera images (camera rendering not running?)")
        m_rgb, m_d, info = self.msgs["rgb"], self.msgs["depth"], self.msgs["info"]
        rgb = np.frombuffer(m_rgb.data, np.uint8).reshape(m_rgb.height, m_rgb.width, -1)[..., :3]
        depth = np.frombuffer(m_d.data, np.float32 if m_d.encoding == "32FC1" else np.uint16).reshape(m_d.height, m_d.width).astype(np.float32)
        if m_d.encoding == "16UC1":
            depth /= 1000.0
        K = (info.k[0], info.k[4], info.k[2], info.k[5])
        stamp = lambda m: m.header.stamp.sec + m.header.stamp.nanosec * 1e-9  # noqa: E731
        T = table_camera_pose()
        checks = {
            "dimensions": {"rgb": [m_rgb.height, m_rgb.width], "depth": [m_d.height, m_d.width], "info": [info.height, info.width],
                           "ok": (m_rgb.height, m_rgb.width) == (m_d.height, m_d.width) == (info.height, info.width)},
            "timestamps": {"rgb": round(stamp(m_rgb), 3), "depth": round(stamp(m_d), 3), "requested_after": round(t_req, 3),
                           "ok": abs(stamp(m_rgb) - stamp(m_d)) < 0.1 and stamp(m_d) >= t_req - 0.2},
            "intrinsics": {"fx": round(K[0], 1), "fy": round(K[1], 1), "cx": round(K[2], 1), "cy": round(K[3], 1),
                           "ok": abs(K[0] - K[1]) < 1.0 and abs(K[2] - m_d.width / 2) < 2 and abs(K[3] - m_d.height / 2) < 2},
            "depth_valid_fraction": round(float(np.isfinite(depth).mean() * (depth > 0).mean()), 3),
        }
        try:
            from rclpy.time import Time
            t = self.tf.lookup_transform("world", "table_camera_optical_frame", Time()).transform.translation
            d = float(np.linalg.norm(np.array([t.x, t.y, t.z]) - T[:3, 3]))
            checks["extrinsics_vs_tf"] = {"position_error_mm": round(d * 1000, 3), "ok": d < self.tf_tol}
        except Exception as exc:  # noqa: BLE001
            checks["extrinsics_vs_tf"] = {"ok": False, "error": str(exc)}
        self.last_checks = checks
        return rgb, depth, K, T, checks

    def __call__(self, ex) -> dict[str, tuple[np.ndarray, float]]:
        rgb, depth, K, T, checks = self.frame(ex)
        bad = [k for k, v in checks.items() if isinstance(v, dict) and not v.get("ok", True)]
        if bad:
            raise RuntimeError(f"table-camera frame failed checks: {bad}: {checks}")
        self.last_detections = self.detector.detect(depth, rgb, K, T)
        return self.detector.identify(self.last_detections)
