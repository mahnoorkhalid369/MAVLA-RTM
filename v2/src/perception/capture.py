"""Timestamp-matched RGB-D and robot/camera metadata. No ROS or object ground truth."""

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


class ObservationError(RuntimeError):
    """Sensor evidence is inconsistent; it must not update object identities or drive motion."""


def optical_pose(position, quaternion_xyzw):
    q = np.asarray(quaternion_xyzw, float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ObservationError('invalid optical camera quaternion')
    x, y, z, w = q / np.linalg.norm(q)
    pose = np.eye(4)
    pose[:3, :3] = [[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                    [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                    [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]]
    pose[:3, 3] = position
    return pose


@dataclass
class Frame:
    camera: str
    rgb: np.ndarray
    depth: np.ndarray
    K: tuple
    stamp: float
    q: np.ndarray
    finger: float
    T_world_camera: np.ndarray | None = None
    depth_stamp: float | None = None
    pose_stamp: float | None = None
    joint_stamp: float | None = None
    optical_frame: str = ""
    epoch: int = 0


class CaptureBuffer:
    PAIR_SKEW_S = 0.003
    METADATA_SKEW_S = 0.050

    def __init__(self):
        self.images = defaultdict(lambda: deque(maxlen=16))
        self.poses = defaultdict(lambda: deque(maxlen=60))
        self.joints = deque(maxlen=120)
        self.intrinsics = defaultdict(lambda: deque(maxlen=16))

    def select(self, camera, after):
        """Newest complete exposure, with matching optical frames/calibration and nearby still-state metadata."""
        for rgb, ts, optical in reversed(self.images[camera, "rgb"]):
            if ts <= after:
                continue
            ds = [(abs(t - ts), a, t, f) for a, t, f in self.images[camera, "depth"]]
            if not ds or not self.joints or not self.poses[camera]:
                continue
            skew, depth, td, df = min(ds, key=lambda v: v[0])
            if skew > self.PAIR_SKEW_S:
                continue
            if optical != df:
                raise ObservationError("RGB and depth optical frame IDs differ")
            colour = min(self.intrinsics[camera, 'rgb'], key=lambda a: abs(a[3] - ts), default=None)
            depth_info = min(self.intrinsics[camera, 'depth'], key=lambda a: abs(a[3] - td), default=None)
            if colour is None or depth_info is None:
                continue
            K, size, frame, tc = colour
            dk, dsize, dframe, tdi = depth_info
            if max(abs(tc - ts), abs(tdi - td)) > self.PAIR_SKEW_S:
                continue
            if (size != dsize or size != rgb.shape[:2] or size != depth.shape or frame != optical
                    or dframe != optical or not np.allclose(K, dk, rtol=0, atol=1e-4)):
                raise ObservationError("RGB/depth calibration, dimensions or optical frames disagree")
            tj, q, finger = min(self.joints, key=lambda a: abs(a[0] - ts))
            tp, pose = min(self.poses[camera], key=lambda a: abs(a[0] - ts))
            if max(abs(tj - ts), abs(tp - ts)) > self.METADATA_SKEW_S:
                continue
            near = [a for a in self.joints if abs(a[0] - ts) <= self.METADATA_SKEW_S]
            if any(np.max(np.abs(a[1] - q)) > np.radians(.1) or abs(a[2] - finger) > np.radians(.5) for a in near):
                continue
            return Frame(camera, rgb.copy(), depth.copy(), tuple(K), ts, q.copy(), finger,
                         pose.copy(), td, tp, tj, optical)
        return None


def validate_frame(frame, pose):
    if (frame.rgb.ndim != 3 or frame.rgb.shape[2] != 3 or frame.rgb.shape[:2] != frame.depth.shape
            or len(frame.K) != 4 or not np.isfinite(frame.K).all() or min(frame.K[:2]) <= 0):
        raise ObservationError("invalid RGB-D image dimensions or intrinsics")
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)
            or not np.allclose(pose[3], [0, 0, 0, 1])):
        raise ObservationError("invalid optical camera pose")


def surface_points(frame, pose, stride=6):
    """Horizontal RGB-D surface samples, for table calibration and local support inference."""
    depth = frame.depth[::stride, ::stride].astype(float)
    if min(depth.shape) < 3:
        return np.empty((0, 3))
    v, u = np.mgrid[0:frame.depth.shape[0]:stride, 0:frame.depth.shape[1]:stride]
    fx, fy, cx, cy = frame.K
    valid = np.isfinite(depth) & (depth > .05) & (depth < 6)
    # Unobserved sky/background has +inf depth in the simulator. Never project it or use a gradient
    # across that gap: inf * 0 and inf - inf otherwise generate NaNs and spurious surface normals.
    safe_depth = np.where(valid, depth, 0.)
    p = np.stack([(u - cx) / fx * safe_depth, (v - cy) / fy * safe_depth, safe_depth], axis=-1) @ pose[:3, :3].T + pose[:3, 3]
    normal_valid = valid.copy()
    normal_valid[1:] &= valid[:-1]
    normal_valid[:-1] &= valid[1:]
    normal_valid[:, 1:] &= valid[:, :-1]
    normal_valid[:, :-1] &= valid[:, 1:]
    normal = np.cross(np.gradient(p, axis=1), np.gradient(p, axis=0))
    nz = np.abs(normal[..., 2]) / np.maximum(np.linalg.norm(normal, axis=-1), 1e-12)
    return p[normal_valid & (nz > .995)]


def check_table_plane(points):
    """The dominant broad horizontal workspace surface must agree with the calibrated table z=0."""
    p = points[(np.linalg.norm(points[:, :2], axis=1) < 1.1) & (np.abs(points[:, 2]) < .30)]
    if len(p) < 100:
        raise ObservationError("not enough visible horizontal table surface to validate camera calibration")
    bins = np.rint(p[:, 2] / .005).astype(int)
    levels, counts = np.unique(bins, return_counts=True)
    for i in np.argsort(-counts):
        patch = p[bins == levels[i]]
        if len(patch) >= 100 and min(np.ptp(patch[:, :2], axis=0)) > .35:
            z = float(np.median(patch[:, 2]))
            if abs(z) > .012:
                raise ObservationError(f"camera/depth calibration mismatch: broad table surface is {z * 1000:.1f} mm from z=0")
            return z
    raise ObservationError("no broad table plane could be confirmed; refresh the camera calibration/view")
