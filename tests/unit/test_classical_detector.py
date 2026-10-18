"""Classical RGB-D detector (perception/classical_detector.py) on a SYNTHETIC table-camera frame:
catalogue objects ray-cast at their real poses from the real camera pose + D435 intrinsics, including
a stack (lime on blue), a 45 deg rotated cube and a cylinder. Expected values come from the scene
definition, never from the detector."""

import numpy as np
import pytest

from control.robot import table_camera_pose
from perception.classical_detector import ClassicalDetector
from sim.scene_objects import OBJECTS

W, H, HFOV = 1280, 720, 69.4
F = W / 2 / np.tan(np.radians(HFOV / 2))
K = (F, F, W / 2, H / 2)
NAMES = ["red_cube", "green_cube", "blue_cube", "lime_cube_stacked", "brown_cube_rotated",
         "yellow_cube", "purple_cylinder", "tan_box_wide"]


def render(names):
    """Depth (m along the optical axis) and RGB of the named catalogue objects on a grey floor."""
    T = table_camera_pose()
    v, u = np.mgrid[0:H, 0:W] + 0.5
    rays = np.stack([(u - K[2]) / K[0], (v - K[3]) / K[1], np.ones_like(u)], axis=-1)
    d = rays @ T[:3, :3].T  # world directions, scaled so the optical-axis component is 1
    o = T[:3, 3]
    depth = np.where(d[..., 2] < 0, -o[2] / np.minimum(d[..., 2], -1e-9), np.inf)
    rgb = np.full((H, W, 3), (150, 150, 155), np.uint8)
    for name in names:
        ob = OBJECTS[name]
        base = ob.base_z
        colour = tuple(int(c * 255) for c in ob.color)
        if ob.shape == "cylinder":
            r = ob.size[0] / 2
            px, py = o[0] - ob.xy[0], o[1] - ob.xy[1]
            a = d[..., 0] ** 2 + d[..., 1] ** 2
            b = 2 * (px * d[..., 0] + py * d[..., 1])
            c = px ** 2 + py ** 2 - r ** 2
            disc = b ** 2 - 4 * a * c
            with np.errstate(invalid="ignore", divide="ignore"):
                t_side = (-b - np.sqrt(disc)) / (2 * a)
                z_side = o[2] + t_side * d[..., 2]
                side = (disc >= 0) & (z_side >= base) & (z_side <= base + ob.height)
                t_top = (base + ob.height - o[2]) / d[..., 2]
                hx, hy = o[0] + t_top * d[..., 0] - ob.xy[0], o[1] + t_top * d[..., 1] - ob.xy[1]
                top = (hx ** 2 + hy ** 2 <= r ** 2) & (t_top > 0)
            t = np.where(top, t_top, np.where(side, t_side, np.inf))
        else:
            a = np.radians(ob.yaw_deg)
            Rb = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            centre = np.array([ob.xy[0], ob.xy[1], base + ob.height / 2])
            ol, dl = Rb.T @ (o - centre), d @ Rb
            half = np.array(ob.size) / 2
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (-half - ol) / dl, (half - ol) / dl
            tn, tf = np.nanmax(np.minimum(t1, t2), axis=-1), np.nanmin(np.maximum(t1, t2), axis=-1)
            t = np.where((tn <= tf) & (tf > 0), tn, np.inf)
        nearer = t < depth
        depth = np.where(nearer, t, depth)
        rgb[nearer] = colour
    return depth.astype(np.float32), rgb


@pytest.fixture(scope="module")
def result():
    depth, rgb = render(NAMES)
    det = ClassicalDetector()
    found = det.detect(depth, rgb, K, table_camera_pose())
    return found, det.identify(found)


def test_every_object_is_found_and_named(result):
    _, poses = result
    assert set(NAMES) <= set(poses), f"missing {set(NAMES) - set(poses)}"


@pytest.mark.parametrize("name", NAMES)
def test_pose_accuracy(result, name):
    _, poses = result
    centre, yaw = poses[name]
    ob = OBJECTS[name]
    truth = np.array([ob.xy[0], ob.xy[1], ob.base_z + ob.height / 2])
    assert np.linalg.norm(centre[:2] - truth[:2]) < 0.006, f"{name}: {np.round(centre * 1000, 1)} vs {np.round(truth * 1000, 1)}"
    assert abs(centre[2] - truth[2]) < 0.006
    if ob.shape == "box":
        assert abs((yaw - ob.yaw_deg + 45) % 90 - 45) < 3.0, f"{name}: yaw {yaw}"


def test_stack_gives_two_objects(result):
    _, poses = result
    assert poses["lime_cube_stacked"][0][2] > poses["blue_cube"][0][2] + 0.03
