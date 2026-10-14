"""Camera transforms and RGB-D grounding (perception/grounding.py) on SYNTHETIC depth images.

A rotated block is ray-cast from the real table-camera pose with D435 intrinsics, so the whole chain
pixel -> camera ray -> world point -> object pose is checked against a known answer.
Frames: world z up; ROS optical frame z forward, x right, y down. Units: m, deg in yaw.
"""

import numpy as np

from control.robot import TABLE_CAMERA_POSITION, TABLE_CAMERA_TARGET, table_camera_pose
from perception.grounding import GroundingError, colour_name, footprint_yaw, grasp_pose, locate

W, H, HFOV = 1280, 720, 69.4
F = W / 2 / np.tan(np.radians(HFOV / 2))
K = (F, F, W / 2, H / 2)


def render_box(T_world_cam, centre_xy, size, yaw_deg, colour=(200, 30, 30)):
    """Depth (m along the optical axis) and RGB of a box standing on the floor (z = 0)."""
    v, u = np.mgrid[0:H, 0:W] + 0.5
    rays_cam = np.stack([(u - K[2]) / K[0], (v - K[3]) / K[1], np.ones_like(u)], axis=-1)
    R, o = T_world_cam[:3, :3], T_world_cam[:3, 3]
    d = rays_cam @ R.T  # world directions; t below is in "optical-axis depth" units since rays_cam z = 1
    t_floor = np.where(d[..., 2] < 0, -o[2] / np.minimum(d[..., 2], -1e-9), np.inf)
    # box: slab test in the box frame
    a = np.radians(yaw_deg)
    Rb = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    centre = np.array([centre_xy[0], centre_xy[1], size[2] / 2])
    ob, db = Rb.T @ (o - centre), d @ Rb  # origin / directions in the box frame
    half = np.array(size) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-half - ob) / db, (half - ob) / db
    t_near = np.nanmax(np.minimum(t1, t2), axis=-1)
    t_far = np.nanmin(np.maximum(t1, t2), axis=-1)
    hit_box = (t_near <= t_far) & (t_far > 0)
    depth = np.where(hit_box & (t_near < t_floor), t_near, t_floor).astype(np.float32)
    rgb = np.full((H, W, 3), (120, 160, 200), np.uint8)
    rgb[hit_box & (t_near < t_floor)] = colour
    return depth, rgb


def box_around(T, xyz, half_px=40):
    c = T[:3, :3].T @ (np.asarray(xyz) - T[:3, 3])
    u, v = K[0] * c[0] / c[2] + K[2], K[1] * c[1] / c[2] + K[3]
    return (int((v - half_px) / H * 1000), int((u - half_px) / W * 1000),
            int((v + half_px) / H * 1000), int((u + half_px) / W * 1000))


def test_table_camera_pose_is_a_proper_rotation_looking_at_its_target():
    T = table_camera_pose()
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(R), 1.0)
    look = np.subtract(TABLE_CAMERA_TARGET, TABLE_CAMERA_POSITION)
    assert np.allclose(R[:, 2], look / np.linalg.norm(look))  # optical z points at the target
    assert abs(R[2, 0]) < 1e-12  # image x axis is horizontal (no roll)
    assert R[2, 1] < 0  # image y (down in the picture) points down in the world


def test_projection_and_back_projection_round_trip():
    T = table_camera_pose()
    p = np.array([0.3, 0.4, 0.05])
    c = T[:3, :3].T @ (p - T[:3, 3])
    u, v = K[0] * c[0] / c[2] + K[2], K[1] * c[1] / c[2] + K[3]
    back = T[:3, :3] @ np.array([(u - K[2]) / K[0] * c[2], (v - K[3]) / K[1] * c[2], c[2]]) + T[:3, 3]
    assert np.linalg.norm(back - p) < 1e-9


def test_locate_rotated_block_from_table_camera():
    T = table_camera_pose()
    for xy, yaw in (((0.0, 0.58), 0.0), ((0.5, -0.25), 30.0), ((-0.3, 0.6), -20.0)):
        depth, rgb = render_box(T, xy, (0.06, 0.06, 0.06), yaw)
        est = locate(box_around(T, (*xy, 0.06)), depth, K, T, rgb)
        assert est.is_object
        assert np.linalg.norm(est.position[:2] - xy) < 0.006  # m
        assert abs(est.top_z - 0.06) < 0.004
        err = (est.yaw_deg - yaw + 45) % 90 - 45  # a square repeats every 90 deg
        assert abs(err) < 3.0, (xy, yaw, est.yaw_deg)
        assert est.colour == "red"


def test_locate_empty_floor_is_not_an_object():
    T = table_camera_pose()
    depth, rgb = render_box(T, (5.0, 5.0), (0.05, 0.05, 0.05), 0.0)  # box far outside the view
    est = locate(box_around(T, (0.2, 0.5, 0.0)), depth, K, T, rgb)
    assert not est.is_object and abs(est.position[2]) < 0.005


def test_locate_without_depth_raises():
    T = table_camera_pose()
    try:
        locate((400, 400, 500, 500), np.full((H, W), np.nan, np.float32), K, T)
    except GroundingError:
        return
    raise AssertionError("expected GroundingError")


def test_grasp_closes_across_the_narrow_side():
    rng = np.random.default_rng(0)
    for deg in (0, 30, -60, 80):
        a = np.radians(deg)
        pts = rng.uniform([-0.05, -0.02], [0.05, 0.02], (3000, 2)) @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        yaw, fp = footprint_yaw(pts)
        g, width, sym = grasp_pose(pts, yaw, fp)
        finger_dir = np.array([-np.sin(np.radians(g)), np.cos(np.radians(g))])
        across = np.percentile(pts @ finger_dir, 98) - np.percentile(pts @ finger_dir, 2)
        assert across < 0.045 and abs(width - across) < 1e-6 and sym == 180.0


def test_colour_names():
    assert colour_name(np.array([229, 25, 26])) == "red"
    assert colour_name(np.array([245, 184, 229])) == "pink"  # table-camera calibrated reference
