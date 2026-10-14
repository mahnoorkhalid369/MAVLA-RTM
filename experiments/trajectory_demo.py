"""Epson VT6L (proxy) joint-trajectory simulation in Isaac Sim 6.x, with a wrist camera.

The arm cycles through a set of joint-space waypoints using smooth (cubic)
interpolation and PD position drives (gripper held open). Joint tracking error
is printed each second.

A RealSense D435-style RGB-D camera is mounted on the J5 wrist housing (see
assets/vt6l/vt6l_proxy.urdf, camera_link). Its live view opens in a second
viewport window; --save-images writes RGB + depth frames to disk.
For grasping, see experiments/pick_place.py.

Usage:
    python -m experiments.trajectory_demo                         # GUI window + camera view
    python -m experiments.trajectory_demo --headless               # no window
    python -m experiments.trajectory_demo --test                   # run ~one trajectory cycle then exit
    python -m experiments.trajectory_demo --save-images out/cam    # save 1 RGB-D frame per second
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--test", action="store_true", help="run one cycle and exit")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
parser.add_argument("--save-images", type=Path, default=None, help="directory for RGB/depth frames")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.sensors.experimental.rtx")
simulation_app.update()

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.rendering_manager import ViewportManager  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera  # noqa: E402
from sim.scene import (  # noqa: E402
    ARM_JOINTS,
    CAM_RESOLUTION,
    FINGER_JOINTS,
    GRIPPER_OPEN,
    ROBOT_PATH,
    add_wrist_camera,
    save_frame,
    setup_scene,
)

PHYSICS_DT = 1.0 / 120.0
SEGMENT_TIME = 2.5  # seconds per waypoint-to-waypoint move

# Joint-space waypoints [J1..J6] in degrees (all within VT6L ranges).
# J5 = -90 points the tool (and wrist camera) straight down at the blocks.
WAYPOINTS_DEG = np.array(
    [
        [0, 0, 0, 0, -90, 0],
        [30, -20, 20, 0, -90, 0],
        [-30, -20, 20, 0, -90, 90],
        [45, -30, 60, 0, -30, 0],
        [-60, -45, 90, 90, 45, 90],
        [0, -90, 150, 0, 30, -180],
        [90, 20, 30, -90, -60, 180],
        [0, 0, 0, 0, -90, 0],
    ],
    dtype=np.float64,
)
WAYPOINTS = np.deg2rad(WAYPOINTS_DEG)


def trajectory(t: float) -> np.ndarray:
    """Cubic (smoothstep) interpolation between waypoints, looping."""
    n_seg = len(WAYPOINTS) - 1
    t = t % (n_seg * SEGMENT_TIME)
    seg = min(int(t // SEGMENT_TIME), n_seg - 1)
    s = (t - seg * SEGMENT_TIME) / SEGMENT_TIME
    s = 3 * s**2 - 2 * s**3
    return WAYPOINTS[seg] + s * (WAYPOINTS[seg + 1] - WAYPOINTS[seg])


def to_wp(values: np.ndarray) -> wp.array:
    return wp.array(np.asarray(values, dtype=np.float32).reshape(1, -1), dtype=wp.float32)


def main() -> None:
    SimulationManager.setup_simulation(dt=PHYSICS_DT, device=args.device)
    simulation_app.run_coroutine(setup_scene())

    cam_path = add_wrist_camera()
    camera = RtxCamera(cam_path)
    sensor = CameraSensor(camera, resolution=CAM_RESOLUTION, annotators=["rgb", "distance_to_image_plane"])
    print(f"[vt6l_sim] wrist camera: {cam_path}")
    if not args.headless:
        ViewportManager.create_viewport_window(
            camera=cam_path, title="D435 wrist camera", resolution=(CAM_RESOLUTION[1], CAM_RESOLUTION[0])
        )
    if args.save_images:
        args.save_images.mkdir(parents=True, exist_ok=True)

    app_utils.play()
    simulation_app.update()

    robot = Articulation(ROBOT_PATH)
    print(f"[vt6l_sim] DOFs: {robot.dof_names}")
    arm_idx = robot.get_dof_indices(ARM_JOINTS)
    finger_idx = robot.get_dof_indices(FINGER_JOINTS)
    robot.set_dof_positions(to_wp(WAYPOINTS[0]), dof_indices=arm_idx)
    robot.set_dof_position_targets(to_wp(WAYPOINTS[0]), dof_indices=arm_idx)
    robot.set_dof_position_targets(to_wp([GRIPPER_OPEN] * len(FINGER_JOINTS)), dof_indices=finger_idx)

    sim_time = 0.0
    step = 0
    frame = 0
    cycle_time = (len(WAYPOINTS) - 1) * SEGMENT_TIME
    max_err = np.zeros(6)
    while simulation_app.is_running():
        simulation_app.update()
        if not (app_utils.is_playing() and SimulationManager.is_simulating()):
            continue
        q_des = trajectory(sim_time)
        robot.set_dof_position_targets(to_wp(q_des), dof_indices=arm_idx)
        q = robot.get_dof_positions(dof_indices=arm_idx).numpy()[0]
        err = np.abs(np.rad2deg(q - q_des))
        max_err = np.maximum(max_err, err)
        if step % int(1.0 / PHYSICS_DT) == 0:
            cam_msg = "camera: warming up"
            rgb, _ = sensor.get_data("rgb")
            depth, _ = sensor.get_data("distance_to_image_plane")
            if rgb is not None and depth is not None:
                rgb, depth = rgb.numpy(), depth.numpy().squeeze()
                valid = depth[np.isfinite(depth)]
                cam_msg = (
                    f"camera: rgb {rgb.shape[1]}x{rgb.shape[0]}, "
                    f"depth {valid.min():.2f}-{valid.max():.2f} m" if valid.size else "camera: no depth"
                )
                if args.save_images:
                    save_frame(args.save_images, frame, rgb, depth)
                frame += 1
            print(
                f"[vt6l_sim] t={sim_time:5.2f}s  q(deg)=[{", ".join(f"{v:7.1f}" for v in np.rad2deg(q))}]  "
                f"|err|max(deg)={err.max():.3f}  {cam_msg}"
            )
        sim_time += PHYSICS_DT
        step += 1
        if args.test and sim_time > cycle_time:
            print(f"[vt6l_sim] cycle done. max tracking error per joint (deg): {np.round(max_err, 3).tolist()}")
            break


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
