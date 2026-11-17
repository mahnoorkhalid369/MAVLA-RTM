"""Epson VT6L (proxy) pick-and-place with a Robotiq 2F-85 gripper and wrist camera.

Picks the red block, lifts it, carries it to PLACE_XY and sets it down.
Straight-line (Cartesian) moves are solved with damped least-squares IK every
physics step (control/kinematics.py); the gripper points straight down throughout.
At the end the block's final position is checked against the target.

Usage:
    python -m experiments.pick_place                         # GUI + wrist camera view
    python -m experiments.pick_place --headless --test        # run once, report, exit
    python -m experiments.pick_place --save-images out/pick   # save RGB-D frames (2/s)
    python -m experiments.pick_place --block-mass 3.0         # payload test (VT6L max 6 kg incl. gripper)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--test", action="store_true", help="run the sequence once, report, and exit")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
parser.add_argument("--save-images", type=Path, default=None, help="directory for RGB/depth frames")
parser.add_argument("--block-mass", type=float, default=None, help="mass (kg) of the picked block")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.sensors.experimental.rtx")
simulation_app.update()

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.core.rendering_manager import ViewportManager  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera  # noqa: E402
from control.kinematics import R_DOWN, Kinematics  # noqa: E402
from sim.scene import (  # noqa: E402
    ARM_JOINTS,
    BLOCKS,
    CAM_RESOLUTION,
    FINGER_JOINTS,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    ROBOT_PATH,
    add_wrist_camera,
    save_frame,
    setup_scene,
)

PHYSICS_DT = 1.0 / 120.0
PICK_BLOCK = "red_cube"
PLACE_XY = np.array([0.30, 0.40])
HOVER_Z = 0.20  # TCP height for travel moves
GRASP_Z = BLOCKS[PICK_BLOCK].height / 2  # TCP at the block centre
PLACE_Z = BLOCKS[PICK_BLOCK].height / 2 + 0.005  # release 5 mm above the table
Q_HOME = np.deg2rad([0, 0, 0, 0, -90, 0])


def build_plan(pick_xy: np.ndarray) -> list[tuple]:
    """(label, kind, target, duration_s, gripper) steps.

    kind "joint": smooth joint-space move to target (joint angles, or a TCP xyz solved by IK)
    kind "line":  straight-line TCP move to target xyz, IK every step
    kind "hold":  stay still (used while the gripper opens/closes)
    """
    pick_hover = np.array([*pick_xy, HOVER_Z])
    place_hover = np.array([*PLACE_XY, HOVER_Z])
    return [
        ("move above block", "joint", pick_hover, 3.0, GRIPPER_OPEN),
        ("descend", "line", np.array([*pick_xy, GRASP_Z]), 2.0, GRIPPER_OPEN),
        ("close gripper", "hold", None, 1.0, GRIPPER_CLOSED),
        ("lift", "line", pick_hover, 2.0, GRIPPER_CLOSED),
        ("carry", "line", place_hover, 3.0, GRIPPER_CLOSED),
        ("lower", "line", np.array([*PLACE_XY, PLACE_Z]), 2.0, GRIPPER_CLOSED),
        ("open gripper", "hold", None, 1.0, GRIPPER_OPEN),
        ("retreat", "line", place_hover, 1.5, GRIPPER_OPEN),
        ("return home", "joint", Q_HOME, 3.0, GRIPPER_OPEN),
    ]


def smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 3 * s**2 - 2 * s**3


def to_wp(values) -> wp.array:
    return wp.array(np.asarray(values, dtype=np.float32).reshape(1, -1), dtype=wp.float32)


def main() -> None:
    kin = Kinematics()
    SimulationManager.setup_simulation(dt=PHYSICS_DT, device=args.device)
    block_mass = BLOCKS[PICK_BLOCK].mass if args.block_mass is None else args.block_mass
    simulation_app.run_coroutine(setup_scene({PICK_BLOCK: block_mass}))

    cam_path = add_wrist_camera()
    sensor = CameraSensor(RtxCamera(cam_path), resolution=CAM_RESOLUTION, annotators=["rgb", "distance_to_image_plane"])
    if not args.headless:
        ViewportManager.create_viewport_window(
            camera=cam_path, title="D435 wrist camera", resolution=(CAM_RESOLUTION[1], CAM_RESOLUTION[0])
        )
    if args.save_images:
        args.save_images.mkdir(parents=True, exist_ok=True)

    app_utils.play()
    simulation_app.update()

    robot = Articulation(ROBOT_PATH)
    block = RigidPrim(f"/World/Objects/{PICK_BLOCK}")
    arm_idx = robot.get_dof_indices(ARM_JOINTS)
    finger_idx = robot.get_dof_indices(FINGER_JOINTS)
    robot.set_dof_positions(to_wp(Q_HOME), dof_indices=arm_idx)
    robot.set_dof_position_targets(to_wp(Q_HOME), dof_indices=arm_idx)
    robot.set_dof_positions(to_wp([GRIPPER_OPEN] * len(FINGER_JOINTS)), dof_indices=finger_idx)
    robot.set_dof_position_targets(to_wp([GRIPPER_OPEN] * len(FINGER_JOINTS)), dof_indices=finger_idx)

    pick_xy = np.array(BLOCKS[PICK_BLOCK].xy)
    plan = build_plan(pick_xy)
    step_i, step_t = 0, 0.0
    q_cmd = Q_HOME.copy()
    q_start, p_start = q_cmd.copy(), kin.fk(q_cmd)[:3, 3]
    q_goal = None
    sim_time, frame, n = 0.0, 0, 0
    lift_z = []
    print(f"[pick_place] picking '{PICK_BLOCK}' ({block_mass} kg) at {pick_xy.tolist()} -> placing at {PLACE_XY.tolist()}")

    while simulation_app.is_running():
        simulation_app.update()
        if not (app_utils.is_playing() and SimulationManager.is_simulating()):
            continue

        if step_i < len(plan):
            label, kind, target, duration, grip = plan[step_i]
            if step_t == 0.0:
                print(f"[pick_place] t={sim_time:5.2f}s  step {step_i + 1}/{len(plan)}: {label}")
                if kind == "joint":
                    if target.shape == (3,):  # TCP position -> joint angles
                        q_goal, ok = kin.ik(target, R_DOWN, q_cmd)
                        if not ok:
                            raise RuntimeError(f"IK failed for '{label}' target {target}")
                    else:
                        q_goal = target
            s = smoothstep(step_t / duration)
            if kind == "joint":
                q_cmd = q_start + s * (q_goal - q_start)
            elif kind == "line":
                p = p_start + s * (target - p_start)
                q_cmd, ok = kin.ik(p, R_DOWN, q_cmd)
                if not ok:
                    raise RuntimeError(f"IK failed during '{label}' at {p}")
            robot.set_dof_position_targets(to_wp(q_cmd), dof_indices=arm_idx)
            robot.set_dof_position_targets(to_wp([grip] * len(FINGER_JOINTS)), dof_indices=finger_idx)
            if label in ("lift", "carry"):
                lift_z.append(block.get_world_poses()[0].numpy()[0][2])

            step_t += PHYSICS_DT
            if step_t >= duration:
                step_i, step_t = step_i + 1, 0.0
                q_start, p_start = q_cmd.copy(), kin.fk(q_cmd)[:3, 3]
                if step_i == len(plan):
                    # Report while physics is still running (it is torn down when the window closes)
                    report(robot, block, finger_idx, lift_z)
                    if args.test:
                        break
                    print("[pick_place] done - close the window to exit")

        if args.save_images and n % int(0.5 / PHYSICS_DT) == 0:
            rgb, _ = sensor.get_data("rgb")
            depth, _ = sensor.get_data("distance_to_image_plane")
            if rgb is not None and depth is not None:
                save_frame(args.save_images, frame, rgb.numpy(), depth.numpy().squeeze())
                frame += 1
        sim_time += PHYSICS_DT
        n += 1


def report(robot: Articulation, block: RigidPrim, finger_idx: wp.array, lift_z: list[float]) -> None:
    final = block.get_world_poses()[0].numpy()[0]
    fingers = robot.get_dof_positions(dof_indices=finger_idx).numpy()[0]
    err_xy = np.linalg.norm(final[:2] - PLACE_XY)
    print(f"[pick_place] block max height while carried: {max(lift_z, default=0.0):.3f} m")
    print(f"[pick_place] block final position: {np.round(final, 4).tolist()}  (target {PLACE_XY.tolist()})")
    print(f"[pick_place] placement error: {err_xy * 1000:.1f} mm   fingers: {np.round(fingers, 4).tolist()}")
    lifted = max(lift_z, default=0.0) > HOVER_Z - 0.1
    print(f"[pick_place] RESULT: {'SUCCESS' if lifted and err_xy < 0.02 else 'FAILED'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
