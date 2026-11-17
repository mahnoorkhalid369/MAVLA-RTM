"""Live simulation: the VT6L scene streaming its sensors over ROS 2 and taking joint commands.

No task runs on its own: the arm starts at its home pose with the gripper open and only moves
when a command arrives on /joint_command (e.g. from ros_nodes/command_interpreter.py, which
turns typed instructions into joint targets). Sensors and TF are published by sim/ros_bridge.py;
the command interface is sim/ros_control.py.

Run from the project folder in a shell where ROS 2 Jazzy is sourced
(launch/start_live.sh does this and also starts the ROS 2 nodes):

    source /opt/ros/jazzy/setup.bash
    python -m sim.run_live                 # GUI
    python -m sim.run_live --headless      # no window
    python -m sim.run_live --duration 30   # stop after 30 s of simulation time
"""

from __future__ import annotations

import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
parser.add_argument("--duration", type=float, default=0.0, help="seconds of sim time to run (0 = until closed)")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

from control.robot import ARM_HOME, ARM_JOINTS, ARM_SPEED, GRIPPER_JOINT, GRIPPER_OPEN, GRIPPER_SPEED  # noqa: E402
from sim.ros_bridge import camera_topics, create_ros2_bridge  # noqa: E402
from safety.joint_limits import JointLimits  # noqa: E402
from sim.ros_control import CameraPosePublisher, GroundTruthPublisher, JointCommandInterface  # noqa: E402
from sim.scene import (  # noqa: E402
    GRIPPER_ROOT,
    MIC_PATH,
    ROBOT_PATH,
    OBJECTS,
    add_table_camera,
    add_wrist_camera,
    find_prim_path,
    setup_scene,
)

PHYSICS_DT = 1.0 / 120.0
STATE_PERIOD = 1.0 / 30.0  # s of simulation time between /joint_states messages
TRUTH_PERIOD = 0.1  # s between object ground-truth messages
CONTROLLED = ARM_JOINTS + [GRIPPER_JOINT]


def to_wp(values) -> wp.array:
    return wp.array(np.asarray(values, dtype=np.float32).reshape(1, -1), dtype=wp.float32)


def main() -> None:
    SimulationManager.setup_simulation(dt=PHYSICS_DT, device=args.device)
    simulation_app.run_coroutine(setup_scene())
    cam_path = add_wrist_camera()
    tf_bodies = [
        find_prim_path(ROBOT_PATH, "base_link"),  # arm articulation root: publishes all its links
        f"{GRIPPER_ROOT}/base_link",
    ]
    table_path, table_optical = add_table_camera()
    static_frames = [
        (find_prim_path(ROBOT_PATH, "camera_color_optical_frame"), find_prim_path(ROBOT_PATH, "camera_link")),
        (f"{MIC_PATH}/microphone_link", None),
        (table_optical, None),
    ]
    cameras = [
        (cam_path, "camera", "camera_color_optical_frame"),  # wrist (eye-in-hand)
        (table_path, "table_camera", "table_camera_optical_frame"),  # on a stand (eye-to-hand)
    ]
    create_ros2_bridge(cameras, tf_bodies, static_frames)

    app_utils.play()
    simulation_app.update()

    robot = Articulation(ROBOT_PATH)
    idx = robot.get_dof_indices(CONTROLLED)
    start = np.concatenate([ARM_HOME, [GRIPPER_OPEN]])
    robot.set_dof_positions(to_wp(start), dof_indices=idx)
    robot.set_dof_position_targets(to_wp(start), dof_indices=idx)
    commands = JointCommandInterface(
        CONTROLLED, start, default_speeds=np.array([ARM_SPEED] * len(ARM_JOINTS) + [GRIPPER_SPEED]),
        limits=JointLimits.from_urdf(),  # independent check of every command against the VT6L limits
    )
    camera_poses = CameraPosePublisher(commands.node, [
        ("table_camera", table_optical), ("camera", find_prim_path(ROBOT_PATH, "camera_color_optical_frame"))])

    # GROUND TRUTH object poses, for evaluation and the known-pose baseline only: perception must not use it
    object_names = list(OBJECTS)
    objects = RigidPrim([f"/World/Objects/{n}" for n in object_names])
    truth = GroundTruthPublisher(commands.node, object_names)

    topics = [t for prefix in ("camera", "table_camera") for t in camera_topics(prefix).values()]
    print("[run_live] streaming on ROS 2:", ", ".join(f"/{t}" for t in topics), "+ /tf, /tf_static, /clock")
    print("[run_live] listening on /joint_command, publishing /joint_states; arm at home pose")
    sim_time = start_time = SimulationManager.get_simulation_time()
    next_note, next_state, next_truth, step = sim_time, sim_time, sim_time, 0
    try:
        while simulation_app.is_running():
            simulation_app.update()
            if not (app_utils.is_playing() and SimulationManager.is_simulating()):
                continue
            # One app update can run SEVERAL physics steps (here 2 at 120 Hz): advance the command
            # interface by the simulation time that really passed, not by one physics dt, or every
            # trajectory plays back slower than planned (it did, at half speed, until 2026-10-14).
            now = SimulationManager.get_simulation_time()
            dt = max(now - sim_time, 0.0)
            sim_time = now
            robot.set_dof_position_targets(to_wp(commands.step(dt)), dof_indices=idx)
            if sim_time >= next_state:
                commands.publish_state(
                    robot.get_dof_positions(dof_indices=idx).numpy()[0],
                    robot.get_dof_velocities(dof_indices=idx).numpy()[0],
                    sim_time=sim_time,
                )
                camera_poses.publish(sim_time)
                next_state += STATE_PERIOD
            if sim_time >= next_truth:
                positions, orientations = objects.get_world_poses()
                truth.publish(sim_time, positions.numpy(), orientations.numpy())
                next_truth += TRUTH_PERIOD
            step += 1
            if sim_time >= next_note:
                print(f"[run_live] sim time {sim_time:6.1f} s")
                next_note += 10.0
            if args.duration and sim_time >= start_time + args.duration:
                break
    finally:
        commands.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
