#!/usr/bin/env python3
"""ROS 2 node: turns text instructions into coordinated arm motions for the simulated VT6L.

    /instruction           std_msgs/String                  in   text, typed now (text_command_node.py),
                                                                 spoken later (a speech-to-text node)
    /joint_states          sensor_msgs/JointState           in   current joint angles from the simulator
    /joint_trajectory      trajectory_msgs/JointTrajectory  out  planned arm motion (all joints coordinated)
    /joint_command         sensor_msgs/JointState           out  gripper targets
    /instruction/feedback  std_msgs/String                  out  what was understood / planned

Pipeline: control/instructions.py (text -> goal) -> control/motion.py (goal -> trajectory, using
FK/IK from control/kinematics.py) -> this node publishes it. None of those modules use ROS.

    source /opt/ros/jazzy/setup.bash
    python3 ros_nodes/command_interpreter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root, for control/

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from builtin_interfaces.msg import Duration  # noqa: E402
from rclpy.executors import ExternalShutdownException  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # noqa: E402

from control.instructions import Command, InstructionParser, plan_command  # noqa: E402
from control.motion import PlanningError, Trajectory  # noqa: E402
from control.robot import ARM_JOINTS, GRIPPER_JOINT, GRIPPER_SPEED  # noqa: E402


class CommandInterpreter(Node):
    def __init__(self) -> None:
        super().__init__("command_interpreter")
        self.parser = InstructionParser()
        self.kin = self.parser.kin
        self.current: dict[str, float] = {}
        self.trajectory_pub = self.create_publisher(JointTrajectory, "joint_trajectory", 10)
        self.command_pub = self.create_publisher(JointState, "joint_command", 10)
        self.feedback_pub = self.create_publisher(String, "instruction/feedback", 10)
        self.create_subscription(String, "instruction", self.on_instruction, 10)
        self.create_subscription(JointState, "joint_states", self.on_joint_states, 10)
        self.get_logger().info("listening on /instruction")

    def on_joint_states(self, msg: JointState) -> None:
        self.current.update(zip(msg.name, msg.position))

    # ----------------------------------------------------------------- planning
    def plan(self, cmd: Command) -> Trajectory | None:
        return plan_command(self.kin, cmd, self.current)

    def publish_trajectory(self, traj: Trajectory) -> None:
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = list(ARM_JOINTS)
        for t, q in zip(traj.times, traj.positions):
            sec = int(t)
            msg.points.append(
                JointTrajectoryPoint(
                    positions=[float(v) for v in q],
                    time_from_start=Duration(sec=sec, nanosec=int(round((t - sec) * 1e9))),
                )
            )
        self.trajectory_pub.publish(msg)

    # ------------------------------------------------------------- instructions
    def on_instruction(self, msg: String) -> None:
        cmd = self.parser.parse(msg.data, self.current)
        reply = cmd.reply
        try:
            traj = self.plan(cmd)
        except PlanningError as err:
            traj, reply = None, f"{reply.splitlines()[0] if reply else msg.data}: not moving, {err}"
        else:
            if traj is not None:
                self.publish_trajectory(traj)
                reply += f" [{traj.description}]"
        if GRIPPER_JOINT in cmd.targets:
            out = JointState()
            out.header.stamp = self.get_clock().now().to_msg()
            out.name = [GRIPPER_JOINT]
            out.position = [float(cmd.targets[GRIPPER_JOINT])]
            out.velocity = [float(GRIPPER_SPEED)]
            self.command_pub.publish(out)
        if reply:
            self.feedback_pub.publish(String(data=reply))
            self.get_logger().info(f"'{msg.data}' -> {reply.splitlines()[0]}")


def main() -> None:
    rclpy.init()
    node = CommandInterpreter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
