#!/usr/bin/env python3
"""ROS 2 node: type instructions in the terminal.

Each line you type is published on /<topic>; replies on /<topic>/feedback are printed as they arrive.

    source /opt/ros/jazzy/setup.bash
    python3 ros_nodes/text_command_node.py               # direct commands -> command_interpreter
    python3 ros_nodes/text_command_node.py --topic task  # tasks for the VLM agent

Type 'help' for the direct commands, 'quit' to exit.
"""

from __future__ import annotations

import argparse
import threading

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


class TextCommandNode(Node):
    def __init__(self, topic: str) -> None:
        super().__init__(f"text_{topic}")
        self.topic = topic
        self.pub = self.create_publisher(String, topic, 10)
        self.create_subscription(String, f"{topic}/feedback", self.on_feedback, 10)
        self.replied = threading.Event()

    def send(self, text: str) -> None:
        self.replied.clear()
        self.pub.publish(String(data=text))

    def on_feedback(self, msg: String) -> None:
        print("\n".join("  " + line for line in msg.data.splitlines()), flush=True)
        self.replied.set()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="instruction", help="'instruction' (direct commands) or 'task' (VLM agent)")
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = TextCommandNode(args.topic)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    target = "the VLM agent (replies arrive as it works)" if args.topic == "task" else "the arm ('help' for commands)"
    print(f"Type instructions for {target}; 'quit' to exit.")
    prompt = "task> " if args.topic == "task" else "vt6l> "
    try:
        while rclpy.ok():
            try:
                text = input(prompt).strip()
            except EOFError:
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            if node.count_subscribers(args.topic) == 0:
                print(f"  nothing is listening on /{args.topic} (is its node running?)")
                continue
            node.send(text)
            if not node.replied.wait(timeout=2.0):
                print("  (no reply yet)")
    except (KeyboardInterrupt, ExternalShutdownException):
        print()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
