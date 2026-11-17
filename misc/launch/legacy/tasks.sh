#!/usr/bin/env bash
# Type tasks for the VLM agent (free-form pick and place, direct commands, overview, forget, ...).
# Needs ./launch/legacy/start_live.sh and ./launch/legacy/start_vlm_agent.sh running.
# Uses the system Python (ROS 2), so it works even with .venv active.
#
#   ./launch/legacy/tasks.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/jazzy/setup.bash
exec /usr/bin/python3 "$ROOT/ros_nodes/text_command_node.py" --topic task "$@"
