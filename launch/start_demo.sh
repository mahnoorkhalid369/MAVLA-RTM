#!/usr/bin/env bash
# Text-driven demo of the perception-redesign slice (agent/, ARCHITECTURE.md rev 3). See DEMO.md.
#   1. the agent loads Gemma, then the YOLO-seg detector (GPU order matters on 12 GB)
#   2. the agent starts Isaac Sim (window unless --headless)
#   3. two camera windows (wrist + table) open once the cameras stream
#   4. type instructions at the "instruction>" prompt; an empty line quits and stops everything
#
#   ./launch/start_demo.sh               # Isaac Sim window + camera windows
#   ./launch/start_demo.sh --headless    # no Isaac Sim window (camera windows still open)
# Outputs: out/demo/<date>/live_<time>/<nn>/ (events.jsonl, images/, result.json)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
GUI="--gui"
[ "$1" = "--headless" ] && GUI=""
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p out/logs

cleanup() {
    trap - EXIT INT TERM
    kill 0 2>/dev/null  # everything started from this script (viewers; the agent stops the sim itself)
}
trap cleanup EXIT INT TERM

(
    until ros2 topic list 2>/dev/null | grep -q '^/table_camera/color/image_raw$'; do sleep 3; done
    /usr/bin/python3 ros_nodes/camera_viewer.py > out/logs/demo_wrist_viewer.log 2>&1 &
    /usr/bin/python3 ros_nodes/camera_viewer.py --title "Table camera - RGB and depth" \
        --color-topic /table_camera/color/image_raw --depth-topic /table_camera/depth/image_rect_raw \
        --depth-min 1.2 --depth-max 3.2 --ros-args -r __node:=table_viewer > out/logs/demo_table_viewer.log 2>&1 &
    wait
) &

"$ROOT/.venv-seg/bin/python" -u -m agent.run --start-sim $GUI
