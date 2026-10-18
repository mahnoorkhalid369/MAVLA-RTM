#!/usr/bin/env bash
# The robot session. Edit agent/settings.py (scene, motion, models, windows, tracing, output), then run:
#
#   ./launch/agent.sh
#
# It validates the settings, prints a summary, loads Gemma and YOLO, starts Isaac Sim with the configured scene,
# opens the demo viewer / camera windows and gives you the "instruction>" prompt. Every agent handoff is printed
# when TRACE is on; the full trace of each instruction is in <OUTPUT_DIR>/<date>/live_<time>/<nn>/events.jsonl.
# An empty instruction line quits and stops the simulator.
#
# Optional, for one run only (agent/settings.py stays the place to configure):
#   ./launch/agent.sh --motion blocked            # any agent.run option overrides the setting (see --help)
#   ./launch/agent.sh --headless                  # no Isaac Sim window
#   DEMO_VIEWER=0 CAMERA_WINDOWS=0 ./launch/agent.sh
#   DEMO_VIEWER_ARGS="--snapshot-every 3 --snapshot-dir out/demo/shots" ./launch/agent.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
SETTINGS="$(env -u PYTHONPATH -u AMENT_PREFIX_PATH "$ROOT/.venv-seg/bin/python" -m agent.settings --shell)" || exit 1
eval "$SETTINGS"
if pgrep -f "sim.run_live|sim.run_scenario" > /dev/null; then
    echo "[agent.sh] a simulator is already running (one Isaac Sim at a time on this GPU); stop it first"; exit 1
fi
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID OMNI_KIT_ACCEPT_EULA=YES
mkdir -p "$LOG_DIR" "$LIVE_DIR"

# windows in their own process group (started once the table camera streams), stdin closed, stopped on exit
setsid bash -c '
    until ros2 topic list 2>/dev/null | grep -q "^/table_camera/color/image_raw$"; do sleep 3; done
    if [ "'"$DEMO_VIEWER"'" = 1 ]; then
        /usr/bin/python3 agent/viewer.py --live-dir "'"$LIVE_DIR"'" '"$DEMO_VIEWER_ARGS"' < /dev/null > "'"$LOG_DIR"'/demo_viewer.log" 2>&1 &
    fi
    if [ "'"$CAMERA_WINDOWS"'" = 1 ]; then
        /usr/bin/python3 ros_nodes/camera_viewer.py < /dev/null > "'"$LOG_DIR"'/demo_wrist_viewer.log" 2>&1 &
        /usr/bin/python3 ros_nodes/camera_viewer.py --title "Table camera - RGB and depth" \
            --color-topic /table_camera/color/image_raw --depth-topic /table_camera/depth/image_rect_raw \
            --depth-min 1.2 --depth-max 3.2 --ros-args -r __node:=table_viewer < /dev/null > "'"$LOG_DIR"'/demo_table_viewer.log" 2>&1 &
    fi
    wait' < /dev/null &
WINDOWS=$!
cleanup() {
    trap - EXIT INT TERM
    kill -- -"$WINDOWS" 2>/dev/null  # only the windows' group (the agent stops Isaac Sim itself)
}
trap cleanup EXIT INT TERM

"$ROOT/.venv-seg/bin/python" -u -m agent.run --settings "$@"
