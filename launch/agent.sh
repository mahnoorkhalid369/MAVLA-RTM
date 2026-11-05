#!/usr/bin/env bash
# The robot session. Edit agent/settings.py (scene, motion, models, windows, tracing, output), then run:
#
#   ./launch/agent.sh
#
# It validates the settings, prints a summary, loads Gemma and YOLO, starts Isaac Sim with the configured scene,
# opens the dashboard (one window: cameras, YOLO masks, task, logs, GPU) and gives you the "instruction>" prompt.
# With GPU_MONITOR on, gpumon records the session in the background (out/gpumon/record/<date>_<time>/: summary.txt,
# timeline.png), as a separate process: the agent itself is unchanged. Every agent handoff is printed
# when TRACE is on; the full trace of each instruction is in <OUTPUT_DIR>/<date>/live_<time>/<nn>/events.jsonl.
# An empty instruction line quits and stops the simulator.
#
# Optional, for one run only (agent/settings.py stays the place to configure):
#   ./launch/agent.sh --motion blocked            # any agent.run option overrides the setting (see --help)
#   ./launch/agent.sh --headless                  # no Isaac Sim window
#   DASHBOARD=0 GPU_MONITOR=0 ./launch/agent.sh   # no dashboard, no GPU recording
#   DEMO_VIEWER=1 CAMERA_WINDOWS=1 ./launch/agent.sh   # the older separate windows
#   DEMO_VIEWER_ARGS="--snapshot-every 3 --snapshot-dir out/demo/shots" ./launch/agent.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
SETTINGS="$(env -u PYTHONPATH -u AMENT_PREFIX_PATH "$ROOT/.venv-seg/bin/python" -m agent.settings --shell)" || exit 1
eval "$SETTINGS"
if ps -eo args= | grep -Eq '^[^ ]*python[0-9.]* (-u )?-m sim\.run_(live|scenario)( |$)'; then
    echo "[agent.sh] a simulator is already running (one Isaac Sim at a time on this GPU); stop it first"; exit 1
fi

# GPU recorder (gpumon, its own process) before anything loads, so the recording starts from idle memory
GPUMON=""
AGENT_ENV=()
if [ "$GPU_MONITOR" = 1 ]; then
    GPUMON_DIR="$ROOT/out/gpumon/record/$(date +%Y-%m-%d_%H%M%S)"
    mkdir -p "$GPUMON_DIR" "$LOG_DIR"
    env -u PYTHONPATH -u AMENT_PREFIX_PATH "$ROOT/.venv-seg/bin/python" -m gpumon record --out "$GPUMON_DIR" \
        --events "$ROOT/$OUTPUT_DIR" --label session < /dev/null > "$LOG_DIR/gpumon.log" 2>&1 &
    GPUMON=$!
    # PyTorch allocated / reserved from inside the agent: a start-up hook on PYTHONPATH (gpumon/torch_hook)
    AGENT_ENV=(GPUMON_TORCH_DIR="$GPUMON_DIR/torch")
fi
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID OMNI_KIT_ACCEPT_EULA=YES
mkdir -p "$LOG_DIR" "$LIVE_DIR"

# windows in their own process group (started once the table camera streams), stdin closed, stopped on exit
setsid bash -c '
    until ros2 topic list 2>/dev/null | grep -q "^/table_camera/color/image_raw$"; do sleep 3; done
    if [ "'"$DASHBOARD"'" = 1 ]; then
        /usr/bin/python3 agent/dashboard.py --live-dir "'"$LIVE_DIR"'" '"${GPUMON_DIR:+--gpumon-dir $GPUMON_DIR}"' '"$DASHBOARD_ARGS"' < /dev/null > "'"$LOG_DIR"'/dashboard.log" 2>&1 &
    fi
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
    if [ -n "$GPUMON" ]; then  # stop the recorder; it writes summary.txt and timeline.png before exiting
        kill -TERM "$GPUMON" 2>/dev/null
        wait "$GPUMON" 2>/dev/null
        echo "[agent.sh] GPU recording of this session: $GPUMON_DIR (summary.txt, timeline.png)"
        sed -n '/^memory/,/^tasks/p' "$GPUMON_DIR/summary.txt" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

if [ -n "$GPUMON" ]; then
    export PYTHONPATH="$ROOT/gpumon/torch_hook${PYTHONPATH:+:$PYTHONPATH}"
fi
env "${AGENT_ENV[@]}" "$ROOT/.venv-seg/bin/python" -u -m agent.run --settings "$@"
