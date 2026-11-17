#!/usr/bin/env bash
# The robot session (v2: master + workers). Edit src/agent/settings.py, then run:
#
#   ./launch/agent.sh
#
# It checks the settings, prints a summary, loads Gemma and YOLO, starts Isaac Sim with the configured scene, opens the
# dashboard and gives you the "instruction>" prompt. An empty line quits and stops the simulator.
# Each instruction's trace: <OUTPUT_DIR>/<date>/live_<time>/<nn>/events.jsonl (and result.json).
#
#   ./launch/agent.sh --motion blocked            # any agent.run option overrides the setting for one run
#   DASHBOARD=0 GPU_MONITOR=1 ./launch/agent.sh   # no dashboard; record the GPU (ECE 565 plugin)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"  # this version's packages (v1/src or v2/src)
cd "$ROOT" || exit 1
SETTINGS="$(env -u AMENT_PREFIX_PATH PYTHONPATH="$ROOT/src" "$ROOT/.venv-seg/bin/python" -m agent.settings --shell)" || exit 1
eval "$SETTINGS"
if ps -eo args= | grep -Eq '^[^ ]*python[0-9.]* (-u )?-m sim\.run_(live|scenario)( |$)'; then
    echo "[agent.sh] a simulator is already running (one Isaac Sim at a time on this GPU); stop it first"; exit 1
fi

# GPU recorder (its own process), started before anything loads so the recording starts from idle memory
GPUMON=""
AGENT_ENV=()
if [ "$GPU_MONITOR" = 1 ]; then
    GPUMON_DIR="$ROOT/out/gpumon/record/$(date +%Y-%m-%d_%H%M%S)"
    mkdir -p "$GPUMON_DIR" "$LOG_DIR"
    env -u AMENT_PREFIX_PATH PYTHONPATH="$ROOT/src" "$ROOT/.venv-seg/bin/python" -m gpumon record --out "$GPUMON_DIR" \
        --events "$ROOT/$OUTPUT_DIR" --label session < /dev/null > "$LOG_DIR/gpumon.log" 2>&1 &
    GPUMON=$!
    AGENT_ENV=(GPUMON_TORCH_DIR="$GPUMON_DIR/torch")
    export PYTHONPATH="$ROOT/gpumon/torch_hook:$PYTHONPATH"
fi
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID OMNI_KIT_ACCEPT_EULA=YES
mkdir -p "$LOG_DIR" "$LIVE_DIR"

# the dashboard, once the table camera streams (its own process group, stopped on exit)
if [ "$DASHBOARD" = 1 ]; then
    setsid bash -c '
        until ros2 topic list 2>/dev/null | grep -q "^/table_camera/color/image_raw$"; do sleep 3; done
        exec /usr/bin/python3 src/agent/dashboard.py --live-dir "'"$LIVE_DIR"'" '"${GPUMON_DIR:+--gpumon-dir $GPUMON_DIR}"' \
            < /dev/null > "'"$LOG_DIR"'/dashboard.log" 2>&1' < /dev/null &
    WINDOWS=$!
fi
cleanup() {
    trap - EXIT INT TERM
    [ -n "$WINDOWS" ] && kill -- -"$WINDOWS" 2>/dev/null
    if [ -n "$GPUMON" ]; then
        kill -TERM "$GPUMON" 2>/dev/null
        wait "$GPUMON" 2>/dev/null
        echo "[agent.sh] GPU recording of this session: $GPUMON_DIR (summary.txt, timeline.png)"
    fi
}
trap cleanup EXIT INT TERM

env "${AGENT_ENV[@]}" "$ROOT/.venv-seg/bin/python" -u -m agent.run --settings "$@"
