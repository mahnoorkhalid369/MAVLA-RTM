#!/usr/bin/env bash
# Start v3/dialogue (Isaac Sim + Gemma + YOLO): the frozen v3 robot with the conversation agents. Uses v2's settings and code as a library; v2 itself is not changed.
#   ./launch.sh [--gui] [--instructions tests/sample.json]
#   SCREEN=0 ./launch.sh ...   without the live screen
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V3="$HERE/.."
V2="$V3/../v2"
LIVE="$V3/out/live"  # the files the live screen reads; run.py writes them
if ps -eo args= | grep -Eq '^[^ ]*python[0-9.]* (-u )?-m sim\.run_(live|scenario)( |$)'; then
    echo "[v3] a simulator is already running (one Isaac Sim at a time on this GPU); stop it first"; exit 1
fi
eval "$(cd "$V2" && env -u AMENT_PREFIX_PATH PYTHONPATH="$V2/src" "$V2/.venv-seg/bin/python" -m agent.settings --shell)" || exit 1
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID OMNI_KIT_ACCEPT_EULA=YES
mkdir -p "$LIVE"

# The live screen is v2's dashboard, a separate read-only window: the table camera with the outlined, numbered
# objects (target green, destination blue) and every log line as it is written. It opens once the camera streams.
if [ "${SCREEN:-1}" = 1 ] && [ -n "$DISPLAY" ]; then
    setsid bash -c 'until ros2 topic list 2>/dev/null | grep -q "^/table_camera/color/image_raw$"; do sleep 3; done
        exec /usr/bin/python3 "$0/src/agent/dashboard.py" --live-dir "$1" $2 > "$1/screen.log" 2>&1' \
        "$V2" "$LIVE" "$SCREEN_OPTIONS" < /dev/null &
    WINDOW=$!
fi
cleanup() {
    trap - EXIT INT TERM
    [ -n "$WINDOW" ] && kill -- -"$WINDOW" 2>/dev/null
}
trap cleanup EXIT INT TERM

export V2_REASONING_TOKENS="${V2_REASONING_TOKENS:-1200}"  # the master's thinking plus answer, in tokens
export PYTHONPATH="$HERE:$V3:$V2/src${PYTHONPATH:+:$PYTHONPATH}"
"$V2/.venv-seg/bin/python" -u "$HERE/run.py" "$@"
