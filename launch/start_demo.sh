#!/usr/bin/env bash
# Text-driven demo of the multi-agent slice (agent/, ARCHITECTURE.md rev 3). See DEMO.md.
#   1. the agent loads Gemma, then the YOLO-seg detector (GPU order matters on 12 GB)
#   2. the agent starts Isaac Sim (window unless --headless)
#   3. the demo viewer opens once the table camera streams: live camera + masks / ids / target /
#      destination + supervisor state + live agent logs (read-only; closing it does not stop the task)
#   4. type instructions at the "instruction>" prompt, answer questions at "answer>" (this terminal);
#      an empty instruction line quits and stops everything
#
#   ./launch/start_demo.sh                    # Isaac Sim window + demo viewer
#   ./launch/start_demo.sh --headless         # no Isaac Sim window (demo viewer still opens)
#   ./launch/start_demo.sh --no-viewer        # terminal only (logs are printed here as well)
#   ./launch/start_demo.sh --camera-windows   # also the raw wrist + table RGB-D windows
#   ./launch/start_demo.sh --scenario ap_occlusion   # an active-perception test scene (sim/scenarios/)
# Outputs: out/demo/<date>/live_<time>/<nn>/ (events.jsonl, images/, result.json); viewer feed out/demo/live/
# Extra viewer options: DEMO_VIEWER_ARGS="--snapshot-every 3 --snapshot-dir out/demo/shots" ./launch/start_demo.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
GUI="--gui"
VIEWER=1
CAMERAS=0
SCENARIO=()
NEXT_IS_SCENARIO=0
for arg in "$@"; do
    if [ "$NEXT_IS_SCENARIO" = 1 ]; then SCENARIO=(--scenario "$arg"); NEXT_IS_SCENARIO=0; continue; fi
    case "$arg" in
        --scenario) NEXT_IS_SCENARIO=1 ;;
        --headless) GUI="" ;;
        --no-viewer) VIEWER=0 ;;
        --camera-windows) CAMERAS=1 ;;
        *) echo "unknown option $arg (use --headless, --no-viewer, --camera-windows, --scenario NAME)"; exit 1 ;;
    esac
done
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p out/logs out/demo/live

# windows in their own process group (started once the table camera streams), stdin closed, stopped on exit
setsid bash -c '
    until ros2 topic list 2>/dev/null | grep -q "^/table_camera/color/image_raw$"; do sleep 3; done
    if [ "'"$VIEWER"'" = 1 ]; then
        /usr/bin/python3 agent/viewer.py --live-dir out/demo/live '"$DEMO_VIEWER_ARGS"' < /dev/null > out/logs/demo_viewer.log 2>&1 &
    fi
    if [ "'"$CAMERAS"'" = 1 ]; then
        /usr/bin/python3 ros_nodes/camera_viewer.py < /dev/null > out/logs/demo_wrist_viewer.log 2>&1 &
        /usr/bin/python3 ros_nodes/camera_viewer.py --title "Table camera - RGB and depth" \
            --color-topic /table_camera/color/image_raw --depth-topic /table_camera/depth/image_rect_raw \
            --depth-min 1.2 --depth-max 3.2 --ros-args -r __node:=table_viewer < /dev/null > out/logs/demo_table_viewer.log 2>&1 &
    fi
    wait' < /dev/null &
WINDOWS=$!
cleanup() {
    trap - EXIT INT TERM
    kill -- -"$WINDOWS" 2>/dev/null  # only the windows' group (the agent stops Isaac Sim itself)
}
trap cleanup EXIT INT TERM

"$ROOT/.venv-seg/bin/python" -u -m agent.run --start-sim $GUI --live-dir out/demo/live "${SCENARIO[@]}"
