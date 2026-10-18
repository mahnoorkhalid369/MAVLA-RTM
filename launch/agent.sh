#!/usr/bin/env bash
# The interactive agent: Gemma -> YOLO-seg -> Isaac Sim (started by the agent) -> demo viewer -> instruction prompt.
# Motion is BLOCKED unless you pass --motion real. Authorization and the simulator's safety layer apply in every mode.
#
#   ./launch/agent.sh                                        # default cube scene, motion blocked
#   ./launch/agent.sh --scenario laboratory                  # laboratory, easy, seed 0
#   ./launch/agent.sh --scenario laboratory --difficulty medium --seed 42
#   ./launch/agent.sh --motion real                          # the robot moves (in the simulator)
#   ./launch/agent.sh --motion dry_run                       # motions are planned and logged, never sent
# Options:
#   --scenario NAME   laboratory (scenario library) or a JSON test scene in scenarios/scenes/ (default: cube scene)
#   --difficulty D    easy | medium | hard (laboratory)      --seed N    layout seed (laboratory)
#   --motion M        blocked (default) | dry_run | real
#   --headless        no Isaac Sim window (the demo viewer still opens)
#   --no-viewer       no demo viewer (logs are printed in this terminal as well)
#   --camera-windows  also the raw wrist and table RGB-D windows
#   anything else is passed to `python -m agent.run` (e.g. --instruction "...", --out DIR)
# Type instructions at "instruction>"; an empty line quits and stops the simulator.
# Outputs: out/demo/<date>/live_<time>/<nn>/ (events.jsonl, images/, result.json); viewer feed out/demo/live/
# Extra viewer options: DEMO_VIEWER_ARGS="--snapshot-every 3 --snapshot-dir out/demo/shots" ./launch/agent.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
GUI="--gui"
VIEWER=1
CAMERAS=0
AGENT_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --headless) GUI="" ;;
        --no-viewer) VIEWER=0 ;;
        --camera-windows) CAMERAS=1 ;;
        --scenario|--difficulty|--seed|--motion) AGENT_ARGS+=("$1" "$2"); shift ;;
        *) AGENT_ARGS+=("$1") ;;
    esac
    shift
done
if pgrep -f "sim.run_live|sim.run_scenario" > /dev/null; then
    echo "[agent.sh] a simulator is already running (one Isaac Sim at a time on this GPU); stop it first"; exit 1
fi
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p out/logs out/demo/live
echo "[agent.sh] ROS_DOMAIN_ID=$ROS_DOMAIN_ID"

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

"$ROOT/.venv-seg/bin/python" -u -m agent.run --start-sim $GUI --live-dir out/demo/live "${AGENT_ARGS[@]}"
