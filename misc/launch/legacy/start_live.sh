#!/usr/bin/env bash
# Start the live setup in one go:
#   1. Isaac Sim: VT6L scene streaming camera + TF over ROS 2 (sim/run_live.py, .venv Python)
#   2. Camera viewers: wrist and table camera, RGB and depth side by side (ros_nodes/camera_viewer.py, system Python)
#   3. Command interpreter: /instruction text -> /joint_command (ros_nodes/command_interpreter.py)
#   4. Microphone node (only with --voice): host microphone -> /microphone/*
# To type instructions, open a second terminal and run:
#   source /opt/ros/jazzy/setup.bash && python3 ros_nodes/text_command_node.py
# Ctrl-C stops them all. Other arguments go to sim/run_live.py (e.g. --headless).
#
#   ./launch/legacy/start_live.sh              # text commands only (microphone off)
#   ./launch/legacy/start_live.sh --voice      # also start the microphone node
#   ./launch/legacy/start_live.sh --headless

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VOICE=0
SIM_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--voice" ]; then VOICE=1; else SIM_ARGS+=("$arg"); fi
done
cd "$ROOT" || exit 1
source /opt/ros/jazzy/setup.bash
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p out/logs

pids=()
cleanup() {
    trap - EXIT INT TERM
    echo; echo "[start_live] stopping..."
    kill -INT "${pids[@]}" 2>/dev/null
    sleep 2
    kill "${pids[@]}" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "[start_live] starting Isaac Sim (first launch takes 1-2 min); log: out/logs/sim.log"
"$ROOT/.venv/bin/python" -u -m sim.run_live "${SIM_ARGS[@]}" > out/logs/sim.log 2>&1 &
pids+=($!)

echo "[start_live] waiting for the camera stream..."
until ros2 topic list 2>/dev/null | grep -q '^/camera/color/image_raw$'; do
    if ! kill -0 "${pids[0]}" 2>/dev/null; then
        echo "[start_live] Isaac Sim exited early; see out/logs/sim.log"; exit 1
    fi
    sleep 2
done

/usr/bin/python3 ros_nodes/camera_viewer.py > out/logs/camera_viewer.log 2>&1 &
pids+=($!)
/usr/bin/python3 ros_nodes/camera_viewer.py --title "Table camera - RGB and depth" \
    --color-topic /table_camera/color/image_raw --depth-topic /table_camera/depth/image_rect_raw \
    --depth-min 1.2 --depth-max 3.2 --ros-args -r __node:=table_viewer > out/logs/table_viewer.log 2>&1 &
pids+=($!)
echo "[start_live] camera viewers started: wrist + table camera (q in a window closes it)"

/usr/bin/python3 ros_nodes/command_interpreter.py > out/logs/command_interpreter.log 2>&1 &
pids+=($!)
echo "[start_live] command interpreter started. To type instructions, open a second terminal and run:"
echo "             ./launch/legacy/tasks.sh   (free-form tasks, needs ./launch/legacy/start_vlm_agent.sh running)"

if [ "$VOICE" = 1 ]; then
    /usr/bin/python3 ros_nodes/microphone_node.py &
    pids+=($!)
else
    echo "[start_live] microphone off (text commands only); use --voice to turn it on"
fi

wait -n
