#!/usr/bin/env bash
# One command for everything: simulation + one camera window + the agent taking typed tasks
# in THIS terminal, with its logs printed here in order.
#   1. Isaac Sim with its window (sim/run_live.py, .venv); log in out/logs/sim.log
#   2. One window: table camera RGB + depth (top row), wrist camera RGB + depth (ros_nodes/cameras_view.py)
#   3. The Gemma agent in the foreground, reading tasks from this terminal (--stdin)
# Type a task and press Enter; 'stop' cancels a task, 'quit' or Ctrl-C stops everything.
#
#   ./launch/run.sh              # Isaac Sim window + camera window
#   ./launch/run.sh --headless   # no Isaac Sim window (less GPU; the cameras still show everything)
# Other arguments go to the agent (e.g. --thinking, --no-thinking).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"  # this version's packages (v1/src or v2/src)
SIM_ARGS=()
AGENT_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--headless" ]; then SIM_ARGS=(--headless); else AGENT_ARGS+=("$arg"); fi
done
cd "$ROOT" || exit 1
if pgrep -f "sim.run_live" > /dev/null; then
    echo "[run] a simulation is already running (start_live.sh?); stop it first"; exit 1
fi
source /opt/ros/jazzy/setup.bash
export OMNI_KIT_ACCEPT_EULA=YES
export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}] {message}"  # short log lines
mkdir -p out/logs

pids=()
cleanup() {
    trap - EXIT INT TERM
    echo; echo "[run] stopping..."
    kill -INT "${pids[@]}" 2>/dev/null
    sleep 2
    kill "${pids[@]}" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "[run] starting Isaac Sim (1-2 min); log: out/logs/sim.log"
"$ROOT/.venv/bin/python" -u -m sim.run_live "${SIM_ARGS[@]}" < /dev/null > out/logs/sim.log 2>&1 &
pids+=($!)
until ros2 topic list 2>/dev/null | grep -q '^/table_camera/color/image_raw$'; do
    if ! kill -0 "${pids[0]}" 2>/dev/null; then
        echo "[run] Isaac Sim exited early; see out/logs/sim.log"; exit 1
    fi
    sleep 2
done

/usr/bin/python3 src/ros_nodes/cameras_view.py < /dev/null > out/logs/cameras_view.log 2>&1 &
pids+=($!)
echo "[run] camera window open; loading the agent (~1 min)..."

"$ROOT/.venv-vlm/bin/python" -u src/ros_nodes/vlm_agent_node.py --stdin "${AGENT_ARGS[@]}"
