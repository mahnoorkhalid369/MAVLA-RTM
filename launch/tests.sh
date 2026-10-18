#!/usr/bin/env bash
# Automated tests, each in the right Python environment. No robot motion in any mode.
#
#   ./launch/tests.sh            # all: frozen-file check + full suite (agent env) + system-Python suite
#   ./launch/tests.sh agent      # full suite in .venv-seg (all tests, incl. agent / model / saved-frame tests)
#   ./launch/tests.sh system     # /usr/bin/python3 suite (agent tests skip: no pydantic / langgraph there)
#   ./launch/tests.sh frozen     # checksums of the frozen manipulation baseline
#   ./launch/tests.sh sim        # Isaac Sim smoke check: laboratory headless, ROS topics, then stop
#   ./launch/tests.sh agent -k controls      # extra arguments go to pytest
# Tests live in tests/ (unit and integration; shared fakes in tests/support/). Research experiments are in
# evaluation/ (./launch/evaluate.sh).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
MODE="${1:-all}"
[ $# -gt 0 ] && shift

agent_suite() { env -u PYTHONPATH -u AMENT_PREFIX_PATH "$ROOT/.venv-seg/bin/python" -m pytest -q "$@"; }
system_suite() { /usr/bin/python3 -m pytest -q "$@"; }
frozen() {
    grep -v '^#' docs/baselines/manipulation_A_2026-10-14/MANIFEST.sha256 | sha256sum -c --quiet && echo "[tests] frozen baseline: OK"
}
sim_smoke() {
    if pgrep -f "sim.run_live|sim.run_scenario" > /dev/null; then
        echo "[tests] a simulator is already running; stop it first (one Isaac Sim at a time)"; return 1
    fi
    source /opt/ros/jazzy/setup.bash
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}" OMNI_KIT_ACCEPT_EULA=YES
    mkdir -p out/logs
    local log=out/logs/tests_sim.log
    echo "[tests] starting the laboratory headless (ROS_DOMAIN_ID=$ROS_DOMAIN_ID, log $log) ..."
    setsid "$ROOT/.venv/bin/python" -u -m sim.run_scenario --scenario laboratory --difficulty easy --seed 42 --headless \
        < /dev/null > "$log" 2>&1 &
    local pid=$! ok=1 t=0
    until ros2 topic list 2>/dev/null | grep -q "^/table_camera/color/image_raw$"; do
        sleep 5; t=$((t + 5))
        if ! kill -0 "$pid" 2>/dev/null || [ $t -ge 600 ]; then echo "[tests] FAIL: the simulator did not come up (see $log)"; ok=0; break; fi
    done
    if [ $ok = 1 ]; then
        local topics; topics="$(ros2 topic list)"
        for t in /table_camera/color/image_raw /table_camera/depth/image_rect_raw /camera/color/image_raw \
                 /camera/depth/image_rect_raw /joint_states /joint_trajectory /clock /sim/ground_truth/objects /safety/events; do
            if grep -qx "$t" <<< "$topics"; then echo "[tests]   ok  $t"; else echo "[tests]   MISSING $t"; ok=0; fi
        done
        if timeout 20 ros2 topic echo --once /joint_states > /dev/null 2>&1; then echo "[tests]   ok  /joint_states publishing"
        else echo "[tests]   FAIL /joint_states silent"; ok=0; fi
        grep -m1 "\[run_scenario\]" "$log" | sed 's/^/[tests]   /'
        if grep -q "Traceback" "$log"; then echo "[tests]   FAIL: traceback in $log"; ok=0; fi
    fi
    kill -INT -- -"$pid" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL -- -"$pid" 2>/dev/null
    [ $ok = 1 ] && echo "[tests] sim: PASS" || { echo "[tests] sim: FAIL"; return 1; }
}

case "$MODE" in
    all) frozen && agent_suite "$@" && system_suite "$@" ;;
    agent) agent_suite "$@" ;;
    system) system_suite "$@" ;;
    frozen) frozen ;;
    sim) sim_smoke ;;
    -h|--help) sed -n 2,11p "$0" ;;
    *) echo "unknown mode $MODE (all | agent | system | frozen | sim)"; exit 1 ;;
esac
