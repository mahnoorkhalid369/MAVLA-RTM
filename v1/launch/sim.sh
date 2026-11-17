#!/usr/bin/env bash
# Isaac Sim only (no agent): a scene streaming cameras, joint states and ground truth over ROS 2 (domain 42).
# Nothing is commanded: the arm stays at HOME unless another process sends a trajectory.
#
#   ./launch/sim.sh                                          # laboratory, easy, seed 0, window
#   ./launch/sim.sh --difficulty hard --seed 42 --headless
#   ./launch/sim.sh --scenario default                       # the frozen default cube scene
#   ./launch/sim.sh --scenario ap_occlusion                  # any scenario plugged in under scenarios/
# Options: --scenario NAME (default laboratory) | --difficulty easy|medium|hard | --seed N | --headless
# Stop with Ctrl+C. Scene ground truth (evaluation only) is written to out/scenarios/live/.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"  # this version's packages (v1/src or v2/src)
cd "$ROOT" || exit 1
SCENARIO=laboratory
DIFFICULTY=easy
SEED=0
HEADLESS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --scenario) SCENARIO="$2"; shift ;;
        --difficulty) DIFFICULTY="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --headless) HEADLESS=(--headless) ;;
        -h|--help) sed -n 2,10p "$0"; exit 0 ;;
        *) echo "unknown option $1 (use --scenario, --difficulty, --seed, --headless)"; exit 1 ;;
    esac
    shift
done
if ps -eo args= | grep -Eq '^[^ ]*python[0-9.]* (-u )?-m sim\.run_(live|scenario)( |$)'; then
    echo "[sim.sh] a simulator is already running (one Isaac Sim at a time on this GPU); stop it first"; exit 1
fi
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export OMNI_KIT_ACCEPT_EULA=YES
echo "[sim.sh] ROS_DOMAIN_ID=$ROS_DOMAIN_ID scenario=$SCENARIO difficulty=$DIFFICULTY seed=$SEED"
exec "$ROOT/.venv/bin/python" -u -m sim.run_scenario --scenario "$SCENARIO" --difficulty "$DIFFICULTY" --seed "$SEED" "${HEADLESS[@]}"
