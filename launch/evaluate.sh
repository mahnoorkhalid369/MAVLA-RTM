#!/usr/bin/env bash
# Research evaluation (evaluation/): each experiment in its own Python environment, outputs under out/.
# These are experiments, not tests; ./launch/tests.sh runs the automated tests.
#
#   ./launch/evaluate.sh list                                      # this help
#   ./launch/evaluate.sh planner      [--out FILE]                 # real Gemma planner check (GPU, no simulator)
#   ./launch/evaluate.sh segmentation [FRAMES_DIR]                 # YOLO + segmentation approaches on saved frames
#                                                                  #   (default: a copy of out/lab_seg_frames)
#   ./launch/evaluate.sh validate-scenarios [validate_scenarios.py options]   # Isaac, headless; default laboratory
#   ./launch/evaluate.sh capture-frames     [lab_capture_frames.py options]   # Isaac, headless
#   ./launch/evaluate.sh probe snapshot|record FILE                # joint states + ground truth over ROS 2
#   ./launch/evaluate.sh lab-exp1 --out out/lab_exp1/RUN [--seed 42]          # Experiment 1 protocol
#                                                                  #   (its inspection phase MOVES the robot in the simulator)
#   ./launch/evaluate.sh score --run DIR --meta FILE --truth FILE  # score an Experiment 1 run
# ROS_DOMAIN_ID defaults to 42. One Isaac Sim at a time.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
NAME="${1:-list}"
[ $# -gt 0 ] && shift
SEG="$ROOT/.venv-seg/bin/python"
ISAAC="$ROOT/.venv/bin/python"
offline() { env -u PYTHONPATH -u AMENT_PREFIX_PATH "$@"; }
with_ros() { source /opt/ros/jazzy/setup.bash; export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}" OMNI_KIT_ACCEPT_EULA=YES; "$@"; }
one_isaac() {
    if pgrep -f "sim.run_live|sim.run_scenario" > /dev/null; then
        echo "[evaluate] a simulator is already running (one Isaac Sim at a time); stop it first"; exit 1
    fi
}
STAMP="$(date +%Y-%m-%d_%H%M%S)"

case "$NAME" in
    planner)
        [ $# -eq 0 ] && set -- --out "out/eval/planner_check_$STAMP.json"
        mkdir -p out/eval
        offline "$SEG" -m evaluation.planner_check "$@" ;;
    segmentation)
        FRAMES="${1:-}"
        if [ -z "$FRAMES" ]; then  # the study writes its results into the frames folder: work on a copy
            FRAMES="out/eval/seg_frames_$STAMP"; mkdir -p out/eval; cp -r out/lab_seg_frames "$FRAMES"
            echo "[evaluate] frames copied to $FRAMES"
        fi
        offline "$SEG" evaluation/seg_generalization.py "$FRAMES" ;;
    validate-scenarios)
        one_isaac
        [ $# -eq 0 ] && set -- --scenario laboratory
        with_ros "$ISAAC" evaluation/validate_scenarios.py "$@" ;;
    capture-frames)
        one_isaac
        with_ros "$ISAAC" evaluation/lab_capture_frames.py "$@" ;;
    probe)
        with_ros "$SEG" -m evaluation.lab_probe "$@" ;;
    lab-exp1)
        one_isaac
        with_ros "$SEG" evaluation/lab_exp1.py "$@" ;;
    score)
        /usr/bin/python3 evaluation/lab_exp1_score.py "$@" ;;
    list|-h|--help)
        sed -n 2,15p "$0" ;;
    *)
        echo "unknown experiment $NAME"; sed -n 2,15p "$0"; exit 1 ;;
esac
