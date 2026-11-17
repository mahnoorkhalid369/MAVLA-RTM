#!/usr/bin/env bash
# One command from any directory. This starts only Isaac Sim, never a physical robot.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$ROOT/.venv-seg/bin/python" -u scripts/pipeline_review.py "$@"
