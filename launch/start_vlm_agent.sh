#!/usr/bin/env bash
# Start the vision-language pick-and-place agent (Gemma 4 E2B-it) next to a running live sim.
#   1. Terminal 1: ./launch/start_live.sh
#   2. Terminal 2: ./launch/start_vlm_agent.sh            (add --voice for spoken tasks)
#   3. Terminal 3: source /opt/ros/jazzy/setup.bash && python3 ros_nodes/text_command_node.py --topic task
# Tasks are typed in terminal 3, e.g. "put the red block on the green block".
# Loading the model takes ~15 s and briefly needs ~6.5 GB of GPU memory (2.3 GB once loaded).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
if [ ! -f models/gemma-4-E2B-it/model.safetensors ]; then
    echo "[start_vlm_agent] model not found in models/gemma-4-E2B-it; it will be downloaded from Hugging Face (10 GB)"
fi
source /opt/ros/jazzy/setup.bash
exec "$ROOT/.venv-vlm/bin/python" -u ros_nodes/vlm_agent_node.py "$@"
