# gpumon — GPU memory monitor (ECE 565 project)

An out-of-process monitor for the VT6L system. It never imports `src/agent/`, `src/vlm/` or `src/sim/`; it watches from outside:

| Source | What it measures |
|---|---|
| NVML (driver) | used / free VRAM, GPU util, PCIe rx/tx, power, SM clock, and **memory per process** grouped as Isaac Sim / agent / other / unattributed |
| psutil | system RAM, resident memory of the simulator and agent processes (CPU offload shows up here) |
| the agent's `events.jsonl` | VLM calls (role, seconds, tokens), instruction reports |
| agent console (bench mode) | model loading, simulator start, ready, per-instruction status, out-of-memory lines |
| `torch_hook/` (opt-in) | PyTorch allocated vs reserved memory from inside the agent |

Why outside: Isaac Sim allocates through Vulkan/RTX, which PyTorch cannot see. Only the driver sees the whole GPU.

The only change in the system itself: `src/vlm/config.py` reads `VLM_*` environment variables (defaults unchanged),
so a benchmark can switch configurations without editing code.

## Plugged in or out (one switch)

gpumon is the ECE 565 course plugin. It is **off by default**, so research runs and other projects never see it.

| | Off (default) | On |
|---|---|---|
| Switch | `GPU_MONITOR = False` in `src/agent/settings.py` | `GPU_MONITOR = True`, or for one run `GPU_MONITOR=1 ./launch/agent.sh` |
| Recorder | not started | `python -m gpumon record` in its own process |
| PyTorch probe | not on `PYTHONPATH` | `gpumon/torch_hook` loaded into the agent |
| Dashboard | camera, task and logs only | plus the GPU column |

The experiments below (`bench`, `throughput`) run only when you call them. Unplugging completely = delete `gpumon/`
and the `GPU_MONITOR` lines: the system still runs (the `VLM_*` variables in `src/vlm/config.py` default to the
settings written there).

## Use (from the repo root, with `.venv-seg/bin/python`)

```bash
# 1. Manual session: record while you use ./launch/agent.sh in another terminal (Ctrl+C or --until-idle to stop)
.venv-seg/bin/python -m gpumon record --label baseline --until-idle
#    with the PyTorch probe, start the agent like this (same folder as the recording's --out):
PYTHONPATH=gpumon/torch_hook GPUMON_TORCH_DIR=out/gpumon/record/<run>/torch ./launch/agent.sh

# 2. Unattended benchmark: configurations x repeats, fixed scene and instructions
.venv-seg/bin/python -m gpumon bench gpumon/examples/exp3_memory.toml --dry-run   # check the commands
.venv-seg/bin/python -m gpumon bench gpumon/examples/exp3_memory.toml             # resumable: done runs are skipped

# 3. Re-analyse
.venv-seg/bin/python -m gpumon report  out/gpumon/bench/exp3_memory/8bit_offload/r01
.venv-seg/bin/python -m gpumon compare out/gpumon/bench/exp3_memory
```

## FlexGen-style throughput experiment (Gemma alone, no simulator)

```bash
.venv-seg/bin/python -m gpumon throughput                       # 4bit_gpu, 4bit_offload, 8bit_offload x batch 1..16
.venv-seg/bin/python -m gpumon throughput --budget-gb 6.8       # memory left next to Isaac Sim, YOLO and the desktop
.venv-seg/bin/python -m gpumon throughput --configs 4bit_gpu 4bit_offload --batches 1 2 4 8 16 32
```

- Baseline `4bit_gpu`: every weight on the GPU. FlexGen's placement `4bit_offload`: the lookup-only per-layer
  embedding table in CPU RAM. Precision: `8bit_offload`, `16bit_offload`.
- Per batch size: a fixed prompt (`--prompt-tokens`), exactly `--gen-tokens` generated per request, `--repeats` runs.
  Throughput = generated tokens / second; efficiency = tokens/s per GiB of GPU memory. A batch that runs out of
  memory ends the sweep: the batch limit of that configuration.
- `--budget-gb` caps PyTorch after loading (the real system also loads the model before the simulator starts).
- Output `out/gpumon/throughput/<date>_<time>/`: `results.csv`, `summary.txt`, `throughput.png`.
- Demo for the presentation (full system, offload on vs off): `python -m gpumon bench gpumon/examples/demo_offload.toml`.

## Terms (OS view)

| Term | Meaning |
|---|---|
| gpumon reads NVML | gpumon asks the GPU driver (NVIDIA Management Library) how much memory each process holds, like `top` for the GPU |
| offload | keep data in a slower level of the memory hierarchy (CPU RAM) and fetch what a step needs over PCIe, like swapping |
| lookup-only table | each token touches one row of it: a small working set, so the rest can live in slower memory |
| allocated vs reserved | memory in tensors vs memory PyTorch's caching allocator holds; the gap is allocator overhead (fragmentation) |
| out of memory | a GPU has no virtual memory to page to: when it is full, the allocation fails and the process exits |
| batch size | requests decoded together; each adds KV cache (the model's memory of earlier tokens), so memory limits it |

## Output

Per run: `samples.csv` (10 Hz), `events.jsonl`, `markers.jsonl`, `console.log`, `torch/`, `summary.json`,
`summary.txt`, `timeline.png`. Per bench: `comparison.csv`, `comparison.txt`, `comparison.png`.

Phases: **load** (Gemma + YOLO) → **sim start** (Isaac Sim + ROS 2) → **working** (instructions; VLM calls shaded).

## Notes on the numbers

- NVML per-process > PyTorch reserved > PyTorch allocated. The gaps are the CUDA context and libraries, then the
  caching allocator's free blocks. A freed tensor does not lower NVML's figure.
- Event times: the agent's `t` is precise but has no absolute anchor. gpumon anchors each trace by the event that
  arrived with the least delay, so VLM-call boundaries are accurate to about the 50 ms poll interval.
- PCIe throughput includes the display (on a laptop the desktop shows ~0.4–0.8 GB/s tx at idle); compare
  "during calls" against "between calls" rather than reading absolute values.
- One Isaac Sim at a time: the bench waits for any running simulator or agent to exit before each run, and stops a
  simulator its own run left behind (never one started before the run).
