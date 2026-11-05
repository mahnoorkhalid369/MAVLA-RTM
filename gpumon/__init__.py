"""gpumon: an out-of-process GPU / RAM monitor for the VT6L system (ECE 565 project).

It never imports agent/, vlm/ or sim/. It observes from outside:
  - NVML (the driver): device memory, utilisation, PCIe traffic, and memory per process (Isaac Sim's Vulkan/RTX
    memory is invisible to PyTorch, so the driver is the only view of the whole GPU);
  - psutil: system RAM and the resident memory of the simulator and agent processes;
  - the agent's own events.jsonl traces: VLM calls (role, seconds, tokens), steps, checks, reports;
  - optionally, PyTorch's allocator counters from inside a process, through gpumon/torch_hook (PYTHONPATH only).

    python -m gpumon record --out out/gpumon/manual_01       # sample while you use ./launch/agent.sh
    python -m gpumon bench gpumon/examples/exp3_memory.toml  # configurations x repeats, then summaries
    python -m gpumon report out/gpumon/manual_01             # summary.json, summary.txt, figures
    python -m gpumon compare out/gpumon/bench/<name>         # one table + figures across configurations
"""
