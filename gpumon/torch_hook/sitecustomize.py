"""Opt-in PyTorch allocator probe. Active only when this folder is on PYTHONPATH and GPUMON_TORCH_DIR is set
(gpumon bench / record --torch-hook do both); the agent's code is not changed.

Python imports `sitecustomize` at start-up. This one starts a daemon thread that waits until the process has
imported torch AND initialised CUDA itself (it never imports torch or touches CUDA first), then writes
<GPUMON_TORCH_DIR>/torch_<pid>.csv: time, allocated, reserved, peak allocated, peak reserved (MiB), num OOMs.

    allocated  memory held by live tensors (weights, activations, KV cache)
    reserved   memory the caching allocator holds from the driver (allocated + cached free blocks)
    NVML's per-process figure is larger still: it also counts the CUDA context and libraries (cuBLAS, kernels).

Any other sitecustomize on the path (e.g. the distribution's) is still run afterwards.
"""

import os
import sys


def _probe(out_dir: str, period: float) -> None:
    import threading
    import time

    def run() -> None:
        try:
            while True:
                torch = sys.modules.get("torch")
                if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_initialized():
                    break
                time.sleep(0.5)
            mib = 2 ** 20
            path = os.path.join(out_dir, f"torch_{os.getpid()}.csv")
            with open(path, "w", buffering=1) as f:
                f.write("# " + " ".join(getattr(sys, "orig_argv", sys.argv))[:400] + "\n")
                f.write("time,allocated_mib,reserved_mib,peak_allocated_mib,peak_reserved_mib,num_ooms\n")
                while True:
                    s = torch.cuda.memory_stats()
                    f.write(f"{time.time():.3f},{s.get('allocated_bytes.all.current', 0) / mib:.1f},"
                            f"{s.get('reserved_bytes.all.current', 0) / mib:.1f},"
                            f"{s.get('allocated_bytes.all.peak', 0) / mib:.1f},"
                            f"{s.get('reserved_bytes.all.peak', 0) / mib:.1f},{s.get('num_ooms', 0)}\n")
                    time.sleep(period)
        except Exception as e:  # never disturb the host process
            try:
                with open(os.path.join(out_dir, f"torch_{os.getpid()}.error"), "w") as f:
                    f.write(repr(e) + "\n")
            except OSError:
                pass

    threading.Thread(target=run, name="gpumon-torch-probe", daemon=True).start()


def _chain() -> None:
    """Run the next sitecustomize on sys.path (the one this file shadows), if any."""
    import importlib.machinery
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    rest = [p for p in sys.path if os.path.abspath(p or ".") != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec is not None and spec.loader is not None:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


if os.environ.get("GPUMON_TORCH_DIR"):
    try:
        os.makedirs(os.environ["GPUMON_TORCH_DIR"], exist_ok=True)
        _probe(os.environ["GPUMON_TORCH_DIR"], float(os.environ.get("GPUMON_TORCH_PERIOD", "0.1")))
    except Exception:
        pass
try:
    _chain()
except Exception:
    pass
