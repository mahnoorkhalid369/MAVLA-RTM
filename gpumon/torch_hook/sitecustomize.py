"""Opt-in PyTorch allocator probe. Active only when this folder is on PYTHONPATH and GPUMON_TORCH_DIR is set
(gpumon bench / record --torch-hook do both); the agent's code is not changed.

Python imports `sitecustomize` at start-up. This one starts a daemon thread that waits until the process has
imported torch AND initialised CUDA itself (it never imports torch or touches CUDA first), then writes
<GPUMON_TORCH_DIR>/torch_<pid>.csv: time, allocated, reserved, peak allocated, peak reserved (MiB), num OOMs.

    allocated  memory held by live tensors (weights, activations, KV cache)
    reserved   memory the caching allocator holds from the driver (allocated + cached free blocks)
    NVML's per-process figure is larger still: it also counts the CUDA context and libraries (cuBLAS, kernels).

It also writes <GPUMON_TORCH_DIR>/vlm_<pid>.json once the VLM has loaded: which model, the bit size its weights are
actually stored in (from the layer types bitsandbytes created, not from the settings), whether the per-layer
embeddings were moved to the CPU, and the load time. For that it wraps vlm.model.VisionLanguageModel.__init__ when
that module is first imported (an import hook; the module's code is not changed).

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


def _describe_vlm(vlm) -> dict:
    """What was loaded, read from the loaded model itself."""
    import collections
    kinds = collections.Counter(type(m).__name__ for m in vlm.model.modules())
    if kinds.get("Linear4bit"):
        q = getattr(getattr(vlm.model, "config", None), "quantization_config", None)
        bits = f"4-bit ({getattr(q, 'bnb_4bit_quant_type', 'nf4')})"
    elif kinds.get("Linear8bitLt"):
        bits = "8-bit (LLM.int8)"
    else:
        dt = next(vlm.model.parameters()).dtype
        bits = {"torch.bfloat16": "16-bit (bf16)", "torch.float16": "16-bit (fp16)", "torch.float32": "32-bit"}.get(str(dt), str(dt))
    cfg = vlm.config
    emb = vlm.model.get_submodule("model.language_model.embed_tokens_per_layer") if cfg.per_layer_embeddings_on_cpu else None
    return {"model": os.path.basename(str(cfg.model_path).rstrip("/")), "weights": bits, "setting": cfg.quantization,
            "quantized_layers": kinds.get("Linear4bit", 0) + kinds.get("Linear8bitLt", 0),
            "embeddings_on_cpu": bool(cfg.per_layer_embeddings_on_cpu),
            "embeddings_device": str(next(emb.parameters()).device) if emb is not None else "cuda",
            "load_seconds": round(float(getattr(vlm, "load_seconds", 0.0)), 1)}


def _watch_vlm(out_dir: str) -> None:
    """Wrap VisionLanguageModel.__init__ as soon as vlm.model is imported; after it returns, write vlm_<pid>.json."""
    import importlib.abc
    import json
    import time

    def wrap(module) -> None:
        cls = getattr(module, "VisionLanguageModel", None)
        if cls is None:
            return
        init = cls.__init__

        def __init__(self, *a, **k):
            init(self, *a, **k)
            try:
                rec = {"time": time.time(), **_describe_vlm(self)}
                with open(os.path.join(out_dir, f"vlm_{os.getpid()}.json"), "w") as f:
                    json.dump(rec, f)
            except Exception as e:  # never disturb the host process
                try:
                    with open(os.path.join(out_dir, f"vlm_{os.getpid()}.error"), "w") as f:
                        f.write(repr(e) + "\n")
                except OSError:
                    pass
        cls.__init__ = __init__

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "vlm.model":
                return None
            sys.meta_path.remove(self)  # find the real spec, then wrap its loader
            try:
                import importlib.util
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            run = spec.loader.exec_module

            def exec_module(module):
                run(module)
                try:
                    wrap(module)
                except Exception:
                    pass
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, Finder())


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
        _watch_vlm(os.environ["GPUMON_TORCH_DIR"])
    except Exception:
        pass
try:
    _chain()
except Exception:
    pass
