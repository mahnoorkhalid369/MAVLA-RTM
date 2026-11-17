"""FlexGen-style throughput experiment: Gemma alone (no simulator), weight placement x precision x batch size.

FlexGen's claim (Sheng et al., ICML 2023): moving weights the GPU does not need every step to CPU memory, and storing
weights in 4 bits, frees GPU memory; the freed memory holds a larger batch (more KV cache), and a larger batch gives
more tokens per second. This experiment measures that chain on one GPU:

    configuration (where the weights live, how many bits)  ->  GPU memory left  ->  largest batch that fits
                                                           ->  tokens per second at each batch size

One process per configuration (placement and precision are fixed at load time). Per batch size it runs a fixed prompt
of --prompt-tokens and generates exactly --gen-tokens (min = max, so every run does the same work), and records time,
tokens/s, PyTorch peak memory, the process's GPU memory from the driver (NVML) and its system RAM (offloaded weights
show up there). A batch that runs out of memory ends that configuration's sweep: that is the batch limit.

--budget-gb emulates the memory left next to Isaac Sim: after the model has loaded (the real system also loads the
model before the simulator starts), PyTorch is capped at that many GB, so a batch whose KV cache does not fit fails
exactly as it would beside the simulator.

    python -m gpumon throughput                                   # default configurations and batch sizes
    python -m gpumon throughput --budget-gb 6.8 --batches 1 2 4 8 16 32
    python -m gpumon throughput --report out/gpumon/throughput/<run>   # redo summary + figure

Output (out/gpumon/throughput/<date>_<time>/): results.csv (one row per batch x repeat), summary.txt,
throughput.png, meta.json, <config>.log (each child's console).
"""

from __future__ import annotations

import csv
import json
import os
import statistics as st
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 ** 2
GIB = 1024 ** 3

# label -> (precision, embedding table in CPU RAM). The baseline is everything on the GPU (FlexGen's "no offload").
CONFIGS = {
    "4bit_gpu": ("4bit", False),       # baseline: 4-bit, all weights on the GPU
    "4bit_offload": ("4bit", True),    # FlexGen placement: lookup-only embedding table in CPU RAM
    "8bit_offload": ("8bit", True),    # precision: 8-bit instead of 4-bit
    "16bit_offload": ("16bit", True),  # precision: no compression
}
DEFAULT_CONFIGS = ["4bit_gpu", "4bit_offload", "8bit_offload"]
DEFAULT_BATCHES = [1, 2, 4, 8, 16]
COLUMNS = ["config", "precision", "offload", "batch", "repeat", "status", "prompt_tokens", "gen_tokens", "seconds",
           "prefill_s", "tokens_per_s", "decode_tokens_per_s", "weights_gib", "peak_alloc_gib", "proc_gpu_peak_gib",
           "device_used_peak_gib", "rss_gib", "budget_gib"]
FILLER = ("The robot looks at the table, finds the red cube next to the green tray, plans a grasp from above, "
          "checks that the gripper is open, moves to the cube and lifts it. ")


# ------------------------------------------------------------------------------------------------ parent
def run(configs: list[str], batches: list[int], prompt_tokens: int, gen_tokens: int, repeats: int,
        model: str, budget_gb: float | None, out: Path) -> int:
    """Run each configuration in its own process, then write the summary and figure."""
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        print(f"[throughput] unknown configuration(s) {unknown}; choose from {list(CONFIGS)}")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    meta = {"configs": configs, "batches": batches, "prompt_tokens": prompt_tokens, "gen_tokens": gen_tokens,
            "repeats": repeats, "model": model, "budget_gb": budget_gb, "start": time.time()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    for label in configs:
        cmd = [sys.executable, "-m", "gpumon", "throughput", "--child", label, "--out", str(out),
               "--batches", *map(str, batches), "--prompt-tokens", str(prompt_tokens),
               "--gen-tokens", str(gen_tokens), "--repeats", str(repeats), "--model", model]
        if budget_gb:
            cmd += ["--budget-gb", str(budget_gb)]
        print(f"[throughput] {label}: loading {model} ...", flush=True)
        with open(out / f"{label}.log", "w") as log:
            code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if code != 0 and not (out / f"{label}.csv").exists():  # died before writing anything: a load failure
            _write_rows(out / f"{label}.csv", [_row(label, batch=0, status="load_failed")])
        print(f"[throughput] {label}: done (exit {code})", flush=True)
    meta["end"] = time.time()
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return 0 if report(out) else 1


# ------------------------------------------------------------------------------------------------- child
def child(label: str, batches: list[int], prompt_tokens: int, gen_tokens: int, repeats: int, model: str,
          budget_gb: float | None, out: Path) -> int:
    """One configuration: load once, sweep the batch sizes, write <label>.csv."""
    import psutil
    import torch

    from vlm.config import VLMConfig
    from vlm.model import VisionLanguageModel

    precision, offload = CONFIGS[label]
    rows: list[dict] = []
    path = out / f"{label}.csv"
    probe = _GpuProbe(os.getpid())
    proc = psutil.Process()
    try:
        vlm = VisionLanguageModel(VLMConfig(model_path=model, quantization=precision,
                                            per_layer_embeddings_on_cpu=offload, thinking=False))
    except RuntimeError as e:  # torch.OutOfMemoryError is a RuntimeError; bitsandbytes raises plain ones
        if not _is_oom(e):
            raise
        print(f"[child] {label}: load ran out of GPU memory: {e}")
        _write_rows(path, [_row(label, batch=0, status="load_oom")])
        return 0
    weights = torch.cuda.memory_allocated() / GIB
    print(f"[child] {label}: loaded in {vlm.load_seconds:.0f} s, weights {weights:.2f} GiB on the GPU")
    if budget_gb:  # cap PyTorch at the budget from here on (the simulator would start now)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, budget_gb * GIB / total))

    ids = _prompt(vlm.processor.tokenizer, prompt_tokens).to(vlm.model.device)
    gen = lambda b, n: vlm.model.generate(input_ids=ids.repeat(b, 1), attention_mask=torch.ones_like(ids).repeat(b, 1),
                                          max_new_tokens=n, min_new_tokens=n, do_sample=False)  # noqa: E731
    with torch.inference_mode():
        try:
            gen(1, 4)  # warm-up: kernels, allocator
        except RuntimeError as e:  # under --budget-gb the weights alone may leave no room for a single request
            if not _is_oom(e):
                raise
            print(f"[child] {label}: no room for even one request (weights {weights:.2f} GiB, budget {budget_gb} GB)")
            _write_rows(path, [_row(label, batch=batches[0], status="oom", weights_gib=round(weights, 3),
                                    budget_gib=budget_gb, prompt_tokens=ids.shape[-1], gen_tokens=gen_tokens)])
            return 0
        for b in batches:
            failed = False
            for rep in range(1, repeats + 1):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                probe.start()
                try:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    gen(b, 1)  # prefill only (one new token)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    gen(b, gen_tokens)
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                except RuntimeError as e:
                    probe.stop()
                    if not _is_oom(e):
                        raise
                    rows.append(_row(label, batch=b, repeat=rep, status="oom", weights_gib=weights,
                                     peak_alloc_gib=torch.cuda.max_memory_allocated() / GIB, budget_gib=budget_gb,
                                     prompt_tokens=ids.shape[-1], gen_tokens=gen_tokens, **probe.peaks()))
                    print(f"[child] {label}: batch {b} ran out of GPU memory")
                    failed = True
                    break
                probe.stop()
                secs, prefill = t2 - t1, t1 - t0
                tokens = b * gen_tokens
                rows.append(_row(
                    label, batch=b, repeat=rep, status="ok", prompt_tokens=ids.shape[-1], gen_tokens=gen_tokens,
                    seconds=round(secs, 3), prefill_s=round(prefill, 3), tokens_per_s=round(tokens / secs, 2),
                    decode_tokens_per_s=round(b * (gen_tokens - 1) / max(secs - prefill, 1e-6), 2),
                    weights_gib=round(weights, 3), peak_alloc_gib=round(torch.cuda.max_memory_allocated() / GIB, 3),
                    rss_gib=round(proc.memory_info().rss / GIB, 2), budget_gib=budget_gb, **probe.peaks()))
                print(f"[child] {label}: batch {b} rep {rep}: {tokens / secs:.1f} tokens/s, "
                      f"peak {torch.cuda.max_memory_allocated() / GIB:.2f} GiB")
            _write_rows(path, rows)
            if failed:
                break
    _write_rows(path, rows)
    return 0


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def _prompt(tokenizer, n: int):
    """A fixed prompt of exactly n tokens (synthetic, like FlexGen's benchmark prompts)."""
    import torch
    text = FILLER
    while len(tokenizer(text)["input_ids"]) < n:
        text += FILLER
    return torch.tensor([tokenizer(text)["input_ids"][:n]])


class _GpuProbe:
    """Peak GPU memory of this process (driver view, NVML) and of the whole device, sampled at 20 Hz."""

    def __init__(self, pid: int) -> None:
        import pynvml
        self.nv, self.pid = pynvml, pid
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self.proc_peak = self.dev_peak = 0.0

    def _sample(self) -> None:
        used = self.nv.nvmlDeviceGetMemoryInfo(self.h).used
        mine = sum((p.usedGpuMemory or 0) for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.h)
                   if p.pid == self.pid)
        self.dev_peak, self.proc_peak = max(self.dev_peak, used), max(self.proc_peak, mine)

    def start(self) -> None:
        self.proc_peak = self.dev_peak = 0.0
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self._sample()
                except self.nv.NVMLError:
                    pass
                self._stop.wait(0.05)
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join()

    def peaks(self) -> dict:
        return {"proc_gpu_peak_gib": round(self.proc_peak / GIB, 3), "device_used_peak_gib": round(self.dev_peak / GIB, 3)}


def _row(label: str, **kv) -> dict:
    precision, offload = CONFIGS[label]
    row = dict.fromkeys(COLUMNS, "")
    row.update(config=label, precision=precision, offload=int(offload), repeat=1)
    row.update({k: ("" if v is None else v) for k, v in kv.items()})
    return row


def _write_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------------------------------ report
def summarise(rows: list[dict]) -> list[dict]:
    """One line per configuration x batch: mean tokens/s over the repeats, memory, and the batch limit."""
    out: dict[tuple, dict] = {}
    for r in rows:
        key = (r["config"], int(r["batch"]))
        s = out.setdefault(key, {"config": r["config"], "batch": int(r["batch"]), "status": r["status"], "tps": [],
                                 "peak": [], "proc": [], "weights": r.get("weights_gib", ""), "rss": r.get("rss_gib", "")})
        if r["status"] == "ok":
            s["tps"].append(float(r["tokens_per_s"]))
            s["peak"].append(float(r["peak_alloc_gib"]))
            if r.get("proc_gpu_peak_gib") not in ("", None):
                s["proc"].append(float(r["proc_gpu_peak_gib"]))
        else:
            s["status"] = r["status"]
    lines = []
    for s in out.values():
        ok = bool(s["tps"])
        lines.append({"config": s["config"], "batch": s["batch"], "status": "ok" if ok and s["status"] == "ok"
                      else s["status"], "tokens_per_s": round(st.mean(s["tps"]), 2) if ok else None,
                      "tokens_per_s_sd": round(st.stdev(s["tps"]), 2) if len(s["tps"]) > 1 else 0.0 if ok else None,
                      "peak_alloc_gib": round(max(s["peak"]), 2) if ok else None,
                      "proc_gpu_peak_gib": round(max(s["proc"]), 2) if s["proc"] else None,
                      "tokens_per_s_per_gib": round(st.mean(s["tps"]) / max(s["proc"] or s["peak"]), 2) if ok else None,
                      "weights_gib": s["weights"], "rss_gib": s["rss"]})
    return sorted(lines, key=lambda l: (l["config"], l["batch"]))


def report(out: Path, figures: bool = True) -> bool:
    rows: list[dict] = []
    for p in sorted(out.glob("*.csv")):
        if p.name == "results.csv":
            continue
        with open(p, newline="") as f:
            rows += list(csv.DictReader(f))
    if not rows:
        print(f"[throughput] no results in {out}")
        return False
    _write_rows(out / "results.csv", rows)
    meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    lines = summarise(rows)
    txt = [f"throughput  {out}",
           f"model {meta.get('model', '?')}  prompt {meta.get('prompt_tokens', '?')} tokens  "
           f"generate {meta.get('gen_tokens', '?')} tokens  repeats {meta.get('repeats', '?')}  "
           f"budget {meta.get('budget_gb') or 'whole GPU'} GB", "",
           f"{'config':<15}{'batch':>6}  {'tokens/s':>12}  {'peak GiB':>9}  {'GPU GiB':>8}  {'tok/s/GiB':>9}  status"]
    for l in lines:
        tps = f"{l['tokens_per_s']:.1f} ± {l['tokens_per_s_sd']:.1f}" if l["tokens_per_s"] is not None else "-"
        txt.append(f"{l['config']:<15}{l['batch']:>6}  {tps:>12}  {_f(l['peak_alloc_gib']):>9}  "
                   f"{_f(l['proc_gpu_peak_gib']):>8}  {_f(l['tokens_per_s_per_gib']):>9}  {l['status']}")
    txt += ["", "largest batch that ran, and its throughput:"]
    for cfg in dict.fromkeys(l["config"] for l in lines):
        ok = [l for l in lines if l["config"] == cfg and l["status"] == "ok"]
        best = max(ok, key=lambda l: l["batch"]) if ok else None
        txt.append(f"  {cfg:<15} " + (f"batch {best['batch']}: {best['tokens_per_s']:.1f} tokens/s" if best
                                       else "did not run (" + next(l["status"] for l in lines if l["config"] == cfg) + ")"))
    (out / "summary.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))
    if figures:
        _figure(out, lines, meta)
    return True


def _f(v) -> str:
    return "-" if v is None else f"{v:.2f}"


def _figure(out: Path, lines: list[dict], meta: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"4bit_gpu": "#a3a29b", "4bit_offload": "#2a78d6", "8bit_offload": "#eb6834", "16bit_offload": "#1baf7a"}
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(12, 4.6))
    for cfg in dict.fromkeys(l["config"] for l in lines):
        ok = [l for l in lines if l["config"] == cfg and l["status"] == "ok"]
        bad = [l for l in lines if l["config"] == cfg and l["status"] != "ok" and l["batch"] > 0]
        c = colours.get(cfg, "#555")
        if ok:
            ax_t.plot([l["batch"] for l in ok], [l["tokens_per_s"] for l in ok], "o-", color=c, label=cfg)
            ax_m.plot([l["batch"] for l in ok], [l["proc_gpu_peak_gib"] or l["peak_alloc_gib"] for l in ok], "o-",
                      color=c, label=cfg)
        for l in bad:  # the batch that ran out of memory
            ax_t.plot([l["batch"]], [0], "x", color=c, markersize=10, mew=2)
    ticks = sorted({l["batch"] for l in lines if l["batch"] > 0})
    for ax in (ax_t, ax_m):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("batch size (requests decoded together)")
        ax.grid(alpha=0.3)
    ax_t.set_ylabel("throughput (generated tokens / s)")
    ax_t.set_title("Throughput (x on the axis = out of GPU memory)")
    ax_m.set_ylabel("GPU memory of the process, peak (GiB)")
    ax_m.set_title("GPU memory")
    if meta.get("budget_gb"):
        ax_m.axhline(meta["budget_gb"], ls="--", color="k", lw=1)
        ax_m.text(1, meta["budget_gb"], f" budget {meta['budget_gb']} GB", va="bottom", fontsize=8)
    ax_t.legend(frameon=False)
    fig.suptitle(f"Gemma alone: prompt {meta.get('prompt_tokens', '?')} tokens, "
                 f"{meta.get('gen_tokens', '?')} generated tokens per request", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "throughput.png", dpi=130)
    plt.close(fig)
