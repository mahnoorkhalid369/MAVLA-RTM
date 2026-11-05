"""Recording sessions and the benchmark runner.

A run folder holds:
    meta.json       device, configuration, environment overrides, git commit, start/end time
    samples.csv     the sampler (gpumon/sampler.py)
    events.jsonl    the agent's events, with arrival times (gpumon/events.py)
    markers.jsonl   lifecycle markers with Unix times (models loading, simulator start, ready, instruction start/end,
                    out-of-memory lines); from the agent's console in bench mode, start/stop only in record mode
    console.log     bench mode: the agent's console, each line prefixed with its Unix time
    torch/          torch_<pid>.csv from the opt-in hook (gpumon/torch_hook)
    agent/          bench mode: the agent's own output folder (--out), with the per-instruction traces
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import psutil

from gpumon import procs
from gpumon.events import EventFollower
from gpumon.sampler import Sampler

ROOT = Path(__file__).resolve().parents[1]
HOOK_DIR = Path(__file__).resolve().parent / "torch_hook"
OOM = re.compile(r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation|"
                 r"out of memory", re.I)
# console line -> marker (searched, not matched: the agent's prompt has no newline, so output can follow it)
CONSOLE_MARKERS = (
    (re.compile(r"\[agent\] loading Gemma"), "gemma_load_start"),
    (re.compile(r"\[agent\] loading the detector"), "detector_load_start"),
    (re.compile(r"\[agent\] models loaded"), "models_loaded"),
    (re.compile(r"\[agent\] waiting for the simulator"), "sim_wait_start"),
    (re.compile(r"\[agent\] ready"), "ready"),
)
FINAL = re.compile(r"\[agent\] ([A-Z_]+): (.*)")


class Recording:
    """Sampler + event follower + marker file for one run folder."""

    def __init__(self, out: Path, events_roots: list[Path], hz: float, meta: dict) -> None:
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.markers = open(self.out / "markers.jsonl", "a", buffering=1)
        self.sampler = Sampler(self.out, hz=hz)
        self.follower = EventFollower(events_roots, self.out)
        self.meta = {"device": self.sampler.device, "total_mib": round(self.sampler.total_mib), "hz": hz,
                     "git": _git(), "argv": sys.argv, "start": time.time(), **meta}

    def mark(self, name: str, **fields) -> None:
        self.markers.write(json.dumps({"time": round(time.time(), 3), "marker": name, **fields}) + "\n")

    def __enter__(self) -> "Recording":
        self.sampler.start()
        self.follower.start()
        self.mark("recording_start")
        self._write_meta()
        return self

    def __exit__(self, *exc) -> None:
        self.mark("recording_stop")
        self.follower.stop()
        self.sampler.stop()
        self.markers.close()
        self.meta["end"] = time.time()
        self._write_meta()

    def _write_meta(self) -> None:
        (self.out / "meta.json").write_text(json.dumps(self.meta, indent=2) + "\n")


def _git() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
        return rev.strip() + ("+dirty" if dirty.strip() else "")
    except OSError:
        return "unknown"


# ------------------------------------------------------------------------------------------------ record mode
def record(out: Path, events_roots: list[Path], hz: float, label: str, duration: float | None,
           until_idle: bool) -> int:
    """Sample until Ctrl+C, --duration, or (until_idle) the agent and simulator have appeared and both exited."""
    seen = False

    def on_term(*_):  # launch/agent.sh stops a background recorder with SIGTERM (bash jobs ignore SIGINT)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    with Recording(out, events_roots, hz, {"mode": "record", "label": label}) as rec:
        print(f"[gpumon] recording {rec.sampler.device} at {hz:g} Hz -> {out}  (Ctrl+C to stop)", flush=True)
        t0 = time.monotonic()
        try:
            while duration is None or time.monotonic() - t0 < duration:
                time.sleep(1.0)
                if until_idle:
                    alive = procs.find("agent") or procs.find("sim")
                    seen = seen or bool(alive)
                    if seen and not alive:
                        rec.mark("processes_exited")
                        time.sleep(3.0)  # record the memory being released
                        break
        except KeyboardInterrupt:
            pass
    print(f"[gpumon] {rec.sampler.rows} samples, {rec.follower.count} events, peak {rec.sampler.peak_used_mib:.0f} MiB",
          flush=True)
    return 0


# ------------------------------------------------------------------------------------------------- bench mode
def load_bench(path: Path) -> dict:
    cfg = tomllib.loads(Path(path).read_text())
    for key in ("name", "instructions", "config"):
        if key not in cfg:
            raise SystemExit(f"{path}: missing '{key}'")
    labels = [c["label"] for c in cfg["config"]]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"{path}: configuration labels must be unique")
    return cfg


def bench(path: Path, only: list[str] | None = None, repeats: int | None = None, dry_run: bool = False) -> int:
    cfg = load_bench(path)
    base = ROOT / cfg.get("out", "out/gpumon/bench") / cfg["name"]
    n = repeats or int(cfg.get("repeats", 1))
    plan = [(c, r) for r in range(1, n + 1) for c in cfg["config"] if not only or c["label"] in only]
    print(f"[gpumon] bench {cfg['name']}: {len(plan)} runs -> {base}", flush=True)
    if dry_run:
        for c, r in plan:
            print(f"  {c['label']} r{r:02d}: {' '.join(_command(cfg, c, base / c['label'] / f'r{r:02d}'))}  "
                  f"env {_env_overrides(cfg, c)}")
        return 0
    failures = 0
    for i, (c, r) in enumerate(plan, 1):  # repeats interleaved (all configs, then again): drift hits all alike
        run_dir = base / c["label"] / f"r{r:02d}"
        if (run_dir / "meta.json").exists() and json.loads((run_dir / "meta.json").read_text()).get("end"):
            print(f"[gpumon] [{i}/{len(plan)}] {c['label']} r{r:02d}: already done, skipped", flush=True)
            continue
        _wait_gpu_quiet(float(cfg.get("cooldown_s", 20)))
        print(f"[gpumon] [{i}/{len(plan)}] {c['label']} r{r:02d}", flush=True)
        result = run_one(cfg, c, run_dir)
        failures += result["exit"] != 0
        print(f"[gpumon]   exit {result['exit']}, finals {result['finals']}, ooms {result['ooms']}, "
              f"{result['seconds']:.0f} s", flush=True)
        from gpumon.report import report
        report(run_dir, figures=bool(cfg.get("figures_per_run", True)))
    from gpumon.report import compare
    compare(base, [c["label"] for c in cfg["config"]])
    return 1 if failures else 0


def _env_overrides(cfg: dict, c: dict) -> dict:
    return {k: str(v) for k, v in {**cfg.get("env", {}), **c.get("env", {})}.items()}


def _command(cfg: dict, c: dict, run_dir: Path) -> list[str]:
    launcher = cfg.get("launcher", "launch/agent.sh")
    args = list(cfg.get("launcher_args", [])) + list(c.get("launcher_args", []))
    if c.get("gemma"):
        args += ["--gemma", str(ROOT / c["gemma"])]
    return [str(ROOT / launcher), *args, "--out", str(run_dir / "agent"), "--live-dir", str(run_dir / "live")]


def run_one(cfg: dict, c: dict, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    # no windows, and no second recorder: launch/agent.sh's own GPU_MONITOR would sample the same GPU
    env = {**os.environ, "DEMO_VIEWER": "0", "CAMERA_WINDOWS": "0", "DASHBOARD": "0", "GPU_MONITOR": "0",
           **_env_overrides(cfg, c)}
    if cfg.get("torch_hook", True):
        env["PYTHONPATH"] = f"{HOOK_DIR}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(HOOK_DIR)
        env["GPUMON_TORCH_DIR"] = str(run_dir / "torch")
    cmd = _command(cfg, c, run_dir)
    todo = list(cfg["instructions"])
    timeout = float(cfg.get("timeout_s", 1800))
    meta = {"mode": "bench", "bench": cfg["name"], "label": c["label"], "config": c, "command": cmd,
            "env_overrides": _env_overrides(cfg, c), "instructions": todo, "torch_hook": cfg.get("torch_hook", True)}
    finals, ooms, started = [], 0, time.time()
    with Recording(run_dir, [run_dir / "agent"], float(cfg.get("hz", 10)), meta) as rec:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        rec.mark("launch", pid=p.pid)

        def send(line: str) -> None:
            try:
                p.stdin.write(line + "\n")
                p.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass

        def on_timeout() -> None:  # a silent hang never reaches the line loop, so the timer interrupts the agent
            rec.mark("timeout")
            try:
                os.killpg(p.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

        watchdog = threading.Timer(timeout, on_timeout)
        watchdog.daemon = True
        watchdog.start()
        with open(run_dir / "console.log", "w", buffering=1) as log:
            for line in p.stdout:  # the line loop ends when the agent closes its stdout (exit)
                now = time.time()
                log.write(f"{now:.3f}\t{line}")
                for rx, name in CONSOLE_MARKERS:
                    if rx.search(line):
                        rec.mark(name)
                if OOM.search(line):
                    ooms += 1
                    rec.mark("oom", line=line.strip()[:300])
                if "[agent] ready" in line and todo:
                    rec.mark("instruction_start", index=1, text=todo[0])
                    send(todo.pop(0))
                m = FINAL.search(line)
                if m:
                    finals.append(m[1])
                    rec.mark("instruction_end", index=len(finals), status=m[1], message=m[2][:300])
                    if todo:
                        rec.mark("instruction_start", index=len(finals) + 1, text=todo[0])
                    send(todo.pop(0) if todo else "")  # next instruction, or the empty line that quits
        watchdog.cancel()
        _stop(p, started, rec)
        rec.meta.update({"exit": p.returncode, "finals": finals, "ooms": ooms})
    return {"exit": p.returncode, "finals": finals, "ooms": ooms, "seconds": time.time() - started}


def _stop(p: subprocess.Popen, started: float, rec: Recording) -> None:
    """End the agent: normal exit, else SIGINT (agent.run then stops Isaac Sim), else SIGKILL. Isaac Sim runs in its
    own session, so a simulator this run started and left behind is stopped too (never one started before it)."""
    try:
        p.wait(120)
    except subprocess.TimeoutExpired:
        for sig, wait in ((signal.SIGINT, 90), (signal.SIGKILL, 10)):
            rec.mark("kill", signal=sig.name)
            try:
                os.killpg(p.pid, sig)
                p.wait(wait)
                break
            except (ProcessLookupError, subprocess.TimeoutExpired):
                continue
    for s in procs.find("sim"):
        try:
            if s.create_time() >= started:
                rec.mark("orphan_sim_killed", pid=s.pid)
                os.killpg(os.getpgid(s.pid), signal.SIGINT)
                s.wait(60)
        except (psutil.NoSuchProcess, ProcessLookupError, psutil.TimeoutExpired):
            pass
    rec.mark("agent_exit", code=p.returncode)
    time.sleep(3.0)  # keep sampling while the memory is released


def _wait_gpu_quiet(cooldown: float) -> None:
    """Between runs: no simulator or agent left, then a short cooldown so the next run starts from idle memory."""
    end = time.monotonic() + 300
    while (procs.find("sim") or procs.find("agent")) and time.monotonic() < end:
        print("[gpumon] waiting for a running simulator / agent to exit ...", flush=True)
        time.sleep(10)
    if procs.find("sim") or procs.find("agent"):
        raise SystemExit("[gpumon] a simulator or agent is still running; stop it before benchmarking")
    time.sleep(cooldown)
