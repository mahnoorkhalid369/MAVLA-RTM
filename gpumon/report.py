"""Turn a run folder into summary.json / summary.txt / timeline.png, and a bench folder into a comparison.

Phases (all in Unix time):
    idle       before the agent was launched (bench) or the first 2 s (record)
    load       'gemma_load_start' -> 'models_loaded'   (Gemma, then YOLO)
    sim_start  'models_loaded' -> 'ready'               (Isaac Sim starts, ROS 2 connects)
    working    'ready' -> agent exit                    (instructions; VLM calls are sub-intervals)
    vlm call   end = aligned time of the 'vlm' event, start = end - its 'seconds'
"""

from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

GROUPS = ("sim", "agent", "other", "unattributed")
# categorical slots 1-3 of the reference palette (light mode), fixed order; driver/unattributed is neutral gray
COLORS = {"sim": "#2a78d6", "agent": "#eb6834", "other": "#1baf7a", "unattributed": "#a3a29b"}
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7"
LABELS = {"sim": "Isaac Sim", "agent": "Agent (Gemma + YOLO)", "other": "Desktop / other",
          "unattributed": "Driver / unattributed"}


# --------------------------------------------------------------------------------------------------- loading
def load_samples(run: Path) -> dict[str, np.ndarray]:
    rows = [r for r in csv.DictReader(line for line in open(run / "samples.csv") if not line.startswith("#"))]
    if not rows:
        raise SystemExit(f"{run}: no samples")
    cols = rows[0].keys()
    return {c: np.array([float(r[c]) if r[c] not in ("", None) else np.nan for r in rows]) for c in cols}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def align(events: list[dict]) -> list[dict]:
    """Add 'time' (Unix) to every event: per trace file, offset = min(arrival - t) (see gpumon/events.py)."""
    offset: dict[str, float] = {}
    for e in events:
        if "t" in e and "arrival" in e:
            d = e["arrival"] - e["t"]
            offset[e["file"]] = min(offset.get(e["file"], d), d)
    for e in events:
        if e.get("file") in offset and "t" in e:
            e["time"] = e["t"] + offset[e["file"]]
    return [e for e in events if "time" in e]


def load_torch(run: Path) -> dict[int, dict[str, np.ndarray]]:
    out = {}
    for f in sorted((run / "torch").glob("torch_*.csv")):
        rows = list(csv.DictReader(line for line in open(f) if not line.startswith("#")))
        if rows:
            out[int(f.stem.split("_")[1])] = {c: np.array([float(r[c]) for r in rows]) for c in rows[0]}
    return out


# --------------------------------------------------------------------------------------------------- metrics
def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.array(xs)
    return {"n": len(xs), "mean": round(float(a.mean()), 3), "std": round(float(a.std(ddof=1)), 3) if len(xs) > 1 else 0.0,
            "p50": round(float(np.percentile(a, 50)), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "max": round(float(a.max()), 3)}


def _window(s: dict, t0: float | None, t1: float | None) -> np.ndarray:
    t = s["time"]
    m = np.ones_like(t, dtype=bool)
    if t0 is not None:
        m &= t >= t0
    if t1 is not None:
        m &= t <= t1
    return m


def _peak(s: dict, mask: np.ndarray, col: str = "used_mib") -> dict | None:
    if not mask.any():
        return None
    i = np.flatnonzero(mask)[np.nanargmax(s[col][mask])]
    return {"mib": round(float(s[col][i]), 1), "time": round(float(s["time"][i]), 3)}


def summarize(run: Path) -> dict:
    run = Path(run)
    meta = json.loads((run / "meta.json").read_text())
    s = load_samples(run)
    markers = load_jsonl(run / "markers.jsonl")
    mk: dict[str, list[dict]] = defaultdict(list)
    for m in markers:
        mk[m["marker"]].append(m)
    first = lambda name: mk[name][0]["time"] if mk.get(name) else None  # noqa: E731
    events = align(load_jsonl(run / "events.jsonl"))
    total = float(np.nanmax(s["total_mib"]))

    launch, gload, loaded, ready = first("launch"), first("gemma_load_start"), first("models_loaded"), first("ready")
    end = first("agent_exit") or float(s["time"][-1])
    idle_mask = _window(s, None, launch) if launch else _window(s, None, s["time"][0] + 2.0)
    phases = {"load": (gload, loaded), "sim_start": (loaded, ready), "working": (ready, end)}

    out: dict = {"run": str(run), "label": meta.get("label"), "mode": meta.get("mode"), "device": meta.get("device"),
                 "total_mib": round(total), "git": meta.get("git"), "env_overrides": meta.get("env_overrides", {}),
                 "duration_s": round(float(s["time"][-1] - s["time"][0]), 1), "samples": int(len(s["time"]))}
    out["idle_mib"] = round(float(np.nanmedian(s["used_mib"][idle_mask])), 1) if idle_mask.any() else None
    out["peak"] = _peak(s, np.ones_like(s["time"], dtype=bool))
    out["min_free_mib"] = round(float(np.nanmin(s["free_mib"])), 1)
    out["phases"] = {}
    for name, (a, b) in phases.items():
        if a is None or b is None:
            continue
        m = _window(s, a, b)
        out["phases"][name] = {"seconds": round(b - a, 2), "peak": _peak(s, m),
                               "peak_sim_mib": round(float(np.nanmax(s["gpu_sim_mib"][m])), 1) if m.any() else None,
                               "peak_agent_mib": round(float(np.nanmax(s["gpu_agent_mib"][m])), 1) if m.any() else None,
                               "min_free_mib": round(float(np.nanmin(s["free_mib"][m])), 1) if m.any() else None}
    if ready:  # steady state: everything loaded and running (excludes the loading spike)
        m = _window(s, ready, end)
        if m.any():
            out["steady"] = {"median_used_mib": round(float(np.nanmedian(s["used_mib"][m])), 1),
                             "sim_mib": round(float(np.nanmedian(s["gpu_sim_mib"][m])), 1),
                             "agent_mib": round(float(np.nanmedian(s["gpu_agent_mib"][m])), 1),
                             "agent_rss_mib": round(float(np.nanmedian(s["rss_agent_mib"][m])), 1),
                             "sim_rss_mib": round(float(np.nanmedian(s["rss_sim_mib"][m])), 1)}
    out["peak_rss_agent_mib"] = round(float(np.nanmax(s["rss_agent_mib"])), 1)
    out["peak_rss_sim_mib"] = round(float(np.nanmax(s["rss_sim_mib"])), 1)
    out["min_ram_available_mib"] = round(float(np.nanmin(s["ram_available_mib"])), 1)

    # VLM calls
    calls = [{"role": e.get("role"), "start": e["time"] - float(e.get("seconds") or 0), "end": e["time"],
              "seconds": float(e.get("seconds") or 0), "tokens": e.get("tokens"), "valid": e.get("valid")}
             for e in events if e.get("kind") == "vlm" and e.get("seconds") is not None]
    by_role: dict[str, list[float]] = defaultdict(list)
    for c in calls:
        by_role[c["role"]].append(c["seconds"])
    toks = [c["tokens"] / c["seconds"] for c in calls if isinstance(c.get("tokens"), (int, float)) and c["seconds"] > 0]
    in_call = np.zeros_like(s["time"], dtype=bool)
    for c in calls:
        in_call |= _window(s, c["start"], c["end"])
    after = _window(s, ready, end) if ready else np.ones_like(in_call)
    out["vlm"] = {"latency_s": _stats([c["seconds"] for c in calls]),
                  "by_role": {r: _stats(v) for r, v in sorted(by_role.items())},
                  "tokens_per_s": _stats(toks),
                  "invalid_answers": sum(1 for c in calls if c.get("valid") is False),
                  "peak_during_calls": _peak(s, in_call & after),
                  "peak_between_calls": _peak(s, ~in_call & after)}
    for col in ("pcie_tx_mbs", "pcie_rx_mbs", "gpu_util"):
        if col in s and np.isfinite(s[col]).any():
            a, b = s[col][in_call & after], s[col][~in_call & after]
            out["vlm"][f"{col}_during"] = round(float(np.nanmean(a)), 1) if a.size else None
            out["vlm"][f"{col}_between"] = round(float(np.nanmean(b)), 1) if b.size else None

    # PyTorch allocator (opt-in hook)
    torch = load_torch(run)
    out["torch"] = {str(pid): {"peak_allocated_mib": round(float(d["peak_allocated_mib"].max()), 1),
                               "peak_reserved_mib": round(float(d["peak_reserved_mib"].max()), 1),
                               "steady_allocated_mib": round(float(np.median(d["allocated_mib"][d["time"] >= (ready or 0)]))
                                                             if (d["time"] >= (ready or 0)).any() else 0.0, 1),
                               "steady_reserved_mib": round(float(np.median(d["reserved_mib"][d["time"] >= (ready or 0)]))
                                                            if (d["time"] >= (ready or 0)).any() else 0.0, 1),
                               "num_ooms": int(d["num_ooms"].max())} for pid, d in torch.items()}

    # outcomes
    finals = [m["status"] for m in mk.get("instruction_end", [])]
    starts = {m["index"]: m["time"] for m in mk.get("instruction_start", [])}
    durations = [m["time"] - starts[m["index"]] for m in mk.get("instruction_end", []) if m["index"] in starts]
    reports = [e.get("status") for e in events if e.get("kind") == "report"]
    out["tasks"] = {"instructions": len(meta.get("instructions", [])) or len(reports), "finals": finals or reports,
                    "completed": sum(1 for f in (finals or reports) if f == "COMPLETED"),
                    "duration_s": _stats(durations)}
    out["feasibility"] = {"models_loaded": loaded is not None if meta.get("mode") == "bench" else None,
                          "ready": ready is not None if meta.get("mode") == "bench" else None,
                          "oom_lines": len(mk.get("oom", [])), "timeout": bool(mk.get("timeout")),
                          "exit": meta.get("exit"),
                          "load_s": round(loaded - gload, 2) if loaded and gload else None,
                          "sim_start_s": round(ready - loaded, 2) if ready and loaded else None}
    out["_series"] = {"calls": calls, "phases": {k: v for k, v in phases.items() if None not in v}}
    return out


# --------------------------------------------------------------------------------------------------- output
def report(run: Path, figures: bool = True) -> dict:
    run = Path(run)
    summ = summarize(run)
    series = summ.pop("_series")
    (run / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    (run / "summary.txt").write_text(text(summ) + "\n")
    if figures:
        timeline(run, series)
    print(text(summ))
    return summ


def text(s: dict) -> str:
    gb = lambda mib: "-" if mib is None else f"{mib / 1024:.2f} GB"  # noqa: E731
    L = [f"run      {s['run']}", f"label    {s.get('label')}   device {s['device']} ({gb(s['total_mib'])})   git {s['git']}"]
    if s.get("env_overrides"):
        L.append(f"env      {' '.join(f'{k}={v}' for k, v in s['env_overrides'].items())}")
    f = s["feasibility"]
    L.append(f"feasible models_loaded={f['models_loaded']} ready={f['ready']} oom_lines={f['oom_lines']} "
             f"timeout={f['timeout']} exit={f['exit']}   load {f['load_s']} s, sim start {f['sim_start_s']} s")
    L.append(f"memory   idle {gb(s['idle_mib'])} | peak {gb(s['peak']['mib'])} | min free {gb(s['min_free_mib'])}")
    for name, p in s["phases"].items():
        L.append(f"  {name:<10} {p['seconds']:>7.1f} s  peak {gb(p['peak']['mib'] if p['peak'] else None)}  "
                 f"(sim {gb(p['peak_sim_mib'])}, agent {gb(p['peak_agent_mib'])}), min free {gb(p['min_free_mib'])}")
    if "steady" in s:
        st_ = s["steady"]
        L.append(f"steady   used {gb(st_['median_used_mib'])} = sim {gb(st_['sim_mib'])} + agent {gb(st_['agent_mib'])} "
                 f"+ rest;  RAM: agent {gb(st_['agent_rss_mib'])}, sim {gb(st_['sim_rss_mib'])}")
    for pid, t in s["torch"].items():
        L.append(f"torch    pid {pid}: allocated {gb(t['steady_allocated_mib'])} / reserved {gb(t['steady_reserved_mib'])} "
                 f"(peaks {gb(t['peak_allocated_mib'])} / {gb(t['peak_reserved_mib'])}), ooms {t['num_ooms']}")
    v = s["vlm"]
    lat = v["latency_s"]
    if lat["n"]:
        L.append(f"vlm      {lat['n']} calls: {lat['mean']:.2f} ± {lat['std']:.2f} s (p50 {lat['p50']:.2f}, "
                 f"p95 {lat['p95']:.2f}, max {lat['max']:.2f}); invalid {v['invalid_answers']}")
        for r, x in v["by_role"].items():
            L.append(f"  {r:<22} n={x['n']:<3} mean {x['mean']:.2f} s  p95 {x['p95']:.2f} s")
        L.append(f"  peak during calls {gb((v['peak_during_calls'] or {}).get('mib'))}, between "
                 f"{gb((v['peak_between_calls'] or {}).get('mib'))};  PCIe rx/tx during {v.get('pcie_rx_mbs_during')}"
                 f"/{v.get('pcie_tx_mbs_during')} MB/s, between {v.get('pcie_rx_mbs_between')}/{v.get('pcie_tx_mbs_between')}")
    t = s["tasks"]
    L.append(f"tasks    {t['completed']}/{t['instructions']} completed  {t['finals']}  "
             + (f"duration {t['duration_s']['mean']:.1f} ± {t['duration_s']['std']:.1f} s" if t["duration_s"]["n"] else ""))
    return "\n".join(L)


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def timeline(run: Path, series: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = load_samples(run)
    t = s["time"] - s["time"][0]
    t0 = s["time"][0]
    fig, (ax, ax_u, ax_p) = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.12})
    layers = [s[f"gpu_{g}_mib"] / 1024 for g in GROUPS]
    ax.stackplot(t, *layers, colors=[COLORS[g] for g in GROUPS], labels=[LABELS[g] for g in GROUPS],
                 edgecolor="white", linewidth=0.4)
    total = float(np.nanmax(s["total_mib"])) / 1024
    ax.axhline(total, color=INK, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(t[-1], total, f"capacity {total:.1f} GB", va="bottom", ha="right", fontsize=8, color=INK)
    for axis in (ax, ax_u, ax_p):
        for c in series["calls"]:
            axis.axvspan(c["start"] - t0, c["end"] - t0, color=BAND, zorder=0, linewidth=0)
    for name, (a, b) in series["phases"].items():
        ax.axvline(a - t0, color=INK2, linewidth=0.8)
        ax.text(a - t0, total * 0.97, f" {name.replace('_', ' ')}", fontsize=8, color=INK2, va="top")
    ax.set_ylim(0, total * 1.06)
    ax.set_ylabel("GPU memory (GB)", color=INK2, fontsize=9)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, frameon=False, ncol=4, borderaxespad=0.3)
    ax_u.plot(t, s["gpu_util"], color=COLORS["sim"], linewidth=1.2)
    ax_u.set_ylabel("GPU util (%)", color=INK2, fontsize=9)
    ax_u.set_ylim(0, 105)
    if np.isfinite(s.get("pcie_rx_mbs", np.array([np.nan]))).any():
        ax_p.plot(t, s["pcie_rx_mbs"], color=COLORS["sim"], linewidth=1.2, label="host → GPU (rx)")
        ax_p.plot(t, s["pcie_tx_mbs"], color=COLORS["agent"], linewidth=1.2, label="GPU → host (tx)")
        ax_p.legend(loc="upper right", fontsize=8, frameon=False)
    ax_p.set_ylabel("PCIe (MB/s)", color=INK2, fontsize=9)
    ax_p.set_xlabel("time since recording start (s)   ·   shaded: VLM calls", color=INK2, fontsize=9)
    for a in (ax, ax_u, ax_p):
        _style(a)
    meta = json.loads((run / "meta.json").read_text())
    fig.suptitle(f"GPU memory timeline: {meta.get('label') or run.name}", x=0.06, y=0.97, ha="left", fontsize=11, color=INK)
    fig.savefig(run / "timeline.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# -------------------------------------------------------------------------------------------------- compare
METRICS = (  # (column, path in summary.json, scale)
    ("peak_gb", ("peak", "mib"), 1 / 1024),
    ("min_free_gb", ("min_free_mib",), 1 / 1024),
    ("load_peak_gb", ("phases", "load", "peak", "mib"), 1 / 1024),
    ("steady_used_gb", ("steady", "median_used_mib"), 1 / 1024),
    ("steady_agent_gb", ("steady", "agent_mib"), 1 / 1024),
    ("agent_rss_gb", ("steady", "agent_rss_mib"), 1 / 1024),
    ("load_s", ("feasibility", "load_s"), 1),
    ("vlm_mean_s", ("vlm", "latency_s", "mean"), 1),
    ("vlm_p95_s", ("vlm", "latency_s", "p95"), 1),
    ("tokens_per_s", ("vlm", "tokens_per_s", "mean"), 1),
    ("task_s", ("tasks", "duration_s", "mean"), 1),
)


def _get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return None
        d = d[k]
    return d


def compare(base: Path, order: list[str] | None = None) -> list[dict]:
    """One row per configuration (mean ± sd over its repeats) -> comparison.csv / comparison.txt / comparison.png.
    order: configuration labels in the bench file's order (others follow alphabetically)."""
    base = Path(base)
    rows = []
    rank = {k: i for i, k in enumerate(order or [])}
    for cfg_dir in sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: (rank.get(p.name, len(rank)), p.name)):
        summ = [json.loads(f.read_text()) for f in sorted(cfg_dir.glob("r*/summary.json"))]
        if not summ:
            continue
        row = {"label": cfg_dir.name, "runs": len(summ),
               "loaded": sum(1 for x in summ if x["feasibility"]["models_loaded"]),
               "ooms": sum(x["feasibility"]["oom_lines"] for x in summ),
               "completed": sum(x["tasks"]["completed"] for x in summ),
               "instructions": sum(x["tasks"]["instructions"] for x in summ), "total_gb": summ[0]["total_mib"] / 1024}
        for col, path, scale in METRICS:
            vals = [v * scale for v in (_get(x, path) for x in summ) if v is not None]
            row[col] = round(st.mean(vals), 3) if vals else None
            row[col + "_sd"] = round(st.stdev(vals), 3) if len(vals) > 1 else None
        rows.append(row)
    if not rows:
        return rows
    with open(base / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fmt = lambda r, c: "-" if r[c] is None else (f"{r[c]:.2f}" + (f"±{r[c + '_sd']:.2f}" if r.get(c + "_sd") else ""))  # noqa: E731
    cols = ["peak_gb", "min_free_gb", "load_peak_gb", "steady_used_gb", "agent_rss_gb", "load_s", "vlm_mean_s",
            "vlm_p95_s", "task_s"]
    lines = [f"{'config':<24}{'runs':>5}{'loaded':>7}{'oom':>5}{'tasks':>8}  " + "".join(f"{c:>15}" for c in cols)]
    for r in rows:
        lines.append(f"{r['label']:<24}{r['runs']:>5}{r['loaded']:>7}{r['ooms']:>5}{r['completed']:>4}/{r['instructions']:<3}  "
                     + "".join(f"{fmt(r, c):>15}" for c in cols))
    (base / "comparison.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    _compare_figure(base, rows)
    return rows


def _compare_figure(base: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("peak_gb", "Peak GPU memory (GB)"), ("steady_used_gb", "Steady GPU memory (GB)"),
              ("vlm_mean_s", "VLM latency, mean (s)"), ("task_s", "Instruction time, mean (s)")]
    labels = [r["label"] for r in rows]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 0.45 * len(rows) + 1.6), sharey=True)
    y = np.arange(len(rows))
    for ax, (col, title) in zip(axes, panels):
        vals = [r[col] if r[col] is not None else 0 for r in rows]
        err = [r.get(col + "_sd") or 0 for r in rows]
        ax.barh(y, vals, height=0.6, color=COLORS["sim"], xerr=err, error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
        for yi, v, e, r in zip(y, vals, err, rows):
            ax.text(v + e, yi, f" {v:.2f}" if r[col] is not None else " n/a", va="center", fontsize=8, color=INK)
        if col in ("peak_gb", "steady_used_gb"):
            ax.axvline(rows[0]["total_gb"], color=INK, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].invert_yaxis()
    fig.savefig(base / "comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
