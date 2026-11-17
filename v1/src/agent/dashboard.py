#!/usr/bin/env python3
"""One-window dashboard (optional, read-only): the main camera with YOLO masks, the live agent logs and a large live
GPU monitor (the data source of the ECE 565 measurements, recorded to disk by gpumon meanwhile).

    +-----------------------------+------------------------------+
    | main camera + YOLO masks,   | GPU MONITOR                  |
    | ids, target / destination   |  used / free / peak / util   |
    | ('c': auto / table / wrist) |  memory per process (bar)    |
    +-----------------------------+  memory over time, stacked,  |
    | task | state | target       |   VLM calls shaded           |
    | live agent logs             |  GPU util / PCIe over time   |
    |                             |  RAM, VLM latency per role   |
    +-----------------------------+------------------------------+
    | status bar                                                 |

Like agent/viewer.py (whose drawing it reuses) it runs as its own process with the system Python and ROS 2, never
reads stdin and never writes anything the agent reads. The GPU panel only READS a gpumon recording
(<gpumon-dir>/samples.csv, written by `python -m gpumon record`, started by launch/agent.sh when GPU_MONITOR=True);
without one it says the monitor is off. The Isaac Sim window stays separate (or headless).

    source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=42
    python3 agent/dashboard.py --gpumon-dir out/gpumon/record/<run>
    keys: q close | c main camera (auto / table / wrist) | arrows PgUp PgDn End: logs
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # the repository (this file is in src/<package>/)
sys.path.insert(0, str(ROOT / "src"))
from agent.viewer import (BG, CAND_C, DEST_C, FG, FONT, LEVEL_COLOUR, MUTED, TARGET_C, RunState,  # noqa: E402
                          Viewer)

W, H, STATUS_H = 1920, 1080, 30
MAIN = (0, 0, 1024, 576)                       # x, y, w, h
TASK = (0, 580, 1024, 56)                      # task / state / target strip above the logs
LOGS = (0, 640, 1024, H - STATUS_H - 640)
GPU = (1032, 0, W - 1032, H - STATUS_H)       # the whole right column
LINE = (70, 70, 70)
# gpumon groups, same hues as gpumon/report.py's figures (BGR here)
GROUP_C = {"sim": (214, 120, 42), "agent": (52, 104, 235), "other": (122, 175, 27), "unattributed": (155, 162, 163)}
GROUP_L = {"sim": "Isaac Sim", "agent": "agent", "other": "desktop", "unattributed": "driver"}
HISTORY_S = 300  # seconds shown in the GPU charts
VLM_SHADE = (70, 58, 40)  # BGR: background of the charts while a VLM call runs
GRID = (58, 54, 50)


def _wall_time(wall) -> float | None:
    """An event's "HH:MM:SS" (today) as unix time."""
    try:
        return datetime.combine(datetime.now().date(), datetime.strptime(wall, "%H:%M:%S").time()).timestamp()
    except (TypeError, ValueError):
        return None


class DashState(RunState):
    """RunState that also keeps the session's VLM calls (role, seconds, tokens) across instructions, and their
    [start, end] in unix time (from the event's wall clock) for shading the GPU charts."""

    def __init__(self) -> None:
        self.vlm: deque = deque(maxlen=200)
        self.vlm_spans: deque = deque(maxlen=200)
        self.first_instruction: float | None = None  # unix time: the task phase starts (model loading is over)
        super().__init__()

    def apply(self, e: dict, frame_id: str | None) -> None:
        if e.get("kind") == "instruction" and self.first_instruction is None:
            self.first_instruction = _wall_time(e.get("wall"))
        if e.get("kind") == "vlm" and e.get("seconds") is not None:
            self.vlm.append((e.get("role", "?"), float(e["seconds"]), e.get("tokens")))
            end = _wall_time(e.get("wall"))  # the event is logged when the call ends
            if end is not None:
                self.vlm_spans.append((end - float(e["seconds"]), end))
        super().apply(e, frame_id)


class GpuFeed:
    """Tails a gpumon samples.csv (append-only) and keeps the last HISTORY_S seconds."""

    def __init__(self, run_dir: Path | None) -> None:
        self.path = Path(run_dir) / "samples.csv" if run_dir else None
        self.pos, self.header, self.partial = 0, None, ""
        self.rows: deque = deque()
        self.peak = 0.0  # whole recording, model loading included
        self.used: list = []  # (time, used_mib) of the whole session, for the task-phase peak
        self.task_t0: float | None = None
        self.task_peak: float | None = None  # max used since task_t0 (the first instruction)
        self.vlm_info: dict | None = None  # gpumon/torch_hook's vlm_<pid>.json: what was loaded

    def poll(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            with open(self.path) as f:
                f.seek(self.pos)
                data = f.read()
                self.pos = f.tell()
        except OSError:
            return
        text = self.partial + data
        lines = text.split("\n")
        self.partial = lines.pop()  # incomplete last line (or "")
        for line in lines:
            if not line or line.startswith("#"):
                continue
            if self.header is None:
                self.header = next(csv.reader(io.StringIO(line)))
                continue
            vals = next(csv.reader(io.StringIO(line)))
            try:
                r = {k: float(v) for k, v in zip(self.header, vals) if v != ""}
            except ValueError:
                continue
            self.rows.append(r)
            u = r.get("used_mib", 0.0)
            self.peak = max(self.peak, u)
            self.used.append((r.get("time", 0.0), u))
            if self.task_t0 is not None and r.get("time", 0.0) >= self.task_t0:
                self.task_peak = max(self.task_peak or 0.0, u)
        while self.rows and self.rows[-1]["time"] - self.rows[0]["time"] > HISTORY_S:
            self.rows.popleft()

    def start_tasks(self, t0: float | None) -> None:
        """The task phase starts at t0: the peak from then on (recomputed once from the stored samples)."""
        if t0 is None or self.task_t0 is not None:
            return
        self.task_t0 = t0
        after = [u for t, u in self.used if t >= t0]
        self.task_peak = max(after) if after else None

    def poll_model(self) -> None:
        if self.vlm_info is not None or self.path is None:
            return
        files = sorted((self.path.parent / "torch").glob("vlm_*.json"), key=lambda f: f.stat().st_mtime)
        if files:
            try:
                self.vlm_info = json.loads(files[-1].read_text())
            except (OSError, ValueError):
                pass

    @property
    def latest(self) -> dict | None:
        return self.rows[-1] if self.rows else None

    @property
    def live(self) -> bool:
        return bool(self.rows) and time.time() - self.rows[-1]["time"] < 3.0


def _put(img, text, x, y, col=FG, scale=0.45, th=1) -> None:
    cv2.putText(img, text, (x, y), FONT, scale, col, th, cv2.LINE_AA)


def fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Letterbox `img` into w x h (keeps the aspect ratio)."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    out = np.zeros((h, w, 3), np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def chart(img, rect, t0: float, t1: float, ymax: float, spans=(), yticks=(), yfmt="{:.0f}", label="") -> callable:
    """Draws the frame, VLM-call shading, grid and labels of one time chart; returns (t, v) -> pixel (x, y)."""
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (24, 22, 20), -1)
    def px(t, v):
        return int(x + w * (t - t0) / (t1 - t0)), int(y + h * (1 - min(max(v / ymax, 0.0), 1.0)))
    for a, b in spans:
        if b >= t0 and a <= t1:
            cv2.rectangle(img, (px(max(a, t0), 0)[0], y), (px(min(b, t1), 0)[0], y + h), VLM_SHADE, -1)
    for v in yticks:
        yy = px(t0, v)[1]
        cv2.line(img, (x, yy), (x + w, yy), GRID, 1)
        _put(img, yfmt.format(v), x + 4, yy - 3 if yy > y + 12 else yy + 12, MUTED, 0.38)
    for k in range(0, int(t1 - t0) + 1, 60):  # a tick every minute, counted back from now
        xx = px(t1 - k, 0)[0]
        cv2.line(img, (xx, y + h), (xx, y + h + 4), MUTED, 1)
        _put(img, "now" if k == 0 else f"-{k // 60} min", max(x, xx - (26 if k == 0 else 20)), y + h + 16, MUTED, 0.36)
    if label:
        _put(img, label, x + w - 8 - cv2.getTextSize(label, FONT, 0.42, 1)[0][0], y + 16, FG, 0.42)
    return px


class Dashboard(Viewer):
    def __init__(self, live_dir: Path, gpumon_dir: Path | None = None, snapshot_every: float = 0.0) -> None:
        super().__init__(live_dir, snapshot_every)
        self.rs = DashState()
        self.gpu = GpuFeed(gpumon_dir)
        self.gpumon_dir = gpumon_dir
        # the GPU column exists only with the ECE 565 plugin on (GPU_MONITOR); without it the window is the left side
        self.width = W if gpumon_dir is not None else MAIN[2]

    def poll_files(self) -> None:
        super().poll_files()
        self.gpu.poll()
        self.gpu.poll_model()
        self.gpu.start_tasks(self.rs.first_instruction)

    # ---------------------------------------------------------------------------------------------- panes
    def draw_main(self, canvas) -> None:
        x, y, w, h = MAIN
        cam = self.active_camera()
        img = self.annotated(cam)
        if img is None:
            canvas[y:y + h, x:x + w] = (20, 20, 20)
            _put(canvas, f"waiting for the {cam} camera ...", x + 40, y + h // 2, MUTED, 0.8, 2)
            return
        canvas[y:y + h, x:x + w] = fit(img, w, h)
        cv2.rectangle(canvas, (x, y + h - 24), (x + w, y + h), (0, 0, 0), -1)
        _put(canvas, self.camera_caption(cam)[:150], x + 8, y + h - 8, (220, 220, 220), 0.4)
        lx = x + 8  # legend of the highlight colours, top-left of the image
        cv2.rectangle(canvas, (x, y), (x + 470, y + 24), (0, 0, 0), -1)
        _put(canvas, f"{cam.upper()} CAMERA + YOLO", lx, y + 17, FG, 0.45, 1)
        lx += 190
        for word, col in (("target", TARGET_C), ("destination", DEST_C), ("asked", CAND_C)):
            cv2.rectangle(canvas, (lx, y + 8), (lx + 16, y + 16), col, -1)
            _put(canvas, word, lx + 20, y + 17, FG, 0.4)
            lx += 30 + cv2.getTextSize(word, FONT, 0.4, 1)[0][0]

    def draw_task(self, canvas) -> None:
        """One strip above the logs: the instruction, the supervisor state and the target / destination."""
        x, y, w, h = TASK
        canvas[y:y + h, x:x + w] = (22, 20, 18)
        rs = self.rs
        _put(canvas, "TASK", x + 10, y + 20, MUTED, 0.45)
        _put(canvas, (rs.instruction or "(waiting for an instruction)")[:70], x + 60, y + 20, FG, 0.55, 1)
        st = rs.state or "-"
        _put(canvas, "STATE", x + 10, y + 46, MUTED, 0.45)
        _put(canvas, st, x + 70, y + 46, LEVEL_COLOUR["success"] if st == "COMPLETED" else FG, 0.55, 2)
        tx = x + 110 + cv2.getTextSize(st, FONT, 0.55, 2)[0][0]
        for label in ("TARGET", "DESTINATION"):
            t, c = self._obj_line(rs.target if label == "TARGET" else rs.dest, label)[0]
            _put(canvas, t[:46], tx, y + 46, c, 0.45)
            tx += cv2.getTextSize(t[:46], FONT, 0.45, 1)[0][0] + 24

    def draw_gpu(self, canvas) -> None:
        x, y, w, h = GPU
        sub = np.full((h, w, 3), BG, np.uint8)
        _put(sub, "GPU MONITOR", 16, 30, FG, 0.7, 2)
        _put(sub, "gpumon: NVML per process + psutil, 10 Hz; recorded to disk"
                  + (f" ({self.gpumon_dir.name})" if self.gpumon_dir else ""), 190, 30, MUTED, 0.42)
        r = self.gpu.latest
        if r is None or not self.gpu.live:
            msg = "waiting for gpumon samples ..." if r is None else "gpumon stopped (no new samples)"
            _put(sub, msg, 16, 60, LEVEL_COLOUR["warn"], 0.5)
            if r is None:
                canvas[y:y + h, x:x + w] = sub
                return
        total, used, free = r["total_mib"], r["used_mib"], r["free_mib"]
        G = 1024.0
        frac = used / total
        col = LEVEL_COLOUR["error"] if frac > 0.9 else LEVEL_COLOUR["warn"] if frac > 0.75 else LEVEL_COLOUR["success"]
        # ---- headline numbers
        tp = self.gpu.task_peak
        tiles = [("VRAM used", f"{used / G:.2f} GB", col), ("free", f"{free / G:.2f} GB", col),
                 ("peak (during tasks)", "-" if tp is None else f"{tp / G:.2f} GB", FG),
                 ("GPU util", f"{r.get('gpu_util', 0):.0f} %", FG), ("power", f"{r.get('power_w', 0):.0f} W", FG)]
        tw = (w - 32) // len(tiles)
        for i, (k, v, c) in enumerate(tiles):
            tx = 16 + i * tw
            cv2.rectangle(sub, (tx, 46), (tx + tw - 8, 112), (24, 22, 20), -1)
            _put(sub, k, tx + 10, 66, MUTED, 0.42)
            _put(sub, v, tx + 10, 100, c, 0.85, 2)
        # ---- what is loaded (from gpumon's in-process probe: the layer types, not the settings)
        mi = self.gpu.vlm_info
        if mi:
            emb = "CPU (offloaded)" if mi.get("embeddings_on_cpu") else "GPU"
            head = f"{mi.get('model', '?')}   weights {mi.get('weights', '?')}"
            _put(sub, "MODEL", 16, 140, MUTED, 0.45)
            _put(sub, head, 80, 140, LEVEL_COLOUR["question"], 0.55, 2)
            _put(sub, f"embeddings on {emb}, load {mi.get('load_seconds', 0):.0f} s",
                 110 + cv2.getTextSize(head, FONT, 0.55, 2)[0][0], 140, FG, 0.45)
        else:
            _put(sub, "MODEL  waiting for the VLM to load (gpumon torch hook)", 16, 140, MUTED, 0.45)
        # ---- memory per process group (stacked bar) with the legend inside
        groups = ("sim", "agent", "other", "unattributed")
        bx, by, bw, bh = 16, 156, w - 32, 34
        cv2.rectangle(sub, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
        cx = bx
        for g in groups:
            seg = int(bw * r.get(f"gpu_{g}_mib", 0.0) / total)
            if seg > 0:
                cv2.rectangle(sub, (cx, by), (cx + seg - 2, by + bh), GROUP_C[g], -1)
                cx += seg
        _put(sub, f"{total / G:.0f} GB", bx + bw - 50, by + 22, FG, 0.45)
        lx = 16
        for g in groups:
            t = f"{GROUP_L[g]} {r.get(f'gpu_{g}_mib', 0.0) / G:.2f} GB"
            cv2.rectangle(sub, (lx, 202), (lx + 14, 216), GROUP_C[g], -1)
            _put(sub, t, lx + 20, 215, FG, 0.48)
            lx += 40 + cv2.getTextSize(t, FONT, 0.48, 1)[0][0]
        rows = list(self.gpu.rows)
        spans = list(self.rs.vlm_spans)
        t1 = rows[-1]["time"]
        t0 = t1 - HISTORY_S
        # ---- memory over time: stacked area per group, capacity line
        _put(sub, f"VRAM over the last {HISTORY_S // 60} min (stacked by process; shaded = Gemma call running)", 16, 244, FG, 0.48)
        rect = (16, 254, w - 32, 320)
        cap = total / G
        px = chart(sub, rect, t0, t1, cap * 1.04, spans, yticks=range(0, int(cap) + 1, 2), yfmt="{} GB")
        if len(rows) > 1:
            base = np.zeros(len(rows))
            for g in groups:
                top = base + np.array([q.get(f"gpu_{g}_mib", 0.0) / G for q in rows])
                upper = [px(q["time"], v) for q, v in zip(rows, top)]
                lower = [px(q["time"], v) for q, v in zip(reversed(rows), base[::-1])]
                cv2.fillPoly(sub, [np.array(upper + lower, np.int32)], GROUP_C[g], cv2.LINE_AA)
                base = top
            cv2.polylines(sub, [np.array([px(q["time"], q["used_mib"] / G) for q in rows], np.int32)], False, FG, 1, cv2.LINE_AA)
        yc = px(t0, cap)[1]
        cv2.line(sub, (rect[0], yc), (rect[0] + rect[2], yc), LEVEL_COLOUR["error"], 2)
        _put(sub, f"capacity {cap:.1f} GB", rect[0] + rect[2] - 150, yc - 6, LEVEL_COLOUR["error"], 0.45)
        # ---- GPU utilisation and PCIe over time
        _put(sub, "GPU utilisation and PCIe traffic (host->GPU)", 16, 614, FG, 0.48)
        rect2 = (16, 624, w - 32, 140)
        px2 = chart(sub, rect2, t0, t1, 100, spans, yticks=(0, 50, 100), yfmt="{}%")
        if len(rows) > 1:
            cv2.polylines(sub, [np.array([px2(q["time"], q.get("gpu_util", 0.0)) for q in rows], np.int32)], False,
                          LEVEL_COLOUR["success"], 2, cv2.LINE_AA)
            pmax = max(100.0, max(q.get("pcie_rx_mbs", 0.0) for q in rows))
            cv2.polylines(sub, [np.array([px2(q["time"], 100 * q.get("pcie_rx_mbs", 0.0) / pmax) for q in rows], np.int32)],
                          False, LEVEL_COLOUR["question"], 1, cv2.LINE_AA)
            _put(sub, f"util {r.get('gpu_util', 0):.0f} %", 420, 614, LEVEL_COLOUR["success"], 0.48)
            _put(sub, f"PCIe {r.get('pcie_rx_mbs', 0):.0f} MB/s (scale max {pmax:.0f})", 540, 614, LEVEL_COLOUR["question"], 0.48)
        # ---- RAM and VLM latency table
        yy = 810
        _put(sub, f"System RAM   agent {r.get('rss_agent_mib', 0) / G:.1f} GB   Isaac Sim {r.get('rss_sim_mib', 0) / G:.1f} GB   "
                  f"available {r.get('ram_available_mib', 0) / G:.0f} GB   (CPU offload shows up here)", 16, yy, FG, 0.45)
        calls = list(self.rs.vlm)
        yy += 30
        if not calls:
            _put(sub, "VLM latency: no Gemma calls yet", 16, yy, MUTED, 0.45)
        else:
            secs = [c[1] for c in calls]
            _put(sub, f"VLM latency   {len(calls)} calls   mean {np.mean(secs):.1f} s   last {secs[-1]:.1f} s ({calls[-1][0]})",
                 16, yy, FG, 0.48)
            by_role: dict = {}
            for role, sec, _tok in calls:
                by_role.setdefault(role, []).append(sec)
            roles = sorted(by_role.items(), key=lambda kv: -sum(kv[1]))[:6]
            smax = max(max(v) for _, v in roles)
            yy += 10
            for i, (role, v) in enumerate(roles):  # role | n | mean | max, with a bar of the mean
                ry = yy + 22 * (i + 1)
                _put(sub, f"{role[:16]:<16}", 16, ry, MUTED, 0.44)
                _put(sub, f"n {len(v):>3}   mean {np.mean(v):6.1f} s   max {max(v):6.1f} s", 170, ry, FG, 0.44)
                bl = int((w - 520) * np.mean(v) / smax)
                cv2.rectangle(sub, (500, ry - 12), (500 + max(bl, 2), ry - 2), GROUP_C["agent"], -1)
        canvas[y:y + h, x:x + w] = sub

    def draw_bar(self, canvas) -> None:
        rs = self.rs
        cv2.rectangle(canvas, (0, H - STATUS_H), (self.width, H), (12, 12, 12), -1)
        col = {"COMPLETED": LEVEL_COLOUR["success"], "FAILED": LEVEL_COLOUR["error"], "CANCELLED": LEVEL_COLOUR["warn"],
               "UNSUPPORTED": LEVEL_COLOUR["warn"], "WAITING FOR USER": CAND_C}.get(rs.status, FG)
        r = self.gpu.latest
        items = [(f"Task: {rs.status}", col), (f"Robot: {'MOVING' if rs.moving else 'STILL'}", FG),
                 (f"HOME: {rs.home}", LEVEL_COLOUR["success"] if rs.home == "VERIFIED" else FG),
                 *([(f"GPU {r['used_mib'] / 1024:.1f}/{r['total_mib'] / 1024:.0f} GB" if r and self.gpu.live
                     else "GPU: waiting", FG)] if self.gpumon_dir is not None else []),
                 (f"main: {self.active_camera()} ({self.camera_mode})", MUTED),
                 ("keys: q close | c camera | arrows/PgUp/PgDn/End logs", MUTED)]
        x = 10
        for t, c in items:
            _put(canvas, t, x, H - 9, c, 0.5)
            x += cv2.getTextSize(t, FONT, 0.5, 1)[0][0] + 28

    def render(self) -> np.ndarray:
        canvas = np.full((H, self.width, 3), BG, np.uint8)
        self.draw_main(canvas)
        self.draw_task(canvas)
        if self.gpumon_dir is not None:
            self.draw_gpu(canvas)
        x, y, w, h = LOGS
        self.draw_logs(canvas, y0=y + 4, y1=y + h - 4, x0=10, chars=125)
        for (ax, ay, aw, ah) in (MAIN, TASK, LOGS) + ((GPU,) if self.gpumon_dir is not None else ()):
            cv2.rectangle(canvas, (ax, ay), (ax + aw - 1, ay + ah - 1), LINE, 1)
        self.draw_bar(canvas)
        return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-dir", default=str(ROOT / "out/demo/live"))
    ap.add_argument("--gpumon-dir", default=None, help="a running gpumon recording (its samples.csv is shown)")
    ap.add_argument("--snapshot-every", type=float, default=0.0, help="save the window to <live-dir>/viewer.png every N s")
    ap.add_argument("--no-window", action="store_true")
    a, _ = ap.parse_known_args()
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    v = Dashboard(Path(a.live_dir), Path(a.gpumon_dir) if a.gpumon_dir else None, a.snapshot_every)
    rclpy.init()
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    node = rclpy.create_node("dashboard")
    subs = []

    def subscribe() -> None:
        for sub in subs:
            node.destroy_subscription(sub)
        q = qos_profile_sensor_data
        subs[:] = [node.create_subscription(Image, "table_camera/color/image_raw", lambda m: v.on_image("table", m), q),
                   node.create_subscription(Image, "camera/color/image_raw", lambda m: v.on_image("wrist", m), q)]
    subscribe()
    last_resub = time.time()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    title = "VT6L agent dashboard (read-only)"
    if not a.no_window:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, v.width, H)
    try:
        while not stop.is_set():
            newest = max((t for _, t in v.frames.values()), default=0.0)
            if time.time() - max(newest, last_resub) > 10.0:  # no frames: re-subscribe (discovery race at start-up)
                subscribe()
                last_resub = time.time()
            v.poll_files()
            canvas = v.render()
            v.maybe_snapshot(canvas)
            if a.no_window:
                time.sleep(0.1)
                continue
            cv2.imshow(title, canvas)
            k = cv2.waitKeyEx(66)
            if k != -1 and not v.key(k):
                break
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        ex.shutdown(timeout_sec=2.0)
        spin.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
