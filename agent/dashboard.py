#!/usr/bin/env python3
"""One-window dashboard (optional, read-only): everything the separate windows showed, plus the GPU monitor.

    +-----------------------------+------------------------------+
    | main camera + YOLO masks,   | the other camera             |
    | ids, target / destination   | (RGB; 'd' toggles depth)     |
    |                             +---------------+--------------+
    +-----------------------------+ task panel    | GPU monitor  |
    | live agent logs             |               |              |
    +-----------------------------+---------------+--------------+
    | status bar                                                 |

Like agent/viewer.py (whose drawing it reuses) it runs as its own process with the system Python and ROS 2, never
reads stdin and never writes anything the agent reads. The GPU panel only READS a gpumon recording
(<gpumon-dir>/samples.csv, written by `python -m gpumon record`, started by launch/agent.sh when GPU_MONITOR=True);
without one it says the monitor is off. The Isaac Sim window stays separate (or headless).

    source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=42
    python3 agent/dashboard.py --gpumon-dir out/gpumon/record/<run>
    keys: q close | c main camera (auto / table / wrist) | d side camera RGB / depth | arrows PgUp PgDn End: logs
"""

from __future__ import annotations

import argparse
import csv
import io
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.viewer import (BG, CAND_C, DEST_C, FG, FONT, LEVEL_COLOUR, MUTED, TARGET_C, RunState,  # noqa: E402
                          Viewer)

W, H, STATUS_H = 1920, 1080, 30
MAIN = (0, 0, 1024, 576)                       # x, y, w, h
LOGS = (0, 580, 1024, H - STATUS_H - 580)
SIDE = (1032, 0, W - 1032, 500)
PANEL = (1032, 504, 460, H - STATUS_H - 504)
GPU = (1496, 504, W - 1496, H - STATUS_H - 504)
LINE = (70, 70, 70)
# gpumon groups, same hues as gpumon/report.py's figures (BGR here)
GROUP_C = {"sim": (214, 120, 42), "agent": (52, 104, 235), "other": (122, 175, 27), "unattributed": (155, 162, 163)}
GROUP_L = {"sim": "Isaac Sim", "agent": "agent", "other": "desktop", "unattributed": "driver"}
DEPTH_RANGE = {"wrist": (0.1, 1.5), "table": (1.2, 3.2)}  # metres, as the old camera windows
HISTORY_S = 120


class DashState(RunState):
    """RunState that also keeps the session's VLM calls (role, seconds, tokens) across instructions."""

    def __init__(self) -> None:
        self.vlm: deque = deque(maxlen=200)
        super().__init__()

    def apply(self, e: dict, frame_id: str | None) -> None:
        if e.get("kind") == "vlm" and e.get("seconds") is not None:
            self.vlm.append((e.get("role", "?"), float(e["seconds"]), e.get("tokens")))
        super().apply(e, frame_id)


class GpuFeed:
    """Tails a gpumon samples.csv (append-only) and keeps the last HISTORY_S seconds."""

    def __init__(self, run_dir: Path | None) -> None:
        self.path = Path(run_dir) / "samples.csv" if run_dir else None
        self.pos, self.header, self.partial = 0, None, ""
        self.rows: deque = deque()
        self.peak = 0.0

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
            self.peak = max(self.peak, r.get("used_mib", 0.0))
        while self.rows and self.rows[-1]["time"] - self.rows[0]["time"] > HISTORY_S:
            self.rows.popleft()

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


def colorize_depth(depth: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    scaled = np.clip((np.nan_to_num(depth) - d_min) / (d_max - d_min), 0.0, 1.0)
    image = cv2.applyColorMap((255 - scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)  # near = warm colours
    image[~valid] = 0
    return image


class Dashboard(Viewer):
    def __init__(self, live_dir: Path, gpumon_dir: Path | None = None, snapshot_every: float = 0.0) -> None:
        super().__init__(live_dir, snapshot_every)
        self.rs = DashState()
        self.gpu = GpuFeed(gpumon_dir)
        self.gpumon_dir = gpumon_dir
        self.depth_msgs: dict = {}  # camera -> latest depth Image message (converted only when shown)
        self.side_depth = False

    def on_depth(self, cam: str, m) -> None:
        with self.lock:
            self.depth_msgs[cam] = m

    def poll_files(self) -> None:
        super().poll_files()
        self.gpu.poll()

    # ---------------------------------------------------------------------------------------------- panes
    def side_camera(self) -> str:
        return "wrist" if self.active_camera() == "table" else "table"

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

    def draw_side(self, canvas) -> None:
        x, y, w, h = SIDE
        cam = self.side_camera()
        title = f"{cam.upper()} CAMERA"
        img = None
        if self.side_depth:
            with self.lock:
                m = self.depth_msgs.get(cam)
            if m is not None:
                try:
                    d = np.frombuffer(m.data, np.float32 if m.encoding == "32FC1" else np.uint16).reshape(m.height, m.width)
                    d = d.astype(np.float32) / (1000.0 if m.encoding == "16UC1" else 1.0)
                    img = colorize_depth(d, *DEPTH_RANGE[cam])
                    title += f" - DEPTH {DEPTH_RANGE[cam][0]:.1f}-{DEPTH_RANGE[cam][1]:.1f} m (near = warm, far = cool)"
                except ValueError:
                    img = None
        else:
            img = self.annotated(cam)  # masks only if the latest detections are from this camera
            title += " - RGB"
        if img is None:
            canvas[y:y + h, x:x + w] = (20, 20, 20)
            _put(canvas, f"waiting for the {cam} camera{' depth' if self.side_depth else ''} ...", x + 30, y + h // 2, MUTED, 0.6)
        else:
            canvas[y:y + h, x:x + w] = fit(img, w, h)
        cv2.rectangle(canvas, (x, y), (x + 12 + 9 * len(title), y + 24), (0, 0, 0), -1)
        _put(canvas, title, x + 8, y + 17)

    def draw_task(self, canvas) -> None:
        x, y, w, h = PANEL
        sub = np.full((h, w, 3), BG, np.uint8)
        self.draw_panel(sub, x0=10, y0=-6, y_end=10_000)  # legend is on the main image instead; overflow is cut
        canvas[y:y + h, x:x + w] = sub

    def draw_gpu(self, canvas) -> None:
        x, y, w, h = GPU
        sub = np.full((h, w, 3), BG, np.uint8)
        _put(sub, "GPU MONITOR (gpumon)", 10, 22, MUTED, 0.5)
        r = self.gpu.latest
        if r is None or not self.gpu.live:
            msg = ("off: set GPU_MONITOR = True in agent/settings.py" if self.gpumon_dir is None
                   else "waiting for gpumon samples ..." if r is None else "gpumon stopped (no new samples)")
            _put(sub, msg, 10, 50, MUTED, 0.42)
            if r is None:
                canvas[y:y + h, x:x + w] = sub
                return
        total, used, free = r["total_mib"], r["used_mib"], r["free_mib"]
        frac = used / total
        col = LEVEL_COLOUR["error"] if frac > 0.9 else LEVEL_COLOUR["warn"] if frac > 0.75 else LEVEL_COLOUR["success"]
        _put(sub, f"{used / 1024:5.2f} / {total / 1024:.1f} GB", 10, 56, col, 0.75, 2)
        _put(sub, f"free {free / 1024:.2f} GB   peak {self.gpu.peak / 1024:.2f} GB   util {r.get('gpu_util', 0):.0f}%",
             10, 80, FG, 0.42)
        # stacked bar: memory per process group, then free
        bx, by, bw, bh = 10, 92, w - 20, 22
        cv2.rectangle(sub, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
        cx = bx
        for g in ("sim", "agent", "other", "unattributed"):
            seg = int(bw * r.get(f"gpu_{g}_mib", 0.0) / total)
            if seg > 0:
                cv2.rectangle(sub, (cx, by), (cx + seg - 2, by + bh), GROUP_C[g], -1)  # 2 px gap between segments
                cx += seg
        ly = by + bh + 20
        for i, g in enumerate(("sim", "agent", "other", "unattributed")):
            lx = 10 + (i % 2) * (w // 2)
            yy = ly + (i // 2) * 20
            cv2.rectangle(sub, (lx, yy - 10), (lx + 12, yy), GROUP_C[g], -1)
            _put(sub, f"{GROUP_L[g]} {r.get(f'gpu_{g}_mib', 0.0) / 1024:.2f} GB", lx + 18, yy, FG, 0.42)
        # trend: used memory over the last HISTORY_S seconds, capacity line on top
        tx, ty, tw, th = 10, ly + 40, w - 20, 120
        cv2.rectangle(sub, (tx, ty), (tx + tw, ty + th), (24, 22, 20), -1)
        rows = list(self.gpu.rows)
        if len(rows) > 1:
            t_end = rows[-1]["time"]
            pts = np.array([[tx + tw * (1 - (t_end - q["time"]) / HISTORY_S), ty + th * (1 - q["used_mib"] / total)]
                            for q in rows], np.int32)
            cv2.polylines(sub, [pts], False, GROUP_C["sim"], 2, cv2.LINE_AA)
        cv2.line(sub, (tx, ty), (tx + tw, ty), (200, 200, 200), 1)
        _put(sub, f"capacity {total / 1024:.0f} GB", tx + tw - 110, ty + 14, MUTED, 0.38)
        _put(sub, f"used, last {HISTORY_S} s", tx + 4, ty + th - 6, MUTED, 0.38)
        yy = ty + th + 24
        _put(sub, f"RAM  agent {r.get('rss_agent_mib', 0) / 1024:.1f} GB  sim {r.get('rss_sim_mib', 0) / 1024:.1f} GB  "
                  f"avail {r.get('ram_available_mib', 0) / 1024:.0f} GB", 10, yy, FG, 0.42)
        calls = list(self.rs.vlm)
        if calls:
            secs = [c[1] for c in calls]
            _put(sub, f"VLM  {len(calls)} calls, mean {np.mean(secs):.1f} s, last {secs[-1]:.1f} s", 10, yy + 22, FG, 0.42)
            for i, (role, s, tok) in enumerate(reversed(calls[-4:])):
                _put(sub, f"  {role[:22]:<22} {s:5.1f} s" + (f"  {tok} tok" if tok else ""), 10, yy + 42 + 18 * i, MUTED, 0.4)
        canvas[y:y + h, x:x + w] = sub

    def draw_bar(self, canvas) -> None:
        rs = self.rs
        cv2.rectangle(canvas, (0, H - STATUS_H), (W, H), (12, 12, 12), -1)
        col = {"COMPLETED": LEVEL_COLOUR["success"], "FAILED": LEVEL_COLOUR["error"], "CANCELLED": LEVEL_COLOUR["warn"],
               "UNSUPPORTED": LEVEL_COLOUR["warn"], "WAITING FOR USER": CAND_C}.get(rs.status, FG)
        r = self.gpu.latest
        items = [(f"Task: {rs.status}", col), (f"Robot: {'MOVING' if rs.moving else 'STILL'}", FG),
                 (f"HOME: {rs.home}", LEVEL_COLOUR["success"] if rs.home == "VERIFIED" else FG),
                 (f"GPU {r['used_mib'] / 1024:.1f}/{r['total_mib'] / 1024:.0f} GB" if r and self.gpu.live else "GPU monitor off", FG),
                 (f"main: {self.active_camera()} ({self.camera_mode})", MUTED),
                 ("keys: q close | c main camera | d side depth | arrows/PgUp/PgDn/End logs", MUTED)]
        x = 10
        for t, c in items:
            _put(canvas, t, x, H - 9, c, 0.5)
            x += cv2.getTextSize(t, FONT, 0.5, 1)[0][0] + 28

    def render(self) -> np.ndarray:
        canvas = np.full((H, W, 3), BG, np.uint8)
        self.draw_main(canvas)
        self.draw_side(canvas)
        self.draw_task(canvas)
        self.draw_gpu(canvas)
        x, y, w, h = LOGS
        self.draw_logs(canvas, y0=y + 4, y1=y + h - 4, x0=10, chars=125)
        for (ax, ay, aw, ah) in (MAIN, SIDE, PANEL, GPU, LOGS):
            cv2.rectangle(canvas, (ax, ay), (ax + aw - 1, ay + ah - 1), LINE, 1)
        self.draw_bar(canvas)
        return canvas

    def key(self, k: int) -> bool:
        if k == ord("d"):
            self.side_depth = not self.side_depth
            return True
        return super().key(k)


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
                   node.create_subscription(Image, "camera/color/image_raw", lambda m: v.on_image("wrist", m), q),
                   node.create_subscription(Image, "table_camera/depth/image_rect_raw", lambda m: v.on_depth("table", m), q),
                   node.create_subscription(Image, "camera/depth/image_rect_raw", lambda m: v.on_depth("wrist", m), q)]
    subscribe()
    last_resub = time.time()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    title = "VT6L agent dashboard (read-only)"
    if not a.no_window:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, W, H)
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
