#!/usr/bin/env python3
"""Demo viewer (optional, read-only): live camera + the agent's latest perception result + live logs.

Runs as its own process with the system Python and ROS 2 (like ros_nodes/camera_viewer.py). It never
reads stdin and never writes anything the agent reads: closing it cannot affect the robot task.

  image   live table (or wrist) camera at the camera's frame rate, with the latest perception result drawn on
          top exactly as Gemma gets it (black outlines, world-id numbers); one result per observation:
          the detector does NOT run here
  panel   task, supervisor state, target / destination, budgets, pending question, robot / HOME status
  logs    every event of the active run, formatted by agent.events (same lines as the terminal)

    source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=42
    python3 agent/viewer.py                      # window; keys: q close viewer, c camera, arrows / PgUp / PgDn scroll logs
    python3 agent/viewer.py --snapshot-every 5   # also save the window to <live-dir>/viewer.png every 5 s
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # the repository (this file is in src/<package>/)
sys.path.insert(0, str(ROOT / "src"))
from agent.events import format_event  # noqa: E402
from perception.somark import draw_layout, layout  # noqa: E402

W, H = 1600, 900
IMG_W, IMG_H = 1056, 594
PANEL_X = IMG_W + 12
LOG_Y0, STATUS_H = IMG_H + 8, 30
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG, FG, MUTED = (32, 28, 24), (235, 235, 230), (160, 160, 150)
LEVEL_COLOUR = {"info": FG, "warn": (60, 180, 255), "error": (80, 80, 255), "success": (120, 220, 120), "question": (255, 200, 80)}
TARGET_C, DEST_C, CAND_C = (90, 230, 90), (255, 170, 60), (60, 200, 255)  # BGR: green, blue, orange
MOTION_STATES = {"APPROACHING", "GRASPING", "TRANSPORTING", "PLACING", "RETREATING", "RETURNING_HOME", "RECOVERING"}


class RunState:
    """Everything the panel shows, derived only from the run's logged events."""

    def __init__(self) -> None:
        self.reset("")

    def reset(self, instruction: str) -> None:
        self.instruction, self.state, self.status = instruction, "-", "WAITING"
        self.target = self.dest = None  # (mark, world id, frame id)
        self.question = None  # (text, candidates) while a question waits for the user
        self.retries, self.budgets_used = {}, {}
        self.moving, self.home, self.last_plan = False, "PENDING", None
        self.lines: deque = deque(maxlen=2000)
        self.steps: list = []  # goal plan: [{"index", "capability", "status"}]

    def apply(self, e: dict, frame_id: str | None) -> None:
        k = e.get("kind")
        if k == "goal_plan":
            self.steps = [s for s in self.steps if s["status"] != "pending"] + \
                [{"index": x["index"], "capability": x["capability"], "status": "pending"} for x in e.get("steps", [])]
        elif k == "step" and e.get("event") in ("start", "end"):
            for st in self.steps:
                if st["index"] == e.get("index"):
                    st["status"] = "running" if e["event"] == "start" else e.get("outcome", "?")
        if k == "state":
            self.state = e["state"]
            if self.state in ("COMPLETED", "FAILED", "CANCELLED", "UNSUPPORTED") and e.get("level") == "goal":
                self.status = self.state
            elif self.status not in ("COMPLETED", "FAILED", "CANCELLED", "UNSUPPORTED"):
                self.status = "RUNNING"
            if self.state in ("RETURNING_HOME",):
                self.home = "MOVING"
        elif k == "localized":  # (mark in the frame it was chosen in, world id, that frame)
            self.target = (e.get("object_mark"), e.get("object_id"), e.get("frame_id"))
            self.dest = (e.get("reference_mark"), e.get("reference_id"), e.get("frame_id"))
        elif k == "question":
            self.question = (e.get("question"), e.get("candidates") or [])
            self.status = "WAITING FOR USER"
        elif k == "ask_user":
            self.question = None
            self.status = "RUNNING"
        elif k == "failure":
            self.budgets_used[e.get("cls")] = f"{e.get('used')}/{e.get('budget')}"
        elif k == "tool" and e.get("tool") == "execute_start":
            self.moving = True
        elif k == "tool" and e.get("tool") == "execute":
            self.moving = False
        elif k == "verification" and e.get("check") == "home":
            self.home = "VERIFIED" if e.get("pass") else "NOT REACHED"
        elif k == "plan":
            self.last_plan = e
        ln = format_event(e)
        if ln is not None:
            self.lines.append(ln)


class Viewer:
    def __init__(self, live_dir: Path, snapshot_every: float = 0.0, snapshot_dir: Path | None = None) -> None:
        self.live_dir, self.snapshot_every, self.snapshot_dir = Path(live_dir), snapshot_every, snapshot_dir
        self.n_snap = 0
        self.frames: dict = {}  # camera -> (bgr, stamp)
        self.fps: dict = {c: deque(maxlen=30) for c in ("table", "wrist")}
        self.lock = threading.Lock()
        self.run_file = None
        self.ev_pos = 0
        self.rs = RunState()
        self.perc, self.perc_mtime, self.labels, self.lab_mtime, self.contours = None, 0, None, 0, {}
        self.capture_rgb = None
        self.perc_updates: deque = deque(maxlen=10)
        self.camera_mode = "auto"
        self.scroll = 0
        self.last_snap = 0.0

    # ------------------------------------------------------------------ inputs (read-only)
    def on_image(self, cam: str, m) -> None:
        try:
            a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[..., :3]
            bgr = cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if m.encoding.lower().startswith("rgb") else a.copy()
        except ValueError:
            return
        with self.lock:
            self.frames[cam] = (bgr, time.time())
            self.fps[cam].append(time.time())

    def poll_files(self) -> None:
        cur = self.live_dir / "current.json"
        try:
            c = json.loads(cur.read_text())
        except (OSError, ValueError):
            c = None
        if c is not None and c.get("events") != self.run_file:  # a new run, or a new agent process (events None)
            self.run_file, self.ev_pos = c.get("events"), 0
            self.rs.reset(c.get("instruction", ""))
            self.scroll = 0
            self.perc_updates.clear()
            self.run_started = c.get("started", 0.0)
        if self.run_file:
            try:
                with open(self.run_file) as f:
                    f.seek(self.ev_pos)
                    while True:
                        line = f.readline()
                        if not line or not line.endswith("\n"):
                            break
                        self.ev_pos = f.tell()
                        try:
                            self.rs.apply(json.loads(line), self.perc and self.perc.get("frame_id"))
                        except ValueError:
                            pass
            except OSError:
                pass
        p = self.live_dir / "perception.json"
        if not p.exists():  # cleared by a new agent process: no detection overlay
            self.perc, self.perc_mtime = None, 0
            self.capture_rgb = None
        try:
            mt = p.stat().st_mtime
            if mt != self.perc_mtime:
                new = json.loads(p.read_text())
                if (not self.perc or new.get("frame_id") != self.perc.get("frame_id")) and new.get("time", 0) >= getattr(self, "run_started", 0.0):
                    self.perc_updates.append(time.time())
                self.perc, self.perc_mtime = new, mt
                self.capture_rgb = cv2.imread(str(self.live_dir / new['rgb_file'])) if new.get('rgb_file') else None
        except (OSError, ValueError):
            pass
        lp = self.live_dir / (self.perc or {}).get('labels_file', 'labels.png')
        if not lp.exists():
            self.labels, self.lab_mtime, self.contours = None, 0, {}
        try:
            mt = lp.stat().st_mtime
            if mt != self.lab_mtime:
                lab = cv2.imread(str(lp), cv2.IMREAD_UNCHANGED)
                if lab is not None:
                    self.labels, self.lab_mtime = lab, mt
                    self.contours = {}
                    for k in np.unique(lab):
                        if k:
                            cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            self.contours[int(k)] = cs
        except OSError:
            pass

    # ---------------------------------------------------------------------- drawing
    def active_camera(self) -> str:
        if self.camera_mode != "auto":
            return self.camera_mode
        return (self.perc or {}).get("camera", "table")

    def mark_of(self, sel) -> int | None:
        """The selected object's mark in the SHOWN perception result: the chosen mark if this is the frame it
        was chosen in, else the mark carrying its world id, else None (not identified in this frame)."""
        if not sel or sel[0] is None or not self.perc:
            return None
        marks = self.perc.get("marks", {})
        for k, v in marks.items():
            if v.get("object_id") is not None and v.get("object_id") == sel[1]:
                return int(k)
        if len(sel) > 2 and sel[2] is not None and sel[2] == self.perc.get("frame_id") and str(sel[0]) in marks:
            return int(sel[0])
        return None

    def highlight(self) -> dict[int, tuple]:
        """mark -> (colour, label) for target / destination / question candidates in the shown result."""
        out = {}
        for sel, col, name in ((self.rs.target, TARGET_C, "TARGET"), (self.rs.dest, DEST_C, "DESTINATION")):
            k = self.mark_of(sel)
            if k is not None:
                out[k] = (col, name)
        if self.rs.question:
            for k in self.rs.question[1]:
                out.setdefault(int(k), (CAND_C, "?"))
        return out

    def annotated(self, cam: str) -> np.ndarray | None:
        """Latest frame of `cam` with the masks / ids of the latest perception result on it (if it is from this
        camera and size); None before the first frame. Also used by agent/dashboard.py."""
        with self.lock:
            fr = self.frames.get(cam)
        if fr is None:
            return None
        img = fr[0].copy()
        if self.rs.moving:
            return img  # live motion: never paint an old object's outlines onto a different exposure
        if self.capture_rgb is not None and (self.perc or {}).get('camera') == cam:
            img = self.capture_rgb.copy()
        h, w = img.shape[:2]
        perc = self.perc or {}
        if self.labels is not None and perc.get("camera") == cam and self.labels.shape[:2] == (h, w):
            # the same drawing Gemma grounds on (perception.somark): black outlines, world-id numbers, real colours
            marks = perc.get("marks", {})
            ids = {k: (marks.get(str(k), {}).get("object_id") or k) for k in self.contours}
            key = (self.lab_mtime, tuple(sorted(ids.items())))
            if getattr(self, "_layout_key", None) != key:
                self._layout = layout({ids[k]: self.labels == k for k in self.contours})
                self._layout_key = key
            img = draw_layout(img, self._layout)
            for k, (col, name) in self.highlight().items():  # target / destination, if the run names one
                cv2.drawContours(img, self.contours.get(k, []), -1, col, 4)
        return img

    def camera_caption(self, cam: str) -> str:
        """One line under the image: frame rate and age of the shown detections."""
        perc = self.perc or {}
        now = time.time()
        f = self.fps[cam]
        fps = (len(f) - 1) / (f[-1] - f[0]) if len(f) > 1 and f[-1] > f[0] else 0.0
        age = f"{now - perc['time']:.0f} s ago" if perc.get("time") else "none yet"
        mode = 'LIVE, overlays hidden during motion' if self.rs.moving else 'CAPTURED model observation' if self.capture_rgb is not None else 'LIVE'
        txt = (f"{cam} {mode} | stream {fps:4.1f} fps | detections: frame {perc.get('frame_id', '-')} ({age}, "
               f"{len(self.perc_updates)} updates this run) | YOLO runs only when the agent observes")
        return txt

    def draw_image(self, canvas) -> None:
        cam = self.active_camera()
        img = self.annotated(cam)
        if img is None:
            canvas[0:IMG_H, 0:IMG_W] = (20, 20, 20)
            cv2.putText(canvas, f"waiting for the {cam} camera ...", (40, IMG_H // 2), FONT, 0.8, MUTED, 2)
            return
        canvas[0:IMG_H, 0:IMG_W] = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        txt = self.camera_caption(cam)
        cv2.rectangle(canvas, (0, IMG_H - 26), (IMG_W, IMG_H), (0, 0, 0), -1)
        cv2.putText(canvas, txt, (8, IMG_H - 8), FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    def _obj_line(self, sel, label) -> list[tuple]:
        if not sel or sel[0] is None:
            return [(f"{label}: -", MUTED)]
        col = TARGET_C if label == "TARGET" else DEST_C
        k = self.mark_of(sel)
        head = f"{label}: object {sel[0]} (world id {sel[1]})"
        if k is None:
            return [(head, col), (f"  not identified in frame {(self.perc or {}).get('frame_id', '-')} (e.g. covered)", MUTED)]
        info = self.perc["marks"][str(k)]
        d = info.get("description") or {}
        c = info.get("centre") or [0, 0, 0]
        s = info.get("score")
        if k != sel[0]:
            head += f", now mark {k}"
        return [(head, col),
                (f"  {d.get('colour', '')} {d.get('name', '')}"[:44], FG),
                (f"  at x {c[0] * 1000:+.0f}, y {c[1] * 1000:+.0f}, z {c[2] * 1000:.0f} mm" + (f", conf {s:.2f}" if s else ""), MUTED)]

    def draw_panel(self, canvas, x0: int = PANEL_X, y0: int = 0, y_end: int = IMG_H) -> None:
        """Task / state / target panel from x0, y0; its colour legend sits just above y_end."""
        x, y = x0, y0 + 28
        rs = self.rs

        def put(t, col=FG, scale=0.55, th=1, dy=24):
            nonlocal y
            cv2.putText(canvas, t, (x, y), FONT, scale, col, th, cv2.LINE_AA)
            y += dy
        put("CURRENT TASK", MUTED, 0.5)
        words, line = (rs.instruction or "(waiting for an instruction)").split(), ""
        for wd in words:
            if len(line) + len(wd) > 40:
                put(line, FG, 0.6, 1)
                line = ""
            line += wd + " "
        put(line, FG, 0.6, 1, 34)
        put("SUPERVISOR STATE", MUTED, 0.5)
        put(rs.state, LEVEL_COLOUR["success"] if rs.state == "COMPLETED" else FG, 0.75, 2, 36)
        if rs.steps:
            put("GOAL PLAN", MUTED, 0.5, 1, 20)
            for st in rs.steps[-5:]:
                col = {"completed": LEVEL_COLOUR["success"], "running": CAND_C, "pending": MUTED}.get(st["status"], LEVEL_COLOUR["warn"])
                put(f"  {st['index']}. {st['capability']}: {st['status']}", col, 0.45, 1, 19)
            y += 6
        for t, c in self._obj_line(rs.target, "TARGET") + self._obj_line(rs.dest, "DESTINATION"):
            put(t, c, 0.5, 1, 22)
        y += 8
        if rs.last_plan:
            g = rs.last_plan.get("grasp") or {}
            put(f"grasp yaw {g.get('yaw_deg')} deg, width {g.get('width_mm')} mm, h {g.get('height_mm')} mm", MUTED, 0.45, 1, 22)
        if rs.budgets_used:
            put("retries used: " + ", ".join(f"{k} {v}" for k, v in rs.budgets_used.items()), LEVEL_COLOUR["warn"], 0.45, 1, 22)
        if rs.question:
            y += 6
            put("QUESTION (answer in the terminal):", CAND_C, 0.5, 2)
            put(rs.question[0][:46], CAND_C, 0.45, 1, 20)
            put(f"candidates: {rs.question[1]}", CAND_C, 0.45, 1, 20)
        lx, ly = x, y_end - 20  # legend: a coloured swatch before each word
        for word, col in (("TARGET", TARGET_C), ("DESTINATION", DEST_C), ("question candidates", CAND_C)):
            cv2.rectangle(canvas, (lx, ly - 11), (lx + 22, ly - 3), col, -1)
            cv2.putText(canvas, word, (lx + 28, ly), FONT, 0.45, FG, 1, cv2.LINE_AA)
            lx += 40 + cv2.getTextSize(word, FONT, 0.45, 1)[0][0]

    def draw_logs(self, canvas, y0: int = LOG_Y0, y1: int = H - STATUS_H - 4, x0: int = 10, chars: int = 190) -> None:
        cv2.putText(canvas, "LIVE AGENT LOGS" + ("  (scrolled: End to follow)" if self.scroll else ""), (x0, y0 + 14), FONT, 0.5, MUTED, 1)
        n = (y1 - y0 - 24) // 19
        lines = list(self.rs.lines)
        end = max(0, len(lines) - self.scroll)
        for i, ln in enumerate(lines[max(0, end - n):end]):
            cv2.putText(canvas, str(ln)[:chars], (x0, y0 + 38 + 19 * i), FONT, 0.45, LEVEL_COLOUR.get(ln.level, FG), 1, cv2.LINE_AA)

    def draw_status(self, canvas) -> None:
        rs = self.rs
        cv2.rectangle(canvas, (0, H - STATUS_H), (W, H), (12, 12, 12), -1)
        col = {"COMPLETED": LEVEL_COLOUR["success"], "FAILED": LEVEL_COLOUR["error"], "CANCELLED": LEVEL_COLOUR["warn"],
               "UNSUPPORTED": LEVEL_COLOUR["warn"],
               "WAITING FOR USER": CAND_C}.get(rs.status, FG)
        items = [(f"Task: {rs.status}", col), (f"Robot: {'MOVING' if rs.moving else 'STILL'}", FG),
                 (f"HOME: {rs.home}", LEVEL_COLOUR["success"] if rs.home == "VERIFIED" else FG),
                 (f"camera view: {self.active_camera()} ({self.camera_mode})", MUTED), ("keys: q close viewer | c camera | arrows/PgUp/PgDn/End logs", MUTED)]
        x = 10
        for t, c in items:
            cv2.putText(canvas, t, (x, H - 9), FONT, 0.5, c, 1, cv2.LINE_AA)
            x += cv2.getTextSize(t, FONT, 0.5, 1)[0][0] + 28

    def render(self) -> np.ndarray:
        canvas = np.full((H, W, 3), BG, np.uint8)
        self.draw_image(canvas)
        self.draw_panel(canvas)
        cv2.line(canvas, (0, IMG_H + 2), (W, IMG_H + 2), (70, 70, 70), 1)
        self.draw_logs(canvas)
        self.draw_status(canvas)
        return canvas

    def key(self, k: int) -> bool:
        """False = close the viewer."""
        if k in (ord("q"), 27):
            return False
        if k == ord("c"):
            self.camera_mode = {"auto": "table", "table": "wrist", "wrist": "auto"}[self.camera_mode]
        elif k in (82, 0x260000, ord("k")):  # up
            self.scroll = min(self.scroll + 1, max(0, len(self.rs.lines) - 1))
        elif k in (84, 0x280000, ord("j")):  # down
            self.scroll = max(0, self.scroll - 1)
        elif k in (85, 0x210000):  # page up
            self.scroll = min(self.scroll + 10, max(0, len(self.rs.lines) - 1))
        elif k in (86, 0x220000):  # page down
            self.scroll = max(0, self.scroll - 10)
        elif k in (87, 0x230000):  # end
            self.scroll = 0
        return True

    def maybe_snapshot(self, canvas) -> None:
        if self.snapshot_every > 0 and time.time() - self.last_snap >= self.snapshot_every:
            if self.snapshot_dir:
                self.snapshot_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(self.snapshot_dir / f"viewer_{self.n_snap:04d}.jpg"), canvas)
                self.n_snap += 1
            else:
                cv2.imwrite(str(self.live_dir / "viewer.png"), canvas)
            self.last_snap = time.time()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-dir", default=str(ROOT / "out/demo/live"))
    ap.add_argument("--snapshot-every", type=float, default=0.0, help="save the window to <live-dir>/viewer.png every N s")
    ap.add_argument("--snapshot-dir", help="with --snapshot-every: keep numbered snapshots in this folder instead")
    ap.add_argument("--no-window", action="store_true", help="no window (with --snapshot-every: headless check)")
    a, _ = ap.parse_known_args()
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    v = Viewer(Path(a.live_dir), a.snapshot_every, Path(a.snapshot_dir) if a.snapshot_dir else None)
    rclpy.init()
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):  # rclpy's own handlers would leave the render loop running
        signal.signal(sig, lambda *_: stop.set())
    node = rclpy.create_node("demo_viewer")
    subs = []

    def subscribe() -> None:
        for sub in subs:
            node.destroy_subscription(sub)
        subs[:] = [node.create_subscription(Image, "table_camera/color/image_raw", lambda m: v.on_image("table", m), qos_profile_sensor_data),
                   node.create_subscription(Image, "camera/color/image_raw", lambda m: v.on_image("wrist", m), qos_profile_sensor_data)]
    subscribe()
    last_resub = time.time()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    title = "Multi-agent robot demo (read-only viewer)"
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
