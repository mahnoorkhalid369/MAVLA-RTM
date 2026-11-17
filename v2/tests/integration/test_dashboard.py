"""One-window dashboard (agent/dashboard.py): runs with the system Python (no ROS, no GPU). It reuses the demo
viewer's overlay, shows the main camera, the task strip, the logs and a full-height GPU monitor, and only reads a
gpumon recording."""

import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from agent.dashboard import GPU, H, MAIN, TASK, W, Dashboard, GpuFeed
from agent.viewer import TARGET_C

COLUMNS = ["time", "used_mib", "free_mib", "total_mib", "gpu_util", "gpu_sim_mib", "gpu_agent_mib", "gpu_other_mib",
           "gpu_unattributed_mib", "rss_agent_mib", "rss_sim_mib", "ram_available_mib"]


def ev(kind, **f):
    return {"wall": "13:05:01", "kind": kind, **f}


def _live_dir(tmp_path: Path) -> Path:
    lab = np.zeros((720, 1280), np.uint16)
    lab[300:400, 500:600], lab[300:400, 800:900] = 1, 2
    cv2.imwrite(str(tmp_path / "labels.png"), lab)
    (tmp_path / "perception.json").write_text(json.dumps({"frame_id": "table_001", "camera": "table", "time": 1.0, "marks": {
        "1": {"object_id": 11, "description": {"name": "cube", "colour": "red"}, "centre": [0.0, 0.58, 0.03], "score": 0.9},
        "2": {"object_id": 12, "description": {"name": "cube", "colour": "green"}, "centre": [0.15, 0.5, 0.025], "score": 0.9}}}))
    run = tmp_path / "run"
    run.mkdir()
    events = [ev("instruction", text="put the red cube on the green cube"), ev("state", state="LOCALIZING"),
              ev("vlm", role="planner", seconds=4.2, tokens=310, valid=True, attempt=1),
              ev("localized", frame_id="table_001", object_mark=1, object_id=11, reference_mark=2, reference_id=12,
                 object={"centre": [0, .58, .03]}, reference={"centre": [.15, .5, .025]}),
              ev("state", state="APPROACHING")]
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (tmp_path / "current.json").write_text(json.dumps({"events": str(run / "events.jsonl"),
                                                       "instruction": "put the red cube on the green cube"}))
    return tmp_path


def _gpumon(tmp_path: Path, n: int = 50) -> Path:
    d = tmp_path / "gpumon"
    d.mkdir()
    now = time.time()
    with open(d / "samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for i in range(n):
            w.writerow({"time": now - (n - i) * 0.1, "used_mib": 8000 + i, "free_mib": 4282 - i, "total_mib": 12282,
                        "gpu_util": 40, "gpu_sim_mib": 3200, "gpu_agent_mib": 4100 + i, "gpu_other_mib": 400,
                        "gpu_unattributed_mib": 300, "rss_agent_mib": 7000, "rss_sim_mib": 9000, "ram_available_mib": 40000})
        f.write(f"{now:.3f},80")  # a line still being written: not read yet
    return d


def test_dashboard_shows_camera_task_and_gpu(tmp_path):
    v = Dashboard(_live_dir(tmp_path), _gpumon(tmp_path))
    v.frames["table"] = (np.full((720, 1280, 3), 90, np.uint8), 0.0)
    v.frames["wrist"] = (np.full((720, 1280, 3), 150, np.uint8), 0.0)
    v.poll_files()
    img = v.render()
    assert img.shape == (H, W, 3)
    x, y, w, h = MAIN
    assert (img[y:y + h, x:x + w] == TARGET_C).all(axis=-1).any()  # target highlighted on the main (table) camera
    tx, ty, tw, th = TASK
    assert img[ty:ty + th, tx:tx + tw].std() > 5  # the task strip is drawn
    assert len(v.gpu.rows) == 50 and v.gpu.latest["gpu_agent_mib"] == 4149 and v.gpu.live
    assert list(v.rs.vlm) == [("planner", 4.2, 310)] and len(v.rs.vlm_spans) == 1
    gx, gy, gw, gh = GPU
    assert img[gy:gy + gh, gx:gx + gw].std() > 10  # the GPU panel is drawn, not empty
    if os.environ.get("DASHBOARD_PNG"):  # look at it: DASHBOARD_PNG=/tmp/x.png pytest ...
        cv2.imwrite(os.environ["DASHBOARD_PNG"], img)


def test_main_camera_cycles(tmp_path):
    v = Dashboard(_live_dir(tmp_path))
    v.key(ord("c"))  # auto -> table
    assert v.active_camera() == "table"
    v.key(ord("c"))  # table -> wrist
    assert v.active_camera() == "wrist"
    v.render()  # no wrist frame yet: a waiting message, no error


def test_without_the_gpu_plugin_the_window_has_no_gpu_column(tmp_path):
    v = Dashboard(_live_dir(tmp_path), None)  # GPU_MONITOR off: the ECE 565 plugin is not plugged in
    v.poll_files()
    assert v.gpu.latest is None and v.render().shape == (H, MAIN[2], 3)


def test_gpu_feed_keeps_partial_lines_for_the_next_poll(tmp_path):
    d = _gpumon(tmp_path, n=3)
    feed = GpuFeed(d)
    feed.poll()
    assert len(feed.rows) == 3
    with open(d / "samples.csv", "a") as f:
        f.write(",0,0,0,0,0,0,0,0,0,0\n")  # completes the partial line
    feed.poll()
    assert len(feed.rows) == 4 and feed.rows[-1]["free_mib"] == 0.0
