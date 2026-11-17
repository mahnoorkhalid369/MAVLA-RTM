"""Live channel from the agent to the (optional) demo viewer: a few small files, written atomically.

    <live_dir>/current.json     which run is active: {"run_dir", "events", "instruction", "started"};
                                events None = a new agent process, no task yet (LiveChannel.reset)
    <live_dir>/perception.json  the latest perception result: frame id, camera, time, and per mark the
                                detector score, measured geometry, the VLM's description, world id
    <live_dir>/labels.png       uint16 image of that frame: pixel = mark number (0 = none)

The agent only writes (never waits for a reader); the viewer only reads. The viewer draws the masks of
the latest *perception result* over the *live* camera image, so the detector's update rate (one
result per observation) and the camera's frame rate stay separate and are both shown.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class LiveChannel:
    def __init__(self, live_dir) -> None:
        self.dir = Path(live_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self) -> None:
        """A new agent process: no task is active yet. The previous process's task and detection overlays are
        removed, so the viewer shows "(waiting for an instruction)" instead of an old run."""
        for name in ("perception.json", "labels.png"):
            (self.dir / name).unlink(missing_ok=True)
        rec = {"run_dir": None, "events": None, "instruction": "", "started": time.time()}
        _atomic_write(self.dir / "current.json", json.dumps(rec).encode())

    def start_run(self, run_dir, instruction: str) -> None:
        rec = {"run_dir": str(run_dir), "events": str(Path(run_dir) / "events.jsonl"), "instruction": instruction,
               "started": time.time()}
        _atomic_write(self.dir / "current.json", json.dumps(rec).encode())

    def perception(self, obs, stamp: float | None = None) -> None:
        """Publish an observation (masks once per frame, the JSON whenever descriptions / ids are added)."""
        try:
            if getattr(obs, "_live_labels_written", False) is False:
                h, w = obs.rgb.shape[:2]
                lab = np.zeros((h, w), np.uint16)
                for k, m in obs.marks.items():
                    lab[m & (lab == 0)] = k
                ok, png = cv2.imencode(".png", lab)
                if ok:
                    _atomic_write(self.dir / "labels.png", png.tobytes())
                obs._live_labels_written = True
            marks = {}
            for k in obs.marks:
                g = obs.measured.get(k, {})
                marks[str(k)] = {"score": (obs.scores or {}).get(k), "description": obs.descriptions.get(k),
                                 "object_id": obs.object_ids.get(k), "centre": g.get("centre"), "top_z": g.get("top_z"),
                                 "footprint": g.get("footprint"), "height": g.get("height")}
            rec = {"frame_id": obs.frame_id, "camera": obs.camera, "time": stamp or time.time(), "marks": marks}
            _atomic_write(self.dir / "perception.json", json.dumps(rec, default=float).encode())
        except Exception as exc:  # the viewer is optional: never let it break the task
            print(f"[agent] live view update skipped: {exc!r}", flush=True)
