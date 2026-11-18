#!/usr/bin/env python3
"""The live screen of v3/dialogue: v2's dashboard (used as a library, not changed) with a second camera panel, so
the table camera and the wrist camera are both visible all the time.

    +-----------------------------+----------------------+
    | main camera + outlines      | the other camera     |
    | ('c' swaps the two)         | (live, no outlines)  |
    +-----------------------------+----------------------+
    | task | state | target                              |
    | live agent logs                                    |
    +----------------------------------------------------+

Read-only like v2's dashboard: its own process, system Python and ROS 2, started by launch.sh.
    keys: q close | c swap the cameras | arrows PgUp PgDn End: logs
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v2" / "src"))
from agent import dashboard as d  # noqa: E402

SIDE = (d.MAIN[2] + 8, 0, 768, d.MAIN[3])  # x, y, w, h: right of the main camera, same height
INSET = (d.MAIN[2] - 328, 32, 320, 240)    # with the GPU column on, the right side is taken: a corner picture


class Screen(d.Dashboard):
    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        if self.gpumon_dir is None:
            self.width = SIDE[0] + SIDE[2]

    def draw_second(self, canvas, rect) -> None:
        x, y, w, h = rect
        cam = "table" if self.active_camera() == "wrist" else "wrist"
        with self.lock:
            frame = self.frames.get(cam)
        canvas[y:y + h, x:x + w] = (20, 20, 20)
        if frame is None:
            d._put(canvas, f"waiting for the {cam} camera ...", x + 20, y + h // 2, d.MUTED, 0.6, 1)
        else:
            canvas[y:y + h, x:x + w] = d.fit(frame[0], w, h)
            age = time.time() - frame[1]
            cv2.rectangle(canvas, (x, y), (x + 250, y + 24), (0, 0, 0), -1)
            d._put(canvas, f"{cam.upper()} CAMERA" + ("  live" if age < 2 else f"  last picture {age:.0f} s ago"), x + 8, y + 17, d.FG, 0.45)
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), d.LINE, 1)

    def render(self) -> np.ndarray:
        if self.gpumon_dir is not None:
            canvas = super().render()
            self.draw_second(canvas, INSET)
            return canvas
        canvas = np.full((d.H, self.width, 3), d.BG, np.uint8)
        self.draw_main(canvas)
        self.draw_second(canvas, SIDE)
        self.draw_task(canvas)
        _, y, _, h = d.LOGS
        self.draw_logs(canvas, y0=y + 4, y1=y + h - 4, x0=10, chars=int(125 * self.width / d.MAIN[2]))
        for ax, ay, aw, ah in (d.MAIN, (0, d.TASK[1], self.width, d.TASK[3]), (0, y, self.width, h)):
            cv2.rectangle(canvas, (ax, ay), (ax + aw - 1, ay + ah - 1), d.LINE, 1)
        self.draw_bar(canvas)
        return canvas


if __name__ == "__main__":
    d.Dashboard = Screen
    sys.exit(d.main())
