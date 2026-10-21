"""JSON-lines event log shared by the supervisor, the workflows and the tools (agent/events.py formats it)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


class EventLog:
    """JSON-lines event log: one object per line with time, kind and fields."""

    def __init__(self, path: Path, echo: bool = False, trace: bool = False) -> None:
        """echo: also print a readable line per event to the terminal (agent.events.format_event).
        trace: with echo, also print the agent handoffs that format_event leaves out (agent.events.format_trace)."""
        self.path, self.t0, self.echo, self.trace = Path(path), time.monotonic(), echo, trace
        self.f = open(self.path, "a", buffering=1)
        self.trail: list[str] = []  # tools and Gemma roles invoked, in order (WorkerResult.tools via mark / since)

    def __call__(self, kind: str, **fields) -> None:
        rec = {"t": round(time.monotonic() - self.t0, 3), "wall": time.strftime("%H:%M:%S"), "kind": kind, **fields}
        line = json.dumps(rec, default=_jsonable)
        self.f.write(line + "\n")
        name = {"tool": fields.get("tool"), "vlm": f"gemma:{fields.get('role')}" if fields.get("attempt") == 1 else None,
                "exploration": "viewpoint_planner", "verification": f"check:{fields.get('check')}"}.get(kind)
        if name:
            self.trail.append(name)
        if self.echo:
            from agent.events import format_event, format_trace
            e = json.loads(line)
            ev = format_event(e)
            if ev is None and self.trace:
                ev = format_trace(e)
            if ev is not None:
                if ev.gap:
                    print(flush=True)
                print(str(ev), flush=True)
                for d in ev.detail:
                    print(" " * 27 + d, flush=True)

    def mark(self) -> int:
        return len(self.trail)

    def since(self, mark: int) -> list[str]:
        """Tools and Gemma roles invoked since mark(), consecutive repeats folded as 'name x n'."""
        out: list[str] = []
        for name in self.trail[mark:]:
            if out and out[-1].split(" x ")[0] == name:
                n = int(out[-1].split(" x ")[1]) + 1 if " x " in out[-1] else 2
                out[-1] = f"{name} x {n}"
            else:
                out.append(name)
        return out

    def close(self) -> None:
        self.f.close()


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
