"""JSON-lines event log shared by the supervisor, the workflows and the tools (agent/events.py formats it)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


class EventLog:
    """JSON-lines event log: one object per line with time, kind and fields."""

    def __init__(self, path: Path, echo: bool = False) -> None:
        """echo: also print a readable line per event to the terminal (agent.events.format_event)."""
        self.path, self.t0, self.echo = Path(path), time.monotonic(), echo
        self.f = open(self.path, "a", buffering=1)

    def __call__(self, kind: str, **fields) -> None:
        rec = {"t": round(time.monotonic() - self.t0, 3), "wall": time.strftime("%H:%M:%S"), "kind": kind, **fields}
        line = json.dumps(rec, default=_jsonable)
        self.f.write(line + "\n")
        if self.echo:
            from agent.events import format_event
            ev = format_event(json.loads(line))
            if ev is not None:
                print(str(ev), flush=True)

    def close(self) -> None:
        self.f.close()


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
