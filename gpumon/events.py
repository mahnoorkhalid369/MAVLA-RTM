"""Follows the agent's events.jsonl traces and copies every event into <out>/events.jsonl with two added fields:

    arrival  Unix time at which gpumon read the line (a few ms after it was written, at the poll rate below)
    file     the trace it came from (one events.jsonl per instruction, each with its own t = 0)

The agent's own 't' is a precise monotonic clock but starts at an unknown Unix time, and its 'wall' has 1 s
resolution. report.align() therefore anchors each file with offset = min(arrival - t): the event read with the
least delay fixes the offset, and every event keeps its precise relative time.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class EventFollower(threading.Thread):
    def __init__(self, roots: list[Path], out_dir: Path, poll_s: float = 0.05, since: float | None = None) -> None:
        """roots: folders searched (recursively) for events.jsonl files; only files modified after `since`
        (default: now) are followed, so traces of earlier sessions are ignored."""
        super().__init__(name="gpumon-events", daemon=True)
        self.roots, self.poll_s = [Path(r) for r in roots], poll_s
        self.since = time.time() - 1.0 if since is None else since
        self.out = open(Path(out_dir) / "events.jsonl", "a", buffering=1)
        self.offsets: dict[Path, int] = {}
        self.partial: dict[Path, str] = {}
        self._halt = threading.Event()
        self._next_scan = 0.0
        self.count = 0

    def stop(self) -> None:
        self._halt.set()
        self.join(5)
        self._read_all()  # whatever was written after the last poll
        self.out.close()

    def _scan(self) -> None:
        for root in self.roots:
            if not root.exists():
                continue
            for f in root.rglob("events.jsonl"):
                if f not in self.offsets:
                    try:
                        if f.stat().st_mtime >= self.since:
                            self.offsets[f] = 0
                    except FileNotFoundError:
                        pass

    def _read_all(self) -> None:
        now = time.time()
        for f, off in list(self.offsets.items()):
            try:
                with open(f, "rb") as fh:
                    fh.seek(off)
                    data = fh.read()
            except FileNotFoundError:
                continue
            if not data:
                continue
            self.offsets[f] = off + len(data)
            text = self.partial.pop(f, "") + data.decode("utf-8", "replace")
            lines = text.split("\n")
            if lines[-1]:  # incomplete last line: keep for the next poll
                self.partial[f] = lines[-1]
            for line in lines[:-1]:
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                e["arrival"], e["file"] = round(now, 3), str(f)
                self.out.write(json.dumps(e) + "\n")
                self.count += 1

    def run(self) -> None:
        while not self._halt.is_set():
            if time.monotonic() >= self._next_scan:  # new instruction folders appear during a session
                self._scan()
                self._next_scan = time.monotonic() + 1.0
            self._read_all()
            self._halt.wait(self.poll_s)
