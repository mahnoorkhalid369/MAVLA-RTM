"""Which process is which: GPU users are grouped by their command line (no PID is passed in by the user)."""

from __future__ import annotations

import re
from pathlib import Path

import psutil

# group name -> command-line pattern. First match wins; everything else on the GPU is "other" (desktop, viewers).
GROUPS = (
    ("sim", re.compile(r"-m sim\.run_(live|scenario)|isaacsim|kit/kit|omni\.isaac")),
    ("agent", re.compile(r"-m agent\.run|-m evaluation\.lab_exp1_agent|agent/run\.py")),
)


def _match(cmdline: list[str]) -> str:
    """Group of a command line. Only Python interpreters count (Isaac Sim and the agent both run as `python -m ...`),
    so a shell, editor or grep whose arguments merely mention these modules is never mistaken for them."""
    if not cmdline or "python" not in Path(cmdline[0]).name:
        return "other"
    cmd = " ".join(cmdline)
    return next((g for g, rx in GROUPS if rx.search(cmd)), "other")


def classify(pid: int, cache: dict[int, str]) -> str:
    """Group of a process ('sim', 'agent' or 'other'); cached per PID (a dead PID keeps its last group)."""
    if pid in cache:
        return cache[pid]
    try:
        cmdline = psutil.Process(pid).cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return "other"  # not cached: it may be readable on the next sample
    group = _match(cmdline)
    cache[pid] = group
    return group


def find(group: str) -> list[psutil.Process]:
    """Running processes of a group (used for the RAM columns, including processes not yet on the GPU)."""
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if _match(p.info["cmdline"] or []) == group:
                out.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return out
