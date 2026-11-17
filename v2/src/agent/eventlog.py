"""The event log of one instruction: one JSON object per line (time, kind, fields) in <run dir>/events.jsonl.

    log = EventLog(path); log("master", call="plan", answer="..."); log.close()

The pick / place skill's progress (state, failure, recovery) is also printed, indented, so the terminal shows what the
arm is doing; everything else is printed by the loop (agent/loop.py) or only kept in the file.
"""

from __future__ import annotations

import json
import time
import threading
from pathlib import Path

import numpy as np

SHOWN = {"state": lambda e: e["state"],
         "failure": lambda e: f"failure {e['cls']}: {e['reason']} (retry {e['used'] + 1} of {e['budget']})"
                              if e["used"] < e["budget"] else f"failure {e['cls']}: {e['reason']}",
         "verification": lambda e: f"check {e.get('check')}: {'pass' if e.get('pass') else 'FAIL'}"
                                   + (f" (wrist camera, Gemma: {e['answer']})" if e.get("answer") else ""),
         "worker": lambda e: grounding_text(e),
         "identity_check": lambda e: identity_text(e),
         "source_confirmation": lambda e: f"source #{e['object_id']} Gemma confirmation: {e['verdict']} ({e['reason']})",
         "plan": lambda e: f"preflight {e['mode']}: grasp {e['grasp']}; placement centre mm {e['target_centre_mm']}; "
                           f"relation {e['relation']}; return HOME preplanned={e['return_home_preplanned']}",
         "tool": lambda e: tool_text(e),
         "identity_retry": lambda e: f"choosing again: {e.get('reason', '')}",
         "catalogue": lambda e: "named: " + "; ".join(f"#{i} {n} ({e['sizes'][i]})" for i, n in e["named"].items())}


def tool_text(e):
    if e.get("tool") == "source_side_binding":
        return f"source #{e['object_id']}: side frame {e['frame_id']} matches local marks {e['matched_marks']}"
    if e.get("tool") == "placement_search":
        if e.get("collision_free"):
            return (f"placement {e['relation']} #{e['reference_id']}: free position mm {e['chosen_mm']}, "
                    f"yaw {e['yaw_deg']:.1f}; complete carry/lower/retreat collision-checked")
        return f"placement #{e['reference_id']}: search exhausted ({e.get('candidates', 0)} free candidates)"
    return None


def grounding_text(e: dict) -> str | None:
    """Words -> numbers: 'pick up the apple' -> 'pick #8' (#8: 64 x 64 mm, 70 mm tall)."""
    if e.get("worker") != "manipulation":
        return None
    sizes = "; ".join(f"#{k}: {v}" for k, v in (e.get("chosen") or {}).items())
    return f"grounding: \"{e.get('message', '')}\" -> {' / '.join(str(e.get('answer', '')).splitlines())}" + \
        (f"  ({sizes})" if sizes else "")


def identity_text(e: dict) -> str:
    """SigLIP 2 on the wrist close-up: 'identity source #8 apple: pass (apple 0.97, milk carton 0.02)'."""
    head = f"identity {e.get('role')} #{e.get('object_id')}"
    if e.get("ok") is None:
        return f"{head}: not checked ({e.get('reason', '')})"
    scores = ", ".join(f"{n} {p:.2f}" for n, p in list((e.get("scores") or {}).items())[:3])
    return f"{head} {e.get('expected')}: {'pass' if e['ok'] else 'FAIL, looks like ' + str(e.get('top'))} ({scores})"


class EventLog:
    def __init__(self, path: Path, echo: bool = True) -> None:
        self.path, self.t0, self.echo = Path(path), time.monotonic(), echo
        self.f = open(self.path, "a", buffering=1)
        self.lock = threading.Lock()

    def __call__(self, kind: str, **fields) -> None:
        now = time.monotonic()
        rec = {"t": round(now - self.t0, 3), "monotonic_s": now,
               "wall": time.strftime("%H:%M:%S"), "kind": kind, **fields}
        with self.lock:
            self.f.write(json.dumps(rec, default=_jsonable) + "\n")
        text = SHOWN[kind](rec) if self.echo and kind in SHOWN else None
        if text:
            print(f"[{rec['wall']}] {'':15}{text}", flush=True)

    def close(self) -> None:
        self.f.close()


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
