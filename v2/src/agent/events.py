"""One readable line per logged event (events.jsonl), for the dashboard's log panel. Pure Python: the dashboard runs
with the system Python.

    line = format_event(event)   # -> Line(wall, component, text, level) or None (events with nothing to show)

The loop's own messages ("say": master, workers, result) are shown as they were printed in the terminal; from the pick /
place skill the states, failures and sensor checks are shown.
"""

from __future__ import annotations

from dataclasses import dataclass

LEVELS = ("info", "warn", "error", "success", "question")


@dataclass
class Line:
    wall: str
    component: str
    text: str
    level: str = "info"

    def __str__(self) -> str:
        return f"[{self.wall}] [{self.component}]".ljust(26) + f" {self.text}"


def format_event(e: dict) -> Line | None:
    k, w = e.get("kind"), e.get("wall", "--:--:--")
    if k == "instruction":
        return Line(w, "USER", e.get("text", ""))
    if k == "say":
        text = e.get("text", "")
        level = ("success" if text.startswith("COMPLETED") else "error" if text.startswith(("FAILED", "STOPPED"))
                 else "error" if "-> failed" in text or "-> no:" in text else "info")
        return Line(w, e.get("who", "").upper(), text, level)
    if k == "state":
        return Line(w, "MANIPULATION", e.get("state", ""), "warn" if e.get("state") == "RECOVERING" else "info")
    if k == "failure":
        return Line(w, "MANIPULATION", f"failure {e.get('cls')}: {e.get('reason')}", "error")
    if k == "verification":
        return Line(w, "VERIFICATION", f"{e.get('check')}: {'pass' if e.get('pass') else 'FAIL'}"
                    + (f" (wrist camera, Gemma: {e['answer']})" if e.get("answer") else ""),
                    "success" if e.get("pass") else "error")
    if k == "worker" and e.get("worker") == "manipulation":  # words -> numbers, with what each number measures
        sizes = "; ".join(f"#{n}: {v}" for n, v in (e.get("chosen") or {}).items())
        answer = " / ".join(str(e.get("answer", "")).splitlines())
        return Line(w, "GROUNDING", f"\"{e.get('message', '')}\" -> {answer}" + (f"  ({sizes})" if sizes else ""))
    if k == "identity_check":
        head = f"{e.get('role')} #{e.get('object_id')}"
        if e.get("ok") is None:
            return Line(w, "IDENTITY", f"{head}: not checked ({e.get('reason', '')})", "warn")
        scores = ", ".join(f"{n} {p:.2f}" for n, p in list((e.get("scores") or {}).items())[:3])
        return Line(w, "IDENTITY", f"{head} {e.get('expected')}: "
                    + ("pass" if e["ok"] else f"FAIL, looks like {e.get('top')}") + f" ({scores})",
                    "success" if e["ok"] else "error")
    if k == "catalogue":
        return Line(w, "CATALOGUE", "; ".join(f"#{i} {n}" for i, n in (e.get("named") or {}).items()))
    if k == "error":
        return Line(w, "AGENT", f"internal error: {e.get('error')}", "error")
    return None
