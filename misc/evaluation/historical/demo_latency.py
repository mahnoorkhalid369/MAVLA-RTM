"""Latency breakdown of agent runs from their events.jsonl / result.json (measured timestamps only).

    python3 evaluation/historical/demo_latency.py out/demo/2026-10-14/live_144109/01 [more run dirs ...] [--json out.json]

Per run: total, time per LangGraph state, VLM calls (count / seconds per role), describe time, perception
(observation) count and time, human-wait time (question -> answer), planning, execution per motion
label, verification, HOME. Times come from the event 't' field (seconds since the run started).
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def analyse(run_dir: Path) -> dict:
    ev = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    res = json.loads((run_dir / "result.json").read_text()) if (run_dir / "result.json").exists() else {}
    out: dict = {"run": str(run_dir), "instruction": ev[0].get("text"), "status": res.get("status"),
                 "total_s": round(ev[-1]["t"] - ev[0]["t"], 1)}
    # time per state: from one state event to the next state event
    states = [(e["t"], e["state"]) for e in ev if e["kind"] == "state"]
    per_state = collections.defaultdict(float)
    for (t0, s), (t1, _) in zip(states, states[1:] + [(ev[-1]["t"], "END")]):
        per_state[s] += t1 - t0
    out["states_s"] = {k: round(v, 1) for k, v in per_state.items()}
    vlm = collections.defaultdict(lambda: [0, 0.0, 0])
    for e in ev:
        if e["kind"] == "vlm":
            v = vlm[e["role"]]
            v[0] += 1
            v[1] += e.get("seconds", 0.0)
            v[2] += 0 if e.get("valid") else 1
    out["vlm"] = {r: {"calls": c, "seconds": round(s, 1), "invalid": i} for r, (c, s, i) in vlm.items()}
    out["vlm_total"] = {"calls": sum(v[0] for v in vlm.values()), "seconds": round(sum(v[1] for v in vlm.values()), 1)}
    # observations: perceive events (YOLO + depth geometry); duration = from the previous event (park done) to it
    obs = []
    for i, e in enumerate(ev):
        if e["kind"] == "tool" and e.get("tool") == "perceive":
            prev = ev[i - 1]["t"] if i else e["t"]
            obs.append({"frame": e["frame_id"], "detections": e["detections"], "objects": e["objects"],
                        "seconds_since_previous_event": round(e["t"] - prev, 2)})
    out["observations"] = obs
    # human wait: question event -> ask_user event
    waits, q = [], None
    for e in ev:
        if e["kind"] == "question":
            q = e["t"]
        elif e["kind"] == "ask_user" and q is not None:
            waits.append(round(e["t"] - q, 2))
            q = None
    out["human_wait_s"] = round(sum(waits), 2)
    # motions: execute events
    mot = collections.defaultdict(float)
    for e in ev:
        if e["kind"] == "tool" and e.get("tool") == "execute":
            mot[e["label"]] += e.get("seconds", 0.0)
    out["execution_s"] = {k: round(v, 1) for k, v in mot.items()}
    out["execution_total_s"] = round(sum(mot.values()), 1)
    planned = [st.get("planned_s", 0) for e in ev if e["kind"] == "tool" and e.get("tool") == "execute" for st in e.get("steps", [])]
    out["planned_motion_s"] = round(sum(planned), 1)
    out["automated_s"] = round(out["total_s"] - out["human_wait_s"], 1)
    return out


def main(args: list[str]) -> int:
    js = None
    if "--json" in args:
        i = args.index("--json")
        js, args = args[i + 1], args[:i] + args[i + 2:]
    rows = [analyse(Path(a)) for a in args]
    for r in rows:
        print(json.dumps(r, indent=1))
    if js:
        Path(js).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
