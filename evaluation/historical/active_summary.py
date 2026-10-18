"""Per-instruction metrics of active-perception sessions from the agent's own events (measured timestamps):
total time, Gemma calls / time, table and wrist observations, viewpoint moves, exploration planning time,
wrist-move time, safety events, outcome.

    python3 evaluation/historical/active_summary.py <run_dir> [<run_dir> ...] > out.json
"""

import json
import sys
from pathlib import Path


def per_instruction(d: Path) -> dict:
    ev = [json.loads(line) for line in (d / "events.jsonl").read_text().splitlines()]
    res = json.loads((d / "result.json").read_text())
    vlm = [e for e in ev if e["kind"] == "vlm"]
    per = [e for e in ev if e["kind"] == "tool" and e.get("tool") == "perceive"]
    expl = [e for e in ev if e["kind"] == "exploration"]
    moves = [e for e in ev if e["kind"] == "tool" and e.get("tool") == "execute" and e.get("label") in ("inspect view", "explore view")]
    parks = [e for e in ev if e["kind"] == "tool" and e.get("tool") == "execute" and e.get("label") == "park"]
    home = [e for e in ev if e["kind"] == "tool" and e.get("tool") == "execute" and e.get("label") == "return HOME"]
    return {"instruction": ev[0].get("text"), "status": res.get("status"), "outcome": res.get("outcome"),
            "total_s": round(ev[-1]["t"] - ev[0]["t"], 1), "plan": [s["capability"] for s in res.get("plan", [])],
            "gemma_calls": len(vlm), "gemma_s": round(sum(e.get("seconds", 0) for e in vlm), 1),
            "table_obs": sum(e["camera"] == "table" for e in per), "wrist_obs": sum(e["camera"] == "wrist" for e in per),
            "viewpoint_plans": len(expl), "viewpoint_plans_ok": sum(bool(e["plan"]["ok"]) for e in expl),
            "viewpoint_plan_s": round(sum(e.get("seconds", 0) for e in expl), 1),
            "viewpoint_moves": len(moves), "viewpoint_move_s": round(sum(e.get("seconds", 0) for e in moves), 1),
            "park_moves": len(parks), "home_moves": len(home), "home_move_s": round(sum(e.get("seconds", 0) for e in home), 1),
            "failed_moves": sum(1 for e in ev if e["kind"] == "tool" and e.get("tool") == "execute" and not e.get("ok")),
            "message": (res.get("message") or "")[:200]}


if __name__ == "__main__":
    rows = []
    for a in sys.argv[1:]:
        for d in sorted(Path(a).glob("[0-9][0-9]")):
            rows.append({"run": str(d), **per_instruction(d)})
    print(json.dumps(rows, indent=1))
