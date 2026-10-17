"""Offline check of the goal planner (real Gemma, no sim): expected capability sequence and pick_place arguments
per instruction, validity after the one retry, latency; and the old parse call's latency for comparison.

    .venv-seg/bin/python -m evaluation.planner_check --out out/perf/planner_check.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = [  # (instruction, expected capabilities, expected pick_place (object, relation, reference) words)
    ("Put the red cube on the green cube.", ["pick_place"], [("red cube", "on", "green cube")]),
    ("Pick up the yellow cube and put it on the tan box.", ["pick_place"], [("yellow cube", "on", "tan box")]),
    ("Put the orange cube on the purple cylinder.", ["pick_place"], [("orange cube", "on", "purple cylinder")]),
    ("Put the pink cube next to the white tall cylinder.", ["pick_place"], [("pink cube", "next_to", "white tall cylinder")]),
    ("Take a look at your surroundings.", ["observe", "describe_scene"], []),
    ("What objects do you see on the table?", ["observe", "describe_scene"], []),
    ("How many cubes are on the table?", ["observe", "answer"], []),
    ("Put the red cube on the green cube, then put the orange cube on the purple cylinder.", ["pick_place", "pick_place"],
     [("red cube", "on", "green cube"), ("orange cube", "on", "purple cylinder")]),
    ("Go back to your home position.", ["return_home"], []),
    ("Where is the red cube?", ["observe", "locate"], []),
    ("Find the banana.", "find", []),
    ("Sing me a song.", [], []),
]


def norm(t):
    return " ".join(str(t or "").lower().replace("the ", "").split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from agent.capabilities import planner_menu
    from agent.vlm_roles import GemmaVLM, Roles, VLMFailure
    events = []
    roles = Roles(GemmaVLM(), lambda k, **f: events.append({"kind": k, **f}))
    rows = []
    for text, caps, pps in CASES:
        events.clear()
        t0 = time.monotonic()
        try:
            p = roles.plan_goal(text, planner_menu())
            from agent.planning import complete_plan
            raw = [s.capability for s in p.steps]
            complete_plan(p, text)
            got = [s.capability for s in p.steps]
            got_pp = [(norm(s.args["object"]["text"]), s.args["relation"], norm(s.args["reference"]["text"]))
                      for s in p.steps if s.capability == "pick_place"]
            err = None
        except VLMFailure as exc:
            p, got, got_pp, err, raw = None, None, [], str(exc), None
        dt = time.monotonic() - t0
        if caps == "find":  # locate (fails honestly if not visible), explore (unsupported) or nothing
            ok = got is not None and got in ([], ["explore"], ["observe", "explore"], ["observe", "locate"], ["locate"])
        else:
            ok = got == caps and got_pp == [(norm(o), r, norm(f)) for o, r, f in pps]
        if caps == ["observe", "describe_scene"] and got == ["describe_scene"]:
            ok = True  # the supervisor observes automatically before describe_scene
        rows.append({"instruction": text, "expected": caps, "got": got, "planner_raw": raw if p else None, "pick_place": got_pp, "ok": ok, "seconds": round(dt, 2),
                     "attempts": len([e for e in events if e["kind"] == "vlm"]), "error": err,
                     "reason": p.reason if p else None})
        print(("OK  " if ok else "BAD ") + f"{dt:5.1f}s {text!r} -> {got} {got_pp or ''} {p.reason[:60] if p and not got else ''}", flush=True)
    t_parse = []
    for text, _, pps in CASES[:4]:
        t0 = time.monotonic()
        roles.parse(text)
        t_parse.append(round(time.monotonic() - t0, 2))
    rep = {"rows": rows, "ok": sum(r["ok"] for r in rows), "n": len(rows),
           "plan_seconds_pick_place": [r["seconds"] for r in rows[:4]], "old_parse_seconds_pick_place": t_parse}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print("ok", rep["ok"], "/", rep["n"], "| plan s (4 pick_place)", rep["plan_seconds_pick_place"], "| old parse s", t_parse)
    return 0


if __name__ == "__main__":
    sys.exit(main())
