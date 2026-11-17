"""Review live smoke evidence. Never changes controller settings or judges image sharpness automatically."""
from pathlib import Path


def smoke_checks(row, events, root):
    executions = [e for e in events if e.get("kind")=="tool" and e.get("tool")=="execute"]
    steps = [s for e in executions for s in e.get("steps", [])]
    required = {"planned_s", "sim_elapsed_s", "wall_s", "real_time_factor"}
    timing_ok = bool(steps) and all(required <= set(s) and s["wall_s"] >= 0 and s["sim_elapsed_s"] >= 0
                                  and (s.get("kind")!="move" or {"motion_wall_s", "settling_wall_s", "settling_sim_s"} <= set(s))
                                  for s in steps)
    pics = [im for e in events if e.get("kind")=="model_input" and e.get("phase")=="placement"
            for im in e.get("images", []) if im and im.get("camera")=="wrist"]
    pictures = []
    for im in pics:
        path = Path(im["path"]) if im.get("path") else None
        if path is not None and not path.is_absolute():
            path = root/path
        pictures.append({"frame_id":im.get("frame_id"), "path":str(path) if path else None,
                         "exists":bool(path and path.is_file())})
    waits = [e for e in events if e.get("kind")=="timing" and e.get("category")=="settling" and e.get("label")=="wait_still"]
    timeout_events = [e for e in events if e.get("kind") in {"failure", "error"}
                      and any(w in str(e).lower() for w in ("timeout", "timed out", "clock stalled", "did not settle"))]
    mechanical = (timing_ok and bool(waits) and all(e.get("ok") for e in waits)
                  and bool(pictures) and all(p["exists"] for p in pictures)
                  and not timeout_events and all(e.get("ok") for e in executions)
                  and row.get("score", {}).get("task_success") is True)
    return {"automatic_checks_pass":mechanical, "motion_steps":len(steps), "timing_complete":timing_ok,
            "successful_settling_captures":sum(bool(w.get("ok")) for w in waits),
            "timeout_events":timeout_events, "pictures":pictures,
            "sharpness_and_stillness_review":"MANUAL: inspect listed wrist pictures; clear object/tray edges, open gripper, no mid-motion view"}


def table(rows):
    profiles = list(dict.fromkeys(r["profile"] for r in rows))
    out = ["| Profile | Scored/runs | Success | Wrong object | Wrong destination | False accept | False reject | Gemma s | Motion s | Settling s | Other s |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in profiles:
        group = [r for r in rows if r["profile"]==name]
        scored = [r for r in group if r.get("score", {}).get("scored")]
        counts = [sum(bool(r["score"].get(k)) for r in scored) for k in
                  ("task_success", "wrong_object", "wrong_destination", "false_accept", "false_reject")]
        times = []
        for r in group:
            t, ex = r["timing"], r["timing"].get("exclusive_s", {})
            g, m, s = (ex.get(k, 0.) for k in ("gemma", "motion", "settling"))
            times.append((g, m, s, max(0.,t["total_s"]-g-m-s)))
        means = [sum(t[i] for t in times)/len(times) for i in range(4)]
        out.append(f"| {name} | {len(scored)}/{len(group)} | " + " | ".join(map(str,counts)) + " | " +
                   " | ".join(f"{v:.1f}" for v in means)+" |")
    return "\n".join(out)+"\n\nTimes are mean wall seconds per run. Unscored runs remain visible in coverage.\n"
