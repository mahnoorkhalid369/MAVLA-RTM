"""Fixed kitchen tasks, independent geometry scoring, and non-overlapping timing totals."""

from collections import defaultdict
import math
import numpy as np

SEEDS = (7, 42, 123)
TASKS = (
    ("apple", "Put the apple in the blue serving tray.", ("apple",), False),
    ("water", "Put the water bottle in the blue serving tray.", ("water_bottle",), False),
    ("cheezit", "Put the cheezit in the blue serving tray.", ("food_003_cracker_box",), False),
    ("milk_alias", "Put the milk bottle in the blue serving tray.", ("milk",), False),
    ("two_items", "Put the apple and the water bottle in the blue serving tray. Leave the pudding on the counter.",
     ("apple", "water_bottle"), False),
    ("basket", "Put the apple in the brown storage basket.", ("apple",), False),
    ("impossible", "Put the banana in the blue serving tray.", (), True),
)
# One factor at a time after deduplication. No untested inference/motion change is the default.
PROFILES = {
    "baseline": {"V2_BASELINE": "1"},
    "deduplicated": {"V2_REUSE_PLACEMENT": "1", "V2_FINISH_PLAN": "1"},
    "headless": {"V2_REUSE_PLACEMENT": "1", "V2_FINISH_PLAN": "1"},
    "no_thinking": {"V2_PLACEMENT_THINKING": "0"},
    "wrist_560": {"V2_WRIST_IMAGE_TOKENS": "560"},
    "reasoning_1024": {"V2_REASONING_TOKENS": "1024"},
    "prefix_cache": {"V2_PREFIX_CACHE": "1"},
    "overlap_home": {"V2_OVERLAP_HOME": "1"},
    "chain": {"V2_CHAIN_TRANSFERS": "1"},
    "carry_125": {"V2_CARRY_SPEED_SCALE": "1.25"},
}
PROFILE_REFERENCE = {"deduplicated": "baseline", "headless": "deduplicated",
                     **{name: "headless" for name in PROFILES if name not in {"baseline", "deduplicated", "headless"}}}


def cases(suite="core"):
    core = [{"id": f"{difficulty}_s{seed}_{name}", "scenario": "snack_kitchen", "difficulty": difficulty,
             "seed": seed, "instruction": text, "objects": list(objects),
             "destination": "storage_basket" if name == "basket" else "serving_tray", "impossible": impossible}
            for difficulty in ("easy", "medium", "hard") for seed in SEEDS
            for name, text, objects, impossible in TASKS]
    regressions = []
    for difficulty in ("easy", "medium"):
        common = {"scenario":"snack_kitchen", "difficulty":difficulty, "seed":42,
                  "destination":"serving_tray", "impossible":False}
        for name, request, objects, extra in (
            ("on_the_tray", "Put the apple on the tray.", ["apple"], {"answers":["the blue serving tray"]}),
            ("occupied_tray", "Put the cheezit in the blue serving tray.", ["food_003_cracker_box"],
             {"setup":{"preplace":{"apple":"serving_tray"}}, "preserve":["apple"]}),
            ("milk_session", "Put the milk bottle on the tray.", ["milk", "food_003_cracker_box"],
             {"instructions":[{"request":"Put the milk bottle on the tray.", "answers":["the blue serving tray"]},
                              {"request":"Put the cheezit on that same tray. Leave the milk bottle where you placed it.",
                               "answers":["the blue serving tray"]}]}),
            ("cheezit_back", "Put the cheezit in the blue serving tray.", ["food_003_cracker_box"],
             {"setup":{"rotate_yaw_deg":{"food_003_cracker_box":180}},
              "view_requirement":"Confirm the saved source wrist crop shows the back of the cracker box; yaw alone does not prove it."}),
        ):
            regressions.append({**common, "id":f"{difficulty}_s42_{name}", "instruction":request,
                                "objects":objects, **extra})
    return {"core":core, "regressions":regressions, "all":core+regressions}[suite]


def rotation(q):
    q = np.asarray(q, float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("invalid ground-truth quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def in_destination(obj, pose, region):
    """Full collision footprint inside the tray and supported at its floor; no language verdict used."""
    if pose is None:
        return False
    p = np.asarray(pose["position"], float)
    if not np.isfinite(p).all():
        return False
    R = rotation(pose["quat_wxyz"])
    # Scenarios use rigid-body origins at the object's support plane. Include tilt in the full body's bounds.
    half = np.array([*obj.footprint, obj.height]) / 2
    centre = p + R @ np.array([0., 0., obj.height/2])
    a = math.radians(region.yaw_deg)
    D = np.array([[math.cos(a), math.sin(a), 0], [-math.sin(a), math.cos(a), 0], [0, 0, 1.]])
    delta = D @ (centre - np.asarray(region.centre))
    extent = np.abs(D @ R) @ half
    bottom = centre[2] - (np.abs(R) @ half)[2]
    return bool(np.all(np.abs(delta[:2]) + extent[:2] <= np.asarray(region.size[:2])/2 + .002)
                and abs(bottom-region.centre[2]) <= .012)


def score(case, scenario, initial, final, result, stable=True, moved_during=()):
    """Unknown evidence remains unscored, rather than being counted as a success or failure."""
    objects = {o.id: o for o in scenario.objects}
    required = set(objects)
    if not required.issubset(initial) or not required.issubset(final) or not stable:
        return {"scored": False, "reason": "missing or unstable ground truth"}
    containers = {r.id: r for r in scenario.regions if r.id in {"serving_tray", "storage_basket"}}
    requested = set(case["objects"])
    if any(k not in objects or not in_destination(objects[k], initial.get(k), containers[case["destination"]])
           for k in case.get("preserve", [])):
        return {"scored":False, "reason":"occupied-tray setup was not observed supported inside the destination"}
    moved = {k for k in required if np.linalg.norm(np.asarray(final[k]["position"])-initial[k]["position"]) > .025}
    moved.update(set(moved_during) & required)
    wrong_object = bool(moved-requested)
    wrong_destination = any(in_destination(objects[k], final[k], r)
                            for k in requested for name, r in containers.items() if name != case["destination"])
    placed = all(in_destination(objects[k], final[k], containers[case["destination"]]) for k in requested)
    placed &= all(in_destination(objects[k], final[k], containers[case["destination"]]) for k in case.get("preserve", []))
    goal_holds = placed and not wrong_object and not case["impossible"]
    accepted = result.get("status") == "COMPLETED"
    return {"scored": True, "task_success": (result.get("status") == "STOPPED" and not moved) if case["impossible"] else goal_holds and accepted,
            "goal_holds": goal_holds, "wrong_object": wrong_object, "wrong_destination": wrong_destination,
            "false_accept": accepted and not goal_holds, "false_reject": not accepted and goal_holds,
            "moved_objects": sorted(moved), "status": result.get("status")}


def timing_summary(events):
    """Sweep interval boundaries so nested spans and overlapped inference/motion never double count wall time."""
    spans = [e for e in events if e.get("kind") == "timing" and e.get("category") != "skill"]
    instrumented = bool(spans)
    if not spans:
        # Historical logs can locate model and execute wall intervals, but cannot
        # separate preprocessing, capture or simulated execution from waits.
        inputs = {e["call_id"]:e for e in events if e.get("kind") == "model_input" and "call_id" in e}
        for e in events:
            if e.get("kind") == "model_output" and e.get("call_id") in inputs:
                spans.append({"category": "gemma", "start_s": inputs[e["call_id"]]["t"], "end_s": e["t"]})
            elif e.get("kind") == "tool" and e.get("tool") == "execute" and "seconds" in e:
                spans.append({"category": "motion", "start_s": max(0.,e["t"]-e["seconds"]), "end_s": e["t"]})
    totals = defaultdict(float)
    states = defaultdict(lambda: defaultdict(float))
    edges = sorted({float(e[k]) for e in spans for k in ("start_s", "end_s")})
    for a, b in zip(edges, edges[1:]):
        active = [e for e in spans if e["start_s"] <= a and e["end_s"] >= b]
        if not active:
            continue
        categories = {e["category"] for e in active}
        if "gemma" in categories and "motion" in categories:
            category = "gemma+settling" if "settling" in categories else "gemma+motion"
        else:
            category = min(active, key=lambda e: e["end_s"]-e["start_s"])["category"]
        totals[category] += b-a
        state = next((e.get("state") for e in active if e.get("state")), "outside_skill")
        states[state][category] += b-a
    motions = [s for e in events if e.get("kind") == "tool" and e.get("tool") == "execute" for s in e.get("steps", [])]
    models = [e for e in events if e.get("kind") == "model_output"]
    elapsed = max((e.get("t", 0.) for e in events), default=0.)
    measured = sum(totals.values())
    return {"total_s": elapsed, "exclusive_s": dict(totals), "other_s": max(0., elapsed-measured),
            "by_state_s": {k: dict(v) for k, v in states.items()}, "motions": motions,
            "model_calls": len(models), "model_metrics": [{k: e.get(k) for k in
            ("call_id", "phase", "input_tokens", "image_count", "thinking_tokens", "generated_tokens", "seconds", "budget_exhausted")}
             for e in models], "instrumented": instrumented,
            "timing_source": "instrumented_spans" if instrumented else "legacy_wall_intervals",
            "simulation_timing_available": bool(motions) and all("sim_elapsed_s" in m for m in motions)}


def compare(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["profile"]].append(row)
    out = {}
    metrics = ("task_success", "wrong_object", "wrong_destination", "false_accept", "false_reject")
    for profile, records in grouped.items():
        reference = PROFILE_REFERENCE.get(profile, "baseline")
        baseline = {r["case_id"]: r for r in grouped.get(reference, []) if r.get("score", {}).get("scored")}
        scored = [r for r in records if r.get("score", {}).get("scored")]
        paired = [(r, baseline[r["case_id"]]) for r in scored if r["case_id"] in baseline]
        paired_clean = bool(paired) and all(
            all(not r["score"][k] or b["score"][k] for k in metrics[1:])
            and (r["score"]["task_success"] or not b["score"]["task_success"]) for r,b in paired)
        out[profile] = {"runs": len(records), "scored": len(scored), "reference": reference,
                        "counts": {k: sum(bool(r["score"][k]) for r in scored) for k in metrics},
                        "mean_s": sum(r["timing"]["total_s"] for r in scored)/len(scored) if scored else None,
                        "paired_cases": len(paired), "no_observed_paired_regression": paired_clean,
                        "fully_paired": bool(paired) and len(paired) == len(records),
                        "paired_mean_saving_s": sum(b["timing"]["total_s"]-r["timing"]["total_s"] for r,b in paired)/len(paired) if paired else None,
                        "note": "Finite test-set result, not a guarantee of equal accuracy."}
    return out


def placement_labels(events, samples, scenario):
    """Label saved wrist checks from contemporaneous simulator poses, outside the agent.

    Bind measured IDs to ground-truth bodies only when nearest-position association is unique.
    Ambiguous or missing associations stay unlabelled, never guessed.
    """
    items = {o.id:o for o in scenario.objects}
    regions = {r.id:r for r in scenario.regions if r.id in {"serving_tray", "storage_basket"}}
    bindings, destinations, out = {}, {}, []
    def nearest(xy, candidates):
        ranked = sorted((float(np.linalg.norm(np.asarray(p)[:2]-xy)), key) for key,p in candidates.items())
        return ranked[0][1] if ranked and ranked[0][0] <= .06 and (len(ranked)==1 or ranked[1][0]-ranked[0][0] > .02) else None
    for e in events:
        if e.get("kind") == "grounded_action":
            if not samples or "monotonic_s" not in e:
                continue
            at = min(samples, key=lambda s: abs(s["wall_s"]-e["monotonic_s"]))
            if abs(at["wall_s"]-e["monotonic_s"]) > 2:
                continue
            g = e.get("object_geometry")
            if g and e.get("object_id") is not None:
                bindings[e["object_id"]] = nearest(np.asarray(g["centre"])[:2], {k:v["position"] for k,v in at["objects"].items()})
            g = e.get("destination_geometry")
            if g and (e.get("destination") or "").startswith("#"):
                destinations[int(e["destination"][1:])] = nearest(np.asarray(g["centre"])[:2], {k:r.centre for k,r in regions.items()})
        if e.get("kind") != "placement_check":
            continue
        key, dest = bindings.get(e.get("object_id")), destinations.get(e.get("reference_id"))
        at = min(samples, key=lambda s: abs(s["sim_time"]-e["sim_time"])) if samples and e.get("sim_time") is not None else None
        label = None
        near = [s for s in samples if e.get("sim_time") is not None and abs(s["sim_time"]-e["sim_time"]) <= .3]
        stationary = (key is not None and len(near) >= 2 and near[-1]["sim_time"]-near[0]["sim_time"] >= .2
                      and all(key in s["objects"] and np.linalg.norm(np.asarray(s["objects"][key]["position"])
                              - at["objects"][key]["position"]) < .005 for s in near)) if at and key in at["objects"] else False
        if (key in items and dest in regions and at is not None and stationary and abs(at["sim_time"]-e["sim_time"]) <= .15
                and e.get("relation") in {"in", "on"}):
            label = in_destination(items[key], at["objects"].get(key), regions[dest])
        out.append({"frame_id": e["frame_id"], "path": e.get("path"), "prompt": e.get("prompt"), "label": label,
                    "source_key": key, "destination_key": dest, "recorded_word": e["word"],
                    "sim_time": e.get("sim_time")})
    return out
