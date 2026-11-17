"""Offline design checks of a generated scenario (no simulator): interpenetration at spawn, workspace and
keep-clear zones, table-camera visibility of each object's top (ray casting against every collider AND the robot
at HOME), top-down grasp and placement feasibility with the project's own planner + collision checker (control/,
unchanged), and wrist-camera viewpoint feasibility with the existing viewpoint planner (perception/viewpoints.py,
unchanged, used read-only).

Intent rules (problems if violated): an object the table camera cannot see at all must have role "occluded";
an object no planned grasp reaches must have role "unreachable"; every free receiving region must accept a
held probe container."""

from __future__ import annotations

import numpy as np

from scenarios.core import (DECOR_MAX_REACH_Z, DECOR_MIN_RADIUS, KEEP_CLEAR, REACH_MAX, Scenario, overlaps, scenario_parts,
                                   world_parts)

__all__ = ["grasp_feasibility", "placement_feasibility", "static_report", "top_visibility", "view_feasibility"]
PROBES = {"tube": (0.034, 0.114), "vial": (0.026, 0.060), "any": (0.034, 0.114),
          "instrument": (0.012, 0.026)}  # held probe: width, height


def _ray_hits(p0, p1, wp, pad=0.0) -> bool:
    a, b = wp.R.T @ (p0 - wp.centre), wp.R.T @ (p1 - wp.centre)
    h, t0, t1, d = wp.half + pad, 0.0, 1.0, b - a
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if abs(a[i]) > h[i]:
                return False
            continue
        u, v = (-h[i] - a[i]) / d[i], (h[i] - a[i]) / d[i]
        t0, t1 = max(t0, min(u, v)), min(t1, max(u, v))
        if t0 > t1:
            return False
    return t1 > 1e-4 and t0 < 0.999


def _robot_spheres():
    from control.collision import CollisionChecker
    from control.robot import ARM_HOME
    c, r, _, _ = CollisionChecker([]).spheres(np.asarray(ARM_HOME, float))
    return c, r


def _ray_hits_spheres(p0, p1, c, r) -> bool:
    d = p1 - p0
    t = np.clip(((c - p0) @ d) / (d @ d), 0.0, 1.0)
    return bool(np.any(np.linalg.norm(p0 + t[:, None] * d - c, axis=1) < r))


def top_visibility(sc: Scenario, camera_xyz=None, robot: bool = True) -> dict:
    """Fraction of 9 points on each object's top face that the table camera sees (other colliders and, by
    default, the robot itself at HOME block the view)."""
    from control.robot import table_camera_pose
    if camera_xyz is None and sc.view.get("table_camera"):  # the scenario mounts its own table camera
        camera_xyz = sc.view["table_camera"][0]
    cam = np.asarray(camera_xyz if camera_xyz is not None else table_camera_pose()[:3, 3], float)
    fx, ob = scenario_parts(sc)
    rc, rr = _robot_spheres() if robot else (np.zeros((0, 3)), np.zeros(0))
    out = {}
    for o in sc.objects:
        top = max((wp for wp in ob if wp.owner == o.id), key=lambda wp: wp.centre[2] + wp.half[2])
        r = min(top.half[:2]) * 0.8
        pts = [top.centre + top.R @ np.array([x * r, y * r, top.half[2] + 0.001]) for x in (-1, 0, 1) for y in (-1, 0, 1)]
        blockers = [wp for wp in fx + ob if wp.owner != o.id]
        seen = [not (any(_ray_hits(p, cam, b) for b in blockers) or _ray_hits_spheres(p, cam, rc, rr)) for p in pts]
        out[o.id] = round(sum(seen) / len(seen), 2)
    return out


def _planner(sc: Scenario):
    """SkillPlanner with every fixture and object as obstacles, ONE name per fixture / object (as the agent
    perceives a rack or a tray as one object); returns (planner, fixture parts)."""
    from control.collision import Box, CollisionChecker
    from control.kinematics import Kinematics
    from control.skills import SkillPlanner

    fx, ob = scenario_parts(sc)
    boxes = [Box(wp.owner, wp.centre, wp.R, wp.half) for wp in fx + ob]
    return SkillPlanner(Kinematics(), collision=CollisionChecker(boxes))


def grasp_width(o, yaw: float) -> float:
    """Object size across the fingers at grip yaw (fingers close along world (-sin yaw, cos yaw)), measured where
    the pads close (within GRIP_BAND of the object's centre along them), as perception measures it."""
    from perception.grounding import GRIP_BAND
    colliders = [p for p in o.parts if p.collide]
    if all(p.shape == "cylinder" for p in colliders):
        return min(o.footprint)
    a = np.radians(yaw)
    across, along = np.array([-np.sin(a), np.cos(a)]), np.array([np.cos(a), np.sin(a)])
    lo, hi = np.inf, -np.inf
    for wp in world_parts(o.id, (0.0, 0.0, 0.0), o.yaw_deg, colliders):  # centred on the object
        corners = np.array([wp.centre[:2] + wp.R[:2, :2] @ (sx * wp.half[0], sy * wp.half[1])
                            for sx in (-1, 1) for sy in (-1, 1)])
        if (corners @ along).max() >= -GRIP_BAND and (corners @ along).min() <= GRIP_BAND:  # under the pads
            lo, hi = min(lo, (corners @ across).min()), max(hi, (corners @ across).max())
    return float(hi - lo) if hi > lo else min(o.footprint)


def grasp_feasibility(sc: Scenario, yaws=(0.0, 45.0, 90.0, 135.0)) -> dict:
    """Per object: the grip yaws for which a collision-checked top-down pick plans from HOME
    (control.skills.SkillPlanner, unchanged). Besides the fixed yaws, the object's own face-aligned yaws are tried
    (the pipeline grips a box square to its measured faces)."""
    from control.motion import PlanningError
    from control.robot import ARM_HOME

    planner = _planner(sc)
    out = {}
    for o in sc.objects:
        top = np.array([o.pos[0], o.pos[1], o.pos[2] + o.height])
        support = None if o.support == "floor" else o.support
        # an object lying across raised rests is measured from the surface the camera sees under it (the pads close
        # in the gap between the rests), as the live pipeline measures it
        height = o.pos[2] + o.height if o.attributes.get("lies_on_rests") else o.height
        ok, why = [], {}
        round_body = all(p.shape == "cylinder" for p in o.parts if p.collide)  # yaw-symmetric: face yaws add nothing
        cands = list(yaws) + ([] if round_body else [round(y, 1) for y in ((o.yaw_deg % 180), (o.yaw_deg + 90) % 180)
                                                      if all(abs(y - z) > 1 for z in yaws)])
        for yaw in cands:
            try:
                planner.pick(ARM_HOME, top, width=grasp_width(o, yaw), yaw_deg=yaw, target=o.id, support=support, height=height)
                ok.append(yaw)
            except PlanningError as exc:
                why[yaw] = str(exc)[:140]
        out[o.id] = {"feasible_yaws": ok, "reason": "" if ok else why[yaws[0]]}
    return out


def placement_feasibility(sc: Scenario, yaws=(0.0, 90.0)) -> dict:
    """Per FREE receiving region (region.accepts set): can a held probe container of the accepted type (PROBES:
    width, height) be lowered onto the region centre, from HOME, by the unchanged planner?"""
    from control.motion import PlanningError
    from control.robot import ARM_HOME
    from control.skills import Held

    planner = _planner(sc)
    out = {}
    for r in sc.regions:
        if not r.accepts or r.occupied_by:
            continue
        w, h = PROBES[r.accepts]
        grasp_z, _ = planner._grasp_z(h, h, w)  # held as the pick would hold it (a thin object: fingertips near its base)
        ok, last = [], ""
        for yaw in yaws:
            held = Held(height=h, grip_depth=h - grasp_z, yaw_deg=yaw, name="probe", width=w, length=w)
            try:
                planner.place(ARM_HOME, held, np.asarray(r.centre, float), yaw_deg=yaw)
                ok.append(yaw)
            except PlanningError as exc:
                last = str(exc)[:140]
        out[r.id] = {"feasible_yaws": ok, "probe": r.accepts, "reason": "" if ok else last}
    return out


def view_feasibility(sc: Scenario, targets: dict) -> dict:
    """Wrist-camera access: targets = {name: world point}; the existing viewpoint planner (camera look-at IK,
    line of sight clear of every collider, collision-checked path from HOME) must find a viewpoint."""
    from perception.viewpoints import ViewpointPlanner
    from control.collision import Box
    from control.robot import ARM_HOME

    fx, ob = scenario_parts(sc)
    boxes = [Box(wp.owner, wp.centre, wp.R, wp.half) for wp in fx + ob]
    vp = ViewpointPlanner()
    out = {}
    for name, p in targets.items():
        plan = vp.plan(np.asarray(p, float), np.asarray(ARM_HOME, float), boxes, target_box_name=name)
        out[name] = {"ok": plan.ok, "reason": plan.reason, "best": plan.summary()["best"]}
    return out


def _min_dist(wp, point=(0.0, 0.0)) -> float:
    """Distance in xy from a point to a part's footprint (cylinders: their bounding square)."""
    local = wp.R[:2, :2].T @ (np.asarray(point, float) - wp.centre[:2])
    q = np.clip(local, -wp.half[:2], wp.half[:2])
    return float(np.linalg.norm(local - q))


def static_report(sc: Scenario, reach: bool = True) -> dict:
    fx, ob = scenario_parts(sc)
    problems = []
    for a, b, mm in overlaps(sc):
        problems.append(f"interpenetration {a} / {b}: {mm} mm")
    keep_clear = [] if sc.view.get("table") else KEEP_CLEAR  # a table scene hides the frozen props (scenarios/build.py)
    for wp in fx + ob:  # props of the frozen default scene must stay clear
        for c, r in keep_clear:
            if _min_dist(wp, c) < r:
                problems.append(f"{wp.name} inside the keep-clear zone at {c}")
    for d in sc.decor:
        for wp in world_parts(d.id, d.pos, d.yaw_deg, d.parts, collide_only=False):
            reach_r = _min_dist(wp)
            bottom, top = wp.centre[2] - wp.half[2], wp.centre[2] + wp.half[2]
            in_arm_height = top > 0.002 and bottom < DECOR_MAX_REACH_Z  # under the table top or above the arm: no matter
            if wp.half[2] > 0.002 and in_arm_height and reach_r < DECOR_MIN_RADIUS:
                problems.append(f"decor {wp.name} within {DECOR_MIN_RADIUS} m of the base ({reach_r:.2f} m)")
    for o in sc.objects:
        r = np.hypot(*o.pos[:2])
        if r > REACH_MAX and o.role != "unreachable":
            problems.append(f"{o.id} at {r:.2f} m is beyond reach but not marked unreachable")
    import time
    timing = {}
    t = time.monotonic()
    vis = top_visibility(sc)
    timing["top_visibility_s"] = round(time.monotonic() - t, 2)
    for o in sc.objects:  # intent: only objects designed to be hidden are invisible to the table camera
        if vis[o.id] == 0.0 and o.role != "occluded":
            problems.append(f"{o.id} is hidden from the table camera but not marked occluded")
        if o.role == "occluded" and vis[o.id] > 0.0:
            problems.append(f"{o.id} is marked occluded but the table camera sees {vis[o.id]:.0%} of its top")
    rep = {"scenario": sc.id, "difficulty": sc.difficulty, "seed": sc.seed, "objects": len(sc.objects),
           "fixtures": len(sc.fixtures), "problems": problems, "table_camera_top_visible": vis}
    rep["timing"] = timing
    if reach:
        t = time.monotonic()
        rep["grasp"] = grasp_feasibility(sc)
        timing["grasp_feasibility_s"] = round(time.monotonic() - t, 2)
        for o in sc.objects:  # intent: only objects designed to be unreachable have no planned grasp
            if not rep["grasp"][o.id]["feasible_yaws"] and o.role != "unreachable":
                problems.append(f"{o.id} has no planned grasp but is not marked unreachable: {rep['grasp'][o.id]['reason']}")
            if rep["grasp"][o.id]["feasible_yaws"] and o.role == "unreachable":
                problems.append(f"{o.id} is marked unreachable but a grasp plans")
        t = time.monotonic()
        rep["placement"] = placement_feasibility(sc)
        timing["placement_feasibility_s"] = round(time.monotonic() - t, 2)
        problems += [f"receiving region {k} refuses a held {v['probe']}: {v['reason']}" for k, v in rep["placement"].items()
                     if not v["feasible_yaws"]]
        targets = {r.id: (r.centre[0], r.centre[1], r.centre[2] + 0.03) for r in sc.regions if r.kind == "inspection"}
        targets.update({o.id: (o.pos[0], o.pos[1], o.pos[2] + o.height) for o in sc.objects if o.role == "occluded"})
        t = time.monotonic()
        rep["wrist_view"] = view_feasibility(sc, targets)
        timing["view_feasibility_s"] = round(time.monotonic() - t, 2)
        problems += [f"no wrist-camera viewpoint for {k}: {v['reason']}" for k, v in rep["wrist_view"].items() if not v["ok"]]
    return rep


def main(argv=None) -> int:
    """python -m scenarios.checks --scenario laboratory --difficulty easy --seed 42 --out design.json
    Runs outside Isaac Sim (pure numpy), so the validator can run it as a subprocess with a timeout."""
    import argparse
    import json
    import time
    from scenarios import generate
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--difficulty", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    sc = generate(a.scenario, a.difficulty, a.seed)
    rep = static_report(sc)
    rep["seconds"] = round(time.monotonic() - t0, 1)
    rep["scenario_version"] = sc.version
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"[checks] {a.scenario} {a.difficulty} seed {a.seed}: {len(rep['problems'])} problem(s) in {rep['seconds']} s "
          f"{rep.get('timing')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
