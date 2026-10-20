"""Inspection capabilities (wrist camera; move the arm, never change objects): inspect one visible object
closely, explore hidden regions (both end with a verified return HOME), and look in a named direction (the arm
stays there so the camera's direction can be measured; a later motion or return_home moves it)."""

from __future__ import annotations

import time

import numpy as np

from agent.capabilities.manipulation import run_return_home
from agent.capabilities.perception import _find_in, _target_mark
from agent.capabilities.specs import Context, StepResult
from perception.somark import isolate


MAX_EXPLORE_VIEWPOINTS = 2


def _wrist_observation(ctx: Context, target_world: np.ndarray, target_id: int | None) -> dict:
    """Fresh wrist frame -> every wrist object folded into the world model (association + fusion rule);
    the wrist mark whose world centre is nearest the look-at point is the inspected one."""
    t = ctx.tools
    wobs = t.perceive("wrist")
    ctx.extra_obs.append(wobs)
    assoc = {}
    for k, g in wobs.measured.items():
        st, oid = t.world.add_observation(g, "wrist", wobs.frame_id, k)
        assoc[k] = {"status": st, "world_id": oid}
    wobs.assoc = assoc
    near = sorted(wobs.measured, key=lambda k: np.hypot(*(np.array(wobs.measured[k]["centre"][:2]) - target_world[:2])))
    mark = near[0] if near and np.hypot(*(np.array(wobs.measured[near[0]]["centre"][:2]) - target_world[:2])) < 0.06 else None
    desc = None
    if mark is not None:
        crop = isolate(wobs.rgb, wobs.marks[mark])
        desc = t.roles.describe_one(crop).model_dump()
        wobs.descriptions[mark] = desc
        if assoc[mark]["world_id"] is not None:
            t.world.objects[assoc[mark]["world_id"]].descriptions_by_camera["wrist"] = desc
    return {"obs": wobs, "assoc": assoc, "mark": mark, "description": desc}


def run_inspect(ctx: Context, args: dict) -> StepResult:
    """Closer look at one described, visible object with the wrist camera. Read-only for the scene; moves
    the arm (collision-checked) and returns HOME (verified) at the end."""
    from perception.viewpoints import ViewpointPlanner
    from control.kinematics import Kinematics
    from control.motion import joint_move
    from control.robot import ARM_SPEED
    from control.skills import MoveStep
    text = (args or {}).get("target", "").strip()
    t, t0 = ctx.tools, time.monotonic()
    if t.world.holding is not None:
        return StepResult("unsafe", "not inspecting while holding an object")
    mark, fail = _target_mark(ctx, text, args)
    if fail is not None:
        return fail
    g = ctx.obs.measured[mark]
    oid = ctx.obs.object_ids.get(mark)
    target = np.array([g["centre"][0], g["centre"][1], g["top_z"]])
    q0 = t.bridge.q()
    vp = ViewpointPlanner()
    tp = time.monotonic()
    plan = vp.plan(target, q0, t.obstacles(), target_box_name=f"obj_{oid}" if oid else None)
    plan_s = round(time.monotonic() - tp, 2)
    ctx.log("exploration", purpose="inspect", target=text, mark=mark, plan=plan.summary(), seconds=plan_s)
    if not plan.ok:
        return StepResult("infeasible", f"no safe inspection viewpoint for {text!r}: {plan.reason}", {"plan": plan.summary()})
    b = plan.best
    res = t.execute([MoveStep(joint_move(Kinematics(), q0, b.q, ARM_SPEED), f"move wrist camera to inspect {text}")], "inspect view")
    if not res.ok:
        return StepResult("unsafe" if "safety" in (res.reason or "").lower() else "failed", f"inspection move failed: {res.reason}")
    t.bridge.wait_still()
    reach = vp.reached(t.bridge.q(), b)
    ctx.log("verification", check="viewpoint", **reach)
    if not reach["pass"]:
        return StepResult("failed", f"viewpoint not reached (max joint error {reach['max_joint_error_deg']} deg)")
    w = _wrist_observation(ctx, target, oid)
    home = run_return_home(ctx, {})
    wm = w["mark"]
    data = {"plan": plan.summary(), "plan_seconds": plan_s, "reach": reach, "wrist_frame": w["obs"].frame_id,
            "association": w["assoc"].get(wm) if wm else None, "home": home.outcome, "seconds": round(time.monotonic() - t0, 1)}
    if wm is None:
        return StepResult("insufficient_info", f"moved the wrist camera to {text!r} but did not measure it there "
                          f"(table measurement kept); {home.summary}", data)
    wg = w["obs"].measured[wm]
    d = w["description"] or {}
    data.update({"table_centre": g["centre"], "wrist_centre": wg["centre"], "wrist_footprint": wg["footprint"],
                 "wrist_height": wg["height"], "wrist_description": d})
    ctx.log("inspection", target=text, mark=mark, wrist_mark=wm, table_centre=g["centre"], wrist_centre=wg["centre"],
            description=d, association=data["association"])
    shift = np.hypot(*(np.array(wg["centre"][:2]) - np.array(g["centre"][:2]))) * 1000
    used = (data["association"] or {}).get("status") == "matched" and t.world.objects[data["association"]["world_id"]].observations[-1]["used_for_geometry"]
    data["bindings"] = {text: oid}
    data["evidence"] = [{"kind": "frame", "id": w["obs"].frame_id, "mark": wm}]
    return StepResult("completed",
                      f"Inspected {text!r} from {b.distance:.2f} m with the wrist camera: at x {wg['centre'][0] * 1000:+.0f} mm, "
                      f"y {wg['centre'][1] * 1000:+.0f} mm ({shift:.0f} mm from the table camera's estimate), about "
                      f"{wg['footprint'][0] * 1000:.0f} x {wg['footprint'][1] * 1000:.0f} mm, {wg['height'] * 1000:.0f} mm tall; "
                      f"close-up: {d.get('colour', '')} {d.get('name', '')}, shape {d.get('shape', '?')!r} [Gemma, from the wrist view]. "
                      f"{'Wrist measurement adopted' if used else 'Table measurement kept (wrist view partial or inconsistent)'}; {home.summary}.",
                      data)


def run_explore(ctx: Context, args: dict) -> StepResult:
    """Look into regions the table camera cannot see: the occlusion shadows behind measured objects (behind the
    object named in `near` if given, else the largest shadows), at most MAX_EXPLORE_VIEWPOINTS collision-checked
    wrist viewpoints, then HOME. Moving the camera is not finding: the result says what was newly seen; the
    supervisor decides the goal with locate."""
    from perception.viewpoints import ViewpointPlanner, shadow_points
    from control.kinematics import Kinematics
    from control.motion import joint_move
    from control.robot import ARM_SPEED
    from control.skills import MoveStep
    text, near = (args or {}).get("target", "").strip(), ((args or {}).get("near") or "").strip()
    t, t0 = ctx.tools, time.monotonic()
    if t.world.holding is not None:
        return StepResult("unsafe", "not exploring while holding an object")
    objs = [{"id": o.id, "geometry": o.geometry} for o in t.world.objects.values() if o.status in ("on_surface", "occluded")]
    if near:  # behind the named object; if several objects fit the name, behind each (bounded)
        r = ctx.tools.resolve(ctx.obs, near)
        marks = [r.id] if r.id is not None else list(r.candidates)
        if not marks:
            return StepResult("insufficient_info", f"cannot find {near!r} to look behind")
        ids = {ctx.obs.object_ids.get(m) for m in marks}
        objs = [o for o in objs if o["id"] in ids]
    regions = shadow_points(objs, n_max=MAX_EXPLORE_VIEWPOINTS)
    if not regions:
        return StepResult("insufficient_info", "no hidden region to look into")
    vp, views, new_total, tried = ViewpointPlanner(), 0, 0, []
    found = None
    for point, behind_id in regions:
        q0 = t.bridge.q()
        tp = time.monotonic()
        from perception.viewpoints import REGION_HEIGHT, REGION_RADIUS
        plan = vp.plan(point, q0, t.obstacles(), region_radius=REGION_RADIUS, region_height=REGION_HEIGHT,
                       elevations=(70.0, 55.0, 40.0))  # steeper views see further behind an occluder
        ctx.log("exploration", purpose="explore", target=text, region=[round(v, 3) for v in point], behind=behind_id,
                plan=plan.summary(), seconds=round(time.monotonic() - tp, 2))
        tried.append({"region": point.tolist(), "behind": behind_id, "plan": plan.summary()})
        if not plan.ok:
            continue
        res = t.execute([MoveStep(joint_move(Kinematics(), q0, plan.best.q, ARM_SPEED), f"move wrist camera to look behind object {behind_id}")], "explore view")
        if not res.ok:
            run_return_home(ctx, {})
            return StepResult("unsafe" if "safety" in (res.reason or "").lower() else "failed", f"exploration move failed: {res.reason}")
        t.bridge.wait_still()
        reach = vp.reached(t.bridge.q(), plan.best)
        ctx.log("verification", check="viewpoint", **reach)
        if not reach["pass"]:
            break
        views += 1
        w = _wrist_observation(ctx, point, None)
        new = [k for k, a in w["obs"].assoc.items() if a["status"] == "new"]
        new_total += len(new)
        found = _find_in(ctx, w["obs"], text) if text else None
        if found is not None:
            break
    home = run_return_home(ctx, {}) if views else None
    ctx.log("explore_result", target=text, viewpoints=views, new_objects=new_total, found=found is not None,
            seconds=round(time.monotonic() - t0, 1))
    data = {"viewpoints": views, "new_objects": new_total, "regions": tried, "found": found is not None}
    if views == 0:
        reasons = "; ".join(r["plan"]["reason"] for r in tried)
        return StepResult("infeasible", f"no safe viewpoint into the hidden region(s): {reasons}", data)
    tail = f" {home.summary}." if home else ""
    return StepResult("completed", f"Looked into {views} hidden region(s) with the wrist camera: {new_total} object(s) not "
                                   f"visible to the table camera" + (f"; {found.summary}" if found else f"; no {text!r} among them") + tail,
                      {**data, "found_summary": found.summary if found else None})


# Look geometry (pure geometry over the robot's kinematics, no scene knowledge): the wrist camera stays high (the base
# is raised and the camera is canted on the J5 housing) and looks outward and down at a floor point in the named
# direction. Camera azimuths are tried closest-first, so the chosen view is the one nearest the requested direction.
LOOK_RADIUS = 1.0  # m: floor point looked at, this far from the base in the named direction
LOOK_DISTANCES = (0.8, 1.0, 1.2)  # m: camera to the look point
LOOK_ELEVATIONS = (35.0, 45.0, 55.0)  # deg: camera above the look point, from the horizontal
LOOK_OFFSETS_DEG = (0.0, 10.0, -10.0, 20.0, -20.0)  # camera azimuth offsets, closest first
DOWN_RADIUS = (0.35, 0.60)  # m: 'down' looks at the floor below the camera, kept inside this ring


def look_target(direction: str, frame: str, q_now) -> tuple[np.ndarray, list, tuple, tuple]:
    """Look-at point, camera azimuth tries (closest first), elevations and distances for the viewpoint planner. The
    camera sits between the base and the look point, so its optical axis points the named way (agent/directions.py)."""
    from agent.directions import camera_pose, direction_vector
    if direction == "down":
        xy = camera_pose(q_now)[:2, 3]
        r = float(np.hypot(*xy))
        xy = (xy / r if r > 1e-3 else np.array([0.0, 1.0])) * float(np.clip(r, *DOWN_RADIUS))
        return np.array([xy[0], xy[1], 0.0]), [None], (85.0,), (0.35, 0.45)
    u = direction_vector(direction, frame)
    target = np.array([u[0], u[1], 0.0]) * LOOK_RADIUS
    back = float(np.degrees(np.arctan2(-u[1], -u[0])))  # from the look point back towards the base
    return target, [(back + d) % 360.0 for d in LOOK_OFFSETS_DEG], LOOK_ELEVATIONS, LOOK_DISTANCES


def run_look(ctx: Context, args: dict) -> StepResult:
    """Point the wrist camera in a named direction of an explicit frame and observe there. Gemma names the direction;
    the viewpoint is chosen and checked by the existing planner (look-at IK, joint limits, clear view, collision-free
    path) and executed through Tools.execute (motion gate, frozen executor, simulator safety layer). The arm stays at
    the viewpoint; the measured camera direction is reported (and verified by the verification worker)."""
    from agent.directions import DIRECTIONS, FRAMES, view_error
    from control.kinematics import Kinematics
    from control.motion import joint_move
    from control.robot import ARM_SPEED
    from control.skills import MoveStep
    from perception.viewpoints import ViewpointPlanner
    from agent.vlm_roles import VLMFailure
    direction, frame = (args or {}).get("direction"), (args or {}).get("frame") or "robot"
    t, t0 = ctx.tools, time.monotonic()
    if direction not in DIRECTIONS or frame not in FRAMES:
        return StepResult("ambiguous", f"which direction? got {direction!r} in frame {frame!r}; I can look "
                          f"{', '.join(DIRECTIONS)} in the {' or '.join(FRAMES)} frame")
    if t.world.holding is not None:
        return StepResult("infeasible", "not moving the camera around while holding an object (place it first)")
    q0 = np.asarray(t.bridge.q(), float)
    target, azimuths, elevations, distances = look_target(direction, frame, q0)
    vp, obstacles = ViewpointPlanner(), t.obstacles()
    tp = time.monotonic()
    for az in azimuths:  # closest to the requested direction first; the first safe view wins
        plan = vp.plan(target, q0, obstacles, azimuths=None if az is None else [az], elevations=elevations,
                       distances=distances)
        if plan.ok:
            break
    plan_s = round(time.monotonic() - tp, 2)
    ctx.log("exploration", purpose="look", target=f"{direction} ({frame} frame)", plan=plan.summary(), seconds=plan_s,
            look_point=[round(float(v), 3) for v in target])
    data = {"direction": direction, "frame": frame, "look_point": [round(float(v), 3) for v in target],
            "plan": plan.summary(), "plan_seconds": plan_s}
    if not plan.ok:
        return StepResult("infeasible", f"no safe viewpoint to look {direction} ({frame} frame): {plan.reason}", data)
    b = plan.best
    res = t.execute([MoveStep(joint_move(Kinematics(), q0, b.q, ARM_SPEED), f"point the wrist camera {direction}")], "look view")
    if not res.ok:
        return StepResult("unsafe" if "safety" in (res.reason or "").lower() else "failed",
                          f"camera move to look {direction} was not executed or failed: {res.reason}", data)
    t.bridge.wait_still()
    q = np.asarray(t.bridge.q(), float)
    reach = vp.reached(q, b)
    view = view_error(q, direction, frame)  # measured: joint feedback -> forward kinematics
    ctx.log("verification", check="viewpoint", **reach)
    ctx.log("verification", check="view_direction", **{"pass": view["pointing"]}, **view)
    data.update({"reach": reach, "view": view})
    if not reach["pass"]:
        return StepResult("failed", f"viewpoint not reached (max joint error {reach['max_joint_error_deg']} deg)", data)
    wobs = t.perceive("wrist")
    new = 0
    for k, g in wobs.measured.items():
        st, _ = t.world.add_observation(g, "wrist", wobs.frame_id, k)
        new += st == "new"
    try:
        t.describe(wobs)
    except (VLMFailure, TimeoutError) as exc:
        ctx.log("tool", tool="describe_failed", camera="wrist", error=str(exc)[:200])
    ctx.obs = wobs  # what the robot now sees: a following describe_scene / answer / locate uses this view
    ctx.extra_obs.append(wobs)
    ctx.state.record_observation(wobs)
    n = len(wobs.marks)
    data.update({"wrist_frame": wobs.frame_id, "objects": n, "new_objects": new, "seconds": round(time.monotonic() - t0, 1),
                 "evidence": [{"kind": "frame", "id": wobs.frame_id, "camera": "wrist"}]})
    cam = view["camera_position_mm"]
    return StepResult("completed",
                      f"Pointed the wrist camera {direction} ({frame} frame): its axis is {view['error_deg']:.0f} deg from "
                      f"that direction (measured from the joint state; camera at x {cam[0]:+d}, y {cam[1]:+d}, z {cam[2]} mm). "
                      f"It sees {n} object(s), {new} not measured before. The arm stays here.", data)
