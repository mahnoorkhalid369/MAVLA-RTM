"""Validate scenario-library scenes in Isaac Sim (headless, no ROS, no agent, no robot motion).

For each difficulty: build the scene exactly as sim/run_scenario.py does (frozen setup_scene + scenarios
install / populate), hold the arm at HOME, settle physics, then check
  robot + gripper articulation present (6 arm joints + finger joint), both cameras render (RGB + depth)
  every object spawned as a rigid body; no object below the floor; settled displacement / tilt (stability)
  interpenetration of the SETTLED poses (pure geometry, scenarios.core.penetration)
and save table-camera, wrist-camera and overview images. With --reload, the same (scenario, difficulty, seed)
is rebuilt in the same process (new stage) and the settled poses are compared (reproducible reload).
Physics runs on the GPU (as the live simulator); the images come from a second build with CPU physics, because
headless without the ROS bridge the GPU pipeline leaves the wrist camera prim at its zero-joint USD pose.
Design checks (visibility, grasp / placement feasibility with the unchanged planner) come from
scenarios/checks.py and are stored next to the results.

    .venv/bin/python evaluation/validate_scenarios.py --scenario laboratory [--difficulty easy medium hard]
           [--seed 42] [--settle 3.0] [--reload] [--no-design]
Results: out/scenarios/<scenario>/<difficulty>_s<seed>/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaacsim import SimulationApp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", required=True)
ap.add_argument("--difficulty", nargs="+", default=["easy", "medium", "hard"])
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--settle", type=float, default=3.0, help="seconds of simulated settling")
ap.add_argument("--reload", action="store_true", help="rebuild each scene a second time and compare")
ap.add_argument("--drop-mm", type=float, default=0.0,
                help="extra physics proof: rebuild with every object this far above its support; it must land back on it")
ap.add_argument("--resume", action="store_true", help="reuse saved stage results (out/.../stages/*.json)")
ap.add_argument("--design-timeout", type=float, default=1800.0, help="seconds the design subprocess may take")
ap.add_argument("--no-design", action="store_true", help="skip the (slow) offline grasp / placement design checks")
args = ap.parse_args()
app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import carb  # noqa: E402

for key, value in (("/rtx/materialDb/syncLoads", True), ("/rtx/hydra/materialSyncLoads", True),
                   ("/omni.kit.plugin/syncUsdLoads", True), ("/rtx-transient/resourcemanager/texturestreaming/async", False),
                   ("/rtx-transient/resourcemanager/enableTextureStreaming", False)):
    carb.settings.get_settings().set(key, value)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import warp as wp  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

from control.robot import ARM_HOME, ARM_JOINTS, GRIPPER_JOINT, GRIPPER_OPEN  # noqa: E402
from scenarios import generate  # noqa: E402
from scenarios.build import install, overview_camera, populate  # noqa: E402
from scenarios.core import WorldPart, penetration, rot_z, scenario_parts  # noqa: E402
from sim.scene import ROBOT_PATH, add_table_camera, add_wrist_camera, setup_scene  # noqa: E402

DT = 1 / 120
STABLE_MM, STABLE_TILT_DEG, DRIFT_MM = 5.0, 3.0, 0.5  # rest error, tilt, drift during one more simulated second


def quat_tilt_yaw(q):
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    return R, tilt, float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def build(sc, device: str):
    install(sc)
    SimulationManager.setup_simulation(dt=DT, device=device)
    app.run_coroutine(setup_scene())
    populate(sc)
    wrist = add_wrist_camera()
    table, _ = add_table_camera()
    over = overview_camera(sc)
    return {"table": table, "wrist": wrist, "overview": over}


def wrist_inspection(sc, robot, idx, out: Path, target_id: str) -> dict:
    """Wrist-camera image from a viewpoint above a fixture (sc.view['wrist']): the pose comes from the existing,
    unmodified viewpoint planner (camera look-at IK + collision-checked against the scene's colliders) and is SET
    directly in this validation (no commanded motion, no agent); the arm goes back to HOME afterwards."""
    from control.collision import Box
    from perception.viewpoints import ViewpointPlanner
    f = next((f for f in sc.fixtures if f.id == target_id), None)
    if f is None:
        return {"ok": False, "reason": f"no fixture {target_id}"}
    fx, ob = scenario_parts(sc)
    boxes = [Box(w.owner, w.centre, w.R, w.half) for w in fx + ob]
    top = max(p.pos[2] + p.size[2] / 2 for p in f.parts)
    vp = ViewpointPlanner().plan(np.array([f.pos[0], f.pos[1], top + 0.05]), np.asarray(ARM_HOME, float), boxes,
                                 target_box_name=f.id)
    if not vp.ok:
        return {"ok": False, "reason": vp.reason}
    q = np.concatenate([vp.best.q, [GRIPPER_OPEN]]).astype(np.float32).reshape(1, -1)
    if not app_utils.is_playing():  # replicator captures pause the timeline; without physics steps PhysX never
        app_utils.play()             # writes the new pose to USD and the render would show the previous pose
    robot.set_dof_positions(wp.array(q, dtype=wp.float32), dof_indices=idx)
    robot.set_dof_position_targets(wp.array(q, dtype=wp.float32), dof_indices=idx)
    for _ in range(30):
        app.update()
    from control.kinematics import Kinematics
    from perception.viewpoints import CAM_TIP
    k = Kinematics(tip=CAM_TIP)
    q_meas = robot.get_dof_positions(dof_indices=idx).numpy()[0]
    import isaacsim.core.experimental.utils.stage as su
    from pxr import UsdGeom
    from sim.scene import find_prim_path as _fp
    T = UsdGeom.XformCache().GetLocalToWorldTransform(su.get_current_stage().GetPrimAtPath(_fp(ROBOT_PATH, "camera_color_optical_frame")))
    usd_err = float(np.linalg.norm(np.array(T.ExtractTranslation()) - k.fk(q_meas[:k.lower.size])[:3, 3])) * 1000
    if usd_err > 5.0:  # the rendered camera would not be where the arm is: do not save a misleading image
        return {"ok": False, "reason": f"render pose differs from the arm's FK by {usd_err:.0f} mm"}
    from sim.scene import find_prim_path
    cam = find_prim_path(ROBOT_PATH, "d435_color")
    rp = rep.create.render_product(cam, (1280, 720))
    a = rep.AnnotatorRegistry.get_annotator("rgb")
    a.attach([rp])
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=8, delta_time=0.0)
    rgb = np.asarray(a.get_data())[..., :3].astype(np.uint8)
    cv2.imwrite(str(out / f"wrist_{target_id}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    a.detach()
    rp.destroy()
    home = np.concatenate([ARM_HOME, [GRIPPER_OPEN]]).astype(np.float32).reshape(1, -1)
    robot.set_dof_positions(wp.array(home, dtype=wp.float32), dof_indices=idx)
    robot.set_dof_position_targets(wp.array(home, dtype=wp.float32), dof_indices=idx)
    return {"ok": True, "target": target_id, "viewpoint": vp.summary()["best"], "rgb_std": round(float(rgb.std()), 1),
            "render_pose_error_mm": round(usd_err, 2)}


def run_once(sc, out: Path | None, drop_mm: float = 0.0, device: str = "cuda") -> dict:
    t0 = time.monotonic()
    rest = {o.id: (o.pos[0], o.pos[1], o.pos[2]) for o in sc.objects}  # where each object should come to rest
    for o in sc.objects:  # drop test: spawn above the support (this Scenario instance only)
        o.pos = (o.pos[0], o.pos[1], o.pos[2] + drop_mm / 1000)
    cams = build(sc, device)
    ids = [o.id for o in sc.objects]
    bodies = RigidPrim([f"/World/Objects/{i}" for i in ids])
    app_utils.play()
    p0, q0 = (a.numpy() for a in bodies.get_world_poses())  # spawn poses, before any physics step
    app.update()
    robot = Articulation(ROBOT_PATH)
    names = list(robot.dof_names)
    ctrl = ARM_JOINTS + [GRIPPER_JOINT]
    idx = robot.get_dof_indices(ctrl)
    start = np.concatenate([ARM_HOME, [GRIPPER_OPEN]]).astype(np.float32).reshape(1, -1)
    robot.set_dof_positions(wp.array(start, dtype=wp.float32), dof_indices=idx)
    robot.set_dof_position_targets(wp.array(start, dtype=wp.float32), dof_indices=idx)
    app.update()

    def advance(seconds):
        t = SimulationManager.get_simulation_time()
        for _ in range(int(seconds / DT) + 10):  # bounded: sim time must advance
            if SimulationManager.get_simulation_time() - t >= seconds:
                break
            app.update()
        return SimulationManager.get_simulation_time() - t
    sim_dt = advance(args.settle)
    p_set, _ = (a.numpy() for a in bodies.get_world_poses())
    sim_dt += advance(1.0)  # drift test: a settled object must not keep moving
    p1, q1 = (a.numpy() for a in bodies.get_world_poses())
    v1 = bodies.get_velocities()[0].numpy()
    q_robot = robot.get_dof_positions(dof_indices=idx).numpy()[0]
    res = {"seconds_build_settle": round(time.monotonic() - t0, 1), "settle_sim_seconds": round(sim_dt, 3),
           "drop_mm": drop_mm, "robot": {"present": all(j in names for j in ctrl), "dofs": len(names),
                     "home_error_deg": round(float(np.degrees(np.abs(q_robot[:6] - ARM_HOME)).max()), 3)},
           "objects": {}, "problems": []}
    settled_parts = []
    for k, o in enumerate(sc.objects):
        spawn = np.array(rest[o.id])
        R, tilt, yaw = quat_tilt_yaw(q1[k])
        disp_spawn = float(np.linalg.norm(p1[k] - spawn)) * 1000
        disp_settle = float(np.linalg.norm(p1[k] - p0[k])) * 1000
        speed = float(np.linalg.norm(v1[k]))  # reported solver velocity (information: resting contacts show a few mm/s)
        drift = float(np.linalg.norm(p1[k] - p_set[k])) * 1000
        stable = disp_spawn <= STABLE_MM and tilt <= STABLE_TILT_DEG and drift <= DRIFT_MM
        res["objects"][o.id] = {"settled_xyz": [round(float(v), 4) for v in p1[k]], "settled_yaw_deg": round(yaw, 2),
                                "tilt_deg": round(tilt, 2), "moved_from_spawn_mm": round(disp_spawn, 2),
                                "moved_while_settling_mm": round(disp_settle, 2), "drift_last_1s_mm": round(drift, 3),
                                "speed_mps": round(speed, 4),
                                "stable": bool(stable), "below_floor": bool(p1[k][2] < -0.002)}
        if not stable:
            res["problems"].append(f"{o.id}: rest error {disp_spawn:.1f} mm, tilt {tilt:.1f} deg, drift {drift:.2f} mm/s")
        if p1[k][2] < -0.002:
            res["problems"].append(f"{o.id} below the floor")
        for i, p in enumerate(o.parts):  # settled colliders (tilt ignored when small)
            if p.collide:
                Rb = rot_z(yaw)
                c = p1[k] + Rb @ np.asarray(p.pos, float)
                h = np.array([p.size[0] / 2, (p.size[0] if p.shape == "cylinder" else p.size[1]) / 2, p.size[2] / 2])
                settled_parts.append(WorldPart(o.id, f"{o.id}/{i}", p.shape, c, Rb @ rot_z(p.yaw_deg), h))
    fx, _ = scenario_parts(sc)
    pen = []
    for i, a in enumerate(settled_parts):
        for b in settled_parts[i + 1:] + fx:
            if a.owner != b.owner:
                d = penetration(a, b)
                if d > 0.0015:  # contact offsets / rest offsets: ~1 mm is resting contact, not penetration
                    pen.append((a.name, b.name, round(d * 1000, 2)))
    res["settled_interpenetration_mm"] = pen
    res["problems"] += [f"settled interpenetration {a} / {b}: {d} mm" for a, b, d in pen]
    if out is not None:  # images: table + wrist (RGB, depth), overview (RGB)
        out.mkdir(parents=True, exist_ok=True)
        cam_res = {}
        for name, path in cams.items():
            rp = rep.create.render_product(path, (1280, 720))
            a_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            a_dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            a_rgb.attach([rp])
            a_dep.attach([rp])
            for _ in range(3):
                rep.orchestrator.step(rt_subframes=8, delta_time=0.0)
            rgb = np.asarray(a_rgb.get_data())[..., :3].astype(np.uint8)
            dep = np.nan_to_num(np.asarray(a_dep.get_data()), nan=0.0, posinf=0.0)
            cv2.imwrite(str(out / f"{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if name != "overview":
                cv2.imwrite(str(out / f"{name}_depth.png"), np.clip(dep * 1000, 0, 65535).astype(np.uint16))
            valid = float(((dep > 0.1) & (dep < 10)).mean())
            cam_res[name] = {"rgb_std": round(float(rgb.std()), 1), "depth_valid_fraction": round(valid, 3),
                             "ok": bool(rgb.std() > 5 and (name == "overview" or valid > 0.5))}
            a_rgb.detach()
            a_dep.detach()
            rp.destroy()
        res["cameras"] = cam_res
        res["wrist_inspection"] = {t: wrist_inspection(sc, robot, idx, out, t) for t in sc.view.get("wrist", [])}
        res["problems"] += [f"camera {n} not rendering properly: {c}" for n, c in cam_res.items() if not c["ok"]]
        res["problems"] += [f"no wrist inspection view of {t}: {w.get('reason')}" for t, w in res["wrist_inspection"].items()
                            if not w["ok"]]
    app_utils.stop()
    app.update()
    return res


class Stage:
    """One validation stage: progress lines with elapsed time, a heartbeat every HEARTBEAT_S, and a hard limit:
    past it faulthandler dumps every thread's Python stack into the log and exits (no silent hang; the dump also
    works where ptrace is restricted)."""

    def __init__(self, level: str, name: str, limit_s: float) -> None:
        self.level, self.name, self.limit = level, name, limit_s

    def __enter__(self):
        import faulthandler
        import threading
        self.t0 = time.monotonic()
        self.done = threading.Event()
        print(f"[validate] {self.level} | {self.name}: start (limit {self.limit:.0f} s)", flush=True)

        def beat():
            while not self.done.wait(HEARTBEAT_S):
                print(f"[validate] {self.level} | {self.name}: still running, {time.monotonic() - self.t0:.0f} s", flush=True)
        threading.Thread(target=beat, daemon=True).start()
        faulthandler.dump_traceback_later(self.limit, exit=True)
        return self

    def __exit__(self, *exc):
        import faulthandler
        faulthandler.cancel_dump_traceback_later()
        self.done.set()
        print(f"[validate] {self.level} | {self.name}: {'FAILED' if exc[0] else 'done'} in "
              f"{time.monotonic() - self.t0:.1f} s", flush=True)
        return False


HEARTBEAT_S = 60.0
STAGE_LIMIT_S = 600.0  # one scene build + settle + images (normally 5-20 s)


def staged(level: str, name: str, out: Path, fn):
    """Run a stage once; with --resume, reuse its saved result (out/stages/<name>.json) from an earlier run."""
    f = out / "stages" / f"{name}.json"
    if args.resume and f.exists():
        print(f"[validate] {level} | {name}: reused {f.relative_to(ROOT)}", flush=True)
        return json.loads(f.read_text())
    with Stage(level, name, STAGE_LIMIT_S):
        r = fn()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(r, indent=1))
    return r


def start_design(level: str, out: Path):
    """Offline design checks (pure numpy, minutes) in a separate plain Python process, in parallel with the scene
    builds: inside the Kit process the same checks ran ~7x slower (measured 2026-10-15: 41.5 min vs 6 min)."""
    import subprocess
    f = out / "stages" / "design.json"
    if args.resume and f.exists():
        return None, f
    f.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "scenarios.checks", "--scenario", args.scenario, "--difficulty", level,
           "--seed", str(args.seed), "--out", str(f)]
    print(f"[validate] {level} | design: started in a subprocess (limit {args.design_timeout:.0f} s)", flush=True)
    return subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True), f


def finish_design(level: str, proc, f: Path) -> dict | None:
    import subprocess
    if proc is None:
        print(f"[validate] {level} | design: reused {f.relative_to(ROOT)}", flush=True)
        return json.loads(f.read_text())
    t0 = time.monotonic()
    while True:
        try:
            log, _ = proc.communicate(timeout=HEARTBEAT_S)
            break
        except subprocess.TimeoutExpired:
            waited = time.monotonic() - t0
            print(f"[validate] {level} | design: still running ({waited:.0f} s after the scene stages)", flush=True)
            if waited > args.design_timeout:
                proc.kill()
                print(f"[validate] {level} | design: KILLED after {waited:.0f} s", flush=True)
                return None
    print((log or "").strip(), flush=True)
    return json.loads(f.read_text()) if proc.returncode == 0 and f.exists() else None


def main() -> int:
    summary = {}
    # All CPU-physics camera stages first: after a GPU-physics build in the same process, a later CPU build no longer
    # writes the arm's pose to USD (measured 2026-10-15: wrist render 436-696 mm away from the arm's FK).
    cam_passes = {}
    for d in args.difficulty:
        out = ROOT / "out" / "scenarios" / args.scenario / f"{d}_s{args.seed}"
        out.mkdir(parents=True, exist_ok=True)
        cam_passes[d] = staged(d, "cameras_cpu", out, lambda: run_once(generate(args.scenario, d, args.seed), out, device="cpu"))
    for d in args.difficulty:
        sc = generate(args.scenario, d, args.seed)
        out = ROOT / "out" / "scenarios" / args.scenario / f"{d}_s{args.seed}"
        out.mkdir(parents=True, exist_ok=True)
        t_level = time.monotonic()
        print(f"[validate] {args.scenario} {d} seed {args.seed}: {len(sc.objects)} objects, {len(sc.fixtures)} fixtures", flush=True)
        design_proc = start_design(d, out) if not args.no_design else None
        # physics on the GPU, as the live simulator runs
        res = staged(d, "physics_gpu", out, lambda: run_once(generate(args.scenario, d, args.seed), None))
        # cameras: a separate build with CPU physics (done first, above). Headless without the ROS bridge, GPU physics
        # does not update the USD pose of the wrist camera prim (measured: it stays at the all-zero joint pose).
        cam_pass = cam_passes[d]
        res["cameras"], res["wrist_inspection"] = cam_pass["cameras"], cam_pass["wrist_inspection"]
        res["cpu_physics_cross_check"] = {"problems": cam_pass["problems"],
                                          "max_rest_error_mm": max(v["moved_from_spawn_mm"] for v in cam_pass["objects"].values()),
                                          "robot_home_error_deg": cam_pass["robot"]["home_error_deg"]}
        res["problems"] += [f"camera pass: {p}" for p in cam_pass["problems"]]
        res["regenerated_identical"] = generate(args.scenario, d, args.seed).metadata() == sc.metadata()
        if args.reload:
            again = staged(d, "reload_gpu", out, lambda: run_once(generate(args.scenario, d, args.seed), None))
            dev = max(float(np.linalg.norm(np.subtract(res["objects"][k]["settled_xyz"], again["objects"][k]["settled_xyz"])))
                      for k in res["objects"]) * 1000
            res["reload"] = {"max_settled_pose_difference_mm": round(dev, 3), "reproducible": dev <= 2.0,
                             "problems_second_build": again["problems"]}
            if dev > 2.0:
                res["problems"].append(f"reload differs by {dev:.1f} mm")
        if args.drop_mm:
            drop = staged(d, "drop_gpu", out, lambda: run_once(generate(args.scenario, d, args.seed), None, args.drop_mm))
            res["drop_test"] = {"drop_mm": args.drop_mm, "problems": drop["problems"],
                                "max_rest_error_mm": max(v["moved_from_spawn_mm"] for v in drop["objects"].values()),
                                "min_fall_measured_mm": min(v["moved_while_settling_mm"] for v in drop["objects"].values()),
                                "max_tilt_deg": max(v["tilt_deg"] for v in drop["objects"].values())}
            res["problems"] += [f"drop test: {p}" for p in drop["problems"]]
        meta = sc.metadata()
        if not args.no_design:
            res["design"] = finish_design(d, *design_proc)
            if res["design"] is None:
                res["problems"].append("design checks did not finish (see the log)")
            else:
                res["problems"] += [f"design: {p}" for p in res["design"]["problems"]]
                for r in meta["regions"]:  # receiving regions validated with the unchanged planner (held probe container)
                    v = res["design"]["placement"].get(r["id"])
                    r["validated_placement"] = None if v is None else {"feasible_yaws": v["feasible_yaws"], "probe": v["probe"]}
                for o in meta["objects"]:
                    o["validated_grasp_yaws"] = res["design"]["grasp"][o["id"]]["feasible_yaws"]
                    o["table_camera_top_visible"] = res["design"]["table_camera_top_visible"][o["id"]]
        meta["settled_ground_truth"] = {k: {kk: v[kk] for kk in ("settled_xyz", "settled_yaw_deg", "tilt_deg")}
                                        for k, v in res["objects"].items()}
        res["seconds_level"] = round(time.monotonic() - t_level, 1)
        (out / "metadata.json").write_text(json.dumps(meta, indent=1))
        (out / "validation.json").write_text(json.dumps(res, indent=1))
        summary[d] = {"objects": len(sc.objects), "problems": res["problems"], "cameras": res.get("cameras"),
                      "robot": res["robot"], "reload": res.get("reload"), "drop_test": res.get("drop_test"),
                      "seconds": res["seconds_level"]}
        print(f"[validate] {d}: {len(res['problems'])} problem(s) in {res['seconds_level']} s {res['problems'][:6]}", flush=True)
    (ROOT / "out" / "scenarios" / args.scenario / f"summary_s{args.seed}.json").write_text(json.dumps(summary, indent=1))
    app.close()
    return 0 if all(not v["problems"] for v in summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
