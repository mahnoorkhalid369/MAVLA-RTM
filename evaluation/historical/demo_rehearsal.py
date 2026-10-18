"""Demo rehearsal: N runs per instruction, a fresh simulator per run, scored against simulator ground
truth AFTER the agent has finished (EVALUATION ONLY: the agent process never sees ground truth).

Per run: start the agent (loads Gemma, then YOLO) -> it flags "models loaded" -> start Isaac Sim
(headless) -> record the objects' true poses -> the agent runs the instruction and exits -> record
the true poses again -> score -> stop the simulator.

    source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=42
    .venv-seg/bin/python -m evaluation.historical.demo_rehearsal --runs 3 --instructions 1 2 3

Scoring (same tolerances as Baseline A and the agent: xy 10 mm, z 5 mm, tilt 5 deg):
  correct object   = the object that moved most is the instruction's object
  correct place    = "on": the object's bottom on the reference's top, centred within xy 10 mm;
                     relations: the moved object stands on the floor, beside the reference (gap <= 0.15 m)
                     and in the stated direction
  verdict vs truth = agent COMPLETED while truth fails = false done; agent FAILED while truth passes = false failed
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# instruction -> (object, relation, reference) in SIMULATOR names: scoring key only
INSTRUCTIONS = {
    1: ("Put the red cube on the green cube.", "red_cube", "on", "green_cube"),
    2: ("Pick up the yellow cube and put it on the tan box.", "yellow_cube", "on", "tan_box_wide"),
    3: ("Put the orange cube on the purple cylinder.", "orange_cube", "on", "purple_cylinder"),
    4: ("Put the pink cube next to the white tall cylinder.", "pink_cube", "next_to", "white_tall_cylinder"),
}
TOL_XY, TOL_Z, TOL_TILT = 0.010, 0.005, 5.0


def tilt_deg(q) -> float:
    w, x, y, z = q
    zz = 1 - 2 * (x * x + y * y)  # world z component of the body's z axis
    return float(np.degrees(np.arccos(np.clip(zz, -1, 1))))


class Truth:
    """Subscribes to /sim/ground_truth/objects (evaluation only)."""

    def __init__(self) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("demo_rehearsal_scorer")
        self.latest = None
        self.node.create_subscription(String, "sim/ground_truth/objects", lambda m: setattr(self, "latest", json.loads(m.data)), 10)
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.node)
        threading.Thread(target=self.ex.spin, daemon=True).start()

    def get(self, after: float = 0.0, timeout: float = 600.0) -> dict | None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.latest and self.latest["sim_time"] > after:
                return self.latest
            time.sleep(0.2)
        return None


def size_of(name: str):
    from sim.scene_objects import OBJECTS
    return OBJECTS[name].size


def score(spec, before: dict, after: dict, result: dict) -> dict:
    _, obj, rel, ref = spec
    b, a = before["objects"], after["objects"]
    moved = {n: float(np.linalg.norm(np.subtract(a[n]["position"], b[n]["position"]))) for n in a}
    mover = max(moved, key=moved.get)
    s = {"moved_most": mover, "moved_mm": round(moved[mover] * 1000, 1),
         "others_moved_mm": {n: round(d * 1000, 1) for n, d in moved.items() if n != mover and d > 0.005}}
    s["correct_object"] = mover == obj and moved[mover] > 0.02
    p, r = np.array(a[obj]["position"]), np.array(a[ref]["position"])
    h_obj, h_ref = size_of(obj)[2], size_of(ref)[2]
    s["tilt_deg"] = round(tilt_deg(a[obj]["quat_wxyz"]), 2)
    if rel == "on":
        xy = float(np.hypot(*(p[:2] - r[:2])))
        dz = float((p[2] - h_obj / 2) - (r[2] + h_ref / 2))
        s.update({"xy_error_mm": round(xy * 1000, 1), "z_error_mm": round(dz * 1000, 1)})
        s["correct_place"] = xy <= TOL_XY and abs(dz) <= TOL_Z and s["tilt_deg"] <= TOL_TILT
    else:
        gap = float(np.hypot(*(p[:2] - r[:2])) - (max(size_of(obj)[:2]) + max(size_of(ref)[:2])) / 2)
        on_floor = abs(p[2] - h_obj / 2) <= TOL_Z
        s.update({"gap_mm": round(gap * 1000, 1), "on_floor": bool(on_floor)})
        s["correct_place"] = on_floor and -0.002 <= gap <= 0.15 and s["tilt_deg"] <= TOL_TILT
        # placement accuracy against the agent's own planned spot
        plan = (result.get("plan") or {}).get("target_centre_mm")
        if plan:
            s["xy_error_mm"] = round(float(np.hypot(p[0] * 1000 - plan[0], p[1] * 1000 - plan[1])), 1)
    truth_ok = s["correct_object"] and s["correct_place"]
    st = result.get("status")
    s["truth_success"] = bool(truth_ok)
    s["agent_status"] = st
    s["false_done"] = st == "COMPLETED" and not truth_ok
    s["false_failed"] = st == "FAILED" and truth_ok
    s["asked"] = st == "CANCELLED"
    s["reference_disturbed_mm"] = round(moved[ref] * 1000, 1)
    return s


def kill_group(p, sig=signal.SIGINT, wait=30) -> None:
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(p.pid, sig)
        p.wait(wait)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def one_run(k: int, n: int, out: Path, weights: str, timeout_s: float, truth: "Truth") -> dict:
    spec = INSTRUCTIONS[k]
    rd = out / f"i{k}_run{n}"
    rd.mkdir(parents=True, exist_ok=True)
    flag = rd / "models_loaded.flag"
    env = {**os.environ}
    t0 = time.monotonic()
    agent = subprocess.Popen([str(ROOT / ".venv-seg/bin/python"), "-u", "-m", "agent.run", "--instruction", spec[0],
                              "--out", str(rd), "--weights", weights, "--sim-flag", str(flag), "--ask", "scripted", "--live-dir", "", "--motion", "real"],
                             cwd=ROOT, stdout=open(rd / "agent.log", "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
    sim = None
    rec = {"instruction": spec[0], "key": k, "run": n, "dir": str(rd)}
    try:
        while not flag.exists():
            if agent.poll() is not None:
                rec["error"] = "agent exited before loading the models"
                return rec
            if time.monotonic() - t0 > 600:
                rec["error"] = "models did not load in 10 min"
                return rec
            time.sleep(1)
        sim = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-u", "-m", "sim.run_live", "--headless"], cwd=ROOT,
                               stdout=open(rd / "sim.log", "w"), stderr=subprocess.STDOUT,
                               env={**env, "OMNI_KIT_ACCEPT_EULA": "YES"}, start_new_session=True)
        truth.latest = None  # only messages from this run's simulator
        before = truth.get(after=1.0)  # after the objects have settled for 1 s of sim time
        if before is None:
            rec["error"] = "no ground truth from the simulator"
            return rec
        try:
            agent.wait(timeout_s)
        except subprocess.TimeoutExpired:
            rec["error"] = f"agent timed out after {timeout_s:.0f} s"
            kill_group(agent)
        time.sleep(2.0)
        after = truth.get(after=truth.latest["sim_time"] if truth.latest else 0.0, timeout=30)
        result = json.loads((rd / "result.json").read_text()) if (rd / "result.json").exists() else {"status": "NO_RESULT"}
        rec["result_status"] = result.get("status")
        rec["message"] = result.get("message")
        rec["failure"] = result.get("failure")
        rec["timeline"] = result.get("timeline")
        rec["verifications"] = result.get("verifications")
        if after:
            rec["score"] = score(spec, before, after, result)
            (rd / "truth_before.json").write_text(json.dumps(before))
            (rd / "truth_after.json").write_text(json.dumps(after))
        rec["seconds"] = round(time.monotonic() - t0, 1)
        return rec
    finally:
        kill_group(agent)
        kill_group(sim)
        time.sleep(5)
        (rd / "score.json").write_text(json.dumps(rec, indent=1, default=str))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instructions", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "out/demo" / time.strftime("%Y-%m-%d") / "rehearsal"))
    from config.models import YOLO_WEIGHTS
    ap.add_argument("--weights", default=str(YOLO_WEIGHTS))
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = Path(a.out) / (a.tag or time.strftime("cycle_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    truth = Truth()
    for k in a.instructions:
        for n in range(1, a.runs + 1):
            r = one_run(k, n, out, a.weights, a.timeout, truth)
            recs.append(r)
            s = r.get("score", {})
            print(f"[rehearsal] i{k} run{n}: agent {r.get('result_status')} | truth success {s.get('truth_success')} | "
                  f"obj {s.get('correct_object')} place {s.get('correct_place')} xy {s.get('xy_error_mm')} mm | "
                  f"false done {s.get('false_done')} | {r.get('error', '')}", flush=True)
            (out / "summary.json").write_text(json.dumps(recs, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
