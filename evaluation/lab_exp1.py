"""Laboratory Experiment 1 (perception baseline, no grasp / placement) on laboratory/easy, under the approved protocol.

    source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=42
    .venv-seg/bin/python evaluation/lab_exp1.py --out out/lab_exp1/run1 [--seed 42]

One agent session (evaluation/lab_exp1_agent.py = agent.run unchanged + runtime guards) gets six instructions:
  Phase A (static, motion gate closed)   1-2  "Describe the objects on the table." (twice: identity stability)
  Phase B (wrist inspection)             3-4  inspections; the gate opens only for that instruction, after a preflight
  Phase C (reasoning, gate closed)       5-6  sample-ID question (unsupported capability), tube / vial question
Preflight before each inspection (all must pass, else the instruction is skipped and recorded):
  robot at HOME (every joint <= 1.5 deg) and gripper open (finger <= 2 deg) from /joint_states; the agent holds nothing
  (previous result); objects unmoved since the start (ground truth, <= 2 mm); a collision-free wrist viewpoint for the
  intended container exists in the GROUND-TRUTH scene (existing ViewpointPlanner, evaluation-side); the simulator's
  command interface (joint-limit safety layer) is up and the agent's executor is ready.
  The agent then plans its own viewpoint against its PERCEIVED obstacles and executes through RosExecutor + the
  simulator safety layer (unchanged); the harness cannot sit between that plan and its execution.
Stop conditions (checked live during motion from /joint_states + ground truth, and after every instruction):
  an object moved > 3 mm (unexpected contact), the gripper closed (> 2 deg), no joint state for > 3 s while moving
  (connection lost), an "unsafe" outcome or safety-layer rejection, robot not back at HOME after an inspection.
  On a stop both processes are terminated at once; failed motions are never retried (agent: max_replans = 0, agent/controls.py).
Outputs: <out>/NN/ (agent per instruction), protocol.jsonl (every gate / preflight / stop decision), phaseB/
(joint + truth recordings, collision re-check), scoring_only/ (ground truth: never read by the agent), score.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# laboratory/easy seed 42: purple caps = tube in B4 and the vial in buffer spot T3; white caps = tubes in A4 (front
# row) and B1. Each reference is unique in that layout; `target` says which container the harness expects (scoring
# and preflight only - the agent resolves the words itself).
INSTRUCTIONS = [
    ("A", "Describe the objects on the table.", None),
    ("A", "Describe the objects on the table.", None),
    ("B", "Inspect the purple-capped vial in the buffer tray.", {"container": "vial", "cap_colour": "purple"}),
    ("B", "Inspect the white-capped tube at the front of the incoming rack.", {"container": "tube", "cap_colour": "white", "row": "A"}),
    ("C", "What sample ID is printed on the cap of the purple-capped vial?", None),
    ("C", "Which objects are sample tubes and which are vials?", None),
]
PER_INSTRUCTION_S = 900
HOME_TOL_DEG, OPEN_FINGER_DEG, UNMOVED_MM, CONTACT_MM, STALE_S = 1.5, 2.0, 2.0, 3.0, 3.0


class Run:
    def __init__(self, out: Path, seed: int) -> None:
        self.out, self.seed = out, seed
        self.scoring = out / "scoring_only"
        self.scoring.mkdir(parents=True, exist_ok=True)
        (out / "phaseB").mkdir(exist_ok=True)
        self.agent = self.sim = None
        self.n = 0
        self.stopped = None
        self.initial_truth = None

    def log(self, **kw) -> None:
        kw = {"wall": time.strftime("%H:%M:%S"), **kw}
        with open(self.out / "protocol.jsonl", "a") as f:
            f.write(json.dumps(kw, default=str) + "\n")
        print(f"[exp1] {json.dumps(kw, default=str)[:300]}", flush=True)

    # ------------------------------------------------------------------------------ processes
    def start(self) -> bool:
        flag = self.out / "models_loaded.flag"
        self.agent = subprocess.Popen([str(ROOT / ".venv-seg/bin/python"), "-u", "-m", "evaluation.lab_exp1_agent", "--out",
                                       str(self.out), "--sim-flag", str(flag), "--ask", "scripted", "--live-dir", ""], cwd=ROOT,
                                      stdin=subprocess.PIPE, text=True, stdout=open(self.out / "agent.log", "w"),
                                      stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.monotonic()
        while not flag.exists():
            if self.agent.poll() is not None or time.monotonic() - t0 > 600:
                self.log(event="abort", reason="the agent did not load its models")
                return False
            time.sleep(1)
        self.log(event="models_loaded", seconds=round(time.monotonic() - t0))
        self.sim = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-u", "-m", "sim.run_scenario", "--scenario", "laboratory",
                                     "--difficulty", "easy", "--seed", str(self.seed), "--headless"], cwd=ROOT,
                                    stdout=open(self.out / "sim.log", "w"), stderr=subprocess.STDOUT,
                                    env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}, start_new_session=True)
        while "[agent] ready" not in (self.out / "agent.log").read_text():
            if self.agent.poll() is not None or self.sim.poll() is not None or time.monotonic() - t0 > 1500:
                self.log(event="abort", reason="agent or simulator did not get ready")
                return False
            time.sleep(2)
        self.log(event="ready", seconds=round(time.monotonic() - t0))
        return True

    def send(self, line: str) -> None:
        self.agent.stdin.write(line + "\n")
        self.agent.stdin.flush()

    def emergency_stop(self, reason: str) -> None:
        self.stopped = reason
        self.log(event="STOP", reason=reason)
        for p in (self.agent, self.sim):
            kill(p, signal.SIGKILL, 5)

    def close(self) -> None:
        if self.agent and self.agent.poll() is None and not self.stopped:
            try:
                self.send("")
                self.agent.wait(60)
            except Exception:
                pass
        kill(self.agent)
        kill(self.sim)

    # --------------------------------------------------------------------------------- state
    def probe(self, name: str) -> dict | None:
        path = self.scoring / f"probe_{name}.json"
        r = subprocess.run([str(ROOT / ".venv-seg/bin/python"), "-m", "evaluation.lab_probe", "snapshot", str(path)],
                           cwd=ROOT, timeout=120, capture_output=True)
        return json.loads(path.read_text()) if r.returncode == 0 and path.exists() else None

    @staticmethod
    def robot(p: dict) -> dict:
        from control.robot import ARM_HOME, ARM_JOINTS, GRIPPER_JOINT
        j = dict(zip(p["joints"]["name"], p["joints"]["position"]))
        q = np.array([j[n] for n in ARM_JOINTS])
        return {"home_error_deg": round(float(np.degrees(np.abs(q - np.asarray(ARM_HOME))).max()), 3),
                "finger_deg": round(float(np.degrees(j[GRIPPER_JOINT])), 2), "q": q.tolist()}

    def moved_mm(self, p: dict) -> dict:
        a, b = self.initial_truth["truth"]["objects"], p["truth"]["objects"]
        return {k: round(float(np.linalg.norm(np.subtract(b[k]["position"], a[k]["position"]))) * 1000, 2) for k in a if k in b}

    def check_state(self, when: str) -> dict:
        p = self.probe(when)
        if p is None or "joints" not in p or "truth" not in p:
            return {"ok": False, "reason": "no joint state / ground truth (connection lost?)"}
        r = self.robot(p)
        moved = self.moved_mm(p)
        worst = max(moved.items(), key=lambda kv: kv[1])
        problems = []
        if r["home_error_deg"] > HOME_TOL_DEG:
            problems.append(f"robot not at HOME ({r['home_error_deg']} deg)")
        if r["finger_deg"] > OPEN_FINGER_DEG:
            problems.append(f"gripper not open ({r['finger_deg']} deg)")
        if worst[1] > UNMOVED_MM:
            problems.append(f"{worst[0]} moved {worst[1]} mm since the start")
        return {"ok": not problems, "problems": problems, "robot": r, "max_object_motion": worst}

    # ------------------------------------------------------------------------------ preflight
    def gt_viewpoint(self, target: dict, q_now) -> dict:
        """Evaluation-side check: does a collision-free wrist viewpoint for the intended container exist in the
        ground-truth scene (existing ViewpointPlanner, ground-truth colliders)?"""
        from perception.viewpoints import ViewpointPlanner
        from control.collision import Box
        from sim.scenario_lib import generate
        from sim.scenario_lib.core import scenario_parts
        sc = generate("laboratory", "easy", self.seed)
        obj = next(o for o in sc.objects if o.attributes.get("container") == target["container"]
                   and o.attributes.get("cap_colour") == target["cap_colour"]
                   and (not target.get("row") or str(o.attributes.get("slot", "")).startswith(target["row"])))
        fx, ob = scenario_parts(sc)
        boxes = [Box(w.owner, w.centre, w.R, w.half) for w in fx + ob]
        plan = ViewpointPlanner().plan(np.array([obj.pos[0], obj.pos[1], obj.pos[2] + obj.height]), np.asarray(q_now), boxes,
                                       target_box_name=obj.id)
        return {"target_id": obj.id, "sample_id": obj.attributes["sample_id"], "ok": plan.ok, "plan": plan.summary()}

    def preflight(self, target: dict) -> dict:
        st = self.check_state(f"preflight_{self.n + 1:02d}")
        prev = json.loads((self.out / f"{self.n:02d}" / "result.json").read_text()) if self.n else {}
        checks = {"state": st, "agent_holds_nothing": prev.get("holding") is None,
                  "safety_layer_up": "listening on /joint_command" in (self.out / "sim.log").read_text(),
                  "executor_ready": "[agent] ready" in (self.out / "agent.log").read_text()}
        if st["ok"]:
            checks["gt_viewpoint"] = self.gt_viewpoint(target, st["robot"]["q"])
        ok = (st["ok"] and checks["agent_holds_nothing"] and checks["safety_layer_up"] and checks["executor_ready"]
              and checks.get("gt_viewpoint", {}).get("ok", False))
        return {"ok": ok, **checks}

    # ------------------------------------------------------------------------------- instructions
    def instruct(self, text: str, motion: bool) -> dict | None:
        self.n += 1
        res = self.out / f"{self.n:02d}" / "result.json"
        rec = rec_f = None
        if motion:
            rec_f = self.out / "phaseB" / f"motion_{self.n:02d}.jsonl"
            rec = subprocess.Popen([str(ROOT / ".venv-seg/bin/python"), "-m", "evaluation.lab_probe", "record", str(rec_f)],
                                   cwd=ROOT, start_new_session=True)
            time.sleep(3)
            self.send("@motion on")
        self.log(event="instruction", index=self.n, text=text, motion_gate=motion)
        t0 = time.monotonic()
        self.send(text)
        try:
            while not res.exists():
                if self.agent.poll() is not None:
                    self.emergency_stop("the agent exited during the instruction")
                    return None
                if time.monotonic() - t0 > PER_INSTRUCTION_S:
                    self.emergency_stop(f"instruction {self.n} timed out")
                    return None
                if motion:
                    why = self.live_check(rec_f, t0)
                    if why:
                        self.emergency_stop(why)
                        return None
                time.sleep(1)
        finally:
            if motion and not self.stopped:
                self.send("@motion off")
            if rec is not None:
                kill(rec, signal.SIGINT, 10)
        r = json.loads(res.read_text())
        self.log(event="result", index=self.n, status=r.get("status"), outcome=r.get("outcome"), seconds=r.get("seconds"),
                 holding=r.get("holding"))
        return r

    def live_check(self, rec_f: Path, t0: float) -> str | None:
        if not rec_f.exists():
            return None
        lines = rec_f.read_text().splitlines()[-400:]
        recs = [json.loads(x) for x in lines if x.strip().endswith("}")]
        joints = [r for r in recs if r["kind"] == "joints"]
        truth = [r for r in recs if r["kind"] == "truth"]
        if time.monotonic() - t0 > 10 and (not joints or time.time() - joints[-1]["wall"] > STALE_S):
            return "no joint state for > 3 s during motion (connection lost?)"
        if joints:
            from control.robot import GRIPPER_JOINT
            j = dict(zip(joints[-1]["name"], joints[-1]["position"]))
            if np.degrees(j[GRIPPER_JOINT]) > OPEN_FINGER_DEG:
                return f"gripper closed unexpectedly ({np.degrees(j[GRIPPER_JOINT]):.1f} deg)"
        if truth:
            moved = self.moved_mm({"truth": truth[-1]})
            k, mm = max(moved.items(), key=lambda kv: kv[1])
            if mm > CONTACT_MM:
                return f"unexpected contact: {k} moved {mm} mm during motion"
        return None

    def after(self, r: dict, motion: bool) -> bool:
        """Stop conditions after an instruction; returns False if the run must stop."""
        ev = [json.loads(x) for x in (self.out / f"{self.n:02d}" / "events.jsonl").read_text().splitlines()]
        safety = [e for e in ev if e["kind"] == "tool" and e.get("tool") == "execute" and "safety" in str(e.get("reason", "")).lower()]
        st = self.check_state(f"after_{self.n:02d}")
        self.log(event="post_check", index=self.n, state=st, safety_rejections=len(safety))
        if r.get("outcome") == "unsafe" or safety:
            self.emergency_stop(f"unsafe outcome / safety-layer rejection in instruction {self.n}")
            return False
        if not st["ok"]:
            self.emergency_stop(f"state after instruction {self.n}: {st.get('problems') or st.get('reason')}")
            return False
        if motion:
            self.recheck_path()
        return True

    def recheck_path(self) -> None:
        """Re-check the EXECUTED joint trajectory (recorded /joint_states) against the ground-truth colliders: zero
        margins = actual contact, default margins = the planner's clearance."""
        from control.collision import Box, CollisionChecker, Margins
        from control.robot import ARM_JOINTS
        from sim.scenario_lib import generate
        from sim.scenario_lib.core import scenario_parts
        rec_f = self.out / "phaseB" / f"motion_{self.n:02d}.jsonl"
        recs = [json.loads(x) for x in rec_f.read_text().splitlines() if x.strip().endswith("}")]
        qs = [[dict(zip(r["name"], r["position"]))[n] for n in ARM_JOINTS] for r in recs if r["kind"] == "joints"]
        fx, ob = scenario_parts(generate("laboratory", "easy", self.seed))
        boxes = [Box(w.owner, w.centre, w.R, w.half) for w in fx + ob]
        zero = CollisionChecker(boxes, margins=Margins(obstacle=0.0, gripper_obstacle=0.0, floor=0.0, pads_floor=0.0, self_=0.0))
        default = CollisionChecker(boxes)
        qs = np.asarray(qs[::3])  # ~10 Hz
        contact = zero.check_path(qs) if len(qs) else None
        margin = default.check_path(qs) if len(qs) else None
        out = {"index": self.n, "samples": len(qs),
               "contact_free": None if contact is None else contact.ok,
               "contacts": [] if contact is None else [str(c) for c in contact.contacts[:5]],
               "within_planner_margins": None if margin is None else margin.ok,
               "margin_violations": [] if margin is None else [str(c) for c in margin.contacts[:5]]}
        (self.out / "phaseB" / f"path_check_{self.n:02d}.json").write_text(json.dumps(out, indent=1))
        self.log(event="executed_path_check", **out)
        if contact is not None and not contact.ok:
            self.emergency_stop(f"executed path intersects the ground-truth scene: {out['contacts'][:2]}")


def kill(p, sig=signal.SIGINT, wait=30):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "42" or not os.environ.get("AMENT_PREFIX_PATH"):
        print("source /opt/ros/jazzy/setup.bash and export ROS_DOMAIN_ID=42 first", flush=True)
        return 2
    busy = subprocess.run(["pgrep", "-f", "sim.run_live|sim.run_scenario|validate_scenarios"], capture_output=True, text=True)
    if busy.stdout.strip():
        print(f"another Isaac Sim process is running (pids {busy.stdout.split()}): one instance at a time", flush=True)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    run = Run(out, a.seed)
    try:
        if not run.start():
            return 1
        shutil.copy(ROOT / "out" / "scenarios" / "live" / f"laboratory_easy_s{a.seed}.json", run.scoring / "scenario_metadata.json")
        first = run.probe("initial")
        if first is None:
            run.emergency_stop("no initial joint state / ground truth")
            return 1
        run.initial_truth = first
        (run.scoring / "truth.json").write_text(json.dumps({k: v for k, v in first["truth"].items() if k != "kind"}, indent=1))
        run.log(event="initial_state", robot=run.robot(first))
        for phase, text, target in INSTRUCTIONS:
            if run.stopped:
                break
            motion = phase == "B"
            if motion:
                pf = run.preflight(target)
                run.log(event="preflight", index=run.n + 1, **pf)
                if not pf["ok"]:
                    run.log(event="skipped", index=run.n + 1, text=text, reason="preflight failed: motion not executed")
                    continue
            r = run.instruct(text, motion)
            if r is None or not run.after(r, motion):
                break
    finally:
        run.close()
    if not run.stopped:
        run.log(event="completed", instructions=run.n)
    subprocess.run([sys.executable, str(ROOT / "evaluation" / "lab_exp1_score.py"), "--run", str(out), "--meta",
                    str(run.scoring / "scenario_metadata.json"), "--truth", str(run.scoring / "truth.json")], cwd=ROOT)
    return 0 if not run.stopped else 1


if __name__ == "__main__":
    raise SystemExit(main())
