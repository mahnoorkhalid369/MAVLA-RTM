"""Checkpoint 3 (evaluation): live wrist-camera inspection of one object through the inspect capability (no
supervisor), scored against simulator ground truth (EVALUATION ONLY: read here, never by the agent code).

    (sim running, ROS sourced)  .venv-seg/bin/python -m evaluation.inspect_check --target "red cube" --truth red_cube --out out/active/cp3.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--truth", required=True, help="simulator object name (scoring only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from agent import capabilities as caps
    from agent.ask import ScriptedResponder
    from config.agent import Budgets
    from agent.detector import YoloDetector
    from agent.eventlog import EventLog
    from agent.ros_bridge import RosBridge
    from agent.tools import Tools
    from agent.vlm_roles import GemmaVLM, Roles
    from agent.world_state import TaskState
    from evaluation.dual_camera_check import Truth
    from execution.ros_executor import RosExecutor
    from sim.scene_objects import OBJECTS

    out = Path(a.out)
    run = out.parent / (out.stem + "_run")
    (run / "images").mkdir(parents=True, exist_ok=True)
    log = EventLog(run / "events.jsonl", echo=True)
    vlm = GemmaVLM()
    det = YoloDetector(str(ROOT / "out/seg/runs/yolo11s_seg_r1/weights/best.pt"))
    bridge, ex = RosBridge(), RosExecutor()
    truth = Truth(bridge.node)
    assert bridge.ready(300) and ex.wait_ready(120)
    while ex.traj_pub.get_subscription_count() == 0:
        ex.spin(0.2)
    ex.wait_sim(2.0)
    tools = Tools(bridge, det, Roles(vlm, log), ex, log, run)
    ts = TaskState(f"inspect {a.target}")
    ctx = caps.Context(tools, log, run, ScriptedResponder(["cancel"]), Budgets(), ts)
    t0 = time.monotonic()
    r_obs = caps.run_observe(ctx, {}, clear_view=False)
    t_global = round(time.monotonic() - t0, 1)
    r = caps.run_inspect(ctx, {"target": a.target})
    g = np.array(truth.latest["objects"][a.truth]["position"]) if truth.latest else None
    top = g[2] + OBJECTS[a.truth].size[2] / 2 if g is not None else None
    d = r.data
    rep = {"target": a.target, "observe": r_obs.summary, "global_seconds": t_global, "outcome": r.outcome, "summary": r.summary,
           "data": d, "truth_centre": None if g is None else g.tolist()}
    if g is not None and "wrist_centre" in d:
        rep["table_xy_err_mm"] = round(float(np.hypot(*(np.array(d["table_centre"][:2]) - g[:2]))) * 1000, 1)
        rep["wrist_xy_err_mm"] = round(float(np.hypot(*(np.array(d["wrist_centre"][:2]) - g[:2]))) * 1000, 1)
        rep["wrist_height_err_mm"] = round((d["wrist_height"] - OBJECTS[a.truth].size[2]) * 1000, 1)
        rep["wrist_footprint_err_mm"] = [round((f - s) * 1000, 1) for f, s in zip(sorted(d["wrist_footprint"]), sorted(OBJECTS[a.truth].size[:2]))]
    rep["safety_events"] = list(ex.safety_events)
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "data"}, indent=1, default=float))
    log.close()
    bridge.close()
    return 0 if r.outcome == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
