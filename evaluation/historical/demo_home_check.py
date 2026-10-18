"""Live check of the return-to-HOME tools on the running simulator (no Gemma, no detector):
park away from HOME -> HOME refused with the gripper closed -> open -> return HOME -> verify from
joint states. Uses agent.tools.Tools exactly as the supervisor does.

    source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=42   (simulator already running)
    .venv-seg/bin/python -m evaluation.historical.demo_home_check --out out/demo/<date>/home_check
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    from agent.eventlog import EventLog
    from agent.ros_bridge import RosBridge
    from agent.tools import Tools
    from control.robot import ARM_HOME, GRIPPER_CLOSED, GRIPPER_OPEN, ARM_SPEED
    from control.motion import joint_move
    from control.skills import GripperStep, MoveStep
    from execution.ros_executor import RosExecutor

    log = EventLog(out / "events.jsonl", echo=True)
    bridge, ex = RosBridge(), RosExecutor()
    assert bridge.ready(300) and ex.wait_ready(120), "simulator not ready"
    while ex.traj_pub.get_subscription_count() == 0:
        ex.spin(0.2)
    ex.wait_sim(3.0)
    t = Tools(bridge, None, None, ex, log, out)
    res = {}
    # 1. park away from HOME (collision-checked, as before verification)
    q = np.array(ARM_HOME, float)
    q[0] = np.radians(135.0)
    q[1] = np.radians(-20.0)
    r = t.execute([MoveStep(joint_move(t.kin, bridge.q(), q, ARM_SPEED), "move away from HOME (test)")], "test setup")
    res["setup_ok"] = r.ok
    res["away_error_deg"] = round(float(np.degrees(np.abs(bridge.q() - np.array(ARM_HOME))).max()), 2)
    # 2. gripper closed -> HOME must be refused
    t.execute([GripperStep(GRIPPER_CLOSED, "close gripper (test)", 1.0)], "test setup")
    res["closed_refused"] = t.return_home()
    res["closed_refused"].pop("result", None)
    # 3. open -> HOME -> verify
    t.execute([GripperStep(GRIPPER_OPEN, "open gripper (test)", 1.0)], "test setup")
    t0 = time.monotonic()
    h = t.return_home()
    res["home_ok"], res["home_reason"], res["home_seconds"] = h["ok"], h.get("reason"), round(time.monotonic() - t0, 1)
    res["verify"] = t.verify_home()
    res["safety_events"] = list(ex.safety_events)
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    log.close()
    bridge.close()
    return 0 if res["home_ok"] and res["verify"]["pass"] and not res["closed_refused"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
