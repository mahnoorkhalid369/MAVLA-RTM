"""Experiment 1 agent process: agent.run.main() with experiment-level guards passed as explicit execution
controls (agent/controls.py; no agent function is replaced at runtime). The harness (evaluation/lab_exp1.py)
drives it through the normal instruction prompt.

Guards (logged to <out>/harness_gates.jsonl):
  motion gate   every robot / gripper command goes through Tools.execute; it is refused (motion "blocked") unless
                the harness opened the gate (line "@motion on") after its own preflight for one inspection
                instruction. Phases A and C therefore cannot move the robot (not even through the supervisor's
                uncertainty-driven inspection).
  no retry      max_replans = 0: a failed motion is never replanned or retried automatically.
  observation   pick / place / pick_place / explore / decompose are disabled (planner menu marks them
  only          "not available"; the supervisor refuses them as "unsupported" if named anyway).
The process never reads ground truth: it is a normal agent session.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

DISABLED = ("pick", "place", "pick_place", "explore", "decompose")


def main() -> int:
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    gate_log = out / "harness_gates.jsonl"
    opened_for = {"instruction": None}  # number of the instruction the gate was opened for

    def record(**kw):
        with open(gate_log, "a") as f:
            f.write(json.dumps({"wall": time.strftime("%H:%M:%S"), **kw}) + "\n")

    from agent.capabilities import SPECS
    from agent.controls import ExecutionControls

    def on_event(event, **kw):
        if event == "motion_allowed":
            kw["instruction"] = opened_for["instruction"]
        record(event=event, **kw)

    controls = ExecutionControls(motion="blocked", disabled=frozenset(DISABLED), max_replans=0,
                                 disabled_reason="in Experiment 1 (observation only)",
                                 blocked_reason="motion gate closed (Experiment 1: no motion in this phase)",
                                 on_event=on_event)

    def directive(line: str) -> None:
        if line.strip() == "@motion on":
            controls.motion = "real"
            opened_for["instruction"] = len(list(out.glob("[0-9][0-9]"))) + 1
        elif line.strip() == "@motion off":
            controls.motion = "blocked"
            opened_for["instruction"] = None
        record(event="gate", directive=line.strip(), motion=controls.motion == "real")

    record(event="guards", max_replans=controls.max_replans, unavailable=[n for n in SPECS if n in controls.disabled],
           motion=controls.motion == "real")

    from agent.run import main as agent_main
    return agent_main(controls=controls, on_directive=directive)


if __name__ == "__main__":
    raise SystemExit(main())
