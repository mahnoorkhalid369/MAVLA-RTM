"""Experiment 1 agent process: agent.run.main() unchanged, with experiment-level guards set at runtime (no agent
file is edited). The harness (evaluation/lab_exp1.py) drives it through the normal instruction prompt.

Guards (logged to <out>/harness_gates.jsonl):
  motion gate   every robot / gripper command goes through Tools.execute; it is refused unless the harness opened
                the gate (line "@motion on") after its own preflight for one inspection instruction. Phases A and C
                therefore cannot move the robot (not even through the supervisor's uncertainty-driven inspection).
  no retry      supervisor MAX_REPLANS = 0: a failed motion is never replanned or retried automatically.
  observation   pick / place / pick_place / explore / decompose are unavailable (planner menu marks them
  only          "not available"; their executors return "unsupported" if named anyway).
The process never reads ground truth: it is a normal agent session.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

GATE = {"motion": False, "opened_for": None}


def main() -> int:
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    gate_log = out / "harness_gates.jsonl"

    def record(**kw):
        with open(gate_log, "a") as f:
            f.write(json.dumps({"wall": time.strftime("%H:%M:%S"), **kw}) + "\n")

    import agent.ask as ask
    import agent.capabilities as caps
    import agent.supervisor as sup
    import agent.tools as tools_mod

    sup.MAX_REPLANS = 0
    for name in ("pick", "place", "pick_place", "explore", "decompose"):
        caps.SPECS[name] = dataclasses.replace(caps.SPECS[name], implemented=False)

    def disabled(name):
        def run(*a, **k):
            record(event="capability_refused", capability=name)
            return caps.StepResult("unsupported", f"{name} is disabled in Experiment 1 (observation only)")
        return run
    caps.run_pick, caps.run_place, caps.run_pick_place = disabled("pick"), disabled("place"), disabled("pick_place")
    caps.run_explore, caps.run_decompose = disabled("explore"), disabled("decompose")

    original_execute = tools_mod.Tools.execute

    def gated_execute(self, steps, label):
        names = [getattr(x, "label", str(x)) for x in steps]
        if not GATE["motion"]:
            record(event="motion_refused", label=label, steps=names)
            from types import SimpleNamespace  # same fields as execution.ros_executor.ExecutionResult
            self.log("tool", tool="execute_refused", label=label, reason="Experiment 1 motion gate closed")
            return SimpleNamespace(ok=False, failed_step=label, reason="motion gate closed (Experiment 1: no motion in this phase)",
                                   log=[], steps=[])
        record(event="motion_allowed", label=label, steps=names, instruction=GATE["opened_for"])
        res = original_execute(self, steps, label)
        record(event="motion_result", label=label, ok=res.ok, reason=res.reason, failed_step=res.failed_step)
        return res
    tools_mod.Tools.execute = gated_execute

    original_get = ask.LineReader.get

    def get(self, timeout=None):
        while True:
            line = original_get(self, timeout)
            if line is None or not line.startswith("@"):
                return line
            if line.strip() == "@motion on":
                GATE["motion"], GATE["opened_for"] = True, len(list(out.glob("[0-9][0-9]"))) + 1
            elif line.strip() == "@motion off":
                GATE["motion"], GATE["opened_for"] = False, None
            record(event="gate", directive=line.strip(), motion=GATE["motion"])
    ask.LineReader.get = get
    record(event="guards", max_replans=sup.MAX_REPLANS,
           unavailable=[n for n, s in caps.SPECS.items() if not s.implemented], motion=GATE["motion"])

    from agent.run import main as agent_main
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    return agent_main()


if __name__ == "__main__":
    raise SystemExit(main())
