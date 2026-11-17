"""The only path from the agent to robot motion: Tools.execute -> frozen execution/ros_executor.RosExecutor
(the simulator's safety layer checks every command independently)."""

from __future__ import annotations

import time
from types import SimpleNamespace


class ExecutionTools:
    def execute(self, steps, label: str):
        """Execute planned motion steps (Tools.controls.motion "real"), or log and refuse them ("dry_run",
        "blocked", agent/controls.py) with a failed result of the same fields as execution.ros_executor's."""
        names = [getattr(x, "label", str(x)) for x in steps]
        if self.controls.motion != "real":
            return self._not_executed(names, label)
        self.controls.notify("motion_allowed", label=label, steps=names)
        t0 = time.monotonic()
        self.log("tool", tool="execute_start", label=label, steps=[getattr(x, "label", str(x)) for x in steps])
        # RosExecutor (frozen) reads its sim clock only while spinning: refresh it after the VLM pauses,
        # otherwise its first step starts from a stale clock (wrong tracking numbers, early step end)
        self.ex.spin(0.3)
        self.motion_count += 1  # counted before executing: a failed command may still have moved the robot
        res = self.ex.execute(steps)
        self.log("tool", tool="execute", label=label, ok=res.ok, failed_step=res.failed_step, reason=res.reason,
                 steps=res.steps, seconds=round(time.monotonic() - t0, 2))
        self.controls.notify("motion_result", label=label, ok=res.ok, reason=res.reason, failed_step=res.failed_step)
        return res

    def _not_executed(self, names: list, label: str):
        """No robot command is sent; the motion counter is unchanged (the robot did not move)."""
        if self.controls.motion == "dry_run":
            reason = "dry run: motion planned and logged, not executed"
            self.log("tool", tool="execute_dry_run", label=label, steps=names)
        else:
            reason = self.controls.blocked_reason
            self.log("tool", tool="execute_refused", label=label, reason=reason)
        self.controls.notify("motion_refused", label=label, steps=names)
        return SimpleNamespace(ok=False, failed_step=label, reason=reason, log=[], steps=[])
