"""The only path from the agent to robot motion: Tools.execute -> frozen execution/ros_executor.RosExecutor
(the simulator's safety layer checks every command independently)."""

from __future__ import annotations

import time


class ExecutionTools:
    def execute(self, steps, label: str):
        t0 = time.monotonic()
        self.log("tool", tool="execute_start", label=label, steps=[getattr(x, "label", str(x)) for x in steps])
        # RosExecutor (frozen) reads its sim clock only while spinning: refresh it after the VLM pauses,
        # otherwise its first step starts from a stale clock (wrong tracking numbers, early step end)
        self.ex.spin(0.3)
        self.motion_count += 1  # counted before executing: a failed command may still have moved the robot
        res = self.ex.execute(steps)
        self.log("tool", tool="execute", label=label, ok=res.ok, failed_step=res.failed_step, reason=res.reason,
                 steps=res.steps, seconds=round(time.monotonic() - t0, 2))
        return res
