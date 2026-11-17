"""The only path from the agent to robot motion: Tools.execute -> execution/ros_executor.RosExecutor
(the simulator's safety layer checks every command independently)."""

from __future__ import annotations

import time
import math
from agent.timing import timed, skill_state
from types import SimpleNamespace


class ExecutionTools:
    @timed("motion")
    def execute(self, steps, label: str):
        """Execute planned motion steps (Tools.controls.motion "real"), or log and refuse them ("dry_run",
        "blocked", agent/controls.py) with a failed result of the same fields as execution.ros_executor's."""
        names = [getattr(x, "label", str(x)) for x in steps]
        if self.controls.motion != "real":
            return self._not_executed(names, label)
        self.controls.notify("motion_allowed", label=label, steps=names)
        t0 = time.monotonic()
        self.log("tool", tool="execute_start", label=label, steps=[getattr(x, "label", str(x)) for x in steps])
        # RosExecutor reads its sim clock only while spinning: refresh it after the VLM pauses,
        # otherwise its first step starts from a stale clock (wrong tracking numbers, early step end)
        self.ex.spin(0.3)
        self.motion_count += 1  # counted before executing: a failed command may still have moved the robot
        # Only certified empty-hand observation/HOME motion preserves a placement receipt.
        observation_move = label in {"return HOME", "look at the placement", "verify placement", "final check",
                                     "inspect source from side", "return to source inspection"}
        finger = getattr(getattr(self, "bridge", None), "finger", None)
        empty = (self.held_object is None and getattr(self, "held_box", None) is None
                 and getattr(getattr(self, "world", None), "holding", None) is None
                 and getattr(self, "unknown_held", None) is None
                 and getattr(self, "unconfirmed_grasp", None) is None
                 and callable(finger) and math.degrees(finger()) <= getattr(self, "OPEN_FINGER_DEG", 2.))
        preserve = observation_move and empty
        if not preserve:
            self.scene_revision = getattr(self, "scene_revision", 0) + 1
        try:
            res = self.ex.execute(steps)
        except Exception:
            self.scene_revision = getattr(self, "scene_revision", 0) + 1
            raise
        if not res.ok:
            self.scene_revision = getattr(self, "scene_revision", 0) + 1
        self.log("tool", tool="execute", label=label, ok=res.ok, failed_step=res.failed_step, reason=res.reason,
                 steps=res.steps, seconds=round(time.monotonic() - t0, 2))
        for step in res.steps:
            if step.get("kind") == "move" and step.get("settling_wall_s", 0) > 0:
                start = step["start_s"] + step["motion_wall_s"]
                self.log("timing", category="settling", label=step["label"], state=skill_state.get(),
                         start_s=start, end_s=step["end_s"], seconds=step["end_s"]-start, ok=step.get("ok"))
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
