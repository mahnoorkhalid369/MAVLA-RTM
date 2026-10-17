"""One robot session = one agent process: the state that lives across instructions, and running one instruction.

Ownership:
    Session      the Tools object (created on the first instruction, kept for the session) and the counters
    Tools        world model, held object, description memory, clarifications, last referent (agent/tools/)
    per instruction (new each time): event log, Gemma roles (call counters), GoalSupervisor, TaskState ledger,
                 LangGraph thread, run directory
Nothing is written to disk for the next session: a new process starts with an empty world and nothing held.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class Session:
    """One robot session = one agent process with one simulator connection. The Tools object - world model,
    held object, last referent, remembered clarifications - lives as long as the session, so a follow-up such as
    "put it in the processing rack" can refer to the previous instruction's object. Each instruction still gets
    its own event log, VLM roles (call counters), supervisor, task ledger, goal conditions and LangGraph thread.
    The held-object claim is re-verified from finger feedback at the start of every instruction (and again before
    a place). Nothing is written to disk for the next session: a new process starts with an empty world and
    nothing held."""

    def __init__(self, bridge, detector, vlm, executor, responder, live=None, memory=None) -> None:
        self.id = time.strftime("session_%Y%m%d_%H%M%S")
        self.bridge, self.detector, self.vlm, self.executor = bridge, detector, vlm, executor
        self.responder, self.live, self.memory = responder, live, memory
        self.tools = None
        self.count = 0

    def run(self, instruction: str, out: Path) -> dict:
        from agent.eventlog import EventLog
        from agent.supervisor import GoalSupervisor
        from agent.tools import Tools
        from agent.vlm_roles import Roles

        (out / "images").mkdir(parents=True, exist_ok=True)
        log = EventLog(out / "events.jsonl", echo=True)
        self.count += 1
        if self.live:
            self.live.start_run(out, instruction)
        try:
            log("session", session_id=self.id, instruction_index=self.count, reused_state=self.tools is not None)
            roles = Roles(self.vlm, log)
            if self.tools is None:
                self.tools = Tools(self.bridge, self.detector, roles, self.executor, log, out, self.live, self.memory)
                self.tools.revalidate_held()
            else:
                self.tools.begin_instruction(log, roles, out)
            report = GoalSupervisor(self.tools, log, out, self.responder).run(instruction, thread_id=f"{self.id}_{self.count:02d}")
        except Exception as exc:  # an unexpected error is a failure, never a success
            log("error", error=repr(exc))
            report = {"instruction": instruction, "status": "FAILED", "failure": {"cls": "internal", "reason": repr(exc)},
                      "message": f"Failed (internal error): {exc!r}. I did not report success."}
        finally:
            log.close()
        report["session"] = {"id": self.id, "instruction_index": self.count}
        (out / "result.json").write_text(json.dumps(report, indent=1, default=str))
        return report
