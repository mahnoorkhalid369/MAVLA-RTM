"""Runners for the agent tests: run the pick-and-place workflow or the goal supervisor on fake tools and
return (report, events). Moved verbatim from tests/test_agent.py (TASK, run -> run_workflow) and
tests/test_supervisor_goal.py (plan, run -> run_supervisor)."""

import json

from agent.ask import ScriptedResponder
from agent.eventlog import EventLog
from agent.pick_place import PickPlaceWorkflow
from agent.supervisor import GoalSupervisor
from agent.vlm_roles import GoalPlan

TASK = {"object": {"text": "a"}, "relation": "on", "reference": {"text": "b"}, "frame": "robot"}


def run_workflow(tmp_path, tools, instruction="put a on b", responder=None, task=None, final_home=True):
    """Runs the pick-and-place skill (the planner's interpretation is given as `task`)."""
    from agent.ask import ScriptedResponder
    log = EventLog(tmp_path / "events.jsonl")
    wf = PickPlaceWorkflow(tools, log, tmp_path, responder or ScriptedResponder(["cancel"]), final_home=final_home)
    r = wf.run(instruction, task or TASK)
    log.close()
    return r, [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]


def plan(*steps, reason=""):
    return GoalPlan(goal="g", steps=[{"do": c, **a} for c, a in steps], success="s", reason=reason)


def run_supervisor(tmp_path, tools, responder=None, instruction="do it"):
    log = EventLog(tmp_path / "events.jsonl")
    r = GoalSupervisor(tools, log, tmp_path, responder or ScriptedResponder(["cancel"])).run(instruction)
    log.close()
    return r, [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
