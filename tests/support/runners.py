"""Runners for the agent tests: run the pick-and-place workflow or the goal supervisor on fake tools and
return (report, events). Moved verbatim from tests/integration/test_agent.py (TASK, run -> run_workflow) and
tests/integration/test_supervisor_goal.py (plan, run -> run_supervisor)."""

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


def scripted_step(c: str, a: dict) -> dict:
    """A scripted Gemma step: Gemma names the worker that implements the operation (tests only; production code never
    infers a worker)."""
    from agent.capabilities import SPECS
    return {"do": c, **({"worker": SPECS[c].worker} if c in SPECS and "worker" not in a else {}), **a}


def plan(*steps, reason="", success=None):
    """A scripted Gemma plan. success: the original goal's criteria; by default (tests only) the measurable effects of
    the steps, as a Gemma that states its goal would."""
    from agent.goals import derive_conditions
    from agent.vlm_roles import PlanStep
    sts = [scripted_step(c, a) for c, a in steps]
    if success is None:
        success = derive_conditions([(p.capability, p.args) for p in (PlanStep.model_validate(x) for x in sts)])
    return GoalPlan(goal="g", steps=sts, success=success, reason=reason)


def problems(t):
    """Scripted Gemma's decisions at the points that needed one (not the routine 'continue' after a subtask that went
    as planned)."""
    return [c for c in t.decide_calls if ", subtask achieved" not in c["event"] and ", subtask done" not in c["event"]]


def run_supervisor(tmp_path, tools, responder=None, instruction="do it"):
    log = EventLog(tmp_path / "events.jsonl")
    r = GoalSupervisor(tools, log, tmp_path, responder or ScriptedResponder(["cancel"])).run(instruction)
    log.close()
    return r, [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
