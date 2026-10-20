"""Execution controls (agent/controls.py): real / dry_run / blocked motion, disabled capabilities, the replanning
limit, harness directives on stdin, and the Experiment 1 wiring. No robot, no simulator, no model: the executor
is a fake that records what it is sent (or fails the test if it is touched at all)."""

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langgraph")

from agent.controls import ExecutionControls  # noqa: E402
from agent.tools import Tools  # noqa: E402
from tests.support.fakes import GoalTools  # noqa: E402
from tests.support.runners import plan  # noqa: E402

STEP = SimpleNamespace(label="move")
PP = {"object": "a", "relation": "on", "target": "b"}


class RecordingExecutor:
    """Stands in for execution.ros_executor.RosExecutor: records every command, moves nothing."""

    def __init__(self) -> None:
        self.spins, self.sent = 0, []

    def spin(self, seconds) -> None:
        self.spins += 1

    def execute(self, steps):
        self.sent.append(list(steps))
        return SimpleNamespace(ok=True, failed_step=None, reason="", steps=[], log=[])


class ForbiddenExecutor:
    """Any use of the executor is a test failure (blocked / dry-run must never reach it)."""

    def spin(self, seconds) -> None:
        raise AssertionError("the executor was touched")

    def execute(self, steps):
        raise AssertionError("a motion was sent to the executor")


def tools_with(executor, controls=None):
    events = []
    t = Tools(None, None, None, executor, lambda kind, **f: events.append({"kind": kind, **f}), Path("."),
              controls=controls)
    return t, events


def run_supervisor(tmp_path, tools, controls, instruction="do it"):
    from agent.ask import ScriptedResponder
    from agent.eventlog import EventLog
    from agent.supervisor import GoalSupervisor
    log = EventLog(tmp_path / "events.jsonl")
    r = GoalSupervisor(tools, log, tmp_path, ScriptedResponder(["cancel"]), controls=controls).run(instruction)
    log.close()
    return r, [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]


# ------------------------------------------------------------------------------------------- motion modes
def test_default_controls_are_normal_operation():
    c = ExecutionControls()
    assert (c.motion, c.disabled, c.max_replans, c.on_event) == ("real", frozenset(), 3, None)


def test_unknown_motion_mode_is_rejected():
    with pytest.raises(ValueError):
        ExecutionControls(motion="fast")


def test_tools_without_controls_execute_for_real():
    ex = RecordingExecutor()
    t, _ = tools_with(ex)
    assert t.controls.motion == "real" and t.execute([STEP], "test").ok and ex.sent == [[STEP]]


def test_real_mode_sends_the_steps_and_counts_the_motion():
    ex, seen = RecordingExecutor(), []
    t, ev = tools_with(ex, ExecutionControls(on_event=lambda e, **k: seen.append(e)))
    r = t.execute([STEP], "test")
    assert r.ok and ex.sent == [[STEP]] and ex.spins == 1 and t.motion_count == 1
    assert seen == ["motion_allowed", "motion_result"]
    assert [e["tool"] for e in ev] == ["execute_start", "execute"]


@pytest.mark.parametrize("mode,event", [("blocked", "execute_refused"), ("dry_run", "execute_dry_run")])
def test_blocked_and_dry_run_never_touch_the_executor(mode, event):
    seen = []
    t, ev = tools_with(ForbiddenExecutor(), ExecutionControls(motion=mode, on_event=lambda e, **k: seen.append((e, k))))
    r = t.execute([STEP], "test")
    assert r.ok is False and r.failed_step == "test" and r.steps == [] and r.log == []
    assert t.motion_count == 0  # the robot did not move: observations stay fresh
    assert [e["tool"] for e in ev] == [event]
    assert seen == [("motion_refused", {"label": "test", "steps": ["move"]})]
    assert ("dry run" in r.reason) == (mode == "dry_run")
    assert "safety" not in r.reason.lower()  # a refusal is not reported as a safety-layer rejection


def test_blocked_reason_is_the_configured_text():
    t, ev = tools_with(ForbiddenExecutor(), ExecutionControls(motion="blocked", blocked_reason="gate closed"))
    assert t.execute([STEP], "x").reason == "gate closed" and ev[0]["reason"] == "gate closed"


def test_motion_mode_can_change_while_the_process_runs():
    ex = RecordingExecutor()
    t, _ = tools_with(ex, ExecutionControls(motion="blocked"))
    assert not t.execute([STEP], "a").ok and ex.sent == []
    t.controls.motion = "real"
    assert t.execute([STEP], "b").ok and ex.sent == [[STEP]]
    t.controls.motion = "blocked"
    assert not t.execute([STEP], "c").ok and len(ex.sent) == 1


def test_return_home_through_a_blocked_gate_reports_failure_without_motion():
    """A whole tool that plans and executes (return_home) is refused at the gate, never sent."""
    import numpy as np
    from control.robot import ARM_HOME
    t, _ = tools_with(ForbiddenExecutor(), ExecutionControls(motion="blocked"))
    q = np.array(ARM_HOME, float) + np.radians(10.0)
    t.bridge = SimpleNamespace(finger=lambda: 0.0, q=lambda: q)
    r = t.return_home()
    assert r["ok"] is False and t.motion_count == 0


# ------------------------------------------------------------------------------------ supervisor settings
def test_disabled_capability_is_refused_as_unsupported_and_never_runs(tmp_path):
    seen = []
    t = GoalTools([plan(("pick_place", PP))])
    c = ExecutionControls(disabled={"pick_place"}, disabled_reason="in this test", on_event=lambda e, **k: seen.append((e, k)))
    r, ev = run_supervisor(tmp_path, t, c)
    assert r["status"] == "UNSUPPORTED" and "pick_place is disabled in this test" in r["message"]
    assert t.executed == [] and seen == [("capability_refused", {"capability": "pick_place"})]
    assert not any(e["kind"] == "skill" for e in ev)  # the pick-and-place workflow never started


def test_disabled_capabilities_show_in_this_supervisors_menu_only(tmp_path):
    from agent import capabilities as caps
    menus = []
    t = GoalTools([plan(("observe", {}), ("describe_scene", {}))])
    plan_goal = t.roles.plan_goal
    t.roles.plan_goal = lambda text, menu, context="": (menus.append(menu), plan_goal(text, menu, context))[1]
    r, _ = run_supervisor(tmp_path, t, ExecutionControls(disabled={"explore"}))
    explore_line = next(line.strip() for line in menus[0].splitlines() if line.strip().startswith("- explore:"))
    assert r["status"] == "COMPLETED" and explore_line.endswith("(not available yet)")
    assert all(s.implemented for s in caps.SPECS.values())  # the registry itself is never changed
    assert caps.planner_menu() == caps.planner_menu(caps.available(()))


def test_disabled_explore_is_not_added_by_the_subtask_policy(tmp_path, monkeypatch):
    from agent import capabilities as caps
    monkeypatch.setattr(caps, "run_locate", lambda ctx, args: caps.StepResult("insufficient_info", "not visible"))
    t = GoalTools([plan(("observe", {}), ("locate", {"object": "zz"})), plan(("observe", {}), ("locate", {"object": "zz"}))])
    r, _ = run_supervisor(tmp_path, t, ExecutionControls(disabled={"explore"}))
    assert "explore" not in [s["capability"] for s in r["plan"]]


@pytest.mark.parametrize("limit,planner_calls", [(0, 2), (1, 2)])  # Gemma is asked either way; the budget limits changes
def test_replanning_limit_comes_from_the_controls(tmp_path, monkeypatch, limit, planner_calls):
    from agent import capabilities as caps
    monkeypatch.setattr(caps, "run_answer", lambda ctx, args: caps.StepResult("failed", "could not answer"))
    steps = (("observe", {}), ("answer", {"question": "how many?"}))
    t = GoalTools([plan(*steps), plan(*steps)])
    r, _ = run_supervisor(tmp_path, t, ExecutionControls(max_replans=limit))
    assert r["status"] == "FAILED" and len(t.plan_calls) == planner_calls and r["replans"] == limit


def test_session_passes_the_same_controls_to_tools_and_supervisor(tmp_path, monkeypatch):
    import agent.supervisor
    import agent.tools
    from agent.session import Session
    from agent.vlm_roles import RecordedVLM
    got = {}

    class FakeTools:
        def __init__(self, *a, controls=None, **k):
            got["tools"] = controls

        def revalidate_held(self):
            return {}

    class FakeSupervisor:
        def __init__(self, *a, controls=None, **k):
            got["supervisor"] = controls

        def run(self, instruction, thread_id):
            return {"status": "COMPLETED", "message": "ok"}

    monkeypatch.setattr(agent.tools, "Tools", FakeTools)
    monkeypatch.setattr(agent.supervisor, "GoalSupervisor", FakeSupervisor)
    c = ExecutionControls(motion="blocked")
    r = Session(None, None, RecordedVLM([]), None, None, controls=c).run("look", tmp_path / "01")
    assert r["status"] == "COMPLETED" and got == {"tools": c, "supervisor": c}


# ------------------------------------------------------------------------------------- harness directives
def test_directive_lines_go_to_the_handler_not_to_the_reader():
    from agent.ask import LineReader
    r = LineReader(io.StringIO("@motion on\nhello\n@motion off\n"))
    got = []
    r.on_directive = got.append
    assert r.get(2) == "hello" and r.get(2) is None and got == ["@motion on", "@motion off"]


def test_without_a_handler_directive_lines_are_ordinary_input():
    from agent.ask import LineReader
    assert LineReader(io.StringIO("@motion on\n")).get(2) == "@motion on"


def test_experiment1_harness_sets_the_same_guards_as_before(tmp_path, monkeypatch):
    """evaluation/lab_exp1_agent.py: blocked motion until "@motion on", five capabilities disabled, no replanning,
    and the harness_gates.jsonl records of the original (monkeypatching) harness."""
    import agent.run
    import evaluation.lab_exp1_agent as harness
    captured = {}
    monkeypatch.setattr(agent.run, "main", lambda controls=None, on_directive=None: captured.update(c=controls, d=on_directive) or 0)
    monkeypatch.setattr(sys, "argv", ["lab_exp1_agent", "--out", str(tmp_path)])
    assert harness.main() == 0
    c, directive = captured["c"], captured["d"]
    assert c.motion == "blocked" and c.max_replans == 0
    assert c.disabled == {"pick", "place", "pick_place", "explore", "decompose"}
    assert c.blocked_reason == "motion gate closed (Experiment 1: no motion in this phase)"
    (tmp_path / "01").mkdir()  # one instruction already ran: the gate opens for instruction 2
    directive("@motion on")
    assert c.motion == "real"
    c.notify("motion_allowed", label="inspect view", steps=["move"])
    directive("@motion off")
    assert c.motion == "blocked"
    recs = [json.loads(line) for line in (tmp_path / "harness_gates.jsonl").read_text().splitlines()]
    assert recs[0]["event"] == "guards" and recs[0]["max_replans"] == 0 and recs[0]["motion"] is False
    assert recs[0]["unavailable"] == ["pick", "place", "decompose", "pick_place", "explore"]  # registry order
    assert [r["event"] for r in recs[1:]] == ["gate", "motion_allowed", "gate"]
    assert recs[1]["motion"] is True and recs[2]["instruction"] == 2 and recs[3]["motion"] is False


# ---------------------------------------------------------------------------------- agent.run entry point
def test_agent_run_blocks_motion_unless_real_is_requested():
    from agent.run import controls_for, parser
    assert controls_for(parser().parse_args([])).motion == "blocked"
    assert controls_for(parser().parse_args(["--motion", "dry_run"])).motion == "dry_run"
    assert controls_for(parser().parse_args(["--motion", "real"])).motion == "real"
    with pytest.raises(SystemExit):
        parser().parse_args(["--motion", "fast"])


def test_harness_controls_take_precedence_over_the_command_line():
    from agent.run import controls_for, parser
    harness = ExecutionControls(motion="blocked", max_replans=0)
    assert controls_for(parser().parse_args(["--motion", "real"]), harness) is harness


def test_simulator_command_for_default_scene_and_laboratory():
    from agent.run import sim_command
    assert sim_command(True)[-2:] == ["-m", "sim.run_live"]
    cmd = sim_command(False, "laboratory", "medium", 42)
    assert cmd[cmd.index("-m") + 1:] == ["sim.run_scenario", "--scenario", "laboratory", "--difficulty", "medium",
                                         "--seed", "42", "--headless"]
