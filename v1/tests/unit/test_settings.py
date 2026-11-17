"""agent/settings.py (the one place to configure a robot session) and what launch/agent.sh builds from it:
validation, the startup summary, agent.run defaults, the Isaac Sim-only target and its guard, and the handoff trace."""

import json
from pathlib import Path

import pytest

from agent import settings
from agent.run import controls_for, parser, sim_command, simulator_on_graph

ROOT = Path(__file__).resolve().parents[2]


def test_the_shipped_settings_are_valid_and_summarised():
    if not (Path(settings.GEMMA_MODEL) / "config.json").exists() or not Path(settings.YOLO_WEIGHTS).exists():
        pytest.skip("models not on this machine")
    assert settings.validate() == []
    text = settings.summary()
    for word in ("scene", "target", "motion", "models", "windows", "tracing", "isaac_sim"):
        assert word in text


@pytest.mark.parametrize("override,needle", [
    ({"TARGET": "physical"}, "Physical-robot execution is not part of this launcher"),
    ({"MOTION": "fast"}, "MOTION"),
    ({"DIFFICULTY": "extreme"}, "DIFFICULTY"),
    ({"SEED": -1}, "SEED"),
    ({"SEED": True}, "SEED"),
    ({"SCENARIO": "no_such_scene"}, "SCENARIO"),
    ({"ROS_DOMAIN_ID": 999}, "ROS_DOMAIN_ID"),
    ({"YOLO_WEIGHTS": "/nonexistent/best.pt"}, "YOLO_WEIGHTS"),
    ({"GEMMA_MODEL": "/nonexistent"}, "GEMMA_MODEL"),
    ({"YOLO_IMAGE_SIZE": 1000}, "YOLO_IMAGE_SIZE"),
    ({"YOLO_CONFIDENCE": 1.5}, "YOLO_CONFIDENCE"),
    ({"ASK": "gemma"}, "ASK"),
    ({"TRACE": "yes"}, "TRACE"),
    ({"MAX_GEMMA_DECISIONS": -1}, "MAX_GEMMA_DECISIONS"),
    ({"MAX_GEMMA_DECISIONS": True}, "MAX_GEMMA_DECISIONS"),
])
def test_invalid_settings_are_reported(override, needle):
    assert any(needle in p for p in settings.validate(override))


@pytest.mark.parametrize("scene", ["laboratory", "default", "ap_occlusion"])
def test_known_scenes_are_accepted(scene):
    assert not any("SCENARIO" in p for p in settings.validate({"SCENARIO": scene}))


def test_launcher_defaults_come_from_the_settings():
    ap = parser()
    ap.set_defaults(**settings.run_defaults())
    a = ap.parse_args(["--settings"])
    assert a.start_sim is settings.START_SIMULATOR and a.motion == settings.MOTION and a.gui is settings.ISAAC_WINDOW
    assert a.difficulty == settings.DIFFICULTY and a.seed == settings.SEED and a.gemma == settings.GEMMA_MODEL
    assert a.weights == settings.YOLO_WEIGHTS and a.trace is settings.TRACE and a.target == "isaac_sim"
    assert a.out.startswith(str(ROOT / settings.OUTPUT_DIR))
    b = ap.parse_args(["--settings", "--motion", "blocked", "--headless"])  # one-run overrides win
    assert b.motion == "blocked" and b.gui is False
    assert controls_for(a).max_replans == settings.MAX_GEMMA_DECISIONS  # the Gemma decision budget reaches the loop
    assert controls_for(ap.parse_args(["--settings", "--max-decisions", "0"])).max_replans == 0


def test_harnesses_without_settings_keep_the_old_defaults():
    """Experiment harnesses start their own simulator and call agent.run without --settings: no second Isaac Sim,
    motion blocked unless they ask for it."""
    a = parser().parse_args([])
    assert a.start_sim is False and a.motion == "blocked" and a.trace is False and a.target == "isaac_sim"


def test_only_isaac_sim_is_a_target():
    with pytest.raises(SystemExit):
        parser().parse_args(["--target", "physical"])


def test_every_named_scenario_starts_through_run_scenario_and_none_starts_the_frozen_simulator():
    cmd = sim_command(True, "default")
    assert cmd[cmd.index("-m") + 1:cmd.index("-m") + 4] == ["sim.run_scenario", "--scenario", "default"]
    assert sim_command(True)[-2:] == ["-m", "sim.run_live"]  # harnesses without a scenario: unchanged


def test_motion_needs_the_simulator_on_the_ros_graph():
    assert simulator_on_graph(["vt6l_sim"])
    assert not simulator_on_graph([])
    assert not simulator_on_graph(["vt6l_agent_executor", "some_robot_driver"])


# ------------------------------------------------------------------------------------------ handoff trace
def ev(kind, **f):
    return {"wall": "10:00:00", "kind": kind, **f}


def test_trace_shows_delegation_input_worker_output_and_failures():
    from agent.events import format_trace
    d = format_trace(ev("delegate", worker="perception", task_id="t3", capability="locate", safety="read_only",
                        args={"target": "blue-capped tube"}, objective="locate blue-capped tube"))
    assert d.component == "SUPERVISOR" and "-> PERCEPTION [t3] locate" in d.text and "blue-capped tube" in d.text
    ok = format_trace(ev("worker_result", worker="perception", task_id="t3", status="completed", outcome="completed",
                         summary="object 4 at x +120 mm", confidence="measured", metrics={"seconds": 2.1, "gemma_calls": 2}))
    assert ok.component == "PERCEPTION" and "<- [t3] completed" in ok.text and ok.level == "success"
    bad = format_trace(ev("worker_result", worker="manipulation", task_id="t4", status="failed", outcome="infeasible",
                          summary="out of reach", failure_reason="out of reach", metrics={}))
    assert "failure: out of reach" in bad.text and bad.level == "error"
    refused = format_trace(ev("authorization", capability="pick", allowed=False, reason="the request is a question"))
    assert refused.component == "AUTHORITY" and "refused pick" in refused.text
    assert format_trace(ev("authorization", capability="observe", allowed=True)) is None
    check = format_trace(ev("goal_check", results=[{"condition": {"pred": "on", "args": ["#1", "#2"]}, "satisfied": False,
                                                    "confidence": "measured"}]))
    assert "on(#1, #2)=NO" in check.text and check.level == "error"
    auth = format_trace(ev("authority", request_kind="action", grants=["none", "sensing", "object_change"], evidence="Put", source="gemma"))
    assert "object_change" in auth.text


def test_event_log_prints_handoffs_only_when_tracing(tmp_path, capsys):
    from agent.eventlog import EventLog
    for trace, expect in ((False, False), (True, True)):
        log = EventLog(tmp_path / f"e{trace}.jsonl", echo=True, trace=trace)
        log("delegate", worker="perception", task_id="t1", capability="observe", safety="read_only", args={}, objective="observe")
        log.close()
        out = capsys.readouterr().out
        assert ("-> PERCEPTION [t1] observe" in out) is expect
        assert json.loads((tmp_path / f"e{trace}.jsonl").read_text())["kind"] == "delegate"  # always saved in full
