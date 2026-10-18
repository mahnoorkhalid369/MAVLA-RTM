"""Demo viewer and event lines (agent/events.py, agent/viewer.py): runs with the system Python (no ROS,
no window). Uses a real event log from the rehearsal when it exists."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from agent.events import format_event
from agent.viewer import IMG_H, IMG_W, TARGET_C, RunState, Viewer

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "out/demo/2026-10-14/rehearsal/cycle4/i1_run1/events.jsonl"


def ev(kind, **f):
    return {"wall": "13:05:01", "kind": kind, **f}


def test_lines_reflect_only_logged_outcomes():
    ok = format_event(ev("verification", check="placement", xy_error_mm=2.0, z_error_mm=-0.8, tilt_deg=0.0, **{"pass": True}))
    bad = format_event(ev("verification", check="placement", xy_error_mm=14.0, z_error_mm=0.1, tilt_deg=0.0, **{"pass": False}))
    assert ok.level == "success" and "confirmed" in ok.text
    assert bad.level == "error" and "OUT OF TOLERANCE" in bad.text and "confirmed" not in bad.text
    home = format_event(ev("verification", check="home", max_error_deg=0.04, tolerance_deg=1.5, tcp_error_mm=0.2, **{"pass": True}))
    assert home.component == "VERIFY" and "reached" in home.text
    assert format_event(ev("vlm", role="describe", valid=True)) is None  # no noise for each valid call
    q = format_event(ev("question", question="Which one?", candidates=[5, 7]))
    a = format_event(ev("ask_user", answer="choice", mark=5, raw="5", source="terminal"))
    assert q.level == "question" and "waiting for the user" in q.text and a.component == "USER" and "5" in a.text
    s = format_event(ev("ask_user", answer="cancel", raw="cancel", source="scripted"))
    assert s.component == "SCRIPT"  # a scripted answer is never shown as the user's


@pytest.mark.skipif(not REAL.exists(), reason="rehearsal log not present")
def test_real_run_log_formats_without_errors_and_one_final_verdict():
    events = [json.loads(line) for line in REAL.read_text().splitlines()]
    lines = [x for x in map(format_event, events) if x is not None]
    assert len(lines) > 15
    finals = [x for x in lines if x.text.startswith(("COMPLETED", "FAILED", "CANCELLED"))]
    assert len(finals) == 1 and finals[0].level == "success"


def test_viewer_draws_masks_highlights_target_and_shows_logs(tmp_path):
    lab = np.zeros((720, 1280), np.uint16)
    lab[300:400, 500:600], lab[300:400, 800:900] = 1, 2
    cv2.imwrite(str(tmp_path / "labels.png"), lab)
    (tmp_path / "perception.json").write_text(json.dumps({"frame_id": "table_001", "camera": "table", "time": 1.0, "marks": {
        "1": {"object_id": 11, "description": {"name": "cube", "colour": "red"}, "centre": [0.0, 0.58, 0.03], "score": 0.9},
        "2": {"object_id": 12, "description": {"name": "cube", "colour": "green"}, "centre": [0.15, 0.5, 0.025], "score": 0.9}}}))
    run = tmp_path / "run"
    run.mkdir()
    events = [ev("instruction", text="put the red cube on the green cube"), ev("state", state="LOCALIZING"),
              ev("localized", frame_id="table_001", object_mark=1, object_id=11, reference_mark=2, reference_id=12, object={"centre": [0, .58, .03]},
                 reference={"centre": [.15, .5, .025]}), ev("state", state="APPROACHING"),
              ev("tool", tool="execute_start", label="approach", steps=["move above object"])]
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (tmp_path / "current.json").write_text(json.dumps({"events": str(run / "events.jsonl"), "instruction": "put the red cube on the green cube"}))
    v = Viewer(tmp_path)
    v.frames["table"] = (np.full((720, 1280, 3), 90, np.uint8), 0.0)
    v.poll_files()
    img = v.render()
    assert v.rs.state == "APPROACHING" and v.rs.moving and v.rs.target == (1, 11, "table_001") and v.rs.dest == (2, 12, "table_001")
    assert v.highlight() == {1: (TARGET_C, "TARGET"), 2: (v.highlight()[2][0], "DESTINATION")}
    left = img[:IMG_H, :IMG_W]
    assert (np.abs(left.astype(int) - np.array(TARGET_C)).sum(axis=2) < 10).sum() > 100  # thick target outline drawn
    assert len(v.rs.lines) >= 4


def test_viewer_without_camera_or_files_does_not_fail(tmp_path):
    v = Viewer(tmp_path)
    v.poll_files()
    assert v.render().shape == (900, 1600, 3)


def test_run_state_ignores_nothing_it_did_not_see():
    rs = RunState()
    rs.apply(ev("state", state="VERIFYING_PLACEMENT"), None)
    assert rs.home == "PENDING" and rs.status == "RUNNING"
    rs.apply(ev("verification", check="home", **{"pass": False}), None)
    assert rs.home == "NOT REACHED"


def test_selection_is_never_shown_on_a_different_object_in_a_newer_frame():
    """Regression (live demo 14:50): in the re-observation mark 4 was another object; the destination
    (world id 4, covered by the placed cube) must not be shown as that object."""
    v = Viewer(Path("/nonexistent"))
    v.rs.dest = (4, 4, "table_001")
    v.rs.target = (3, 3, "table_001")
    v.perc = {"frame_id": "table_002", "marks": {"4": {"object_id": 5, "description": {"name": "cylinder"}},
                                                 "14": {"object_id": 3, "description": {"name": "block"}}}}
    assert v.mark_of(v.rs.dest) is None and v.mark_of(v.rs.target) == 14
    assert "not identified" in v._obj_line(v.rs.dest, "DESTINATION")[1][0]
    v.perc["frame_id"] = "table_001"
    v.perc["marks"]["4"]["object_id"] = None  # same frame, world ids not assigned yet: the chosen mark is right
    assert v.mark_of(v.rs.dest) == 4


def test_goal_plan_and_steps_are_tracked_and_formatted():
    rs = RunState()
    for e in [ev("state", state="INTERPRETING", level="goal"),
              ev("goal_plan", goal="report", steps=[{"index": 1, "capability": "observe", "args": {}},
                                                   {"index": 2, "capability": "describe_scene", "args": {}}]),
              ev("step", event="start", index=1, capability="observe", safety="read_only"),
              ev("step", event="end", index=1, capability="observe", outcome="completed", seconds=3.0),
              ev("state", state="COMPLETED"),  # a skill-level terminal state does not end the goal
              ev("state", state="UNSUPPORTED", level="goal")]:
        rs.apply(e, None)
    assert [s["status"] for s in rs.steps] == ["completed", "pending"] and rs.status == "UNSUPPORTED"
    lines = [str(format_event(e)) for e in [ev("goal_plan", goal="g", steps=[], reason="cannot sing")]]
    assert "cannot sing" in lines[0] and "[PLANNER]" in lines[0]


def stale_run(live_dir: Path) -> None:
    """What an earlier agent process leaves behind: its task, its events and its detection overlay."""
    lab = np.zeros((720, 1280), np.uint16)
    lab[300:400, 500:600] = 1
    cv2.imwrite(str(live_dir / "labels.png"), lab)
    (live_dir / "perception.json").write_text(json.dumps({"frame_id": "table_001", "camera": "table", "time": 1.0,
                                                          "marks": {"1": {"object_id": 4, "score": 0.9}}}))
    run = live_dir / "old_run"
    run.mkdir()
    (run / "events.jsonl").write_text(json.dumps(ev("instruction", text="sort all the boxes")) + "\n"
                                      + json.dumps(ev("state", state="FAILED", level="goal")) + "\n")
    (live_dir / "current.json").write_text(json.dumps({"events": str(run / "events.jsonl"),
                                                       "instruction": "sort all the boxes, group by colors", "started": 1.0}))


def test_a_fresh_agent_launch_clears_the_previous_task_from_an_open_viewer(tmp_path):
    """Regression (2026-10-18): a new agent process showed the previous process's task and detection overlay."""
    from agent.live import LiveChannel
    stale_run(tmp_path)
    v = Viewer(tmp_path)
    v.poll_files()
    assert v.rs.instruction.startswith("sort all the boxes") and v.perc is not None and v.labels is not None
    LiveChannel(tmp_path)  # agent.run starts: a new session
    v.poll_files()
    assert v.rs.instruction == "" and v.rs.status == "WAITING" and v.rs.lines == deque_of(v.rs.lines, [])
    assert v.perc is None and v.labels is None and v.contours == {} and v.run_file is None
    assert v.render().shape == (900, 1600, 3)


def test_a_viewer_opened_after_a_fresh_launch_waits_for_an_instruction(tmp_path):
    from agent.live import LiveChannel
    stale_run(tmp_path)
    LiveChannel(tmp_path)
    assert not (tmp_path / "perception.json").exists() and not (tmp_path / "labels.png").exists()
    v = Viewer(tmp_path)
    v.poll_files()
    assert v.rs.instruction == "" and v.perc is None and v.labels is None
    assert json.loads((tmp_path / "current.json").read_text())["events"] is None


def test_the_next_instruction_after_a_fresh_launch_is_shown(tmp_path):
    from agent.live import LiveChannel
    stale_run(tmp_path)
    live = LiveChannel(tmp_path)
    v = Viewer(tmp_path)
    v.poll_files()
    run = tmp_path / "new_run"
    run.mkdir()
    (run / "events.jsonl").write_text(json.dumps(ev("instruction", text="look")) + "\n")
    live.start_run(run, "Take a look at your surroundings.")
    v.poll_files()
    assert v.rs.instruction == "Take a look at your surroundings." and v.run_file == str(run / "events.jsonl")


def deque_of(d, items):
    """The run-state log is a deque; compare its contents."""
    return type(d)(items, maxlen=d.maxlen)
