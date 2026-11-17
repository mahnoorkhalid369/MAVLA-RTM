"""Production executor with a synthetic clock/joints; no ROS node or motion commands.

Source ROS before running this file. No simulator is needed.
"""
from types import SimpleNamespace
import numpy as np
import pytest

pytest.importorskip("rclpy")
from execution import ros_executor
from control.motion import Trajectory
from control.skills import MoveStep, GripperStep


def executor(monkeypatch, rate=0.2, rejected=False):
    clock = SimpleNamespace(wall=0.)
    monkeypatch.setattr(ros_executor, "time", SimpleNamespace(monotonic=lambda:clock.wall))
    ex = ros_executor.RosExecutor.__new__(ros_executor.RosExecutor)
    ex.sim_time, ex.safety_events = 10., []
    ex.reach_tol, ex.settle_tol, ex.timeout_factor = np.radians(1.5), np.radians(.1), 3.
    ex.goal, ex.sent_at, ex.finger = np.zeros(6), 10., 0.
    ex.q = lambda: ex.goal.copy() if ex.sim_time-ex.sent_at >= 1. else np.zeros(6)
    def spin(dt):
        clock.wall += dt
        ex.sim_time += dt*rate
    def send(step):
        ex.goal, ex.sent_at = step.trajectory.positions[-1], ex.sim_time
        if rejected:
            ex.safety_events.append("REJECTED synthetic path")
    ex.spin, ex.send_trajectory = spin, send
    ex.set_gripper = lambda p: setattr(ex, "finger", p)
    ex.gripper = lambda:ex.finger
    return ex


def test_planned_simulated_wall_and_gripper_time_are_recorded(monkeypatch):
    ex = executor(monkeypatch)
    move = MoveStep(Trajectory(np.array([0., 1.]), np.array([np.zeros(6), np.ones(6)*.1])), "test move")
    r = ex.execute([move, GripperStep(.1, "test fingers", .2)])
    assert r.ok and len(r.steps) == 2
    assert r.steps[0]["planned_s"] == 1.
    assert r.steps[0]["sim_elapsed_s"] >= 1.
    assert r.steps[0]["wall_s"] >= 5.
    assert r.steps[0]["real_time_factor"] == pytest.approx(.2)
    assert r.steps[1]["kind"] == "gripper" and r.steps[1]["sim_elapsed_s"] >= .2


@pytest.mark.parametrize("mode", ["stalled", "rejected"])
def test_failed_execution_still_records_timing_and_returns(monkeypatch, mode):
    ex = executor(monkeypatch, rate=0. if mode == "stalled" else .2, rejected=mode == "rejected")
    move = MoveStep(Trajectory(np.array([0., 1.]), np.array([np.zeros(6), np.ones(6)*.1])), "test move")
    r = ex.execute([move])
    assert not r.ok and len(r.steps) == 1 and not r.steps[0]["ok"]
    assert r.steps[0]["wall_s"] > 0


def test_agent_does_not_subscribe_to_ground_truth_without_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(ros_executor.rclpy, "ok", lambda:True)
    topics = []
    node = SimpleNamespace(create_subscription=lambda msg, topic, callback, qos: topics.append(topic) or object(),
                           create_publisher=lambda *a:object())
    ex = ros_executor.RosExecutor(node=node)
    assert "sim/ground_truth/objects" not in topics
    ex._subscribe_truth()
    ex._subscribe_truth()
    assert topics.count("sim/ground_truth/objects") == 1
