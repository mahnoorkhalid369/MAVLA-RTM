"""Perception on SAVED simulator frames (out/lab_seg_frames, captured by evaluation/lab_capture_frames.py): RGB,
depth, intrinsics, camera pose, joint state, the simulator's instance masks and the settled object poses.
Ground truth is used here only to score the agent's code (evaluation), never given to it.

- localization: perception.geometry.measure on each object's true mask -> centre and top within the agent's own
  placement tolerances (config.agent.Tolerances: xy 10 mm, z 5 mm)
- integration: agent.tools.Tools.perceive + update_world on a replayed frame (bridge and detector replaced by the
  saved frame and its true masks; everything else is the production code)
- YOLO: the production detector on a saved frame (skipped without the weights / ultralytics)
Skipped when the frames are not on this machine (out/ is not in git)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "out" / "lab_seg_frames"
SCENES = ["easy_s42", "medium_s42", "hard_s42"]  # the Experiment 1 seed at every difficulty
pytestmark = pytest.mark.skipif(not (FRAMES / "easy_s42" / "table" / "rgb.png").exists(),
                                reason="saved lab frames not on this machine (out/lab_seg_frames)")


def load(scene: str, view: str = "table"):
    v = FRAMES / scene / view
    cam = json.loads((v / "camera.json").read_text())
    rgb = cv2.cvtColor(cv2.imread(str(v / "rgb.png")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(v / "depth_mm.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    ins = cv2.imread(str(v / "instance.png"), cv2.IMREAD_UNCHANGED)
    labels = json.loads((v / "instance.json").read_text())
    meta = json.loads((FRAMES / scene / "metadata.json").read_text())
    objects = {}  # object id -> true mask (visible pixels)
    for i, path in labels.items():
        if "/World/Objects/" in path:
            name = path.split("/World/Objects/")[1].split("/")[0]
            m = ins == int(i)
            if name in meta["settled"] and m.sum() >= 200:
                objects[name] = objects.get(name, np.zeros(ins.shape, bool)) | m
    return SimpleNamespace(rgb=rgb, depth=depth, K=tuple(cam["K"]), T=np.array(cam["T_world_cam"]),
                           q=np.array(cam["q"]), objects=objects, meta=meta)


def truth(meta, name):
    """Settled base position and top height of a scenario object (evaluation only)."""
    dims = {o["id"]: o for o in meta["objects"]}[name]["dimensions"]
    p = meta["settled"][name]["position"]
    return np.array(p[:2]), p[2] + dims["height"]


@pytest.mark.parametrize("scene", SCENES)
def test_measured_positions_match_the_settled_poses(scene):
    from config.agent import Tolerances
    from control.collision import CollisionChecker
    from control.kinematics import Kinematics
    from control.robot import gripper_gap
    from perception.geometry import RobotVolume, measure
    f = load(scene)
    tol = Tolerances()
    robot = RobotVolume(CollisionChecker(), f.q, gripper_gap(0.0), None, Kinematics(tip="tool0").fk(f.q))
    assert len(f.objects) >= 8
    for name, mask in f.objects.items():
        g, why = measure(mask, f.depth, f.K, f.T, robot)
        assert g is not None, f"{name}: {why}"
        xy, top = truth(f.meta, name)
        assert np.hypot(*(np.array(g.centre[:2]) - xy)) <= tol.xy_m, name
        assert abs(g.top_z - top) <= tol.z_m, name
        assert g.tilt_deg is not None, f"{name}: top face not measured"


class SavedFrameBridge:
    """agent.ros_bridge.RosBridge replaced by one saved table frame (the robot is not moving)."""

    def __init__(self, f):
        self.f = f

    def snapshot(self, camera="table", timeout=10.0):
        from agent.ros_bridge import Frame
        return Frame("table", self.f.rgb, self.f.depth, self.f.K, 0.0, self.f.q, 0.0)

    def q(self):
        return self.f.q

    def finger(self):
        return 0.0


class TrueMaskDetector:
    """The YOLO detector replaced by the simulator's object masks: deterministic input for the rest of the pipeline."""

    def __init__(self, masks):
        self.masks = masks

    def detect(self, rgb):
        from perception.detector import Detection
        return [Detection(m, 0.9) for m in self.masks]


def test_tools_perceive_and_world_model_on_a_replayed_frame(tmp_path):
    pytest.importorskip("pydantic")
    from agent.tools import Tools
    f = load("easy_s42")
    (tmp_path / "images").mkdir()
    events = []
    t = Tools(SavedFrameBridge(f), TrueMaskDetector(list(f.objects.values())), None, None,
              lambda kind, **k: events.append({"kind": kind, **k}), tmp_path)
    obs = t.perceive("table")
    assert len(obs.marks) == len(f.objects) and not obs.excluded
    assert (tmp_path / "images" / f"{obs.frame_id}_som.jpg").exists()
    t.update_world(obs)
    assert len(t.world.objects) == len(f.objects) and set(obs.object_ids) == set(obs.marks)
    measured = [np.array(o.geometry["centre"][:2]) for o in t.world.objects.values()]
    for name in f.objects:  # every true object has exactly one world object at its position
        xy, _ = truth(f.meta, name)
        assert sum(np.hypot(*(m - xy)) <= 0.010 for m in measured) == 1, name
    obs2 = t.perceive("table")  # a second look at the unchanged scene keeps the same ids
    t.update_world(obs2)
    assert len(t.world.objects) == len(f.objects) and sorted(obs2.object_ids.values()) == sorted(obs.object_ids.values())
    assert t.is_fresh(obs2) and t.motion_count == 0


def test_yolo_detector_on_a_saved_frame():
    pytest.importorskip("ultralytics")
    torch = pytest.importorskip("torch")
    from config.models import DETECTOR_CONF, DETECTOR_IMGSZ, YOLO_WEIGHTS
    from perception.detector import YoloDetector
    if not YOLO_WEIGHTS.exists():
        pytest.skip("YOLO weights not on this machine")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    f = load("easy_s42")
    dets = YoloDetector(str(YOLO_WEIGHTS), DETECTOR_IMGSZ, DETECTOR_CONF, device=device).detect(f.rgb)
    assert dets, "no detections on the laboratory table view"
    assert all(d.mask.shape == f.rgb.shape[:2] and d.mask.dtype == bool and DETECTOR_CONF <= d.score <= 1.0 for d in dets)
