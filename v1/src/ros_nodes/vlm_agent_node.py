#!/usr/bin/env python3
"""ROS 2 node: vision-language pick-and-place agent (Gemma 4 E2B-it).

A task (typed text on /task, or speech from the microphone with --voice) plus the current wrist
camera frame go to the model, which returns a plan of pick / place / home steps with 2-D boxes.
Each box is located in 3-D from the depth image, every motion is planned, then executed.

Runs with the VLM environment and ROS 2 Jazzy:
    source /opt/ros/jazzy/setup.bash
    .venv-vlm/bin/python ros_nodes/vlm_agent_node.py            # typed tasks on /task
    .venv-vlm/bin/python ros_nodes/vlm_agent_node.py --voice    # also spoken tasks

Topics
  in   /task                          std_msgs/String       task in plain language ("stop" cancels)
  in   /camera/color/image_raw        sensor_msgs/Image     rgb8
  in   /camera/depth/image_rect_raw   sensor_msgs/Image     32FC1 metres
  in   /camera/color/camera_info      sensor_msgs/CameraInfo
  in   /joint_states, /clock
  in   /microphone/audio, /microphone/speech_active      (with --voice)
  out  /joint_trajectory, /joint_command                  arm motions, gripper
  out  /task/feedback                 std_msgs/String       progress and answers
  out  /task/annotated                sensor_msgs/Image     camera frame with the model's boxes

Module layout: vlm/ (model, prompt, plan parsing), perception/grounding.py (box -> 3-D),
control/skills.py + control/motion.py (planning). This file is only the ROS glue.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the repository (this file is in src/ros_nodes/)
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from builtin_interfaces.msg import Duration  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from PIL import ImageDraw  # noqa: E402
from rclpy.executors import ExternalShutdownException  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, JointState  # noqa: E402
from std_msgs.msg import Bool, Int16MultiArray, String  # noqa: E402
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # noqa: E402

from control.instructions import InstructionParser, plan_command  # noqa: E402
from control.kinematics import Kinematics  # noqa: E402
from control.motion import PlanningError, joint_move  # noqa: E402
from control.robot import (  # noqa: E402
    ARM_JOINTS,
    ARM_SPEED,
    GRIPPER_JOINT,
    GRIPPER_SPEED,
    OBSERVE_POSE,
    VIEW_POSES,
    table_camera_pose,
)
from control.skills import GripperStep, MoveStep, SkillPlanner  # noqa: E402
from perception.grounding import GroundingError, locate  # noqa: E402
from perception.world_memory import SHAPES, WorldMemory, matches  # noqa: E402
from vlm.plan import PlanError  # noqa: E402

TASK_LOG_DIR = ROOT / "out" / "tasks"
MEMORY_FILE = ROOT / "out" / "memory" / "world.json"  # long-term object memory (survives restarts)
SCAN_HEADINGS_DEG = (-150, -100, -50, 0, 50, 100, 150)  # J1 sweep for "scan" (J1 limit +-170)
MAX_OBJECT_TOP = 0.14  # m: anything taller is a prop (the microphone), not a test object
SEARCH_STEP_DEG = 45.0  # J1 step between search views (camera sees ~60 deg of table)
EXPLORE_WORDS = ("explore", "explore the surroundings", "explore surroundings", "explore around")
OVERVIEW_WORDS = ("overview", "survey", "look around", "what is around you", "where is everything")
SEARCH_LIMIT_DEG = 165.0  # stay inside J1 +-170
MEMORY_WORDS = ("memory", "what do you remember", "remember", "list objects")
VOICE_DIR = ROOT / "out" / "voice"  # every utterance the agent heard, as .wav (play: aplay <file>)
MAX_UTTERANCE_S = 12.0
MIN_UTTERANCE_S = 0.6
# Log tags and their indent level: who speaks at level 0, what the agent is doing under it
LOG_LEVELS = {"USER": 0, "AGENT": 0, "PLANNING": 1, "PERCEPTION": 1, "MOTION": 1, "GRASP": 1, "MEMORY": 1, "WARN": 1}
REFINE_RADIUS = 0.07  # m: the wrist camera looks for the object within this distance of the coarse estimate
REFINE_MAX_SHIFT = 0.05  # m: a wrist estimate further than this from the coarse one is something else
ALIGN_TOLERANCE = (0.003, 3.0)  # (m, deg): re-align above the object if the wrist camera disagrees by more


@dataclass
class Request:
    text: str | None = None
    audio: np.ndarray | None = None  # float32, 16 kHz
    explored: bool = False  # already explored for a missing object: do not ask again


@dataclass
class Snapshot:
    rgb: np.ndarray
    depth: np.ndarray
    K: tuple[float, float, float, float]
    q: np.ndarray  # arm joints at capture time
    T: np.ndarray  # 4x4 pose of the camera's optical frame in the world at capture time
    camera: str  # "wrist" (eye-in-hand) | "table" (eye-to-hand)


class VLMAgent(Node):
    def __init__(self, planner, voice: bool, memory_file: Path = MEMORY_FILE, stdin: bool = False) -> None:
        super().__init__("vlm_agent")
        self.planner = planner
        self.kin = Kinematics()
        self.camera_kin = Kinematics(tip="camera_color_optical_frame")
        self.skills = SkillPlanner(self.kin)
        self.memory = WorldMemory(memory_file)
        self.direct = InstructionParser()  # direct commands: move left 100, look up, turn around...
        self.pending: dict | None = None  # a task waiting for the user to say where to look
        self.lock = threading.Lock()
        self.images = {"wrist": {}, "table": {}}  # camera -> {"rgb", "depth", "K"}
        self.joints: dict[str, float] = {}
        self.sim_time = 0.0
        self.requests: queue.Queue[Request] = queue.Queue(maxsize=1)
        self.busy = threading.Event()
        self.cancel = threading.Event()
        self.audio_chunks: list[np.ndarray] = []
        self.speaking = False
        self.too_long = False  # current sound has gone on too long to be an instruction
        self.last_noise_warning = -1e9

        self.create_subscription(String, "task", self.on_task, 10)
        for camera, prefix in (("wrist", "camera"), ("table", "table_camera")):
            self.create_subscription(Image, f"{prefix}/color/image_raw",
                                     lambda m, c=camera: self.on_rgb(c, m), qos_profile_sensor_data)
            self.create_subscription(Image, f"{prefix}/depth/image_rect_raw",
                                     lambda m, c=camera: self.on_depth(c, m), qos_profile_sensor_data)
            self.create_subscription(CameraInfo, f"{prefix}/color/camera_info",
                                     lambda m, c=camera: self.on_info(c, m), qos_profile_sensor_data)
        self.create_subscription(JointState, "joint_states", self.on_joints, 10)
        self.create_subscription(Clock, "clock", self.on_clock, qos_profile_sensor_data)
        if voice:
            self.create_subscription(Int16MultiArray, "microphone/audio", self.on_audio, 50)
            self.create_subscription(Bool, "microphone/speech_active", self.on_speech, 50)
        self.trajectory_pub = self.create_publisher(JointTrajectory, "joint_trajectory", 10)
        self.command_pub = self.create_publisher(JointState, "joint_command", 10)
        self.feedback_pub = self.create_publisher(String, "task/feedback", 10)
        self.annotated_pub = self.create_publisher(Image, "task/annotated", 2)
        threading.Thread(target=self.worker, daemon=True).start()
        if stdin:
            threading.Thread(target=self.read_stdin, daemon=True).start()
        self.say(
            f"ready{' (listening to the microphone)' if voice else ''}: "
            f"{'type a task here' if stdin else 'send a task on /task'}. "
            f"Memory: {len(self.memory.objects)} objects ('scan' to look around, 'memory' to list, 'forget' to clear)"
        )

    # --------------------------------------------------------------- inputs
    def on_rgb(self, camera: str, msg: Image) -> None:
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[..., :3]
        with self.lock:
            self.images[camera]["rgb"] = img.copy()

    def on_depth(self, camera: str, msg: Image) -> None:
        dtype = np.float32 if msg.encoding == "32FC1" else np.uint16
        d = np.frombuffer(msg.data, dtype).reshape(msg.height, msg.width).astype(np.float32)
        if msg.encoding == "16UC1":
            d /= 1000.0
        with self.lock:
            self.images[camera]["depth"] = d

    def on_info(self, camera: str, msg: CameraInfo) -> None:
        with self.lock:
            self.images[camera]["K"] = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def on_joints(self, msg: JointState) -> None:
        with self.lock:
            self.joints.update(zip(msg.name, msg.position))

    def on_clock(self, msg: Clock) -> None:
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def on_task(self, msg: String) -> None:
        text = msg.data.strip()
        self.log("USER", text)
        if text.lower() in ("stop", "cancel", "abort"):
            self.cancel.set()
            self.say("stopping after the current motion")
            return
        self.submit(Request(text=text))

    def read_stdin(self) -> None:
        """Tasks typed in this terminal (--stdin): same as messages on /task; 'quit' or Ctrl-D exits."""
        for line in sys.stdin:
            text = line.strip()
            if text.lower() in ("quit", "exit"):
                break
            if text:
                self.on_task(String(data=text))
        os.kill(os.getpid(), signal.SIGINT)

    def on_audio(self, msg: Int16MultiArray) -> None:
        if self.speaking and not self.too_long:
            self.audio_chunks.append(np.asarray(msg.data, np.int16))
            if sum(len(c) for c in self.audio_chunks) / 16000 > MAX_UTTERANCE_S:
                # Sound that never pauses is noise, not an instruction: drop it and wait for silence
                self.too_long, self.audio_chunks = True, []
                if time.monotonic() - self.last_noise_warning > 60.0:
                    self.last_noise_warning = time.monotonic()
                    self.say(
                        f"the microphone has heard sound for over {MAX_UTTERANCE_S:.0f} s without a pause, so "
                        "it is ignoring it as noise. If you are not speaking, the microphone gain is too high "
                        "(see the microphone node's warning)."
                    )

    def on_speech(self, msg: Bool) -> None:
        if msg.data and not self.speaking:
            self.speaking, self.too_long, self.audio_chunks = True, False, []
        elif not msg.data and self.speaking:
            self.speaking = False
            if self.too_long or not self.audio_chunks:
                return
            audio = np.concatenate(self.audio_chunks).astype(np.float32) / 32768.0
            if len(audio) / 16000 < MIN_UTTERANCE_S:
                return
            if self.busy.is_set():
                self.get_logger().debug("ignoring speech while busy")
                return
            self.submit(Request(audio=audio))

    def submit(self, request: Request) -> None:
        if self.busy.is_set():
            self.say("busy with the previous task; type 'stop' to cancel it")
            return
        try:
            self.requests.put_nowait(request)
        except queue.Full:
            self.say("a task is already waiting")

    # -------------------------------------------------------------- helpers
    def log(self, tag: str, text: str, detail: bool = False) -> None:
        """One log entry: '[TAG] text', indented by its level (see LOG_LEVELS; detail = one deeper).
        Printed here and published on /task/feedback."""
        head = "    " * (LOG_LEVELS.get(tag, 1) + detail) + f"[{tag}] "
        lines = str(text).splitlines() or [""]
        msg = head + lines[0] + "".join(f"\n{' ' * len(head)}{line}" for line in lines[1:])
        self.feedback_pub.publish(String(data=msg))
        print(msg, flush=True)

    def say(self, text: str) -> None:
        self.log("AGENT", text)

    def wait_still(self, tolerance_deg: float = 0.05, window: float = 0.3, timeout_sim: float = 6.0) -> None:
        """Wait until the arm has stopped moving: a photo taken while it moves gives wrong 3-D positions."""
        start, last_q, still_since = self.sim_time, None, None
        while self.sim_time - start < timeout_sim:
            with self.lock:
                q = np.array([self.joints.get(j, np.nan) for j in ARM_JOINTS])
            if last_q is not None and np.max(np.abs(q - last_q)) < np.deg2rad(tolerance_deg):
                still_since = still_since if still_since is not None else self.sim_time
                if self.sim_time - still_since >= window:
                    return
            else:
                still_since = None
            last_q = q
            time.sleep(0.05)

    def snapshot(self, camera: str = "wrist") -> Snapshot:
        """One consistent RGB + depth + pose capture, taken once the arm has stopped moving."""
        self.wait_still()
        time.sleep(0.2)  # let a fresh camera frame arrive after the arm stopped
        with self.lock:
            img = self.images[camera]
            if not all(k in img for k in ("rgb", "depth", "K")):
                raise RuntimeError(f"no {camera} camera images yet; is the simulation running?")
            if not all(j in self.joints for j in ARM_JOINTS):
                raise RuntimeError("no joint states yet; is the simulation running?")
            q = np.array([self.joints[j] for j in ARM_JOINTS])
            T = table_camera_pose() if camera == "table" else self.camera_kin.fk(q)
            return Snapshot(img["rgb"].copy(), img["depth"].copy(), img["K"], q, T, camera)

    def wait_sim(self, seconds: float) -> None:
        start, deadline = self.sim_time, time.monotonic() + max(30.0, 20 * seconds)
        while self.sim_time - start < seconds and time.monotonic() < deadline:
            time.sleep(0.02)

    def wait_reached(self, q_goal: np.ndarray, tol_deg: float = 1.5, timeout_sim: float = 10.0) -> bool:
        start = self.sim_time
        while self.sim_time - start < timeout_sim:
            with self.lock:
                q = np.array([self.joints.get(j, np.nan) for j in ARM_JOINTS])
            if np.max(np.abs(q - q_goal)) < np.deg2rad(tol_deg):
                return True
            time.sleep(0.02)
        return False

    def publish_trajectory(self, step: MoveStep) -> None:
        msg = JointTrajectory()
        msg.joint_names = list(ARM_JOINTS)
        for t, q in zip(step.trajectory.times, step.trajectory.positions):
            sec = int(t)
            msg.points.append(JointTrajectoryPoint(
                positions=[float(v) for v in q], time_from_start=Duration(sec=sec, nanosec=int((t - sec) * 1e9))))
        self.trajectory_pub.publish(msg)

    def set_gripper(self, position: float) -> None:
        msg = JointState(name=[GRIPPER_JOINT], position=[float(position)], velocity=[float(GRIPPER_SPEED)])
        self.command_pub.publish(msg)

    def annotate(self, snap: Snapshot, items: list[tuple[str, tuple, tuple]], stamp: str) -> None:
        """Draw (text, box 0-1000, rgb) items on the camera frame; save it and publish /task/annotated."""
        img = PILImage.fromarray(snap.rgb)
        draw = ImageDraw.Draw(img)
        h, w = snap.rgb.shape[:2]
        for text, (y0, x0, y1, x1), colour in items:
            rect = (x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000)
            draw.rectangle(rect, outline=colour, width=4)
            draw.text((rect[0] + 4, max(rect[1] - 16, 0)), text, fill=colour)
        TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)
        img.save(TASK_LOG_DIR / f"{stamp}.png")
        out = Image(height=h, width=w, encoding="rgb8", step=w * 3, data=np.asarray(img).tobytes())
        out.header.frame_id = "camera_color_optical_frame"
        self.annotated_pub.publish(out)

    def execute(self, steps: list) -> bool:
        """Run planned motion / gripper steps in order; False if cancelled."""
        for s in steps:
            if self.cancel.is_set():
                self.say("stopped")
                return False
            if isinstance(s, GripperStep):
                self.set_gripper(s.position)
                self.wait_sim(s.settle)
            else:
                self.publish_trajectory(s)
                self.wait_sim(s.trajectory.duration)
                if not self.wait_reached(s.trajectory.positions[-1]):
                    self.say(f"warning: '{s.label}' did not reach its goal within tolerance")
        return True

    # --------------------------------------------------------------- worker
    def worker(self) -> None:
        while rclpy.ok():
            request = self.requests.get()
            self.busy.set()
            self.cancel.clear()
            try:
                self.run_task(request)
            except Exception as exc:  # keep the node alive, report the problem
                self.say(f"task failed: {exc}")
            finally:
                self.busy.clear()

    def run_task(self, request: Request) -> None:
        text = request.text
        if request.audio is not None:  # speech: transcribe first, then treat like typed text
            wav = save_wav(request.audio)
            self.get_logger().info(f"heard {len(request.audio) / 16000:.1f} s of sound ({wav}), transcribing...")
            text = self.planner.transcribe(request.audio)
            if text is None:
                self.get_logger().info("  -> no clear speech in it (ignored)")
                return
            self.log("USER", f"(spoken) {text}")
            if text.strip().lower().rstrip(".!") in ("stop", "cancel", "abort"):
                self.say("nothing is running to stop")
                return
        command = text.strip().lower().rstrip(".!?")
        if command == "scan":
            self.scan()
            return
        if command in ("forget", "clear memory", "forget everything"):
            self.memory.clear()
            self.say("memory cleared")
            return
        if command in MEMORY_WORDS:
            searched = self.memory.describe_searched()
            self.say("I remember:\n" + self.memory.describe() + (f"\n{searched}" if searched else ""))
            return
        if self.pending is not None:  # answer to "where should I look?"
            pending, self.pending = self.pending, None
            hint = parse_direction(command)
            if hint is not None:
                if self.explore(pending["missing"], *hint):
                    self.run_task(Request(text=pending["task"], explored=True))
                return
            self.say("ok, dropping the previous task")  # not an answer: treat it as a new task
        if command in OVERVIEW_WORDS:
            self.survey()
            return
        if command in EXPLORE_WORDS:
            self.explore(None, None, None)
            return
        if self.run_direct(text):
            return

        self.survey()  # table camera: update memory with every object it can see right now
        while True:  # plan -> find every named object (ask where to look if one is missing)
            snap = self.snapshot("table")
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.say(f"planning '{text}' from the camera view and {len(self.memory.objects)} remembered objects...")
            try:
                plan = self.planner.plan(snap.rgb, text=text, memory=self.memory.describe())
            except PlanError as exc:
                self.say(f"the model's answer was not a usable plan: {exc}")
                return
            self.say(f"model ({self.planner.last_seconds:.1f} s): {plan.reply or '(no reply)'}")
            TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)
            (TASK_LOG_DIR / f"{stamp}.json").write_text(json.dumps(
                {"task": text, "spoken": request.audio is not None, "memory": self.memory.describe(),
                 "reply": plan.reply, "raw": plan.raw}, indent=2))

            missing = plan.steps[0].label if plan.steps and plan.steps[0].action == "search" else None
            T_world_camera = snap.T
            targets, items, detections = {}, [], None
            for i, step in enumerate(plan.steps if missing is None else []):
                if step.action in ("home", "move"):
                    continue
                if step.box is None and step.object_id is None and step.position_mm is None:
                    if detections is None:  # the model only named it: find it in this same frame
                        detections = self.planner.detect(snap.rgb)
                    step.box = self.find_in_view(step.label, detections, snap, T_world_camera)
                    if step.box is None:
                        remembered = self.memory.find(step.label)
                        if remembered is None:
                            missing = step.label
                            break
                        step.object_id = remembered.id
                try:
                    targets[i] = self.resolve(step, snap, T_world_camera)
                except (GroundingError, LookupError, ValueError) as exc:
                    self.annotate(snap, items, stamp)
                    self.say(f"step {i + 1} ({step.describe()}): {exc}; not moving")
                    return
                t = targets[i]
                if step.action == "pick":
                    self.say(f"target {t['label']}: {t['pose']} | {direction_text(t['position'])}")
                if step.box is not None:
                    items.append((f"{i + 1}. {step.describe()}: {t['pose']}", step.box,
                                  (255, 60, 60) if step.action == "pick" else (60, 220, 90)))
            self.annotate(snap, items, stamp)
            if missing is None:
                break
            self.memory.mark_not_found(missing, np.degrees(snap.q[0]), snap.camera)
            if request.explored:
                self.say(f"found the {missing} while exploring but cannot locate it from here; please point the "
                         "camera at it ('look left 30', ...) and ask again")
                return
            self.pending = {"task": text, "missing": missing}
            self.say(f"I can't find the {missing} from the table camera or in my memory (it may be hidden, "
                     "e.g. under the arm). Where should I look? Answer "
                     "left / right / front / behind (add 'near' or 'far' if you like), or 'search' to explore "
                     "on my own.")
            return
        if not plan.steps:
            return

        # Plan every motion before moving (reach, gripper width, singularities)
        q, held, sequence = snap.q, None, []
        try:
            for i, step in enumerate(plan.steps):
                if step.action == "pick":
                    t = targets[i]
                    steps, q, held = self.skills.pick(q, t["position"], width=t["width"])
                elif step.action == "place":
                    steps, q = self.skills.place(q, held, targets[i]["position"])
                elif step.action == "move":
                    cmd = self.direct.parse(step.label, dict(zip(ARM_JOINTS, q)))
                    if cmd.reply.startswith("did not understand"):
                        raise PlanningError(f"unknown move '{step.label}'")
                    traj = plan_command(self.kin, cmd, dict(zip(ARM_JOINTS, q)))
                    steps = [MoveStep(traj, step.label)] if traj is not None else []
                    q = traj.positions[-1] if traj is not None else q
                else:
                    steps, q = self.skills.home(q)
                sequence.append((i, step, steps, held))
        except PlanningError as exc:
            self.say(f"not moving: {exc}")
            return

        holding_id = None
        for n, (i, step, steps, held_now) in enumerate(sequence, 1):
            where = ""
            if i in targets:
                p = targets[i]["position"]
                where = f" at ({p[0] * 1000:.0f}, {p[1] * 1000:.0f}, {p[2] * 1000:.0f}) mm"
            self.say(f"step {n}/{len(sequence)}: {step.describe()}{where}")
            if not self.execute(steps):
                return
            # keep the memory in step with what the robot did
            if step.action == "pick" and targets[i]["memory_id"] is not None:
                holding_id = targets[i]["memory_id"]
                self.memory.pick(holding_id)
            elif step.action == "place" and holding_id is not None:
                support = targets[i]["position"]
                self.memory.place(holding_id, [support[0], support[1], support[2] + held_now.height])
                holding_id = None
        self.say("done")

    def find_in_view(self, query: str, detections, snap: Snapshot, T_world_camera: np.ndarray):
        """Box of the detection that best fits `query`, judged by MEASURED colour (then the model's label)."""
        best, best_score = None, 0
        for label, box in detections:
            try:
                est = locate(box, snap.depth, snap.K, T_world_camera, snap.rgb)
            except GroundingError:
                continue
            if not est.is_object:
                continue
            measured = measured_label(label, est)
            if not matches(query, measured):  # wrong shape, or a colour that is not even related
                continue
            exact = est.colour in query.lower().split()  # measured colour is exactly the one asked for
            agrees = matches(query, label)  # the model's own label fits too
            score = 2 * exact + agrees  # exact colour, or a related one (yellow/gold) that the model agrees with
            if score >= 1 and score > best_score:
                best, best_score = box, score
        return best

    def resolve(self, step, snap: Snapshot, T_world_camera: np.ndarray) -> dict:
        """3-D target of a plan step: {"position", "width", "memory_id"}."""
        if step.box is not None:
            est = locate(step.box, snap.depth, snap.K, T_world_camera, snap.rgb)
            if step.action == "place" and not est.is_object and step.label not in ("table", "the table", ""):
                raise LookupError(f"no {step.label} found where the model pointed (only bare table there)")
            memory_id = None
            if est.is_object:
                label = measured_label(step.label or "object", est)
                memory_id = self.memory.observe(label, est.position, est.size_xy, est.yaw_deg).id
            elif step.action == "pick":
                raise ValueError("the box contains no object above the table")
            label = measured_label(step.label or "object", est) if est.is_object else step.label
            return {"position": est.position, "width": est.size_xy[1], "memory_id": memory_id, "pose": est.pose_text(),
                    "label": label}
        if step.object_id is not None:
            obj = self.memory.get(step.object_id)
            if obj is None:
                raise LookupError(f"there is no object #{step.object_id} in memory")
            if obj.status != "on_table":
                raise LookupError(f"#{obj.id} {obj.label} is not on the table")
            return {"position": np.array(obj.position), "width": obj.size_xy[1], "memory_id": obj.id,
                    "pose": obj.pose_text() + f" (remembered #{obj.id})", "label": obj.label}
        if step.position_mm is not None:
            x, y = (v / 1000.0 for v in step.position_mm)
            return {"position": np.array([x, y, 0.0]), "width": 0.0, "memory_id": None,
                    "pose": f"x {x * 1000:.0f} mm, y {y * 1000:.0f} mm (table spot)", "label": "table spot"}
        raise ValueError("no target given")

    def survey(self) -> list[tuple[str, object]]:
        """Table camera: detect + locate every object it can see and update memory. Logs a direction
        for each (front / left / right / behind, bearing, distance, reachable)."""
        try:
            snap = self.snapshot("table")
        except RuntimeError as exc:
            self.say(f"table camera unavailable ({exc}); using memory and the wrist camera")
            return []
        detections = self.planner.detect(snap.rgb)
        seen, items, lines = [], [], []
        for label, box in detections:
            try:
                est = locate(box, snap.depth, snap.K, snap.T, snap.rgb)
            except GroundingError:
                continue
            if not est.is_object or not 0.25 < np.hypot(*est.position[:2]) < 1.5:
                continue  # floor, or the robot base
            if est.top_z > MAX_OBJECT_TOP or "microphone" in label:
                continue  # the robot arm or the microphone stand
            label = measured_label(label, est)
            obj = self.memory.observe(label, est.position, est.size_xy, est.yaw_deg)
            self.memory.found(label)
            seen.append((label, obj))
            items.append((f"#{obj.id} {label}", box, (255, 200, 40)))
            lines.append(f"  #{obj.id} {label}: {est.pose_text()} | {direction_text(est.position)}")
        self.annotate(snap, items, time.strftime("table_%Y%m%d_%H%M%S"))
        self.say(f"table camera view ({self.planner.last_seconds:.1f} s): {len(seen)} objects\n" + "\n".join(lines))
        return seen

    def look(self, heading_deg: float, view: str, q: np.ndarray, tag: str) -> list[tuple[str, object]]:
        """Point the camera (view 'near' / 'mid' / 'far' at a J1 heading), detect, locate and remember objects.

        Returns [(label, memory object)] for the objects seen in this view.
        """
        pose = np.asarray(VIEW_POSES[view], float).copy()
        pose[0] = np.deg2rad(heading_deg)
        label = f"look at {heading_deg:+.0f} deg, {view}"
        if not self.execute([MoveStep(joint_move(self.kin, q, pose, ARM_SPEED), label)]):
            raise InterruptedError
        snap = self.snapshot()  # waits until the arm is still
        detections = self.planner.detect(snap.rgb)
        T_world_camera = snap.T
        seen, items = [], []
        for label, box in detections:
            try:
                est = locate(box, snap.depth, snap.K, T_world_camera, snap.rgb)
            except GroundingError:
                continue
            if not est.is_object or not 0.25 < np.hypot(*est.position[:2]) < 1.5:
                continue  # floor, or the robot itself
            if est.top_z > MAX_OBJECT_TOP or "microphone" in label:
                continue  # the microphone stand
            label = measured_label(label, est)
            obj = self.memory.observe(label, est.position, est.size_xy, est.yaw_deg)
            seen.append((label, obj))
            items.append((f"#{obj.id} {label}: {est.pose_text()}", box, (255, 200, 40)))
        self.annotate(snap, items, time.strftime(f"{tag}_%Y%m%d_%H%M%S"))
        return seen

    def current_q(self) -> np.ndarray:
        with self.lock:
            return np.array([self.joints[j] for j in ARM_JOINTS])

    def run_direct(self, text: str) -> bool:
        """Direct commands (move left 100, look up 20, turn around, observe, open...) without the model."""
        with self.lock:
            current = dict(self.joints)
        cmd = self.direct.parse(text, current)
        if not cmd.reply or cmd.reply.startswith("did not understand"):
            return False
        try:
            traj = plan_command(self.kin, cmd, current)
        except PlanningError as exc:
            self.say(f"{cmd.reply.splitlines()[0]}: not moving, {exc}")
            return True
        self.say(cmd.reply + (f" [{traj.description}]" if traj is not None else ""))
        if GRIPPER_JOINT in cmd.targets:
            self.set_gripper(cmd.targets[GRIPPER_JOINT])
        if traj is not None:
            self.execute([MoveStep(traj, text)])
        return True

    def scan(self) -> None:
        """Map the table: mid and far views at 7 headings; every object seen goes into memory."""
        views = [(h, v) for h in SCAN_HEADINGS_DEG for v in ("mid", "far")]
        self.say(f"scanning {len(views)} views...")
        before, q = len(self.memory.objects), self.current_q()
        try:
            for k, (heading, view) in enumerate(views, 1):
                seen = self.look(heading, view, q, f"scan{k}")
                q = VIEW_POSES[view].copy()
                q[0] = np.deg2rad(heading)
                self.say(f"view {k}/{len(views)} (J1 {heading:+d} deg, {view}): "
                         + (", ".join(f"#{o.id} {label}" for label, o in seen) if seen else "nothing"))
        except InterruptedError:
            return
        self.execute([MoveStep(joint_move(self.kin, q, np.asarray(OBSERVE_POSE, float), ARM_SPEED), "back to observe")])
        self.say(f"scan done: {len(self.memory.objects) - before} new, {len(self.memory.objects)} remembered\n"
                 + self.memory.describe())

    def explore(self, query: str | None, sector: str | None, distance: str | None) -> bool:
        """Look around for `query` (None: map everything) across near / mid / far views at every heading.

        The user's hint comes first (sector 'left' / 'right' / 'front' / 'behind', distance 'near' / 'far'),
        then the rest, nearest heading first. Views where it was already looked for are skipped, and each
        empty view is remembered as "not found here". Stops at the first match; True if found.
        """
        q = self.current_q()
        here = np.degrees(q[0])
        headings = [float(h) for h in np.arange(-SEARCH_LIMIT_DEG, SEARCH_LIMIT_DEG + 1, SEARCH_STEP_DEG)]
        view_order = {"near": ["near", "mid", "far"], "far": ["far", "mid", "near"]}.get(distance, ["mid", "far", "near"])
        views = [(h, v) for h in headings for v in view_order]
        in_sector = {
            "left": lambda h: 30 <= h, "right": lambda h: h <= -30,
            "front": lambda h: abs(h) <= 60, "behind": lambda h: abs(h) >= 105,
        }.get(sector, lambda h: True)
        views.sort(key=lambda hv: (not in_sector(hv[0]), abs(hv[0] - here), view_order.index(hv[1])))
        if query is not None:
            skipped = [hv for hv in views if self.memory.was_searched(query, *hv)]
            views = [hv for hv in views if hv not in skipped]
            hint = f" ({sector or 'all directions'}{', ' + distance if distance else ''} first)"
            note = f"; skipping {len(skipped)} views already searched" if skipped else ""
            self.say(f"exploring for the {query}: {len(views)} views{hint}{note}")
        else:
            self.say(f"exploring the surroundings: {len(views)} views")
        try:
            for k, (heading, view) in enumerate(views, 1):
                seen = self.look(heading, view, q, f"explore{k}")
                q = VIEW_POSES[view].copy()
                q[0] = np.deg2rad(heading)
                where = f"view {k}/{len(views)} (J1 {heading:+.0f} deg, {view})"
                if query is None:
                    self.say(f"{where}: " + (", ".join(f"#{o.id} {label}" for label, o in seen) or "nothing"))
                    continue
                hits = [(label, obj) for label, obj in seen if matches(query, label)]
                if hits:
                    label, obj = hits[0]
                    self.memory.found(query)
                    self.say(f"{where}: found {label} (#{obj.id}): {obj.pose_text()}")
                    return True
                self.memory.mark_not_found(query, heading, view)
                self.say(f"{where}: " + (", ".join(label for label, _ in seen) or "nothing") + "; not it")
        except InterruptedError:
            return False
        if query is not None:
            self.say(f"looked in every view and found no {query}; I remember where I looked "
                     "('memory' lists it, 'forget' clears it)")
        else:
            self.say("exploration done\n" + self.memory.describe())
        return False


# m from the base, from the grasp-feasibility map (evaluation/grasp_feasibility_map.py, 2026-10-14):
# top-down picks plan for every grip yaw from REACH_MIN to REACH_MAX; between REACH_PARTIAL and
# REACH_MIN it depends on the bearing and the grip yaw
REACH_PARTIAL, REACH_MIN, REACH_MAX = 0.375, 0.45, 0.90


def direction_text(position) -> str:
    """Holistic direction of a world point as seen from the robot: side, bearing (the J1 angle that faces
    it), distance, and whether a top-down pick can reach it."""
    x, y = float(position[0]), float(position[1])
    bearing, dist = np.degrees(np.arctan2(-x, y)), np.hypot(x, y)
    side = "front" if abs(bearing) <= 45 else "left" if 45 < bearing <= 135 else "right" if -135 <= bearing < -45 else "behind"
    reach = ("reachable" if REACH_MIN <= dist <= REACH_MAX else "near edge: depends on the grip angle"
             if REACH_PARTIAL <= dist < REACH_MIN else "too close" if dist < REACH_PARTIAL else "out of reach")
    if abs(bearing) > 170:
        reach = "behind the J1 limit"
    return f"{side}, bearing {bearing:+.0f} deg, {dist * 1000:.0f} mm from the base, {reach}"


def parse_direction(text: str) -> tuple[str | None, str | None] | None:
    """User's answer to 'where should I look?' -> (sector, distance); None if it is not such an answer."""
    words = set(text.lower().replace(",", " ").split())
    sector = next((v for k, v in (("left", "left"), ("right", "right"), ("behind", "behind"), ("back", "behind"),
                                  ("front", "front"), ("forward", "front"), ("ahead", "front")) if k in words), None)
    distance = "near" if words & {"near", "close", "nearby"} else "far" if words & {"far", "away", "distant"} else None
    anywhere = words & {"search", "explore", "anywhere", "everywhere", "around", "know", "idea", "sure"}
    if sector is None and distance is None and not anywhere:
        return None
    return sector, distance


def measured_label(model_label: str, est) -> str:
    """'<measured colour> <shape>': the depth+colour measurement is more reliable than the model's colour
    word (it confuses orange / yellow); the shape word still comes from the model."""
    if est.colour is None:
        return model_label
    shape = next((w for w in model_label.lower().split() if w in SHAPES), "object")
    return f"{est.colour} {SHAPES.get(shape, shape)}"


def save_wav(audio: np.ndarray, rate: int = 16000) -> Path:
    import wave

    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    path = VOICE_DIR / time.strftime("utterance_%Y%m%d_%H%M%S.wav")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
    return path.relative_to(ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", action="store_true", help="also take spoken tasks from the microphone node")
    parser.add_argument("--stdin", action="store_true", help="also read tasks typed in this terminal")
    parser.add_argument("--memory", type=Path, default=MEMORY_FILE, help="object memory file (JSON)")
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default=None)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None,
                        help="Gemma reasoning mode on/off (default: vlm/config.py)")
    args, ros_args = parser.parse_known_args()

    from vlm.config import VLMConfig
    from vlm.model import VisionLanguageModel
    from vlm.planner import TaskPlanner

    config = VLMConfig()
    if args.thinking is not None:
        config.thinking = args.thinking
    if args.quantization:
        config.quantization = args.quantization
    print(f"[vlm_agent] loading {config.model_path} ({config.quantization}, thinking {'on' if config.thinking else 'off'})...", flush=True)
    model = VisionLanguageModel(config)
    print(f"[vlm_agent] model loaded in {model.load_seconds:.1f} s, GPU {model.memory_gb():.1f} GB", flush=True)

    rclpy.init(args=ros_args)
    node = VLMAgent(TaskPlanner(model), voice=args.voice, memory_file=args.memory, stdin=args.stdin)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
