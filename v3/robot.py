"""The robot: cameras, measurements and motion. Everything physical is here and nothing is decided here.

It wraps v2's tested code (segmentation, depth geometry, motion planner, collision checks, executor) and
gives the agents five plain actions: look, transfer, pick, home, holding.
"""

import time

import cv2
import numpy as np

from agent import run as v2
from agent.tools import perception as v2_perception
from control.motion import PlanningError
from control.robot import GRIPPER_OPEN, finger_angle_for_gap
from control.skills import GripperStep
from perception.detector import Detection

GRIP_SQUEEZE_M = 0.016  # the fingers close to (object width - this); stopping short of it means contact
_v2_parts = v2_perception.parent_of_parts


def parts(masks, measured):
    """Which YOLO outlines are a part of another object: {part: body}.

    v2's rule, plus the case it misses: YOLO outlines a bottle's cap separately from the bottle. The cap is a
    smaller outline whose centre is inside a tall body's footprint, reaching at most 35 mm above the body's top
    and touching it in the picture. Joined, they are measured as one object again."""
    parents = _v2_parts(masks, measured)
    for k, g in measured.items():
        if k in parents or k in parents.values():
            continue
        bodies = []
        for j, h in measured.items():
            if j == k or j in parents or masks[j].sum() < 1.5 * masks[k].sum():
                continue
            apart = np.hypot(g["centre"][0] - h["centre"][0], g["centre"][1] - h["centre"][1])
            near = cv2.dilate(masks[j].astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
            if (h["height"] > 1.3 * max(h["footprint"]) and max(g["footprint"]) < min(h["footprint"])
                    and -0.006 < g["top_z"] - h["top_z"] < 0.035 and apart < 0.5 * min(h["footprint"])
                    and (masks[k] & near).any()):
                bodies.append(j)
        if len(bodies) == 1:
            parents[k] = bodies[0]
    return parents


v2_perception.parent_of_parts = parts  # v2's perceive() now uses the rule above; no v2 file is changed


def solid(patch, smallest=150):
    """The compact pieces of a pixel patch, largest first; thin rims and specks are dropped."""
    opened = cv2.morphologyEx(patch.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    sizes = {i: stats[i, cv2.CC_STAT_AREA] for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= smallest}
    return [labels == i for i in sorted(sizes, key=sizes.get, reverse=True)]


class Outlines:
    """YOLO's outlines, corrected with what the robot knows it put down.

    The table camera does not move, so after a release the pixels that came closer to the camera since the last
    table picture are the object just put down. That patch is remembered for as long as its depth stays the same.
    1. YOLO often draws the tray's outline, or one ring around a group, over a put-down object. The patch is cut
       out of every such outline, so a tray is measured as a tray and not as tall as the carton standing on it.
    2. When YOLO draws no outline of its own for a put-down object, the patch is its outline.
    3. An extra outline that only lies on other outlines is dropped."""

    def __init__(self, tools):
        self.yolo, self.snapshot = tools.detector, tools.bridge.snapshot
        self.frame = self.depth = self.home = None  # the frame looked at; the last table picture's depth; the arm's pose in it
        self.released, self.placed = False, []  # an object was just put down; the patches of put-down objects
        tools.detector, tools.bridge.snapshot = self, self.take

    def take(self, camera):
        self.frame = self.snapshot(camera)
        return self.frame

    def detect(self, rgb):
        found = self.yolo.detect(rgb)
        if self.frame is None or self.frame.camera != "table":
            return found
        depth, q = np.nan_to_num(self.frame.depth, posinf=0.0), np.asarray(self.frame.q)
        self.home = q if self.home is None else self.home
        if np.abs(q - self.home).max() < 0.03:  # only pictures with the arm in the same pose are compared
            if self.depth is not None:
                changed = np.abs(depth - self.depth) > 0.01
                new = solid(changed & (depth > 0) & (depth < self.depth))[:1] if self.released else []
                for patch in new:  # the new object hides part of what was put down before it
                    self.placed = [old & ~patch for old in self.placed]
                self.placed = [p for p in self.placed if p.sum() >= 150 and changed[p].mean() < 0.5] + new
            self.depth, self.released = depth, False
        cut = []  # what is left of the outlines that reached over a put-down object
        for patch in self.placed:
            size, wide = patch.sum(), cv2.dilate(patch.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            inside = [d for d in found if (d.mask & patch).sum() > 0.5 * d.mask.sum()]  # its outline, or its cap's
            own = [d for d in inside if (d.mask & patch).sum() > 0.5 * size]
            for d in [d for d in found if (d.mask & patch).any() and not any(d is i for i in inside)]:
                rest = np.any(solid(d.mask & ~wide) or [np.zeros_like(patch)], axis=0)  # a tray, a ring around a group
                reached = (d.mask & patch).sum() > 0.2 * size or any(d is c for c in cut)
                found, cut = [f for f in found if f is not d], [c for c in cut if c is not d]
                if rest.any():
                    found.append(Detection(rest, d.score))
                    cut += [found[-1]] if reached else []
            if not own:  # YOLO drew no outline of its own for it: the patch is its outline
                found.append(Detection(patch, 1.0))
        for d in sorted(cut, key=lambda d: d.mask.sum()):  # a ring's rest lies on the tray: not an object
            others = np.any([f.mask for f in found if f is not d], axis=0)
            if (d.mask & others).sum() > 0.6 * d.mask.sum():
                found = [f for f in found if f is not d]
        return [d for d in found if not self.group(d, found)]

    @staticmethod
    def group(outline, found):
        """Is this an extra outline drawn over parts of two or more objects that have their own outlines?"""
        size = outline.mask.sum()
        parts = [d.mask & outline.mask for d in found if d is not outline]
        parts = [p for p in parts if p.sum() > 0.15 * size]
        return len(parts) >= 2 and np.any(parts, axis=0).sum() > 0.6 * size


class Robot:
    def __init__(self, tools):
        self.t = tools
        self.outlines = Outlines(tools)
        self.moved = False  # has the arm moved during this instruction
        self.choose = None  # agents.py puts the worker's spot choice here: (picture, letters, object, destination) -> number
        self.spot = None  # the spot the worker chose for the object being moved now
        free = tools._free_spots
        tools._free_spots = lambda rid, held, **options: self._spots(*free(rid, held, **options), rid)
        self.say = lambda who, text, started=None: None  # run.py puts its log function here

    # ------------------------------------------------------------------ seeing
    def look(self):
        """One table picture. Returns (plain picture, numbered picture, {id: measurements + close-up})."""
        t = self.t
        obs = t.perceive("table")
        t.update_world(obs)
        numbered, objects = obs.rgb.copy(), {}
        self.view, self.boxes = obs, {}  # kept to draw the free spots of a destination on this picture
        for mark, oid in sorted(obs.object_ids.items(), key=lambda item: item[1]):
            if oid == t.held_object or mark not in obs.marks:
                continue
            g, mask = t.world.objects[oid].geometry, obs.marks[mark].astype(np.uint8)
            x, y, w, h = self.boxes[oid] = cv2.boundingRect(mask)
            pad = max(int(0.4 * max(w, h)), 20)
            crop = obs.rgb[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
            scale = max(1, int(np.ceil(448 / max(crop.shape[:2]))))
            objects[oid] = {
                "width": round(g["footprint"][0] * 1000), "length": round(g["footprint"][1] * 1000),
                "height": round(g["height"] * 1000), "x": round(g["centre"][0] * 1000), "y": round(g["centre"][1] * 1000),
                "crop": cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC),
            }
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(numbered, contours, -1, (0, 0, 0), 2)
            tag = f"#{oid}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            tx, ty = x, max(y - 6, th + 4)
            cv2.rectangle(numbered, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 3), (0, 0, 0), -1)
            cv2.putText(numbered, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(t.run_dir / "images" / f"{obs.frame_id}_numbered.jpg"), cv2.cvtColor(numbered, cv2.COLOR_RGB2BGR))
        return obs.rgb, numbered, objects

    def holding(self):
        """The id of the object in the gripper, or None (from the finger sensors)."""
        return self.t.held_object

    # ------------------------------------------------------------------ moving
    def _do(self, steps, label):
        self.moved, started = True, time.monotonic()
        self.t.log("state", state=label.upper())  # the live screen shows what the arm is doing now
        result = self.t.execute(steps, label)
        self.say("robot", label if result.ok else f"{label} FAILED: {result.reason}", started)
        if not result.ok:
            raise RuntimeError(f"{label} failed: {result.failed_step}: {result.reason}")

    def _grasp(self, plan):
        """Hover, descend, close, lift. True when the fingers hold the object after the lift."""
        t = self.t
        close = GripperStep(finger_angle_for_gap(max(plan.held.width - GRIP_SQUEEZE_M, 0.0)), "close gripper", 1.0)
        self._do(plan.hover + plan.approach[:1], "approach")
        self._do([close], "close")
        touching = t.verify_contact(close.position).get("pass")
        self.say("robot", "finger check after closing: " + ("the fingers touch the object" if touching else "NO contact"))
        t.set_held(plan, close.position)
        self._do(plan.lift, "lift")
        held = t.confirm_held(close.position, plan.object_id, "retention")["pass"]
        self.say("robot", "finger check after the lift: " + ("the object is held" if held else "NOTHING is held"))
        if held:
            return True
        t.clear_held()
        self._do([GripperStep(GRIPPER_OPEN, "open gripper", 0.8)], "open")  # a closed gripper cannot go HOME
        return False

    def _ready(self, object_id, reference_id=None):
        """Before a pick: empty open gripper, arm at HOME. Marks the chosen objects on the live screen."""
        started = time.monotonic()
        self.t.log("localized", object_mark=object_id, object_id=object_id, reference_mark=reference_id, reference_id=reference_id)
        ready = self.t.prepare_pick()
        self.say("robot", "ready: gripper open and empty, arm at HOME" if ready["ok"] else "NOT ready to start", started)
        return ready

    def _plan(self, planner, *arguments):
        started = time.monotonic()
        plan = planner(*arguments, self.t.bridge.q())
        self.say("robot", "motion planned and collision-checked", started)
        return plan

    def _spots(self, spots, inside, rid):
        """v2 lists the free spots on a tray or in a basket, nearest its centre first, and tries the first twelve.
        v3 lets the worker choose: up to twelve spots spread over the destination are drawn on the table picture
        as lettered dots, and the spot the worker picks is tried first, then the ones nearest to it."""
        if not spots or self.choose is None:
            return spots, inside
        xy = np.asarray(spots)
        if self.spot is None:
            shown = [0]
            while len(shown) < 12:  # each time the spot farthest from those already shown, while they stay apart
                apart = np.linalg.norm(xy[:, None] - xy[shown], axis=2).min(axis=1)
                if apart.max() < 0.035:
                    break
                shown.append(int(apart.argmax()))
            g, frame = self.t.world.objects[rid].geometry, self.view.frame
            (fx, fy, cx, cy), camera = frame.K, self.t._camera_pose(frame)
            z = g["top_z"] if self.relation == "on" else g["top_z"] - g["height"]
            seen = (np.c_[xy[shown], np.full(len(shown), z)] - camera[:3, 3]) @ camera[:3, :3]
            dots = np.c_[fx * seen[:, 0] / seen[:, 2] + cx, fy * seen[:, 1] / seen[:, 2] + cy]
            rows = sorted(range(len(shown)), key=lambda i: (round(dots[i, 1] / 30), dots[i, 0]))  # lettered as one reads
            shown, dots = [shown[i] for i in rows], dots[rows]
            x, y, w, h = self.boxes[rid]
            left, top = int(min(x, dots[:, 0].min())) - 40, int(min(y, dots[:, 1].min())) - 40
            right, bottom = int(max(x + w, dots[:, 0].max())) + 40, int(max(y + h, dots[:, 1].max())) + 40
            left, top = max(left, 0), max(top, 0)
            crop = self.view.rgb[top:bottom, left:right]
            zoom = max(1, round(900 / max(crop.shape[:2])))
            picture = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
            letters = [chr(65 + i) for i in range(len(shown))]
            for letter, (u, v) in zip(letters, (dots - (left, top)) * zoom):
                cv2.circle(picture, (int(u), int(v)), 15, (255, 255, 255), -1)
                cv2.circle(picture, (int(u), int(v)), 15, (0, 0, 0), 2)
                (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.putText(picture, letter, (int(u - tw / 2), int(v + th / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imwrite(str(self.t.run_dir / "images" / f"{self.view.frame_id}_spots_{self.picking}.jpg"),
                        cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))
            chosen = self.choose(picture, letters, self.picking, rid)
            self.spot = xy[shown[chosen]] if chosen is not None else xy[0]
            self.t.log("tool", tool="spot_choice", object_id=self.picking, reference_id=rid, letters=len(letters),
                       chosen=letters[chosen] if chosen is not None else None,
                       spot_mm=[round(float(v) * 1000, 1) for v in self.spot])
        nearest = np.argsort(np.linalg.norm(xy - self.spot, axis=1), kind="stable")
        return [spots[i] for i in nearest], inside

    def transfer(self, object_id, relation, reference_id):
        """Pick one object and place it. Returns (ok, what happened) for the master."""
        t, self.spot, self.picking, self.relation = self.t, None, object_id, relation
        try:
            ready = self._ready(object_id, reference_id)
            if not ready["ok"]:
                return False, f"could not start from HOME with an empty gripper: {ready.get('reason')}"
            plan = self._plan(t.plan_pick_place, object_id, relation, reference_id, "robot")
            if not self._grasp(plan):
                return self._home(False, "the gripper closed but did not hold the object after lifting; nothing was moved")
            self._do(plan.place[:1], "carry")
            self._do(plan.place[1:2], "lower")
            self._do(plan.place[2:-1], "release")
            self.outlines.released = True
            t.assume_placed(plan)
            self._do(plan.place[-1:], "retreat")
            t.clear_held()
            return self._home(True, "the object was carried to the destination and released there")
        except PlanningError as error:
            return self._home(False, f"no safe motion was found: {error}")
        except RuntimeError as error:
            return self._home(False, str(error))

    def pick(self, object_id):
        """Pick one object and keep holding it."""
        t = self.t
        try:
            ready = self._ready(object_id)
            if not ready["ok"]:
                return False, f"could not start from HOME with an empty gripper: {ready.get('reason')}"
            plan = self._plan(t.plan_pick, object_id)
            if self._grasp(plan):
                return True, "the object is held in the gripper above the table"
            return self._home(False, "the gripper closed but did not hold the object after lifting")
        except PlanningError as error:
            return self._home(False, f"no safe motion was found: {error}")
        except RuntimeError as error:
            return self._home(False, str(error))

    def home(self):
        return self._home(True, "the arm is at HOME")

    def _home(self, ok, text):
        """Every action ends at HOME so the next table picture is not blocked by the arm."""
        t = self.t
        if t.held_object is not None:
            return ok, text + "; the arm stayed where it is because it is holding an object"
        self.moved, started = True, time.monotonic()
        t.log("state", state="RETURNING_HOME")
        result = t.return_home()
        if not result["ok"]:  # the path started from an old joint reading: plan once more from where the arm is now
            result = t.return_home()
        at_home = result["ok"] and t.verify_home()["pass"]
        self.say("robot", "back at HOME, confirmed by the joint sensors" if at_home else "did NOT reach HOME", started)
        if at_home:
            return ok, text + "; the arm is back at HOME"
        return False, text + f"; the arm did NOT reach HOME ({result.get('reason') or 'joint check failed'})"


def start(scenario, difficulty, seed, out, gui=False, live=None):
    """Load the models, start Isaac Sim and connect. Returns (robot, gemma, tools, stop function).

    live: the folder the live screen reads (v2's dashboard); v2's tools write each table picture's outlines there."""
    from argparse import Namespace
    from agent.controls import ExecutionControls
    from agent.eventlog import EventLog
    from agent.gemma import Gemma
    from agent.live import LiveChannel
    from agent.tools import Tools
    from config import models
    from perception.detector import YoloDetector

    out.mkdir(parents=True, exist_ok=True)
    print("[v3] loading Gemma and YOLO ...", flush=True)
    gemma = Gemma(None)
    detector = YoloDetector(str(models.YOLO_WEIGHTS), models.DETECTOR_IMGSZ, models.DETECTOR_CONF)
    print("[v3] starting Isaac Sim ...", flush=True)
    sim = v2.start_sim(Namespace(scenario=scenario, difficulty=difficulty, seed=seed, gui=gui), out / "sim.log")
    try:
        from agent.ros_bridge import RosBridge
        from execution.ros_executor import RosExecutor
        bridge, executor = RosBridge(), RosExecutor()
        if not v2.connect(bridge, executor):
            raise RuntimeError("the simulator did not become ready")
        run_dir = out / "00_start"
        (run_dir / "images").mkdir(parents=True, exist_ok=True)
        log = EventLog(run_dir / "events.jsonl", echo=False)
        gemma.log, gemma.run_dir = log, run_dir
        tools = Tools(bridge, detector, gemma, executor, log, run_dir, LiveChannel(live) if live else None,
                      ExecutionControls(motion="real"))
        tools.matcher = None
        tools.revalidate_held()
        return Robot(tools), gemma, tools, lambda: v2.stop_sim(sim)
    except BaseException:
        v2.stop_sim(sim)
        raise
