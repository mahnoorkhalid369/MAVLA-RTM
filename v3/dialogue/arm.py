"""The arm of v3/dialogue: the frozen v3 robot (outlines, measuring, motion) plus what the conversation needs.

Gemma marks an object and a place in the table picture. This file only measures and keeps the arm safe: which
measured object a box means, where a marked place is on the table in millimetres, the nearest spot there where the
object fits and the arm can put it down, and a picture that shows that spot back. Nothing is decided here.
"""

import dataclasses

import cv2
import numpy as np

import v3_config as config  # not "config": that name is v2's settings folder
import robot as frozen  # v3/robot.py, not edited
from agent.tools import manipulation as v2_manipulation
from agent.tools.manipulation import REACH_MAX_M, REACH_MIN_M
from control.motion import PlanningError
from control.skills import SkillParams


@dataclasses.dataclass
class Speeds(SkillParams):
    """v2's motion settings with the arm speeds of v3_config.py."""
    tcp_speed: float = config.LINE_SPEED
    place_speed: float = config.LOWER_SPEED
    joint_speed: float = config.JOINT_SPEED


v2_manipulation.SkillParams = Speeds  # v2's planner takes its settings from here: this version's speeds, v2's files unchanged
v2_manipulation.ARM_SPEED = config.JOINT_SPEED  # the moves from and to HOME


FINGER_ROOM_M = 0.03  # free room kept between a put-down object and its neighbours, for the open fingers


class Robot(frozen.Robot):
    def __init__(self, tools):
        super().__init__(tools)
        self.target = self.support = self.moving = None  # the marked place (world x, y), the object under it (None: the table), the object to move
        self.names = {}  # object id -> name, filled by the agents, for the sentences below
        self._v2_place = tools._plan_place
        tools._plan_place = self._place

    # ------------------------------------------------------------------ measuring
    def look(self):
        """As the frozen robot, and measures for every object how deep it is inside (a tray, a box without lid)."""
        rgb, numbered, objects = super().look()
        frame = self.view.frame
        fx, fy, cx, cy = frame.K
        self.K, self.T = frame.K, self.t._camera_pose(frame)
        v, u = np.mgrid[:frame.depth.shape[0], :frame.depth.shape[1]]
        z = np.where(np.isfinite(frame.depth) & (frame.depth > 0), frame.depth, np.nan)
        world = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1) @ self.T[:3, :3].T + self.T[:3, 3]
        for mark, oid in self.view.object_ids.items():
            if oid not in objects:
                continue
            g = self.t.world.objects[oid].geometry
            points = world[self.view.marks[mark].astype(bool)]
            points = points[np.isfinite(points[:, 2])]
            a = np.radians(g.get("yaw_deg") or 0.0)
            along = (points[:, :2] - g["centre"][:2]) @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            share = np.abs(along / (np.asarray(g["extents"][:2]) / 2)).max(axis=1)  # 0 in the middle, 1 at the edge
            middle, rim = points[share < 0.6, 2], points[(share > 0.8) & (share < 1.05), 2]
            deep = float(np.percentile(rim, 90) - np.median(middle)) if len(middle) > 50 and len(rim) > 50 else 0.0
            objects[oid]["hollow"] = max(round(deep * 1000), 0)  # mm the inside lies below the rim
        self.objects = objects
        return rgb, numbered, objects

    def meant(self, box, skip=None):
        """The measured object a box (x0, y0, x1, y1 in pixels) drawn on the table picture means: the one whose
        outline it covers best. Returns (object id, overlap from 0 to 1)."""
        best = (None, 0.0)
        for oid, (x, y, w, h) in self.boxes.items():
            if oid == skip or oid not in self.objects:
                continue
            both = max(0, min(box[2], x + w) - max(box[0], x)) * max(0, min(box[3], y + h) - max(box[1], y))
            overlap = both / (w * h + (box[2] - box[0]) * (box[3] - box[1]) - both + 1e-9)
            if overlap > best[1]:
                best = (oid, overlap)
        return best

    def zoom(self, box, around=0):
        """A close-up of the table picture around a box, with `around` more pixels of table on every side.
        Returns (picture, (left, top, width, height, scale))."""
        h, w = self.view.rgb.shape[:2]
        pad = max(60, int(0.25 * max(box[2] - box[0], box[3] - box[1]))) + around
        left, top = max(int(box[0]) - pad, 0), max(int(box[1]) - pad, 0)
        crop = self.view.rgb[top:min(int(box[3]) + pad, h), left:min(int(box[2]) + pad, w)]
        scale = max(1, round(900 / max(crop.shape[:2])))
        picture = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return picture, (left, top, crop.shape[1], crop.shape[0], scale)

    def _pixel(self, point):
        fx, fy, cx, cy = self.K
        seen = (np.asarray(point, float) - self.T[:3, 3]) @ self.T[:3, :3]
        return fx * seen[0] / seen[2] + cx, fy * seen[1] / seen[2] + cy

    # ------------------------------------------------------------------ placing at a marked place
    def aim(self, object_id, support, u, v):
        """Plan, without moving, to pick an object and put it down as near as possible to pixel (u, v) of the table
        picture: on or in the measured object `support`, or on the table when it is None. Returns (plan, problem)."""
        t = self.t
        if support is None:
            height, relation = 0.0, "next_to"  # v2 needs a word here; _place below does the work
        else:
            g = t.world.objects[support].geometry
            inside = self.objects[support]["hollow"] > 10
            height, relation = (t._container_floor(g), "in") if inside else (g["top_z"], "on")
        fx, fy, cx, cy = self.K
        ray = self.T[:3, :3] @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0])  # where the camera ray meets that height
        self.full = False
        self.support, self.moving, self.target = support, object_id, (self.T[:3, 3] + (height - self.T[2, 3]) / ray[2] * ray)[:2]
        try:
            ready = self._ready(object_id, support)
            if not ready["ok"]:
                return None, f"the arm could not start from HOME with an empty gripper: {ready.get('reason')}"
            return self._plan(t.plan_pick_place, object_id, relation, support, "robot"), None
        except PlanningError as error:
            if getattr(self, "full", False):
                return None, "no place is left there where it fits with room for the open fingers beside the objects already standing there"
            return None, f"no safe way was found to pick it up and put it down there: {error}"

    def _spots(self, spots, inside, rid):
        """v2 lists the free spots on a tray or in a box and tries the first twelve: nearest the marked place first."""
        if not spots or self.target is None:
            return spots, inside
        spots = [spot for spot in spots if self._crowded(spot) is None]
        self.full = not spots  # nothing left where it fits with room for the fingers
        near = np.argsort(np.linalg.norm(np.asarray(spots).reshape(-1, 2) - self.target, axis=1), kind="stable")
        return [spots[i] for i in near], inside

    def _crowded(self, xy):
        """The neighbour that leaves too little room for the open fingers when the object is put down at xy, or None."""
        mine = max(self.objects[self.moving]["width"], self.objects[self.moving]["length"]) / 2000
        for oid, o in self.objects.items():
            if oid not in (self.moving, self.support):
                if np.hypot(o["x"] / 1000 - xy[0], o["y"] / 1000 - xy[1]) < mine + max(o["width"], o["length"]) / 2000 + FINGER_ROOM_M:
                    return oid
        return None

    def why_moved(self):
        """Why the object cannot stand exactly at the marked place, as a sentence."""
        reach, crowd = float(np.hypot(*self.target)), self._crowded(self.target)
        if not REACH_MIN_M <= reach <= REACH_MAX_M:
            return (f"the spot you pointed at is {reach * 1000:.0f} mm from the robot's base, and the arm puts objects down "
                    f"safely only between {REACH_MIN_M * 1000:.0f} and {REACH_MAX_M * 1000:.0f} mm from it")
        if crowd is not None:
            return f"the spot you pointed at is too close to the {self.names.get(crowd, 'object beside it')} for the open fingers"
        return "at the spot you pointed at it would not stand fully inside, or the arm cannot put it down without touching something"

    def _place(self, planner, q, held, g, relation, reference_id, frame, region=None):
        """On a tray, in a box or on an object v2 plans the placement. On the table: the marked place itself, then
        places around it up to 120 mm away, the first that is free, in reach and passes the collision checks."""
        if self.target is None or self.support is not None:
            return self._v2_place(planner, q, held, g, relation, reference_id, frame, region)
        last = "none is free and in reach"
        for distance in np.arange(0.0, 0.13, 0.02):
            for angle in np.radians(np.arange(0, 360, 30 if distance else 360)):
                xy = self.target + distance * np.array([np.cos(angle), np.sin(angle)])
                if REACH_MIN_M <= np.hypot(*xy) <= REACH_MAX_M and not self.t._too_close(xy, g, exclude=None) and self._crowded(xy) is None:
                    try:
                        place, _ = planner.place(q, held, np.array([xy[0], xy[1], 0.0]))
                        return place, [xy[0], xy[1], held.height / 2], held.height
                    except PlanningError as error:
                        last = error
        raise PlanningError(f"no free, reachable, collision-free spot on the table within 120 mm of the marked place ({last})")

    def preview(self, plan, picture, view):
        """The close-up with a white outline where the object would stand: its base, and thinner its top.
        Returns (picture, mm from the marked place)."""
        left, top, _, _, scale = view
        o, (x, y) = self.objects[plan.object_id], plan.target_centre[:2]
        radius, floor = max(o["width"], o["length"]) / 2000, plan.target_top - plan.held.height
        shown = picture.copy()
        for height, width in ((plan.target_top, 1), (floor, 3)):
            ring = np.array([self._pixel((x + radius * np.cos(a), y + radius * np.sin(a), height))
                             for a in np.linspace(0, 2 * np.pi, 36, endpoint=False)])
            ring = ((ring - (left, top)) * scale).astype(np.int32)
            cv2.polylines(shown, [ring], True, (0, 0, 0), width + 4)
            cv2.polylines(shown, [ring], True, (255, 255, 255), width)
        cv2.imwrite(str(self.t.run_dir / "images" / f"{self.view.frame_id}_place_{plan.object_id}.jpg"),
                    cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))
        return shown, round(float(np.hypot(x - self.target[0], y - self.target[1])) * 1000)

    def carry_out(self, plan):
        """Do a plan from aim(): grip, lift, carry, put down, back to HOME. Returns (ok, what happened)."""
        t = self.t
        try:
            if not self._grasp(plan):
                return self._home(False, "the gripper closed but did not hold the object after lifting; it was not moved")
            self._do(plan.place[:1], "carry")
            self._do(plan.place[1:2], "lower")
            self._do(plan.place[2:-1], "release")
            self.outlines.released = True
            t.assume_placed(plan)
            self._do(plan.place[-1:], "retreat")
            t.clear_held()
            x, y = plan.target_centre[:2]
            return self._home(True, f"let go of at x {x * 1000:+.0f}, y {y * 1000:+.0f} mm; where it stands now has not been checked")
        except (PlanningError, RuntimeError) as error:
            return self._home(False, str(error))
        finally:
            self.target = None


def start(*arguments, **options):
    """Load the models and start Isaac Sim exactly as v3 does, with this file's Robot."""
    frozen.Robot = Robot
    return frozen.start(*arguments, **options)
