"""Pick / place / home skills: 3-D object positions -> a sequence of planned steps.

No ROS, no model. Every motion is planned before anything moves (control/motion.py), so a plan
that would fail half-way is rejected up front. The ROS node then executes the steps in order.

    skills = SkillPlanner(Kinematics())
    steps, q, held = skills.pick(q_now, top_centre_xyz, yaw_deg=30)   # approach, descend, grip, lift
    more, q = skills.place(q, held, support_xyz)                       # carry, lower, release, retreat

The agent splits a pick in two so it can look again in between: hover() brings the gripper straight
above the object (vertical, turned to the object's yaw), the wrist camera re-measures it, then
grasp() descends from there. The gripper points straight down (90 deg to the floor); only its
rotation about the vertical (yaw) changes, so the fingers close across two parallel faces.

Top-down is the default. tilted_pick() is the fallback for an object top-down cannot reach: the
gripper leans up to SkillParams.max_tilt_deg about its own closing axis, so the pads stay parallel
to the same two vertical faces; it approaches along its tilted axis and lifts straight up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control.collision import GRIPPER_OPEN_WIDTH, CollisionChecker
from control.kinematics import R_DOWN, Kinematics, _axis_angle
from control.motion import PlanningError, Trajectory, joint_move, linear_move
from safety.joint_limits import forbidden_on_path
from control.robot import (
    ARM_HOME,
    ARM_SPEED,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    TCP_SPEED,
    finger_angle_for_gap,
    fingertip_below_tcp,
)

GROUND_Z = 0.0  # table height (m)


@dataclass
class MoveStep:
    trajectory: Trajectory
    label: str


@dataclass
class GripperStep:
    position: float  # finger_joint target (rad)
    label: str
    settle: float = 1.0  # seconds to let the fingers close/open


@dataclass
class Held:
    height: float  # object height (m)
    grip_depth: float  # TCP is this far below the object's top while holding it (m)
    yaw_deg: float = 0.0  # gripper yaw while holding it (deg about vertical)
    name: str | None = None  # obstacle name of the object (collision checks ignore its old position)
    width: float = 0.0  # across the fingers (m)
    length: float = 0.0  # along the fingers' faces (m)
    tilt_deg: float = 0.0  # gripper tilt from vertical about its closing axis while holding it (0 = top-down)
    body_size: tuple | None = None  # full object bounds in gripper axes, not just the local finger width
    body_offset: tuple = (0.0, 0.0)  # body centre relative to the selected grasp centre in gripper XY

    def collision_body(self):
        """(pose in tool0, size) for CollisionChecker: the object moves with the gripper."""
        size = self.body_size or (self.width or 0.01, self.length or self.width or 0.01, self.height)
        T, dimensions = CollisionChecker.held_box_in_tool(size, self.grip_depth, tilt_deg=self.tilt_deg)
        T[:3, 3] += T[:3, :2] @ np.array(self.body_offset)
        return T, dimensions


def grip_rotation(yaw_deg: float, tilt_deg: float = 0.0) -> np.ndarray:
    """Gripper pointing straight down, turned yaw_deg about the vertical. At yaw 0 (R_DOWN) the fingers
    close along world Y; at yaw a they close along (-sin a, cos a). tilt_deg leans it from vertical about
    that (horizontal) closing axis: the pads stay parallel to the same vertical faces."""
    a = np.radians(yaw_deg)
    rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    R = rz @ R_DOWN
    return _axis_angle(R[:, 0], np.radians(tilt_deg)) @ R if tilt_deg else R


@dataclass
class SkillParams:
    approach: float = 0.12  # approach / retreat height above the object or support (m)
    grip_depth: float = 0.025  # hold this far below the top: 2F-85 pads centred on a 60 mm block
    tip_clearance: float = 0.004  # fingertips stay this far above the supporting surface once CLOSED on the object
    min_contact: float = 0.010  # at least this much pad length on the object's sides
    grip_squeeze: float = 0.008  # close to (object width - this): bounded grip force, contact detectable
    # (closing swings the 2F-85 tips up to 14 mm further down: control/robot.py fingertip_below_tcp)
    place_clearance: float = 0.005  # release this far above the support
    max_grip_width: float = 0.080  # 2F-85 opens to 85 mm; keep 5 mm margin
    max_tilt_deg: float = 40.0  # tilted-grasp fallback: at most this far from vertical
    tilt_step_deg: float = 10.0  # tilts tried: step, 2 step, ... up to max_tilt_deg (the smallest that works wins)
    pad_half_width: float = 0.0135  # 2F-85 pad half-width along the faces: a tilted pad's corner dips this x sin(tilt)
    tcp_speed: float = TCP_SPEED  # m/s on straight lines
    place_speed: float = 0.025  # m/s for controlled final lowering
    joint_speed: float = ARM_SPEED  # rad/s on joint moves
    carry_speed_scale: float = 1.0  # evaluation-only speedup; descent/lowering retain their speeds


class SkillPlanner:
    def __init__(self, kin: Kinematics, params: SkillParams | None = None,
                 collision: CollisionChecker | None = None) -> None:
        """collision: if given, every planned motion is checked along its whole path (robot, gripper,
        held object vs floor, obstacles and itself); a colliding plan raises PlanningError."""
        self.kin = kin
        self.p = params or SkillParams()
        self.collision = collision

    def _check(self, step: "MoveStep", opening: float = GRIPPER_OPEN_WIDTH, held: Held | None = None,
               ignore: set | None = None) -> None:
        """Every planned motion, every sample: Epson's forbidden J1/J2/J3 combinations, then (with a collision
        checker) collisions."""
        why = forbidden_on_path(step.trajectory.positions)
        if why:
            raise PlanningError(f"{step.label}: {why}")
        if self.collision is None:
            return
        res = self.collision.check_path(step.trajectory.positions, opening,
                                        held.collision_body() if held is not None else None, ignore)
        if not res.ok:
            raise PlanningError(f"{step.label}: collision, {'; '.join(map(str, res.contacts[:3]))}")

    def _moving_object(self, name: str | None) -> set:
        """Ignore an object's ORIGINAL position (it is in the gripper now) for every robot part."""
        if self.collision is None or name is None:
            return set()
        parts = [p.name for p in self.collision.parts] + ["palm", "pads", "linkages", "held"]
        return {(part, name) for part in parts}

    # -------------------------------------------------------------- building blocks
    def _symmetric_yaws(self, yaw_deg: float, symmetry_deg: float) -> list[float]:
        return [(yaw_deg + k * symmetry_deg + 180.0) % 360.0 - 180.0 for k in range(int(round(360.0 / symmetry_deg)))]

    def _goto_above(self, q: np.ndarray, xyz: np.ndarray, label: str, yaw_deg: float = 0.0,
                    symmetry_deg: float = 180.0, opening: float = GRIPPER_OPEN_WIDTH, held: "Held | None" = None,
                    ignore: set | None = None, offset_xy=None) -> tuple[MoveStep, np.ndarray, float]:
        """Joint move to xyz pointing down. A parallel gripper grips the same at yaw and yaw + 180 deg
        (a square object also at yaw + 90): try those equivalent yaws, keep the one the arm reaches with
        the least joint travel (mostly: the least wrist J6 spin). Returns (step, q, yaw used)."""
        candidates = []
        for yaw in self._symmetric_yaws(yaw_deg, symmetry_deg):
            tcp = np.array(xyz, float).copy()
            if offset_xy is not None:
                tcp[:2] -= grip_rotation(yaw)[:2, :2] @ np.array(offset_xy)
            q_goal, ok = self.kin.ik_best(tcp, grip_rotation(yaw), q, extra_seeds=[ARM_HOME])
            if ok:
                q_goal = self._nearest_turn(q_goal, q)
                cost = float(np.abs(np.asarray(q_goal) - q).sum())
                candidates.append((cost, q_goal, yaw))
        if not candidates:
            raise PlanningError(f"{label}: {np.round(xyz * 1000).astype(int).tolist()} mm is out of reach")
        failure = None
        for _, q_goal, yaw in sorted(candidates, key=lambda item: item[0]):
            try:
                return self._joint_step(q, q_goal, label, opening, held, ignore), q_goal, yaw
            except PlanningError as exc:
                failure = exc
        raise failure

    def _joint_step(self, q: np.ndarray, q_goal: np.ndarray, label: str, opening: float = GRIPPER_OPEN_WIDTH,
                    held: "Held | None" = None, ignore: set | None = None) -> MoveStep:
        """Collision-checked joint move q -> q_goal, or via the (high, central) home pose if the direct move
        sweeps through something."""
        if held is not None and label == "carry above target":
            # A closed-gripper carrying waypoint, distinct from the empty-hand HOME command.
            # Every sample includes the complete payload collision body; no obstacle override.
            speed = np.minimum(self.p.joint_speed * self.p.carry_speed_scale, self.kin.max_velocity)
            a = joint_move(self.kin, q, ARM_HOME, speed)
            b = joint_move(self.kin, ARM_HOME, q_goal, speed)
            traj = Trajectory(np.concatenate([a.times, a.times[-1] + b.times[1:]]),
                              np.vstack([a.positions, b.positions[1:]]), "via carrying waypoint")
            step = MoveStep(traj, label + " (via carrying waypoint)")
            self._check(step, opening, held, ignore)
            return step
        traj = joint_move(self.kin, q, q_goal, self.p.joint_speed)
        step = MoveStep(traj, label)
        try:
            self._check(step, opening, held, ignore)
        except PlanningError as direct:
            # the direct joint move sweeps through something: try going via the (high, central) home pose
            a = joint_move(self.kin, q, ARM_HOME, self.p.joint_speed)
            b = joint_move(self.kin, ARM_HOME, q_goal, self.p.joint_speed)
            via = Trajectory(np.concatenate([a.times, a.times[-1] + b.times[1:]]),
                             np.vstack([a.positions, b.positions[1:]]), f"via home: {a.description} + {b.description}")
            step = MoveStep(via, f"{label} (via home)")
            try:
                self._check(step, opening, held, ignore)
            except PlanningError:
                raise direct from None
        return step

    def _nearest_turn(self, q_goal: np.ndarray, q_now: np.ndarray) -> np.ndarray:
        """J6 (flange) at +-360 deg is the same pose: take the turn closest to where the wrist is now."""
        q_goal = np.asarray(q_goal, float).copy()
        options = q_goal[5] + 2 * np.pi * np.arange(-2, 3)
        options = options[(options >= self.kin.lower[5]) & (options <= self.kin.upper[5])]
        q_goal[5] = options[np.argmin(np.abs(options - q_now[5]))]
        return q_goal

    def _line(self, q: np.ndarray, xyz: np.ndarray, label: str, yaw_deg: float = 0.0,
              tilt_deg: float = 0.0, speed: float | None = None) -> tuple[MoveStep, np.ndarray]:
        try:
            traj = linear_move(self.kin, q, xyz, grip_rotation(yaw_deg, tilt_deg), speed=self.p.tcp_speed if speed is None else speed)
        except PlanningError as exc:
            raise PlanningError(f"{label}: {exc}") from exc
        return MoveStep(traj, label), traj.positions[-1]

    # ----------------------------------------------------------------------- skills
    def check_width(self, width: float) -> None:
        if width > self.p.max_grip_width:
            raise PlanningError(
                f"pick: the object is {width * 1000:.0f} mm across the fingers; the gripper fits at most "
                f"{self.p.max_grip_width * 1000:.0f} mm"
            )

    def hover(self, q: np.ndarray, top_centre: np.ndarray, yaw_deg: float = 0.0, symmetry_deg: float = 180.0,
              label: str = "move above object") -> tuple[list, np.ndarray, float]:
        """Open the gripper and go straight above top_centre, pointing down, turned to yaw_deg (or an
        equivalent yaw, see _goto_above). Returns (steps, q, yaw used)."""
        above = np.array([top_centre[0], top_centre[1], float(top_centre[2]) + self.p.approach])
        step, q, yaw = self._goto_above(q, above, label, yaw_deg, symmetry_deg)
        return [GripperStep(GRIPPER_OPEN, "open gripper", 0.5), step], q, yaw

    def grasp(self, q: np.ndarray, top_centre: np.ndarray, yaw_deg: float = 0.0, target: str | None = None,
              width: float = 0.0, length: float = 0.0, support: str | None = None,
              height: float | None = None, tilt_deg: float = 0.0, body_size=None,
              body_offset=(0., 0.)) -> tuple[list, list, np.ndarray, Held]:
        """From the pre-grasp pose: (descend + close steps, lift steps, q after lift, Held).

        height: the object's own height (m); default: it stands on the floor (height = top z).
        target: the object's obstacle name. The open fingers must clear it on the way down (the
        simplified linkage shapes may touch its top); once closed it moves with the gripper.
        support: what it stands on (None = floor); the object and the pads may touch it as the lift starts.
        tilt_deg: gripper tilt about its closing axis (0 = top-down). Tilted, the fingertips sit higher by
        cos(tilt) and the pads' lower corners dip pad_half_width x sin(tilt): both kept tip_clearance above
        what the object stands on. The lift is straight up either way."""
        top_z = float(top_centre[2])
        grasp_z, height = self._grasp_z(top_z, height, width, tilt_deg)
        above = np.array([top_centre[0], top_centre[1], top_z + self.p.approach])
        grasp = np.array([top_centre[0], top_centre[1], grasp_z])
        down, q = self._line(q, grasp, "descend to grasp" + (f" (tilted {tilt_deg:+.0f} deg)" if tilt_deg else ""),
                             yaw_deg, tilt_deg)
        self._check(down, ignore={("linkages", target)} if target else None)
        # Position + force like a real Robotiq: close to just past the object's width. Driving to fully
        # closed at full force makes the simulated 2F-85 linkage flex (pads swing round small objects and
        # the finger angle stops meaning the pad gap). The fingers stopping short of this target = contact.
        close_target = finger_angle_for_gap(max(width - self.p.grip_squeeze, 0.0)) if width > 0 else GRIPPER_CLOSED
        close = GripperStep(close_target, "close gripper", 1.0)
        up, q = self._line(q, above, "lift", yaw_deg, tilt_deg)
        held = Held(height=height, grip_depth=top_z - grasp_z, yaw_deg=yaw_deg, name=target,
                    width=width, length=length or width, tilt_deg=tilt_deg, body_size=body_size, body_offset=body_offset)
        lifting_off = {("held", "floor")} | ({("held", support), ("pads", support)} if support else set())
        self._check(up, opening=width or GRIPPER_OPEN_WIDTH, held=held, ignore=self._moving_object(target) | lifting_off)
        return [down, close], [up], q, held

    def _grasp_z(self, top_z: float, height: float | None, width: float, tilt_deg: float = 0.0) -> tuple[float, float]:
        """TCP height for the grasp and the object height; refuses a grasp whose pads would barely touch it."""
        height = top_z - GROUND_Z if height is None else float(height)
        if height < 0.01:
            raise PlanningError("pick: the box contains no object above the table")
        closed_gap = width if width > 0 else GRIPPER_OPEN_WIDTH
        base_z = top_z - height  # what it stands on (floor or another object)
        t = np.radians(abs(tilt_deg))
        # plain floats: these end up in the skill's checkpointed state, which cannot store numpy scalars (live
        # 2026-10-26: "Type is not msgpack serializable: numpy.float64" failed a pick that had succeeded)
        tip_drop = float(fingertip_below_tcp(closed_gap) * np.cos(t))  # tips below the TCP, measured vertically
        corner_dip = float(self.p.pad_half_width * np.sin(t))  # a tilted pad's lower corner
        grasp_z = max(top_z - min(self.p.grip_depth, height / 2), base_z + tip_drop + corner_dip + self.p.tip_clearance)
        pad_bottom = grasp_z - tip_drop
        contact = top_z - max(pad_bottom, top_z - height)  # pad length touching the object's sides
        if contact < self.p.min_contact:
            raise PlanningError(f"pick: only {contact * 1000:.0f} mm of the pads would touch the object "
                                f"(need {self.p.min_contact * 1000:.0f} mm)")
        return float(grasp_z), float(height)

    def tilted_pick(self, q: np.ndarray, top_centre: np.ndarray, yaw_deg: float, symmetry_deg: float = 180.0,
                    target: str | None = None, width: float = 0.0, length: float = 0.0, support: str | None = None,
                    height: float | None = None, body_size=None) -> tuple[list, list, list, np.ndarray, Held, list]:
        """Fallback grasp tilted from vertical about the closing axis, for an object a top-down grasp cannot reach.

        Candidates: every equivalent yaw (symmetry_deg) x lean either way x tilt step, 2 step, ... max_tilt_deg; the
        lean direction is not fixed, it follows the object's faces. Each candidate must pass IK and joint limits for
        the pre-grasp pose, the straight descent along the tilted tool axis and the straight vertical lift, and the
        collision checks of every motion (robot, gripper and fingers vs table, obstacles and itself; the held object
        on the lift), with the fingertips tip_clearance above what the object stands on. The smallest tilt that
        passes wins, then the least joint travel.

        Returns (hover steps, [descend, close], [lift], q after lift, Held, candidates); candidates = [{"yaw_deg",
        "tilt_deg", "ok", "reason"}] for the log. Raises PlanningError (with the candidates) when none passes."""
        self.check_width(width)
        top_z = float(top_centre[2])
        seeds = [ARM_HOME] + self._seeds_towards(top_centre)
        candidates, best = [], None
        n = int(round(self.p.max_tilt_deg / self.p.tilt_step_deg))
        for mag in [self.p.tilt_step_deg * i for i in range(1, n + 1)]:
            for yaw in self._symmetric_yaws(yaw_deg, symmetry_deg):
                for tilt in (mag, -mag):
                    rec = {"yaw_deg": round(yaw, 1), "tilt_deg": round(tilt, 1)}
                    try:
                        grasp_z, _ = self._grasp_z(top_z, height, width, tilt)
                        R = grip_rotation(yaw, tilt)
                        grasp_pt = np.array([top_centre[0], top_centre[1], grasp_z])
                        # back off along the tool axis (R[:, 2] points from the TCP into the object)
                        pre = grasp_pt - R[:, 2] * (top_z + self.p.approach - grasp_z)
                        q_pre, ok = self.kin.ik_best(pre, R, q, extra_seeds=seeds)
                        if not ok:
                            raise PlanningError(f"move to tilted pre-grasp: {np.round(pre * 1000).astype(int).tolist()} mm "
                                                "is out of reach")
                        q_pre = self._nearest_turn(q_pre, q)
                        move = self._joint_step(q, q_pre, f"move to tilted pre-grasp ({tilt:+.0f} deg)")
                        approach, lift, q_end, held = self.grasp(q_pre, top_centre, yaw, target, width, length,
                                                                support, height, tilt, body_size=body_size)
                    except PlanningError as exc:
                        candidates.append({**rec, "ok": False, "reason": str(exc)})
                        continue
                    candidates.append({**rec, "ok": True, "reason": ""})
                    cost = float(np.abs(np.asarray(q_pre) - q).sum())
                    if best is None or cost < best[0]:
                        best = (cost, [GripperStep(GRIPPER_OPEN, "open gripper", 0.5), move], approach, lift, q_end, held)
            if best is not None:
                break
        if best is None:
            err = PlanningError(f"no tilted grasp up to {self.p.max_tilt_deg:.0f} deg passed the checks "
                                f"({len(candidates)} candidates)")
            err.candidates = candidates
            raise err
        _, hover, approach, lift, q_end, held = best
        return hover, approach, lift, q_end, held, candidates

    # arm shapes (J2, J3, J5) to start IK from, the arm turned towards the object: reaching out, folded, overhead
    SEED_SHAPES = ((-0.5, 0.5, -1.2), (0.6, 1.2, -1.2), (-1.0, 2.0, 1.2), (-2.4, 2.5, -2.0))

    def _seeds_towards(self, xyz: np.ndarray) -> list[np.ndarray]:
        """IK seeds with J1 turned towards xyz (J1 points the arm along (-sin J1, cos J1)); an object behind the base
        is not found from the forward-facing default seeds. At the J1 limit both ends are tried."""
        lo, hi = self.kin.lower, self.kin.upper
        j1 = float(np.arctan2(-xyz[0], xyz[1]))
        j1s = [j1] if lo[0] <= j1 <= hi[0] else [lo[0], hi[0]]
        return [np.clip(np.array([a, j2, j3, 0.0, j5, 0.0]), lo, hi) for a in j1s for j2, j3, j5 in self.SEED_SHAPES]

    def pick(self, q: np.ndarray, top_centre: np.ndarray, width: float = 0.0,
             yaw_deg: float = 0.0, target: str | None = None, length: float = 0.0,
             support: str | None = None, height: float | None = None) -> tuple[list, np.ndarray, Held]:
        """Grasp an object whose top face is centred at top_centre (world, m), from above, in one go.

        width: the object's size across the fingers (m); wider than max_grip_width is refused.
        yaw_deg: gripper yaw (deg) so the fingers close across the object's faces.
        """
        self.check_width(width)
        steps, q, yaw = self.hover(q, top_centre, yaw_deg)
        close, lift, q, held = self.grasp(q, top_centre, yaw, target, width, length, support, height)
        return steps + close + lift, q, held

    def place(self, q: np.ndarray, held: Held, support: np.ndarray,
              yaw_deg: float | None = None) -> tuple[list, np.ndarray]:
        """Put the held object down with its bottom on `support` (top of an object, or a surface point).

        yaw_deg: gripper yaw while placing (default: keep the yaw it was picked with)."""
        yaw = held.yaw_deg if yaw_deg is None else yaw_deg
        release_z = float(support[2]) + held.height - held.grip_depth + self.p.place_clearance
        above_centre = np.array([*support[:2], release_z + self.p.approach])
        steps: list = []
        moving = self._moving_object(held.name)
        closed = held.width or GRIPPER_OPEN_WIDTH
        step, q, yaw = self._goto_above(q, above_centre, "carry above target", yaw, 180.0, closed, held, moving,
                                      offset_xy=held.body_offset)
        xy = np.array(support[:2]) - grip_rotation(yaw)[:2, :2] @ np.array(held.body_offset)
        above = np.array([*xy, release_z + self.p.approach])
        release = np.array([*xy, release_z])
        steps.append(step)
        step, q = self._line(q, release, "lower", yaw, speed=self.p.place_speed)
        self._check(step, closed, held, moving)
        steps.append(step)
        steps.append(GripperStep(GRIPPER_OPEN, "open gripper", 0.8))
        step, q = self._line(q, above, "retreat", yaw)
        self._check(step, ignore=moving)  # the released object now stands between the open pads
        steps.append(step)
        return steps, q

    def home(self, q: np.ndarray, pose: np.ndarray = ARM_HOME) -> tuple[list, np.ndarray]:
        step = MoveStep(joint_move(self.kin, q, pose, self.p.joint_speed), "return home")
        why = forbidden_on_path(step.trajectory.positions)
        if why:
            raise PlanningError(f"return home: {why}")
        self._check(step)
        return [step], np.asarray(pose, float)


if __name__ == "__main__":
    kin = Kinematics()
    skills = SkillPlanner(kin)
    q = ARM_HOME.copy()
    pick_steps, q, held = skills.pick(q, np.array([0.0, 0.58, 0.06]), yaw_deg=30.0)  # red block top, turned 30 deg
    place_steps, q = skills.place(q, held, np.array([0.15, 0.50, 0.06]))  # on top of the green block
    home_steps, q = skills.home(q)
    total = 0.0
    for s in pick_steps + place_steps + home_steps:
        dur = s.trajectory.duration if isinstance(s, MoveStep) else s.settle
        total += dur
        print(f"  {s.label:22s} {dur:5.1f} s" + (f"  ({s.trajectory.description})" if isinstance(s, MoveStep) else ""))
    print(f"held object: height {held.height * 1000:.0f} mm, gripped {held.grip_depth * 1000:.0f} mm below its top; total {total:.1f} s")
    try:
        skills.pick(ARM_HOME, np.array([0.0, 1.4, 0.06]))
    except PlanningError as e:
        print("rejected:", e)
