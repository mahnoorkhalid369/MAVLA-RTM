"""Turn text instructions into motion goals for the VT6L arm and the 2F-85 gripper.

Plain Python (no ROS, no Isaac Sim): ros_nodes/command_interpreter.py wraps it in a ROS 2 node.
Typed text is the input for now; a speech-to-text node can feed the same /instruction topic later.

The parser only decides WHAT is wanted:
  * Command.targets    joint-space goal {joint name: rad} (arm joints and/or gripper)
  * Command.cartesian  TCP goal (position + orientation, straight line or joint move)
HOW to get there (synchronised joint motion, straight-line IK) is control/motion.py.

Positions are in millimetres in the robot base frame (World): +X right, +Y forward
(the arm's home reach direction), +Z up. Run `python3 -m control.instructions` to self-test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from control.kinematics import R_DOWN, Kinematics
from control.motion import PlanningError, Trajectory, joint_move, linear_move
from control.robot import (
    ARM_HOME,
    ARM_JOINTS,
    ARM_SPEED,
    GRIPPER_CLOSED,
    GRIPPER_JOINT,
    GRIPPER_OPEN,
    OBSERVE_POSE,
    TCP_SPEED,
)

LOOK_STEP_DEG = 15.0
NUM = r"([+-]?\d+(?:\.\d+)?)"
UNITS = {"mm": 0.001, "cm": 0.01, "m": 1.0}
DIRECTIONS = {  # base-frame unit vectors
    "up": (0, 0, 1), "down": (0, 0, -1), "left": (-1, 0, 0), "right": (1, 0, 0),
    "forward": (0, 1, 0), "forwards": (0, 1, 0), "back": (0, -1, 0), "backward": (0, -1, 0), "backwards": (0, -1, 0),
}
HELP = """Joint commands (degrees, J1-J6):
  j5 -60               move J5 to -60          j1 30 j5 -45   several joints at once
  j1 by 20             move J1 by +20          pose 0 -20 20 0 -90 0   all six joints
  look left|right|up|down [deg]   turn the camera view (J1 / J5), default 15
Gripper (TCP) commands, millimetres in the robot base frame (+X right, +Y forward, +Z up):
  move up|down|left|right|forward|back 100 [mm|cm|m]   straight line, keeps the tool orientation
  move to 0 580 200    straight line to a point       goto 0 580 200   joint move to a point
                       (add 'down' to also turn the tool straight down: goto 0 500 400 down)
  point down           turn the tool to point straight down (stays in place)
Gripper:  open | close | gripper 50%
Other:    home | observe | turn around | stop | status | speed 45 (joint deg/s) | tcp speed 80 (mm/s)
All joints in a move start and stop together."""


@dataclass
class CartesianGoal:
    position: np.ndarray  # m, base frame
    rotation: np.ndarray | None  # 3x3, None = keep the current orientation
    linear: bool  # True: straight line; False: synchronised joint move


@dataclass
class Command:
    targets: dict[str, float] = field(default_factory=dict)  # joint name -> absolute target (rad)
    cartesian: CartesianGoal | None = None
    reply: str = ""
    speed: float = ARM_SPEED  # joint speed limit for this move (rad/s)
    tcp_speed: float = TCP_SPEED  # TCP speed for straight lines (m/s)


class InstructionParser:
    def __init__(self) -> None:
        self.kin = Kinematics()
        self.limits = {n: (lo, hi) for n, lo, hi in zip(self.kin.joint_names, self.kin.lower, self.kin.upper)}
        self.limits[GRIPPER_JOINT] = (GRIPPER_OPEN, GRIPPER_CLOSED)
        self.speed = ARM_SPEED
        self.tcp_speed = TCP_SPEED

    # ---------------------------------------------------------------- helpers
    def _set(self, cmd: Command, joint: str, value: float, notes: list[str]) -> None:
        lo, hi = self.limits[joint]
        clipped = float(np.clip(value, lo, hi))
        if abs(clipped - value) > 1e-9:
            notes.append(f"{_label(joint)} limited to {np.rad2deg(lo):.0f}..{np.rad2deg(hi):.0f} deg")
        cmd.targets[joint] = clipped

    def _arm(self, current: dict[str, float]) -> np.ndarray | None:
        if not all(j in current for j in ARM_JOINTS):
            return None
        return np.array([current[j] for j in ARM_JOINTS])

    def tcp(self, current: dict[str, float]) -> np.ndarray | None:
        """4x4 TCP pose from the current joint angles (forward kinematics)."""
        q = self._arm(current)
        return None if q is None else self.kin.fk(q)

    def _status(self, current: dict[str, float]) -> str:
        arm = "  ".join(f"{_label(j)} {np.rad2deg(current.get(j, np.nan)):6.1f}" for j in ARM_JOINTS)
        grip = current.get(GRIPPER_JOINT)
        closed = "unknown" if grip is None else f"{100 * grip / GRIPPER_CLOSED:.0f}% closed"
        T = self.tcp(current)
        where = "" if T is None else (
            f"\nTCP at ({', '.join(f'{v * 1000:.0f}' for v in T[:3, 3])}) mm, tool axis "
            f"({', '.join(f'{v:+.2f}' for v in T[:3, 2])})"
        )
        return f"{arm}  | gripper {closed}{where}"

    # ------------------------------------------------------------------ parse
    def parse(self, text: str, current: dict[str, float]) -> Command:
        raw = text.strip()
        t = raw.lower().replace("°", " ").replace(",", " ")
        t = re.sub(r"\b(degrees?|deg)\b", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        cmd, notes = Command(speed=self.speed, tcp_speed=self.tcp_speed), []
        not_running = "joint positions not received yet; is the simulation running?"

        if not t:
            return cmd
        if t in ("help", "?", "h"):
            cmd.reply = HELP
            return cmd
        if t in ("status", "where", "joints", "state"):
            cmd.reply = self._status(current)
            return cmd
        if t in ("stop", "hold", "freeze"):
            for j in ARM_JOINTS:
                if j in current:
                    cmd.targets[j] = current[j]
            cmd.reply = "holding the current pose"
            return cmd
        if t in ("observe", "look at the table", "observe pose"):
            for j, v in zip(ARM_JOINTS, OBSERVE_POSE):
                self._set(cmd, j, v, notes)
            cmd.reply = "moving to the observe pose (camera sees most of the table)"
            return self._finish(cmd, notes)
        if t in ("home", "go home", "home pose"):
            for j, v in zip(ARM_JOINTS, ARM_HOME):
                self._set(cmd, j, v, notes)
            cmd.reply = "moving to home pose"
            return self._finish(cmd, notes)

        if m := re.fullmatch(r"(?:set )?(tcp|linear|line) speed " + NUM + r"(?: ?mm/s)?", t):
            self.tcp_speed = float(np.clip(float(m.group(2)), 5.0, 250.0)) / 1000.0
            cmd.reply = f"TCP speed set to {self.tcp_speed * 1000:.0f} mm/s"
            return cmd
        if m := re.fullmatch(r"(?:set )?speed " + NUM, t):
            self.speed = float(np.deg2rad(np.clip(float(m.group(1)), 5.0, 120.0)))
            cmd.reply = f"joint speed set to {np.rad2deg(self.speed):.0f} deg/s"
            return cmd

        if re.fullmatch(r"(?:open|release)(?: the)?(?: gripper)?|gripper open", t):
            self._set(cmd, GRIPPER_JOINT, GRIPPER_OPEN, notes)
            cmd.reply = "opening gripper"
            return self._finish(cmd, notes)
        if re.fullmatch(r"(?:close|grip|grab|grasp)(?: the)?(?: gripper)?|gripper close", t):
            self._set(cmd, GRIPPER_JOINT, GRIPPER_CLOSED, notes)
            cmd.reply = "closing gripper"
            return self._finish(cmd, notes)
        if m := re.fullmatch(r"gripper " + NUM + r" ?%", t):
            frac = np.clip(float(m.group(1)) / 100.0, 0.0, 1.0)
            self._set(cmd, GRIPPER_JOINT, GRIPPER_OPEN + frac * (GRIPPER_CLOSED - GRIPPER_OPEN), notes)
            cmd.reply = f"gripper to {100 * frac:.0f}% closed"
            return self._finish(cmd, notes)

        # ---- TCP (Cartesian) goals: need the current pose from forward kinematics
        if m := re.fullmatch(r"(?:move )?(" + "|".join(DIRECTIONS) + r") (?:by )?" + NUM + r" ?(mm|cm|m)?", t):
            T = self.tcp(current)
            if T is None:
                cmd.reply = not_running
                return cmd
            step = float(m.group(2)) * UNITS[m.group(3) or "mm"]
            goal = T[:3, 3] + step * np.array(DIRECTIONS[m.group(1)], float)
            cmd.cartesian = CartesianGoal(goal, None, linear=True)
            cmd.reply = f"moving {m.group(1)} {step * 1000:.0f} mm in a straight line"
            return cmd
        if m := re.fullmatch(r"(move to|go to|goto) " + NUM + " " + NUM + " " + NUM + r" ?(mm|cm|m)?((?: (?:linear|line|joint|down))*)", t):
            if self.tcp(current) is None:
                cmd.reply = not_running
                return cmd
            scale = UNITS[m.group(5) or "mm"]
            goal = np.array([float(m.group(i)) for i in (2, 3, 4)]) * scale
            options = m.group(6).split()
            linear = ("linear" in options or "line" in options) or (m.group(1) == "move to" and "joint" not in options)
            rotation = R_DOWN if "down" in options else None
            cmd.cartesian = CartesianGoal(goal, rotation, linear=linear)
            kind = "straight line" if linear else "joint move"
            tool = ", tool pointing down" if rotation is not None else ""
            cmd.reply = f"{kind} to ({', '.join(f'{v * 1000:.0f}' for v in goal)}) mm{tool}"
            return cmd
        if re.fullmatch(r"(?:point|tool|turn tool|face) down", t):
            T = self.tcp(current)
            if T is None:
                cmd.reply = not_running
                return cmd
            cmd.cartesian = CartesianGoal(T[:3, 3].copy(), R_DOWN, linear=True)
            cmd.reply = "turning the tool to point straight down"
            return cmd

        if t in ("turn around", "look behind", "look behind you", "look back", "turn back"):
            if "j1_joint" not in current:
                cmd.reply = not_running
                return cmd
            behind = current["j1_joint"] + (np.pi if current["j1_joint"] <= 0 else -np.pi)
            self._set(cmd, "j1_joint", behind, notes)
            cmd.reply = f"turning around: J1 -> {np.rad2deg(cmd.targets['j1_joint']):.0f} deg"
            return self._finish(cmd, notes)
        if m := re.fullmatch(r"look (left|right|up|down)(?: by)?(?: " + NUM + ")?", t):
            direction, step = m.group(1), np.deg2rad(float(m.group(2) or LOOK_STEP_DEG))
            # J1 positive turns the arm to its left; J5 towards 0 raises the view from straight down
            joint, sign = {"left": ("j1_joint", 1), "right": ("j1_joint", -1), "up": ("j5_joint", 1), "down": ("j5_joint", -1)}[direction]
            if joint not in current:
                cmd.reply = not_running
                return cmd
            self._set(cmd, joint, current[joint] + sign * step, notes)
            cmd.reply = f"looking {direction}: {_label(joint)} -> {np.rad2deg(cmd.targets[joint]):.1f} deg"
            return self._finish(cmd, notes)

        if m := re.fullmatch(r"pose" + (" " + NUM) * 6, t):
            for j, v in zip(ARM_JOINTS, m.groups()):
                self._set(cmd, j, np.deg2rad(float(v)), notes)
            cmd.reply = "moving to pose " + " ".join(f"{np.rad2deg(cmd.targets[j]):.1f}" for j in ARM_JOINTS)
            return self._finish(cmd, notes)

        # Joint moves: "j5 -60", "move joint 5 to -60", "rotate j1 by 20", "j1 30 j5 -45"
        joint_re = r"\b(?:j|joint ?)([1-6])\b (?:to |= ?)?(by )?" + NUM
        moves = re.findall(joint_re, t)
        leftover = re.sub(joint_re, "", t)
        leftover = re.sub(r"\b(move|set|rotate|turn|and|please|then)\b", "", leftover).strip()
        if moves and not leftover:
            parts = []
            for idx, by, value in moves:
                joint = ARM_JOINTS[int(idx) - 1]
                if by:
                    if joint not in current:
                        cmd.reply = not_running
                        return cmd
                    target = current[joint] + np.deg2rad(float(value))
                else:
                    target = np.deg2rad(float(value))
                self._set(cmd, joint, target, notes)
                parts.append(f"{_label(joint)} -> {np.rad2deg(cmd.targets[joint]):.1f} deg")
            cmd.reply = ", ".join(parts)
            return self._finish(cmd, notes)

        cmd.reply = f"did not understand '{raw}'. Type 'help' for the list of commands."
        return cmd

    def _finish(self, cmd: Command, notes: list[str]) -> Command:
        if notes:
            cmd.reply += " (" + "; ".join(notes) + ")"
        return cmd


def plan_command(kin: Kinematics, cmd: Command, current: dict[str, float]) -> Trajectory | None:
    """Arm motion for a parsed command: synchronised joint move, straight line, or IK + joint move.

    Returns None when the command has no arm motion (e.g. only the gripper, or a status question).
    """
    arm_targets = {j: v for j, v in cmd.targets.items() if j in ARM_JOINTS}
    if not arm_targets and cmd.cartesian is None:
        return None
    if not all(j in current for j in ARM_JOINTS):
        raise PlanningError("joint positions not received yet; is the simulation running?")
    q_now = np.array([current[j] for j in ARM_JOINTS])
    if cmd.cartesian is None:  # joint-space goal: synchronise all joints
        q_goal = np.array([arm_targets.get(j, current[j]) for j in ARM_JOINTS])
        return joint_move(kin, q_now, q_goal, cmd.speed)
    goal = cmd.cartesian
    if goal.linear:
        return linear_move(kin, q_now, goal.position, goal.rotation, speed=cmd.tcp_speed)
    rotation = kin.fk(q_now)[:3, :3] if goal.rotation is None else goal.rotation
    q_goal, ok = kin.ik_best(goal.position, rotation, q_now, extra_seeds=[ARM_HOME])
    if not ok:
        hint = "" if goal.rotation is not None else " (try adding 'down' to point the tool straight down)"
        raise PlanningError(f"that point is out of reach with this tool orientation{hint}")
    return joint_move(kin, q_now, q_goal, cmd.speed)


def _label(joint: str) -> str:
    return "gripper" if joint == GRIPPER_JOINT else joint.split("_")[0].upper()


if __name__ == "__main__":
    parser = InstructionParser()
    pose = dict(zip(ARM_JOINTS, ARM_HOME)) | {GRIPPER_JOINT: 0.0}
    for text in ["status", "j5 -60", "rotate j1 by 20", "move joint 2 to -200", "look left", "close",
                 "gripper 50%", "home", "move down 100", "move left 5 cm", "move to 300 400 200",
                 "goto 300 400 200", "goto 0 500 400 down", "move to 0 500 300 mm joint down", "point down", "tcp speed 120", "speed 60", "dance"]:
        cmd = parser.parse(text, pose)
        targets = {k: round(float(np.rad2deg(v)), 1) for k, v in cmd.targets.items()}
        goal = "" if cmd.cartesian is None else f" goal {np.round(cmd.cartesian.position * 1000, 1).tolist()} mm linear={cmd.cartesian.linear}"
        print(f"> {text}\n  {cmd.reply}  {targets if targets else ''}{goal}")
