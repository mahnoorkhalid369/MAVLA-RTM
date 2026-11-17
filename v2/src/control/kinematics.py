"""Forward / inverse kinematics for the VT6L proxy arm, read straight from the URDF.

Pure numpy (no Isaac Sim imports), so it can be used and tested outside the simulator.

    kin = Kinematics()                       # base_link -> tcp
    T = kin.fk(q)                            # 4x4 pose of the TCP in the base frame
    q = kin.ik(position, rotation, q_seed)   # damped least-squares IK
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# Epson VT6L from the manufacturer's manual (see the URDF header); vt6l_proxy.urdf is the old C8L proxy
URDF_PATH = Path(__file__).resolve().parents[2] / "assets" / "vt6l" / "vt6l.urdf"

# Tool pointing straight down with tool0 X = world +Y: the orientation the arm reaches
# from J1 = 0 with J6 ~ 0, so the gripper keeps its mounted orientation relative to the
# J5 wrist camera (2F-85 fingers close along tool0 X -> world Y).
R_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def _rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s, C = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def _rotation_error(R_target: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) taking R to R_target, in the base frame."""
    R_err = R_target @ R.T
    angle = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    v = np.array([R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]])
    return v * angle / (2.0 * np.sin(angle))


class Kinematics:
    def __init__(self, urdf: Path = URDF_PATH, base: str = "base_link", tip: str = "tcp") -> None:
        joints = {}
        for j in ET.parse(urdf).getroot().findall("joint"):
            origin = j.find("origin")
            xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = _rpy_to_matrix(*rpy), xyz
            axis = j.find("axis")
            limit = j.find("limit")
            joints[j.find("child").get("link")] = {
                "name": j.get("name"),
                "type": j.get("type"),
                "parent": j.find("parent").get("link"),
                "origin": T,
                "axis": np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else None,
                "limits": (float(limit.get("lower")), float(limit.get("upper"))) if limit is not None else None,
                "velocity": float(limit.get("velocity", "nan")) if limit is not None else None,  # rad/s
            }
        # Walk tip -> base to get the chain
        chain, link = [], tip
        while link != base:
            chain.append(joints[link])
            link = joints[link]["parent"]
        self.chain = chain[::-1]
        self.active = [j for j in self.chain if j["type"] in ("revolute", "prismatic")]
        self.joint_names = [j["name"] for j in self.active]
        self.lower = np.array([j["limits"][0] for j in self.active])
        self.upper = np.array([j["limits"][1] for j in self.active])
        self.max_velocity = np.array([j["velocity"] for j in self.active])  # rad/s, from the URDF

    def fk(self, q: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        qi = iter(q)
        for j in self.chain:
            T = T @ j["origin"]
            if j["type"] == "revolute":
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j["axis"], next(qi))
                T = T @ R
        return T

    def ik(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        q_seed: np.ndarray,
        iterations: int = 200,
        tol: float = 1e-5,
        damping: float = 1e-3,
    ) -> tuple[np.ndarray, bool]:
        """Damped least-squares IK. Returns (q, converged)."""
        q = np.array(q_seed, dtype=float)
        eps = 1e-6
        for _ in range(iterations):
            T = self.fk(q)
            err = np.concatenate([position - T[:3, 3], _rotation_error(rotation, T[:3, :3])])
            if np.linalg.norm(err) < tol:
                return q, True
            J = np.zeros((6, len(q)))
            for i in range(len(q)):
                dq = q.copy()
                dq[i] += eps
                Td = self.fk(dq)
                J[:3, i] = (Td[:3, 3] - T[:3, 3]) / eps
                J[3:, i] = _rotation_error(Td[:3, :3], T[:3, :3]) / eps
            step = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            q = np.clip(q + step, self.lower, self.upper)
        return q, False

    def ik_best(
        self, position: np.ndarray, rotation: np.ndarray, q_now: np.ndarray, extra_seeds: list[np.ndarray] | None = None
    ) -> tuple[np.ndarray, bool]:
        """IK from several starting poses; returns the valid solution closest to q_now.

        Local IK can stall when started far from the answer; trying the current pose, the
        given seeds and a few spread-out poses makes a reachable target much more likely to solve.
        """
        seeds = [np.asarray(q_now, float)] + list(extra_seeds or [])
        for j1 in (0.0, np.pi / 4, -np.pi / 4):
            seeds.append(np.array([j1, -0.5, 0.5, 0.0, -1.2, 0.0]))
        best, best_cost = None, np.inf
        for seed in seeds:
            q, ok = self.ik(position, rotation, np.clip(seed, self.lower, self.upper))
            if ok:
                cost = float(np.max(np.abs(q - q_now)))
                if cost < best_cost:
                    best, best_cost = q, cost
        return (best, True) if best is not None else (np.asarray(q_now, float), False)


if __name__ == "__main__":
    # Self-test: IK to a few downward-pointing TCP poses, verified by FK
    kin = Kinematics()
    print("joints:", kin.joint_names)
    seed = np.deg2rad([0, 0, 0, 0, -90, 0])
    for target in ([0.0, 0.58, 0.20], [0.0, 0.58, 0.03], [0.30, 0.40, 0.035]):
        q, ok = kin.ik(np.array(target), R_DOWN, seed)
        T = kin.fk(q)
        print(
            f"target {target} -> ok={ok} q(deg)={np.round(np.rad2deg(q), 1).tolist()} "
            f"pos err={np.linalg.norm(T[:3, 3] - target) * 1000:.3f} mm"
        )
