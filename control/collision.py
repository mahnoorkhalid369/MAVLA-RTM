"""Collision checking for planned motions (no simulator, numpy only).

What is checked, along a whole path (conservative advancement: between checked configurations no
robot point moves further than the clearance there, so nothing is skipped; see check_path):
  robot links (URDF <collision> boxes / cylinders), the wrist camera, the Robotiq 2F-85 (palm, pads,
  linkages, opening-dependent) and a held object
    vs the floor (z = 0), vs obstacles (oriented boxes: test objects, camera stand, microphone),
    vs each other (self-collision).

Representation (like cuRobo): the robot is covered by spheres, obstacles are exact oriented boxes.
Sphere tests vectorise well, so a whole trajectory is checked in a fraction of a second. Spheres
slightly under-cover box corners; the margins below compensate. The guarantee holds for this model:
how well spheres + simplified link shapes match the real robot is a separate (CAD) question.

Self-collision exclusions: parts joined by a joint (parent/child links, the gripper on the flange,
the camera on the J5 housing), upper arm vs forearm (only J3 moves them relative to each other and
Epson's J3 limits are the fold limits), and part pairs that already overlap in the zero pose
(intentional overlaps of simplified shapes). `CollisionChecker.excluded_pairs` lists them for review.

    checker = CollisionChecker(obstacles=[Box("red_cube", centre, np.eye(3), half)])
    hit = checker.check(q, opening=0.085)                       # [] or [Contact(...)]
    hit = checker.check_path(trajectory.positions, opening=0.085, ignore={("pads", "red_cube")})
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from control.kinematics import URDF_PATH, _axis_angle, _rpy_to_matrix
from control.robot import ARM_JOINTS, FINGERTIP_BELOW_TCP_OPEN, fingertip_below_tcp

GRIPPER_OPEN_WIDTH = 0.087  # m between the pads when open (2F-85 asset: inner faces at +-43.5 mm)
GRIPPER_MOUNT_YAW_DEG = 90.0  # sim/scene.py: 2F-85 turned 90 deg about the flange axis
MAX_STEP_DEG = 1.0  # path checks start from samples this close in every joint, then subdivide as needed
PATH_RESOLUTION = 0.001  # m: conservative path checks guarantee (margin - this) clearance between samples

# 2F-85 geometry in its base frame (z from the flange towards the fingertips, fingers close along y),
# measured from the Isaac Sim asset's part bounding boxes with the gripper open:
# palm x+-37.5 y+-42.4 z -3.4..90; inner finger (pad) x+-13.5, y 43.5..74.8, z 91.4..148.4;
# outer knuckle y 23..68 z 44..64.5; outer finger y 47.7..77.2 z 44..103.4; inner knuckle runs
# diagonally from (y 6.7, z 55.4) to (y 56.9, z 109.9). Linkages are modelled at the open pose.
PAD_THICKNESS = 0.0313
_PALM = (np.array([0.0, 0.0, 0.0433]), np.array([0.0375, 0.0424, 0.0467]))  # centre, half size


@dataclass
class Box:
    """Oriented box obstacle: centre (world, m), rotation (columns = box axes in world), half sizes."""
    name: str
    centre: np.ndarray
    R: np.ndarray
    half: np.ndarray


@dataclass
class Contact:
    a: str  # robot part
    b: str  # robot part, obstacle name or "floor"
    depth: float  # m: how far inside the margin (positive = penetration beyond the margin)

    def __str__(self) -> str:
        return f"{self.a} with {self.b} ({self.depth * 1000:.1f} mm)"


@dataclass
class Margins:
    """Clearance required (m). Fingertips may come close to the floor during a grasp."""
    obstacle: float = 0.010
    gripper_obstacle: float = 0.003
    floor: float = 0.020
    pads_floor: float = 0.002
    self_: float = 0.005


@dataclass
class _Part:
    name: str  # group name used in reports and ignore pairs ("j3_link", "pads", "held", ...)
    link: str  # frame it moves with
    centres: np.ndarray  # (n, 3) in the link frame
    radii: np.ndarray  # (n,)


@dataclass
class Path_Result:
    ok: bool
    index: int = -1  # first colliding sample
    contacts: list[Contact] = field(default_factory=list)


def box_spheres(T: np.ndarray, size) -> tuple[np.ndarray, np.ndarray]:
    """Spheres covering a box (size = full edge lengths) posed by T: radius = half the thinnest edge,
    centres on a grid with spacing <= radius."""
    half = np.asarray(size, float) / 2
    r = float(half.min())
    axes = [np.linspace(-h + r, h - r, max(1, int(np.ceil((2 * h - 2 * r) / r)) + 1)) if h > r else np.zeros(1)
            for h in half]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid @ T[:3, :3].T + T[:3, 3], np.full(len(grid), r)


def cylinder_spheres(T: np.ndarray, radius: float, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Spheres covering a cylinder along its local z."""
    if length >= 2 * radius:  # long: a row of radius-r spheres
        z = np.linspace(-length / 2 + radius, length / 2 - radius, max(1, int(np.ceil((length - 2 * radius) / radius)) + 1))
        local, r = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1), np.full(len(z), radius)
    else:  # flat disc: rings of thickness-sized spheres
        rs = length / 2
        pts = [np.zeros(3)]
        for ring in np.arange(rs, radius - rs + 1e-9, rs):
            n = max(6, int(np.ceil(2 * np.pi * ring / rs)))
            a = np.linspace(0, 2 * np.pi, n, endpoint=False)
            pts += list(np.stack([ring * np.cos(a), ring * np.sin(a), np.zeros(n)], axis=1))
        local, r = np.array(pts), np.full(len(pts), rs)
    return local @ T[:3, :3].T + T[:3, 3], r


def _rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return T


def _box_T(centre) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = centre
    return T


class CollisionChecker:
    def __init__(self, obstacles: list[Box] | None = None, urdf: Path = URDF_PATH,
                 margins: Margins | None = None) -> None:
        self.margins = margins or Margins()
        self.obstacles: list[Box] = list(obstacles or [])
        self._parse(urdf)
        self.excluded_pairs = self._exclusions()

    # ------------------------------------------------------------------ model
    def _parse(self, urdf: Path) -> None:
        root = ET.parse(urdf).getroot()
        self.joints = {}  # child link -> (parent, origin T, axis, name, type)
        for j in root.findall("joint"):
            o = j.find("origin")
            T = np.eye(4)
            if o is not None:
                T[:3, :3] = _rpy_to_matrix(*[float(v) for v in o.get("rpy", "0 0 0").split()])
                T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            axis = j.find("axis")
            self.joints[j.find("child").get("link")] = (
                j.find("parent").get("link"), T,
                np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else None,
                j.get("name"), j.get("type"))
        self.links = ["base_link"]  # parents before children
        while len(self.links) < len(self.joints) + 1:
            for child, (parent, *_rest) in self.joints.items():
                if parent in self.links and child not in self.links:
                    self.links.append(child)
        self.parts: list[_Part] = []
        for link in root.findall("link"):
            centres, radii = [], []
            for c in link.findall("collision"):
                o = c.find("origin")
                T = np.eye(4)
                if o is not None:
                    T[:3, :3] = _rpy_to_matrix(*[float(v) for v in o.get("rpy", "0 0 0").split()])
                    T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
                g = c.find("geometry")
                if g.find("box") is not None:
                    p, r = box_spheres(T, [float(v) for v in g.find("box").get("size").split()])
                elif g.find("cylinder") is not None:
                    cyl = g.find("cylinder")
                    p, r = cylinder_spheres(T, float(cyl.get("radius")), float(cyl.get("length")))
                else:
                    continue  # meshes are not supported here
                centres.append(p)
                radii.append(r)
            if centres:
                self.parts.append(_Part(link.get("name"), link.get("name"), np.vstack(centres), np.concatenate(radii)))
        self.parent_of = {child: parent for child, (parent, *_rest) in self.joints.items()}

    def _gripper_parts(self, opening: float) -> list[_Part]:
        """2F-85 spheres in the tool0 frame for a given pad opening (m)."""
        mount = _rot_z(GRIPPER_MOUNT_YAW_DEG)
        shift = GRIPPER_OPEN_WIDTH / 2 - opening / 2  # how far each pad has moved in from open
        palm = box_spheres(mount @ _box_T(_PALM[0]), 2 * _PALM[1])
        pads, links = [], []
        for side in (1.0, -1.0):
            y_pad = side * (opening / 2 + PAD_THICKNESS / 2)
            pads.append(box_spheres(mount @ _box_T([0.0, y_pad, 0.1199 + fingertip_below_tcp(opening) - FINGERTIP_BELOW_TCP_OPEN]),
                                    [0.027, PAD_THICKNESS, 0.057]))  # closing swings the pads down (calibrated)
            links.append(box_spheres(mount @ _box_T([0.0, side * 0.0455, 0.0543]), [0.0164, 0.045, 0.0205]))  # outer knuckle
            links.append(box_spheres(mount @ _box_T([0.0, side * (0.0625 - shift), 0.0737]), [0.0272, 0.0295, 0.0593]))  # outer finger
            for t in np.linspace(0.0, 1.0, 6):  # inner knuckle, diagonal
                y, z = 0.0067 + t * (0.0569 - 0.0067 - shift), 0.0554 + t * (0.1099 - 0.0554)
                for x in (-0.009, 0.0, 0.009):
                    links.append((mount[:3, :3] @ [x, side * y, z] + mount[:3, 3], np.array([0.0085])))
        cat = lambda items: (np.vstack([np.atleast_2d(p) for p, _ in items]), np.concatenate([r for _, r in items]))  # noqa: E731
        return [_Part("palm", "tool0", *palm), _Part("pads", "tool0", *cat(pads)), _Part("linkages", "tool0", *cat(links))]

    @staticmethod
    def held_box_in_tool(size, grip_depth: float, tcp_offset: float = 0.1337, yaw_offset_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Pose (4x4 in tool0) and full size of an object held by the pads: its top is grip_depth above the TCP."""
        T = _rot_z(yaw_offset_deg)
        T[2, 3] = tcp_offset - grip_depth + size[2] / 2
        return T, np.asarray(size, float)

    def link_poses(self, q: np.ndarray) -> dict[str, np.ndarray]:
        qmap = dict(zip(ARM_JOINTS, q))
        poses = {"base_link": np.eye(4)}
        for link in self.links[1:]:
            parent, T, axis, name, jtype = self.joints[link]
            M = poses[parent] @ T
            if jtype in ("revolute", "continuous"):
                R = np.eye(4)
                R[:3, :3] = _axis_angle(axis, qmap[name])
                M = M @ R
            poses[link] = M
        return poses

    def spheres(self, q, opening: float = GRIPPER_OPEN_WIDTH, held=None):
        """All robot spheres in the world: (centres (N,3), radii (N,), part index (N,), part names)."""
        poses = self.link_poses(np.asarray(q, float))
        parts = self.parts + self._gripper_parts(opening)
        if held is not None:
            T_held, size = held
            parts = parts + [_Part("held", "tool0", *box_spheres(T_held, size))]
        cs, rs, idx = [], [], []
        for i, p in enumerate(parts):
            T = poses[p.link]
            cs.append(p.centres @ T[:3, :3].T + T[:3, 3])
            rs.append(p.radii)
            idx.append(np.full(len(p.radii), i))
        return np.vstack(cs), np.concatenate(rs), np.concatenate(idx), [p.name for p in parts]

    # -------------------------------------------------------------- exclusions
    def _attached_to(self, name: str) -> str:
        return {"palm": "j6_link", "pads": "j6_link", "linkages": "j6_link", "held": "j6_link"}.get(name, name)

    def _exclusions(self) -> set[frozenset]:
        names = [p.name for p in self.parts] + ["palm", "pads", "linkages", "held"]
        excluded = {frozenset(("palm", "pads")), frozenset(("palm", "linkages")), frozenset(("pads", "linkages")),
                    frozenset(("held", "pads")), frozenset(("held", "linkages")), frozenset(("held", "palm"))}
        for a in names:  # joined by a joint (parent / child link, or attached to the same link)
            for b in names:
                la, lb = self._attached_to(a), self._attached_to(b)
                if a != b and (la == lb or self.parent_of.get(la) == lb or self.parent_of.get(lb) == la):
                    excluded.add(frozenset((a, b)))
        # Upper arm vs forearm: their relative pose depends on J3 alone, and Epson's J3 range (-51..+190)
        # is the mechanism's own fold limit. Within it, a contact between these two simplified boxes is an
        # artefact of the shapes (e.g. the observe pose at J3 -50), not a collision.
        excluded.add(frozenset(("j2_link", "j4_link")))
        # part pairs that overlap in the zero pose: intentional overlaps of simplified shapes
        self._self_excluded = excluded
        for c in self._self_contacts(np.zeros(6), GRIPPER_OPEN_WIDTH, None, margin=0.0):
            excluded.add(frozenset((c.a, c.b)))
        return excluded

    # ------------------------------------------------------------------ checks
    def _self_contacts(self, q, opening, held, margin) -> list[Contact]:
        c, r, idx, names = self.spheres(q, opening, held)
        return self._self_check(c, r, idx, names, margin, set())[0]

    def _self_check(self, c, r, idx, names, margin, ignore) -> tuple[list[Contact], float]:
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1) - r[:, None] - r[None, :]
        out, clear = [], np.inf
        n = len(names)
        for i in range(n):
            for j in range(i + 1, n):
                pair = frozenset((names[i], names[j]))
                if pair in self._self_excluded or (names[i], names[j]) in ignore or (names[j], names[i]) in ignore:
                    continue
                block = d[np.ix_(idx == i, idx == j)]
                if block.size:
                    clear = min(clear, float(block.min() - margin))
                    if block.min() < margin:
                        out.append(Contact(names[i], names[j], float(margin - block.min())))
        return out, clear

    def evaluate(self, q, opening: float = GRIPPER_OPEN_WIDTH, held=None,
                 ignore: set | None = None) -> tuple[list[Contact], float, float]:
        """(contacts, environment clearance, self clearance) at one configuration. Clearances are the
        smallest (distance - required margin) in metres: negative = a contact."""
        ignore = ignore or set()
        m = self.margins
        c, r, idx, names = self.spheres(q, opening, held)
        out: list[Contact] = []
        env_clear = np.inf
        gripper = {"palm", "pads", "linkages", "held"}
        # floor (the base stands on it)
        z = c[:, 2] - r
        for i, name in enumerate(names):
            if name == "base_link" or (name, "floor") in ignore:
                continue
            margin = m.pads_floor if name in ("pads", "held") else m.floor
            zi = z[idx == i]
            if zi.size:
                env_clear = min(env_clear, float(zi.min() - margin))
                if zi.min() < margin:
                    out.append(Contact(name, "floor", float(margin - zi.min())))
        # obstacles
        for box in self.obstacles:
            local = (c - box.centre) @ box.R
            outside = np.linalg.norm(local - np.clip(local, -box.half, box.half), axis=1)
            inside = np.all(np.abs(local) <= box.half, axis=1)
            depth_in = np.min(box.half - np.abs(local), axis=1)
            dist = np.where(inside, -depth_in, outside) - r
            for i, name in enumerate(names):
                if (name, box.name) in ignore:
                    continue
                di = dist[idx == i]
                margin = m.gripper_obstacle if name in gripper else m.obstacle
                if di.size:
                    env_clear = min(env_clear, float(di.min() - margin))
                    if di.min() < margin:
                        out.append(Contact(name, box.name, float(margin - di.min())))
        self_hits, self_clear = self._self_check(c, r, idx, names, m.self_, ignore)
        return out + self_hits, env_clear, self_clear

    def check(self, q, opening: float = GRIPPER_OPEN_WIDTH, held=None, ignore: set | None = None) -> list[Contact]:
        """Contacts at one configuration. ignore: {(part, obstacle)} pairs allowed to touch,
        e.g. ("pads", "red_cube") while grasping it or ("held", "floor") while lifting it."""
        return self.evaluate(q, opening, held, ignore)[0]

    # ------------------------------------------------------------ swept paths
    def lever_arms(self, held=None) -> np.ndarray:
        """Upper bound (m), per arm joint, on the distance from that joint's axis to any point of the
        parts it moves (sum of the link offsets along the chain + the sphere's offset and radius;
        triangle inequality, so it holds in every configuration)."""
        poses0 = {link: self.joints[link] for link in self.links[1:]}
        parts = self.parts + self._gripper_parts(GRIPPER_OPEN_WIDTH)
        if held is not None:
            parts = parts + [_Part("held", "tool0", *box_spheres(*held))]
        arms = np.zeros(len(ARM_JOINTS))
        for k, jname in enumerate(ARM_JOINTS):
            joint_child = next(link for link, (_, _, _, name, _) in poses0.items() if name == jname)
            for p in parts:
                # walk from the part's link up to the joint's child link, summing offsets
                link, length, outboard = p.link, 0.0, False
                while link != "base_link":
                    if link == joint_child:
                        outboard = True
                        break
                    parent, T, *_ = self.joints[link]
                    length += float(np.linalg.norm(T[:3, 3]))
                    link = parent
                if outboard:
                    arms[k] = max(arms[k], length + float(np.max(np.linalg.norm(p.centres, axis=1) + p.radii)))
        return arms

    def check_path(self, qs, opening: float = GRIPPER_OPEN_WIDTH, held=None, ignore: set | None = None) -> Path_Result:
        """Check a joint path (straight in joint space between the given samples) CONSERVATIVELY.

        Conservative advancement: between two configurations no robot point moves further than
        B = sum_j |dq_j| * lever_arm_j. A segment is safe if the environment clearance at its ends is
        at least B (2B for self-collision: both parts move); otherwise it is split in half, down to
        PATH_RESOLUTION. Within the sphere model, the swept motion therefore never comes closer than
        (margin - PATH_RESOLUTION) to anything checked, between samples as well as at them."""
        qs = np.atleast_2d(np.asarray(qs, float))
        dense = [qs[0]]
        for a, b in zip(qs[:-1], qs[1:]):
            n = int(np.ceil(np.degrees(np.max(np.abs(b - a))) / MAX_STEP_DEG))
            dense += [a + (b - a) * s for s in np.linspace(0, 1, n + 1)[1:]] if n else []
        arms = self.lever_arms(held)
        evaluated = [self.evaluate(q, opening, held, ignore) for q in dense[:1]]
        if evaluated[0][0]:
            return Path_Result(False, 0, evaluated[0][0])
        prev = (dense[0], evaluated[0][1], evaluated[0][2])
        self.last_subdivisions = 0
        for k, q in enumerate(dense[1:], start=1):
            hits, env, slf = self.evaluate(q, opening, held, ignore)
            if hits:
                return Path_Result(False, k, hits)
            bad = self._segment(prev, (q, env, slf), arms, opening, held, ignore, 0)
            if bad:
                return Path_Result(False, k, bad)
            prev = (q, env, slf)
        return Path_Result(True)

    def _segment(self, a, b, arms, opening, held, ignore, depth) -> list[Contact]:
        qa, ea, sa = a
        qb, eb, sb = b
        bound = float(np.sum(np.abs(qb - qa) * arms))
        if (min(ea, eb) >= bound and min(sa, sb) >= 2 * bound) or bound < PATH_RESOLUTION or depth > 20:
            return []
        self.last_subdivisions += 1
        qm = (qa + qb) / 2
        hits, em, sm = self.evaluate(qm, opening, held, ignore)
        if hits:
            return hits
        mid = (qm, em, sm)
        return self._segment(a, mid, arms, opening, held, ignore, depth + 1) or \
            self._segment(mid, b, arms, opening, held, ignore, depth + 1)
