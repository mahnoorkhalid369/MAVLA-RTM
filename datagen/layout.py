"""Scene sampling for synthetic instance-segmentation data (numpy only, no Isaac Sim): what goes in one
image — objects (shape family, size, resting face, pose, material, stacks, one held in the gripper),
the robot's joint state, the camera, lights, floor and RGB effects. Isaac Sim only renders it
(datagen/generate.py). Everything is drawn from a generator seeded by (dataset seed, split, index).

Splits: "train" / "val" use the training families; "test_novel" uses families never used in
training. Exact copies of the evaluation scene's objects (sim/scene_objects.py, read ONLY to exclude
them) are re-coloured.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from control.collision import Box, CollisionChecker
from control.kinematics import Kinematics
from control.robot import (
    ARM_JOINTS,
    TABLE_CAMERA_POSITION,
    TABLE_CAMERA_TARGET,
    finger_angle_for_gap,
)

PROCEDURAL_TRAIN = ("box", "cylinder", "sphere", "capsule", "cone", "plate", "l_shape")
PROCEDURAL_NOVEL = ("torus", "hex_prism", "tri_prism", "pyramid")


@dataclass
class Material:
    colour: tuple[float, float, float]  # linear-ish 0-1 base colour
    roughness: float
    metallic: float
    texture: int | None = None  # index into the texture pool
    keep_asset: bool = False  # YCB: keep the asset's own material


@dataclass
class ObjectSpec:
    uid: int
    family: str  # procedural family or "ycb"
    asset: str | None  # YCB / Isaac asset name (file stem) or None
    dims: tuple[float, float, float]  # extent along the object's own x, y, z after scaling (m)
    rest: tuple[int, int]  # resting face: (axis that points up 0/1/2, sign +1/-1)
    yaw_deg: float
    position: tuple[float, float, float]  # (x, y, z of the bottom) in the world (m)
    material: Material
    stacked_on: int | None = None
    held: bool = False
    up_height: float = 0.0  # height above its support (m)
    scale: float = 1.0
    extra: dict = field(default_factory=dict)


@dataclass
class SceneSpec:
    split: str
    index: int
    seed: int
    objects: list[ObjectSpec]
    robot_q: list[float]
    finger_rad: float
    camera: dict  # {"view": "table"|"wrist", "eye", "target", "roll_deg"} -> camera_pose()
    lights: dict
    floor: dict
    rgb_effects: dict


def image_seed(seed: int, split: str, index: int) -> int:
    return int(hashlib.sha256(f"{seed}/{split}/{index}".encode()).hexdigest()[:12], 16)


def look_at(eye, target, roll_deg: float = 0.0) -> np.ndarray:
    """4x4 pose of a ROS optical frame (z forward, x right, y down) at eye looking at target."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    if np.linalg.norm(x) < 1e-6:  # looking straight down
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    a = np.radians(roll_deg)
    roll = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R @ roll, eye
    return T


def camera_pose(cam: dict) -> np.ndarray:
    return look_at(cam["eye"], cam["target"], cam.get("roll_deg", 0.0))


def _rot_up(axis: int, sign: int) -> np.ndarray:
    """Rotation turning the object's local `axis` (sign) to world +z: the resting face."""
    v = np.zeros(3)
    v[axis] = sign
    target = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(v, target))
    if c > 0.999:
        return np.eye(3)
    if c < -0.999:
        return np.diag([1.0, -1.0, -1.0])
    k = np.cross(v, target)
    s = np.linalg.norm(k)
    k /= s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + K * s + K @ K * (1 - c)


def object_rotation(o: ObjectSpec) -> np.ndarray:
    a = np.radians(o.yaw_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return Rz @ _rot_up(*o.rest)


def world_extent(o: ObjectSpec) -> np.ndarray:
    """Axis-aligned world extent (x, y, z) of the object's (scaled) local box in its pose."""
    return np.abs(object_rotation(o)) @ np.asarray(o.dims)


class SceneSampler:
    def __init__(self, cfg: dict, asset_dims: dict[str, tuple[float, float, float]]) -> None:
        """asset_dims: {asset stem: local bbox size (m) at scale 1} for the YCB / Isaac assets."""
        self.cfg = cfg
        self.asset_dims = asset_dims
        self.kin = Kinematics()
        self.checker = CollisionChecker()
        from sim.scene_objects import OBJECTS  # ONLY to keep exact evaluation-object copies out of the data

        self.eval_objects = OBJECTS

    # ---------------------------------------------------------------- objects
    def _material(self, rng, keep_ok: bool) -> Material:
        c = self.cfg["materials"]
        keep = keep_ok and rng.random() < c["ycb_keep_material_probability"]
        metallic = float(rng.uniform(0.6, 1.0)) if rng.random() < c["metallic_probability"] else float(rng.uniform(0, 0.1))
        tex = int(rng.integers(c["texture_pool"])) if rng.random() < c["textured_object_probability"] else None
        return Material(tuple(float(v) for v in rng.uniform(0, 1, 3)), float(rng.uniform(0.05, 1.0)), metallic, tex, keep)

    def _family(self, rng, split: str) -> str:
        fams = self.cfg["families"]["novel" if split == "test_novel" else "train"]
        return fams[int(rng.integers(len(fams)))]

    def _dims(self, rng, family: str) -> tuple[tuple[float, float, float], str | None, float, dict]:
        lo, hi = self.cfg["objects"]["size_range_m"]
        s = lambda: float(rng.uniform(lo, hi))  # noqa: E731
        extra = {}
        if family in ("box",):
            return (s(), s(), s()), None, 1.0, extra
        if family == "plate":
            return (s() * 1.3, s() * 1.3, float(rng.uniform(0.006, 0.02))), None, 1.0, extra
        if family in ("cylinder", "hex_prism", "tri_prism"):
            d = s()
            return (d, d, s()), None, 1.0, extra
        if family == "sphere":
            d = s()
            return (d, d * rng.uniform(0.8, 1.0), d * rng.uniform(0.8, 1.0)), None, 1.0, extra
        if family in ("capsule",):
            d = float(rng.uniform(lo, hi * 0.6))
            return (d, d, d + s()), None, 1.0, extra
        if family in ("cone", "pyramid"):
            d = s()
            return (d, d, s()), None, 1.0, extra
        if family == "l_shape":
            a, b, t, h = s(), s(), float(rng.uniform(0.015, 0.04)), s()
            extra["thickness"] = t
            return (a, b, h), None, 1.0, extra
        if family == "torus":
            d = float(rng.uniform(0.05, 0.16))
            return (d, d, d / 3), "torus", d / 1.5, extra
        if family in ("ycb_train", "ycb_novel"):
            names = self.cfg["ycb"]["novel" if family == "ycb_novel" else "train"]
            name = names[int(rng.integers(len(names)))]
            scale = float(rng.uniform(*self.cfg["ycb"]["scale"]))
            return tuple(float(v) * scale for v in self.asset_dims[name]), name, scale, extra
        raise ValueError(family)

    def _is_eval_copy(self, family: str, dims, colour) -> bool:
        """Same family (box / cylinder / sphere), size within tolerance and similar colour as one of the
        evaluation objects -> that exact object must not appear in training."""
        tol, ctol = self.cfg["eval_exclusion"]["size_tol_m"], self.cfg["eval_exclusion"]["colour_tol"]
        shape = {"box": "box", "cylinder": "cylinder", "sphere": "sphere"}.get(family)
        if shape is None:
            return False
        for o in self.eval_objects.values():
            if o.shape == shape and np.all(np.abs(np.sort(dims) - np.sort(o.size)) < tol) and \
                    np.all(np.abs(np.asarray(colour) - np.asarray(o.color)) < ctol):
                return True
        return False

    def _place(self, rng, placed: list[ObjectSpec], o: ObjectSpec, region) -> bool:
        ext = world_extent(o)
        r_obj = 0.5 * float(np.hypot(ext[0], ext[1]))
        stack_p = self.cfg["objects"]["stack_probability"]
        supports = [p for p in placed if p.family in ("box", "plate", "cylinder", "hex_prism", "ycb_train", "ycb_novel")
                    and p.stacked_on is None and not p.held and min(world_extent(p)[:2]) > max(ext[:2]) * 0.8]
        if supports and rng.random() < stack_p:
            s = supports[int(rng.integers(len(supports)))]
            se = world_extent(s)
            o.position = (s.position[0] + float(rng.uniform(-0.1, 0.1)) * se[0], s.position[1] + float(rng.uniform(-0.1, 0.1)) * se[1],
                          s.position[2] + se[2])
            o.stacked_on = s.uid
            return True
        for _ in range(40):
            xy = region(rng)
            if xy is None:
                continue
            ok = all(np.hypot(xy[0] - p.position[0], xy[1] - p.position[1]) >= r_obj + 0.5 * np.hypot(*world_extent(p)[:2]) + 0.002
                     for p in placed if p.stacked_on is None and not p.held)
            excl = all(np.hypot(xy[0] - c[0], xy[1] - c[1]) >= c[2] + r_obj for c in self.cfg["workspace"]["keep_clear"])
            if ok and excl:
                o.position = (float(xy[0]), float(xy[1]), 0.0)
                return True
        return False

    # ----------------------------------------------------------------- camera
    def _camera(self, rng) -> dict:
        if rng.random() < self.cfg["views"]["table"]:
            j = self.cfg["views"]["table_jitter"]
            eye = np.asarray(TABLE_CAMERA_POSITION) + rng.uniform(-j["pos_m"], j["pos_m"], 3)
            target = np.asarray(TABLE_CAMERA_TARGET, float)
            dist = np.linalg.norm(target - eye)
            # turn the viewing direction by up to rot_deg: move the target perpendicular to the view
            off = rng.normal(size=3)
            off -= off.dot(target - eye) / dist ** 2 * (target - eye)
            off *= np.tan(np.radians(rng.uniform(0, j["rot_deg"]))) * dist / max(np.linalg.norm(off), 1e-9)
            return {"view": "table", "eye": eye.tolist(), "target": (target + off).tolist(),
                    "roll_deg": float(rng.uniform(-j["rot_deg"], j["rot_deg"]))}
        w = self.cfg["views"]["wrist"]
        a, r = rng.uniform(-np.pi, np.pi), rng.uniform(*w["radius_m"])
        eye = np.array([r * np.cos(a), r * np.sin(a), rng.uniform(*w["height_m"])])
        out = np.array([np.cos(a), np.sin(a)]) * rng.uniform(*w["look_out_m"])
        target = np.array([eye[0] + out[0], eye[1] + out[1], 0.0]) + np.r_[rng.uniform(-0.1, 0.1, 2), 0]
        return {"view": "wrist", "eye": eye.tolist(), "target": target.tolist(), "roll_deg": float(rng.uniform(*w["roll_deg"]))}

    def _region(self, cam: dict):
        """Sampler of floor xy inside the camera's view (projected) and the workspace annulus."""
        T = camera_pose(cam)
        res = self.cfg["render"]["resolution"]
        f = res[0] / 2 / np.tan(np.radians(self.cfg["render"]["hfov_deg"] / 2))
        R, o = T[:3, :3], T[:3, 3]
        ws = self.cfg["workspace"]
        centre = np.array(cam["target"][:2])

        def sample(rng):
            xy = centre + rng.uniform(-1.0, 1.0, 2) * (0.9 if cam["view"] == "table" else 0.45)
            if not ws["r_min"] <= np.hypot(*xy) <= ws["r_max"]:
                return None
            c = R.T @ (np.array([xy[0], xy[1], 0.03]) - o)
            if c[2] <= 0.05:
                return None
            u, v = f * c[0] / c[2] + res[0] / 2, f * c[1] / c[2] + res[1] / 2
            m = ws["frame_margin_px"]
            return xy if (-m <= u <= res[0] + m and -m <= v <= res[1] + m) else None
        return sample

    # ------------------------------------------------------------------ robot
    def _robot(self, rng, objects: list[ObjectSpec], held: ObjectSpec | None):
        boxes = [Box(f"o{o.uid}", np.asarray(o.position) + [0, 0, world_extent(o)[2] / 2], np.eye(3), world_extent(o) / 2 + 0.01)
                 for o in objects if not o.held]
        self.checker.obstacles = boxes
        lo, hi = self.kin.lower, self.kin.upper
        for _ in range(60):
            if held is not None:  # gripper pointing down somewhere above the floor, object in the pads
                xy = rng.uniform(-0.7, 0.7, 2)
                if not 0.45 <= np.hypot(*xy) <= 0.85:
                    continue
                from control.skills import grip_rotation
                yaw = float(rng.uniform(-90, 90))
                q, ok = self.kin.ik_best(np.array([xy[0], xy[1], rng.uniform(0.2, 0.45)]), grip_rotation(yaw),
                                         np.radians([0, -10, -30, 0, -60, 0]))
                if not ok:
                    continue
            else:
                q = rng.uniform(lo * 0.8, hi * 0.8)
                q[1] = rng.uniform(np.radians(-80), np.radians(30))  # keep the arm mostly up: realistic poses
            width = held.dims[0] if held is not None else None
            opening = (width + 0.004) if width else 0.087
            hb = None
            if held is not None:
                grip = min(0.025, held.dims[2] / 2)
                hb = self.checker.held_box_in_tool(held.dims, grip)
            hits = self.checker.check(q, opening, hb, ignore={("held", "floor")})
            if not hits:
                finger = finger_angle_for_gap(max(width - 0.004, 0.0)) if width else float(rng.uniform(0, np.radians(30)))
                return [float(v) for v in q], finger
        return [0.0, -0.3, -0.5, 0.0, -1.2, 0.0], 0.0  # fallback: a compact pose (checked later by masks only)

    # ------------------------------------------------------------------- main
    def sample(self, split: str, index: int) -> SceneSpec:
        seed = image_seed(self.cfg["seed"], split, index)
        rng = np.random.default_rng(seed)
        cam = self._camera(rng)
        region = self._region(cam)
        n = int(rng.integers(self.cfg["objects"]["count"][0], self.cfg["objects"]["count"][1] + 1))
        objects: list[ObjectSpec] = []
        uid = 1
        held_spec = None
        want_held = rng.random() < self.cfg["objects"]["held_probability"]
        for k in range(n * 3):
            if len(objects) >= n:
                break
            fam = self._family(rng, split)
            dims, asset, scale, extra = self._dims(rng, fam)
            irregular = fam in ("ycb_train", "ycb_novel", "torus", "l_shape", "cone", "pyramid", "capsule")
            rest = (int(rng.integers(3)), int(rng.choice([-1, 1]))) if irregular and rng.random() < 0.5 else (2, 1)
            mat = self._material(rng, keep_ok=asset is not None and fam.startswith("ycb"))
            for _ in range(5):
                if not self._is_eval_copy(fam, dims, mat.colour):
                    break
                mat.colour = tuple(float(v) for v in rng.uniform(0, 1, 3))
            o = ObjectSpec(uid, fam, asset, tuple(float(v) for v in dims), rest, float(rng.uniform(-180, 180)), (0.0, 0.0, 0.0),
                           mat, scale=scale, extra=extra)
            if want_held and held_spec is None and fam in ("box", "cylinder", "ycb_train", "ycb_novel", "hex_prism") \
                    and o.dims[0] <= 0.08 and o.dims[2] >= 0.03:
                o.held, o.rest = True, (2, 1)
                held_spec = o
                objects.append(o)
                uid += 1
                continue
            if self._place(rng, objects, o, region):
                objects.append(o)
                uid += 1
        q, finger = self._robot(rng, objects, held_spec)
        if held_spec is not None:
            T = self.kin.fk(np.asarray(q))  # tcp pose; object hangs between the pads, top grip_depth above the TCP
            grip = min(0.025, held_spec.dims[2] / 2)
            held_spec.position = tuple(float(v) for v in T[:3, 3] + [0, 0, grip - held_spec.dims[2]])
            held_spec.extra["R_world"] = (T[:3, :3] @ np.diag([1, -1, -1])).tolist()  # object upright, turned with the gripper
        light = self.cfg["lights"]
        lights = {"dome_intensity": float(rng.uniform(*light["dome_intensity"])),
                  "dome_tint": rng.uniform(0.85, 1.0, 3).tolist(),
                  "extra": [{"kind": ["sphere", "rect", "distant"][int(rng.integers(3))],
                             "position": [float(rng.uniform(-1.5, 1.5)), float(rng.uniform(-1.5, 1.5)), float(rng.uniform(0.8, 3.0))],
                             "intensity": float(rng.uniform(*light["extra_intensity"])), "colour": rng.uniform(0.8, 1.0, 3).tolist(),
                             "angle_deg": [float(rng.uniform(20, 70)), float(rng.uniform(-180, 180))]}
                            for _ in range(int(rng.integers(light["extra"][0], light["extra"][1] + 1)))]}
        fl = self.cfg["floor"]
        floor = {"kind": fl["kinds"][int(rng.integers(len(fl["kinds"])))], "texture": int(rng.integers(fl["texture_pool"])),
                 "tint": rng.uniform(0.7, 1.0, 3).tolist(), "roughness": float(rng.uniform(0.2, 1.0)),
                 "metallic": float(rng.uniform(0, 0.3)) if rng.random() < 0.8 else float(rng.uniform(0.5, 0.9)),
                 "uv_scale": float(rng.uniform(*fl["uv_scale"])), "uv_rot_deg": float(rng.uniform(0, 90))}
        fx = self.cfg["rgb_effects"]
        effects = {"noise_sigma": float(rng.uniform(*fx["noise_sigma"])),
                   "blur_px": float(rng.uniform(*fx["blur_px"])) if rng.random() < fx["blur_probability"] else 0.0,
                   "motion_blur": bool(rng.random() < 0.5), "motion_angle_deg": float(rng.uniform(0, 180)),
                   "exposure": float(rng.uniform(*fx["exposure"])), "gamma": float(rng.uniform(*fx["gamma"])),
                   "rng_seed": int(rng.integers(2**31))}
        return SceneSpec(split, index, seed, objects, q, float(finger), cam, lights, floor, effects)


def apply_rgb_effects(img: np.ndarray, fx: dict) -> np.ndarray:
    """Camera effects on a rendered sRGB image (uint8): exposure, gamma, blur (defocus or motion), noise."""
    import cv2

    rng = np.random.default_rng(fx["rng_seed"])
    x = img.astype(np.float32) / 255.0
    x = np.clip(x * fx["exposure"], 0, 1) ** fx["gamma"]
    if fx["blur_px"] > 0.3:
        k = max(3, int(round(fx["blur_px"] * 2)) | 1)
        if fx["motion_blur"]:
            kern = np.zeros((k, k), np.float32)
            kern[k // 2, :] = 1.0
            M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), fx["motion_angle_deg"], 1.0)
            kern = cv2.warpAffine(kern, M, (k, k))
            x = cv2.filter2D(x, -1, kern / max(kern.sum(), 1e-6))
        else:
            x = cv2.GaussianBlur(x, (k, k), fx["blur_px"] / 2)
    x = x + rng.normal(0, fx["noise_sigma"] / 255.0, x.shape).astype(np.float32)
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)
