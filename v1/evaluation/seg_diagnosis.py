"""Segmentation diagnosis on SAVED images (EVALUATION ONLY; no simulator, no retraining, production code unchanged).

Input: a current-system-test folder (raw table RGB-D, the 18 raw YOLO masks of that frame, oracle container
masks, scenario metadata + ground truth in scoring_only/). Ground truth is used only to score.

Approaches (none uses ground truth):
  A  baseline        YOLO masks -> perception.geometry.measure (production path)
  B  protrusion split inside each YOLO mask: depth points standing > PROTRUDE above the mask's dominant surface
                     (histogram mode of z) form separate instances (8-connected image components); the rest stays as
                     the holder. Generic "objects in / on another object" rule, no object knowledge.
  B2 B + world split each B instance smaller than a holder is split again by connectivity of its points in a
                     top-down world grid (objects that only touch in the oblique image, e.g. a row of tubes)
  C  height map      detector-free: a top-down height map of the bench (CELL grid), a grey-scale opening with an
                     OPEN-wide structuring element estimates the local support, cells standing > PROTRUDE above it form
                     instances (components); back-projected to image masks.
Scored per container: recall (an instance within 30 mm of the true centre AND IoU >= 0.2 with the oracle mask),
IoU, xy / top errors; per method: instances produced, those matching a container, whole objects (a fixture, decor
or the microphone counted once), fragments; runtime.

    .venv-seg/bin/python evaluation/seg_diagnosis.py out/lab_exp1/current_system_test
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
PROTRUDE = 0.015  # m above the local support
MIN_PX = 60  # image component
CELL = 0.004  # height-map cell (m)
OPEN = 0.10  # height-map opening width (m): wider than any container, narrower than racks / trays
MATCH_XY, MATCH_IOU = 0.030, 0.2


def world_points(depth, K, T):
    fx, fy, cx, cy = K
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    z = depth
    pc = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)
    return pc @ T[:3, :3].T + T[:3, 3]


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def approach_a(raw_masks, depth, K, T, robot):
    from perception.geometry import measure
    out = []
    for m in raw_masks:
        g, _ = measure(m, depth, K, T, robot)
        if g is not None:
            out.append((m, g.to_dict()))
    return out


def approach_b(raw_masks, depth, K, T, robot, W):
    from perception.geometry import measure
    out = []
    for m in raw_masks:
        g, _ = measure(m, depth, K, T, robot)
        if g is None:
            continue
        z = W[..., 2][m & (depth > 0.1)]
        if len(z) < MIN_PX:
            out.append((m, g.to_dict()))
            continue
        hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.002, 0.002))
        surface = float(edges[np.argmax(hist)] + 0.001)  # dominant surface of this mask
        prot = m & (W[..., 2] > surface + PROTRUDE) & (depth > 0.1)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(prot.astype(np.uint8), connectivity=8)
        parts = [lab == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_PX]
        rest = m & ~prot
        if len(parts) == 0 or (len(parts) == 1 and parts[0].sum() > 0.6 * m.sum()):
            out.append((m, g.to_dict()))  # nothing stands out: the mask is one object
            continue
        for p in parts:
            gp, _ = measure(p, depth, K, T, robot)
            if gp is not None:
                out.append((p, gp.to_dict()))
        gr, _ = measure(rest, depth, K, T, robot)
        if gr is not None:
            out.append((rest, gr.to_dict()))
    return out


def split_by_world_xy(part, W, depth):
    """B2: one image component may hold several objects that only touch in the oblique view (a row of tubes along the
    viewing direction); split it by connectivity of its points in a top-down world grid (CELL)."""
    sel = part & (depth > 0.1)
    pts = W[sel]
    if len(pts) < MIN_PX:
        return [part]
    ij = np.floor((pts[:, :2] - pts[:, :2].min(axis=0)) / CELL).astype(int)
    grid = np.zeros(ij.max(axis=0) + 1, np.uint8)
    grid[ij[:, 0], ij[:, 1]] = 1
    n, lab = cv2.connectedComponents(grid, connectivity=8)
    if n <= 2:
        return [part]
    pix = np.argwhere(sel)
    out = []
    for i in range(1, n):
        k = lab[ij[:, 0], ij[:, 1]] == i
        if k.sum() >= MIN_PX // 2:
            m = np.zeros(part.shape, bool)
            m[pix[k, 0], pix[k, 1]] = True
            out.append(m)
    return out or [part]


def approach_b2(raw_masks, depth, K, T, robot, W):
    from perception.geometry import measure
    out = []
    for m, g in approach_b(raw_masks, depth, K, T, robot, W):
        pieces = split_by_world_xy(m, W, depth) if g["top_z"] > 0 and m.sum() < 5000 else [m]
        if len(pieces) == 1:
            out.append((m, g))
            continue
        for p in pieces:
            gp, _ = measure(p, depth, K, T, robot)
            if gp is not None:
                out.append((p, gp.to_dict()))
    return out


def approach_c(depth, K, T, robot, W):
    from perception.geometry import measure
    valid = (depth > 0.1) & (depth < 5)
    P = W[valid]
    r = np.hypot(P[:, 0], P[:, 1])
    keep = (r > 0.30) & (r < 1.0) & (P[:, 2] > 0.003) & (P[:, 2] < 0.40)
    keep[keep] &= ~robot.contains(P[keep])
    pix = np.argwhere(valid)[keep]
    P = P[keep]
    x0, y0 = -1.0, -0.4
    ij = np.floor((P[:, :2] - [x0, y0]) / CELL).astype(int)
    shape = (int(2.0 / CELL), int(1.4 / CELL))
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
    ij, P, pix = ij[ok], P[ok], pix[ok]
    H = np.zeros(shape, np.float32)
    np.maximum.at(H, (ij[:, 0], ij[:, 1]), P[:, 2].astype(np.float32))
    k = int(OPEN / CELL) | 1
    S = cv2.morphologyEx(H, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    prot = ((H - S) > PROTRUDE).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(prot, connectivity=8)
    cell_lab = lab[ij[:, 0], ij[:, 1]]
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 6:
            continue
        sel = (cell_lab == i) & (P[:, 2] > S[ij[:, 0], ij[:, 1]] + PROTRUDE * 0.5)
        if sel.sum() < MIN_PX:
            continue
        m = np.zeros(depth.shape, bool)
        m[pix[sel, 0], pix[sel, 1]] = True
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
        g, _ = measure(m, depth, K, T, robot)
        if g is not None:
            out.append((m, g.to_dict()))
    return out


def score(instances, gt, truth, oracle, meta):
    from evaluation.lab_exp1_score import fixture_at
    per, used = {}, set()
    for gid in gt:
        x, y, z = truth[gid]["position"]
        best = None
        for i, (m, g) in enumerate(instances):
            d = float(np.hypot(g["centre"][0] - x, g["centre"][1] - y))
            j = iou(m, oracle[gid])
            if d <= MATCH_XY and j >= MATCH_IOU and (best is None or j > best[2]):
                best = (i, d, j, g)
        if best is None:
            per[gid] = {"found": False}
            continue
        i, d, j, g = best
        used.add(i)
        top_true = z + gt[gid]["dimensions"]["height"]
        per[gid] = {"found": True, "iou": round(j, 2), "xy_error_mm": round(d * 1000, 1),
                    "top_error_mm": round((g["top_z"] - top_true) * 1000, 1),
                    "height_mm_measured_vs_true": [round(g["height"] * 1000, 1), round(gt[gid]["dimensions"]["height"] * 1000, 1)]}
    whole, fragments, seen_fixtures = 0, 0, set()
    for i, (m, g) in enumerate(instances):
        if i in used:
            continue
        c = g["centre"][:2]
        fx = fixture_at(meta, c)
        if fx is None:
            whole += 1  # decor beyond reach, the microphone, other real things: one object each
        elif fx not in seen_fixtures and g["footprint"][0] > 0.08:
            seen_fixtures.add(fx)
            whole += 1  # the fixture itself (racks, trays, pad)
        else:
            fragments += 1  # a piece of a fixture or a second instance of it
    n = len(instances)
    found = sum(v["found"] for v in per.values())
    return {"instances": n, "containers_found": found, "recall": round(found / len(gt), 3),
            "container_precision": round(found / max(found + fragments, 1), 3),
            "whole_non_container_objects": whole, "fragments": fragments,
            "mean_iou": round(float(np.mean([v["iou"] for v in per.values() if v["found"]] or [0])), 2),
            "max_xy_error_mm": max([v["xy_error_mm"] for v in per.values() if v["found"]] or [None], key=lambda v: v or 0),
            "per_container": per}


def main(folder: str) -> int:
    from perception.geometry import RobotVolume
    from control.collision import CollisionChecker
    from control.kinematics import Kinematics
    from control.robot import ARM_HOME, gripper_gap, table_camera_pose
    base = Path(folder)
    out = base.parent / "segmentation_diagnosis"
    out.mkdir(exist_ok=True)
    res = json.loads((base / "results.json").read_text())
    K = tuple(res["step1"]["table_camera"]["K"])
    rgb = cv2.cvtColor(cv2.imread(str(base / "raw" / "table_rgb.png")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(base / "raw" / "table_depth_mm.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    raw = [(int(p.name.split("_")[1]), cv2.imread(str(p), 0) > 0) for p in sorted((base / "masks").glob("raw_*.png"))]
    raw_masks = [m for _, m in sorted(raw)]
    meta = json.loads((base / "scoring_only" / "scenario_metadata.json").read_text())
    truth = json.loads((base / "scoring_only" / "probe_initial.json").read_text())["truth"]["objects"]
    gt = {o["id"]: o for o in meta["objects"] if o["category"] in ("sample_tube", "sample_vial")}
    oracle = {g: cv2.imread(str(base / "masks" / f"oracle_{g}.png"), 0) > 0 for g in gt}
    T = table_camera_pose()
    q = np.asarray(ARM_HOME, float)
    robot = RobotVolume(CollisionChecker(), q, gripper_gap(0.0), None, Kinematics(tip="tool0").fk(q))
    W = world_points(depth, K, T)
    report = {"inputs": str(base), "params": {"PROTRUDE_m": PROTRUDE, "MIN_PX": MIN_PX, "CELL_m": CELL, "OPEN_m": OPEN},
              "raw_masks": len(raw_masks)}
    # raw-mask inventory: which containers each raw mask covers (where instances are lost)
    inv = []
    for i, m in enumerate(raw_masks):
        cov = {g: round(float((m & oracle[g]).sum() / max(oracle[g].sum(), 1)), 2) for g in gt}
        inv.append({"raw": i, "pixels": int(m.sum()), "covers": {g: c for g, c in cov.items() if c > 0.05}})
    report["raw_inventory"] = inv
    report["lost_at"] = {g: ("no raw mask of its own: lost in YOLO's output (only a holder mask covers it)"
                             if not any(c["covers"].get(g, 0) > 0.3 and c["pixels"] < 4 * oracle[g].sum() for c in inv)
                             else "has its own raw mask") for g in gt}
    for name, fn in (("A_baseline", lambda: approach_a(raw_masks, depth, K, T, robot)),
                     ("B_protrusion_split", lambda: approach_b(raw_masks, depth, K, T, robot, W)),
                     ("B2_protrusion_split_world_xy", lambda: approach_b2(raw_masks, depth, K, T, robot, W)),
                     ("C_height_map", lambda: approach_c(depth, K, T, robot, W))):
        t0 = time.monotonic()
        inst = fn()
        secs = round(time.monotonic() - t0, 2)
        s = score(inst, gt, truth, oracle, meta)
        s["seconds"] = secs
        report[name] = s
        vis = rgb.copy()
        rng = np.random.default_rng(0)
        for m, _ in inst:
            vis[m] = (0.45 * vis[m] + 0.55 * rng.integers(40, 255, 3)).astype(np.uint8)
        for g in gt:
            cnts, _ = cv2.findContours(oracle[g].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (255, 0, 255) if s["per_container"][g]["found"] else (255, 0, 0), 2)
        cv2.imwrite(str(out / f"{name}.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"[seg] {name}: recall {s['recall']}, container precision {s['container_precision']}, instances {s['instances']}, "
              f"fragments {s['fragments']}, mean IoU {s['mean_iou']}, {secs} s", flush=True)
    (out / "results.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
