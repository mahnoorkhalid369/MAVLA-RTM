"""Replay source association and side-view planning on the saved Cheez-It failure; no robot/model calls.

From v2: OPENBLAS_NUM_THREADS=1 .venv-seg/bin/python scripts/replay_source_view.py
"""

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perception.geometry import back_project
from perception.source_evidence import foreground
from perception.viewpoints import ViewpointPlanner
from perception.world_model import ObjectRecord, WorldModel


def replay():
    fixture = ROOT / "tests/fixtures/cheezit_side_foreground"
    metadata = json.loads(fixture.with_suffix(".json").read_text())
    data = np.load(fixture.with_suffix(".npz"))
    world = WorldModel()
    oid = metadata["source_id"]
    world.objects[oid] = ObjectRecord(oid, metadata["geometry"])
    for i, g in enumerate(metadata["other_geometry"], 100):
        world.objects[i] = ObjectRecord(i, g)
    mask = data["labels"] == metadata["source_mark"]
    frame = SimpleNamespace(depth=data["depth"], K=data["K"])
    pose = data["T_world_cam"]
    raw = back_project(mask,frame.depth,frame.K,pose)
    _, points, stats = foreground(mask,frame,pose,support_z=metadata["geometry"]["support_z"])
    box = world.objects[oid].box()
    corners = [box.centre + box.R @ (box.half * [x,y,z])
               for x in (-1,1) for y in (-1,1) for z in (-1,1)]
    started = time.monotonic()
    plan = ViewpointPlanner().plan(np.asarray(metadata["geometry"]["centre"]),np.array(metadata["inspection_q"]),
             [o.box() for o in world.objects.values()],target_box_name=f"obj_{oid}",distances=(.35,.45),
             elevations=(25.,35.),min_view_angle_deg=20.,target_points=corners,
             camera_K=metadata["inspection_camera_K"],image_shape=metadata["inspection_image_shape"],
             body_visibility_fraction=2/3)
    return {"source_run": metadata["source_run"], "scope": "Saved depth association and planning only; no motion, new camera capture or model inference. Other-object volumes are conservative; wrist-local/unexplained obstacles unavailable in this fixture.",
            "raw_binding": world.associate_partial(raw), "foreground_binding": world.associate_partial(points),
            "foreground": stats, "planning_wall_s": round(time.monotonic()-started,3), "viewpoint": plan.summary()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    result = replay()
    text = json.dumps(result,indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text)
    print(text,end="")
    return 0 if result["foreground_binding"][:2] == ("matched",3) and result["viewpoint"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
