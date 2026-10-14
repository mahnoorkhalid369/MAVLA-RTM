"""Offline dry run (evaluation): measure today's capture with the agent geometry (ground-truth masks), then plan
the four rehearsal instructions with the agent planner. Object names are the scoring key only."""
import json, sys, time, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, '.')
from agent.geometry import measure, snap_supports
from agent.tools import Tools
from control.robot import ARM_HOME
from control.motion import PlanningError
d = 'data/perception/eval_layouts/today/'
A = json.load(open(d + 'annotations.json')); im = A['images'][0]
T = np.array(im['camera_pose_world']); f = 640 / np.tan(np.radians(34.7)); K = (f, f, 640.0, 360.0)
depth = cv2.imread(d + 'depth/today_000.png', cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
ids = cv2.imread(d + 'masks/today_000.png', cv2.IMREAD_UNCHANGED)
ms = {}
for a in A['annotations']:
    g, _ = measure(ids == a['object_uid'], depth, K, T, None)
    if g: ms[a['name']] = g.to_dict()
snap_supports(list(ms.values()))
t = Tools(None, None, None, None, lambda *a, **k: None, Path('.'))
oid = {n: t.world.observe(g, None) for n, g in ms.items()}  # names = evaluation key only
q = np.array(ARM_HOME, float); q[0] = np.radians(170)
for o, rel, r in [("red_cube", "on", "green_cube"), ("yellow_cube", "on", "tan_box_wide"), ("orange_cube", "on", "purple_cylinder"),
                  ("pink_cube", "next_to", "white_tall_cylinder")]:
    t0 = time.time()
    try:
        p = t.plan_pick_place(oid[o], rel, oid[r], "robot", q)
        print(o, rel, r, "OK", p.grasp, np.round(np.array(p.target_centre) * 1000), f"{time.time()-t0:.1f}s")
    except PlanningError as e:
        print(o, rel, r, "PLANNING ERROR", e, f"{time.time()-t0:.1f}s")
