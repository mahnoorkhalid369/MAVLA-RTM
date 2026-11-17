"""Replay a saved sensor capture through production perception, without ROS, Gemma or arm commands.

From v2: PYTHONPATH=src .venv-seg/bin/python scripts/replay_rgbd.py path/to/table_001_depth.npz
The adjacent table_001_rgb.png supplies RGB unless the NPZ contains an rgb array.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from agent.prompts import REVISION
from agent.ros_bridge import Frame
from agent.tools import Tools
from agent.workers import Scene
from config.models import YOLO_WEIGHTS, DETECTOR_IMGSZ, DETECTOR_CONF
from perception.detector import YoloDetector
from perception.somark import draw_marks, grid, isolate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture', type=Path)
    ap.add_argument('--camera', choices=('table', 'wrist'), default='table')
    ap.add_argument('--rgb', type=Path)
    ap.add_argument('--out', type=Path, default=Path('out/perception_v4_review'))
    ap.add_argument('--repeats', type=int, default=4)
    ap.add_argument('--weights', type=Path, default=YOLO_WEIGHTS)
    ap.add_argument('--source-ids', help='comma-separated board IDs for optional read-only transfer planning')
    ap.add_argument('--destination-id', type=int)
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error('--repeats must be positive')
    if args.source_ids and args.destination_id is None:
        ap.error('--source-ids requires --destination-id')
    capture = np.load(args.capture)
    for key in ('depth', 'K', 'q', 'T_world_cam'):
        if key not in capture:
            ap.error(f'capture lacks {key}; use a complete calibrated v4 capture')
    if 'rgb' in capture:
        rgb = capture['rgb']
    else:
        path = args.rgb or args.capture.with_name(args.capture.name.replace('_depth.npz', '_rgb.png'))
        bgr = cv2.imread(str(path))
        if bgr is None:
            ap.error(f'no RGB image at {path}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    q = capture['q']
    frame = Frame(args.camera, rgb, capture['depth'], capture['K'], float(capture['stamp']) if 'stamp' in capture else 0.,
                  q, float(capture['finger']) if 'finger' in capture else 0., T_world_camera=capture['T_world_cam'])
    args.out.mkdir(parents=True, exist_ok=True)
    detector = YoloDetector(str(args.weights), DETECTOR_IMGSZ, DETECTOR_CONF, device='cpu')
    events = []
    tools = Tools(SimpleNamespace(snapshot=lambda camera: frame, q=lambda: q), detector, None, None,
                  lambda kind, **data: events.append({'kind': kind, **data}), args.out)
    scene, ids, stable = Scene(tools, args.out), None, True
    for _ in range(args.repeats):
        if args.camera == 'table':
            obs = scene.current(force=True)
        else:
            obs = tools.perceive('wrist')
            tools.update_world(obs)
            numbered = draw_marks(obs.rgb, {obs.object_ids[k]: m for k, m in obs.marks.items()})
            cv2.imwrite(str(args.out / f'{obs.frame_id}_ids.png'), cv2.cvtColor(numbered, cv2.COLOR_RGB2BGR))
        current = dict(obs.object_ids)
        stable = stable and (ids is None or current == ids)
        ids = current
    if obs.marks:
        board = grid({obs.object_ids[k]: isolate(obs.rgb, m) for k, m in obs.marks.items()}, tile=224, cols=3)
        cv2.imwrite(str(args.out / 'candidate_board.png'), cv2.cvtColor(board, cv2.COLOR_RGB2BGR))
    plans = {}
    if args.source_ids:
        from control.motion import PlanningError
        for oid in map(int, args.source_ids.split(',')):
            try:
                plan = tools.plan_pick_place(oid, 'in', args.destination_id, 'robot', q)
                plans[oid] = {'pass': True, 'grasp': plan.grasp, 'return_home_preplanned': bool(plan.home)}
            except PlanningError as exc:
                plans[oid] = {'pass': False, 'reason': str(exc)}
    report = {'policy': REVISION, 'capture': str(args.capture.resolve()), 'weights': str(args.weights.resolve()),
              'weights_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              'observations': args.repeats, 'stable_ids': stable, 'accepted': len(obs.measured),
              'excluded': obs.excluded, 'grouped_parts': obs.grouped_parts, 'world': tools.world.summary(),
              'plans': plans,
              'scope': 'saved RGB-D perception and optional offline plans; no Gemma inference, ROS transport, physics or arm commands'}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.out / 'events.json').write_text(json.dumps(events, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('policy', 'accepted', 'stable_ids', 'observations', 'excluded', 'grouped_parts', 'plans')}, indent=2))
    return 0 if stable and all(p['pass'] for p in plans.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
