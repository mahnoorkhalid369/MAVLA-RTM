"""Offline check of the VLM roles on today's capture (evaluation): round-1 detector masks ->
isolated crops -> Gemma descriptions -> resolve + confirm the rehearsal references. Ground truth
(mask ids -> scene names) is used ONLY to score the answers afterwards."""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perception.detector import YoloDetector  # noqa: E402
from config.models import YOLO_WEIGHTS  # noqa: E402
from perception.somark import draw_marks, draw_names, isolate  # noqa: E402
from agent.vlm_roles import GemmaVLM, Roles  # noqa: E402
from seg.evaluate import iou_matrix  # noqa: E402

TEXTS = {"red cube": "red_cube", "green cube": "green_cube", "yellow cube": "yellow_cube", "tan box": "tan_box_wide",
         "orange cube": "orange_cube", "purple cylinder": "purple_cylinder", "pink cube": "pink_cube",
         "white tall cylinder": "white_tall_cylinder"}


def main() -> int:
    out = ROOT / "out/demo/2026-10-14/vlm_check"
    out.mkdir(parents=True, exist_ok=True)
    events = []
    log = lambda kind, **f: events.append({"kind": kind, **f})  # noqa: E731
    roles = Roles(GemmaVLM(), log)
    det = YoloDetector(str(YOLO_WEIGHTS))
    d = ROOT / "data/perception/eval_layouts/today"
    A = json.loads((d / "annotations.json").read_text())
    rgb = cv2.cvtColor(cv2.imread(str(d / "images/today_000.png")), cv2.COLOR_BGR2RGB)
    ids = cv2.imread(str(d / "masks/today_000.png"), cv2.IMREAD_UNCHANGED)
    dets = det.detect(rgb)
    marks = {k + 1: x.mask for k, x in enumerate(sorted(dets, key=lambda x: -x.score))}
    gt = {a["name"]: ids == a["object_uid"] for a in A["annotations"]}
    names = list(gt)
    M = iou_matrix(list(marks.values()), np.stack([gt[n] for n in names]))
    truth = {k: (names[int(np.argmax(M[i]))] if M[i].max() >= 0.5 else None) for i, k in enumerate(marks)}
    t0 = time.time()
    desc = {}
    for k, m in marks.items():
        c = isolate(rgb, m)
        cv2.imwrite(str(out / f"obj_{k}.jpg"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
        desc[k] = roles.describe_one(c).model_dump()
    t_desc = time.time() - t0
    som = draw_marks(rgb, marks)
    cv2.imwrite(str(out / "names.jpg"), cv2.cvtColor(draw_names(som, marks, {k: f"{v['colour']} {v['name']}" for k, v in desc.items()}), cv2.COLOR_RGB2BGR))
    rows = []
    for text, name in TEXTS.items():
        r = roles.resolve(som, [{"id": k, **v} for k, v in desc.items()], text, list(marks))
        conf = roles.confirm(isolate(rgb, marks[r.id]), text).match if r.id is not None else None
        rows.append({"text": text, "truth": name, "resolved": r.id, "resolved_truth": truth.get(r.id), "ambiguous": r.ambiguous,
                     "candidates": r.candidates, "confirm": conf, "correct": truth.get(r.id) == name})
        print(rows[-1], flush=True)
    rep = {"detections": len(marks), "describe_seconds_total": round(t_desc, 1),
           "descriptions": {k: {**v, "truth": truth[k]} for k, v in desc.items()}, "references": rows,
           "correct": sum(r["correct"] for r in rows), "n": len(rows),
           "vlm_invalid_answers": sum(1 for e in events if e["kind"] == "vlm" and not e["valid"]),
           "vlm_calls": sum(1 for e in events if e["kind"] == "vlm")}
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    for k, v in rep["descriptions"].items():
        print(k, v)
    print("correct", rep["correct"], "/", rep["n"], "describe s", rep["describe_seconds_total"], "invalid", rep["vlm_invalid_answers"], "/", rep["vlm_calls"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
