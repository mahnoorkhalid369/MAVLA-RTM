"""Experiment 1 scorer (evaluation/lab_exp1_score.py) on a synthetic agent run built from the real lab metadata."""

import json

from evaluation.lab_exp1_score import score
from scenarios import generate


def fake_run(tmp_path):
    meta = generate("laboratory", "easy", 42).metadata()
    gt = [o for o in meta["objects"] if o["category"] in ("sample_tube", "sample_vial")]
    truth = {"sim_time": 3.0, "objects": {o["id"]: {"position": o["spawn_pose"]["xyz"], "quat_wxyz": [1, 0, 0, 0]} for o in gt}}

    def instruction(name, world_offset, extra_events=(), answer=None):
        d = tmp_path / name
        d.mkdir()
        objs, descs = [], {}
        for k, o in enumerate(gt[:-1], 1):  # the last container is missed by the detector
            x, y, z = o["spawn_pose"]["xyz"]
            h = o["dimensions"]["height"]
            objs.append({"mark": k, "world_id": k + world_offset, "centre_m": [x + 0.004, y, z + h / 2], "height_m": h})
            descs[str(k + world_offset)] = {"name": "test tube" if o["attributes"]["container"] == "tube" else "vial",
                                            "colour": o["attributes"]["cap_colour"] if k % 2 else "grey"}  # by world id
        fx = next(f for f in meta["fixtures"] if f["id"] == "inspection_station")
        objs.append({"mark": 99, "world_id": 99, "centre_m": [fx["pos"][0], fx["pos"][1], 0.006], "height_m": 0.012})
        st = {"facts": {"camera": "table", "objects": objs}, "hypotheses": {"descriptions": descs}}
        (d / "task_state.json").write_text(json.dumps(st))
        ev = [{"kind": "vlm", "role": "describe", "seconds": 2.0, "tokens": {"prompt": 100, "answer": 20}}, *extra_events]
        if answer:
            ev.append({"kind": "answer", "question": "id?", "answer": answer})
        (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev))
        (d / "result.json").write_text(json.dumps({"instruction": name, "status": "COMPLETED", "seconds": 10.0}))
    instruction("01", 0)
    instruction("02", 0)
    vial = next(o for o in gt if o["attributes"]["container"] == "vial")
    real_id = vial["attributes"]["sample_id"]
    instruction("03", 50, answer=f"The cap says {real_id}; the other one is X000")
    return meta, truth, gt, real_id


def test_scorer_counts_detections_identity_colours_labels_and_hallucinations(tmp_path):
    meta, truth, gt, real_id = fake_run(tmp_path)
    rep = score(tmp_path, meta, truth)
    s = rep["summary"]
    assert s["detected"] == f"{len(gt) - 1}/{len(gt)}" and s["max_xy_error_mm"] == 4.0
    assert s["cap_colour_named"] == f"{(len(gt)) // 2}/{len(gt) - 1}"  # odd marks name the right colour
    assert s["identity_stable"] == "0/7"  # instruction 03 re-numbered every world id
    first = rep["instructions"][0]
    assert first["unmatched_measurements"][0]["fixture"] == "inspection_station"  # a fixture seen as an object
    assert rep["hallucinated_ids"] == ["X000"]
    assert [k for k, v in rep["label_read_correctly"].items() if v] == [vial["id"] for vial in gt
                                                                         if vial["attributes"]["sample_id"] == real_id]
    assert rep["gemma_by_role"]["describe"]["calls"] == 3


def test_scorer_identity_is_stable_when_world_ids_repeat(tmp_path):
    meta, truth, gt, _ = fake_run(tmp_path)
    (tmp_path / "03" / "task_state.json").unlink()  # only 01 and 02 observe the table
    rep = score(tmp_path, meta, truth)
    assert rep["summary"]["identity_stable"] == f"{len(gt) - 1}/{len(gt) - 1}"


def test_descriptions_are_looked_up_by_world_id_not_by_mark(tmp_path):
    """Regression (Experiment 1 erratum): world ids differ from marks (a fixture seen as two masks shares one world
    id, so every later mark is shifted). Descriptions are stored by world id; reading them by mark misattributes."""
    meta = generate("laboratory", "easy", 42).metadata()
    gt = [o for o in meta["objects"] if o["category"] in ("sample_tube", "sample_vial")][:2]
    truth = {"sim_time": 3.0, "objects": {o["id"]: {"position": o["spawn_pose"]["xyz"], "quat_wxyz": [1, 0, 0, 0]} for o in gt}}
    d = tmp_path / "01"
    d.mkdir()
    objs, descs = [], {}
    for k, o in enumerate(gt, 1):
        x, y, z = o["spawn_pose"]["xyz"]
        objs.append({"mark": k, "world_id": k + 10, "centre_m": [x, y, z + 0.05], "height_m": 0.1})
        descs[str(k + 10)] = {"name": o["attributes"]["container"], "colour": o["attributes"]["cap_colour"]}  # correct one
        descs[str(k)] = {"name": "board", "colour": "nothing like it"}  # what a mark-keyed lookup would wrongly read
    (d / "task_state.json").write_text(json.dumps({"facts": {"camera": "table", "objects": objs},
                                                   "hypotheses": {"descriptions": descs}}))
    (d / "events.jsonl").write_text("")
    (d / "result.json").write_text(json.dumps({"status": "COMPLETED", "seconds": 1.0}))
    rep = score(tmp_path, meta, truth)
    assert rep["summary"]["cap_colour_named"] == "2/2" and rep["summary"]["type_named"] == "2/2"
