"""Review follow-ups: conservative defaults, genuine initial fixtures, session continuity and smoke gates."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from agent.performance import Performance
from evaluation.benchmark import PROFILES, cases, score
from evaluation.fixtures import apply_setup
from evaluation.smoke import smoke_checks, table

ROOT = Path(__file__).resolve().parents[2]


def load_runner():
    spec = importlib.util.spec_from_file_location("test_review_runner", ROOT/"scripts/eval_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normal_sessions_default_to_baseline_but_deduplicated_remains_explicit(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("V2_"):
            monkeypatch.delenv(key)
    assert not Performance().reuse_placement and not Performance().finish_plan
    p = Performance.from_env()
    assert not p.reuse_placement and not p.finish_plan
    for k,v in PROFILES["deduplicated"].items():
        monkeypatch.setenv(k,v)
    assert Performance.from_env().reuse_placement and Performance.from_env().finish_plan
    monkeypatch.setenv("V2_BASELINE", "1")
    assert not Performance.from_env().reuse_placement and not Performance.from_env().finish_plan


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_regressions_are_real_setups_and_scripted_requests_without_expanding_core(difficulty):
    import scenarios
    assert len(cases()) == 63 and len(cases("all")) == 71
    subset = {c["id"].split("_s42_")[1]:c for c in cases("regressions") if c["difficulty"]==difficulty}
    assert set(subset)=={"on_the_tray", "occupied_tray", "milk_session", "cheezit_back"}
    assert subset["on_the_tray"]["instruction"] == "Put the apple on the tray."
    original = scenarios.generate("snack_kitchen", difficulty, 42)
    scene = apply_setup(scenarios.generate("snack_kitchen", difficulty,42), subset["occupied_tray"]["setup"])
    apple = next(o for o in scene.objects if o.id=="apple")
    tray = next(r for r in scene.regions if r.id=="serving_tray")
    assert apple.pos[:2] == tray.centre[:2] and apple.pos[2] >= tray.centre[2]
    key="food_003_cracker_box"
    turned=apply_setup(scenarios.generate("snack_kitchen",difficulty,42),subset["cheezit_back"]["setup"])
    before=next(o for o in original.objects if o.id==key)
    after=next(o for o in turned.objects if o.id==key)
    assert (after.yaw_deg-before.yaw_deg)%360 == 180
    assert after.pos == before.pos
    requests=subset["milk_session"]["instructions"]
    assert len(requests)==2 and "milk bottle" in requests[0]["request"] and "same tray" in requests[1]["request"]


def test_on_the_tray_scores_supported_containment_and_occupied_fixture_must_be_valid():
    import copy
    import scenarios
    from tests.unit.test_pipeline_performance import truth_for, positioned
    on=next(c for c in cases("regressions") if c["id"]=="easy_s42_on_the_tray")
    sc=scenarios.generate("snack_kitchen","easy",42)
    initial=truth_for(sc)
    final=copy.deepcopy(initial)
    tray=next(r for r in sc.regions if r.id=="serving_tray")
    final["apple"]=positioned(tray)
    assert score(on,sc,initial,final,{"status":"COMPLETED"})["task_success"]
    occupied=next(c for c in cases("regressions") if c["id"]=="easy_s42_occupied_tray")
    assert not score(occupied,sc,initial,final,{"status":"COMPLETED"})["scored"]
    ready=apply_setup(sc,occupied["setup"])
    initial=truth_for(ready)
    final=copy.deepcopy(initial)
    final["food_003_cracker_box"]=positioned(tray)
    assert score(occupied,ready,initial,final,{"status":"COMPLETED"})["task_success"]
    assert score(occupied,ready,initial,final,{"status":"COMPLETED"}, moved_during={"apple"})["wrong_object"]


def test_combining_session_logs_preserves_time_between_requests(tmp_path):
    for n,start in ((1,100.),(2,110.)):
        p=tmp_path/f"{n:02d}"
        p.mkdir()
        (p/"events.jsonl").write_text(json.dumps({"kind":"instruction","t":0.,"monotonic_s":start})+"\n"+
                                      json.dumps({"kind":"say","t":5.,"monotonic_s":start+5})+"\n")
    events=load_runner().session_events(tmp_path)
    assert [e["instruction_index"] for e in events]==[1,1,2,2]
    assert events[-1]["t"]==15.


def smoke_events(path):
    return [{"kind":"tool","tool":"execute","ok":True,"steps":[
        {"kind":"move","label":"lower","planned_s":9.,"sim_elapsed_s":9.,"wall_s":43.,"real_time_factor":9/43,
         "motion_wall_s":42.,"settling_wall_s":1.,"settling_sim_s":.2}]},
        {"kind":"model_input","phase":"placement","images":[{"camera":"wrist","path":str(path),"frame_id":"f"}]},
        {"kind":"timing","category":"settling","label":"wait_still","ok":True}]


def test_smoke_never_passes_missing_timing_timeout_or_missing_picture(tmp_path):
    image=tmp_path/"wrist.jpg"
    image.touch()
    row={"score":{"task_success":True}}
    events=smoke_events(image)
    assert smoke_checks(row,events,tmp_path)["automatic_checks_pass"]
    del events[0]["steps"][0]["sim_elapsed_s"]
    assert not smoke_checks(row,events,tmp_path)["automatic_checks_pass"]
    events=smoke_events(image)+[{"kind":"failure","reason":"simulation clock stalled"}]
    assert not smoke_checks(row,events,tmp_path)["automatic_checks_pass"]
    image.unlink()
    assert not smoke_checks(row,smoke_events(image),tmp_path)["automatic_checks_pass"]


def test_table_exposes_missing_scores_and_time_split():
    rows=[{"profile":"baseline","score":{"scored":True,"task_success":True},
           "timing":{"total_s":100.,"exclusive_s":{"gemma":40.,"motion":30.,"settling":10.}}},
          {"profile":"baseline","score":{"scored":False},"timing":{"total_s":100.,"exclusive_s":{}}}]
    text=table(rows)
    assert "1/2" in text and "Settling s" in text and "False reject" in text
    assert "20.0" in text and "60.0" in text


@pytest.mark.parametrize("first_succeeds",[True,False])
def test_scripted_session_reuses_tools_and_stops_if_first_request_fails(monkeypatch,tmp_path,first_succeeds):
    from agent import run, gemma, catalogue, loop, tools, ros_bridge
    from perception import detector
    requests=[{"request":"Put the milk bottle on the tray.","answers":["the blue serving tray"]},
              {"request":"Put the cheezit on that same tray.","answers":["the blue serving tray"]}]
    script=tmp_path/"requests.json"
    script.write_text(json.dumps(requests))
    monkeypatch.setattr(gemma,"Gemma",lambda *a:SimpleNamespace())
    monkeypatch.setattr(detector,"YoloDetector",lambda *a:None)
    monkeypatch.setattr(ros_bridge,"RosBridge",lambda:SimpleNamespace())
    monkeypatch.setitem(sys.modules,"execution.ros_executor",SimpleNamespace(RosExecutor=lambda:None))
    monkeypatch.setattr(run,"connect",lambda *a:True)
    monkeypatch.setattr(catalogue,"build",lambda *a:[])
    class SessionTools:
        def __init__(self,*a):
            self.persistent_catalogue={"milk":"white milk carton"}
        def revalidate_held(self):
            pass
        def begin_instruction(self,*a):
            pass
    monkeypatch.setattr(tools,"Tools",SessionTools)
    called=[]
    def instruction(request,t,g,log,out,ask_user):
        called.append((request,id(t)))
        assert t.persistent_catalogue["milk"]=="white milk carton"
        assert ask_user("Which tray?")=="the blue serving tray"
        return {"status":"COMPLETED" if first_succeeds else "STOPPED"}
    monkeypatch.setattr(loop,"run_instruction",instruction)
    result=run.main(["--instructions-file",str(script),"--out",str(tmp_path/"session"),"--live-dir","","--no-identity-check"])
    assert len(called)==(2 if first_succeeds else 1)
    assert len({key for _,key in called})==1
    assert result==(0 if first_succeeds else 1)


def test_launched_agent_keeps_sourced_ros_python_paths(monkeypatch,tmp_path):
    import subprocess
    ros_path=tmp_path/"ros_site"
    ros_path.mkdir()
    (ros_path/"synthetic_ros_dependency.py").write_text("marker = 'ROS dependency visible'\n")
    monkeypatch.setenv("PYTHONPATH",str(ros_path))
    monkeypatch.setenv("V2_OVERLAP_HOME","1")
    env=load_runner().agent_environment(cases()[0],"baseline",88)
    child=subprocess.run([sys.executable,"-c","import synthetic_ros_dependency as d; print(d.marker)"],
                         env=env,capture_output=True,text=True)
    assert child.returncode==0 and "ROS dependency visible" in child.stdout
    assert "V2_OVERLAP_HOME" not in env and env["V2_BASELINE"]=="1"


def test_each_collector_run_uses_a_fresh_child_process(monkeypatch,tmp_path):
    import os,subprocess
    runner=load_runner()
    original=subprocess.Popen
    def cpu_worker(cmd,**kwargs):
        folder=cmd[cmd.index('--out')+1]
        request=cmd[cmd.index('--collect-one')+1]
        profile=cmd[cmd.index('--profiles')+1]
        program="import json,os,sys; from pathlib import Path; p=Path(sys.argv[1]); p.mkdir(); c=json.loads(Path(sys.argv[2]).read_text()); (p/'evaluation.json').write_text(json.dumps({'case_id':c['id'],'profile':sys.argv[3],'worker_pid':os.getpid()}))"
        return original([sys.executable,'-c',program,folder,request,profile],**kwargs)
    monkeypatch.setattr(runner.subprocess,'Popen',cpu_worker)
    a=runner.collect_isolated(cases()[0],'baseline',tmp_path/'a',88,10)
    b=runner.collect_isolated(cases()[0],'deduplicated',tmp_path/'b',89,10)
    assert a['worker_pid']!=os.getpid() and b['worker_pid']!=a['worker_pid']
    assert a['profile']=='baseline' and b['profile']=='deduplicated'
