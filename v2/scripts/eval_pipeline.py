#!/usr/bin/env python3
"""Fixed live ablations. Ground truth stays in this evaluation process, never in agent inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.benchmark import PROFILES, cases, compare, score, timing_summary, placement_labels
from evaluation.fixtures import apply_setup


def read_events(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a writer may be midway through its final line
    return rows


def session_events(agent_folder):
    events = []
    for path in sorted(agent_folder.glob("[0-9][0-9]/events.jsonl")):
        events.extend({**e, "instruction_index":int(path.parent.name)} for e in read_events(path))
    if events and all("monotonic_s" in e for e in events):
        start = events[0]["monotonic_s"]-events[0]["t"]
        events = [{**e, "t":e["monotonic_s"]-start} for e in events]
    return events


def agent_environment(case, profile, domain):
    import scenarios
    env = {k:v for k,v in os.environ.items() if not k.startswith("V2_")}
    env.update(PROFILES[profile])
    python_paths = [str(ROOT/"src"), *env.get("PYTHONPATH", "").split(os.pathsep)]
    env.update(ROS_DOMAIN_ID=str(domain), TABLE_CAMERA=scenarios.table_camera(case["scenario"]),
               WORKCELL_LAYOUT=scenarios.layout(case["scenario"]),
               PYTHONPATH=os.pathsep.join(dict.fromkeys(p for p in python_paths if p)))
    return env


def stop_owned_sim(folder):
    record = folder/"agent/sim.process.json"
    if not record.exists():
        return
    owner = json.loads(record.read_text())
    def same_process():
        try:
            stat = Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(")", 1)[1].split()
            return stat[19] == owner["start_ticks"] and stat[0] != "Z"
        except FileNotFoundError:
            return False
    if same_process():
        try:
            os.killpg(owner["pid"], signal.SIGINT)
            deadline = time.monotonic()+35
            while same_process() and time.monotonic() < deadline:
                time.sleep(.1)
            if same_process():
                os.killpg(owner["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass


def collect(case, profile, folder, domain, timeout):
    import scenarios
    import rclpy
    from std_msgs.msg import String
    from rclpy.qos import qos_profile_sensor_data
    import numpy as np

    os.environ["ROS_DOMAIN_ID"] = str(domain)
    env = agent_environment(case, profile, domain)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"case.json").write_text(json.dumps(case, indent=2))
    cmd = [str(ROOT/".venv-seg/bin/python"), "-u", "-m", "agent.run", "--start-sim", "--motion", "real",
           "--scenario", case["scenario"], "--difficulty", case["difficulty"], "--seed", str(case["seed"]),
           "--instruction", case["instruction"], "--out", str(folder/"agent"), "--live-dir", ""]
    if profile in {"baseline", "deduplicated"}:
        cmd.append("--gui")
    script = case.get("instructions")
    if script is None and case.get("answers"):
        script = [{"request":case["instruction"], "answers":case["answers"]}]
    if script is not None:
        command_index = cmd.index("--instruction")
        del cmd[command_index:command_index+2]
        script_path = folder/"user_requests.json"
        script_path.write_text(json.dumps(script, indent=2))
        cmd += ["--instructions-file", str(script_path.resolve())]
    if case.get("setup"):
        setup_path = folder/"scene_setup.json"
        setup_path.write_text(json.dumps(case["setup"], indent=2))
        cmd += ["--evaluation-setup", str(setup_path.resolve())]
    # No dashboard process is launched: all profiles share that condition.
    (folder/"command.json").write_text(json.dumps({"argv": cmd, "overrides": PROFILES[profile], "domain": domain}, indent=2))
    rclpy.init()
    node = rclpy.create_node("pipeline_evaluator")
    samples, safety = [], []
    def truth(msg):
        samples.append({"wall_s": time.monotonic(), **json.loads(msg.data)})
    node.create_subscription(String, "sim/ground_truth/objects", truth, qos_profile_sensor_data)
    node.create_subscription(String, "safety/events", lambda m: safety.append(m.data), 10)
    process = None
    timed_out = False
    wall_start = time.monotonic()
    try:
        # A dedicated domain must be empty before a fresh experiment is started.
        discovery_end = time.monotonic()+1.
        while time.monotonic() < discovery_end:
            rclpy.spin_once(node, timeout_sec=.1)
        if samples or node.get_publishers_info_by_topic("joint_states"):
            raise RuntimeError(f"ROS domain {domain} is already in use; choose an unused --domain")
        with (folder/"console.log").open("w") as console:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=console, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            while process.poll() is None:
                rclpy.spin_once(node, timeout_sec=.1)
                if time.monotonic()-wall_start > timeout:
                    timed_out = True
                    process.send_signal(signal.SIGINT)  # agent finally stops only its own simulator
                    try:
                        process.wait(45)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(45)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        stop_owned_sim(folder)
        node.destroy_node()
        rclpy.try_shutdown()
    (folder/"truth.jsonl").write_text("".join(json.dumps(s)+"\n" for s in samples))
    events = session_events(folder/"agent")
    results = [json.loads(p.read_text()) for p in sorted((folder/"agent").glob("[0-9][0-9]/result.json"))]
    expected_instructions = len(script) if script is not None else 1
    status = ("COMPLETED" if len(results) == expected_instructions and all(r["status"] == "COMPLETED" for r in results)
              else next((r["status"] for r in results if r["status"] != "COMPLETED"), "INCOMPLETE_SESSION"))
    result = {"status":status if results else ("TIMEOUT" if timed_out else "ERROR")}
    starts = [e["monotonic_s"] for e in events if e.get("tool") == "execute_start"]
    before = [s for s in samples if not starts or s["wall_s"] <= min(starts)]
    initial = before[-1]["objects"] if before else {}
    final = samples[-1]["objects"] if samples else {}
    recent = [s for s in samples if samples[-1]["sim_time"]-.5 <= s["sim_time"]] if samples else []
    stable = (len(recent) >= 2 and recent[-1]["sim_time"]-recent[0]["sim_time"] >= .2
              and all(k in s["objects"] and np.linalg.norm(np.asarray(s["objects"][k]["position"])-v["position"]) < .005
                      for k,v in final.items() for s in recent))
    scenario = apply_setup(scenarios.generate(case["scenario"], case["difficulty"], case["seed"]), case.get("setup", {}))
    (folder/"placement_labels.json").write_text(json.dumps(placement_labels(events, samples, scenario), indent=2))
    action_start = min(starts) if starts else min((e["monotonic_s"] for e in events), default=float("inf"))
    moved_during = {k for s in samples if s["wall_s"] >= action_start for k,v in s["objects"].items()
                    if k in initial and np.linalg.norm(np.asarray(v["position"])-initial[k]["position"]) > .025}
    scoring = score(case, scenario, initial, final, result, stable=stable, moved_during=moved_during)
    if timed_out or not results:
        scoring = {"scored": False, "reason": result["status"]}
    row = {"case_id": case["id"], "profile": profile, "score": scoring, "timing": timing_summary(events),
           "startup_and_run_s": time.monotonic()-wall_start, "safety_events": safety, "folder": str(folder)}
    row["instruction_results"] = [{"request":r.get("request"), "status":r["status"]} for r in results]
    row["view_requirement"] = case.get("view_requirement")
    row["source_identity_checks"] = [{k:e.get(k) for k in ("frame_id", "object_id", "expected", "top", "ok", "crop")}
                                     for e in events if e.get("kind")=="identity_check" and e.get("role")=="source"]
    row["source_identity_false_rejections"] = None  # classify from the saved crop, not the selected target alone
    row["timing"]["per_requested_object_s"] = row["timing"]["total_s"]/len(case["objects"]) if case["objects"] else None
    (folder/"evaluation.json").write_text(json.dumps(row, indent=2))
    return row


def collect_isolated(case, profile, folder, domain, timeout):
    """Each case/profile owns a fresh ROS context and process; do not reuse DDS state between runs."""
    folder = folder.resolve()
    folder.parent.mkdir(parents=True, exist_ok=True)
    request = folder.parent/(folder.name+".request.json")
    request.write_text(json.dumps(case, indent=2))
    cmd = [str(ROOT/".venv-seg/bin/python"), "-u", str(Path(__file__).resolve()),
           "--collect-one", str(request), "--profiles", profile, "--out", str(folder),
           "--domain", str(domain), "--timeout", str(timeout)]
    with (folder.parent/(folder.name+".collector.log")).open("w") as output:
        worker = subprocess.Popen(cmd, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = worker.wait()
        finally:
            if worker.poll() is None:
                worker.send_signal(signal.SIGINT)
                try:
                    worker.wait(80)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
                stop_owned_sim(folder)
    if code:
        raise RuntimeError(f"collector exited {code}; see {folder.parent/(folder.name+'.collector.log')}")
    return json.loads((folder/"evaluation.json").read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, help="write the fixed test set without loading models or ROS")
    ap.add_argument("--suite", choices=["core", "regressions", "all"], default="core", help="default remains 63 core cases; --case can select from either suite")
    ap.add_argument("--collect-one", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--run", action="store_true", help="execute live simulator experiments")
    ap.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=["baseline", "deduplicated", "headless"])
    ap.add_argument("--case", action="append", help="case ID to run (repeat for a subset)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=2400, help="wall seconds per case including startup")
    ap.add_argument("--domain", type=int, default=88, help="dedicated ROS domain; leave 42 for interactive use")
    ap.add_argument("--out", type=Path, default=ROOT/"out/pipeline_benchmark")
    ap.add_argument("--summarize", nargs="+", type=Path, help="existing evaluation folders or JSONL event logs")
    a = ap.parse_args()
    if a.manifest:
        a.manifest.parent.mkdir(parents=True, exist_ok=True)
        a.manifest.write_text(json.dumps({"schema": 2, "default_suite":"core", "profiles": PROFILES, "cases": cases("all")}, indent=2)+"\n")
        print(f"Wrote {len(cases('all'))} cases (63 core + 8 regression) to {a.manifest}")
    if a.summarize:
        reports = []
        for path in a.summarize:
            if path.suffix == ".jsonl":
                reports.append({"path": str(path), **timing_summary(read_events(path))})
            else:
                reports.extend(json.loads(p.read_text()) for p in path.rglob("evaluation.json"))
        payload = compare(reports) if reports and "profile" in reports[0] else reports
        print(json.dumps(payload, indent=2))
    if not a.run and not a.collect_one:
        return
    if not 0 <= a.domain <= 232 or a.repeat < 1:
        ap.error("invalid domain or repeat")
    # Fail before loading models if the GPU cannot be reached.
    check = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True)
    if check.returncode:
        raise SystemExit("No accessible NVIDIA driver; live benchmark was not started.")
    if a.collect_one:
        if len(a.profiles) != 1:
            ap.error("an isolated collector requires exactly one profile")
        collect(json.loads(a.collect_one.read_text()), a.profiles[0], a.out.resolve(), a.domain, a.timeout)
        return
    selected = [c for c in cases("all" if a.case else a.suite) if not a.case or c["id"] in a.case]
    if not selected or (a.case and set(a.case)-{c["id"] for c in selected}):
        ap.error("unknown case ID")
    rows = []
    a.out.mkdir(parents=True, exist_ok=True)
    for repeat in range(a.repeat):
        for c in selected:
            for profile in a.profiles:
                folder = a.out/profile/f"{c['id']}_r{repeat+1}"
                if folder.exists():
                    raise SystemExit(f"Refusing to overwrite existing experiment: {folder}")
                row = collect_isolated(c, profile, folder, a.domain, a.timeout)
                row["case_id"] += f"_r{repeat+1}"
                (folder/"evaluation.json").write_text(json.dumps(row, indent=2))
                rows.append(row)
                (a.out/"comparison.json").write_text(json.dumps(compare(rows), indent=2))
                print(profile, row["case_id"], row["score"], flush=True)


if __name__ == "__main__":
    main()
