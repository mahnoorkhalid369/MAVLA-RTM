"""THE place to configure a robot session. Edit the values below, then run ./launch/agent.sh

That starts the live system: Gemma -> YOLO -> Isaac Sim (this scene) -> ROS 2 cameras and joints -> the master and its
workers -> the instruction prompt. The values are checked at startup and summarised before anything loads.
Safety does not depend on these settings: collision checking (control/collision.py), the joint-limit safety layer
inside the simulator (safety/) and the sensor checks apply in every configuration; MOTION only chooses whether planned
motions are sent at all. Limits of one instruction are in agent/loop.py; Gemma's loading settings in vlm/config.py.

    python -m agent.settings           # check and print the summary
    python -m agent.settings --shell   # the values launch/agent.sh needs
"""

from __future__ import annotations

from pathlib import Path

from config import models as _models

ROOT = Path(__file__).resolve().parents[2]  # this version's folder (v1/ or v2/)

# scene
SCENARIO = "workcell"        # any scenario in scenarios/ (python -c "import scenarios; print(scenarios.available())")
SCENARIO = "snack_kitchen"

DIFFICULTY = "easy"          # separated milk/water/snack scene; "easy" | "medium" | "hard"
SEED = 42                    # procedural scenes: layout seed

# execution
TARGET = "isaac_sim"         # the only supported target; physical-robot execution is not part of this launcher
START_SIMULATOR = True       # the agent starts Isaac Sim itself after loading the models
MOTION = "real"              # "real" | "dry_run" (planned and logged, never sent) | "blocked" (nothing is sent)
ROS_DOMAIN_ID = 42           # keeps this session apart from other ROS 2 sessions on the machine

# models
GEMMA_MODEL = str(ROOT / "models" / "gemma-4-E4B-it")
YOLO_WEIGHTS = str(_models.YOLO_WEIGHTS)
YOLO_IMAGE_SIZE = _models.DETECTOR_IMGSZ
YOLO_CONFIDENCE = _models.DETECTOR_CONF
IDENTITY_CHECK = True        # SigLIP 2 (models/siglip2-base-patch16-224, ~0.9 GB GPU): the wrist close-up of the chosen
                             # object / destination is checked against the catalogue names before descent and release

# windows and output
ISAAC_WINDOW = True          # show the Isaac Sim window (False: headless)
DASHBOARD = True             # one window: cameras, YOLO masks and ids, task, logs (agent/dashboard.py)
GPU_MONITOR = False          # ECE 565 plugin (gpumon/): records GPU / RAM / latency in its own process
OUTPUT_DIR = "out/demo"      # <OUTPUT_DIR>/<date>/live_<time>/<nn>/ per instruction
LIVE_DIR = "out/demo/live"   # files the dashboard reads
LOG_DIR = "out/logs"         # dashboard / recorder logs

NAMES = ("SCENARIO", "DIFFICULTY", "SEED", "TARGET", "START_SIMULATOR", "MOTION", "ROS_DOMAIN_ID", "GEMMA_MODEL",
         "YOLO_WEIGHTS", "YOLO_IMAGE_SIZE", "YOLO_CONFIDENCE", "IDENTITY_CHECK", "ISAAC_WINDOW", "DASHBOARD", "GPU_MONITOR", "OUTPUT_DIR",
         "LIVE_DIR", "LOG_DIR")


def validate(values: dict | None = None) -> list[str]:
    import scenarios
    v = {k: globals()[k] for k in NAMES}
    v.update(values or {})
    errors = []
    if v["TARGET"] != "isaac_sim":
        errors.append(f"TARGET={v['TARGET']!r}: only 'isaac_sim' is supported")
    if v["SCENARIO"] not in scenarios.available():
        errors.append(f"SCENARIO={v['SCENARIO']!r}: not in scenarios/ ({', '.join(scenarios.available())})")
    if v["DIFFICULTY"] not in ("easy", "medium", "hard"):
        errors.append(f"DIFFICULTY={v['DIFFICULTY']!r}: easy, medium or hard")
    if not isinstance(v["SEED"], int) or isinstance(v["SEED"], bool) or v["SEED"] < 0:
        errors.append(f"SEED={v['SEED']!r}: a non-negative integer")
    if v["MOTION"] not in ("real", "dry_run", "blocked"):
        errors.append(f"MOTION={v['MOTION']!r}: real, dry_run or blocked")
    if not isinstance(v["ROS_DOMAIN_ID"], int) or not 0 <= v["ROS_DOMAIN_ID"] <= 232:
        errors.append(f"ROS_DOMAIN_ID={v['ROS_DOMAIN_ID']!r}: an integer 0-232")
    if not (Path(v["GEMMA_MODEL"]) / "config.json").exists():
        errors.append(f"GEMMA_MODEL={v['GEMMA_MODEL']!r}: no model there (config.json missing)")
    if not Path(v["YOLO_WEIGHTS"]).exists():
        errors.append(f"YOLO_WEIGHTS={v['YOLO_WEIGHTS']!r}: file not found")
    if not isinstance(v["YOLO_IMAGE_SIZE"], int) or v["YOLO_IMAGE_SIZE"] <= 0 or v["YOLO_IMAGE_SIZE"] % 32:
        errors.append(f"YOLO_IMAGE_SIZE={v['YOLO_IMAGE_SIZE']!r}: a positive multiple of 32")
    if not isinstance(v["YOLO_CONFIDENCE"], (int, float)) or not 0.0 < v["YOLO_CONFIDENCE"] < 1.0:
        errors.append(f"YOLO_CONFIDENCE={v['YOLO_CONFIDENCE']!r}: between 0 and 1")
    for k in ("START_SIMULATOR", "ISAAC_WINDOW", "DASHBOARD", "GPU_MONITOR", "IDENTITY_CHECK"):
        if not isinstance(v[k], bool):
            errors.append(f"{k}={v[k]!r}: True or False")
    for k in ("OUTPUT_DIR", "LIVE_DIR", "LOG_DIR"):
        if not isinstance(v[k], str) or not v[k]:
            errors.append(f"{k}={v[k]!r}: a non-empty path")
    return errors


def summary() -> str:
    motion = {"real": "REAL - planned motions are executed in Isaac Sim", "dry_run": "DRY RUN - planned and logged, never sent",
              "blocked": "BLOCKED - nothing is sent to the robot"}[MOTION]
    windows = ", ".join(n for n, on in (("Isaac Sim window", ISAAC_WINDOW), ("dashboard", DASHBOARD)) if on) or "none"
    return "\n".join([
        "[settings] agent/settings.py (v2: master + workers)",
        f"[settings]   scene      {SCENARIO} ({DIFFICULTY}, seed {SEED})",
        f"[settings]   target     {TARGET} (ROS_DOMAIN_ID {ROS_DOMAIN_ID}; simulator started by the agent: {START_SIMULATOR})",
        f"[settings]   motion     {motion}",
        f"[settings]   models     Gemma {Path(GEMMA_MODEL).name} | YOLO {Path(YOLO_WEIGHTS).parent.parent.name} "
        f"(imgsz {YOLO_IMAGE_SIZE}, conf {YOLO_CONFIDENCE})" + (" | SigLIP 2 identity check" if IDENTITY_CHECK else ""),
        f"[settings]   windows    {windows}",
        f"[settings]   gpu        {'gpumon records this session' if GPU_MONITOR else 'not recorded'}",
        f"[settings]   output     {OUTPUT_DIR}/<date>/live_<time>/<nn>/ (events.jsonl, result.json, images/)",
    ])


def run_defaults() -> dict:
    import time
    return {"scenario": SCENARIO, "difficulty": DIFFICULTY, "seed": SEED, "motion": MOTION, "start_sim": START_SIMULATOR,
            "gui": ISAAC_WINDOW, "weights": YOLO_WEIGHTS, "imgsz": YOLO_IMAGE_SIZE, "conf": YOLO_CONFIDENCE,
            "gemma": GEMMA_MODEL, "live_dir": LIVE_DIR, "identity_check": IDENTITY_CHECK,
            "out": str(ROOT / OUTPUT_DIR / time.strftime("%Y-%m-%d") / time.strftime("live_%H%M%S"))}


def shell() -> str:
    import shlex

    import scenarios
    flag = {True: "1", False: "0"}  # the window switches can be overridden for one run: DASHBOARD=0 ./launch/agent.sh
    return "\n".join([f"ROS_DOMAIN_ID={ROS_DOMAIN_ID}", f": ${{DASHBOARD:={flag[DASHBOARD]}}}",
                      f": ${{GPU_MONITOR:={flag[GPU_MONITOR]}}}", f"OUTPUT_DIR={OUTPUT_DIR}", f"LIVE_DIR={LIVE_DIR}",
                      f"LOG_DIR={LOG_DIR}", f"export TABLE_CAMERA={scenarios.table_camera(SCENARIO)}",
                      f"export WORKCELL_LAYOUT={shlex.quote(scenarios.layout(SCENARIO))}"])


if __name__ == "__main__":
    import sys
    problems = validate()
    if problems:
        print("[settings] agent/settings.py has problems:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        raise SystemExit(1)
    print(shell() if "--shell" in sys.argv else summary())
