"""THE place to configure a robot session. Edit the values below, then run:

    ./launch/agent.sh

That starts the complete live system: Gemma -> YOLO -> Isaac Sim (this scene) -> ROS 2 cameras and joints ->
supervisor and workers -> interactive instruction prompt, with every agent handoff traced in the terminal.
The values are validated at startup and summarised before anything is loaded.

Safety does not depend on these settings. Request authorization (agent/authority.py), collision checking
(control/collision.py), the joint-limit safety layer (safety/, enforced inside the simulator), the executor's
reach checks and goal verification apply in every configuration. MOTION only chooses whether planned motions are
sent at all.

Lower-level constants (thresholds, tolerances, budgets) are in config/. Gemma's loading settings are in vlm/config.py.
"""

from __future__ import annotations

from pathlib import Path

from config import models as _models

ROOT = Path(__file__).resolve().parents[1]

SCENARIO = "default"
TARGET = "isaac_sim"
MOTION = "real"
TRACE = True

# ---------------------------------------------------------------------------------------------- scene
SCENARIO = "laboratory"      # any scenario in scenarios/ (python -c "import scenarios; print(scenarios.available())"):
                             # "default" (frozen cube scene), "laboratory", "ap_occlusion", "ap_categories", ...
DIFFICULTY = "easy"          # laboratory: "easy" | "medium" | "hard"
SEED = 42                    # laboratory layout seed (42 = the Experiment 1 scene)

# ----------------------------------------------------------------------------------------- execution
TARGET = "isaac_sim"         # the only supported target. Physical-robot execution is NOT part of this launcher.
START_SIMULATOR = True       # True: the agent starts Isaac Sim itself after loading the models (one command)
MOTION = "real"              # "real": planned motions are executed in Isaac Sim through the existing executor
                             # "dry_run": planned and logged, never sent | "blocked": nothing is sent
ROS_DOMAIN_ID = 42           # keeps this session apart from other ROS 2 sessions on the machine

# -------------------------------------------------------------------------------------------- models
GEMMA_MODEL = str(ROOT / "models" / "gemma-4-E2B-it")   # local Gemma 4 E2B-it (4-bit, see vlm/config.py)
YOLO_WEIGHTS = str(_models.YOLO_WEIGHTS)                # YOLO11s-seg, class-agnostic instance masks
YOLO_IMAGE_SIZE = _models.DETECTOR_IMGSZ
YOLO_CONFIDENCE = _models.DETECTOR_CONF

# ----------------------------------------------------------------------------------- windows and I/O
ISAAC_WINDOW = True          # show the Isaac Sim window (False: headless, faster)
DEMO_VIEWER = True           # the agent viewer: live camera, detections, supervisor state, logs
CAMERA_WINDOWS = True        # raw wrist and table RGB-D windows
ASK = "terminal"             # who answers clarification questions: "terminal" (you) | "scripted" (always cancel)
MAX_GEMMA_DECISIONS = 3      # per instruction: how often Gemma may decide again after its first plan (failure,
                             # uncertain result, goal not shown); 0 = stop at the first problem

# ----------------------------------------------------------------------------------- tracing and output
TRACE = True                 # print every agent handoff: request, delegation input, worker output, review, failures
OUTPUT_DIR = "out/demo"      # one folder per session: <OUTPUT_DIR>/<date>/live_<time>/<nn>/ (events.jsonl = full trace)
LIVE_DIR = "out/demo/live"   # files the viewer reads
LOG_DIR = "out/logs"         # viewer / camera-window logs

# =========================================================================================================
# Validation and summary (no need to edit below this line)

DIFFICULTIES = ("easy", "medium", "hard")
MOTIONS = ("real", "dry_run", "blocked")
TARGETS = ("isaac_sim",)


def _scenario_ok(name: str) -> bool:
    import scenarios
    return name in scenarios.available()  # every scenario plugged in under scenarios/


def validate(values: dict | None = None) -> list[str]:
    """Problems with the settings (empty = valid). `values` overrides the module values (tests)."""
    v = {k: globals()[k] for k in ("SCENARIO", "DIFFICULTY", "SEED", "TARGET", "START_SIMULATOR", "MOTION", "ROS_DOMAIN_ID",
                                   "GEMMA_MODEL", "YOLO_WEIGHTS", "YOLO_IMAGE_SIZE", "YOLO_CONFIDENCE", "ISAAC_WINDOW",
                                   "DEMO_VIEWER", "CAMERA_WINDOWS", "ASK", "MAX_GEMMA_DECISIONS", "TRACE", "OUTPUT_DIR",
                                   "LIVE_DIR", "LOG_DIR")}
    v.update(values or {})
    errors = []
    if v["TARGET"] not in TARGETS:
        errors.append(f"TARGET={v['TARGET']!r}: only 'isaac_sim' is supported. Physical-robot execution is not part of "
                      "this launcher and is never started from it.")
    if not _scenario_ok(v["SCENARIO"]):
        import scenarios
        errors.append(f"SCENARIO={v['SCENARIO']!r}: not a scenario in scenarios/ (available: {', '.join(scenarios.available())})")
    if v["DIFFICULTY"] not in DIFFICULTIES:
        errors.append(f"DIFFICULTY={v['DIFFICULTY']!r}: one of {DIFFICULTIES}")
    if not isinstance(v["SEED"], int) or isinstance(v["SEED"], bool) or v["SEED"] < 0:
        errors.append(f"SEED={v['SEED']!r}: a non-negative integer")
    if v["MOTION"] not in MOTIONS:
        errors.append(f"MOTION={v['MOTION']!r}: one of {MOTIONS}")
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
    if v["ASK"] not in ("terminal", "scripted"):
        errors.append(f"ASK={v['ASK']!r}: 'terminal' or 'scripted'")
    m = v["MAX_GEMMA_DECISIONS"]
    if not isinstance(m, int) or isinstance(m, bool) or not 0 <= m <= 10:
        errors.append(f"MAX_GEMMA_DECISIONS={m!r}: an integer 0-10")
    for k in ("START_SIMULATOR", "ISAAC_WINDOW", "DEMO_VIEWER", "CAMERA_WINDOWS", "TRACE"):
        if not isinstance(v[k], bool):
            errors.append(f"{k}={v[k]!r}: True or False")
    for k in ("OUTPUT_DIR", "LIVE_DIR", "LOG_DIR"):
        if not isinstance(v[k], str) or not v[k]:
            errors.append(f"{k}={v[k]!r}: a non-empty path")
    return errors


def summary() -> str:
    import scenarios
    k = scenarios.kind(SCENARIO)
    scene = f"{SCENARIO} ({DIFFICULTY}, seed {SEED})" if k == "procedural" else f"{SCENARIO} ({k} scene)"
    motion = {"real": "REAL - planned motions are executed in Isaac Sim (authorization, collision checks, joint limits apply)",
              "dry_run": "DRY RUN - planned and logged, never sent", "blocked": "BLOCKED - nothing is sent to the robot"}[MOTION]
    windows = ", ".join(n for n, on in (("Isaac Sim window", ISAAC_WINDOW), ("demo viewer", DEMO_VIEWER),
                                        ("camera windows", CAMERA_WINDOWS)) if on) or "none (headless)"
    return "\n".join([
        "[settings] agent/settings.py",
        f"[settings]   scene      {scene}",
        f"[settings]   target     {TARGET} (ROS_DOMAIN_ID {ROS_DOMAIN_ID}; simulator started by the agent: {START_SIMULATOR})",
        f"[settings]   motion     {motion}",
        f"[settings]   models     Gemma {Path(GEMMA_MODEL).name} | YOLO {Path(YOLO_WEIGHTS).parent.parent.name} "
        f"(imgsz {YOLO_IMAGE_SIZE}, conf {YOLO_CONFIDENCE})",
        f"[settings]   windows    {windows}",
        f"[settings]   gemma      decides the plan, and again up to {MAX_GEMMA_DECISIONS} time(s) per instruction",
        f"[settings]   tracing    {'every agent handoff in the terminal' if TRACE else 'progress lines only'}; full trace in "
        f"{OUTPUT_DIR}/<date>/live_<time>/<nn>/events.jsonl",
    ])


def run_defaults() -> dict:
    """agent.run argument defaults taken from these settings (agent.run --settings)."""
    import time
    return {"scenario": SCENARIO, "difficulty": DIFFICULTY, "seed": SEED, "motion": MOTION,
            "start_sim": START_SIMULATOR, "gui": ISAAC_WINDOW, "weights": YOLO_WEIGHTS, "imgsz": YOLO_IMAGE_SIZE,
            "conf": YOLO_CONFIDENCE, "gemma": GEMMA_MODEL, "ask": ASK, "max_decisions": MAX_GEMMA_DECISIONS, "trace": TRACE, "live_dir": LIVE_DIR,
            "out": str(ROOT / OUTPUT_DIR / time.strftime("%Y-%m-%d") / time.strftime("live_%H%M%S")), "target": TARGET}


def shell() -> str:
    """Values the launcher needs before Python starts the agent (eval'd by launch/agent.sh)."""
    flag = {True: "1", False: "0"}  # the window switches can be overridden for one run: DEMO_VIEWER=0 ./launch/agent.sh
    return "\n".join([f"ROS_DOMAIN_ID={ROS_DOMAIN_ID}", f": ${{DEMO_VIEWER:={flag[DEMO_VIEWER]}}}",
                      f": ${{CAMERA_WINDOWS:={flag[CAMERA_WINDOWS]}}}", f"LIVE_DIR={LIVE_DIR}", f"LOG_DIR={LOG_DIR}"])


if __name__ == "__main__":  # python -m agent.settings [--shell]: validate, then print the summary (or shell values)
    import sys
    problems = validate()
    if problems:
        print("[settings] agent/settings.py has problems:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        raise SystemExit(1)
    print(shell() if "--shell" in sys.argv else summary())
