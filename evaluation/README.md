# evaluation/: experiments, diagnostics and scoring

**Rules for everything in this folder:**

- Simulator ground truth is allowed **here only**. The agent never reads it.
- Nothing here changes agent behaviour by replacing functions at runtime. Experiment switches are
  `agent/controls.py` `ExecutionControls`, and `tests/test_structure.py` enforces this.
- Outputs go under `out/` (not in git).

## Active tools (current agent and laboratory)

| Script | Purpose | Environment |
|---|---|---|
| `lab_exp1.py` | Laboratory Experiment 1 protocol harness (phases, preflights, stop conditions) | `.venv-seg` + ROS |
| `lab_exp1_agent.py` | the agent process of Experiment 1 (`agent.run.main` with blocked motion, disabled capabilities, no replanning) | `.venv-seg` + ROS |
| `lab_exp1_score.py` | scores Experiment 1 against the scene metadata and ground truth | system Python |
| `lab_probe.py`, `truth_snapshot.py` | read joint states and `/sim/ground_truth/objects` over ROS | `.venv-seg` + ROS |
| `lab_capture_frames.py` | captures RGB-D + instance masks of laboratory scenes for offline perception work (→ `out/lab_seg_frames`) | `.venv` (Isaac) |
| `seg_diagnosis.py`, `seg_generalization.py` | segmentation approaches A / B / B2 / C on saved frames | `.venv-seg` |
| `validate_scenarios.py` | scenario-library validation in Isaac (spawn, settling, reload, cameras, design checks via `python -m scenarios.checks`) | `.venv` (Isaac) |
| `scenario_design.py` | design check of the JSON test scenes (`scenarios/scenes/`) | system Python |
| `planner_check.py` | real-Gemma planner check on fixed instructions (no simulator) | `.venv-seg` (GPU) |
| `demo_live_driver.py` | drives `launch/start_demo.sh` as an operator for an end-to-end demo test | `.venv-seg` + ROS |

## Historical (earlier phases; still runnable, not maintained; they reproduce reported results)

| Script | Phase / report |
|---|---|
| `demo_rehearsal.py`, `demo_summary.py`, `demo_latency.py` | demo rehearsal, default scene (`docs/perception_redesign/DEMO_REPORT.md`) |
| `demo_describe_bench.py`, `demo_vlm_check.py`, `demo_plan_dryrun.py`, `demo_home_check.py` | demo-phase benchmarks and checks |
| `dual_camera_check.py`, `inspect_check.py`, `active_summary.py` | active-perception checkpoints (`docs/architecture/ACTIVE_PERCEPTION.md`) |
| `grasp_feasibility_map.py` | grasp-feasibility map (frozen baseline `docs/baselines/manipulation_A_2026-10-14/`) |
