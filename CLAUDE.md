# Epson VT6L manipulation project (Isaac Sim 6.1 + ROS 2 Jazzy)

Current state: restructured and simplified codebase awaiting manual acceptance — start with README.md, then
docs/validation/FINAL_MAIN_READINESS.md, docs/architecture/MODULE_GUIDE.md and docs/DEVELOPER_GUIDE.md.
Entry points: launch/agent.sh (motion blocked unless --motion real), launch/sim.sh, launch/tests.sh, launch/evaluate.sh.

- Overall audit, milestones and results: docs/history/AUDIT_AND_PLAN.md
- Frozen manipulation baseline: docs/baselines/manipulation_A_2026-10-14/ — do NOT modify any file
  listed in its MANIFEST.sha256 (check with `grep -v '^#' docs/baselines/manipulation_A_2026-10-14/MANIFEST.sha256 | sha256sum -c`).
- Offline tests: `./launch/tests.sh` (frozen check + `.venv-seg` full suite + `/usr/bin/python3` suite; all must pass).
  Tests are in tests/unit/ and tests/integration/; research experiments in evaluation/ (not tests).
- GPU is 12 GB: never run two Isaac Sim instances, load the VLM before heavy rendering, never train while the sim runs.
- Test in ROS_DOMAIN_ID=42 so the user's own session (default domain) is not disturbed; ask before stopping it.
