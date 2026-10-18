# Epson VT6L manipulation project (Isaac Sim 6.1 + ROS 2 Jazzy)

Current state: restructured codebase awaiting manual acceptance — read docs/validation/FINAL_MAIN_READINESS.md first,
then docs/architecture/MODULE_GUIDE.md (where things are) and docs/DEVELOPER_GUIDE.md (commands).

- Overall audit, milestones and results: docs/AUDIT_AND_PLAN.md
- Frozen manipulation baseline: docs/baselines/manipulation_A_2026-10-14/ — do NOT modify any file
  listed in its MANIFEST.sha256 (check with `grep -v '^#' docs/baselines/manipulation_A_2026-10-14/MANIFEST.sha256 | sha256sum -c`).
- Offline tests: `/usr/bin/python3 -m pytest` (all must pass).
  Full suite incl. agent tests: `env -u PYTHONPATH -u AMENT_PREFIX_PATH .venv-seg/bin/python -m pytest` (all must pass).
- GPU is 12 GB: never run two Isaac Sim instances, load the VLM before heavy rendering, never train while the sim runs.
- Test in ROS_DOMAIN_ID=42 so the user's own session (default domain) is not disturbed; ask before stopping it.
