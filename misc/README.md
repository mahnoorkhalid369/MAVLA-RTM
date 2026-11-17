# misc: archived code (not used by the running system)

Nothing in agent/, sim/, perception/, control/, scenarios/ or the launchers imports these. They are kept for the
record (how the detector was trained, earlier baselines) and to be re-run if needed.

| Folder | What it was |
|---|---|
| `datagen/` | synthetic training-data generator (Isaac Sim scenes -> COCO / YOLO datasets) |
| `seg/` | YOLO11-seg / Mask R-CNN training and evaluation (produced `out/seg/runs/yolo11s_seg_r1`, the detector in use) |
| `experiments/` | early known-pose pick-and-place baselines (Baseline B) and the payload study |
| `evaluation/historical/` | demo rehearsal, latency and earlier checks |
| `launch/legacy/` | start scripts of the legacy ROS 2 node stack |
| `scripts/` | early camera / robot milestone validation |
| `tests/` | the tests of the code above (not collected: pytest runs `tests/` only) |
| `epson-robot-ros2/` | external Epson clone (not in git; the VT6L model in `assets/` is self-contained) |

Running one again: these files still assume they sit at the repository root (their `ROOT` and imports such as
`from seg.data import ...`). Move the folder back to the root first, or run from `misc/` with
`PYTHONPATH=misc:. python -m seg.train_yolo ...` and check its paths.
