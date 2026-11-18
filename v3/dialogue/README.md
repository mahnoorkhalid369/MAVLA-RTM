# v3/dialogue: the agents talk, the code measures

A parallel direction beside the frozen v3 (`../FROZEN.sha256` lists the frozen files; none of them is edited here).
Gemma decides and the agents talk in plain sentences. Code only measures, checks and moves.

## Run

```bash
cd v3/dialogue
./launch.sh                                    # type instructions; an empty line quits
./launch.sh --instructions tests/sample.json   # run a list of requests
HEADLESS=1 ./launch.sh                         # without the Isaac Sim window (it opens by default)
```

Logs and pictures of every request: `v3/out/v3d/<date>/run_<time>/<nn>/` (`events.jsonl`, `images/`), and `results.json` per session.
The frozen v3 still starts with `cd v3 && ./launch.sh`. One simulator at a time.

## Files

| File | What it holds |
|---|---|
| `prompts.py` | Every prompt of the master and the workers |
| `agents.py` | The shared conversation and who speaks when |
| `arm.py` | The robot: measuring, picture point to millimetres, fit and collision checks, motion. It imports the frozen `../robot.py` |
| `v3_config.py` | The arm speeds of this version: 1.5 times v2's (45 deg/s, 0.12 m/s, lowering 0.0375 m/s) |
| `run.py`, `launch.sh` | Start-up and logging |
| `screen.py` | The live screen: the table camera with outlines, the wrist camera beside it, the logs below. Key `c` swaps the two cameras. It uses v2's dashboard as a library |
| `tests/test_dialogue.py` | Scripted tests with a fake Gemma and a fake robot (no GPU) |
| `tests/gemma_only.py` | The real Gemma on a saved table picture, without the simulator |
| `tests/sample.json` | Sample requests for `--instructions` |

## One transfer, step by step

1. **Master** reads the request, the table picture and the measured facts, plans all objects, and sends one message at a time.
2. **Manipulation worker** answers in words: `pick up: ...` and `put down: ...` (a thing) or `on the table by: ...` (a thing nearby).
3. The worker **points** at the object and at the destination in the table picture, each in a short call of its own.
4. **Robot (code)** finds the measured object under each box and shows a close-up of the destination.
5. The worker describes the place in one sentence and points at it on the close-up.
6. The robot turns the point into millimetres with the depth camera, moves it to the nearest spot where the object fits and the arm can put it down safely, draws that spot as a white outline, and says why it moved it.
7. The worker answers `fine: ...` or `move it: ...` (at most two corrections).
8. The robot carries the object over and reports to the master in a sentence.
9. At the end the **verification worker** describes the table, and the master decides: done, try again, ask the user, or stop.

## The look after every release

The robot only knows where it let go of an object. Whether the object stands there is found out by looking again:
after every release, and whenever the master asks, the verification worker describes the place.

- It is not told what the team believes happened: it reads the user's words, the question and the measurements.
- It sees a plain close-up of the places the robot worked at in this request, not the whole table.
- The master treats that description, not the robot's report, as what really stands where.

Why, measured with Gemma alone on the last picture of 8 recorded runs (30 object cases, 11 of them outside the
tray; `out/v3d/experiments/verify_2026-11-18/`):

| What the checking worker is given | Right, of 30 | Run with two things beside the tray, of 4 |
|---|---|---|
| Whole conversation, whole-table picture (before) | 26 | 2 |
| No conversation, whole-table picture | 29 | 3 |
| No conversation, whole table with numbered outlines | 27 | 2 |
| No conversation, close-up of the tray only (now) | 30 | 4 |
| No conversation, close-up with outlines | 26 | 4 |
| No conversation, whole table and close-up together | 29 | 3 |

Known limits: an object lying on its side was not reported as fallen over, and things are sometimes misnamed.

## Tests

```bash
cd v3/dialogue
PYTHONPATH=. ../../v2/.venv-seg/bin/python tests/test_dialogue.py          # 10 scripted checks, no GPU
env -u AMENT_PREFIX_PATH PYTHONPATH=.:../../v2/src ../../v2/.venv-seg/bin/python tests/gemma_only.py \
    ../out/v3d/<date>/run_<time>/01 "Put the milk carton on the light blue serving tray, in a corner."
```

## Live results so far (2026-11-18, "place apple, water, milk and fruit jar on the blue tray")

Seven live runs; none ended as completed. In one run all four objects stood upright on the tray, after which the master
ordered an object nobody asked for. In the last run three were placed and the fruit cup slipped out of the gripper.

Known weak points:
- Gemma points at an area, not a spot: the code moved the pointed place by 34 to 222 mm, and the worker nearly always answers "fine".
- After a problem the master tends to report or ask the user instead of trying again.
- The first plan takes about 2 minutes, a four-object request 14 to 18 minutes.
- Round objects (apple, fruit cup) sometimes slip out of the gripper; this also happens in the frozen v3.
- Not yet run live: placing on the table beside an object, a question only, an absent object.
