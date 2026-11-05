# Open issues from the manual tests of 2026-10-26 (evening)

The user's manual tests in the work-cell scene, with Gemma 4 E4B (4-bit) and session `session_20261026_183003`
(`out/demo/2026-10-26/live_182936/01–05`), went through these instructions in order:

1. **pick the pink block**: completed.
2. **place it on the red tray**: the block went into the tray, but the robot reported failure.
3. **place red block in red tray**: failed; nothing moved.
4. (not shown; the red block was picked)
5. **place the block in the red tray**: the block was put into the **green** tray, and the robot reported failure.

The physical actions mostly worked. What failed is how the system keeps track of the world, checks results, and keeps
its plans tied to what the user asked. Gemma made some mistakes too, but the architecture let each mistake spread
instead of containing it. This document records every failure seen, its cause, and the part of the architecture
responsible.

**Status (later on 2026-10-26):** A1, A2, A4, B1, C1, D3, E3 and F1 are fixed (`ROBUSTNESS_FIXES_2026-10-26.md` section 10).
A3, B2, B3, C2, C3, D1, D2, D4, E1, E2, F2 and F3 are still open.
**2026-11-04:** D4 is fixed: a decomposition whose conditions need an object to change, but whose steps only sense, is
rejected with the reason and Gemma plans again (`check_progress`); conditions missing their arguments (`holding()`) are
rejected too, and are "unknown" (not a crash) if one reaches the goal check.

---

## A. The world model accepts unverified and wrong information

| # | Seen | Cause |
|---|---|---|
| A1 | After instruction 2, object #4 ("pink block") had the **red tray's** size and position (202 × 142 mm, 40 mm high, at −249, 711). | The placement check measured the tray instead of the block (see C1). Even though the check **failed**, the measurement was still written into the placed object's record (`verify_placement`). |
| A2 | The red tray existed **twice**: #4 (now named "pink block") and #15 ("pink tray"), at the same position. | Because of A1, two world objects describe one physical thing. Nothing checks that one place holds one object. |
| A3 | The real pink block, dropped into the tray, became a new object, #16 "pink gemstone". Its old record (#4) was never associated with it. | Nothing tracks an object **through** an action: once released, it is found again only by appearance and position. |
| A4 | Instruction 5 planned into the red tray, but the red tray already held the pink block. The planner found "no collision-free way to lower" and retried three times. | The world model has no notion of **contents**: what is in a container, and whether it has free space. "Occupied" appears only as a collision, retried as if transient. |

**Architectural point:** the world model should change only on verified evidence, keep one record per physical object,
follow an object through pick → hold → place, and know which object is inside which container.

## B. Object identity does not last across instructions

| # | Seen | Cause |
|---|---|---|
| B1 | Instruction 2's goal check took "pink block" to be **#11, a red cube** 934 mm from the tray. | Links between words and objects ("pink block" = #4) live only within one instruction. The goal check looked the words up again, in a frame taken **before** the place, where the pink block was in the gripper and excluded. Gemma picked the most pink-looking object left. |
| B2 | Gemma asked "Which object do you mean by 'it'?" although its context said "the robot is HOLDING object #4 (pink block)". | The held object reaches Gemma as a sentence of context, not as part of the request's meaning. |
| B3 | The same object has a world id and a different number in each camera picture. Messages mix them. | Two numbering systems. |

**Architectural point:** what a word refers to should be settled once, as a world id, and kept for the session until
evidence says otherwise. The goal should be stored in world ids as soon as they are known, not re-resolved from words.

## C. Verification checks "something near the target", not the object that was moved

| # | Seen | Cause |
|---|---|---|
| C1 | Instruction 2: "placement z −120.5 mm, out of tolerance". It had measured the tray's rim. | The check takes the measured object nearest the planned position. For a destination that is a container, that is the container. The `in` fix (skipping the container) does not cover `on`. |
| C2 | Goal conditions were evaluated on stale or wrong objects (B1), so a successful action read as failed. | The goal check re-resolves words instead of using what the action actually moved. |
| C3 | Instruction 2: "on(pink block, red tray)" can never be true for a block **inside** a tray. | The user said "on" and Gemma kept it literally. For a container, "on" (on the rim) and "in" are physically different, and the user meant "in". |

**Architectural point:** an action's result should be verified on **the object that action moved** (known by id from the
grasp), and checked against what the action did, not re-identified by proximity or by words.

## D. Plans can drift away from the user's goal

| # | Seen | Cause |
|---|---|---|
| D1 | Instruction 5: the user's goal was the red tray (Gemma wrote "pink tray"). After the red tray was blocked, the decomposition chose **#2, the green tray**. The block was placed there and the camera confirmed it. The original goal `in(#12, pink tray)` stayed unmet. | A decomposition may name different objects from the original goal, and the supervisor accepts it. |
| D2 | Instruction 3: the decomposition chose the light green tray (#2) as "the red tray". | The red tray was mislabelled in the world model (A1), so Gemma chose among wrong descriptions. |
| D3 | Instruction 3: the first plan was `place` with an **empty gripper**. Gemma's own reason assumed it still held a block. | The operations Gemma may choose are not filtered by the robot's state. `place` is offered when nothing is held, and `pick` when something is. The worker refuses afterwards. |
| D4 | Decompositions planned only `observe`, although the prompt asks for actions. | The planning worker accepts a decomposition that does not progress the goal. |

**Architectural point:** after the first plan, the objects in the goal are fixed. A plan or decomposition that changes
them must come back to the user, not run. The operations offered should match the robot's state: `place` only when
holding, `pick` only when empty.

## E. Recovery wanders until the budget ends

| # | Seen | Cause |
|---|---|---|
| E1 | In instructions 2, 3 and 5, after a failure, Gemma chose observe → describe → observe until the 3 changes of course were used up. | Recovery options are not tied to the failure. For "the goal check says not met", no option fits, so Gemma looks again. The no-progress question fires only when nothing at all changed, and new frames or new goal conditions count as change. |
| E2 | Instruction 2: the user answered "the task is done already". Gemma doubted it and kept re-observing. | The user has no authority over completion. Only a measurement can complete a task, and here the measurement was wrong (C1, B1). |
| E3 | Instruction 5: occupied tray → "no collision-free way" → retried three times, identically. | A persistent obstacle (the tray's contents) is treated as a transient planning failure (see A4). |

**Architectural point:** each failure class should come with the options that can work. "Goal unmet although the step
succeeded" should go back to the user ("I put it in the green tray; you asked for the red one, which is occupied. Move
the pink block out first?"). The user should be able to confirm completion, recorded as the user's confirmation, not as
a measurement.

## F. Smaller defects

- **F1:** the clarify step failed twice in instruction 1: Gemma sent `"request": null` with a question, and the schema
  requires text (`Clarification.request`).
- **F2:** the completion message in instruction 1 quoted the pick ("holding object #4") correctly. In instruction 2 the
  world model kept #4 "held" at its old pick position after the place.
- **F3:** Gemma called the red tray "pink tray" in its restatement. The tray renders pinkish in the simulator, so the
  colour words drift.

---

## Direction for the architecture work

1. **World model with object permanence:** one record per physical object; updated only on verified evidence; follows an
   object by id through pick, hold and place; knows container contents and free space.
2. **Session-level references:** words resolved once to world ids and kept for the session; the goal stored in world ids
   from the first plan.
3. **Action-scoped verification:** verify the moved object (by id) against what the action did; for containers,
   "on/in" semantics checked against the container's contents.
4. **Goal-locked planning:** replans and decompositions may change *how*, not *which objects*; a change of objects goes
   to the user.
5. **State-filtered operations:** the menu Gemma chooses from contains only operations whose preconditions hold now.
   Gemma still chooses.
6. **Failure-class recovery:** each failure names the options that can work; "goal unmet after a successful step" goes to
   the user; the user can confirm completion.

These keep the master–worker structure (Gemma decides; workers execute and report) and add no matching rules. They
change what the master is given and what a worker's result may change.
