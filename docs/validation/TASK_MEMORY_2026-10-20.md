# Task-scoped memory and goal ownership: validation (2026-10-20)

**Commits** on `experiment/lab-exp1`:

| Commit | Change |
|---|---|
| `2941231` | task memory, goal ownership, worker choice, per-subtask decision |
| `6b80317` | criteria notation and shapes |
| `992817a` | held object from bindings |
| `09e6cc0` | continue semantics |
| `e8d8b3f` | worker label |
| this commit | output limits, report |

**Safety.** No physical hardware was used, no frozen file changed, and nothing was merged.

## 1. What changed

| Requirement | Implementation |
|---|---|
| Fresh task memory per instruction | `TaskState` is created by `GoalSupervisor.run`; `finalized` is set at the end; the world model and held object stay session-level |
| Original goal kept apart from the plan | Gemma's `success` criteria (required whenever there are steps), plus `answered()` for a question, go into `conditions` / `original_goal`. Fixed: a re-plan never changes them; a decomposition can only add measured criteria |
| Gemma chooses the worker | every step has `worker` and `do`; the menu groups operations under each worker's responsibility; `check_worker` only confirms the worker implements the operation |
| Structured feedback | `WorkerResult`, plus a task-memory exchange: attempted, tool status, subtask result, what happened, evidence, uncertainty, failure, held object before / after |
| Order of the loop | delegate → execute → WorkerResult → subtask verification → task-memory update → original-goal evaluation → COMPLETED, or Gemma decides |
| Gemma's view | `TaskState.memory_text()`: original goal, criteria met / unmet, every subtask, failures (and whether anything changed since), planned steps, robot state |
| No unchanged repetition | a re-plan whose first subtask already failed is sent back unless motion, a new frame or your answer has happened since |
| Three kinds of success | tool (`exchanges[].tool`), subtask (`exchanges[].subtask`), task (the original criteria shown); only the last completes a task |

The original criteria use your words; plans often act on measured ids. An object reference no worker has bound is
grounded by perception (Gemma resolve) for the goal check. This never applies to `located()`.

## 2. Required scenarios (deterministic mocked workers, scripted Gemma; `tests/integration/test_feedback_loop.py`)

| # | Test | Shows |
|---|---|---|
| 1 | `test_1_a_worker_succeeds_and_gemma_moves_on_to_the_next_needed_worker` | perception locates; goal `on(a, b)` still unmet; Gemma hands the next subtask to the manipulation worker; COMPLETED on a measured `on` |
| 2 | `test_2_uncertainty_makes_gemma_ask_for_useful_new_evidence` | an uncertain answer (Gemma names objects to check) → Gemma: inspection worker, then perception answers again |
| 3 | `test_3_a_failed_worker_is_not_repeated_with_unchanged_evidence` | an identical retry is rejected; a re-plan that observes first is accepted |
| 4 | `test_4_a_successful_worker_does_not_complete_an_unfinished_task` | "Pick the second block" planned as an observation: tool and subtask succeed, `holding(...)` stays unmet, not COMPLETED |
| 5 | `test_5_replanning_keeps_the_original_goal` | a re-plan to an observation plus "finish": the criteria are unchanged and the task is not completed |
| 6 | `test_6_completed_only_with_measured_evidence_for_the_original_goal` | the skill reports done but the placement is measured wrong → not COMPLETED; a measured `on` → COMPLETED |

Also tested:

- `test_each_instruction_gets_a_fresh_task_memory`;
- the objective-guard regression (`test_replan_that_drops_the_objective_cannot_complete_the_goal`, no longer xfail);
- `test_words_bound_to_the_held_object_in_this_task_name_it`;
- `test_continue_keeps_the_plan_and_gemma_is_told_*`;
- `test_a_worker_named_by_its_menu_label_*`;
- `test_criteria_in_the_prompts_notation_*`.

**Regression results:**

| Suite | Result |
|---|---|
| `.venv-seg` | 464 passed |
| system Python | 236 passed, 31 skipped |
| frozen baseline | OK |

## 3. Trace: Gemma → worker → feedback → task memory → Gemma → next worker → final verification

This trace uses mocked workers and a scripted Gemma decision; it is from `test_1`. Some lines are cut.

```
[USER]          goal: 'Put a on b.'
[AUTHORITY]     request is a action -> may: none, sensing, object_change (evidence Put a on b.; user_request)
[GEMMA]         goal: g | plan: 1. locate target='a', relation=None, anchor=None
[SUPERVISOR]    goal conditions (original goal): on(a, b)
[SUPERVISOR]    step 1: needs a fresh observation first
[SUPERVISOR]    -> PERCEPTION [t1] observe: sub-goal 'fresh observation for step 1 (locate)'; input {"clear_view": false}; constraints allowed_effects=['none', 'sensing', 'object_change'], effect=non
[PERCEPTION]    <- [t1] completed / completed (0.0 s, 0 Gemma calls, confidence measured): 3 objects measured (frame t); the arm hides 1 of 36 workspace points from the camera
[SUPERVISOR]    -> PERCEPTION [t2] locate: sub-goal 'locate a'; input {"target": "a", "relation": null, "anchor": null}; expect located(a); constraints allowed_effects=['none', 'sensing', 'object_cha
[PERCEPTION]    'a': object 1 at (550, 150, 20) mm, crop check yes
[PERCEPTION]    <- [t2] completed / completed (0.0 s, 0 Gemma calls, confidence measured): 'a' is object 1 at x +550 mm, y +150 mm (0.57 m from the robot), 40 mm tall [position measured; identity fro
[SUPERVISOR]    -> VERIFICATION [t3] verify: sub-goal 'verify step 1 (locate)'; input {"conditions": [{"pred": "located", "args": ["a"]}], "scope": "subgoal"}; expect located(a); constraints allowed_
[VERIFICATION]  <- [t3] completed / completed (0.0 s, 0 Gemma calls, confidence measured): met: located(a)
[SUPERVISOR]    task memory: step 1 (perception locate) tool completed, subtask achieved; holding None; 0 failure(s) so far
[PERCEPTION]    'b' -> #2 (grounded for the goal check (frame t))
[SUPERVISOR]    -> VERIFICATION [t4] verify: sub-goal "verify the user's original goal"; input {"conditions": [{"pred": "on", "args": ["a", "b"], "source": "gemma"}], "scope": "goal"}; expect on(a, b
[VERIFICATION]  <- [t4] failed / failed (0.0 s, 0 Gemma calls, confidence measured): not met: on(a, b): support 50 mm from the top, centre 300 mm from its centre; failure: not met: on(a, b): support 
[VERIFICATION]  original goal so far: on(a, b)=NO [measured]
[SUPERVISOR]    review step 1 locate (completed): ask Gemma - tool completed, subtask achieved, 1 original criteria not yet shown
[SUPERVISOR]    -> GEMMA: what next? (step 1 (perception worker: locate target=a): tool completed, subtask achieved: 'a' is object 1 at x +550 mm, y +150 mm (0.57 m from the robot), 40 mm tall [posit
[GEMMA]         decision: REPLAN - new steps: pick_place 'a' on 'b'; reason: a is found: move it onto b
[SUPERVISOR]    Gemma's new plan accepted
[GEMMA]         goal: g | plan: 2. pick_place 'a' on 'b'
[SUPERVISOR]    plan updated (re-planned by Gemma (the original goal is unchanged)): 1. locate [done]; 2. pick_place [pending]
[SUPERVISOR]    -> MANIPULATION [t5] pick_place: sub-goal 'pick_place a, b'; input {"object": {"text": "a", "relation": null, "anchor": null}, "relation": "on", "reference": {"text": "b", "relation":
[MANIPULATION]  <- [t5] completed / completed (0.07 s, 0 Gemma calls, confidence measured): Done (verified by the table camera): placed 3.0 mm from the planned spot, top +1.0 mm, tilt 0.4 deg; back a
[SUPERVISOR]    -> VERIFICATION [t6] verify: sub-goal 'verify step 2 (pick_place)'; input {"conditions": [{"pred": "on", "args": ["a", "b"]}], "scope": "subgoal"}; expect on(a, b); constraints allowe
[VERIFICATION]  <- [t6] completed / completed (0.0 s, 0 Gemma calls, confidence measured): met: on(a, b)
[SUPERVISOR]    task memory: step 2 (manipulation pick_place) tool completed, subtask achieved; holding None; 0 failure(s) so far
[SUPERVISOR]    -> VERIFICATION [t7] verify: sub-goal "verify the user's original goal"; input {"conditions": [{"pred": "on", "args": ["a", "b"], "source": "gemma"}], "scope": "goal"}; expect on(a, b
[VERIFICATION]  <- [t7] completed / completed (0.0 s, 0 Gemma calls, confidence measured): met: on(a, b)
[VERIFICATION]  original goal so far: on(a, b)=yes [measured]
[SUPERVISOR]    review step 2 pick_place (completed): original goal verified - tool completed, subtask achieved
[SUPERVISOR]    -> VERIFICATION [t8] verify: sub-goal "verify the user's original goal"; input {"conditions": [{"pred": "on", "args": ["a", "b"], "source": "gemma"}], "scope": "goal"}; expect on(a, b
[VERIFICATION]  <- [t8] completed / completed (0.0 s, 0 Gemma calls, confidence measured): met: on(a, b)
[VERIFICATION]  goal check: on(a, b)=yes [measured]
[SUPERVISOR]    review verify (completed): task verified - on(a, b) [measured]
[SUPERVISOR]    state COMPLETED: task complete
[SUPERVISOR]    COMPLETED: Done (verified by the table camera): placed 3.0 mm from the planned spot, top +1.0 mm, tilt 0.4 deg; back at HOME (max joint error 0.1 deg).
('a' is object 1 at x +550 mm, y +150 mm (0.57 m from the robot), 40 mm tall [position measured; identity from Gemma, confirmed on its crop])
(goal verified: on(a, b) [measured])
```

## 4. Real Gemma 4 E2B

**Planner benchmark** (`evaluation/planner_check.py`, 20 cases, plan role only;
`out/eval/taskmemory/planner_final.json`):

| Measure | Result |
|---|---|
| Correct plans (operation sequence and arguments) | 8/20 |
| Original-goal criteria that capture the request | 13/18 |
| Valid JSON | 19/20 |

Format problems found and fixed during measurement:

- criteria written in predicate notation;
- a condition's own keys nested under `args`;
- the worker written as its menu label;
- plans cut off at the old 300-token output limit.

**Isaac Sim** (headless, default scene, real motion in the simulator only; traces in `out/eval/taskmemory/live_*`):

| Run | Instruction | What the loop did | Result |
|---|---|---|---|
| 082258 #1 | Pick up the red block. | manipulation `pick` (168 s) → subtask `holding` measured → original goal measured | **COMPLETED** |
| 082258 #2 | Put it on the blue block. | `pick_place #6` → "already in the gripper" (no motion) → Gemma chose the planning worker (`decompose`), which failed on invalid JSON → three identical re-plans rejected (no new evidence). "blue block" grounded to #14; `on` measured NO | FAILED (Gemma never chose `place`; its criteria `holding` + `on` contradict each other) |
| 082258 #3–5 | look / pick / home | the gripper still held the block: `inspect`, `pick` and HOME refused deterministically | FAILED (correct refusals) |
| 151307 | Put the red block on the blue block. | `pick` verified → original `on(red_block, blue_block)` measured NO (156 mm) → Gemma: `continue` (with extra steps, not used) → `pick_place` → "already in the gripper" (held object recognised from the task's bindings) → Gemma re-planned `pick_place` again with "manipulation worker" (label rejected at the time) | FAILED |
| 152316 | Find the red block, then inspect it. | perception `observe` → describe → planning `decompose` (failed on invalid JSON) → `pick` refused by authorization (a description request) → four identical re-plans rejected | FAILED (Gemma added an objective nobody asked for: `holding(red block)`) |

In every live run:

- tool, subtask and task outcomes were recorded separately;
- each subtask's effect was verified;
- each decision used the task memory;
- the original goal was never dropped;
- no failure was repeated without new evidence;
- nothing was reported COMPLETED without measured evidence.

**No complete multi-worker live success with real Gemma was obtained.** The architecture's chain is shown with
mocked workers (section 3).

## 5. Remaining limitations

- **Gemma's decisions are the bottleneck:**
  - it does not choose `place` for a held object;
  - it repeats failed subtasks, which the guard rejects;
  - it adds criteria that were not asked for, or that contradict each other;
  - it answers `continue` with new steps.

  None of this is patched with rules; it ends as an honest failure.
- **Per-subtask Gemma calls add time** (6–15 s per decision). The pick itself takes about 170 s, including 22 Gemma
  object descriptions.
- **Criteria quality:** 13/18. Free-text criteria are never verifiable, so those tasks never complete.
- **Grounding** uses Gemma resolve without the crop confirmation, once per frame. A wrong grounding makes a criterion
  false, never true by itself, except for criteria that a wrong binding happens to satisfy.
- **Precondition outcomes.** Picking or inspecting while holding a *different* object is still `unsafe` (stops
  without Gemma), by design for safety.
