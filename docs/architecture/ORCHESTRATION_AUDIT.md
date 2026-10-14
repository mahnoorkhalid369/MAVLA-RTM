# Orchestration audit (Checkpoint 1, 2026-10-14)

Branch `feature/orchestration` from `feature/active-perception` (tag `before-orchestration` = 3411b59). Other
refs verified: `main`, `refactor/goal-supervisor`, tags `baseline-before-goal-supervisor`, `before-active-perception`.

## Reproduced failure (user's live run `out/demo/2026-10-14/live_211215/04`, `/05`)
"pick the dark brown block": planner output `pick_place 'dark brown block' on 'null'` → the skill grounded
"null" as an object (Gemma "resolved" it) → irrelevant question → 120 s timeout → CANCELLED (268.6 s). Second
attempt: the user answered "12" (a cube 0.34 m from the base) → "carry above target … out of reach" → the
planning budget re-observed and replanned the same infeasible placement, the user's answer was not kept, the
same question was asked again → CANCELLED (397.1 s).

Root causes (code):
1. **No pick-only capability** — the planner's only manipulation verb was `pick_place`; `PickPlaceWorkflow` is
   one chain grasp → carry → place → HOME.
2. **Placeholder strings pass validation** — `plan_goal` checked non-empty fields; `"null"` is non-empty.
3. **Deterministic infeasibility is retried** — every `PlanningError` uses the planning budget (re-observe +
   replan), including "out of reach" / "too wide", which no re-observation can change.
4. **Clarifications are not remembered** — answers live only in the workflow run that asked.
5. **No state across instructions** — `agent/run.py` builds a new `Tools` (world model, held object) per
   instruction: "put it beside …" after "pick up …" cannot know what "it" is or that it is held.
6. **Goal success = all steps succeeded** — no explicit desired final state is checked (step evidence only).

## What exists (worker-like components)
| Target role | Existing code | Gap |
|---|---|---|
| Supervisor | `agent/supervisor.py` GoalSupervisor (LangGraph: interpret → dispatch → step verify → replan → goal verify) | steps are calls, not typed requests/results; no worker status; no goal conditions |
| Planning | `Roles.plan_goal` (one Gemma call, flat capability steps, validation, completion rules) | plans before seeing the scene; cannot decompose multi-object goals over measured objects |
| Perception | `capabilities.run_observe/describe/answer/locate`, `Tools`, `WorldModel` | answers carry no uncertainty; no request for a closer look |
| Exploration | `agent/exploration.py`, `run_inspect/run_explore` | only triggered by "not visible" |
| Manipulation | `PickPlaceWorkflow` (validated chain), `Tools.plan_pick_place`, `return_home` | no separate grasp / place-held operations; `park_for_view` path check ignores a held object |
| Verification | contact / placement / HOME checks inside the skill; `_verifying_goal` = all steps done | no goal predicates over measured state |
| Shared state | `TaskState` per instruction; `WorldModel` per instruction; `DescriptionMemory` per session | no session-level task memory (held object, last referent) |
| Events / viewer | `EventLog`, `events.py`, `viewer.py` | no worker / subtask / goal-condition view |

## Smallest coherent change set
1. Typed `WorkerRequest` / `WorkerResult`, a worker registry over the existing capabilities, a task ledger in
   `TaskState` (CP2).
2. `PickPlaceWorkflow` modes `pick` (grasp + lift + retention check, stays holding, no HOME) and `place`
   (destination only, carry from the current held state) reusing the validated nodes; session-persistent
   `Tools`/world/held object; deterministic infeasibility not retried; clarifications remembered (CP3).
3. Goal conditions (predicates) produced by the planner or derived from steps, evaluated over measured state
   (CP4).
4. Supervisor loop over workers; scene-grounded decomposition step for multi-object goals; uncertainty-driven
   inspection for questions (CP5).
