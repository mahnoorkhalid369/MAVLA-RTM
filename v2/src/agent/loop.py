"""One instruction as a state machine: the master plans, the workers do the steps, the master decides, until the request
is done or the budget is used up.

    LOOK      the table camera measures every object (Scene, agent/workers.py); the names come from the session's
              catalogue (agent/catalogue.py: Gemma named each object once, when it was first seen)
    PLAN      the master turns the request into steps, "<worker>: <message>", and writes its memory (agent/master.py)
    DISPATCH  the next step goes to the worker named at its start; the worker answers in sentences
    RECOVER   one supported measured recovery resumes the frozen skill stage; unresolved physical failures STOP
    REVIEW    a step that went as asked -> DISPATCH the next one; the last step a verification that says yes -> CHECK;
              anything else -> DECIDE
    DECIDE    the master sees the request, the camera picture of the table NOW, its memory, the plan left, the latest
              reports and the budget left, and answers: continue / plan (new steps) / ask the user / done / stop, with its memory rewritten
    ASK       the question goes to the user; the answer becomes the latest report -> DECIDE
    FINISH    "done": if manipulation was requested, verification must measure its outcomes first (otherwise the
              failure stops with the measured cause); all requests then go to CHECK
    CHECK     Gemma looks at the table now (and the wrist camera's view of what the gripper holds) with only the user's
              own words and later user updates: yes -> COMPLETED; no -> DECIDE once per changed state

The master retries, changes the step or uses another worker as it judges; the budget bounds it: MAX_STEPS steps and
MAX_DECISIONS decisions per instruction (it is told what is left). Messages to the user are in words: any internal
number (#12, R2) that slips in is taken out. Result: {"status": "COMPLETED" | "FAILED" | "STOPPED", "message"} in
<run dir>/result.json with every report; every Gemma answer and its memory are in <run dir>/events.jsonl.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict

from agent import master
from agent import prompts
from agent.plan_validation import requests_placement
from agent.gemma import VLMFailure
from agent.contracts import TaskContext
from agent.performance import options
from agent.workers import Manipulation, Perception, Report, Scene, Verification
from perception.capture import ObservationError

MAX_STEPS, MAX_DECISIONS = 40, 20
LATEST = 4  # reports the master sees at each decision (its memory carries the rest)


def to_user(text: str) -> str:
    text = re.sub(r"\s*\([^()]*(?:#\d+|\bR\d+\b)[^()]*\)", "", text)  # "(#12)", "(R3 through R11)"
    return re.sub(r"#\d+", "that object", text)


def run_instruction(request: str, tools, gemma, log, run_dir, ask_user=input) -> dict:
    gemma.log, gemma.run_dir = log, run_dir
    scene = Scene(tools, run_dir)
    context = TaskContext()
    performance = options(tools)
    manipulation = Manipulation(tools, gemma, scene, log, context)
    verification = Verification(tools, context, scene, log, gemma)
    workers = {"perception": Perception(gemma, scene, log).run, "manipulation": manipulation.run,
               "verification": verification.run}
    reports: list[str] = []
    worker_reports = []
    failures = {}
    perception_rounds = 0
    failed_goal_checks = set()
    clarifications = []
    identity_unresolved = False
    questions = 0
    steps: list = []
    planned_actions, active_plan, step_ids = [], [], []
    memory, executed, decisions, verified = "", 0, 0, False

    def say(who: str, text: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] [{who.upper():12s}] {text}", flush=True)
        log("say", who=who, text=text)

    def finish(status: str, message: str) -> dict:
        say("result", f"{status}: {to_user(message)}")
        result = {"request": request, "status": status, "message": to_user(message), "reports": reports, "memory": memory,
                  "worker_reports": worker_reports, "execution_history": context.history,
                  "outcomes": [asdict(o) for o in context.outcomes.values()],
                  "unresolved_actions": sorted(context.pending, key=str),
                  "motion_requested": context.motion_requested, "motion_attempted": context.motion_attempted}
        result["planned_actions"] = planned_actions
        result["performance"] = performance.manifest()
        result["effective_request"], result["clarifications"] = effective_request(), list(clarifications)
        (run_dir / "result.json").write_text(json.dumps(result, indent=1))
        return result

    def budget() -> str:
        return f"{MAX_STEPS - executed} steps and {MAX_DECISIONS - decisions} decisions"

    def effective_request():
        return request + (prompts.USER_UPDATES.format(updates="\n".join(clarifications)) if clarifications else "")

    def held_name():
        return tools.object_words.get(tools.held_object, "object") if tools.held_object is not None else None

    def completion_version():
        return context.revision, getattr(tools, "scene_revision", getattr(tools, "motion_count", 0)), tools.held_object

    def record_plan(new_steps):
        active_plan.clear()
        step_ids.clear()
        for w, m in new_steps:
            index = len(planned_actions)
            planned_actions.append({"worker": w, "message": m, "status": "pending"})
            active_plan.append(index)
            step_ids.append(index)

    try:
        policy = prompts.manifest()
        (run_dir / "prompts.json").write_text(json.dumps(policy, indent=1))
        log("prompt_policy", revision=policy["revision"], sha256=policy["sha256"])
        log("performance_policy", **performance.manifest())
        say("policy", f"{policy['revision']} master={policy['sha256']['master'][:12]}")
        state = "LOOK"
        while True:
            log("loop", state=state)
            if state == "LOOK":
                say("perception", "looking at the table ...")
                obs = scene.current()
                say("perception", f"{len(obs.marks)} objects measured")
                state = "PLAN"

            elif state == "PLAN":
                try:
                    steps, memory = master.plan(gemma, request, scene.text() + "\n" + context.summary(held_name()), obs.rgb, log,
                                                image_info=scene.image_info(obs))
                    context.motion_requested |= any(w == "manipulation" for w, _ in steps)
                except VLMFailure as exc:
                    return finish("FAILED", f"no plan: {exc}")
                say("master", "plan: " + " | ".join(f"{w}: {m}" for w, m in steps))
                if memory:
                    say("memory", memory)
                record_plan(steps)
                state = "DISPATCH"

            elif state == "DISPATCH":
                executed += 1
                if executed > MAX_STEPS:
                    return finish("STOPPED", f"the budget of {MAX_STEPS} steps is used up; {memory}")
                worker, message = steps.pop(0)
                action_index = step_ids.pop(0)
                if worker == "manipulation":
                    context.motion_requested = True
                    manipulation.require_transfer = requests_placement(effective_request())
                    manipulation.defer_home = bool(performance.chain_transfers and steps and steps[0][0] == "manipulation")
                say(worker, message)
                motion_before = getattr(tools, "motion_count", 0)
                try:
                    if worker == "perception" and perception_rounds >= 2:
                        report = Report("unsure", "visual question budget exhausted; use the evidence already returned, "
                                        "ask a specific user question, or stop", code="perception_budget")
                    else:
                        report = workers[worker](message)
                    if worker == "perception":
                        perception_rounds += 1
                    elif worker == "manipulation" and (report.ok or getattr(tools, "motion_count", 0) != motion_before):
                        perception_rounds = 0
                except VLMFailure as exc:
                    report = Report("failed", str(exc))
                say(worker, f"-> {report}")
                reports.append(f"{worker}: {message} -> {report}")
                worker_reports.append({"worker": worker, "message": message, **asdict(report)})
                log("handoff_result", worker=worker, message=message, **asdict(report))
                planned_actions[action_index]["status"] = "done" if report.ok else "failed"
                if worker == "manipulation" and report.code in ("grounding_invalid", "cannot_act"):
                    identity_unresolved = True
                if worker == "manipulation" and not report.ok:
                    recovery = report.details.get("recovery")
                    if recovery in ("inspect_wrist", "reconcile_held"):
                        state = "RECOVER"
                        continue
                    if recovery == "stop":
                        return finish("STOPPED", f"{report.text}. {report.details.get('required_change', '')}".strip())
                if worker == "perception" and perception_rounds >= 4:
                    return finish("STOPPED", "repeated visual questions did not resolve the task; clarification is needed")
                if not report.ok and worker in ("manipulation", "verification"):
                    context.note_failure(f"{worker}: {message}", report.text)
                if not report.ok:
                    key = (worker, " ".join(message.lower().split()), report.code or report.text,
                           getattr(tools, "motion_count", 0), scene.evidence_key())
                    failures[key] = failures.get(key, 0) + 1
                    if failures[key] >= 2:
                        return finish("STOPPED", f"the same worker request failed twice without changed scene evidence or robot motion: {report.text}")
                if worker == "manipulation":  # any arm action, even a failed one, may have changed the scene
                    verified = False
                if worker == "verification":
                    verified = report.word == "yes"
                state = "REVIEW"

            elif state == "RECOVER":
                executed += 1
                if executed > MAX_STEPS:
                    return finish("STOPPED", "recovery budget exhausted")
                say("recovery", report.details["required_change"])
                log("recovery_dispatch", worker="manipulation", request=asdict(report))
                report = manipulation.recover(report)
                planned_actions[action_index]["status"] = "done" if report.ok else "failed"
                verified = False
                say("manipulation", f"-> {report}")
                reports.append(f"measured manipulation recovery -> {report}")
                worker_reports.append({"worker": "manipulation", "message": "measured local recovery", **asdict(report)})
                log("handoff_result", worker="manipulation", message="measured local recovery", **asdict(report))
                if not report.ok:
                    return finish("STOPPED", report.text + ". " + report.details.get("required_change", "No verified recovery is available."))
                state = "REVIEW"

            elif state == "REVIEW":
                if verified and worker == "verification" and not steps:
                    state, answer = "CHECK", report.text
                    continue
                if (performance.finish_plan and report.ok and not steps and context.motion_requested
                        and not identity_unresolved and active_plan
                        and context.placements and not context.pending
                        and all(c.kind == "home" or (c.kind == "placed" and verification._receipt(c) is not None)
                                for c in context.outcomes.values())
                        and all(planned_actions[i]["status"] == "done" for i in active_plan)):
                    log("completion_fast_path", planned_actions=[planned_actions[i] for i in active_plan])
                    state, answer = "FINISH", report.text
                    continue
                state = "DISPATCH" if report.ok and steps else "DECIDE"

            elif state == "DECIDE":
                decisions += 1
                if decisions > MAX_DECISIONS:
                    return finish("STOPPED", f"the budget of {MAX_DECISIONS} decisions is used up; {memory}")
                try:
                    observation = scene.current()
                    constraint = prompts.IDENTITY_DISPATCH_BLOCKED if identity_unresolved else ""
                    action, value, memory = master.decide(gemma, effective_request(), scene.text() + "\n" + context.summary(held_name()) + constraint, memory,
                                                          "\n".join(f"{w}: {m}" for w, m in steps), reports[-LATEST:],
                                                          budget(), log, observation.rgb, image_info=scene.image_info(observation))
                except VLMFailure as exc:
                    return finish("FAILED", f"the master gave no usable decision: {exc}")
                say("master", action + (f": {value}" if isinstance(value, str) else ""))
                if memory:
                    say("memory", memory)
                if identity_unresolved and action not in ("ask", "stop"):
                    return finish("STOPPED", "object/action identity is unresolved; another dispatch requires a specific clarification")
                if action == "continue":
                    if steps:
                        state = "DISPATCH"
                    else:
                        reports.append("master: continue, but no steps are left: plan, ask, done or stop")
                elif action == "plan":
                    steps = value
                    record_plan(steps)
                    context.motion_requested |= any(w == "manipulation" for w, _ in steps)
                    say("master", "new plan: " + " | ".join(f"{w}: {m}" for w, m in steps))
                    state = "DISPATCH"
                elif action == "ask":
                    state, question = "ASK", value
                elif action == "stop":
                    return finish("STOPPED", value)
                elif action == "done":
                    state, answer = "FINISH", value

            elif state == "ASK":
                questions += 1
                if questions > 2:
                    return finish("STOPPED", "clarification budget exhausted; the unresolved request needs new information")
                try:
                    reply = ask_user(f"[{time.strftime('%H:%M:%S')}] [ROBOT ASKS  ] {to_user(question)}\nyour answer> ")
                except EOFError:
                    return finish("STOPPED", "user clarification is required: " + question)
                reports.append(f"you asked the user: {question} -> the user answered: {reply.strip()}")
                clarifications.append(prompts.USER_CLARIFICATION.format(question=question, reply=reply.strip()))
                context.revision += 1
                identity_unresolved = False
                perception_rounds = 0
                state = "DECIDE"

            elif state == "FINISH":
                if completion_version() in failed_goal_checks:
                    return finish("STOPPED", "completion was already rejected for this state; no physical progress or goal clarification has changed it")
                if context.motion_requested and not verified:
                    report = verification.run()
                    say("verification", f"-> {report}")
                    worker_reports.append({"worker": "verification", "message": "completion gate", **asdict(report)})
                    log("handoff_result", worker="verification", message="completion gate", **asdict(report))
                    verified = report.word == "yes"
                    if not verified:
                        reports.append(f"you said done; verification: {report} -> so the request is NOT done yet")
                        return finish("STOPPED", "completion gate failed: " + report.text)
                state = "CHECK"

            elif state == "CHECK":  # measured; now the user's own words against the pictures
                version = completion_version()
                if version in failed_goal_checks:
                    return finish("STOPPED", "the same unchanged goal check already failed; further completion claims cannot resolve it")
                report = verification.final_check(effective_request())
                say("final check", f"-> {report}")
                worker_reports.append({"worker": "verification", "message": "original goal check", **asdict(report)})
                if report.code == "sensor_epoch":
                    return finish("STOPPED", report.text)
                if report.word == "yes":
                    return finish("COMPLETED", answer)
                reports.append(f"final check against the user's words: {report} -> so the request is NOT done yet")
                failed_goal_checks.add(version)
                verified = False
                state = "DECIDE"

    except ObservationError as exc:
        log('perception_rejected', reason=str(exc))
        return finish('STOPPED', 'sensor/scene evidence rejected before further motion: ' + str(exc))
    except Exception as exc:
        log("error", error=repr(exc))
        return finish("FAILED", f"instruction failed: {type(exc).__name__}: {exc}")
