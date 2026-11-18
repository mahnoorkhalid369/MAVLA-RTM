"""Start v3/dialogue (the frozen v3 robot with the conversation agents): load the models, start Isaac Sim, then run the user's instructions one after another.

    ./launch.sh                                  type instructions; an empty line quits
    ./launch.sh --instructions tests/sample.json run a list: [{"request": "...", "answers": ["..."]}]

Every log line shows the time since the request started, how long that step took, and who it is from:
    [ 1:03.0   17.0 s] [master -> manipulation    ] Put the cracker box on the tray, at the far side.
"""

import argparse
import json
import time
from pathlib import Path

import arm as robot_module
from agents import Agents

ROOT = Path(__file__).resolve().parents[1]  # v3


class Console:
    """The one log function. It prints each line and writes it to the request's event log, which the live screen shows."""
    log, start = None, time.monotonic()  # the event log and the start time of the request being handled

    def __call__(self, who, text, started=None):
        now = time.monotonic()
        clock, took = now - self.start, "" if started is None else f"{now - started:.1f} s"
        head = f"[{int(clock // 60):2d}:{clock % 60:04.1f} {took:>8}] [{who:<26}] "
        print(head + text.replace("\n", "\n" + " " * len(head)), flush=True)
        for line in text.splitlines() if self.log else []:
            self.log("say", who=who, text=line + (f"  ({took})" if took else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="snack_kitchen")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gui", action="store_true", help="show the Isaac Sim window")
    ap.add_argument("--instructions", type=Path, help="JSON list of {request, answers}")
    ap.add_argument("--live", type=Path, default=ROOT / "out/live", help="folder the live screen reads")
    ap.add_argument("--out", type=Path, default=ROOT / "out/v3d" / time.strftime("%Y-%m-%d") / time.strftime("run_%H%M%S"))
    a = ap.parse_args()

    from agent.eventlog import EventLog
    scripted = json.loads(a.instructions.read_text()) if a.instructions else None
    robot, gemma, tools, stop = robot_module.start(a.scenario, a.difficulty, a.seed, a.out, a.gui, a.live)
    results = []
    try:
        answers, say = [], Console()

        def ask_user(question):
            say.log("question", question=question, candidates=[])
            started = time.monotonic()
            if scripted is None:
                reply = input("> ").strip()
            else:
                reply = answers.pop(0) if answers else ""
            say.log("ask_user")
            return reply or "(the user did not answer)"

        robot.say = say
        agents = Agents(gemma, robot, ask_user, say)
        n = 0
        while True:
            if scripted is not None:
                if n == len(scripted):
                    break
                request, answers[:] = scripted[n]["request"], scripted[n].get("answers", [])
            else:
                request = input("\ninstruction> ").strip()
                if not request:
                    break
            n += 1
            run_dir = a.out / f"{n:02d}"
            (run_dir / "images").mkdir(parents=True, exist_ok=True)
            log = EventLog(run_dir / "events.jsonl", echo=False)
            gemma.log, gemma.run_dir = log, run_dir
            tools.begin_instruction(log, run_dir)
            tools.live.start_run(run_dir, request)  # the live screen now follows this request's log
            print(f"\n===== {n}. {request}", flush=True)
            log("instruction", text=request)
            start, calls = time.monotonic(), getattr(gemma, "call_count", 0)
            say.log, say.start = log, start
            try:
                status, answer = agents.run(request)
            except Exception as error:  # a crash in one instruction must not end the session
                status, answer = "ERROR", repr(error)
            results.append({"request": request, "status": status, "answer": answer,
                            "seconds": round(time.monotonic() - start), "model_calls": getattr(gemma, "call_count", 0) - calls})
            say("result", f"{status}: {answer}  ({results[-1]['model_calls']} model calls)", start)
            log("state", state=status, level="goal")
            say.log = None
            log.close()
            (a.out / "results.json").write_text(json.dumps(results, indent=1))
    finally:
        stop()
    print("\n===== results")
    for r in results:
        print(f"{r['status']:<10} {r['seconds']:>4} s  {r['request']}")
    print(f"logs: {a.out}")


if __name__ == "__main__":
    main()
