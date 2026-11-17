"""python -m gpumon {record,bench,report,compare,throughput} ... (see gpumon/README.md)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gpumon", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="sample while you run the system yourself (another terminal)")
    r.add_argument("--out", default=None, help="run folder (default out/gpumon/record/<date>_<time>)")
    r.add_argument("--events", action="append", default=None,
                   help="folder searched for the agent's events.jsonl (repeatable; default out/demo)")
    r.add_argument("--hz", type=float, default=10.0)
    r.add_argument("--label", default="manual")
    r.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    r.add_argument("--until-idle", action="store_true",
                   help="stop once the agent / simulator have started and both exited")
    r.add_argument("--no-report", action="store_true")

    b = sub.add_parser("bench", help="run a benchmark file: configurations x repeats, unattended")
    b.add_argument("file", type=Path)
    b.add_argument("--only", nargs="+", help="configuration labels to run")
    b.add_argument("--repeats", type=int, help="override the file's repeats")
    b.add_argument("--dry-run", action="store_true", help="print the commands only")

    p = sub.add_parser("report", help="summary + timeline for one run folder")
    p.add_argument("run", type=Path)
    p.add_argument("--no-figures", action="store_true")

    c = sub.add_parser("compare", help="comparison table + figure for a bench folder")
    c.add_argument("base", type=Path)

    t = sub.add_parser("throughput", help="FlexGen-style experiment: Gemma alone, placement x precision x batch size")
    t.add_argument("--configs", nargs="+", default=None, help="configurations (default: 4bit_gpu 4bit_offload 8bit_offload)")
    t.add_argument("--batches", nargs="+", type=int, default=None, help="batch sizes (default 1 2 4 8 16)")
    t.add_argument("--prompt-tokens", type=int, default=512)
    t.add_argument("--gen-tokens", type=int, default=64)
    t.add_argument("--repeats", type=int, default=3)
    t.add_argument("--model", default="models/gemma-4-E4B-it")
    t.add_argument("--budget-gb", type=float, default=None,
                   help="cap PyTorch at this many GB after loading (memory left next to Isaac Sim, e.g. 6.8)")
    t.add_argument("--out", default=None, help="run folder (default out/gpumon/throughput/<date>_<time>)")
    t.add_argument("--report", type=Path, default=None, help="only redo summary + figure for this run folder")
    t.add_argument("--child", default=None, help=argparse.SUPPRESS)  # internal: one configuration per process

    a = ap.parse_args(argv)
    if a.cmd == "record":
        from gpumon.report import report
        from gpumon.runner import record
        out = Path(a.out) if a.out else ROOT / "out/gpumon/record" / time.strftime("%Y-%m-%d_%H%M%S")
        roots = [Path(e) for e in (a.events or [str(ROOT / "out/demo")])]
        code = record(out, roots, a.hz, a.label, a.duration, a.until_idle)
        if not a.no_report:
            report(out)
        return code
    if a.cmd == "bench":
        from gpumon.runner import bench
        return bench(a.file, a.only, a.repeats, a.dry_run)
    if a.cmd == "report":
        from gpumon.report import report
        report(a.run, figures=not a.no_figures)
        return 0
    if a.cmd == "compare":
        from gpumon.report import compare
        return 0 if compare(a.base) else 1
    if a.cmd == "throughput":
        from gpumon import throughput as tp
        if a.report:
            return 0 if tp.report(a.report) else 1
        out = Path(a.out) if a.out else ROOT / "out/gpumon/throughput" / time.strftime("%Y-%m-%d_%H%M%S")
        args = (a.batches or tp.DEFAULT_BATCHES, a.prompt_tokens, a.gen_tokens, a.repeats, a.model, a.budget_gb, out)
        if a.child:
            return tp.child(a.child, *args)
        return tp.run(a.configs or tp.DEFAULT_CONFIGS, *args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
