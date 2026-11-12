"""Command line entry point.

    evalkit run --agent examples.toy_agent:agent --dataset examples/golden.jsonl \
                --baseline baseline.json --out run.json --min-pass-rate 0.9

Exit code is 1 when the pass rate is below ``--min-pass-rate`` or when a case
that passed in the baseline now fails, so the command works as a CI gate.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from . import report
from .dataset import load_golden_set
from .judge import KeywordJudge, OpenAICompatibleJudge
from .runner import run


def _import(spec: str):
    module, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"--agent must look like package.module:function, got {spec!r}")
    sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(module), attr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evalkit")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a golden set against an agent")
    r.add_argument("--agent", required=True, help="module:function returning AgentResponse")
    r.add_argument("--dataset", required=True)
    r.add_argument("--tags", nargs="*", help="only run cases with these tags")
    r.add_argument("--judge", choices=["none", "keyword", "openai"], default="none")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--baseline", help="previous run JSON to diff against")
    r.add_argument("--out", default="run.json")
    r.add_argument("--report", default="report.md")
    r.add_argument("--min-pass-rate", type=float, default=0.0)
    r.add_argument("--allow-regressions", action="store_true")
    args = p.parse_args(argv)

    agent = _import(args.agent)
    judge = {"none": None, "keyword": KeywordJudge(), "openai": None}[args.judge]
    if args.judge == "openai":
        judge = OpenAICompatibleJudge()

    cases = load_golden_set(args.dataset, args.tags)
    result = run(cases, agent, judge, workers=args.workers)
    report.save(result, args.out)

    comparison = None
    if args.baseline and Path(args.baseline).exists():
        comparison = report.diff(report.load(args.out), report.load(args.baseline))
    md = report.markdown(result, comparison)
    Path(args.report).write_text(md, encoding="utf-8")
    print(md)

    if result.pass_rate < args.min_pass_rate:
        print(f"FAIL: pass rate {result.pass_rate:.1%} is below {args.min_pass_rate:.0%}", file=sys.stderr)
        return 1
    if comparison and comparison["regressions"] and not args.allow_regressions:
        print(f"FAIL: regressions against baseline: {comparison['regressions']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
