"""Run a golden set against an agent and summarise the result."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .assertions import AgentResponse, CheckResult, Judge, run_checks
from .dataset import TestCase

Agent = Callable[[str, dict[str, Any]], AgentResponse]


@dataclass
class CaseResult:
    id: str
    question: str
    tags: list[str]
    passed: bool
    checks: list[CheckResult]
    route: str | None = None
    execution_path: list[str] = field(default_factory=list)
    latency_s: float | None = None
    node_latency_s: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def failed_layer(self) -> str | None:
        if self.error:
            return "ERROR"
        failed = [c.layer for c in self.checks if not c.passed]
        return min(failed) if failed else None


@dataclass
class RunResult:
    cases: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        return sum(c.passed for c in self.cases) / len(self.cases) if self.cases else 0.0

    def by_layer(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.cases:
            if c.failed_layer:
                counts[c.failed_layer] = counts.get(c.failed_layer, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {"pass_rate": self.pass_rate, "cases": [asdict(c) for c in self.cases]}


def _run_one(case: TestCase, agent: Agent, judge: Judge | None) -> CaseResult:
    started = time.perf_counter()
    try:
        resp = agent(case.question, case.context)
    except Exception as exc:  # the agent crashing is a result, not a harness bug
        return CaseResult(case.id, case.question, case.tags, False, [], error=f"{type(exc).__name__}: {exc}")
    elapsed = resp.latency_s if resp.latency_s is not None else time.perf_counter() - started
    checks = run_checks(case.question, case.expect, resp, judge)
    return CaseResult(
        id=case.id,
        question=case.question,
        tags=case.tags,
        passed=all(c.passed for c in checks),
        checks=checks,
        route=resp.route,
        execution_path=list(resp.execution_path or []),
        latency_s=round(elapsed, 3),
        node_latency_s=dict(resp.node_latency_s or {}),
    )


def run(cases: list[TestCase], agent: Agent, judge: Judge | None = None, workers: int = 1) -> RunResult:
    """Run every case. ``workers`` > 1 runs cases in parallel threads (keep it low for rate-limited endpoints)."""
    if workers <= 1:
        return RunResult([_run_one(c, agent, judge) for c in cases])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return RunResult(list(pool.map(lambda c: _run_one(c, agent, judge), cases)))
