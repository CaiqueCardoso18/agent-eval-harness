"""Layered checks for one agent response.

The layers go from cheap and deterministic to expensive and fuzzy:

    L1  routing        did the classifier send the question to the right place?
    L2  execution path did the graph walk the expected nodes (and avoid forbidden ones)?
    L3  content        required / forbidden phrases, canned answers, grounding in retrieved docs
    L4  judge          an LLM (or any callable) scores the answer against a rubric

A failure in an early layer is the most useful signal: if routing is wrong,
the answer text is wrong for a boring reason and an LLM judge only adds noise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol

from .dataset import Expectation


@dataclass
class AgentResponse:
    """What the agent under test returns. Adapt your agent to this shape."""

    answer: str
    route: str | None = None
    execution_path: list[str] | None = None
    documents: list[str] | None = None
    latency_s: float | None = None
    node_latency_s: dict[str, float] | None = None


@dataclass
class CheckResult:
    layer: str
    name: str
    passed: bool
    detail: str = ""


class Judge(Protocol):
    def __call__(self, question: str, answer: str, rubric: str) -> tuple[bool, str]: ...


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(step in it for step in needle)


def check_routing(exp: Expectation, resp: AgentResponse) -> list[CheckResult]:
    if exp.route is None:
        return []
    ok = (resp.route or "").lower() == exp.route.lower()
    return [CheckResult("L1", "route", ok, f"expected {exp.route!r}, got {resp.route!r}")]


def check_path(exp: Expectation, resp: AgentResponse) -> list[CheckResult]:
    path = resp.execution_path or []
    out: list[CheckResult] = []
    if exp.path_exact is not None:
        out.append(CheckResult("L2", "path_exact", path == exp.path_exact, f"got {path}"))
    if exp.path_contains:
        ok = _is_subsequence(exp.path_contains, path)
        out.append(CheckResult("L2", "path_contains", ok, f"want {exp.path_contains} in order, got {path}"))
    for node in exp.path_excludes:
        out.append(CheckResult("L2", f"path_excludes:{node}", node not in path, f"got {path}"))
    return out


def _grounding_score(answer: str, documents: list[str]) -> float:
    """Share of the answer's content words that appear in the retrieved documents.

    Crude on purpose: it is a cheap, deterministic tripwire for answers that
    were clearly not built from the retrieved context. Use the L4 judge for nuance.
    """
    words = {w for w in re.findall(r"[a-zA-Z0-9\-]{4,}", answer.lower())}
    if not words:
        return 1.0
    corpus = _norm(" ".join(documents))
    return sum(1 for w in words if w in corpus) / len(words)


def check_content(exp: Expectation, resp: AgentResponse, grounding_threshold: float = 0.5) -> list[CheckResult]:
    answer = _norm(resp.answer)
    out: list[CheckResult] = []
    for phrase in exp.must_mention:
        out.append(CheckResult("L3", f"mentions:{phrase}", _norm(phrase) in answer))
    for phrase in exp.must_not_mention:
        out.append(CheckResult("L3", f"avoids:{phrase}", _norm(phrase) not in answer))
    if exp.canned is not None:
        out.append(CheckResult("L3", "canned_exact", answer == _norm(exp.canned), "canned answers must match verbatim"))
    if exp.grounded:
        docs = resp.documents or []
        if not docs:
            out.append(CheckResult("L3", "grounded", False, "no documents retrieved"))
        else:
            score = _grounding_score(resp.answer, docs)
            out.append(CheckResult("L3", "grounded", score >= grounding_threshold, f"overlap={score:.2f}"))
    return out


def check_judge(exp: Expectation, question: str, resp: AgentResponse, judge: Judge | None) -> list[CheckResult]:
    if not exp.judge_rubric or judge is None:
        return []
    ok, reason = judge(question, resp.answer, exp.judge_rubric)
    return [CheckResult("L4", "judge", ok, reason)]


def run_checks(
    question: str,
    exp: Expectation,
    resp: AgentResponse,
    judge: Judge | None = None,
    stop_on_structural_failure: bool = True,
) -> list[CheckResult]:
    """Run every layer that the expectation asks for.

    With ``stop_on_structural_failure`` the L4 judge is skipped when L1/L2
    already failed, which saves judge calls and keeps reports readable.
    """
    results = check_routing(exp, resp) + check_path(exp, resp)
    structural_ok = all(r.passed for r in results)
    results += check_content(exp, resp)
    if structural_ok or not stop_on_structural_failure:
        results += check_judge(exp, question, resp, judge)
    return results


JudgeFactory = Callable[[], Judge]
