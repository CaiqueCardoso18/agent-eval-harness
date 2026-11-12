"""Golden dataset loading.

A golden set is a JSONL file, one test case per line:

    {
      "id": "product-001",
      "question": "Which drill bit fits the TE 30?",
      "tags": ["product", "smoke"],
      "expect": {
        "route": "product",
        "path_contains": ["classifier", "retriever", "answer"],
        "path_excludes": ["internal"],
        "must_mention": ["TE 30"],
        "must_not_mention": ["I am not sure"],
        "grounded": true,
        "judge_rubric": "Recommends a compatible bit and does not invent prices."
      }
    }

Every key under ``expect`` is optional. A case only gets the checks it asks for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Expectation:
    route: str | None = None
    path_contains: list[str] = field(default_factory=list)
    path_excludes: list[str] = field(default_factory=list)
    path_exact: list[str] | None = None
    must_mention: list[str] = field(default_factory=list)
    must_not_mention: list[str] = field(default_factory=list)
    canned: str | None = None
    grounded: bool = False
    judge_rubric: str | None = None


@dataclass
class TestCase:
    id: str
    question: str
    expect: Expectation
    tags: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


_KNOWN = set(Expectation.__dataclass_fields__)


def _parse(raw: dict[str, Any], line_no: int) -> TestCase:
    if "id" not in raw or "question" not in raw:
        raise ValueError(f"line {line_no}: every case needs 'id' and 'question'")
    expect_raw = raw.get("expect", {})
    unknown = set(expect_raw) - _KNOWN
    if unknown:
        raise ValueError(f"line {line_no} ({raw['id']}): unknown expect keys {sorted(unknown)}")
    return TestCase(
        id=str(raw["id"]),
        question=str(raw["question"]),
        expect=Expectation(**expect_raw),
        tags=list(raw.get("tags", [])),
        context=dict(raw.get("context", {})),
    )


def load_golden_set(path: str | Path, tags: list[str] | None = None) -> list[TestCase]:
    """Load a JSONL golden set, optionally keeping only cases that carry one of ``tags``."""
    cases: list[TestCase] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            case = _parse(json.loads(line), line_no)
            if case.id in seen:
                raise ValueError(f"line {line_no}: duplicate id {case.id!r}")
            seen.add(case.id)
            if tags and not set(tags) & set(case.tags):
                continue
            cases.append(case)
    return cases
