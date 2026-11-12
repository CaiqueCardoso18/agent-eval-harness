import json
from pathlib import Path

import pytest

from evalkit import AgentResponse, Expectation, load_golden_set, run, run_checks
from evalkit.cli import main
from evalkit.judge import KeywordJudge, parse_verdict
from evalkit.report import diff

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "examples" / "golden.jsonl"


def test_golden_set_loads_and_filters_by_tag():
    assert len(load_golden_set(GOLDEN)) == 10
    assert {c.id for c in load_golden_set(GOLDEN, tags=["smoke"])} == {"returns-001", "shipping-001"}


def test_duplicate_ids_are_rejected(tmp_path):
    f = tmp_path / "dup.jsonl"
    f.write_text('{"id": "a", "question": "q"}\n{"id": "a", "question": "q2"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_golden_set(f)


def test_unknown_expect_keys_are_rejected(tmp_path):
    f = tmp_path / "bad.jsonl"
    f.write_text('{"id": "a", "question": "q", "expect": {"rout": "x"}}\n')
    with pytest.raises(ValueError, match="unknown expect keys"):
        load_golden_set(f)


def test_path_contains_respects_order():
    exp = Expectation(path_contains=["classifier", "answer"])
    ok = run_checks("q", exp, AgentResponse("a", execution_path=["classifier", "retriever", "answer"]))
    bad = run_checks("q", exp, AgentResponse("a", execution_path=["answer", "classifier"]))
    assert all(c.passed for c in ok)
    assert not all(c.passed for c in bad)


def test_ungrounded_answer_fails_l3():
    exp = Expectation(grounded=True)
    resp = AgentResponse("Our store opens at nine tomorrow morning", documents=["Refunds take 30 days"])
    results = run_checks("q", exp, resp)
    assert [c.name for c in results if not c.passed] == ["grounded"]


def test_judge_is_skipped_when_routing_already_failed():
    calls = []

    def judge(q, a, r):
        calls.append(1)
        return True, "ok"

    exp = Expectation(route="knowledge", judge_rubric="anything")
    run_checks("q", exp, AgentResponse("a", route="other"), judge)
    assert calls == []


@pytest.mark.parametrize(
    "reply,expected",
    [('{"pass": true, "reason": "fine"}', True), ('```json\n{"pass": false}\n```', False), ("looks good to me", False), ('{"pass": "yes"}', False)],
)
def test_judge_verdict_parsing_is_strict(reply, expected):
    assert parse_verdict(reply)[0] is expected


def test_toy_agent_passes_its_golden_set():
    import sys

    sys.path.insert(0, str(ROOT / "examples"))
    from toy_agent import agent

    result = run(load_golden_set(GOLDEN), agent, KeywordJudge())
    assert result.pass_rate == 1.0, [c.id for c in result.cases if not c.passed]


def test_diff_finds_regressions_and_node_slowdown():
    base = {"pass_rate": 1.0, "cases": [{"id": "a", "passed": True, "route": "x", "latency_s": 1.0, "node_latency_s": {"llm": 1.0}}]}
    cur = {"pass_rate": 0.0, "cases": [{"id": "a", "passed": False, "route": "y", "latency_s": 2.0, "node_latency_s": {"llm": 2.0}}]}
    d = diff(cur, base)
    assert d["regressions"] == ["a"] and d["route_changes"] == ["a"]
    assert d["node_slowdown"]["llm"] == pytest.approx(2.0)


def test_cli_gate_fails_below_threshold(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT / "examples")
    out, rep = tmp_path / "run.json", tmp_path / "report.md"
    args = ["run", "--agent", "toy_agent:agent", "--dataset", str(GOLDEN), "--out", str(out), "--report", str(rep)]
    assert main(args + ["--min-pass-rate", "0.9", "--judge", "keyword"]) == 0
    assert json.loads(out.read_text())["pass_rate"] == 1.0
    assert "Pass rate" in rep.read_text()
