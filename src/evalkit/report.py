"""Reports and baseline comparison.

Comparing against a saved baseline run is what turns an eval into a
regression test: you see which cases broke, which got fixed, and whether a
model or prompt change made specific graph nodes slower.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .runner import RunResult


def save(run: RunResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(run.to_dict(), indent=2, default=str), encoding="utf-8")


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, round(q * (len(values) - 1)))
    return values[idx]


def latency_summary(cases: list[dict[str, Any]]) -> dict[str, float]:
    lat = [c["latency_s"] for c in cases if c.get("latency_s") is not None]
    if not lat:
        return {}
    return {"avg": statistics.mean(lat), "p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95), "max": max(lat)}


def node_latency(cases: list[dict[str, Any]]) -> dict[str, float]:
    per_node: dict[str, list[float]] = {}
    for c in cases:
        for node, secs in (c.get("node_latency_s") or {}).items():
            per_node.setdefault(node, []).append(secs)
    return {n: statistics.mean(v) for n, v in per_node.items()}


def diff(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    cur = {c["id"]: c for c in current["cases"]}
    base = {c["id"]: c for c in baseline["cases"]}
    common = cur.keys() & base.keys()
    regressions = sorted(i for i in common if base[i]["passed"] and not cur[i]["passed"])
    fixes = sorted(i for i in common if not base[i]["passed"] and cur[i]["passed"])
    route_changes = sorted(i for i in common if base[i].get("route") != cur[i].get("route"))
    nb, nc = node_latency(baseline["cases"]), node_latency(current["cases"])
    slowdown = {n: nc[n] / nb[n] for n in nb.keys() & nc.keys() if nb[n] > 0}
    return {
        "pass_rate": {"baseline": baseline["pass_rate"], "current": current["pass_rate"]},
        "regressions": regressions,
        "fixes": fixes,
        "route_changes": route_changes,
        "new_cases": sorted(cur.keys() - base.keys()),
        "node_slowdown": dict(sorted(slowdown.items(), key=lambda kv: -kv[1])),
        "latency": {"baseline": latency_summary(baseline["cases"]), "current": latency_summary(current["cases"])},
    }


def markdown(run: RunResult, comparison: dict[str, Any] | None = None) -> str:
    lines = [
        "# Eval report",
        "",
        f"**Pass rate:** {run.pass_rate:.1%} ({sum(c.passed for c in run.cases)}/{len(run.cases)})",
        "",
    ]
    layers = run.by_layer()
    if layers:
        lines += ["| First failing layer | Cases |", "| --- | --- |"]
        lines += [f"| {layer} | {n} |" for layer, n in layers.items()]
        lines.append("")
    failed = [c for c in run.cases if not c.passed]
    if failed:
        lines += ["## Failures", ""]
        for c in failed:
            reasons = c.error or "; ".join(f"{k.layer} {k.name}: {k.detail}".strip() for k in c.checks if not k.passed)
            lines.append(f"- `{c.id}`: {reasons}")
        lines.append("")
    if comparison:
        pr = comparison["pass_rate"]
        lines += [
            "## Against baseline",
            "",
            f"Pass rate {pr['baseline']:.1%} -> {pr['current']:.1%}",
            f"- Regressions: {', '.join(comparison['regressions']) or 'none'}",
            f"- Fixes: {', '.join(comparison['fixes']) or 'none'}",
            f"- Route changes: {', '.join(comparison['route_changes']) or 'none'}",
        ]
        slow = {n: x for n, x in comparison["node_slowdown"].items() if x >= 1.3}
        if slow:
            lines += ["", "| Node | x slower |", "| --- | --- |"]
            lines += [f"| {n} | {x:.2f} |" for n, x in slow.items()]
        lines.append("")
    return "\n".join(lines)
