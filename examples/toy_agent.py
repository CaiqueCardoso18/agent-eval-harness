"""A tiny rule-based "support agent" so the harness runs without any API key.

It mimics the shape of a LangGraph RAG agent: a classifier routes the
question, a retriever pulls documents, a node writes the answer, and the
response carries the execution path and per-node timings.

Swap it for your real agent: wrap your graph call so it returns AgentResponse.
"""

from __future__ import annotations

import time

from evalkit import AgentResponse

KNOWLEDGE = {
    "returns": "Items can be returned within 30 days with the original receipt. Refunds go back to the original payment method.",
    "shipping": "Standard shipping takes 3 to 5 business days. Express shipping takes 1 business day.",
    "warranty": "Power tools carry a 2 year warranty that covers manufacturing defects, not wear and tear.",
}

CANNED_LEGAL = "I can't give legal advice. Please contact our legal team."
INJECTION_MARKERS = ("ignore previous", "ignore all", "system prompt", "you are now")


def _classify(question: str) -> str:
    q = question.lower()
    if any(m in q for m in INJECTION_MARKERS):
        return "blocked"
    if any(w in q for w in ("lawsuit", "liable", "liability", "sue")):
        return "legal"
    if any(w in q for w in ("return", "refund", "shipping", "deliver", "warranty", "broken")):
        return "knowledge"
    return "other"


def _retrieve(question: str) -> list[str]:
    q = question.lower()
    hits = []
    if "return" in q or "refund" in q:
        hits.append(KNOWLEDGE["returns"])
    if "ship" in q or "deliver" in q:
        hits.append(KNOWLEDGE["shipping"])
    if "warranty" in q or "broken" in q:
        hits.append(KNOWLEDGE["warranty"])
    return hits


def agent(question: str, context: dict | None = None) -> AgentResponse:
    timings: dict[str, float] = {}
    path = ["classifier"]

    t = time.perf_counter()
    route = _classify(question)
    timings["classifier"] = time.perf_counter() - t

    if route == "blocked":
        path.append("guardrail")
        return AgentResponse("I can only help with questions about our products and orders.", route, path, [], node_latency_s=timings)
    if route == "legal":
        path.append("canned")
        return AgentResponse(CANNED_LEGAL, route, path, [], node_latency_s=timings)
    if route == "other":
        path.append("fallback")
        return AgentResponse("I don't know that one. I can help with returns, shipping and warranty.", route, path, [], node_latency_s=timings)

    t = time.perf_counter()
    docs = _retrieve(question)
    timings["retriever"] = time.perf_counter() - t
    path += ["retriever", "answer"]
    answer = " ".join(docs) if docs else "I couldn't find that in our policies."
    return AgentResponse(answer, route, path, docs, node_latency_s=timings)
