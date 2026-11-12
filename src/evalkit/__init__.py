"""evalkit: layered, regression-first evaluation for LLM agents."""

from .assertions import AgentResponse, CheckResult, run_checks
from .dataset import Expectation, TestCase, load_golden_set
from .runner import RunResult, run

__all__ = [
    "AgentResponse",
    "CheckResult",
    "Expectation",
    "RunResult",
    "TestCase",
    "load_golden_set",
    "run",
    "run_checks",
]
__version__ = "0.1.0"
