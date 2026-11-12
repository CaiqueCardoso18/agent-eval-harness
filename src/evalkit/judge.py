"""LLM-as-a-judge.

Any OpenAI-compatible chat endpoint works (OpenAI, Azure OpenAI through a
gateway, vLLM, Ollama). Configure it with environment variables:

    EVALKIT_JUDGE_URL    e.g. https://api.openai.com/v1/chat/completions
    EVALKIT_JUDGE_KEY    bearer token
    EVALKIT_JUDGE_MODEL  e.g. gpt-4.1

The judge must answer with JSON: {"pass": true|false, "reason": "..."}.
Temperature is 0 and the verdict is parsed strictly, so a malformed reply
counts as a failure instead of silently passing.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

PROMPT = """You are grading an AI assistant's answer.

Question:
{question}

Answer:
{answer}

Rubric (the answer passes only if it satisfies all of it):
{rubric}

Reply with JSON only: {{"pass": true or false, "reason": "one short sentence"}}"""


def parse_verdict(text: str) -> tuple[bool, str]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return False, f"judge reply was not JSON: {text[:80]!r}"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False, f"judge reply was not valid JSON: {text[:80]!r}"
    verdict = data.get("pass")
    if not isinstance(verdict, bool):
        return False, "judge reply had no boolean 'pass'"
    return verdict, str(data.get("reason", ""))


class OpenAICompatibleJudge:
    def __init__(self, url: str | None = None, key: str | None = None, model: str | None = None, timeout: float = 60):
        self.url = url or os.environ["EVALKIT_JUDGE_URL"]
        self.key = key or os.environ.get("EVALKIT_JUDGE_KEY", "")
        self.model = model or os.environ.get("EVALKIT_JUDGE_MODEL", "gpt-4.1")
        self.timeout = timeout

    def __call__(self, question: str, answer: str, rubric: str) -> tuple[bool, str]:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "user", "content": PROMPT.format(question=question, answer=answer, rubric=rubric)}],
            }
        ).encode()
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.load(resp)
        return parse_verdict(payload["choices"][0]["message"]["content"])


class KeywordJudge:
    """Offline stand-in for demos and CI without API keys.

    Passes when every quoted phrase in the rubric ("like this") appears in the answer.
    """

    def __call__(self, question: str, answer: str, rubric: str) -> tuple[bool, str]:
        wanted = re.findall(r'"([^"]+)"', rubric)
        missing = [w for w in wanted if w.lower() not in answer.lower()]
        return (not missing, "ok" if not missing else f"missing {missing}")
