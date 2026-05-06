"""Relevance LLM-judge: score response quality for live (and demo) runs.

Conciseness is only valuable if relevance holds. A judge rates each response on
relevance, completeness, and conciseness (1-5). `LLMJudge` uses a real model;
`MockJudge` is a deterministic heuristic so the offline pipeline still works.
"""
from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .metrics import compute_metrics


@dataclass
class JudgeResult:
    relevance: int
    completeness: int
    conciseness: int
    overall: float
    rationale: str = ""


def _overall(relevance: int, completeness: int, conciseness: int) -> float:
    return round(0.4 * relevance + 0.3 * completeness + 0.3 * conciseness, 2)


def _clamp(value, default: int = 3) -> int:
    try:
        return max(1, min(5, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


class Judge(ABC):
    name: str = "judge"

    @abstractmethod
    def score(self, prompt: str, response: str) -> JudgeResult:
        ...


class MockJudge(Judge):
    """Deterministic heuristic judge (no API calls) for demos and tests.

    Rewards on-topic, complete answers but penalizes padding/length on the
    conciseness axis — surfacing the completeness-vs-conciseness trade-off.
    """

    name = "mock"

    def score(self, prompt: str, response: str) -> JudgeResult:
        m = compute_metrics(response)
        words = m["word_count"]
        padding = int(m["hedge_count"] + m["caveat_count"] + m["filler_count"])
        rng = random.Random(hash((prompt, response)) & 0xFFFFFFFF)

        relevance = 1 if words == 0 else 5 - rng.randint(0, 1)
        completeness = max(1, min(5, 1 + words // 12))
        length_penalty = max(0, (words - 25) // 15)
        conciseness = max(1, min(5, 5 - padding - length_penalty))

        return JudgeResult(
            relevance, completeness, conciseness,
            _overall(relevance, completeness, conciseness),
            rationale=f"heuristic: words={words}, padding={padding}",
        )


class LLMJudge(Judge):
    """LLM-as-judge backed by any Provider (OpenAI/Anthropic)."""

    RUBRIC = (
        "You are a strict evaluator. Rate the assistant's response to a user question.\n"
        "Question:\n{question}\n\nResponse:\n{response}\n\n"
        "Score each dimension as an integer from 1 (poor) to 5 (excellent):\n"
        "- relevance: does it answer the actual question?\n"
        "- completeness: is the key information present?\n"
        "- conciseness: is it free of padding, hedging, and restatement?\n"
        'Return ONLY JSON: {{"relevance":n,"completeness":n,"conciseness":n,"rationale":"<=15 words"}}'
    )

    def __init__(self, provider, temperature: float = 0.0):
        self.provider = provider
        self.temperature = temperature
        self.name = f"llm:{getattr(provider, 'provider', '?')}"

    def score(self, prompt: str, response: str) -> JudgeResult:
        judge_prompt = self.RUBRIC.format(question=prompt, response=response)
        result = self.provider.generate(judge_prompt, temperature=self.temperature)
        data = _parse_json(getattr(result, "text", ""))
        relevance = _clamp(data.get("relevance"))
        completeness = _clamp(data.get("completeness"))
        conciseness = _clamp(data.get("conciseness"))
        return JudgeResult(
            relevance, completeness, conciseness,
            _overall(relevance, completeness, conciseness),
            rationale=str(data.get("rationale", ""))[:300],
        )
