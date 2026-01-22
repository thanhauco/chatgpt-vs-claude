"""Model providers: real OpenAI / Anthropic plus an offline mock for demos."""
from __future__ import annotations

import os
import re
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GenerationResult:
    provider: str
    model: str
    text: str
    latency_s: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class Provider(ABC):
    provider: str = "base"
    model: str = "unknown"

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> GenerationResult:
        ...


# --------------------------------------------------------------------------- #
# Offline mock provider                                                        #
# --------------------------------------------------------------------------- #
_RESTATEMENT = [
    "That's a great question.",
    "Thanks for asking.",
    "Let me help you with that.",
]
_CONTEXT = [
    "For some additional context, this topic has a few dimensions worth understanding.",
    "Before diving in, it helps to frame the broader picture here.",
]
_HEDGES = [
    "Generally speaking, the answer can vary somewhat depending on context.",
    "In most cases this is typically true, though not always.",
]
_EXAMPLES = [
    "For example, you might consider how this applies in everyday scenarios.",
    "As an illustration, imagine a common situation where this comes up.",
]
_CAVEATS = [
    "It's important to note that the right approach can depend on your specific situation.",
    "Keep in mind that there are exceptions to consider.",
    "That said, your mileage may vary depending on context.",
]
_SAFETY = [
    "As always, consider consulting a qualified professional for advice specific to your needs.",
    "Please double-check against an authoritative source before relying on this.",
]
_SUMMARY = [
    "In summary, the key point is captured above, with surrounding detail for completeness.",
    "Overall, that covers the essentials and a bit more for good measure.",
]

_PADDING_POOLS = [_RESTATEMENT, _CONTEXT, _HEDGES, _EXAMPLES, _CAVEATS, _SAFETY, _SUMMARY]


def _as_points(core_answer: str) -> List[str]:
    parts = [p.strip(" .") for p in re.split(r"[.;]\s+", core_answer.strip()) if p.strip(" .")]
    if len(parts) < 3:
        parts = (parts + [
            "Key consideration to keep in mind",
            "Practical next step you can take",
            "Common pitfall to avoid",
        ])[:3]
    return parts


def _truncate_words(text: str, n: int) -> str:
    words = text.split()
    if len(words) <= n:
        return text
    return " ".join(words[:n]).rstrip(",;:") + "."


def _stable_seed(*parts) -> int:
    import hashlib

    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


class MockProvider(Provider):
    """Deterministic synthetic generator used when no API keys are present.

    Verbosity decreases as a technique's ``conciseness_level`` rises and varies
    slightly with temperature. ``base_verbosity`` models a product-level default
    (e.g. one assistant tuned toward longer answers than another).
    """

    def __init__(self, provider: str, model: str, base_verbosity: int):
        self.provider = provider
        self.model = model
        self.base_verbosity = base_verbosity

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        core_answer: str = "Here is the answer.",
        conciseness_level: int = 0,
        technique_id: str = "baseline",
        seed: int = 0,
        **kwargs,
    ) -> GenerationResult:
        start = time.perf_counter()
        rng = random.Random(_stable_seed(self.provider, prompt, technique_id, round(temperature, 3), seed))

        if technique_id == "one_sentence":
            text = core_answer.strip()
            if not text.endswith((".", "!", "?")):
                text += "."
        elif technique_id == "bullets_3":
            text = "\n".join(f"- {p}" for p in _as_points(core_answer)[:3])
        else:
            budget = max(0, self.base_verbosity - conciseness_level)
            budget += rng.randint(0, 1) + int(round(temperature * 1.5))
            pools = _PADDING_POOLS[:]
            rng.shuffle(pools)
            chosen: List[str] = []
            for pool in pools:
                if len(chosen) >= budget:
                    break
                chosen.append(rng.choice(pool))
            text = " ".join([core_answer.strip()] + chosen)
            if technique_id == "word_limit":
                text = _truncate_words(text, 40)

        latency = (time.perf_counter() - start) + rng.uniform(0.2, 0.8)
        return GenerationResult(self.provider, self.model, text, round(latency, 3))


# --------------------------------------------------------------------------- #
# Real providers (lazy imports so the repo runs without the SDKs installed)    #
# --------------------------------------------------------------------------- #
class OpenAIProvider(Provider):
    provider = "openai"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        from openai import OpenAI

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> GenerationResult:
        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        latency = time.perf_counter() - start
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return GenerationResult(
            self.provider, self.model, text, round(latency, 3),
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )


class AnthropicProvider(Provider):
    provider = "anthropic"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, max_tokens: int = 1024):
        import anthropic

        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> GenerationResult:
        start = time.perf_counter()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.perf_counter() - start
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        usage = getattr(resp, "usage", None)
        return GenerationResult(
            self.provider, self.model, text, round(latency, 3),
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
        )
