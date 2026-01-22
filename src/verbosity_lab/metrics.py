"""Verbosity and conciseness metrics for model responses."""
from __future__ import annotations

import re
from typing import Dict, List

try:  # optional, more accurate token counts
    import tiktoken  # type: ignore

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken is optional
    _ENC = None

HEDGE_WORDS: List[str] = [
    "might", "may", "could", "perhaps", "possibly", "probably", "likely",
    "generally", "typically", "usually", "often", "sometimes", "arguably",
    "somewhat", "relatively", "tends to", "can be", "i think", "i believe",
    "it seems", "in general", "more or less", "to some extent",
]

CAVEAT_PHRASES: List[str] = [
    "it's important to", "it is important to", "keep in mind", "note that",
    "please note", "as an ai", "i cannot", "i can't", "however", "that said",
    "of course", "remember that", "be sure to", "disclaimer", "bear in mind",
    "consult a professional", "consult a qualified", "it depends",
]

FILLER_PHRASES: List[str] = [
    "in order to", "the fact that", "at the end of the day", "needless to say",
    "as you can see", "it's worth noting", "it is worth noting", "basically",
    "essentially", "in conclusion", "to summarize", "to sum up", "overall",
    "as mentioned", "as previously stated", "for all intents and purposes",
]

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[^.!?]+[.!?]?")
_LIST_RE = re.compile(r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s+")


def count_tokens(text: str) -> int:
    """Token count via tiktoken when available, else a word-based heuristic."""
    if _ENC is not None:
        try:
            return len(_ENC.encode(text))
        except Exception:  # pragma: no cover
            pass
    return int(round(len(_WORD_RE.findall(text)) * 1.3))


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.findall(text) if s.strip()]


def _count_occurrences(haystack: str, needles: List[str]) -> int:
    return sum(haystack.count(n) for n in needles)


def compute_metrics(text: str) -> Dict[str, float]:
    """Return a flat dict of verbosity metrics for a single response."""
    text = text or ""
    lower = text.lower()
    words = _WORD_RE.findall(lower)
    word_count = len(words)
    sentence_count = len(split_sentences(text))
    unique = len(set(words))
    list_items = len(_LIST_RE.findall(text))

    hedge = _count_occurrences(lower, HEDGE_WORDS)
    caveat = _count_occurrences(lower, CAVEAT_PHRASES)
    filler = _count_occurrences(lower, FILLER_PHRASES)

    denom = max(word_count, 1)
    return {
        "char_count": len(text),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "token_count": count_tokens(text),
        "avg_words_per_sentence": round(word_count / max(sentence_count, 1), 2),
        "unique_word_count": unique,
        "type_token_ratio": round(unique / denom, 4),
        "list_item_count": list_items,
        "hedge_count": hedge,
        "caveat_count": caveat,
        "filler_count": filler,
        "hedge_density": round(hedge / denom, 4),
        "caveat_density": round(caveat / denom, 4),
        "filler_density": round(filler / denom, 4),
    }
