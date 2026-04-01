"""Derived scores: verbosity-bias, padding score, summaries, transparency table."""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from . import config


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def add_verbosity_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Extra words vs the leanest answer to the *same* prompt: (words - min) / min."""
    df = df.copy()
    mins = df.groupby("prompt_id")["word_count"].transform("min").clip(lower=1)
    df["verbosity_bias"] = ((df["word_count"] - mins) / mins).round(3)
    return df


def add_padding_score(df: pd.DataFrame) -> pd.DataFrame:
    """0-100 composite of verbosity-bias plus hedge / caveat / filler density."""
    if "verbosity_bias" not in df.columns:
        df = add_verbosity_bias(df)
    else:
        df = df.copy()
    score = (
        0.40 * _minmax(df["verbosity_bias"])
        + 0.25 * _minmax(df["hedge_density"])
        + 0.20 * _minmax(df["caveat_density"])
        + 0.15 * _minmax(df["filler_density"])
    )
    df["padding_score"] = (100 * score).round(1)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = add_padding_score(df)
    agg_spec = dict(
        responses=("response", "count"),
        avg_words=("word_count", "mean"),
        avg_tokens=("token_count", "mean"),
        avg_sentences=("sentence_count", "mean"),
        avg_hedge_density=("hedge_density", "mean"),
        avg_caveat_density=("caveat_density", "mean"),
        verbosity_bias=("verbosity_bias", "mean"),
        padding_score=("padding_score", "mean"),
    )
    if "answer_coverage" in df.columns:
        agg_spec["avg_coverage"] = ("answer_coverage", "mean")
    if "signal_efficiency" in df.columns:
        agg_spec["avg_efficiency"] = ("signal_efficiency", "mean")
    agg = df.groupby(["provider", "model"]).agg(**agg_spec).reset_index()
    for c in agg.columns:
        if agg[c].dtype.kind == "f":
            agg[c] = agg[c].round(3)
    return agg


def transparency_table() -> Tuple[pd.DataFrame, dict]:
    cfg = config.load_transparency()
    dims = cfg["dimensions"]
    providers = cfg["providers"]
    rows = []
    for d in dims:
        row = {
            "dimension": d["name"],
            "weight": d.get("weight", 1),
            "description": d.get("description", ""),
        }
        for pid, pdata in providers.items():
            row[pdata.get("label", pid)] = pdata["scores"].get(d["id"])
        rows.append(row)
    return pd.DataFrame(rows), cfg
