"""Statistical analysis helpers for the verbosity comparison."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

try:  # optional: enables exact p-values
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:  # pragma: no cover - scipy is optional
    _scipy_stats = None


def describe_verbosity(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution summary of response length per provider."""
    out = (
        df.groupby("provider")
        .agg(
            n=("word_count", "size"),
            mean_words=("word_count", "mean"),
            std_words=("word_count", "std"),
            median_words=("word_count", "median"),
            p90_words=("word_count", lambda s: s.quantile(0.9)),
            mean_tokens=("token_count", "mean"),
        )
        .reset_index()
    )
    return out.round(2)


def cohens_d(a, b) -> float:
    """Standardized mean difference (pooled SD)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def _interpret_d(d: float) -> str:
    if np.isnan(d):
        return "n/a"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def welch_ttest(a, b):
    """Welch's t-test. Returns (t, p); p is None if scipy is unavailable."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if _scipy_stats is not None:
        t, p = _scipy_stats.ttest_ind(a, b, equal_var=False)
        return float(t), float(p)
    na, nb = len(a), len(b)
    se = np.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
    t = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    return float(t), None


def compare_providers(df: pd.DataFrame, metric: str = "word_count",
                      a: str = "openai", b: str = "anthropic") -> Dict:
    """Effect size + significance of `metric` between two providers."""
    sa = df[df["provider"] == a][metric].dropna()
    sb = df[df["provider"] == b][metric].dropna()
    if len(sa) == 0 or len(sb) == 0:
        return {}
    d = cohens_d(sa, sb)
    t, p = welch_ttest(sa, sb)
    mean_a, mean_b = sa.mean(), sb.mean()
    pct = 100 * (mean_a - mean_b) / mean_b if mean_b else float("nan")
    return {
        "metric": metric, "a": a, "b": b,
        "mean_a": round(mean_a, 2), "mean_b": round(mean_b, 2),
        "diff": round(mean_a - mean_b, 2), "pct_diff": round(pct, 1),
        "cohens_d": round(d, 3), "effect": _interpret_d(d),
        "t_stat": (round(t, 3) if t == t else None),
        "p_value": (round(p, 5) if p is not None else None),
    }


def technique_effectiveness(df: pd.DataFrame) -> pd.DataFrame:
    """Rank techniques by average word-count reduction vs the baseline prompt."""
    base = df[df["technique_id"] == "baseline"]["word_count"].mean()
    g = (
        df.groupby(["technique_id", "technique", "conciseness_level"])["word_count"]
        .mean()
        .reset_index()
        .rename(columns={"word_count": "avg_words"})
    )
    g["reduction_pct"] = (100 * (1 - g["avg_words"] / base)).round(1) if base else 0.0
    g = g.sort_values("reduction_pct", ascending=False).reset_index(drop=True)
    g["rank"] = g.index + 1
    g["avg_words"] = g["avg_words"].round(1)
    return g[["rank", "technique_id", "technique", "conciseness_level", "avg_words", "reduction_pct"]]


def best_recipe(df: pd.DataFrame) -> Optional[dict]:
    eff = technique_effectiveness(df)
    non_base = eff[eff["technique_id"] != "baseline"]
    if non_base.empty:
        return None
    top = non_base.iloc[0]
    return {"technique": str(top["technique"]), "reduction_pct": float(top["reduction_pct"])}


def temperature_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Linear length sensitivity to temperature per provider."""
    rows = []
    for prov, sub in df.groupby("provider"):
        x = sub["temperature"].to_numpy(dtype=float)
        y = sub["word_count"].to_numpy(dtype=float)
        if len(np.unique(x)) >= 2:
            slope = float(np.polyfit(x, y, 1)[0])
            r = float(np.corrcoef(x, y)[0, 1])
        else:
            slope, r = float("nan"), float("nan")
        rows.append({
            "provider": prov,
            "slope_words_per_temp": round(slope, 2),
            "pearson_r": round(r, 3),
        })
    return pd.DataFrame(rows)
