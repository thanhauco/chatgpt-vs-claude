import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verbosity_lab import analysis  # noqa: E402


def _make_df():
    rows = []
    for prov, base in [("openai", 40), ("anthropic", 20)]:
        for tid, words in [("baseline", base), ("be_concise", base // 2)]:
            for temp, w in [(0.0, words - 2), (1.0, words + 2)]:
                rows.append({
                    "provider": prov, "prompt_id": "p1",
                    "technique_id": tid, "technique": tid, "conciseness_level": 0,
                    "temperature": temp, "word_count": w, "token_count": w,
                })
    return pd.DataFrame(rows)


def test_cohens_d_sign_and_magnitude():
    d = analysis.cohens_d([1, 2, 3], [7, 8, 9])
    assert d < 0
    assert abs(d) > 0.8  # large


def test_compare_providers_detects_gap():
    out = analysis.compare_providers(_make_df(), "word_count")
    assert out["a"] == "openai"
    assert out["diff"] > 0  # openai longer in fixture
    assert out["effect"] in {"negligible", "small", "medium", "large"}


def test_technique_effectiveness_baseline_zero():
    eff = analysis.technique_effectiveness(_make_df())
    base_row = eff[eff["technique_id"] == "baseline"].iloc[0]
    assert base_row["reduction_pct"] == 0.0
    assert (eff["reduction_pct"] >= 0).all()


def test_temperature_trend_has_slope():
    trend = analysis.temperature_trend(_make_df())
    assert set(trend["provider"]) == {"openai", "anthropic"}
    assert "slope_words_per_temp" in trend.columns
