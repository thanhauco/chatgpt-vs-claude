import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verbosity_lab.metrics import answer_coverage, compute_metrics  # noqa: E402


def test_word_and_sentence_count():
    m = compute_metrics("Paris is the capital. It is in France.")
    assert m["sentence_count"] == 2
    assert m["word_count"] == 8


def test_hedge_and_caveat_detection():
    text = "It's important to note that this might generally depend on context."
    m = compute_metrics(text)
    assert m["caveat_count"] >= 1
    assert m["hedge_count"] >= 1
    assert 0 <= m["hedge_density"] <= 1


def test_empty_text_is_safe():
    m = compute_metrics("")
    assert m["word_count"] == 0
    assert m["type_token_ratio"] == 0


def test_list_items_counted():
    m = compute_metrics("- one\n- two\n- three")
    assert m["list_item_count"] == 3


def test_answer_coverage_full_and_partial():
    assert answer_coverage("Paris is the capital of France.", "Paris.") == 1.0
    partial = answer_coverage(
        "Something completely unrelated.",
        "Run git reset soft HEAD keeps changes staged",
    )
    assert partial < 0.5


def test_answer_coverage_empty_core_is_full():
    assert answer_coverage("anything", "") == 1.0
