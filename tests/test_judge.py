import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verbosity_lab.judge import MockJudge  # noqa: E402


def test_mock_judge_ranges():
    r = MockJudge().score("What is the capital of France?", "Paris is the capital of France.")
    for v in (r.relevance, r.completeness, r.conciseness):
        assert 1 <= v <= 5
    assert 1 <= r.overall <= 5


def test_mock_judge_penalizes_padding_on_conciseness():
    j = MockJudge()
    short = j.score("q", "Paris.")
    padded = j.score("q", "Paris. " + "It is important to note this generally depends. " * 6)
    assert short.conciseness >= padded.conciseness


def test_mock_judge_rewards_completeness_with_content():
    j = MockJudge()
    terse = j.score("q", "Yes.")
    fuller = j.score("q", "Yes, because the configuration enables the feature across all environments here.")
    assert fuller.completeness >= terse.completeness
