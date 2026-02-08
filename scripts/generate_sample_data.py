"""Generate synthetic comparison data with offline mock providers (no API keys)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verbosity_lab import config  # noqa: E402
from verbosity_lab.experiment import run_experiment  # noqa: E402
from verbosity_lab.providers import MockProvider  # noqa: E402


def main() -> None:
    prompts = config.load_prompts()
    techniques = config.load_techniques()
    temperatures = [0.0, 0.2, 0.5, 0.7, 1.0]
    providers = [
        # base_verbosity models a product-level default: GPT tuned slightly longer.
        MockProvider("openai", "gpt-mock", base_verbosity=6),
        MockProvider("anthropic", "claude-mock", base_verbosity=4),
    ]

    print("Generating synthetic comparison data (mock providers)...")
    df = run_experiment(providers, prompts, techniques, temperatures, repeats=2)

    out = config.DATA_DIR / "sample_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
