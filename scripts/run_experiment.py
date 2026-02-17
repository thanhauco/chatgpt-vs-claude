"""Run the verbosity comparison experiment against mock or real providers."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from verbosity_lab import config  # noqa: E402
from verbosity_lab.experiment import run_experiment  # noqa: E402
from verbosity_lab.providers import (  # noqa: E402
    AnthropicProvider,
    MockProvider,
    OpenAIProvider,
)


def build_providers(names):
    out = []
    for n in names:
        if n == "openai":
            out.append(OpenAIProvider())
        elif n == "anthropic":
            out.append(AnthropicProvider())
        elif n == "mock-openai":
            out.append(MockProvider("openai", "gpt-mock", base_verbosity=6))
        elif n == "mock-anthropic":
            out.append(MockProvider("anthropic", "claude-mock", base_verbosity=4))
        else:
            raise SystemExit(f"Unknown provider: {n}")
    return out


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description="Run the verbosity comparison experiment.")
    p.add_argument(
        "--providers", nargs="+", default=["mock-openai", "mock-anthropic"],
        help="Any of: openai anthropic mock-openai mock-anthropic",
    )
    p.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.2, 0.5, 0.7, 1.0])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--out", default=str(config.DATA_DIR / "results.csv"))
    args = p.parse_args()

    providers = build_providers(args.providers)
    df = run_experiment(
        providers,
        config.load_prompts(),
        config.load_techniques(),
        args.temperatures,
        repeats=args.repeats,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
