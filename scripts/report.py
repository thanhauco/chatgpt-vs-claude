"""Generate a console + self-contained HTML verbosity comparison report."""
import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verbosity_lab import analysis, config, scoring  # noqa: E402

CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;max-width:920px;color:#1c1c1c;padding:0 1rem}
h1{margin-bottom:0}.sub{color:#666;margin-bottom:1.5rem}
h2{margin-top:2rem;border-bottom:2px solid #eee;padding-bottom:4px}
table{border-collapse:collapse;width:100%;margin:1rem 0}
th,td{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:14px}
th{background:#f4f4f4}
.callout{background:#eefaf2;border-left:4px solid #10a37f;padding:10px 14px;margin:1rem 0}
.disclaimer{color:#777;font-size:12px;margin-top:2rem;border-top:1px solid #eee;padding-top:1rem}
""".strip()


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0)


def main() -> None:
    p = argparse.ArgumentParser(description="Build a verbosity comparison report.")
    p.add_argument("--input", default=str(config.DATA_DIR / "sample_results.csv"))
    p.add_argument("--out", default=str(ROOT / "reports" / "verbosity_report.html"))
    args = p.parse_args()

    df = scoring.add_padding_score(pd.read_csv(args.input))
    summary = scoring.summarize(df)
    cmp = analysis.compare_providers(df, "word_count")
    eff = analysis.technique_effectiveness(df)
    trend = analysis.temperature_trend(df)
    transparency, cfg = scoring.transparency_table()
    recipe = analysis.best_recipe(df)

    # --- Console ---------------------------------------------------------- #
    print("\n=== Verbosity leaderboard ===")
    print(summary.to_string(index=False))
    if cmp:
        more = cmp["a"] if cmp["diff"] > 0 else cmp["b"]
        less = cmp["b"] if cmp["diff"] > 0 else cmp["a"]
        print(
            f"\n{more} averages {abs(cmp['diff']):.0f} more words than {less} "
            f"(Cohen's d={cmp['cohens_d']}, {cmp['effect']} effect)."
        )
    if recipe:
        print(f"Best technique: {recipe['technique']} (~{recipe['reduction_pct']:.0f}% shorter).")

    # --- HTML ------------------------------------------------------------- #
    narrative = "Not enough data for a provider comparison."
    if cmp:
        more = cmp["a"] if cmp["diff"] > 0 else cmp["b"]
        less = cmp["b"] if cmp["diff"] > 0 else cmp["a"]
        narrative = (
            f"On identical prompts, <b>{more}</b> averages "
            f"<b>{abs(cmp['diff']):.0f} more words</b> ({cmp['pct_diff']:+.0f}%) than "
            f"<b>{less}</b> — a <i>{cmp['effect']}</i> effect (Cohen's d = {cmp['cohens_d']})."
        )
    recipe_html = ""
    if recipe:
        recipe_html = (
            f"<div class='callout'><b>Recommended recipe:</b> "
            f"\u201c{recipe['technique']}\u201d reduces output by "
            f"~{recipe['reduction_pct']:.0f}% versus an unconstrained prompt.</div>"
        )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>ChatGPT vs Claude \u2014 Verbosity Report</title>
<style>{CSS}</style></head>
<body>
<h1>ChatGPT vs Claude \u2014 Verbosity Report</h1>
<div class="sub">Generated {date.today().isoformat()} \u00b7 {len(df)} responses</div>
<p>{narrative}</p>
{recipe_html}
<h2>Verbosity leaderboard</h2>{_table(summary)}
<h2>Most effective conciseness techniques</h2>{_table(eff[['rank', 'technique', 'avg_words', 'reduction_pct']])}
<h2>Temperature sensitivity</h2>{_table(trend)}
<h2>RLHF transparency scorecard</h2>{_table(transparency.drop(columns=['description']))}
<div class="disclaimer">{cfg.get('meta', {}).get('disclaimer', '')}
Data source: {Path(args.input).name}. The default sample data is synthetic.</div>
</body></html>"""

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\nHTML report -> {out}")


if __name__ == "__main__":
    main()
