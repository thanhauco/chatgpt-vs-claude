"""Experiment runner: prompts x techniques x temperatures x providers."""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

import pandas as pd

from .metrics import compute_metrics
from .providers import Provider


def run_experiment(
    providers: Sequence[Provider],
    prompts: List[Dict],
    techniques: List[Dict],
    temperatures: Sequence[float],
    repeats: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    rows = []
    combos = list(itertools.product(providers, prompts, techniques, temperatures, range(repeats)))
    total = len(combos)

    for i, (prov, prompt, tech, temp, rep) in enumerate(combos, 1):
        full_prompt = (prompt["prompt"] + tech.get("suffix", "")).strip()
        result = prov.generate(
            full_prompt,
            temperature=temp,
            core_answer=prompt.get("core_answer", "Here is the answer."),
            conciseness_level=tech.get("conciseness_level", 0),
            technique_id=tech["id"],
            seed=rep,
        )
        row = {
            "provider": result.provider,
            "model": result.model,
            "prompt_id": prompt["id"],
            "category": prompt.get("category", "general"),
            "technique_id": tech["id"],
            "technique": tech.get("name", tech["id"]),
            "conciseness_level": tech.get("conciseness_level", 0),
            "temperature": temp,
            "repeat": rep,
            "latency_s": result.latency_s,
            "response": result.text,
        }
        row.update(compute_metrics(result.text))
        rows.append(row)
        if verbose and (i % 50 == 0 or i == total):
            print(f"  [{i}/{total}] generated")

    return pd.DataFrame(rows)
