"""Paths and config loaders."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"


def load_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts() -> List[Dict[str, Any]]:
    return load_yaml(CONFIG_DIR / "prompts.yaml")["prompts"]


def load_techniques() -> List[Dict[str, Any]]:
    return load_yaml(CONFIG_DIR / "techniques.yaml")["techniques"]


def load_transparency() -> Dict[str, Any]:
    return load_yaml(CONFIG_DIR / "rlhf_transparency.yaml")
