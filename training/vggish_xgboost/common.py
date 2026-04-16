"""Shared helpers for the VGGish + XGBoost pipeline."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np


def project_root() -> Path:
    """Return repository root path from this module location."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load JSON configuration file."""
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path

    with open(cfg_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload: Dict[str, Any], output_path: str | Path) -> None:
    """Write dictionary as formatted JSON."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def ensure_dir(path: str | Path) -> Path:
    """Create directory and return its resolved path."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def set_seed(seed: int) -> None:
    """Set reproducible seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
