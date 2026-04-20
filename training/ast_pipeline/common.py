"""Shared helpers for the standalone AST pipeline."""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


DEFAULT_CONFIG = "training/ast_pipeline/config_ast_pipeline.json"


def project_root() -> Path:
    """Return repository root path from this module location."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load JSON config and normalize path entries to absolute paths."""
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path

    with open(cfg_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    paths = config.get("paths", {})
    root = project_root()
    resolved_paths: Dict[str, str] = {}
    for key, value in paths.items():
        p = Path(value)
        if not p.is_absolute():
            p = root / p
        resolved_paths[key] = str(p)

    config["paths"] = resolved_paths
    config["_config_path"] = str(cfg_path)
    return config


def ensure_dir(path: str | Path) -> Path:
    """Create directory and return its resolved path."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_output_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    """Create and return all AST output directories."""
    paths = config["paths"]
    required = [
        "output_root",
        "manifests_dir",
        "checkpoints_dir",
        "models_dir",
        "results_dir",
        "predictions_dir",
        "runtime_logs_dir",
    ]

    resolved: Dict[str, Path] = {}
    for key in required:
        resolved[key] = ensure_dir(paths[key])
    return resolved


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set reproducible random seeds across CPU/GPU stacks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    except Exception:
        pass


def save_json(payload: Dict[str, Any], output_path: str | Path) -> None:
    """Write dictionary payload as formatted JSON."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def configure_logger(log_file: str | Path, logger_name: str = "ast_pipeline") -> logging.Logger:
    """Configure console + file logger for AST stages."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def save_run_snapshot(config: Dict[str, Any], results_dir: str | Path) -> Path:
    """Persist resolved runtime configuration for reproducibility."""
    snapshot = dict(config)
    snapshot["run_started_utc"] = datetime.utcnow().isoformat() + "Z"

    output_path = Path(results_dir) / "run_config_snapshot.json"
    save_json(snapshot, output_path)
    return output_path
