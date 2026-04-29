"""Configuration for the AST web application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from training.ast_pipeline.common import load_config, project_root


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


@dataclass(frozen=True)
class AppConfig:
    """Runtime settings for the web app."""

    root: Path
    ast_config_path: Path
    ast_config: Dict[str, Any]
    base_checkpoint_path: Path
    runtime_root: Path
    uploads_dir: Path
    predictions_path: Path
    feedback_log_path: Path
    feedback_buffer_path: Path
    model_state_path: Path
    continual_runs_dir: Path
    feedback_trigger_size: int
    continual_epochs: int
    continual_batch_size: int
    continual_learning_rate: float
    continual_weight_decay: float
    continual_freeze_backbone: bool
    continual_max_grad_norm: float


def load_app_config() -> AppConfig:
    """Load app configuration from environment variables and the AST config."""
    root = project_root()
    ast_config_path = _resolve(
        root,
        os.getenv("AST_APP_CONFIG", "training/ast_pipeline/config_ast_pipeline.json"),
    )
    ast_config = load_config(ast_config_path)

    paths = ast_config["paths"]
    base_checkpoint_path = _resolve(
        root,
        os.getenv(
            "AST_APP_CHECKPOINT",
            str(Path(paths["checkpoints_dir"]) / "best.pt"),
        ),
    )
    runtime_root = _resolve(root, os.getenv("AST_APP_RUNTIME_DIR", "logs/app"))

    return AppConfig(
        root=root,
        ast_config_path=ast_config_path,
        ast_config=ast_config,
        base_checkpoint_path=base_checkpoint_path,
        runtime_root=runtime_root,
        uploads_dir=runtime_root / "uploads",
        predictions_path=runtime_root / "predictions.jsonl",
        feedback_log_path=runtime_root / "feedback_log.jsonl",
        feedback_buffer_path=runtime_root / "feedback_buffer.jsonl",
        model_state_path=runtime_root / "model_state.json",
        continual_runs_dir=runtime_root / "continual_runs",
        feedback_trigger_size=max(1, int(os.getenv("AST_FEEDBACK_TRIGGER_SIZE", "8"))),
        continual_epochs=max(1, int(os.getenv("AST_CONTINUAL_EPOCHS", "2"))),
        continual_batch_size=max(1, int(os.getenv("AST_CONTINUAL_BATCH_SIZE", "2"))),
        continual_learning_rate=float(os.getenv("AST_CONTINUAL_LR", "1e-5")),
        continual_weight_decay=float(os.getenv("AST_CONTINUAL_WEIGHT_DECAY", "0.0")),
        continual_freeze_backbone=_env_bool("AST_CONTINUAL_FREEZE_BACKBONE", True),
        continual_max_grad_norm=float(os.getenv("AST_CONTINUAL_MAX_GRAD_NORM", "1.0")),
    )


def ensure_runtime_dirs(config: AppConfig) -> None:
    """Create runtime directories used by the web app."""
    for path in [
        config.runtime_root,
        config.uploads_dir,
        config.continual_runs_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

