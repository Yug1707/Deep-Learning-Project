"""Orchestrate standalone AST pipeline stages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

try:
    from training.ast_pipeline.common import DEFAULT_CONFIG
except ImportError:  # pragma: no cover - script execution fallback
    from common import DEFAULT_CONFIG


def _project_root() -> Path:
    """Return repository root from this script location."""
    return Path(__file__).resolve().parents[2]


def _resolve_config_path(config_arg: str) -> str:
    """Resolve config path against caller cwd, then return absolute string path."""
    cfg = Path(config_arg)
    if not cfg.is_absolute():
        cfg = Path.cwd() / cfg
    return str(cfg.resolve())


def _run(module_name: str, args: List[str]) -> None:
    command: List[str] = [sys.executable, "-m", module_name] + args
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True, cwd=_project_root())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AST pipeline stages")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to AST config")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "build", "train", "eval", "predict"],
        help="Pipeline stage to run",
    )
    parser.add_argument("--audio", type=str, default=None, help="Single audio path for predict stage")
    parser.add_argument("--audio-dir", type=str, default=None, help="Audio directory for predict stage")
    parser.add_argument("--manifest", type=str, default=None, help="Manifest CSV for predict stage")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint for eval/predict")
    args = parser.parse_args()

    resolved_config = _resolve_config_path(args.config)

    stage_map = {
        "build": ["training.ast_pipeline.build_dataset"],
        "train": ["training.ast_pipeline.train_ast"],
        "eval": ["training.ast_pipeline.evaluate_ast"],
        "predict": ["training.ast_pipeline.predict_ast"],
        "all": [
            "training.ast_pipeline.build_dataset",
            "training.ast_pipeline.train_ast",
            "training.ast_pipeline.evaluate_ast",
        ],
    }

    for module in stage_map[args.stage]:
        command_args: List[str] = ["--config", resolved_config]

        if module.endswith("evaluate_ast") and args.checkpoint:
            command_args.extend(["--checkpoint", args.checkpoint])

        if module.endswith("predict_ast"):
            if args.checkpoint:
                command_args.extend(["--checkpoint", args.checkpoint])
            if args.audio:
                command_args.extend(["--audio", args.audio])
            if args.audio_dir:
                command_args.extend(["--audio-dir", args.audio_dir])
            if args.manifest:
                command_args.extend(["--manifest", args.manifest])

        _run(module, command_args)

    print("AST pipeline stage execution complete")


if __name__ == "__main__":
    main()
