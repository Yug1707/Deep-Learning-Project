"""Orchestrate the full VGGish + XGBoost pipeline stages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_CONFIG = "training/vggish_xgboost/config_vggish_xgboost.json"


def _run(script_path: Path, config_path: str) -> None:
    command: List[str] = [sys.executable, str(script_path), "--config", config_path]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VGGish + XGBoost pipeline stages")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to VGGish pipeline config",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "build", "extract", "train", "eval"],
        help="Pipeline stage to run",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent

    stage_map = {
        "build": [base / "build_dataset.py"],
        "extract": [base / "extract_embeddings.py"],
        "train": [base / "train_xgboost.py"],
        "eval": [base / "evaluate_xgboost.py"],
        "all": [
            base / "build_dataset.py",
            base / "extract_embeddings.py",
            base / "train_xgboost.py",
            base / "evaluate_xgboost.py",
        ],
    }

    for script in stage_map[args.stage]:
        _run(script, args.config)

    print("Pipeline stage execution complete")


if __name__ == "__main__":
    main()
