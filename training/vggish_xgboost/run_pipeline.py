"""Orchestrate the full VGGish + XGBoost pipeline stages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List


DEFAULT_CONFIG = "training/vggish_xgboost/config_vggish_xgboost.json"


def _run(module_name: str, config_path: str) -> None:
    command: List[str] = [sys.executable, "-m", module_name, "--config", config_path]
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

    stage_map = {
        "build": ["training.vggish_xgboost.build_dataset"],
        "extract": ["training.vggish_xgboost.extract_embeddings"],
        "train": ["training.vggish_xgboost.train_xgboost"],
        "eval": ["training.vggish_xgboost.evaluate_xgboost"],
        "all": [
            "training.vggish_xgboost.build_dataset",
            "training.vggish_xgboost.extract_embeddings",
            "training.vggish_xgboost.train_xgboost",
            "training.vggish_xgboost.evaluate_xgboost",
        ],
    }

    for script in stage_map[args.stage]:
        _run(script, args.config)

    print("Pipeline stage execution complete")


if __name__ == "__main__":
    main()
