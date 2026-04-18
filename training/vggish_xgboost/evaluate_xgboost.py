"""Evaluate trained XGBoost model on validation/test pooled embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score, multilabel_confusion_matrix

from training.vggish_xgboost.common import ensure_dir, load_config, project_root, save_json
from utils.metrics import MultiLabelMetrics


def _load_split(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {
        "X": data["X"],
        "y": data["y"],
        "track_ids": data["track_ids"],
    }


def _evaluate_split(
    split: str,
    model,
    data: Dict[str, np.ndarray],
    threshold: float,
    class_map: Dict[str, str],
) -> Dict:
    if data["X"].shape[0] == 0:
        return {
            "split": split,
            "num_samples": 0,
            "metrics": MultiLabelMetrics(threshold=threshold).calculate_all_metrics(
                predictions=torch.zeros((0, data["y"].shape[1]), dtype=torch.float32),
                targets=torch.zeros((0, data["y"].shape[1]), dtype=torch.float32),
            ),
            "per_class_confusion": [],
        }

    probs = model.predict_proba(data["X"])
    y_true = (data["y"] >= 0.5).astype(np.int64)
    preds = (probs >= threshold).astype(np.int64)

    metrics_calc = MultiLabelMetrics(threshold=threshold)
    metrics = metrics_calc.calculate_all_metrics(
        predictions=torch.tensor(probs, dtype=torch.float32),
        targets=torch.tensor(y_true, dtype=torch.float32),
    )

    confusion = multilabel_confusion_matrix(y_true, preds)
    per_class: List[Dict] = []
    for class_idx, matrix in enumerate(confusion):
        tn, fp, fn, tp = matrix.ravel().tolist()
        class_f1 = f1_score(
            y_true[:, class_idx],
            preds[:, class_idx],
            zero_division=0,
        )
        per_class.append(
            {
                "class_index": class_idx,
                "genre_id": class_map.get(str(class_idx), str(class_idx)),
                "support": int(y_true[:, class_idx].sum()),
                "f1": float(class_f1),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    return {
        "split": split,
        "num_samples": int(data["X"].shape[0]),
        "metrics": metrics,
        "per_class_confusion": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate One-vs-Rest XGBoost model")
    parser.add_argument(
        "--config",
        type=str,
        default="training/vggish_xgboost/config_vggish_xgboost.json",
        help="Path to VGGish pipeline config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    root = project_root()

    paths = config["paths"]
    threshold = float(config["evaluation"]["threshold"])

    pooled_dir = root / paths["embeddings_dir"] / "pooled"
    models_dir = root / paths["models_dir"]
    results_dir = ensure_dir(root / paths["results_dir"])
    manifests_dir = root / paths["output_root"] / "manifests"

    model = joblib.load(models_dir / "xgboost_ovr.joblib")

    class_mapping = load_config(manifests_dir / "class_mapping.json")
    class_map = class_mapping.get("index_to_genre", {})

    report: Dict[str, Dict] = {}
    for split in ["val", "test"]:
        split_data = _load_split(pooled_dir / f"{split}_pooled.npz")
        report[split] = _evaluate_split(split, model, split_data, threshold, class_map)

    save_json(report, results_dir / "evaluation_report.json")

    print("Evaluation complete")
    print({k: v["metrics"] for k, v in report.items()})


if __name__ == "__main__":
    main()
