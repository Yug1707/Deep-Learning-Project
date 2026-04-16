"""Train One-vs-Rest XGBoost on pooled VGGish embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import torch
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier

from training.vggish_xgboost.common import ensure_dir, load_config, project_root, save_json, set_seed
from utils.metrics import MultiLabelMetrics


def _load_split(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {
        "X": data["X"],
        "y": data["y"],
        "track_ids": data["track_ids"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train One-vs-Rest XGBoost for multi-label genre classification")
    parser.add_argument(
        "--config",
        type=str,
        default="training/vggish_xgboost/config_vggish_xgboost.json",
        help="Path to VGGish pipeline config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))

    root = project_root()
    paths = config["paths"]
    xgb_cfg = config["xgboost"]
    threshold = float(config["evaluation"]["threshold"])

    pooled_dir = root / paths["embeddings_dir"] / "pooled"
    models_dir = ensure_dir(root / paths["models_dir"])

    train = _load_split(pooled_dir / "train_pooled.npz")
    val = _load_split(pooled_dir / "val_pooled.npz")

    base_estimator = XGBClassifier(
        objective="binary:logistic",
        n_estimators=int(xgb_cfg["n_estimators"]),
        max_depth=int(xgb_cfg["max_depth"]),
        learning_rate=float(xgb_cfg["learning_rate"]),
        subsample=float(xgb_cfg["subsample"]),
        colsample_bytree=float(xgb_cfg["colsample_bytree"]),
        reg_lambda=float(xgb_cfg["reg_lambda"]),
        min_child_weight=float(xgb_cfg["min_child_weight"]),
        random_state=int(xgb_cfg["random_state"]),
        n_jobs=int(xgb_cfg["n_jobs"]),
        eval_metric=xgb_cfg.get("eval_metric", "logloss"),
        tree_method="hist",
    )

    model = OneVsRestClassifier(base_estimator)
    model.fit(train["X"], train["y"])

    val_probs = model.predict_proba(val["X"])

    metrics_calc = MultiLabelMetrics(threshold=threshold)
    val_metrics = metrics_calc.calculate_all_metrics(
        predictions=torch.tensor(val_probs, dtype=torch.float32),
        targets=torch.tensor(val["y"], dtype=torch.float32),
    )

    model_path = models_dir / "xgboost_ovr.joblib"
    joblib.dump(model, model_path)

    summary = {
        "model_path": str(model_path),
        "train_samples": int(train["X"].shape[0]),
        "val_samples": int(val["X"].shape[0]),
        "feature_dim": int(train["X"].shape[1]) if train["X"].size else 0,
        "threshold": threshold,
        "validation_metrics": val_metrics,
    }
    save_json(summary, models_dir / "training_summary.json")

    print("Training complete")
    print(summary)


if __name__ == "__main__":
    main()
