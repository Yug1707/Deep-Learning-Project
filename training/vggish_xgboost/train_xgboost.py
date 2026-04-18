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


def _validate_split(name: str, split: Dict[str, np.ndarray]) -> None:
    X = split["X"]
    y = split["y"]

    if X.ndim != 2:
        raise ValueError(f"{name} X must be 2D, got shape {X.shape}")
    if y.ndim != 2:
        raise ValueError(f"{name} y must be 2D, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{name} X/y row mismatch: {X.shape[0]} vs {y.shape[0]}")


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

    _validate_split("train", train)
    _validate_split("val", val)

    if train["X"].shape[0] == 0:
        raise ValueError("Train split is empty. Run extraction and check failures before training")

    if train["y"].shape[1] == 0:
        raise ValueError("Train labels have zero classes")

    train_y_int = (train["y"] >= 0.5).astype(np.int64)
    val_y_int = (val["y"] >= 0.5).astype(np.int64)

    class_positives = train_y_int.sum(axis=0)
    no_positive = np.where(class_positives == 0)[0]
    all_positive = np.where(class_positives == train_y_int.shape[0])[0]
    if no_positive.size > 0:
        raise ValueError(
            f"Cannot train: classes with no positive samples in train split: {no_positive.tolist()}"
        )
    if all_positive.size > 0:
        raise ValueError(
            f"Cannot train: classes with no negative samples in train split: {all_positive.tolist()}"
        )

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
    model.fit(train["X"], train_y_int)

    if val["X"].shape[0] > 0:
        val_probs = model.predict_proba(val["X"])
    else:
        val_probs = np.zeros((0, train_y_int.shape[1]), dtype=np.float32)

    metrics_calc = MultiLabelMetrics(threshold=threshold)
    val_metrics = metrics_calc.calculate_all_metrics(
        predictions=torch.tensor(val_probs, dtype=torch.float32),
        targets=torch.tensor(val_y_int, dtype=torch.float32),
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
