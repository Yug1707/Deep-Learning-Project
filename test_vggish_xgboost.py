"""Smoke tests for VGGish + XGBoost pipeline components."""

from __future__ import annotations

import numpy as np

from utils.vggish_extractor import pool_embeddings
from utils.metrics import MultiLabelMetrics
from training.vggish_xgboost.build_dataset import build_balanced_subset_indices


def test_pool_embeddings() -> None:
    frames = np.random.randn(32, 128).astype(np.float32)
    pooled_mean = pool_embeddings(frames, mode="mean")
    pooled_max = pool_embeddings(frames, mode="max")

    assert pooled_mean.shape == (128,)
    assert pooled_max.shape == (128,)


def test_balancer() -> None:
    labels = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.int64,
    )

    selected_idx, class_counts, target = build_balanced_subset_indices(labels, seed=42)

    assert selected_idx.size > 0
    assert target > 0
    assert np.all(class_counts <= target)

    # Regression check: reported class counts must match selected samples.
    recomputed = labels[selected_idx].sum(axis=0)
    assert np.array_equal(class_counts, recomputed)


def test_metrics_precision_at_k_small_class_count() -> None:
    try:
        import torch
    except ImportError:
        print("Skipping metrics top-k test: torch not installed")
        return

    metrics = MultiLabelMetrics(threshold=0.5)

    # 8 classes only, while calculate_all_metrics internally asks for @10.
    predictions = torch.rand(12, 8)
    targets = torch.randint(0, 2, (12, 8)).float()

    output = metrics.calculate_all_metrics(predictions, targets)

    assert "precision_at_5" in output
    assert "precision_at_10" in output
    assert 0.0 <= output["precision_at_5"] <= 1.0
    assert 0.0 <= output["precision_at_10"] <= 1.0


def test_xgboost_fit_predict() -> None:
    try:
        from sklearn.multiclass import OneVsRestClassifier
        from xgboost import XGBClassifier
    except ImportError:
        print("Skipping XGBoost smoke test: dependencies not installed")
        return

    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 128)).astype(np.float32)
    y = np.zeros((80, 3), dtype=np.int64)

    y[:, 0] = (X[:, 0] > 0.0).astype(np.int64)
    y[:, 1] = (X[:, 1] + X[:, 2] > 0.3).astype(np.int64)
    y[:, 2] = (X[:, 3] - X[:, 4] > -0.2).astype(np.int64)

    model = OneVsRestClassifier(
        XGBClassifier(
            objective="binary:logistic",
            n_estimators=30,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            tree_method="hist",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
    )

    model.fit(X[:60], y[:60])
    probs = model.predict_proba(X[60:])

    assert probs.shape == (20, 3)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def run_all_tests() -> None:
    test_pool_embeddings()
    print("[PASS] pooling")

    test_balancer()
    print("[PASS] balancing")

    test_metrics_precision_at_k_small_class_count()
    print("[PASS] metrics top-k")

    test_xgboost_fit_predict()
    print("[PASS] xgboost smoke")

    print("All VGGish + XGBoost smoke tests passed")


if __name__ == "__main__":
    run_all_tests()
