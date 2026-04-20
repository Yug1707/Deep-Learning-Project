"""Build balanced top-k manifests for standalone AST training."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from training.ast_pipeline.common import (
    DEFAULT_CONFIG,
    ensure_output_dirs,
    load_config,
    save_json,
    set_seed,
)


def get_audio_path(track_id: int, audio_dir: str) -> Path:
    """Return expected FMA file path for track id."""
    track_str = f"{track_id:06d}"
    return Path(audio_dir) / track_str[:3] / f"{track_str}.mp3"


def build_balanced_subset_indices(labels: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Select rows while capping each class to the minimum positive support."""
    positive_counts = labels.sum(axis=0).astype(int)
    if np.any(positive_counts <= 0):
        raise ValueError("Cannot balance labels when one or more classes have zero positives")

    target = int(positive_counts.min())
    num_classes = labels.shape[1]

    selected = np.zeros(labels.shape[0], dtype=bool)
    class_counts = np.zeros(num_classes, dtype=np.int64)

    rng = np.random.default_rng(seed)
    for _ in range(6):
        changed = False
        for idx in rng.permutation(labels.shape[0]):
            if selected[idx]:
                continue

            pos = np.where(labels[idx] == 1)[0]
            if pos.size == 0:
                continue
            if np.any(class_counts[pos] >= target):
                continue

            selected[idx] = True
            class_counts[pos] += 1
            changed = True

        if np.all(class_counts >= target) or not changed:
            break

    selected_idx = np.where(selected)[0].astype(np.int64)
    return selected_idx, class_counts, target


def _safe_stratify_targets(labels: np.ndarray) -> np.ndarray | None:
    primary = np.array([int(np.where(row == 1)[0][0]) for row in labels], dtype=np.int64)
    counts = np.bincount(primary, minlength=labels.shape[1])
    if np.any(counts < 2):
        return None
    return primary


def _split_indices(
    labels: np.ndarray,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must be 1.0")

    all_indices = np.arange(labels.shape[0])
    stratify_all = _safe_stratify_targets(labels)

    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=(1.0 - train_size),
        random_state=seed,
        stratify=stratify_all,
    )

    temp_labels = labels[temp_idx]
    stratify_temp = _safe_stratify_targets(temp_labels)
    val_ratio_within_temp = val_size / (val_size + test_size)

    val_idx_rel, test_idx_rel = train_test_split(
        np.arange(temp_idx.shape[0]),
        test_size=(1.0 - val_ratio_within_temp),
        random_state=seed,
        stratify=stratify_temp,
    )

    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]

    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Split configuration produced an empty partition")

    return train_idx, val_idx, test_idx


def _to_manifest(records: Sequence[Dict], indices: np.ndarray, num_classes: int) -> pd.DataFrame:
    rows: List[Dict] = []
    for idx in indices:
        rec = records[int(idx)]
        row = {
            "track_id": rec["track_id"],
            "audio_path": rec["audio_path"],
        }
        for class_idx in range(num_classes):
            row[f"label_{class_idx}"] = int(rec["labels"][class_idx])
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AST manifests")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to AST pipeline config")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))

    dirs = ensure_output_dirs(config)
    manifests_dir = dirs["manifests_dir"]

    dataset_cfg = config["dataset"]
    metadata_csv = Path(config["paths"]["metadata_csv"])
    audio_dir = Path(config["paths"]["audio_dir"])

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_csv}")

    tracks = pd.read_csv(metadata_csv, index_col=0, header=[0, 1])
    subset = tracks[tracks[("set", "subset")] == dataset_cfg["subset"]]

    genres_raw = subset[("track", "genres_all")].dropna()
    genres_parsed = genres_raw.apply(ast.literal_eval)

    genre_counter: Counter = Counter()
    for genre_list in genres_parsed:
        genre_counter.update(int(g) for g in genre_list)

    top_k = int(dataset_cfg["top_k_genres"])
    top_genres = [genre_id for genre_id, _ in genre_counter.most_common(top_k)]
    genre_to_index = {genre_id: idx for idx, genre_id in enumerate(top_genres)}

    records: List[Dict] = []
    for track_id, genre_list in genres_parsed.items():
        audio_path = get_audio_path(int(track_id), str(audio_dir))
        if not audio_path.exists():
            continue

        label_vec = np.zeros(top_k, dtype=np.int64)
        for genre_id in genre_list:
            if genre_id in genre_to_index:
                label_vec[genre_to_index[genre_id]] = 1

        if label_vec.sum() == 0:
            continue

        records.append(
            {
                "track_id": int(track_id),
                "audio_path": str(audio_path),
                "labels": label_vec,
            }
        )

    if not records:
        raise RuntimeError("No valid records found after filtering")

    labels = np.stack([rec["labels"] for rec in records], axis=0)
    positive_counts_before = labels.sum(axis=0).astype(int)
    valid_class_indices = np.where(positive_counts_before > 0)[0]

    if valid_class_indices.size == 0:
        raise RuntimeError("No positive labels remain after audio filtering")

    if valid_class_indices.size < labels.shape[1]:
        labels = labels[:, valid_class_indices]
        top_genres = [top_genres[int(i)] for i in valid_class_indices]
        top_k = len(top_genres)
        for rec_idx, rec in enumerate(records):
            records[rec_idx]["labels"] = rec["labels"][valid_class_indices]

    if bool(dataset_cfg["balancing"].get("enabled", False)):
        selected_idx, class_counts, target = build_balanced_subset_indices(labels, seed=int(config["seed"]))
        records = [records[i] for i in selected_idx]
        labels = np.stack([rec["labels"] for rec in records], axis=0)
    else:
        class_counts = labels.sum(axis=0).astype(int)
        target = int(class_counts.min())

    train_idx, val_idx, test_idx = _split_indices(
        labels=labels,
        train_size=float(dataset_cfg["train_size"]),
        val_size=float(dataset_cfg["val_size"]),
        test_size=float(dataset_cfg["test_size"]),
        seed=int(config["seed"]),
    )

    train_manifest = _to_manifest(records, train_idx, top_k)
    val_manifest = _to_manifest(records, val_idx, top_k)
    test_manifest = _to_manifest(records, test_idx, top_k)

    train_manifest.to_csv(manifests_dir / "train_manifest.csv", index=False)
    val_manifest.to_csv(manifests_dir / "val_manifest.csv", index=False)
    test_manifest.to_csv(manifests_dir / "test_manifest.csv", index=False)

    class_mapping = {
        "top_k_genres": top_k,
        "genre_ids": top_genres,
        "index_to_genre": {str(idx): str(genre_id) for idx, genre_id in enumerate(top_genres)},
        "genre_to_index": {str(genre_id): idx for idx, genre_id in enumerate(top_genres)},
    }
    save_json(class_mapping, manifests_dir / "class_mapping.json")

    summary = {
        "total_records": int(labels.shape[0]),
        "class_positive_counts": class_counts.tolist(),
        "target_positive_count": int(target),
        "split_sizes": {
            "train": int(train_manifest.shape[0]),
            "val": int(val_manifest.shape[0]),
            "test": int(test_manifest.shape[0]),
        },
        "manifests_dir": str(manifests_dir),
    }
    save_json(summary, manifests_dir / "dataset_summary.json")

    print("AST manifests created successfully")
    print(summary)


if __name__ == "__main__":
    main()
