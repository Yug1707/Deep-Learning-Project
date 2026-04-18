"""Extract VGGish embeddings and save pooled features by split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from training.vggish_xgboost.common import ensure_dir, load_config, project_root, save_json, set_seed
from utils.vggish_extractor import VGGishExtractor, pool_embeddings
from utils.vggish_audio import get_audio_duration, load_audio_16k_mono


def _label_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("label_")]


def _extract_split(
    manifest_path: Path,
    split: str,
    extractor: VGGishExtractor,
    sample_rate: int,
    res_type: str,
    min_audio_seconds: float,
    pooling: str,
    frames_dir: Path,
    pooled_dir: Path,
) -> Dict[str, int]:
    df = pd.read_csv(manifest_path)
    label_cols = _label_columns(df)

    pooled_vectors: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    track_ids: List[int] = []
    failures: List[Dict[str, str]] = []
    index_rows: List[Dict[str, str]] = []

    for row in tqdm(df.itertuples(index=False), total=df.shape[0], desc=f"Extracting {split}"):
        track_id = int(row.track_id)
        audio_path = Path(row.audio_path)

        try:
            waveform = load_audio_16k_mono(
                audio_path,
                sample_rate=sample_rate,
                res_type=res_type,
            )
            duration = get_audio_duration(waveform, sample_rate)
            if duration < min_audio_seconds:
                raise ValueError(
                    f"Audio too short ({duration:.2f}s < {min_audio_seconds:.2f}s minimum)"
                )

            frames = extractor.extract_from_waveform(waveform, sample_rate=sample_rate)
            if frames.size == 0:
                raise ValueError("No VGGish frames extracted from audio")

            pooled = pool_embeddings(frames, mode=pooling)

            frame_path = frames_dir / f"{split}_{track_id}.npy"
            np.save(frame_path, frames)

            pooled_vectors.append(pooled)
            labels.append(np.array([getattr(row, col) for col in label_cols], dtype=np.float32))
            track_ids.append(track_id)

            index_rows.append(
                {
                    "split": split,
                    "track_id": track_id,
                    "audio_path": str(audio_path),
                    "frame_embedding_path": str(frame_path),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "split": split,
                    "track_id": track_id,
                    "audio_path": str(audio_path),
                    "error": str(exc),
                }
            )

    if pooled_vectors:
        X = np.stack(pooled_vectors, axis=0).astype(np.float32)
        y = np.stack(labels, axis=0).astype(np.float32)
    else:
        X = np.zeros((0, 128), dtype=np.float32)
        y = np.zeros((0, len(label_cols)), dtype=np.float32)

    np.savez_compressed(
        pooled_dir / f"{split}_pooled.npz",
        X=X,
        y=y,
        track_ids=np.array(track_ids, dtype=np.int64),
    )

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(pooled_dir / f"{split}_index.csv", index=False)

    if failures:
        pd.DataFrame(failures).to_csv(pooled_dir / f"{split}_failures.csv", index=False)

    return {
        "split": split,
        "input_rows": int(df.shape[0]),
        "succeeded": int(X.shape[0]),
        "failed": int(len(failures)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract VGGish embeddings for train/val/test splits")
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
    audio_cfg = config["audio"]
    vggish_cfg = config["vggish"]

    output_root = root / paths["output_root"]
    manifests_dir = output_root / "manifests"

    embeddings_dir = ensure_dir(root / paths["embeddings_dir"])
    frames_dir = ensure_dir(embeddings_dir / "frames")
    pooled_dir = ensure_dir(embeddings_dir / "pooled")

    extractor = VGGishExtractor(device=vggish_cfg.get("device", "auto"))

    summary_rows: List[Dict[str, int]] = []
    for split in ["train", "val", "test"]:
        manifest_path = manifests_dir / f"{split}_manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        split_summary = _extract_split(
            manifest_path=manifest_path,
            split=split,
            extractor=extractor,
            sample_rate=int(audio_cfg["sample_rate"]),
            res_type=str(audio_cfg.get("res_type", "soxr_hq")),
            min_audio_seconds=float(config["dataset"].get("min_audio_seconds", 0.0)),
            pooling=vggish_cfg.get("pooling", "mean"),
            frames_dir=frames_dir,
            pooled_dir=pooled_dir,
        )
        summary_rows.append(split_summary)

    summary = {"splits": summary_rows}
    save_json(summary, embeddings_dir / "extraction_summary.json")

    print("Embedding extraction complete")
    print(summary)


if __name__ == "__main__":
    main()
