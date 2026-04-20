"""Data utilities for AST training and batch inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import warnings

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SUPPORTED_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


def label_columns(df: pd.DataFrame) -> List[str]:
    """Return all manifest columns that encode multi-label targets."""
    return [col for col in df.columns if col.startswith("label_")]


def load_audio_file(audio_path: str | Path, sample_rate: int, res_type: str = "soxr_hq") -> np.ndarray:
    """Load audio as mono waveform float32 at target sample rate."""
    path = Path(audio_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Audio file is missing or invalid: {path}")

    with warnings.catch_warnings():
        # librosa may emit noisy backend fallback/deprecation warnings on malformed files.
        warnings.filterwarnings("ignore", message="PySoundFile failed.*")
        warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load.*")
        waveform, _ = librosa.load(
            str(path),
            sr=sample_rate,
            mono=True,
            res_type=res_type,
        )

    if waveform.ndim != 1:
        waveform = np.mean(waveform, axis=0)
    return waveform.astype(np.float32)


def pad_or_trim(waveform: np.ndarray, target_samples: int, start: int = 0) -> np.ndarray:
    """Extract fixed-length crop and zero-pad when the source is shorter."""
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")

    if waveform.size == 0:
        return np.zeros(target_samples, dtype=np.float32)

    if waveform.shape[0] >= target_samples:
        end = min(start + target_samples, waveform.shape[0])
        cropped = waveform[start:end]
        if cropped.shape[0] < target_samples:
            padded = np.zeros(target_samples, dtype=np.float32)
            padded[: cropped.shape[0]] = cropped
            return padded
        return cropped.astype(np.float32)

    padded = np.zeros(target_samples, dtype=np.float32)
    padded[: waveform.shape[0]] = waveform
    return padded


def chunk_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    chunk_seconds: float,
    hop_seconds: float,
    max_chunks: int | None = None,
) -> List[np.ndarray]:
    """Split waveform into fixed windows for scalable inference."""
    chunk_samples = int(round(chunk_seconds * sample_rate))
    hop_samples = int(round(hop_seconds * sample_rate))
    if chunk_samples <= 0 or hop_samples <= 0:
        raise ValueError("chunk_seconds and hop_seconds must be positive")

    if waveform.size == 0:
        return [np.zeros(chunk_samples, dtype=np.float32)]

    if waveform.shape[0] <= chunk_samples:
        return [pad_or_trim(waveform, chunk_samples)]

    chunks: List[np.ndarray] = []
    start = 0
    last_start = waveform.shape[0] - chunk_samples
    while start <= last_start:
        chunks.append(pad_or_trim(waveform, chunk_samples, start=start))
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
        start += hop_samples

    if not chunks:
        chunks.append(pad_or_trim(waveform, chunk_samples, start=0))

    return chunks


@dataclass
class ManifestRow:
    track_id: int
    audio_path: str
    labels: np.ndarray


class ASTManifestDataset(Dataset):
    """Manifest-backed dataset that yields fixed-length waveforms and labels."""

    def __init__(
        self,
        manifest_path: str | Path,
        sample_rate: int,
        clip_seconds: float,
        res_type: str,
        mode: str,
        seed: int,
        min_audio_seconds: float,
    ) -> None:
        if mode not in {"train", "val", "test", "infer"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {self.manifest_path}")

        self.sample_rate = int(sample_rate)
        self.clip_seconds = float(clip_seconds)
        self.clip_samples = int(round(self.sample_rate * self.clip_seconds))
        self.res_type = str(res_type)
        self.mode = mode
        self.seed = int(seed)
        self.min_audio_seconds = float(min_audio_seconds)
        self.min_audio_samples = int(round(self.min_audio_seconds * self.sample_rate))
        self.epoch = 0
        self.max_decode_retries = 3

        df = pd.read_csv(self.manifest_path)
        self._label_cols = label_columns(df)

        rows: List[ManifestRow] = []
        missing_count = 0
        for row in df.itertuples(index=False):
            audio_path = str(getattr(row, "audio_path"))
            audio_path_obj = Path(audio_path)
            if not audio_path_obj.exists() or not audio_path_obj.is_file():
                missing_count += 1
                continue

            labels = (
                np.array([getattr(row, col) for col in self._label_cols], dtype=np.float32)
                if self._label_cols
                else np.zeros((0,), dtype=np.float32)
            )
            rows.append(
                ManifestRow(
                    track_id=int(getattr(row, "track_id", 0)),
                    audio_path=audio_path,
                    labels=labels,
                )
            )
        self.rows = rows

        if not self.rows:
            raise RuntimeError(
                f"No usable rows in manifest {self.manifest_path}. "
                "Check audio paths and dataset files."
            )

        if missing_count > 0:
            warnings.warn(
                f"Skipped {missing_count} missing/invalid files in {self.manifest_path.name}",
                stacklevel=2,
            )

    @property
    def num_labels(self) -> int:
        return len(self._label_cols)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def _select_start(self, length: int, idx: int) -> int:
        if length <= self.clip_samples:
            return 0

        max_start = length - self.clip_samples
        if self.mode == "train":
            rng_seed = self.seed + (self.epoch * 1_000_003) + (idx * 9_973)
            rng = np.random.default_rng(rng_seed)
            return int(rng.integers(0, max_start + 1))

        return max_start // 2

    def _fallback_index(self, idx: int, attempt: int) -> int:
        if len(self.rows) <= 1:
            return idx

        if self.mode == "train":
            rng_seed = self.seed + (self.epoch * 1_000_003) + (idx * 9_973) + (attempt * 101)
            rng = np.random.default_rng(rng_seed)
            return int(rng.integers(0, len(self.rows)))

        return int((idx + attempt) % len(self.rows))

    def __getitem__(self, idx: int) -> Dict[str, object]:
        last_error: str | None = None
        for attempt in range(self.max_decode_retries):
            row_idx = idx if attempt == 0 else self._fallback_index(idx, attempt)
            row = self.rows[row_idx]
            try:
                waveform = load_audio_file(
                    row.audio_path,
                    sample_rate=self.sample_rate,
                    res_type=self.res_type,
                )

                if waveform.shape[0] < self.min_audio_samples:
                    waveform = pad_or_trim(waveform, self.min_audio_samples)

                start = self._select_start(waveform.shape[0], row_idx)
                fixed = pad_or_trim(waveform, self.clip_samples, start=start)

                return {
                    "waveform": fixed,
                    "labels": torch.tensor(row.labels, dtype=torch.float32),
                    "track_id": row.track_id,
                    "audio_path": row.audio_path,
                    "is_valid": True,
                }
            except Exception as exc:
                last_error = str(exc)

        row = self.rows[idx]
        return {
            "waveform": np.zeros(self.clip_samples, dtype=np.float32),
            "labels": torch.tensor(row.labels, dtype=torch.float32),
            "track_id": row.track_id,
            "audio_path": row.audio_path,
            "is_valid": False,
            "error": last_error or "audio load failure",
        }


class ASTBatchCollator:
    """Convert raw waveforms into AST feature tensors."""

    def __init__(self, feature_extractor, sample_rate: int):
        self.feature_extractor = feature_extractor
        self.sample_rate = int(sample_rate)

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, object] | None:
        valid_items = [item for item in batch if bool(item.get("is_valid", True))]
        if not valid_items:
            return None

        waveforms = [item["waveform"] for item in valid_items]
        labels = torch.stack([item["labels"] for item in valid_items], dim=0)
        track_ids = [int(item["track_id"]) for item in valid_items]
        audio_paths = [str(item["audio_path"]) for item in valid_items]

        model_inputs = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        return {
            "model_inputs": model_inputs,
            "labels": labels,
            "track_ids": track_ids,
            "audio_paths": audio_paths,
            "invalid_count": len(batch) - len(valid_items),
        }


def discover_audio_files(audio_dir: str | Path) -> List[Path]:
    """Recursively discover supported audio files under a directory."""
    base = Path(audio_dir)
    if not base.exists():
        raise FileNotFoundError(f"Audio directory not found: {base}")

    return sorted([p for p in base.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS])
