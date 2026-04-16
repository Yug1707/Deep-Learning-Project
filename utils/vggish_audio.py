"""Audio utilities for the VGGish pipeline."""

from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np


DEFAULT_SAMPLE_RATE = 16000


def load_audio_16k_mono(
    path: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    res_type: str = "soxr_hq",
) -> np.ndarray:
    """Load audio as mono waveform at 16 kHz for VGGish."""
    audio, _ = librosa.load(
        str(path),
        sr=sample_rate,
        mono=True,
        res_type=res_type,
    )
    return audio.astype(np.float32)


def slice_audio(
    audio: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
) -> List[np.ndarray]:
    """Slice waveform into overlapping windows."""
    if audio.size == 0:
        return []

    window = int(window_seconds * sample_rate)
    hop = int(hop_seconds * sample_rate)

    if window <= 0 or hop <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")

    if audio.shape[0] < window:
        pad = np.zeros(window - audio.shape[0], dtype=np.float32)
        return [np.concatenate([audio, pad])]

    windows: List[np.ndarray] = []
    for start in range(0, audio.shape[0] - window + 1, hop):
        windows.append(audio[start : start + window])

    if not windows:
        windows.append(audio[:window])

    return windows


def get_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """Return waveform duration in seconds."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return float(audio.shape[0]) / float(sample_rate)
