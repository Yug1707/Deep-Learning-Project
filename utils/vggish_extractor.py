"""PyTorch VGGish wrapper for embedding extraction."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from utils.vggish_audio import load_audio_16k_mono


class VGGishExtractor:
    """Extract frame-level VGGish embeddings from audio."""

    def __init__(self, device: Optional[str] = None):
        if device is None or device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved = device

        self.device = torch.device(resolved)

        try:
            from torchvggish import vggish as torchvggish_model
            from torchvggish import vggish_input
        except ImportError as exc:
            raise ImportError(
                "torchvggish is not installed. Install dependencies from requirements_vggish.txt"
            ) from exc

        self._vggish_input = vggish_input
        self.model = torchvggish_model()
        self.model = self.model.to(self.device)
        self.model.eval()

    def extract_from_waveform(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Return frame embeddings with shape (num_frames, 128)."""
        if audio is None or audio.size == 0:
            return np.zeros((0, 128), dtype=np.float32)

        examples = self._vggish_input.waveform_to_examples(audio, sample_rate)
        if examples is None or len(examples) == 0:
            return np.zeros((0, 128), dtype=np.float32)

        inputs = torch.tensor(examples, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            embeddings = self.model(inputs)

        return embeddings.detach().cpu().numpy().astype(np.float32)

    def extract_from_file(self, audio_path: Path, sample_rate: int = 16000) -> np.ndarray:
        """Load audio file and return frame embeddings."""
        waveform = load_audio_16k_mono(Path(audio_path), sample_rate=sample_rate)
        return self.extract_from_waveform(waveform, sample_rate)


def pool_embeddings(frame_embeddings: np.ndarray, mode: str = "mean") -> np.ndarray:
    """Pool frame embeddings to a single vector with shape (128,)."""
    if frame_embeddings.size == 0:
        return np.zeros((128,), dtype=np.float32)

    if mode == "mean":
        return np.mean(frame_embeddings, axis=0).astype(np.float32)
    if mode == "max":
        return np.max(frame_embeddings, axis=0).astype(np.float32)

    raise ValueError(f"Unsupported pooling mode: {mode}")
