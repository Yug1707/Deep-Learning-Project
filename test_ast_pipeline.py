"""Smoke tests for standalone AST pipeline utilities."""

from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import wave

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from training.ast_pipeline.common import load_config
from training.ast_pipeline.data import ASTBatchCollator, ASTManifestDataset, chunk_waveform, label_columns


class _DummyFeatureExtractor:
    def __call__(self, waveforms, sampling_rate: int, return_tensors: str = "pt", padding: bool = True):
        max_len = max(len(w) for w in waveforms)
        out = np.zeros((len(waveforms), max_len), dtype=np.float32)
        for i, w in enumerate(waveforms):
            out[i, : len(w)] = w
        return {"input_values": torch.tensor(out)}


def _write_test_wav(path: Path, sample_rate: int = 16000, seconds: float = 0.4) -> None:
    num_samples = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)

        frames = bytearray()
        for i in range(num_samples):
            sample = int(12000 * np.sin((2.0 * np.pi * 440.0 * i) / sample_rate))
            frames.extend(struct.pack("<h", sample))
        handle.writeframes(bytes(frames))


def test_load_config_defaults() -> None:
    cfg = load_config("training/ast_pipeline/config_ast_pipeline.json")
    assert "paths" in cfg
    assert "output_root" in cfg["paths"]
    assert cfg["ast"]["model_name"] == "MIT/ast-finetuned-audioset-10-10-0.4593"


def test_chunk_waveform_shapes() -> None:
    sr = 16000
    wave = np.random.randn(sr * 23).astype(np.float32)
    chunks = chunk_waveform(
        waveform=wave,
        sample_rate=sr,
        chunk_seconds=10.0,
        hop_seconds=5.0,
        max_chunks=4,
    )

    assert len(chunks) == 3
    assert all(chunk.shape[0] == sr * 10 for chunk in chunks)


def test_label_columns() -> None:
    df = pd.DataFrame(
        {
            "track_id": [1],
            "audio_path": ["x.mp3"],
            "label_0": [1],
            "label_1": [0],
        }
    )
    cols = label_columns(df)
    assert cols == ["label_0", "label_1"]


def test_collator_schema() -> None:
    collator = ASTBatchCollator(feature_extractor=_DummyFeatureExtractor(), sample_rate=16000)
    batch = [
        {
            "waveform": np.random.randn(1000).astype(np.float32),
            "labels": torch.tensor([1.0, 0.0]),
            "track_id": 11,
            "audio_path": "a.wav",
        },
        {
            "waveform": np.random.randn(900).astype(np.float32),
            "labels": torch.tensor([0.0, 1.0]),
            "track_id": 12,
            "audio_path": "b.wav",
        },
    ]
    output = collator(batch)

    assert "model_inputs" in output
    assert "labels" in output
    assert output["labels"].shape == (2, 2)
    assert output["model_inputs"]["input_values"].shape[0] == 2


def test_ast_forward_shape_optional() -> None:
    try:
        from transformers import ASTConfig, ASTForAudioClassification
    except Exception:
        print("Skipping AST forward shape test: transformers not installed")
        return

    cfg = ASTConfig(
        num_labels=4,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=128,
        max_length=128,
        num_mel_bins=128,
        patch_size=16,
        frequency_stride=10,
        time_stride=10,
    )
    model = ASTForAudioClassification(cfg)
    inputs = torch.randn(2, cfg.max_length, cfg.num_mel_bins)
    logits = model(input_values=inputs).logits

    assert logits.shape == (2, 4)


def test_dataloader_multiworker_with_corrupt_audio() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        valid_audio = base / "valid.wav"
        corrupt_audio = base / "corrupt.mp3"
        manifest = base / "manifest.csv"

        _write_test_wav(valid_audio)
        corrupt_audio.write_text("not a valid audio stream", encoding="utf-8")

        pd.DataFrame(
            [
                {"track_id": 1, "audio_path": str(corrupt_audio), "label_0": 1, "label_1": 0},
                {"track_id": 2, "audio_path": str(valid_audio), "label_0": 0, "label_1": 1},
            ]
        ).to_csv(manifest, index=False)

        dataset = ASTManifestDataset(
            manifest_path=manifest,
            sample_rate=16000,
            clip_seconds=1.0,
            res_type="soxr_hq",
            mode="val",
            seed=7,
            min_audio_seconds=0.0,
        )
        collator = ASTBatchCollator(feature_extractor=_DummyFeatureExtractor(), sample_rate=16000)
        loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=collator, shuffle=False)

        batch = next(iter(loader))
        assert batch is not None
        assert batch["labels"].shape[1] == 2
        assert batch["labels"].shape[0] >= 1


def run_all_tests() -> None:
    test_load_config_defaults()
    print("[PASS] config load")

    test_chunk_waveform_shapes()
    print("[PASS] chunking")

    test_label_columns()
    print("[PASS] label columns")

    test_collator_schema()
    print("[PASS] collator schema")

    test_ast_forward_shape_optional()
    print("[PASS] optional AST forward")

    test_dataloader_multiworker_with_corrupt_audio()
    print("[PASS] dataloader robustness")

    print("All AST pipeline smoke tests passed")


if __name__ == "__main__":
    run_all_tests()
