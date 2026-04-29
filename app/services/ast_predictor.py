"""AST model loading and inference for uploaded audio."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from app.config import AppConfig
from app.schemas import GenreInfo, GenreScore
from app.services.genres import index_lookup
from training.ast_pipeline.data import chunk_waveform, load_audio_file
from training.ast_pipeline.model import (
    apply_classifier_dropout,
    create_ast_model,
    load_feature_extractor,
)


def _resolve_model_source(models_dir: Path, model_name: str) -> tuple[str, bool]:
    required = ["config.json", "preprocessor_config.json"]
    if models_dir.exists() and all((models_dir / name).exists() for name in required):
        return str(models_dir), True
    return model_name, False


class ASTGenrePredictor:
    """Lazy-loaded AST predictor with support for checkpoint reloads."""

    def __init__(self, config: AppConfig, genres: List[GenreInfo]) -> None:
        self.config = config
        self.genres = genres
        self._genres_by_index = index_lookup(genres)
        self._lock = threading.RLock()
        self._model = None
        self._feature_extractor = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.active_checkpoint_path = self._load_active_checkpoint_path()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._feature_extractor is not None

    def _load_active_checkpoint_path(self) -> Path:
        state_path = self.config.model_state_path
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                active = payload.get("active_checkpoint_path")
                if active:
                    return Path(active)
            except Exception:
                pass
        return self.config.base_checkpoint_path

    def write_active_checkpoint_path(self, checkpoint_path: Path, run_id: str | None = None) -> None:
        payload = {
            "active_checkpoint_path": str(checkpoint_path),
            "run_id": run_id,
        }
        self.config.model_state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.model_state_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load(self) -> None:
        """Load model and feature extractor if they are not already loaded."""
        with self._lock:
            if self.is_loaded:
                return

            checkpoint_path = self.active_checkpoint_path
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    "Missing AST checkpoint. Expected "
                    f"{checkpoint_path}. Train AST first with "
                    "`python -m training.ast_pipeline.run_pipeline --stage all` "
                    "or set AST_APP_CHECKPOINT to a trained checkpoint."
                )

            ast_cfg = self.config.ast_config["ast"]
            models_dir = Path(self.config.ast_config["paths"]["models_dir"])
            model_source, local_only = _resolve_model_source(models_dir, ast_cfg["model_name"])

            feature_extractor = load_feature_extractor(
                model_source,
                cache_dir=ast_cfg.get("cache_dir"),
                local_files_only=local_only,
            )
            model = create_ast_model(
                model_name=model_source,
                num_labels=len(self.genres),
                cache_dir=ast_cfg.get("cache_dir"),
                local_files_only=local_only,
                hidden_dropout_prob=ast_cfg.get("hidden_dropout_prob"),
                attention_probs_dropout_prob=ast_cfg.get("attention_probs_dropout_prob"),
            )
            apply_classifier_dropout(
                model,
                dropout_prob=float(ast_cfg.get("classifier_dropout_prob", 0.3)),
            )

            checkpoint = torch.load(checkpoint_path, map_location=self._device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(self._device)
            model.eval()

            self._feature_extractor = feature_extractor
            self._model = model

    def reload(self, checkpoint_path: Path | None = None, run_id: str | None = None) -> None:
        """Reload the model from a new or existing checkpoint."""
        with self._lock:
            if checkpoint_path is not None:
                self.active_checkpoint_path = checkpoint_path
                self.write_active_checkpoint_path(checkpoint_path, run_id=run_id)
            self._model = None
            self._feature_extractor = None
            self.load()

    def _batched_probabilities(
        self,
        chunks: List[np.ndarray],
        sample_rate: int,
        batch_size: int,
    ) -> np.ndarray:
        if self._model is None or self._feature_extractor is None:
            raise RuntimeError("AST predictor is not loaded")

        outputs: List[np.ndarray] = []
        self._model.eval()
        with torch.no_grad():
            for start in range(0, len(chunks), batch_size):
                chunk_batch = chunks[start : start + batch_size]
                features = self._feature_extractor(
                    chunk_batch,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding=True,
                )
                model_inputs = {
                    key: value.to(self._device)
                    for key, value in features.items()
                    if isinstance(value, torch.Tensor)
                }
                logits = self._model(**model_inputs).logits
                outputs.append(torch.sigmoid(logits).cpu().numpy())

        return np.concatenate(outputs, axis=0)

    def predict(self, audio_path: Path, threshold: float | None = None, top_k: int = 5) -> Dict:
        """Predict genres for one audio file."""
        self.load()

        audio_cfg = self.config.ast_config["audio"]
        ast_cfg = self.config.ast_config["ast"]
        inference_cfg = self.config.ast_config["inference"]
        sample_rate = int(audio_cfg["sample_rate"])
        decision_threshold = (
            float(threshold)
            if threshold is not None
            else float(self.config.ast_config["evaluation"]["threshold"])
        )
        batch_size = int(inference_cfg.get("batch_size", 8))
        top_k = max(1, min(int(top_k), len(self.genres)))

        waveform = load_audio_file(
            audio_path,
            sample_rate=sample_rate,
            res_type=str(audio_cfg.get("res_type", "soxr_hq")),
        )
        chunks = chunk_waveform(
            waveform=waveform,
            sample_rate=sample_rate,
            chunk_seconds=float(ast_cfg.get("chunk_seconds", audio_cfg.get("clip_seconds", 10.0))),
            hop_seconds=float(ast_cfg.get("chunk_hop_seconds", 5.0)),
            max_chunks=int(ast_cfg.get("max_chunks_per_track", 12)),
        )

        probs_per_chunk = self._batched_probabilities(
            chunks=chunks,
            sample_rate=sample_rate,
            batch_size=batch_size,
        )
        mean_probs = probs_per_chunk.mean(axis=0)
        predicted_indices = np.where(mean_probs >= decision_threshold)[0].tolist()
        top_indices = np.argsort(mean_probs)[::-1][:top_k]

        def score_for(index: int) -> GenreScore:
            genre = self._genres_by_index[int(index)]
            return GenreScore(
                class_index=genre.class_index,
                genre_id=genre.genre_id,
                name=genre.name,
                probability=float(mean_probs[int(index)]),
            )

        return {
            "threshold": decision_threshold,
            "num_chunks": int(len(chunks)),
            "predicted_genres": [score_for(index) for index in predicted_indices],
            "top_k": [score_for(index) for index in top_indices],
            "raw_probabilities": [float(value) for value in mean_probs.tolist()],
        }

