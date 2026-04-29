"""Background continual learning for AST feedback samples."""

from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from app.config import AppConfig
from app.schemas import GenreInfo, TrainerStatus
from app.services.ast_predictor import ASTGenrePredictor, _resolve_model_source
from app.services.feedback_store import FeedbackStore
from training.ast_pipeline.data import ASTBatchCollator, ASTManifestDataset
from training.ast_pipeline.model import (
    apply_classifier_dropout,
    create_ast_model,
    freeze_backbone,
    load_feature_extractor,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


class ContinualTrainer:
    """Runs feedback-triggered AST fine-tuning in the background."""

    def __init__(self, config: AppConfig, genres: List[GenreInfo], store: FeedbackStore) -> None:
        self.config = config
        self.genres = genres
        self.store = store
        self._lock = threading.RLock()
        self._state = "idle"
        self._current_run_id: str | None = None
        self._last_run_id: str | None = None
        self._last_error: str | None = None
        self._last_checkpoint_path: str | None = None

    def status(self) -> TrainerStatus:
        with self._lock:
            return TrainerStatus(
                state=self._state,
                current_run_id=self._current_run_id,
                last_run_id=self._last_run_id,
                last_error=self._last_error,
                last_checkpoint_path=self._last_checkpoint_path,
            )

    def should_trigger(self) -> bool:
        return self.store.buffer_size() >= self.config.feedback_trigger_size

    def mark_started(self, run_id: str) -> None:
        with self._lock:
            self._state = "training"
            self._current_run_id = run_id
            self._last_error = None

    def mark_finished(self, run_id: str, checkpoint_path: Path) -> None:
        with self._lock:
            self._state = "idle"
            self._current_run_id = None
            self._last_run_id = run_id
            self._last_checkpoint_path = str(checkpoint_path)

    def mark_failed(self, run_id: str, error: str) -> None:
        with self._lock:
            self._state = "failed"
            self._current_run_id = None
            self._last_run_id = run_id
            self._last_error = error

    def is_training(self) -> bool:
        with self._lock:
            return self._state == "training"

    def maybe_start(self, background_tasks, predictor: ASTGenrePredictor) -> bool:
        if self.is_training() or not self.should_trigger():
            return False

        run_id = f"feedback_{_utc_stamp()}"
        self.mark_started(run_id)
        background_tasks.add_task(self.run, run_id, predictor)
        return True

    def _write_manifest(self, entries: List[Dict], run_dir: Path) -> Path:
        manifest_path = run_dir / "feedback_manifest.csv"
        rows: List[Dict] = []
        for row_idx, entry in enumerate(entries):
            labels = {f"label_{i}": 0 for i in range(len(self.genres))}
            for class_index in entry["target_indices"]:
                labels[f"label_{int(class_index)}"] = 1
            rows.append(
                {
                    "track_id": row_idx,
                    "audio_path": entry["audio_path"],
                    **labels,
                }
            )

        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        return manifest_path

    def run(self, run_id: str, predictor: ASTGenrePredictor) -> None:
        """Fine-tune AST on buffered feedback and reload the prediction server."""
        run_dir = self.config.continual_runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        processed_entries = self.store.read_buffer()
        if not processed_entries:
            self.mark_failed(run_id, "Feedback buffer was empty when training started")
            return

        try:
            manifest_path = self._write_manifest(processed_entries, run_dir)
            checkpoint_path = predictor.active_checkpoint_path
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing active checkpoint: {checkpoint_path}")

            ast_cfg = self.config.ast_config["ast"]
            audio_cfg = self.config.ast_config["audio"]
            models_dir = Path(self.config.ast_config["paths"]["models_dir"])
            model_source, local_only = _resolve_model_source(models_dir, ast_cfg["model_name"])

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)

            if self.config.continual_freeze_backbone:
                freeze_backbone(model)

            dataset = ASTManifestDataset(
                manifest_path=manifest_path,
                sample_rate=int(audio_cfg["sample_rate"]),
                clip_seconds=float(audio_cfg["clip_seconds"]),
                res_type=str(audio_cfg.get("res_type", "soxr_hq")),
                mode="train",
                seed=int(self.config.ast_config["seed"]),
                min_audio_seconds=float(self.config.ast_config["dataset"].get("min_audio_seconds", 0.0)),
            )
            collator = ASTBatchCollator(
                feature_extractor=feature_extractor,
                sample_rate=int(audio_cfg["sample_rate"]),
            )
            loader = DataLoader(
                dataset,
                batch_size=self.config.continual_batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=device.type == "cuda",
                collate_fn=collator,
                drop_last=False,
            )

            optimizer = AdamW(
                [param for param in model.parameters() if param.requires_grad],
                lr=self.config.continual_learning_rate,
                weight_decay=self.config.continual_weight_decay,
            )
            criterion = nn.BCEWithLogitsLoss()
            history: List[Dict] = []

            model.train()
            for epoch in range(1, self.config.continual_epochs + 1):
                dataset.set_epoch(epoch)
                total_loss = 0.0
                batches = 0

                for batch in loader:
                    if batch is None:
                        continue

                    labels = batch["labels"].to(device)
                    model_inputs = {
                        key: value.to(device)
                        for key, value in batch["model_inputs"].items()
                        if isinstance(value, torch.Tensor)
                    }

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(**model_inputs).logits
                    loss = criterion(logits, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.continual_max_grad_norm)
                    optimizer.step()

                    total_loss += float(loss.item())
                    batches += 1

                history.append(
                    {
                        "epoch": epoch,
                        "loss": total_loss / max(1, batches),
                        "batches": batches,
                    }
                )

            new_checkpoint_path = run_dir / "ast_feedback_latest.pt"
            payload = {
                "epoch": int(checkpoint.get("epoch", 0)) + self.config.continual_epochs,
                "best_f1": float(checkpoint.get("best_f1", checkpoint.get("val_f1", 0.0))),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": self.config.ast_config,
                "history": {
                    "continual_feedback": history,
                },
                "continual_metadata": {
                    "run_id": run_id,
                    "source_checkpoint": str(checkpoint_path),
                    "feedback_count": len(processed_entries),
                    "feedback_ids": [entry["feedback_id"] for entry in processed_entries],
                    "freeze_backbone": self.config.continual_freeze_backbone,
                },
            }
            torch.save(payload, new_checkpoint_path)

            summary = {
                "run_id": run_id,
                "checkpoint_path": str(new_checkpoint_path),
                "feedback_count": len(processed_entries),
                "history": history,
            }
            with open(run_dir / "training_summary.json", "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)

            predictor.reload(new_checkpoint_path, run_id=run_id)
            self.store.remove_from_buffer(entry["feedback_id"] for entry in processed_entries)
            self.mark_finished(run_id, new_checkpoint_path)
        except Exception:
            error = traceback.format_exc()
            with open(run_dir / "error.log", "w", encoding="utf-8") as handle:
                handle.write(error)
            self.mark_failed(run_id, error)

