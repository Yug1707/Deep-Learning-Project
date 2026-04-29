"""Feedback persistence and replay buffer management."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from app.config import AppConfig
from app.schemas import FeedbackRequest, GenreInfo
from app.services.genres import genre_lookup


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


class FeedbackStore:
    """Stores predictions, user feedback, and the active continual-learning buffer."""

    def __init__(self, config: AppConfig, genres: List[GenreInfo]) -> None:
        self.config = config
        self.genres = genres
        self._genres_by_id = genre_lookup(genres)
        self._lock = threading.RLock()

    def record_prediction(self, prediction_id: str, payload: Dict) -> None:
        record = {
            "prediction_id": prediction_id,
            "created_at": utc_now(),
            **payload,
        }
        with self._lock:
            _append_jsonl(self.config.predictions_path, record)

    def get_prediction(self, prediction_id: str) -> Dict | None:
        with self._lock:
            for row in _read_jsonl(self.config.predictions_path):
                if row.get("prediction_id") == prediction_id:
                    return row
        return None

    def append_feedback(self, request: FeedbackRequest) -> Dict:
        prediction = self.get_prediction(request.prediction_id)
        if prediction is None:
            raise KeyError(f"Unknown prediction_id: {request.prediction_id}")

        if request.is_correct:
            target_genre_ids = [
                item["genre_id"]
                for item in prediction.get("predicted_genres", [])
            ]
        else:
            target_genre_ids = request.corrected_genre_ids or []
            if not target_genre_ids:
                raise ValueError("corrected_genre_ids is required when prediction is wrong")

        unknown = [genre_id for genre_id in target_genre_ids if genre_id not in self._genres_by_id]
        if unknown:
            raise ValueError(f"Unknown genre_id values: {unknown}")

        target_indices = [
            self._genres_by_id[genre_id].class_index
            for genre_id in target_genre_ids
        ]
        feedback_id = str(uuid.uuid4())
        record = {
            "feedback_id": feedback_id,
            "prediction_id": request.prediction_id,
            "created_at": utc_now(),
            "audio_path": prediction["audio_path"],
            "is_correct": bool(request.is_correct),
            "target_genre_ids": target_genre_ids,
            "target_indices": sorted(set(target_indices)),
            "notes": request.notes or "",
            "original_prediction": {
                "predicted_genres": prediction.get("predicted_genres", []),
                "top_k": prediction.get("top_k", []),
                "threshold": prediction.get("threshold"),
            },
        }

        with self._lock:
            _append_jsonl(self.config.feedback_log_path, record)
            _append_jsonl(self.config.feedback_buffer_path, record)

        return record

    def buffer_size(self) -> int:
        with self._lock:
            return len(_read_jsonl(self.config.feedback_buffer_path))

    def read_buffer(self) -> List[Dict]:
        with self._lock:
            return _read_jsonl(self.config.feedback_buffer_path)

    def remove_from_buffer(self, feedback_ids: Iterable[str]) -> None:
        remove_ids = set(feedback_ids)
        with self._lock:
            rows = [
                row
                for row in _read_jsonl(self.config.feedback_buffer_path)
                if row.get("feedback_id") not in remove_ids
            ]
            self.config.feedback_buffer_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.feedback_buffer_path, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

