"""API schemas for the AST web application."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GenreInfo(BaseModel):
    class_index: int
    genre_id: str
    name: str


class GenreScore(GenreInfo):
    probability: float


class PredictionResponse(BaseModel):
    prediction_id: str
    audio_path: str
    threshold: float
    num_chunks: int
    predicted_genres: List[GenreScore]
    top_k: List[GenreScore]


class FeedbackRequest(BaseModel):
    prediction_id: str
    is_correct: bool
    corrected_genre_ids: Optional[List[str]] = Field(default=None)
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    accepted: bool
    buffer_size: int
    trigger_size: int
    training_started: bool
    message: str


class TrainerStatus(BaseModel):
    state: str
    current_run_id: Optional[str] = None
    last_run_id: Optional[str] = None
    last_error: Optional[str] = None
    last_checkpoint_path: Optional[str] = None


class AppStatus(BaseModel):
    ast_config_path: str
    active_checkpoint_path: str
    checkpoint_exists: bool
    model_loaded: bool
    num_genres: int
    feedback_buffer_size: int
    feedback_trigger_size: int
    trainer: TrainerStatus

