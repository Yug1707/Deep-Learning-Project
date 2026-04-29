"""FastAPI server for AST genre prediction and feedback learning."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import ensure_runtime_dirs, load_app_config
from app.schemas import (
    AppStatus,
    FeedbackRequest,
    FeedbackResponse,
    GenreInfo,
    PredictionResponse,
)
from app.services.ast_predictor import ASTGenrePredictor
from app.services.continual_trainer import ContinualTrainer
from app.services.feedback_store import FeedbackStore
from app.services.genres import load_genres


config = load_app_config()
ensure_runtime_dirs(config)
genres = load_genres(config.ast_config)
store = FeedbackStore(config, genres)
predictor = ASTGenrePredictor(config, genres)
trainer = ContinualTrainer(config, genres, store)

app = FastAPI(title="AST Genre Feedback Server")

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def _model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/status", response_model=AppStatus)
def status() -> AppStatus:
    active_checkpoint = predictor.active_checkpoint_path
    return AppStatus(
        ast_config_path=str(config.ast_config_path),
        active_checkpoint_path=str(active_checkpoint),
        checkpoint_exists=active_checkpoint.exists(),
        model_loaded=predictor.is_loaded,
        num_genres=len(genres),
        feedback_buffer_size=store.buffer_size(),
        feedback_trigger_size=config.feedback_trigger_size,
        trainer=trainer.status(),
    )


@app.get("/api/genres", response_model=list[GenreInfo])
def list_genres() -> list[GenreInfo]:
    return genres


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}:
        return suffix
    return ".wav"


@app.post("/api/predict", response_model=PredictionResponse)
def predict(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(default=None),
    top_k: int = Form(default=5),
) -> PredictionResponse:
    prediction_id = str(uuid.uuid4())
    audio_path = config.uploads_dir / f"{prediction_id}{_safe_suffix(file.filename)}"

    try:
        with open(audio_path, "wb") as handle:
            shutil.copyfileobj(file.file, handle)
    finally:
        file.file.close()

    try:
        result = predictor.predict(audio_path, threshold=threshold, top_k=top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    response = PredictionResponse(
        prediction_id=prediction_id,
        audio_path=str(audio_path),
        threshold=float(result["threshold"]),
        num_chunks=int(result["num_chunks"]),
        predicted_genres=result["predicted_genres"],
        top_k=result["top_k"],
    )
    store.record_prediction(
        prediction_id=prediction_id,
        payload=_model_to_dict(response),
    )
    return response


@app.post("/api/feedback", response_model=FeedbackResponse)
def feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
) -> FeedbackResponse:
    try:
        store.append_feedback(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    training_started = trainer.maybe_start(background_tasks, predictor)
    buffer_size = store.buffer_size()
    if training_started:
        message = "Feedback saved. Continual AST training started."
    else:
        remaining = max(0, config.feedback_trigger_size - buffer_size)
        message = f"Feedback saved. {remaining} more feedback item(s) until training."

    return FeedbackResponse(
        accepted=True,
        buffer_size=buffer_size,
        trigger_size=config.feedback_trigger_size,
        training_started=training_started,
        message=message,
    )


@app.post("/api/reload", response_model=AppStatus)
def reload_model() -> AppStatus:
    try:
        predictor.reload()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}") from exc
    return status()
