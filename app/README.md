# AST Web Feedback App

FastAPI frontend/backend for AST-only genre prediction, user feedback, and feedback-triggered fine-tuning.

## Expected AST Artifacts

By default the app loads:

- Config: `training/ast_pipeline/config_ast_pipeline.json`
- Checkpoint: `logs/ast_pipeline/checkpoints/best.pt`
- Class mapping: `logs/ast_pipeline/manifests/class_mapping.json`

If the checkpoint is elsewhere, set:

```bash
$env:AST_APP_CHECKPOINT = "path/to/best.pt"
```

## Run

```bash
pip install -r requirements_ast.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Continual Learning

Runtime files are written under `logs/app/`.

- Uploaded audio: `logs/app/uploads/`
- Prediction records: `logs/app/predictions.jsonl`
- Full feedback log: `logs/app/feedback_log.jsonl`
- Training buffer: `logs/app/feedback_buffer.jsonl`
- Continual runs: `logs/app/continual_runs/`

When the buffer reaches `AST_FEEDBACK_TRIGGER_SIZE` feedback items, the server starts a background AST fine-tuning run. The new checkpoint is saved under that run directory, then the predictor reloads from it. The original AST checkpoint is not overwritten.

Useful overrides:

```bash
$env:AST_FEEDBACK_TRIGGER_SIZE = "8"
$env:AST_CONTINUAL_EPOCHS = "2"
$env:AST_CONTINUAL_BATCH_SIZE = "2"
$env:AST_CONTINUAL_LR = "1e-5"
$env:AST_CONTINUAL_FREEZE_BACKBONE = "true"
```

