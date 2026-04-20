# AST Pipeline

Standalone Audio Spectrogram Transformer (AST) pipeline for multi-label genre classification.

## Stages

- Build manifests
- Train AST classifier
- Evaluate on val/test
- Run batch inference

## Run

```bash
python -m training.ast_pipeline.run_pipeline --stage build
python -m training.ast_pipeline.run_pipeline --stage train
python -m training.ast_pipeline.run_pipeline --stage eval
python -m training.ast_pipeline.run_pipeline --stage predict --audio-dir fma_small/000
```

Use a custom config with `--config training/ast_pipeline/config_ast_pipeline.json`.

All artifacts are written under `logs/ast_pipeline` by default.
