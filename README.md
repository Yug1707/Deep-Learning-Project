# Deep-Learning-Project

## Standalone AST Pipeline

This repository now includes a standalone Audio Spectrogram Transformer (AST) pipeline for multi-label music genre classification.

- Package: `training/ast_pipeline`
- Default config: `training/ast_pipeline/config_ast_pipeline.json`
- Dedicated artifact root: `logs/ast_pipeline`

### Why this is non-intrusive

- The AST pipeline is isolated in its own package.
- It does not modify CRNN or VGGish/XGBoost training/inference code paths.
- Predictions, logs, checkpoints, manifests, and reports are written under `logs/ast_pipeline` only.

### Stage commands

```bash
# Build deterministic train/val/test manifests
python -m training.ast_pipeline.run_pipeline --stage build

# Train AST
python -m training.ast_pipeline.run_pipeline --stage train

# Evaluate val/test using best checkpoint
python -m training.ast_pipeline.run_pipeline --stage eval

# Run full build+train+eval sequence
python -m training.ast_pipeline.run_pipeline --stage all
```

### Batch inference

```bash
# Directory-based batch inference
python -m training.ast_pipeline.run_pipeline --stage predict --audio-dir fma_small/000

# Single-file inference
python -m training.ast_pipeline.run_pipeline --stage predict --audio fma_small/000/000002.mp3

# Manifest-driven inference
python -m training.ast_pipeline.run_pipeline --stage predict --manifest logs/ast_pipeline/manifests/test_manifest.csv
```

### Reproducibility

- Seeded behavior is controlled by `seed` in the AST config.
- Runtime config snapshots are stored in `logs/ast_pipeline/results/run_config_snapshot.json`.
- Train/val/test split generation is deterministic for the same config + seed.

### Dependencies

Install additive AST dependencies from:

```bash
pip install -r requirements_ast.txt
```