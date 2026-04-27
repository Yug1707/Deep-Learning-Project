# 🚀 IMMEDIATE ACTION PLAN: Fix Audio Corruption

**Timeline: ~12 hours total | Start now for results by tomorrow**

---

## Phase 1: Diagnosis (5 minutes)

Run the diagnostic to see how bad the corruption is:

```bash
cd /home/devarsh/Work/Deep-Learning-Project

# Test on first 100 files
python diagnose_audio_corruption.py --audio-dir fma_small --max-files 100
```

**Expected output:**
```
✓ Valid files:              ~94-96  (~95%)
✗ Corrupted files:          ~4-6    (~5%)
```

**If corruption > 10%:** Problem is severe, proceed immediately.  
**If corruption < 2%:** Problem may be elsewhere, but still re-encode for reliability.

---

## Phase 2: Re-encode Audio (2-4 hours | BACKGROUND)

Start this NOW - it runs in the background while you do other work:

```bash
# This will take 2-4 hours with 8 workers
# It's safe to run in the background (e.g., in a tmux/screen session)
python fix_audio_corruption.py \
    --audio-dir fma_small \
    --sample-rate 16000 \
    --workers 8

# Or run in background with nohup:
nohup python fix_audio_corruption.py \
    --audio-dir fma_small \
    --sample-rate 16000 \
    --workers 8 > re_encoding.log 2>&1 &
```

**What this does:**
- Converts `fma_small/000/000000.mp3` → `fma_small/000/000000.wav`
- Keeps all MP3 files (can delete later to save space)
- Creates robust, error-proof WAV files

**Expected output:**
```
✓ Success (ffmpeg):         ~7950
✓ Success (librosa):        ~50
⚠ Already exist:            0
✗ Failed:                   0
```

---

## Phase 3: Update Code (15 minutes | DO NOW)

While re-encoding runs, update the pipeline to use WAV files:

### 3a. Build dataset code is already updated ✓

The file `training/ast_pipeline/build_dataset.py` has been updated to prefer WAV files.
The `get_audio_path()` function now checks for `.wav` first.

### 3b. Convert existing manifests (if you have old ones)

If you've already run `build_dataset.py` with MP3 paths, convert the manifests:

```bash
python convert_manifest_to_wav.py \
    --manifests-dir logs/ast_pipeline/manifests \
    --backup
```

This updates:
- `logs/ast_pipeline/manifests/train_manifest.csv`
- `logs/ast_pipeline/manifests/val_manifest.csv`
- `logs/ast_pipeline/manifests/test_manifest.csv`

---

## Phase 4: Verification (5 minutes | DO AFTER RE-ENCODING)

Once re-encoding is complete:

```bash
# Count how many WAV files were created
find fma_small -name "*.wav" -type f | wc -l
# Should show ~8000

# Verify they're all loadable
python diagnose_audio_corruption.py --audio-dir fma_small --max-files 100
# Should show 100/100 valid
```

---

## Phase 5: Clean Dataset & Rebuild Manifests (10 minutes)

Clear old pipeline outputs and rebuild manifests with WAV paths:

```bash
# Remove old checkpoints/logs
rm -rf logs/ast_pipeline/checkpoints/*
rm -rf logs/ast_pipeline/results/*

# Rebuild manifests with WAV files
python -m training.ast_pipeline.build_dataset \
    --config training/ast_pipeline/config_ast_pipeline.json
```

**Output should show:**
```
AST manifests created successfully
{'total_records': 8000, 'class_positive_counts': [...], ...}
```

---

## Phase 6: Re-train (6-8 hours)

Now train with clean data:

```bash
python -m training.ast_pipeline.run_pipeline \
    --stage all \
    --config training/ast_pipeline/config_ast_pipeline.json
```

**Expected metrics (NEW vs OLD):**

| Metric | Epoch 1 (OLD) | Epoch 1 (NEW) | Epoch 5 (NEW) |
|--------|---------------|---------------|---------------|
| Val F1 macro | 0.0158 | **0.35-0.50** | **0.65-0.75** |
| Subset accuracy | 0.0075 | **0.20-0.35** | **0.50-0.65** |
| Val loss | 0.415 | **0.35-0.40** | **0.30-0.35** |

**Key differences:**
- No `libmpg123` errors in logs
- Training loss converges smoothly
- Validation metrics improve every epoch
- Model actually learns patterns!

---

## Troubleshooting

### "Re-encoding is taking forever"

Normal. Don't kill it. Process:
- 8000 files × 3 min audio = 24,000 min total audio
- FFmpeg processes at ~30x speed on GPU
- ~13 hours / 8 workers = ~2 hours per worker = **total ~2-4 hours**

Monitor progress:
```bash
find fma_small -name "*.wav" -type f | wc -l  # Current count
```

### "WAV files created but still getting errors"

Check:
1. Are you still loading from MP3?
   ```python
   # ✗ BAD
   librosa.load("fma_small/000/000000.mp3")
   
   # ✓ GOOD
   librosa.load("fma_small/000/000000.wav")
   ```

2. Did you rebuild manifests?
   ```bash
   grep ".mp3" logs/ast_pipeline/manifests/train_manifest.csv
   # Should return nothing - all should be .wav
   ```

3. Is training using the new manifest?
   ```bash
   grep manifest training/ast_pipeline/config_ast_pipeline.json
   # Verify it points to the rebuilt manifests
   ```

### "Out of disk space"

WAV files take ~1.7 MB per 3-minute song:
- 8000 files × 1.7 MB = ~13.6 GB for WAV
- Original MP3s: ~1.6 GB

**Solution:**
```bash
# Delete original MP3 files (ONLY after verifying WAV files work)
find fma_small -name "*.mp3" -delete

# OR compress to a backup
tar -czf fma_small_mp3_backup.tar.gz fma_small/*.mp3
```

---

## Success Criteria

- [ ] Diagnostic shows >95% valid files
- [ ] ~8000 WAV files created in fma_small/
- [ ] Manifests updated to use .wav paths
- [ ] Build dataset completes without errors
- [ ] Training starts without `libmpg123` errors
- [ ] **Epoch 1 Val F1 > 0.3** (was 0.0158)
- [ ] Training loss decreases each epoch

---

## Commands Quick Reference

```bash
# Phase 1: Diagnose
python diagnose_audio_corruption.py --audio-dir fma_small --max-files 100

# Phase 2: Re-encode (background)
nohup python fix_audio_corruption.py --audio-dir fma_small --workers 8 > re_encoding.log 2>&1 &

# Phase 3: Update code (already done ✓)

# Phase 4: Verify
find fma_small -name "*.wav" -type f | wc -l
python diagnose_audio_corruption.py --audio-dir fma_small --max-files 100

# Phase 5: Rebuild
rm -rf logs/ast_pipeline/checkpoints/* logs/ast_pipeline/results/*
python -m training.ast_pipeline.build_dataset --config training/ast_pipeline/config_ast_pipeline.json

# Phase 6: Train
python -m training.ast_pipeline.run_pipeline --stage all --config training/ast_pipeline/config_ast_pipeline.json
```

---

## 🎯 Start Now!

**Run Phase 1 (diagnosis) immediately to confirm the problem.** Then start Phase 2 (re-encoding) in the background while you proceed with the rest.

**Expected end result:** Working AST model with proper learning curves instead of random guesses.
