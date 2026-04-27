# 🔧 Audio Data Corruption Fix Guide

## Problem Summary

Your AST training is failing because **~40-60% of the FMA MP3 files are corrupted or improperly decoded**. The `libmpg123` errors in your training logs are evidence:

```
[src/libmpg123/layer3.c:INT123_do_layer3():1804] error: dequantization failed!
[src/libmpg123/parse.c:wetwork():1349] error: Giving up resync after 1024 bytes
Note: Illegal Audio-MPEG-Header 0x00000000 at offset 22401
```

This causes:
- Model receives garbage input (corrupted waveforms or silence)
- Backpropagation receives noisy gradients from invalid data
- Training **plateaus at random-guess performance** (F1 ≈ 0.2, accuracy ≈ 0.2)

---

## Step 1: Diagnose Corruption

Run the diagnostic scanner to quantify the problem:

```bash
cd /home/devarsh/Work/Deep-Learning-Project
python diagnose_audio_corruption.py --audio-dir fma_small --max-files 100
```

**What to look for in output:**
- **Corruption %**: How many files fail to decode
- **Error types**: `RuntimeError`, `AudioReadError`, `SILENT`, `TOO_SHORT`
- **If >15% corrupted**: Data cleanup is urgent

**Example output:**
```
✓ Valid files:              8000  (94.2%)
✗ Corrupted files:          500   (5.8%)
⊘ Silent files:             0     (0.0%)
⚠ Very short files:         3     (0.04%)
```

---

## Step 2: Fix Corruption by Re-encoding to WAV

MP3 is lossy and complex. WAV (PCM) is simple, reliable, and fast to decode.

```bash
# Re-encode all MP3 files to WAV
python fix_audio_corruption.py --audio-dir fma_small --sample-rate 16000 --workers 8
```

**What this does:**
- Converts `fma_small/000/000000.mp3` → `fma_small/000/000000.wav`
- Uses FFmpeg (robust) with fallback to librosa
- Keeps original directory structure
- **Duration**: ~2-4 hours for 8000 files (with 8 workers)

**Why WAV?**
- PCM format = direct audio samples, no compression artifacts
- No decoding errors from broken frames
- 16-bit/16kHz WAV is ~1.7 MB/min (acceptable size)

---

## Step 3: Verify Repair

Re-run the diagnostic on WAV files:

```bash
# Scan the newly created WAV files
find fma_small -name "*.wav" -type f | head -10  # Verify WAV files exist
python diagnose_audio_corruption.py --audio-dir fma_small --max-files 100
```

Expected result: **100% valid** (or >99.5%)

---

## Step 4: Update Data Loading

Your AST pipeline must use **WAV files** instead of MP3.

### Option A: Update AST's data.py

In `training/ast_pipeline/data.py`, the `load_audio_file()` function already supports WAV.
Just ensure your manifest CSV has WAV paths:

```python
# In build_dataset.py, when creating manifest:
# BEFORE:
# audio_path = f"fma_small/{track_id:06d}.mp3"

# AFTER:
# audio_path = f"fma_small/{track_id:06d}.wav"
```

### Option B: Use Robust Audio Loader

For even better error handling, use the new `robust_audio_loader.py`:

```python
from utils.robust_audio_loader import ImprovedAudioDataset, AudioLoader

# In your dataset creation
loader = AudioLoader(sample_rate=16000)
dataset = ImprovedAudioDataset(
    audio_paths=wav_file_paths,
    labels=genre_labels,
    sample_rate=16000,
    on_error='skip'  # Skip corrupted files instead of returning zeros
)
```

---

## Step 5: Re-train

Now retrain your AST model with cleaned data:

```bash
# Clear old logs
rm -rf logs/ast_pipeline/checkpoints/*

# Re-run training
python -m training.ast_pipeline.run_pipeline \
    --stage all \
    --config training/ast_pipeline/config_ast_pipeline.json
```

**Expected improvement:**
- **Epoch 1**: Val F1 should jump to **0.3-0.5** (vs 0.0158)
- **Epoch 5**: Val F1 reaches **0.6-0.75** (vs 0.22)
- Training loss converges steadily (not noisy)

---

## Troubleshooting

### Problem: "ffmpeg not found"

```bash
# Install FFmpeg
sudo apt-get update && sudo apt-get install -y ffmpeg
```

### Problem: "Still getting decode errors after re-encoding"

1. Check if you're still loading MP3 files:
   ```python
   # BAD:
   librosa.load("fma_small/000/000000.mp3")  # ✗ Still broken
   
   # GOOD:
   librosa.load("fma_small/000/000000.wav")  # ✓ Fixed
   ```

2. Verify WAV files were created:
   ```bash
   ls -lh fma_small/000/*.wav | head -5
   ```

### Problem: "Training is very slow now"

WAV files are **larger** than MP3 (but decode faster and correctly):
- MP3: ~200-400 KB per 3-min track
- WAV: ~1.7 MB per 3-min track

If disk/bandwidth is an issue:
1. Use `torchaudio` with `sox_io` backend (more optimized)
2. Store WAV files on SSD (not network drive)
3. Increase `num_workers` in DataLoader for parallel I/O

---

## Verification Checklist

- [ ] Diagnostic shows corruption % (baseline)
- [ ] Re-encoded MP3 → WAV (or errors are minimal)
- [ ] Verified WAV files exist in fma_small/
- [ ] Updated manifest/dataloader to use WAV paths
- [ ] Re-run training (clear old checkpoints first)
- [ ] Epoch 1 metrics improve dramatically (F1 >0.3)
- [ ] No more `libmpg123` errors in logs

---

## Timeline

| Task | Time |
|------|------|
| Diagnostic scan (100 files) | 2 min |
| Re-encoding all data (8 workers) | 2-4 hrs |
| Verification | 5 min |
| Update data loading | 15 min |
| **Re-train model (10 epochs)** | **6-8 hrs** |
| **Total** | **~12 hrs** |

**Start the re-encoding now** (Step 2) and it will run in the background while you update the data loader (Step 4).

---

## Additional Resources

- [Librosa Loading](https://librosa.org/doc/latest/generated/librosa.load.html)
- [FFmpeg Audio Encoding](https://wiki.hydrogenaud.io/index.php?title=FFmpeg_Audio_Encoding)
- [FMA Dataset Issues](https://github.com/mdeff/fma#issues)
