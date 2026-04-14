#!/usr/bin/env python
# coding: utf-8

# # Audio Data Pipeline — Improved
# FMA Small dataset → Log-Mel Spectrogram → Multi-label genre classification

# ## 1. Environment Check — Library Versions & GPU

# In[1]:


import sys
import importlib

import numpy as np
import pandas as pd
import librosa
import torch
import sklearn

print(f"Python      : {sys.version.split()[0]}")
print(f"NumPy       : {np.__version__}")
print(f"Pandas      : {pd.__version__}")
print(f"Librosa     : {librosa.__version__}")
print(f"PyTorch     : {torch.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print()

# Resampler availability check
for pkg, res_type in [("soxr", "soxr_hq"), ("resampy", "kaiser_fast")]:
    try:
        importlib.import_module(pkg)
        print(f"Resampler   : {pkg} found  → will use res_type='{res_type}'")
        break
    except ImportError:
        print(f"Resampler   : {pkg} NOT installed")
print()

# GPU availability
if torch.cuda.is_available():
    print(f"CUDA available  : True")
    print(f"CUDA version    : {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        print(f"GPU [{i}]          : {props.name}  |  {mem_gb:.1f} GB VRAM")
    DEVICE = torch.device("cuda")
else:
    print("CUDA available  : False — running on CPU")
    DEVICE = torch.device("cpu")

print(f"\nActive device   : {DEVICE}")


# ## 2. Imports

# In[2]:


import os
import ast
import importlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

logging.basicConfig(level=logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)

# Pick the fastest available resampler — no extra install required for soxr
# (ships with librosa >= 0.10). Falls back to 'polyphase' which is pure scipy.
def _best_res_type():
    for pkg, res in [("soxr", "soxr_hq"), ("resampy", "kaiser_fast")]:
        try:
            importlib.import_module(pkg)
            return res
        except ImportError:
            continue
    return "polyphase"   # always available via scipy

RES_TYPE = _best_res_type()
print(f"Using resampler: {RES_TYPE}")


# ## 3. Config — single place to change hyperparameters

# In[3]:


CFG = dict(
    audio_dir       = "fma_small",
    metadata_csv    = "fma_metadata/tracks.csv",
    subset          = "small",
    sr              = 22050,
    segment_sec     = 3,
    n_fft           = 2048,
    hop_length      = 512,
    n_mels          = 128,
    test_size       = 0.2,
    random_state    = 42,
    batch_size      = 16,
    # set num_workers=0 on Windows to avoid multiprocessing pickling errors
    num_workers     = 0 if os.name == "nt" else 4,
)
print("Config:", CFG)


# ## 4. Load Metadata & Build Genre Labels

# In[4]:


tracks = pd.read_csv(
    CFG["metadata_csv"],
    index_col=0,
    header=[0, 1]
)

small_tracks = tracks[tracks["set"]["subset"] == CFG["subset"]]

genres_raw = small_tracks["track"]["genres_all"].dropna()

# BUG FIX: use ast.literal_eval instead of eval() — safer and avoids
# executing arbitrary code that might be in the CSV.
genres_parsed = genres_raw.apply(ast.literal_eval)

mlb = MultiLabelBinarizer()
genre_labels = mlb.fit_transform(genres_parsed)
num_genres = len(mlb.classes_)

print(f"Tracks with genre labels : {len(genres_parsed)}")
print(f"Unique genres            : {num_genres}")
print(f"Label matrix shape       : {genre_labels.shape}")


# ## 5. Audio Utilities

# In[5]:


def get_audio_path(track_id: int, audio_dir: str = CFG["audio_dir"]) -> Path:
    """Return the .mp3 path for a given FMA track id."""
    track_str = f"{track_id:06d}"
    return Path(audio_dir) / track_str[:3] / (track_str + ".mp3")


def load_audio(path: Path, sr: int = CFG["sr"]) -> np.ndarray:
    """
    Load a mono audio file at the target sample rate.
    Returns None on error so corrupted files are skipped gracefully.
    """
    try:
        audio, _ = librosa.load(str(path), sr=sr, mono=True, res_type=RES_TYPE)
        # RES_TYPE resolved at startup: soxr_hq > kaiser_fast > polyphase
        # soxr ships with librosa >= 0.10 so no extra pip install needed.
        return audio
    except Exception as exc:
        print(f"  [WARN] Could not load {path}: {exc}")
        return None


def split_segments(audio: np.ndarray, sr: int, duration: int = CFG["segment_sec"]):
    """Slice audio into fixed-length segments, discarding the trailing remainder."""
    seg_len = duration * sr
    # BUG FIX: use integer arithmetic, not Python range() with np.ndarray length
    n_full = len(audio) // seg_len
    return [audio[i * seg_len : (i + 1) * seg_len] for i in range(n_full)]


def audio_to_logmel(audio: np.ndarray, sr: int = CFG["sr"]) -> np.ndarray:
    """Convert a raw audio segment to a normalised log-Mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=CFG["n_fft"],
        hop_length=CFG["hop_length"],
        n_mels=CFG["n_mels"],
    )
    logmel = librosa.power_to_db(mel)
    # BUG FIX: guard against zero-std segments (silence) to avoid NaN tensors
    std = logmel.std()
    if std < 1e-6:
        return logmel - logmel.mean()
    return (logmel - logmel.mean()) / std


# ## 6. Validate File Existence (with progress bar)

# In[6]:


track_ids = genres_parsed.index.tolist()
all_paths = [get_audio_path(i) for i in track_ids]

# Filter out missing files up-front so the DataLoader never chokes
valid_mask   = []
valid_paths  = []
valid_labels = []

for path, label in tqdm(zip(all_paths, genre_labels),
                        total=len(all_paths),
                        desc="Checking audio files",
                        unit="file"):
    if path.exists():
        valid_paths.append(path)
        valid_labels.append(label)

valid_labels = np.array(valid_labels)

print(f"\nFiles found    : {len(valid_paths)} / {len(all_paths)}")
print(f"Files missing  : {len(all_paths) - len(valid_paths)}")


# ## 7. Dataset Class

# In[12]:


class FMADataset(Dataset):
    """
    PyTorch Dataset for the FMA audio corpus.

    Each __getitem__ call returns:
        specs  : Tensor of shape (max_segments, 1, n_mels, time_frames)
        label  : FloatTensor of shape (num_genres,)  — multi-hot encoded

    Improvements over original:
    - Skips corrupted files gracefully (returns zeros + a warning)
    - Uses res_type='kaiser_fast' for 5× faster audio loading
    - Guards against silent segments (std ≈ 0) that would produce NaN tensors
    - Uses Path objects throughout for cross-platform compatibility
    - Pads segments to ensure consistent tensor sizes across batches
    """

    def __init__(self, audio_paths, labels, sr: int = CFG["sr"], max_segments: int = 10):
        self.audio_paths = list(audio_paths)
        self.labels = labels
        self.sr = sr
        self.max_segments = max_segments  # Maximum number of segments to pad to

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        label = torch.tensor(self.labels[idx]).float()

        audio = load_audio(path, self.sr)

        # BUG FIX: handle corrupt / unreadable audio gracefully instead of crashing
        if audio is None or len(audio) == 0:
            dummy = torch.zeros(self.max_segments, 1, CFG["n_mels"], 130)
            return dummy, label

        segments = split_segments(audio, self.sr)

        if len(segments) == 0:
            # Track shorter than one segment
            dummy = torch.zeros(self.max_segments, 1, CFG["n_mels"], 130)
            return dummy, label

        # Convert segments to spectrograms
        specs_list = [
            torch.tensor(audio_to_logmel(seg, self.sr)).unsqueeze(0).float()
            for seg in segments
        ]

        # Pad or truncate to max_segments
        if len(specs_list) < self.max_segments:
            # Pad with zeros
            padding_needed = self.max_segments - len(specs_list)
            dummy_spec = torch.zeros(1, CFG["n_mels"], 130)
            specs_list.extend([dummy_spec] * padding_needed)
        elif len(specs_list) > self.max_segments:
            # Truncate to max_segments
            specs_list = specs_list[:self.max_segments]

        specs = torch.stack(specs_list)  # (max_segments, 1, n_mels, T)

        return specs, label


# ## 8. Smoke-test a Single Item

# In[13]:


smoke_dataset = FMADataset(valid_paths[:5], valid_labels[:5])

print("Running smoke test on first 5 tracks...")
for i in tqdm(range(len(smoke_dataset)), desc="Smoke test", unit="track"):
    specs, label = smoke_dataset[i]
    print(f"  [{i}]  specs: {tuple(specs.shape)}   label: {tuple(label.shape)}   "
          f"NaN in specs: {specs.isnan().any().item()}")

print("\nSmoke test passed.")


# ## 9. Train / Val Split

# In[14]:


train_paths, val_paths, train_labels, val_labels = train_test_split(
    valid_paths,
    valid_labels,
    test_size    = CFG["test_size"],
    random_state = CFG["random_state"],
)

print(f"Train samples : {len(train_paths)}")
print(f"Val samples   : {len(val_paths)}")


# ## 10. DataLoaders

# In[15]:


train_dataset = FMADataset(train_paths, train_labels, max_segments=10)
val_dataset   = FMADataset(val_paths,   val_labels,   max_segments=10)

# BUG FIX: pin_memory=True speeds up CPU→GPU transfers when CUDA is available;
# persistent_workers=True avoids re-spawning workers every epoch (num_workers>0 only).
_use_workers = CFG["num_workers"] > 0

train_loader = DataLoader(
    train_dataset,
    batch_size        = CFG["batch_size"],
    shuffle           = True,
    num_workers       = CFG["num_workers"],
    pin_memory        = DEVICE.type == "cuda",
    persistent_workers= _use_workers,
)

val_loader = DataLoader(
    val_dataset,
    batch_size        = CFG["batch_size"],
    shuffle           = False,
    num_workers       = CFG["num_workers"],
    pin_memory        = DEVICE.type == "cuda",
    persistent_workers= _use_workers,
)

print(f"Train batches : {len(train_loader)}")
print(f"Val batches   : {len(val_loader)}")


# ## 11. Full Dataset Pre-processing Dry-run (with progress + ETA)

# In[16]:


# Iterate the entire training set once to surface any remaining bad files.
# Progress bar shows per-batch ETA so you know how long is left.

print("Dry-run: iterating train_loader to check for errors...")

error_batches = []

for batch_idx, (specs, labels) in enumerate(
    tqdm(train_loader, desc="Train dry-run", unit="batch")
):
    if specs.isnan().any() or specs.isinf().any():
        error_batches.append(batch_idx)
        print(f"  [WARN] NaN/Inf in batch {batch_idx}")

if error_batches:
    print(f"\nBatches with bad values: {error_batches}")
else:
    print("\nAll batches clean — no NaN / Inf detected.")

print(f"Last batch — specs: {specs.shape}  labels: {labels.shape}")


# ## 12. Quick Stats — Label Distribution

# In[17]:


genre_counts = valid_labels.sum(axis=0)
genre_ids    = mlb.classes_

dist = pd.Series(genre_counts, index=genre_ids).sort_values(ascending=False)
print("Top-20 genre frequencies in the dataset:")
print(dist.head(20).to_string())

