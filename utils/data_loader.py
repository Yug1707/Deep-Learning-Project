"""
Data loading utilities that can be imported from notebooks.
This module provides the essential data pipeline functionality.
"""

import os
import ast
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# Configuration
CFG = {
    'audio_dir': "fma_small",
    'metadata_csv': "fma_metadata/tracks.csv",
    'subset': "small",
    'sr': 22050,
    'segment_sec': 3,
    'n_fft': 2048,
    'hop_length': 512,
    'n_mels': 128,
    'test_size': 0.2,
    'random_state': 42,
    'batch_size': 16,
    'num_workers': 0 if os.name == "nt" else 4,
}

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_audio_path(track_id: int, audio_dir: str = CFG["audio_dir"]) -> Path:
    """Return the .mp3 path for a given FMA track id."""
    track_str = f"{track_id:06d}"
    return Path(audio_dir) / track_str[:3] / (track_str + ".mp3")

def load_audio(path: Path, sr: int = CFG["sr"]) -> np.ndarray:
    """Load a mono audio file at the target sample rate."""
    try:
        audio, _ = librosa.load(str(path), sr=sr, mono=True, res_type='soxr_hq')
        return audio
    except Exception as exc:
        print(f"  [WARN] Could not load {path}: {exc}")
        return None

def split_segments(audio: np.ndarray, sr: int, duration: int = CFG["segment_sec"]):
    """Slice audio into fixed-length segments."""
    seg_len = duration * sr
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
    std = logmel.std()
    if std < 1e-6:
        return logmel - logmel.mean()
    return (logmel - logmel.mean()) / std

class FMADataset(Dataset):
    """PyTorch Dataset for the FMA audio corpus."""
    
    def __init__(self, audio_paths, labels, sr: int = CFG["sr"], max_segments: int = 10):
        self.audio_paths = list(audio_paths)
        self.labels = labels
        self.sr = sr
        self.max_segments = max_segments

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        label = torch.tensor(self.labels[idx]).float()

        audio = load_audio(path, self.sr)

        if audio is None or len(audio) == 0:
            dummy = torch.zeros(self.max_segments, 1, CFG["n_mels"], 130)
            return dummy, label

        segments = split_segments(audio, self.sr)

        if len(segments) == 0:
            dummy = torch.zeros(self.max_segments, 1, CFG["n_mels"], 130)
            return dummy, label

        specs_list = [
            torch.tensor(audio_to_logmel(seg, self.sr)).unsqueeze(0).float()
            for seg in segments
        ]

        if len(specs_list) < self.max_segments:
            padding_needed = self.max_segments - len(specs_list)
            dummy_spec = torch.zeros(1, CFG["n_mels"], 130)
            specs_list.extend([dummy_spec] * padding_needed)
        elif len(specs_list) > self.max_segments:
            specs_list = specs_list[:self.max_segments]

        specs = torch.stack(specs_list)
        return specs, label

def load_fma_data():
    """
    Load and prepare the FMA dataset.
    
    Returns:
        Tuple of (train_loader, val_loader, num_genres, mlb)
    """
    print("Loading FMA dataset...")
    
    # Load metadata
    tracks = pd.read_csv(CFG["metadata_csv"], index_col=0, header=[0, 1])
    small_tracks = tracks[tracks["set"]["subset"] == CFG["subset"]]
    genres_raw = small_tracks["track"]["genres_all"].dropna()
    genres_parsed = genres_raw.apply(ast.literal_eval)
    
    # Create multi-label binarizer
    mlb = MultiLabelBinarizer()
    genre_labels = mlb.fit_transform(genres_parsed)
    num_genres = len(mlb.classes_)
    
    # Get audio paths
    track_ids = genres_parsed.index.tolist()
    all_paths = [get_audio_path(i) for i in track_ids]
    
    # Filter existing files
    valid_paths = []
    valid_labels = []
    
    for path, label in tqdm(zip(all_paths, genre_labels), total=len(all_paths), desc="Checking files"):
        if path.exists():
            valid_paths.append(path)
            valid_labels.append(label)
    
    valid_labels = np.array(valid_labels)
    
    # Train/val split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        valid_paths, valid_labels, test_size=CFG["test_size"], random_state=CFG["random_state"]
    )
    
    # Create datasets
    train_dataset = FMADataset(train_paths, train_labels, max_segments=10)
    val_dataset = FMADataset(val_paths, val_labels, max_segments=10)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=CFG["batch_size"], shuffle=True, 
        num_workers=CFG["num_workers"], pin_memory=DEVICE.type == "cuda"
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=DEVICE.type == "cuda"
    )
    
    print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} validation samples")
    print(f"✓ Number of genres: {num_genres}")
    print(f"✓ Device: {DEVICE}")
    
    return train_loader, val_loader, num_genres, mlb, train_dataset, val_dataset

if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, num_genres, mlb, train_dataset, val_dataset = load_fma_data()
    print(f"Data loading test successful!")
