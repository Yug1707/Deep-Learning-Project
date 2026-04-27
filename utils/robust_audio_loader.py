#!/usr/bin/env python3
"""
Improved data loading with robust error handling and audio validation.
Supports both MP3 and WAV formats with fallback mechanisms.
"""

import warnings
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


class AudioValidator:
    """Validate audio quality and detect issues."""
    
    @staticmethod
    def validate_waveform(
        waveform: np.ndarray,
        sr: int,
        min_duration_sec: float = 0.5,
        max_duration_sec: float = 600,
        min_rms: float = 1e-5,
        verbose: bool = False
    ) -> Tuple[bool, str]:
        """
        Validate audio waveform.
        
        Returns:
            (is_valid, reason)
        """
        if waveform is None or waveform.size == 0:
            return False, "Empty waveform"
        
        duration = len(waveform) / sr
        
        # Check duration
        if duration < min_duration_sec:
            return False, f"Too short ({duration:.2f}s < {min_duration_sec}s)"
        if duration > max_duration_sec:
            return False, f"Too long ({duration:.2f}s > {max_duration_sec}s)"
        
        # Check for silence
        rms_energy = np.sqrt(np.mean(waveform ** 2))
        if rms_energy < min_rms:
            return False, f"Silent (RMS={rms_energy:.2e} < {min_rms})"
        
        # Check for NaN/Inf
        if not np.isfinite(waveform).all():
            return False, "Contains NaN or Inf values"
        
        return True, "Valid"


class AudioLoader:
    """Load audio files with multiple backends and fallbacks."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        mono: bool = True,
        dtype: str = 'float32',
        verbosity: int = 0
    ):
        self.sr = sample_rate
        self.mono = mono
        self.dtype = dtype
        self.verbosity = verbosity
    
    def load_librosa(self, path: Path) -> Optional[np.ndarray]:
        """Load using librosa (supports many formats via audioread)."""
        if not HAS_LIBROSA:
            return None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                y, _ = librosa.load(
                    str(path),
                    sr=self.sr,
                    mono=self.mono,
                    res_type='soxr_hq'
                )
            return y.astype(self.dtype)
        except Exception as e:
            if self.verbosity > 1:
                print(f"  librosa failed: {type(e).__name__}")
            return None
    
    def load_soundfile(self, path: Path) -> Optional[np.ndarray]:
        """Load using soundfile (WAV/FLAC support)."""
        if not HAS_SOUNDFILE:
            return None
        try:
            y, sr = sf.read(str(path))
            
            # Resample if needed
            if sr != self.sr and HAS_LIBROSA:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
            
            # Convert to mono
            if self.mono and y.ndim > 1:
                y = np.mean(y, axis=1)
            
            return y.astype(self.dtype)
        except Exception as e:
            if self.verbosity > 1:
                print(f"  soundfile failed: {type(e).__name__}")
            return None
    
    def load(self, path: Path, validate: bool = True) -> Tuple[Optional[np.ndarray], str]:
        """
        Load audio with fallback mechanisms.
        
        Returns:
            (waveform, error_message or "success")
        """
        path = Path(path)
        if not path.exists():
            return None, f"File not found: {path}"
        
        # Try based on format
        if path.suffix.lower() == '.wav':
            # Try soundfile first for WAV
            waveform = self.load_soundfile(path)
            if waveform is not None:
                if validate:
                    is_valid, reason = AudioValidator.validate_waveform(waveform, self.sr)
                    if is_valid:
                        return waveform, "success"
                    else:
                        return None, reason
                return waveform, "success"
            # Fallback to librosa
            waveform = self.load_librosa(path)
            if waveform is not None:
                if validate:
                    is_valid, reason = AudioValidator.validate_waveform(waveform, self.sr)
                    if is_valid:
                        return waveform, "success"
                    else:
                        return None, reason
                return waveform, "success"
        else:
            # Try librosa first for MP3/other formats
            waveform = self.load_librosa(path)
            if waveform is not None:
                if validate:
                    is_valid, reason = AudioValidator.validate_waveform(waveform, self.sr)
                    if is_valid:
                        return waveform, "success"
                    else:
                        return None, reason
                return waveform, "success"
            # Fallback to soundfile
            waveform = self.load_soundfile(path)
            if waveform is not None:
                if validate:
                    is_valid, reason = AudioValidator.validate_waveform(waveform, self.sr)
                    if is_valid:
                        return waveform, "success"
                    else:
                        return None, reason
                return waveform, "success"
        
        return None, "Failed all decode attempts"


class ImprovedAudioDataset(Dataset):
    """Dataset with robust error handling and optional caching."""
    
    def __init__(
        self,
        audio_paths: List[Path],
        labels: np.ndarray,
        sample_rate: int = 16000,
        segment_length_sec: Optional[float] = None,
        max_segments: int = 10,
        return_metadata: bool = False,
        on_error: str = 'skip'  # 'skip' or 'zero'
    ):
        """
        Args:
            audio_paths: List of audio file paths
            labels: Label array (N_samples, N_classes)
            sample_rate: Target sample rate
            segment_length_sec: If set, split audio into segments
            max_segments: Max segments per audio
            return_metadata: If True, return path and validity info
            on_error: How to handle decode errors ('skip' returns None, 'zero' returns zeros)
        """
        self.audio_paths = [Path(p) for p in audio_paths]
        self.labels = labels
        self.sr = sample_rate
        self.segment_length_sec = segment_length_sec
        self.max_segments = max_segments
        self.return_metadata = return_metadata
        self.on_error = on_error
        
        self.loader = AudioLoader(sample_rate=sample_rate, verbosity=0)
        self.validator = AudioValidator()
        
        # Pre-validate dataset
        self._valid_indices = self._validate_dataset()
    
    def _validate_dataset(self) -> List[int]:
        """Check which files are loadable."""
        valid = []
        for i, path in enumerate(self.audio_paths):
            waveform, status = self.loader.load(path, validate=True)
            if waveform is not None:
                valid.append(i)
        return valid
    
    def __len__(self) -> int:
        if self.on_error == 'skip':
            return len(self._valid_indices)
        return len(self.audio_paths)
    
    def _get_valid_idx(self, idx: int) -> int:
        """Map dataset index to valid file index."""
        if self.on_error == 'skip':
            return self._valid_indices[idx]
        return idx
    
    def __getitem__(self, idx: int) -> dict:
        actual_idx = self._get_valid_idx(idx)
        path = self.audio_paths[actual_idx]
        label = torch.tensor(self.labels[actual_idx]).float()
        
        # Try to load
        waveform, status = self.loader.load(path, validate=True)
        
        if waveform is None:
            if self.on_error == 'zero':
                # Return zero tensor
                specs = torch.zeros((self.max_segments, 128, 130))  # Placeholder shape
                return {
                    'waveform': specs,
                    'labels': label,
                    'track_id': actual_idx,
                    'is_valid': False,
                    'error': status,
                    'path': str(path)
                }
            else:
                # Should not reach here if on_error='skip'
                raise RuntimeError(f"Failed to load {path}: {status}")
        
        # Segment if requested
        if self.segment_length_sec is not None:
            segment_samples = int(self.segment_length_sec * self.sr)
            segments = self._segment_audio(waveform, segment_samples)
            waveform = segments  # Return segments instead
        
        return {
            'waveform': torch.tensor(waveform, dtype=torch.float32),
            'labels': label,
            'track_id': actual_idx,
            'is_valid': True,
            'error': None,
            'path': str(path)
        }
    
    def _segment_audio(self, waveform: np.ndarray, segment_samples: int) -> np.ndarray:
        """Split audio into fixed-length segments."""
        n_segments = min((len(waveform) // segment_samples) + 1, self.max_segments)
        
        segments = []
        for i in range(n_segments):
            start = i * segment_samples
            end = min(start + segment_samples, len(waveform))
            segment = waveform[start:end]
            
            # Pad to exact length
            if len(segment) < segment_samples:
                segment = np.pad(segment, (0, segment_samples - len(segment)), mode='constant')
            
            segments.append(segment)
        
        # Pad to max_segments
        while len(segments) < self.max_segments:
            segments.append(np.zeros(segment_samples))
        
        return np.array(segments[:self.max_segments])


if __name__ == "__main__":
    # Quick test
    print("AudioLoader test...")
    loader = AudioLoader(sample_rate=16000)
    
    # Test with a sample file (if available)
    test_file = Path("fma_small/000/000000.mp3")
    if test_file.exists():
        waveform, status = loader.load(test_file, validate=True)
        if waveform is not None:
            print(f"✓ Loaded {test_file}: {len(waveform)} samples, {len(waveform)/16000:.2f}s")
        else:
            print(f"✗ Failed to load {test_file}: {status}")
    else:
        print(f"Test file not found: {test_file}")
