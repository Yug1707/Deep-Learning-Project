#!/usr/bin/env python3
"""
Fix audio corruption by re-encoding MP3 files to WAV format.
WAV (PCM) files are much more reliable than MP3, reducing decoder errors.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class AudioRepair:
    """Re-encode corrupted audio files to WAV format."""
    
    def __init__(self, audio_dir: str = "fma_small", output_dir: str = None, sr: int = 16000):
        self.audio_dir = Path(audio_dir)
        self.sr = sr
        self.output_dir = Path(output_dir) if output_dir else self.audio_dir
        self.stats = defaultdict(int)
        
    def has_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        result = subprocess.run(['which', 'ffmpeg'], capture_output=True)
        return result.returncode == 0
    
    def reencode_with_ffmpeg(self, input_file: Path, output_file: Path) -> bool:
        """Use ffmpeg to re-encode MP3 to WAV."""
        try:
            cmd = [
                'ffmpeg', '-i', str(input_file),
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', str(self.sr),      # Target sample rate
                '-ac', '1',                # Mono
                '-y',                      # Overwrite
                str(output_file)
            ]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
    
    def reencode_with_librosa(self, input_file: Path, output_file: Path) -> bool:
        """Use librosa to re-encode audio."""
        if not HAS_LIBROSA:
            return False
        try:
            import soundfile as sf
            y, sr = librosa.load(str(input_file), sr=self.sr, mono=True)
            if len(y) > 0:
                sf.write(str(output_file), y, self.sr, subtype='PCM_16')
                return True
        except Exception:
            pass
        return False
    
    def process_file(self, mp3_path: Path) -> dict:
        """Process a single MP3 file. Return success status."""
        # Keep same directory structure, just change extension
        wav_path = mp3_path.with_suffix('.wav')
        
        # Skip if WAV already exists
        if wav_path.exists():
            return {'path': str(mp3_path), 'status': 'already_exists', 'size': wav_path.stat().st_size}
        
        # Try ffmpeg first (more reliable)
        if self.has_ffmpeg():
            if self.reencode_with_ffmpeg(mp3_path, wav_path):
                return {'path': str(mp3_path), 'status': 'success_ffmpeg', 'size': wav_path.stat().st_size}
        
        # Fallback to librosa
        if HAS_LIBROSA and self.reencode_with_librosa(mp3_path, wav_path):
            return {'path': str(mp3_path), 'status': 'success_librosa', 'size': wav_path.stat().st_size}
        
        return {'path': str(mp3_path), 'status': 'failed', 'size': 0}
    
    def process_dataset(self, max_files: int = None, workers: int = 4) -> dict:
        """Process all MP3 files in dataset."""
        print(f"\n🔧 Re-encoding audio files to WAV...")
        print(f"Source: {self.audio_dir}")
        print(f"Target sample rate: {self.sr} Hz\n")
        
        # Find all MP3 files
        mp3_files = sorted(self.audio_dir.glob('**/*.mp3'))
        
        if max_files:
            mp3_files = mp3_files[:max_files]
        
        if not mp3_files:
            print("❌ No MP3 files found!")
            return {'error': 'no_files_found'}
        
        print(f"Found {len(mp3_files)} MP3 files\n")
        
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.process_file, mp3): mp3 for mp3 in mp3_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Re-encoding"):
                try:
                    result = future.result()
                    results.append(result)
                    self.stats[result['status']] += 1
                except Exception as e:
                    results.append({'status': 'error', 'error': str(e)})
                    self.stats['error'] += 1
        
        return {'results': results, 'stats': dict(self.stats)}
    
    def generate_report(self) -> str:
        """Generate summary report."""
        total = sum(self.stats.values())
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      AUDIO RE-ENCODING REPAIR REPORT                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

RE-ENCODING SUMMARY
-------------------
Total files processed:      {total:6d}
✓ Success (ffmpeg):         {self.stats.get('success_ffmpeg', 0):6d}
✓ Success (librosa):        {self.stats.get('success_librosa', 0):6d}
⚠ Already exist:            {self.stats.get('already_exists', 0):6d}
✗ Failed:                   {self.stats.get('failed', 0):6d}
✗ Errors:                   {self.stats.get('error', 0):6d}

NEXT STEPS
----------
1. Update your DataLoader to use WAV files instead of MP3:
   
   OLD (broken):
     audio, sr = librosa.load("path/audio.mp3", sr=16000)
   
   NEW (fixed):
     audio, sr = librosa.load("path/audio.wav", sr=16000)

2. Re-run your AST training - metrics should improve significantly.

3. Monitor for any remaining load failures in training logs.

"""
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Re-encode audio files to WAV to fix corruption issues"
    )
    parser.add_argument('--audio-dir', default='fma_small',
                       help='Path to audio directory')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Target sample rate')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Max files to process (for testing)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    repair = AudioRepair(
        audio_dir=args.audio_dir,
        sr=args.sample_rate
    )
    
    if not repair.has_ffmpeg():
        print("⚠️  ffmpeg not found. Installing...")
        subprocess.run(['apt-get', 'update'], capture_output=True)
        subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], capture_output=True)
        print("✓ ffmpeg installed\n")
    
    results = repair.process_dataset(
        max_files=args.max_files,
        workers=args.workers
    )
    
    report = repair.generate_report()
    print(report)


if __name__ == "__main__":
    main()
