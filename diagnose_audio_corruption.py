#!/usr/bin/env python3
"""
Diagnostic tool to detect and report audio file corruption in the FMA dataset.
Identifies files that fail to decode and provides actionable statistics.
"""

import os
import sys
from pathlib import Path
import warnings
import json
import subprocess
from collections import defaultdict
import argparse

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*PySoundFile.*')
warnings.filterwarnings('ignore', message='.*audioread.*')


class AudioDiagnostics:
    """Diagnose and report audio file corruption."""
    
    def __init__(self, audio_dir: str = "fma_small", sr: int = 16000):
        self.audio_dir = Path(audio_dir)
        self.sr = sr
        self.results = {
            'valid_files': [],
            'corrupted_files': [],
            'silent_files': [],
            'very_short_files': [],
            'errors': defaultdict(int),
        }
        self.min_duration = 0.5  # seconds
        
    def test_librosa_load(self, audio_path: Path) -> tuple:
        """Try to load file with librosa. Returns (success, duration, error_msg)."""
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                waveform, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)
            
            duration = len(waveform) / sr
            
            # Check if silent
            rms_energy = np.sqrt(np.mean(waveform ** 2))
            if rms_energy < 1e-4:  # essentially silent
                return False, duration, 'SILENT'
            
            # Check if very short
            if duration < self.min_duration:
                return False, duration, f'TOO_SHORT ({duration:.3f}s < {self.min_duration}s)'
            
            return True, duration, None
        except Exception as e:
            error_type = type(e).__name__
            return False, 0.0, error_type
    
    def test_ffmpeg_probe(self, audio_path: Path) -> tuple:
        """Validate file with ffprobe. Returns (success, error_msg)."""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                 '-of', 'default=noprint_wrappers=1:nokey=1:nokey=1', str(audio_path)],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return True, None
            return False, 'ffprobe_failed'
        except subprocess.TimeoutExpired:
            return False, 'ffprobe_timeout'
        except FileNotFoundError:
            return None, 'ffprobe_not_found'  # Skip this check
        except Exception as e:
            return False, type(e).__name__
    
    def scan_dataset(self, max_files: int = None) -> dict:
        """Scan all audio files and categorize them."""
        print(f"\n📊 Scanning audio files in {self.audio_dir}...")
        
        # Find all MP3 files
        all_files = sorted(self.audio_dir.glob('**/*.mp3'))
        
        if max_files:
            all_files = all_files[:max_files]
        
        if not all_files:
            print("❌ No MP3 files found!")
            return self.results
        
        print(f"Found {len(all_files)} MP3 files to scan\n")
        
        for audio_path in tqdm(all_files, desc="Scanning files"):
            # Test with librosa
            success, duration, error = self.test_librosa_load(audio_path)
            
            if success:
                self.results['valid_files'].append({
                    'path': str(audio_path),
                    'duration': duration
                })
            else:
                if error == 'SILENT':
                    self.results['silent_files'].append({
                        'path': str(audio_path),
                        'duration': duration
                    })
                elif 'TOO_SHORT' in error:
                    self.results['very_short_files'].append({
                        'path': str(audio_path),
                        'duration': duration,
                        'reason': error
                    })
                else:
                    self.results['corrupted_files'].append({
                        'path': str(audio_path),
                        'error': error
                    })
                    self.results['errors'][error] += 1
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate a summary report."""
        total = len(self.results['valid_files']) + len(self.results['corrupted_files']) + \
                len(self.results['silent_files']) + len(self.results['very_short_files'])
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        AUDIO CORRUPTION DIAGNOSTIC REPORT                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

SUMMARY
-------
Total files scanned:        {total:6d}
✓ Valid files:              {len(self.results['valid_files']):6d}  ({100*len(self.results['valid_files'])/total if total>0 else 0:.1f}%)
✗ Corrupted files:          {len(self.results['corrupted_files']):6d}  ({100*len(self.results['corrupted_files'])/total if total>0 else 0:.1f}%)
⊘ Silent files:             {len(self.results['silent_files']):6d}  ({100*len(self.results['silent_files'])/total if total>0 else 0:.1f}%)
⚠ Very short files:         {len(self.results['very_short_files']):6d}  ({100*len(self.results['very_short_files'])/total if total>0 else 0:.1f}%)

CORRUPTION BREAKDOWN
--------------------
"""
        for error_type, count in sorted(self.results['errors'].items(), key=lambda x: x[1], reverse=True):
            report += f"  • {error_type:30s} {count:5d} files\n"
        
        report += f"""
RECOMMENDATIONS
---------------
1. Your training is receiving {len(self.results['corrupted_files']) + len(self.results['silent_files'])} unusable files.
2. This explains the poor metrics: garbage input → garbage output.
3. Next steps:
   a) Create a cleaned dataset by re-encoding corrupted files to WAV format
   b) Discard corrupted files beyond recovery (~5-10% loss is acceptable)
   c) Re-run training with the cleaned dataset
   
4. For re-encoding, see fix_audio_corruption.py script.

⚠️  DATA QUALITY IMPACT
If >20% of your dataset is corrupted, your model cannot learn meaningful patterns.
The AST model falls back to returning zeros for failed files, polluting gradients.

"""
        return report
    
    def save_results(self, output_file: str = "audio_diagnostic_results.json"):
        """Save detailed results to JSON."""
        # Convert to serializable format
        serializable = {
            'valid_files': self.results['valid_files'][:100],  # Sample first 100
            'corrupted_files': self.results['corrupted_files'][:100],
            'silent_files': self.results['silent_files'][:100],
            'very_short_files': self.results['very_short_files'][:100],
            'error_summary': dict(self.results['errors']),
            'counts': {
                'total_valid': len(self.results['valid_files']),
                'total_corrupted': len(self.results['corrupted_files']),
                'total_silent': len(self.results['silent_files']),
                'total_short': len(self.results['very_short_files']),
            }
        }
        with open(output_file, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"✓ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose audio file corruption in FMA dataset"
    )
    parser.add_argument('--audio-dir', default='fma_small', 
                       help='Path to audio directory')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Sample rate for loading')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Max files to scan (for quick testing)')
    parser.add_argument('--output', default='audio_diagnostic_results.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    diagnostics = AudioDiagnostics(
        audio_dir=args.audio_dir,
        sr=args.sample_rate
    )
    
    results = diagnostics.scan_dataset(max_files=args.max_files)
    report = diagnostics.generate_report()
    print(report)
    diagnostics.save_results(args.output)


if __name__ == "__main__":
    main()
