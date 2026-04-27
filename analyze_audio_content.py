#!/usr/bin/env python3
"""
Deep analysis of audio content quality and label distribution.
Checks for silent files, duration distribution, and class balance issues.
"""

import os
import sys
from pathlib import Path
import warnings
import ast
import json
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

warnings.filterwarnings('ignore')


class AudioContentAnalysis:
    """Analyze audio content quality beyond file corruption."""
    
    def __init__(self, audio_dir: str = "fma_small", metadata_csv: str = "fma_metadata/tracks.csv"):
        self.audio_dir = Path(audio_dir)
        self.metadata_csv = Path(metadata_csv)
        self.sr = 16000
        self.min_rms = 1e-5
        self.min_duration = 0.5
        
    def analyze_audio_content(self, max_files: int = 1000):
        """Analyze audio quality and duration characteristics."""
        print(f"\n🔍 Analyzing audio content quality (first {max_files} files)...\n")
        
        all_files = sorted(self.audio_dir.glob('**/*.mp3'))[:max_files]
        
        stats = {
            'total': len(all_files),
            'silent': 0,
            'very_short': 0,
            'short': 0,
            'normal': 0,
            'long': 0,
            'durations': [],
            'rms_energies': [],
            'silent_files': [],
            'short_files': [],
        }
        
        for audio_path in tqdm(all_files, desc="Analyzing content"):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    y, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)
                
                duration = len(y) / sr
                rms_energy = np.sqrt(np.mean(y ** 2))
                
                stats['durations'].append(duration)
                stats['rms_energies'].append(rms_energy)
                
                # Categorize
                if rms_energy < self.min_rms:
                    stats['silent'] += 1
                    stats['silent_files'].append(str(audio_path))
                elif duration < self.min_duration:
                    stats['very_short'] += 1
                    stats['short_files'].append(str(audio_path))
                elif duration < 1.0:
                    stats['short'] += 1
                    stats['short_files'].append(str(audio_path))
                elif duration > 300:  # >5 min
                    stats['long'] += 1
                else:
                    stats['normal'] += 1
                    
            except Exception as e:
                print(f"Error analyzing {audio_path}: {e}")
        
        return stats
    
    def analyze_label_distribution(self):
        """Check class balance and label statistics."""
        print(f"\n📊 Analyzing label distribution...\n")
        
        if not self.metadata_csv.exists():
            print(f"⚠️  Metadata not found: {self.metadata_csv}")
            return None
        
        try:
            tracks = pd.read_csv(self.metadata_csv, index_col=0, header=[0, 1])
            small_tracks = tracks[tracks[("set", "subset")] == "small"]
            genres_raw = small_tracks[("track", "genres_all")].dropna()
            genres_parsed = genres_raw.apply(ast.literal_eval)
            
            # Count genre distribution
            genre_counter = Counter()
            for genre_list in genres_parsed:
                genre_counter.update(genre_list)
            
            # Multi-label stats
            label_counts = [len(g) for g in genres_parsed]
            
            stats = {
                'total_tracks': len(genres_parsed),
                'unique_genres': len(genre_counter),
                'top_genres': dict(genre_counter.most_common(10)),
                'avg_labels_per_track': np.mean(label_counts),
                'max_labels_per_track': max(label_counts),
                'min_labels_per_track': min(label_counts),
                'label_distribution': Counter(label_counts)
            }
            
            return stats
        except Exception as e:
            print(f"Error analyzing labels: {e}")
            return None
    
    def generate_report(self, audio_stats, label_stats):
        """Generate comprehensive analysis report."""
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    AUDIO CONTENT QUALITY ANALYSIS REPORT                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

AUDIO CONTENT STATISTICS
------------------------
Total files analyzed:       {audio_stats['total']:6d}
✓ Normal files:             {audio_stats['normal']:6d}  ({100*audio_stats['normal']/audio_stats['total']:.1f}%)
⊘ Silent files (RMS<1e-5):  {audio_stats['silent']:6d}  ({100*audio_stats['silent']/audio_stats['total']:.1f}%)
⚠ Very short (<0.5s):       {audio_stats['very_short']:6d}  ({100*audio_stats['very_short']/audio_stats['total']:.1f}%)
⚠ Short (0.5-1s):          {audio_stats['short']:6d}  ({100*audio_stats['short']/audio_stats['total']:.1f}%)
🔊 Long files (>5min):      {audio_stats['long']:6d}  ({100*audio_stats['long']/audio_stats['total']:.1f}%)

AUDIO DURATION DISTRIBUTION
---------------------------
"""
        if audio_stats['durations']:
            durations = np.array(audio_stats['durations'])
            report += f"""Mean duration:              {np.mean(durations):6.2f} seconds
Median duration:            {np.median(durations):6.2f} seconds
Min duration:               {np.min(durations):6.2f} seconds
Max duration:               {np.max(durations):6.2f} seconds
Std deviation:              {np.std(durations):6.2f} seconds
25th percentile:            {np.percentile(durations, 25):6.2f} seconds
75th percentile:            {np.percentile(durations, 75):6.2f} seconds
"""
        
        report += f"""
RMS ENERGY DISTRIBUTION (Audio Loudness)
--------------------------------------
"""
        if audio_stats['rms_energies']:
            rms = np.array(audio_stats['rms_energies'])
            report += f"""Mean RMS energy:            {np.mean(rms):6.2e}
Min RMS energy:             {np.min(rms):6.2e}
Max RMS energy:             {np.max(rms):6.2e}
Silent threshold (1e-5):    Used for detecting silent files
"""
        
        if label_stats:
            report += f"""
LABEL/CLASS DISTRIBUTION
-----------------------
Total tracks with labels:   {label_stats['total_tracks']:6d}
Unique genres:              {label_stats['unique_genres']:6d}
Avg labels per track:       {label_stats['avg_labels_per_track']:6.2f}
Max labels per track:       {label_stats['max_labels_per_track']:6d}
Min labels per track:       {label_stats['min_labels_per_track']:6d}

Top 10 genres by frequency:
"""
            for i, (genre_id, count) in enumerate(label_stats['top_genres'].items(), 1):
                report += f"  {i:2d}. Genre {genre_id:4d}: {count:5d} tracks\n"
            
            # Check imbalance
            counts = list(label_stats['top_genres'].values())
            if len(counts) > 1:
                imbalance_ratio = max(counts) / (min(counts) + 1e-6)
                report += f"\nClass imbalance ratio:      {imbalance_ratio:.2f}x (top/bottom)"
                if imbalance_ratio > 5:
                    report += " ⚠️  SEVERE IMBALANCE - This hurts model training!"
                elif imbalance_ratio > 2:
                    report += " ⚠️  MODERATE IMBALANCE - Consider weighted loss"
        
        report += f"""
POTENTIAL ISSUES DETECTED
------------------------
"""
        issues = []
        
        if audio_stats['silent'] > audio_stats['total'] * 0.05:
            issues.append(f"• {audio_stats['silent']} silent files (>{5}%) will produce meaningless spectrograms")
        
        if audio_stats['very_short'] + audio_stats['short'] > audio_stats['total'] * 0.1:
            short_count = audio_stats['very_short'] + audio_stats['short']
            issues.append(f"• {short_count} very short files (<1s) may not have enough info for learning")
        
        mean_duration = np.mean(audio_stats['durations']) if audio_stats['durations'] else 0
        if mean_duration < 30:
            issues.append(f"• Mean audio duration is only {mean_duration:.1f}s - very short!")
        
        if not issues:
            issues.append("✓ No major content quality issues detected")
        
        for issue in issues:
            report += issue + "\n"
        
        report += f"""
⚠️  INTERPRETATION
If files load successfully but model performance is terrible:
1. Check RMS energy distribution - many silent files?
2. Check duration distribution - too short (<30s)?
3. Check class imbalance - some genres rare?
4. The problem might not be file corruption, but data characteristics.

RECOMMENDATION
The audio appears to be loadable. If training metrics are still poor:
- Try training with class-weighted loss (imbalanced classes)
- Check if spectrograms are computed correctly (visualize samples)
- Verify labels are correct (confusion matrix)
- Consider that the model needs more training time or hyperparameter tuning
"""
        return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze audio content quality")
    parser.add_argument('--audio-dir', default='fma_small')
    parser.add_argument('--metadata-csv', default='fma_metadata/tracks.csv')
    parser.add_argument('--max-files', type=int, default=1000)
    args = parser.parse_args()
    
    analyzer = AudioContentAnalysis(
        audio_dir=args.audio_dir,
        metadata_csv=args.metadata_csv
    )
    
    audio_stats = analyzer.analyze_audio_content(max_files=args.max_files)
    label_stats = analyzer.analyze_label_distribution()
    
    report = analyzer.generate_report(audio_stats, label_stats)
    print(report)
    
    # Save results
    with open('audio_content_analysis.json', 'w') as f:
        json.dump({
            'audio_stats': {
                'total': audio_stats['total'],
                'silent': audio_stats['silent'],
                'very_short': audio_stats['very_short'],
                'short': audio_stats['short'],
                'normal': audio_stats['normal'],
                'long': audio_stats['long'],
                'mean_duration': float(np.mean(audio_stats['durations'])),
                'mean_rms': float(np.mean(audio_stats['rms_energies'])),
            },
            'label_stats': label_stats
        }, f, indent=2)
    print("\n✓ Results saved to audio_content_analysis.json")


if __name__ == "__main__":
    main()
