#!/usr/bin/env python3
"""
Convert manifest CSV files from MP3 paths to WAV paths.
Run this after re-encoding audio files to update your manifests.
"""

import argparse
from pathlib import Path
import pandas as pd


def convert_manifest_paths(manifest_file: str, audio_dir: str = "fma_small") -> pd.DataFrame:
    """Convert manifest audio_path from .mp3 to .wav."""
    df = pd.read_csv(manifest_file)
    
    def mp3_to_wav(path: str) -> str:
        """Convert .mp3 path to .wav."""
        return path.replace('.mp3', '.wav')
    
    df['audio_path'] = df['audio_path'].apply(mp3_to_wav)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Convert manifest files from MP3 to WAV paths"
    )
    parser.add_argument('--manifests-dir', default='logs/ast_pipeline/manifests',
                       help='Directory containing manifest CSV files')
    parser.add_argument('--backup', action='store_true',
                       help='Create backup of original files')
    args = parser.parse_args()
    
    manifests_dir = Path(args.manifests_dir)
    if not manifests_dir.exists():
        print(f"❌ Manifests directory not found: {manifests_dir}")
        return
    
    manifest_files = [
        manifests_dir / "train_manifest.csv",
        manifests_dir / "val_manifest.csv",
        manifests_dir / "test_manifest.csv",
    ]
    
    for manifest_file in manifest_files:
        if not manifest_file.exists():
            print(f"⚠️  Skipping missing file: {manifest_file}")
            continue
        
        print(f"Converting {manifest_file.name}...", end=" ")
        
        # Backup if requested
        if args.backup:
            backup_file = manifest_file.with_suffix('.csv.bak')
            manifest_file.rename(backup_file)
            print(f"(backed up to {backup_file.name})", end=" ")
        
        # Convert
        df = convert_manifest_paths(str(manifest_file))
        df.to_csv(manifest_file, index=False)
        
        print("✓")
        
        # Verify conversion
        df_check = pd.read_csv(manifest_file)
        n_wav = sum(1 for p in df_check['audio_path'] if p.endswith('.wav'))
        print(f"  → {n_wav}/{len(df_check)} files now point to .wav")


if __name__ == "__main__":
    main()
