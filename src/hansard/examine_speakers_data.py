#!/usr/bin/env python3
"""
Examine the structure of speaker data files
"""

import pandas as pd
from pathlib import Path

def examine_speakers():
    """Examine speaker data structure from parsed debates"""

    # Look at a sample year
    speaker_path = Path("src/hansard/data/processed_fixed/metadata/speakers_1950.parquet")

    if not speaker_path.exists():
        print(f"File not found: {speaker_path}")
        # Try another year
        speaker_path = Path("src/hansard/data/processed_fixed/metadata/speakers_1920.parquet")

    if not speaker_path.exists():
        # List available years
        meta_dir = Path("src/hansard/data/processed_fixed/metadata/")
        speaker_files = sorted(meta_dir.glob("speakers_*.parquet"))
        if speaker_files:
            print(f"Found {len(speaker_files)} speaker files")
            speaker_path = speaker_files[len(speaker_files)//2]  # Pick middle year
            print(f"Using {speaker_path}")
        else:
            print("No speaker files found")
            return None

    print(f"Loading {speaker_path}...")
    df = pd.read_parquet(speaker_path)

    print(f"\n=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print(f"\n=== FIRST 10 ROWS ===")
    print(df.head(10))

    print(f"\n=== DATA TYPES ===")
    print(df.dtypes)

    print(f"\n=== UNIQUE VALUES ===")
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"{col}: {df[col].nunique()} unique")

    # Check speaker names
    if 'speaker' in df.columns:
        print(f"\n=== TOP SPEAKERS ===")
        print(df['speaker'].value_counts().head(20))

    # Check if there's any gender info
    gender_cols = [c for c in df.columns if 'gender' in c.lower()]
    if gender_cols:
        print(f"\n=== EXISTING GENDER INFO ===")
        for col in gender_cols:
            print(f"{col}: {df[col].value_counts()}")

    return df

if __name__ == "__main__":
    df = examine_speakers()