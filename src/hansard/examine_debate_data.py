#!/usr/bin/env python3
"""
Examine the structure of debate data files
"""

import pandas as pd
from pathlib import Path

def examine_debates():
    """Examine debate data structure from parsed debates"""

    # Look at a sample year
    debate_path = Path("src/hansard/data/processed_fixed/metadata/debates_1950.parquet")

    if not debate_path.exists():
        print(f"File not found: {debate_path}")
        # Try another year
        debate_path = Path("src/hansard/data/processed_fixed/metadata/debates_1920.parquet")

    if not debate_path.exists():
        # List available years
        meta_dir = Path("src/hansard/data/processed_fixed/metadata/")
        debate_files = sorted(meta_dir.glob("debates_*.parquet"))
        if debate_files:
            print(f"Found {len(debate_files)} debate files")
            debate_path = debate_files[len(debate_files)//2]  # Pick middle year
            print(f"Using {debate_path}")
        else:
            print("No debate files found")
            return None

    print(f"Loading {debate_path}...")
    df = pd.read_parquet(debate_path)

    print(f"\n=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print(f"\n=== FIRST 3 ROWS ===")
    print(df.head(3))

    print(f"\n=== DATA TYPES ===")
    print(df.dtypes)

    print(f"\n=== UNIQUE VALUES ===")
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['title', 'topics', 'meta_tags', 'speakers', 'reference_columns']:
            try:
                print(f"{col}: {df[col].nunique()} unique")
            except TypeError:
                print(f"{col}: contains unhashable types")

    # Check speakers column if it exists
    if 'speakers' in df.columns:
        print(f"\n=== SPEAKERS COLUMN ===")
        print(f"Type: {type(df['speakers'].iloc[0])}")
        print(f"Sample speakers from first debate:")
        if pd.api.types.is_list_like(df['speakers'].iloc[0]):
            print(f"  {df['speakers'].iloc[0][:5]}")  # First 5 speakers

    # Check text columns
    text_cols = [c for c in df.columns if 'text' in c.lower() or 'content' in c.lower()]
    if text_cols:
        print(f"\n=== TEXT COLUMNS ===")
        for col in text_cols:
            sample = df[col].iloc[0] if len(df) > 0 else ""
            print(f"{col}: {len(str(sample))} chars in first row")

    return df

if __name__ == "__main__":
    df = examine_debates()