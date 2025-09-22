#!/usr/bin/env python3
"""
Examine existing conversation turns data
"""

import pandas as pd
from pathlib import Path

def examine_turns():
    """Examine the structure of conversation turns data"""

    turns_path = Path("src/hansard/debate_datasets/conversation_turns_1950_sample.parquet")
    metadata_path = Path("src/hansard/debate_datasets/debate_metadata_1950_sample.parquet")

    print("=== CONVERSATION TURNS DATA ===")
    if turns_path.exists():
        turns_df = pd.read_parquet(turns_path)
        print(f"Shape: {turns_df.shape}")
        print(f"Columns: {list(turns_df.columns)}")
        print(f"\nFirst 5 rows:")
        print(turns_df.head())

        print(f"\nData types:")
        print(turns_df.dtypes)

        # Check unique debates
        if 'debate_id' in turns_df.columns:
            print(f"\nUnique debates: {turns_df['debate_id'].nunique()}")
            print(f"Turns per debate (avg): {len(turns_df) / turns_df['debate_id'].nunique():.1f}")

        # Check speakers
        if 'speaker' in turns_df.columns:
            print(f"\nUnique speakers: {turns_df['speaker'].nunique()}")
            print(f"Top speakers:")
            print(turns_df['speaker'].value_counts().head(10))

        # Check text length distribution
        if 'text' in turns_df.columns:
            turns_df['text_len'] = turns_df['text'].str.len()
            print(f"\nText length distribution:")
            print(turns_df['text_len'].describe())

    else:
        print(f"Turns file not found: {turns_path}")
        turns_df = None

    print("\n=== DEBATE METADATA ===")
    if metadata_path.exists():
        meta_df = pd.read_parquet(metadata_path)
        print(f"Shape: {meta_df.shape}")
        print(f"Columns: {list(meta_df.columns)}")
        print(f"\nFirst 3 rows:")
        print(meta_df.head(3))
    else:
        print(f"Metadata file not found: {metadata_path}")
        meta_df = None

    return turns_df, meta_df

if __name__ == "__main__":
    turns_df, meta_df = examine_turns()