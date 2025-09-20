#!/usr/bin/env python3
"""
Create sampled datasets from the full Hansard corpus.
Generates both JSONL and parquet files for samples of 500, 5000, 10000, and 100000 debates.
"""

import pandas as pd
import json
import random
import os
from pathlib import Path
import numpy as np
from typing import List, Dict, Any


def load_master_parquet() -> pd.DataFrame:
    """Load the master debates parquet file."""
    parquet_path = Path("data/processed_fixed/metadata/debates_master.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Master parquet file not found: {parquet_path}")
    
    print(f"Loading master parquet from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} debates from master parquet")
    return df


def load_jsonl_files(years: List[int]) -> List[Dict[Any, Any]]:
    """Load JSONL files for specified years."""
    all_debates = []
    
    for year in sorted(set(years)):
        jsonl_path = Path(f"data/processed_fixed/content/{year}/debates_{year}.jsonl")
        if jsonl_path.exists():
            print(f"Loading JSONL for year {year}")
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                year_debates = [json.loads(line) for line in f]
                all_debates.extend(year_debates)
                print(f"  Loaded {len(year_debates):,} debates from {year}")
        else:
            print(f"  Warning: JSONL file not found for year {year}: {jsonl_path}")
    
    print(f"Total debates loaded from JSONL files: {len(all_debates):,}")
    return all_debates


def create_sample_directory(sample_size: int) -> Path:
    """Create directory for sample files."""
    sample_dir = Path(f"data/sampled_datasets/sample_{sample_size}")
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


def save_sample_parquet(sampled_df: pd.DataFrame, sample_size: int, sample_dir: Path):
    """Save sampled parquet file."""
    parquet_path = sample_dir / f"debates_sample_{sample_size}.parquet"
    sampled_df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(sampled_df):,} debates to parquet: {parquet_path}")


def save_sample_jsonl(sampled_debates: List[Dict[Any, Any]], sample_size: int, sample_dir: Path):
    """Save sampled JSONL file."""
    jsonl_path = sample_dir / f"debates_sample_{sample_size}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for debate in sampled_debates:
            f.write(json.dumps(debate) + '\n')
    print(f"Saved {len(sampled_debates):,} debates to JSONL: {jsonl_path}")


def create_sample_metadata(sample_size: int, sampled_df: pd.DataFrame, sample_dir: Path):
    """Create metadata file for the sample."""
    metadata = {
        "sample_size": sample_size,
        "total_debates": len(sampled_df),
        "year_range": f"{sampled_df['year'].min()}-{sampled_df['year'].max()}",
        "chambers": sampled_df['chamber'].value_counts().to_dict(),
        "years_covered": len(sampled_df['year'].unique()),
        "total_words": int(sampled_df['word_count'].sum()) if 'word_count' in sampled_df.columns else None,
        "avg_words_per_debate": float(sampled_df['word_count'].mean()) if 'word_count' in sampled_df.columns else None
    }
    
    metadata_path = sample_dir / f"sample_{sample_size}_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")


def create_sampled_dataset(sample_size: int):
    """Create a sampled dataset of specified size."""
    print(f"\n=== Creating sample dataset of {sample_size:,} debates ===")
    
    # Load master parquet
    df = load_master_parquet()
    
    # Sample the parquet data
    if sample_size >= len(df):
        print(f"Warning: Sample size {sample_size:,} >= total debates {len(df):,}, using all debates")
        sampled_df = df.copy()
    else:
        sampled_df = df.sample(n=sample_size, random_state=42).sort_values(['year', 'file_name'])
    
    # Get unique years from sample
    sample_years = sampled_df['year'].unique()
    print(f"Sample covers {len(sample_years)} years: {min(sample_years)}-{max(sample_years)}")
    
    # Load corresponding JSONL data
    all_debates = load_jsonl_files(sample_years)
    
    # Create lookup for JSONL data by content_hash
    jsonl_lookup = {debate['content_hash']: debate for debate in all_debates}
    
    # Match sampled debates with JSONL data
    sampled_debates = []
    missing_hashes = []
    
    for _, row in sampled_df.iterrows():
        content_hash = row['content_hash']
        if content_hash in jsonl_lookup:
            sampled_debates.append(jsonl_lookup[content_hash])
        else:
            missing_hashes.append(content_hash)
    
    if missing_hashes:
        print(f"Warning: {len(missing_hashes)} debates not found in JSONL files")
    
    print(f"Successfully matched {len(sampled_debates):,} debates with JSONL content")
    
    # Create output directory
    sample_dir = create_sample_directory(sample_size)
    
    # Save files
    save_sample_parquet(sampled_df, sample_size, sample_dir)
    save_sample_jsonl(sampled_debates, sample_size, sample_dir)
    create_sample_metadata(sample_size, sampled_df, sample_dir)
    
    print(f"✓ Completed sample dataset of {sample_size:,} debates")


def main():
    """Create all sample datasets."""
    print("Creating sampled datasets from Hansard corpus")
    print("=" * 50)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Change to hansard directory
    os.chdir(Path(__file__).parent)
    
    sample_sizes = [500, 5000, 10000, 100000]
    
    for sample_size in sample_sizes:
        try:
            create_sampled_dataset(sample_size)
        except Exception as e:
            print(f"Error creating sample of {sample_size:,}: {e}")
            continue
    
    print("\n" + "=" * 50)
    print("Sample dataset creation completed!")
    print("\nCreated datasets:")
    for sample_size in sample_sizes:
        sample_dir = Path(f"data/sampled_datasets/sample_{sample_size}")
        if sample_dir.exists():
            print(f"  - {sample_size:,} debates: {sample_dir}")


if __name__ == "__main__":
    main()