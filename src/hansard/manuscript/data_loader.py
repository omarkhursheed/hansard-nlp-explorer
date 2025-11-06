#!/usr/bin/env python3
"""
Simple data loader for derived_complete dataset.
Replaces complex UnifiedDataLoader with direct parquet access.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


def get_data_dir() -> Path:
    """Get data-hansard directory path."""
    project_root = Path(__file__).resolve().parents[3]
    return project_root / 'data-hansard'


def load_speeches(
    year_range: Optional[Tuple[int, int]] = None,
    gender: Optional[str] = None,
    chamber: Optional[str] = None,
    sample: Optional[int] = None
) -> pd.DataFrame:
    """
    Load speeches from derived_complete/speeches_complete.

    Args:
        year_range: (start_year, end_year) tuple, e.g., (1990, 2005)
        gender: Filter by gender ('m', 'f', or 'male', 'female')
        chamber: Filter by chamber ('Commons', 'Lords')
        sample: Random sample size (for testing)

    Returns:
        DataFrame with columns: speech_id, speaker, gender, text, word_count,
        year, chamber, party, constituency, etc.
    """
    data_dir = get_data_dir() / 'derived_complete' / 'speeches_complete'

    # Determine years to load
    if year_range:
        years = range(year_range[0], year_range[1] + 1)
    else:
        # Load all available years
        year_files = sorted(data_dir.glob('speeches_*.parquet'))
        years = [int(f.stem.split('_')[1]) for f in year_files]

    # Load parquet files
    dfs = []
    for year in years:
        file_path = data_dir / f'speeches_{year}.parquet'
        if file_path.exists():
            df = pd.read_parquet(file_path)
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    # Combine all years
    speeches = pd.concat(dfs, ignore_index=True)

    # Apply filters
    if gender:
        # Handle both 'm'/'f' and 'male'/'female', case-insensitive
        gender_code = gender[0].upper()  # Data uses uppercase M/F
        speeches = speeches[speeches['gender'] == gender_code]

    if chamber:
        speeches = speeches[speeches['chamber'] == chamber]

    # Apply sampling if requested
    if sample and len(speeches) > sample:
        speeches = speeches.sample(n=sample, random_state=42)

    return speeches


def load_debates(
    year_range: Optional[Tuple[int, int]] = None,
    chamber: Optional[str] = None
) -> pd.DataFrame:
    """
    Load debate metadata from derived_complete/debates_complete.

    Args:
        year_range: (start_year, end_year) tuple
        chamber: Filter by chamber ('Commons', 'Lords')

    Returns:
        DataFrame with debate-level information
    """
    data_dir = get_data_dir() / 'derived_complete' / 'debates_complete'

    # Determine years to load
    if year_range:
        years = range(year_range[0], year_range[1] + 1)
    else:
        year_files = sorted(data_dir.glob('debates_*.parquet'))
        years = [int(f.stem.split('_')[1]) for f in year_files]

    # Load parquet files
    dfs = []
    for year in years:
        file_path = data_dir / f'debates_{year}.parquet'
        if file_path.exists():
            df = pd.read_parquet(file_path)
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    debates = pd.concat(dfs, ignore_index=True)

    # Apply filters
    if chamber:
        debates = debates[debates['chamber'] == chamber]

    return debates


if __name__ == '__main__':
    # Quick test
    print("Testing data_loader...")

    # Test speech loading
    print("\n1. Loading speeches from 2000:")
    speeches = load_speeches(year_range=(2000, 2000))
    print(f"   Loaded {len(speeches):,} speeches")
    print(f"   Columns: {speeches.columns.tolist()}")

    # Test gender filter
    print("\n2. Loading female speeches from 1990-2000:")
    female_speeches = load_speeches(year_range=(1990, 2000), gender='f')
    print(f"   Loaded {len(female_speeches):,} female speeches")
    print(f"   Total words: {female_speeches['word_count'].sum():,}")

    # Test chamber filter
    print("\n3. Loading Commons speeches from 2000:")
    commons = load_speeches(year_range=(2000, 2000), chamber='Commons')
    print(f"   Loaded {len(commons):,} Commons speeches")

    print("\nData loader test complete!")
