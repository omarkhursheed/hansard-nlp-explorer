#!/usr/bin/env python3
"""
Examine the house_members_gendered_updated.parquet - our source of truth for MP gender data
"""

import pandas as pd
from pathlib import Path

def examine_truth_data():
    """Examine the structure and contents of our truth gender data"""

    data_path = Path("src/hansard/data/house_members_gendered_updated.parquet")

    if not data_path.exists():
        print(f"File not found: {data_path}")
        return None

    print("Loading house_members_gendered_updated.parquet (source of truth)...")
    df = pd.read_parquet(data_path)

    print(f"\n=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print(f"\n=== FIRST 5 ROWS ===")
    print(df.head())

    print(f"\n=== DATA TYPES ===")
    print(df.dtypes)

    # Gender column analysis
    if 'gender' in df.columns:
        print(f"\n=== GENDER DISTRIBUTION ===")
        print(df['gender'].value_counts())
        print(f"Gender coverage: {df['gender'].notna().sum()}/{len(df)} ({100*df['gender'].notna().sum()/len(df):.1f}%)")

    # Name columns
    name_cols = [c for c in df.columns if 'name' in c.lower()]
    print(f"\n=== NAME COLUMNS ===")
    for col in name_cols:
        try:
            print(f"{col}: {df[col].nunique()} unique values")
            print(f"  Sample: {df[col].dropna().head(3).tolist()}")
        except TypeError:
            print(f"{col}: contains unhashable types (arrays/lists)")
            print(f"  Sample: {df[col].dropna().head(1).tolist()}")

    # Date columns
    date_cols = [c for c in df.columns if any(x in c.lower() for x in ['date', 'year', 'from', 'to'])]
    print(f"\n=== DATE/TIME COLUMNS ===")
    for col in date_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"{col}: {df[col].min()} to {df[col].max()}")
        else:
            print(f"{col}: {df[col].nunique()} unique values")

    print(f"\n=== MISSING VALUES ===")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values")

    # Check for unique identifier
    print(f"\n=== CHECKING FOR UNIQUE IDENTIFIERS ===")
    for col in df.columns:
        if df[col].dtype not in ['object'] or 'other_names' in col or 'aliases' in col:
            continue
        try:
            if df[col].nunique() == len(df):
                print(f"  {col} is a unique identifier")
        except TypeError:
            pass

    return df

if __name__ == "__main__":
    df = examine_truth_data()
    if df is not None:
        print(f"\nTotal MPs with gender data: {len(df)}")