"""
Quick script to inspect checkpoint data for PyArrow issues.
"""
import pandas as pd
import sys

checkpoint_path = "outputs/llm_classification/full_results_v6_context_3_expanded.parquet.checkpoint"

print(f"Loading checkpoint from {checkpoint_path}...")
df = pd.read_parquet(checkpoint_path)

print(f"\nCheckpoint has {len(df)} speeches")
print(f"\nColumns: {list(df.columns)}")

# Inspect top_quote column
print("\n" + "="*60)
print("TOP_QUOTE COLUMN ANALYSIS")
print("="*60)

# Check for None values
none_count = df['top_quote'].isna().sum()
print(f"None/NaN values: {none_count}")

# Check data types
print(f"\nUnique types in top_quote column:")
types_seen = df['top_quote'].apply(type).value_counts()
print(types_seen)

# Sample some values
print(f"\nFirst 10 top_quote values:")
for idx, val in enumerate(df['top_quote'].head(10)):
    print(f"{idx}: {type(val).__name__:15s} | {val}")

# Check for problematic values
print(f"\nSearching for non-dict, non-None values...")
problematic = []
for idx, val in enumerate(df['top_quote']):
    if val is not None and not isinstance(val, dict):
        problematic.append((idx, type(val).__name__, val))
        if len(problematic) >= 5:  # Show first 5
            break

if problematic:
    print(f"Found {len(problematic)} problematic values:")
    for idx, typ, val in problematic:
        print(f"  Row {idx}: {typ} | {val}")
else:
    print("No problematic values found!")

# Check reasons column too
print("\n" + "="*60)
print("REASONS COLUMN ANALYSIS")
print("="*60)

none_count_reasons = df['reasons'].isna().sum()
print(f"None/NaN values: {none_count_reasons}")

print(f"\nUnique types in reasons column:")
types_seen_reasons = df['reasons'].apply(type).value_counts()
print(types_seen_reasons)

# Sample some values
print(f"\nFirst 10 reasons values:")
for idx, val in enumerate(df['reasons'].head(10)):
    print(f"{idx}: {type(val).__name__:15s} | {val if isinstance(val, list) else '...'}")
