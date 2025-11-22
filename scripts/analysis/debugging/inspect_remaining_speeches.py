"""
Inspect the speeches that haven't been processed yet (4001-6531).
"""
import pandas as pd

input_path = "outputs/llm_classification/full_input_context_3_expanded.parquet"

print(f"Loading input data from {input_path}...")
df = pd.read_parquet(input_path)
print(f"Loaded {len(df)} total speeches")

# Get speeches 4001-6531 (0-indexed: 4000-6530)
remaining = df.iloc[4000:]
print(f"\nAnalyzing {len(remaining)} remaining speeches (4001-{len(df)})")

print("\n" + "="*60)
print("REMAINING SPEECHES CHARACTERISTICS")
print("="*60)

# Check for unusual characteristics
print(f"\nColumn data types:")
for col in remaining.columns:
    print(f"  {col}: {remaining[col].dtype}")

# Check for None/NaN values
print(f"\nNull value counts:")
for col in remaining.columns:
    null_count = remaining[col].isna().sum()
    if null_count > 0:
        print(f"  {col}: {null_count}")

# Check target_text and context_text lengths
print(f"\nText length statistics:")
print(f"  target_text length: min={remaining['target_text'].str.len().min()}, max={remaining['target_text'].str.len().max()}, mean={remaining['target_text'].str.len().mean():.1f}")
print(f"  context_text length: min={remaining['context_text'].str.len().min()}, max={remaining['context_text'].str.len().max()}, mean={remaining['context_text'].str.len().mean():.1f}")

# Find any speeches with unusually long text (might cause API issues)
long_target = remaining[remaining['target_text'].str.len() > 10000]
long_context = remaining[remaining['context_text'].str.len() > 50000]

if len(long_target) > 0:
    print(f"\n{len(long_target)} speeches with target_text > 10k chars:")
    for idx, row in long_target.head(5).iterrows():
        print(f"  Speech {row['speech_id']}: {len(row['target_text'])} chars")

if len(long_context) > 0:
    print(f"\n{len(long_context)} speeches with context_text > 50k chars:")
    for idx, row in long_context.head(5).iterrows():
        print(f"  Speech {row['speech_id']}: {len(row['context_text'])} chars")

# Check for special characters that might cause JSON encoding issues
print(f"\nChecking for problematic characters...")
problematic = []
for idx, row in remaining.iterrows():
    target = row['target_text']
    context = row['context_text']

    # Check for null bytes or other problematic chars
    if '\x00' in target or '\x00' in context:
        problematic.append((row['speech_id'], 'null byte'))
    elif '\ufffd' in target or '\ufffd' in context:
        problematic.append((row['speech_id'], 'replacement character'))

if problematic:
    print(f"Found {len(problematic)} speeches with problematic characters:")
    for speech_id, issue in problematic[:10]:
        print(f"  {speech_id}: {issue}")
else:
    print("No obvious problematic characters found")

# Save a subset for manual inspection
print(f"\nSaving speeches 4001-4200 for closer inspection...")
subset = remaining.iloc[:200]
subset.to_parquet("outputs/llm_classification/speeches_4001_4200.parquet", index=False)
print(f"Saved to outputs/llm_classification/speeches_4001_4200.parquet")
