"""
Check what the context_text actually contains.
"""
import pandas as pd

input_path = "outputs/llm_classification/full_input_context_3_expanded.parquet"

print(f"Loading input data...")
df = pd.read_parquet(input_path)

# Check first 10 speeches
print("\nFirst 10 speeches context_text:")
for idx, row in df.head(10).iterrows():
    print(f"  Speech {row['speech_id']}: len={len(row['context_text'])}, value='{row['context_text'][:100]}'")

# Check speeches around 4000
print("\nSpeeches 3995-4005 context_text:")
for idx in range(3995, min(4005, len(df))):
    row = df.iloc[idx]
    print(f"  Speech {row['speech_id']}: len={len(row['context_text'])}, value='{row['context_text'][:100]}'")

# Check the unique context_text values
unique_contexts = df['context_text'].unique()
print(f"\nTotal unique context_text values: {len(unique_contexts)}")
if len(unique_contexts) <= 10:
    print("All unique values:")
    for ctx in unique_contexts:
        print(f"  '{ctx}' (length={len(ctx)})")

# Count by length
print("\nContext_text length distribution:")
length_counts = df['context_text'].str.len().value_counts().sort_index()
print(length_counts)

# Check context_before and context_after
print("\nContext window settings:")
print(f"  context_before values: {df['context_before'].unique()}")
print(f"  context_after values: {df['context_after'].unique()}")
