"""
Find the exact speech that causes PyArrow serialization errors.
"""
import pandas as pd
import json

# Load the checkpoint to get examples of successful data
checkpoint_path = "outputs/llm_classification/full_results_v6_context_3_expanded.parquet.checkpoint"
print("Loading checkpoint...")
checkpoint_df = pd.read_parquet(checkpoint_path)
print(f"Loaded {len(checkpoint_df)} speeches from checkpoint")

# Inspect all speeches for type consistency
print("\n" + "="*60)
print("CHECKING ALL SPEECHES FOR TYPE INCONSISTENCIES")
print("="*60)

problematic_speeches = []

for idx, row in checkpoint_df.iterrows():
    speech_id = row['speech_id']
    top_quote = row['top_quote']
    reasons = row['reasons']

    # Check top_quote
    if top_quote is not None and not isinstance(top_quote, dict):
        problematic_speeches.append({
            'index': idx,
            'speech_id': speech_id,
            'field': 'top_quote',
            'type': type(top_quote).__name__,
            'value': top_quote
        })

    # Check reasons
    if reasons is not None and not isinstance(reasons, (list, type(checkpoint_df['reasons'].iloc[0]))):
        problematic_speeches.append({
            'index': idx,
            'speech_id': speech_id,
            'field': 'reasons',
            'type': type(reasons).__name__,
            'value': reasons
        })

if problematic_speeches:
    print(f"\nFound {len(problematic_speeches)} problematic speeches:")
    for prob in problematic_speeches[:10]:  # Show first 10
        print(f"  Speech {prob['speech_id']} (row {prob['index']}): {prob['field']} is {prob['type']}, value: {prob['value']}")
else:
    print("\nNo type inconsistencies found in checkpoint!")

# Now try to re-save the checkpoint to see if it fails
print("\n" + "="*60)
print("TESTING CHECKPOINT RE-SAVE")
print("="*60)

try:
    test_path = "outputs/llm_classification/test_resave.parquet"
    checkpoint_df.to_parquet(test_path, index=False)
    print(f"Successfully re-saved checkpoint to {test_path}")
except Exception as e:
    print(f"Failed to re-save checkpoint: {e}")

# Now test adding one more speech
print("\n" + "="*60)
print("TESTING INCREMENTAL ADDITION")
print("="*60)

# Create a mock new speech result with all required fields
new_speech = {
    'stance': 'for',
    'reasons': [],
    'top_quote': {'text': 'test quote', 'source': 'TARGET'},
    'confidence': 0.8,
    'context_helpful': True,
    'speech_id': 'test_speech_001',
    'debate_id': 'test_debate_001',
    'speaker': 'Test Speaker',
    'canonical_name': 'Test Name',
    'gender': 'M',
    'party': 'Test Party',
    'year': 2000,
    'decade': 2000,
    'date': '2000-01-01',
    'chamber': 'Commons',
    'confidence_level': 'high',
    'word_count': 100,
    'model': 'openai/gpt-4o-mini',
    'prompt_version': 'v6',
    'tokens_used': 500,
    'api_success': True,
}

# Add to checkpoint
extended_df = pd.concat([checkpoint_df, pd.DataFrame([new_speech])], ignore_index=True)

try:
    test_path = "outputs/llm_classification/test_extended.parquet"
    extended_df.to_parquet(test_path, index=False)
    print(f"Successfully saved extended dataset to {test_path}")
except Exception as e:
    print(f"Failed to save extended dataset: {e}")
    print(f"\nError type: {type(e).__name__}")

    # Try to find which row is problematic
    print("\nTesting each row individually...")
    for i in range(len(extended_df)):
        try:
            test_df = extended_df.iloc[[i]]
            test_df.to_parquet(f"outputs/llm_classification/test_row_{i}.parquet", index=False)
        except Exception as row_e:
            print(f"  Row {i} (speech_id={extended_df.iloc[i]['speech_id']}) FAILED: {row_e}")
            print(f"    top_quote type: {type(extended_df.iloc[i]['top_quote'])}")
            print(f"    top_quote value: {extended_df.iloc[i]['top_quote']}")
            break

print("\n" + "="*60)
print("DONE")
print("="*60)
