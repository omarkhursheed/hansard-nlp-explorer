"""
Analyze the v6 classification results.
"""
import pandas as pd

results_path = "outputs/llm_classification/full_results_v6_context_3_expanded.parquet"

print(f"Loading results from {results_path}...")
df = pd.read_parquet(results_path)

print(f"\n{'='*60}")
print(f"CLASSIFICATION RESULTS SUMMARY")
print(f"{'='*60}")
print(f"Total speeches: {len(df)}")
print(f"Successful: {df['api_success'].sum()}")
print(f"Failed: {(~df['api_success']).sum()}")

print(f"\n{'='*60}")
print(f"STANCE DISTRIBUTION")
print(f"{'='*60}")
stance_counts = df['stance'].value_counts()
print(stance_counts)
print(f"\nPercentages:")
for stance, count in stance_counts.items():
    print(f"  {stance}: {count} ({100*count/len(df):.1f}%)")

print(f"\n{'='*60}")
print(f"CONFIDENCE DISTRIBUTION")
print(f"{'='*60}")
print(df['confidence'].describe())

# Confidence by stance
print(f"\nMean confidence by stance:")
for stance in ['for', 'against', 'both', 'neutral', 'irrelevant']:
    if stance in df['stance'].values:
        mean_conf = df[df['stance'] == stance]['confidence'].mean()
        print(f"  {stance}: {mean_conf:.3f}")

print(f"\n{'='*60}")
print(f"IRRELEVANT SPEECHES - CONFIDENCE CHECK")
print(f"{'='*60}")
irrelevant = df[df['stance'] == 'irrelevant']
print(f"Total irrelevant: {len(irrelevant)}")
if len(irrelevant) > 0:
    print(f"Confidence distribution for irrelevant:")
    print(f"  Mean: {irrelevant['confidence'].mean():.3f}")
    print(f"  Median: {irrelevant['confidence'].median():.3f}")
    print(f"  Min: {irrelevant['confidence'].min():.3f}")
    print(f"  Max: {irrelevant['confidence'].max():.3f}")

    # Count high vs low confidence irrelevant
    high_conf = (irrelevant['confidence'] >= 0.7).sum()
    med_conf = ((irrelevant['confidence'] >= 0.4) & (irrelevant['confidence'] < 0.7)).sum()
    low_conf = (irrelevant['confidence'] < 0.4).sum()
    print(f"\n  High confidence (>=0.7): {high_conf} ({100*high_conf/len(irrelevant):.1f}%)")
    print(f"  Medium confidence (0.4-0.7): {med_conf} ({100*med_conf/len(irrelevant):.1f}%)")
    print(f"  Low confidence (<0.4): {low_conf} ({100*low_conf/len(irrelevant):.1f}%)")

print(f"\n{'='*60}")
print(f"CONTEXT HELPFULNESS")
print(f"{'='*60}")
context_helpful = df['context_helpful'].value_counts()
print(context_helpful)

print(f"\n{'='*60}")
print(f"ERRORS")
print(f"{'='*60}")
errors = df[~df['api_success']]
if len(errors) > 0:
    print(f"Total errors: {len(errors)}")
    if 'error' in df.columns:
        print(f"\nError types:")
        print(errors['error'].value_counts())
else:
    print("No errors!")

print(f"\n{'='*60}")
print(f"SAMPLE CLASSIFICATIONS")
print(f"{'='*60}")

# Show a few examples of each stance
for stance in ['for', 'against', 'both', 'neutral', 'irrelevant']:
    sample = df[df['stance'] == stance].head(2)
    if len(sample) > 0:
        print(f"\n{stance.upper()} examples:")
        for idx, row in sample.iterrows():
            print(f"  Speech {row['speech_id']}: confidence={row['confidence']:.2f}, context_helpful={row.get('context_helpful', 'N/A')}")
            if 'top_quote' in row and isinstance(row['top_quote'], dict):
                quote_text = row['top_quote'].get('text', '')[:80]
                print(f"    Top quote: '{quote_text}...'")
