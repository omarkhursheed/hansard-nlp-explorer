"""
Detailed review of v6 classification results.
"""
import pandas as pd
import json

results_path = "outputs/llm_classification/full_results_v6_context_3_expanded.parquet"

print(f"Loading results...")
df = pd.read_parquet(results_path)

print(f"\n{'='*70}")
print(f"1. TEMPORAL DISTRIBUTION")
print(f"{'='*70}")
if 'year' in df.columns:
    # Group by decade and stance
    decade_stance = df.groupby(['decade', 'stance']).size().unstack(fill_value=0)
    print("\nStance by decade:")
    print(decade_stance)

    print("\nSuffrage-relevant speeches by decade (excluding irrelevant):")
    relevant = df[df['stance'] != 'irrelevant']
    decade_counts = relevant.groupby('decade').size()
    print(decade_counts)

print(f"\n{'='*70}")
print(f"2. GENDER DISTRIBUTION")
print(f"{'='*70}")
if 'gender' in df.columns:
    # Remove irrelevant speeches for clearer analysis
    relevant = df[df['stance'] != 'irrelevant']
    gender_stance = relevant.groupby(['gender', 'stance']).size().unstack(fill_value=0)
    print("\nStance by gender (excluding irrelevant):")
    print(gender_stance)

    # Calculate percentages
    print("\nPercentages within each gender:")
    gender_pct = gender_stance.div(gender_stance.sum(axis=1), axis=0) * 100
    print(gender_pct.round(1))

print(f"\n{'='*70}")
print(f"3. EXAMINING FAILURES")
print(f"{'='*70}")
failures = df[~df['api_success']]
if len(failures) > 0:
    print(f"Total failures: {len(failures)}")
    for idx, row in failures.iterrows():
        print(f"\nSpeech {row['speech_id']}:")
        print(f"  Error: {row.get('error', 'N/A')}")
        print(f"  Error detail: {row.get('error_detail', 'N/A')[:100]}")
else:
    print("No failures!")

print(f"\n{'='*70}")
print(f"4. INTERESTING CASES - BOTH STANCE")
print(f"{'='*70}")
both_cases = df[df['stance'] == 'both'].sort_values('confidence', ascending=False)
print(f"Total 'both' cases: {len(both_cases)}")
print(f"\nTop 5 highest confidence 'both' cases:")
for idx, row in both_cases.head(5).iterrows():
    print(f"\nSpeech {row['speech_id']} (confidence={row['confidence']:.2f}):")
    print(f"  Year: {row.get('year', 'N/A')}, Speaker: {row.get('speaker', 'N/A')}")
    if 'top_quote' in row and isinstance(row['top_quote'], dict):
        quote = row['top_quote'].get('text', '')[:150]
        print(f"  Quote: '{quote}...'")
    if 'reasons' in row and isinstance(row['reasons'], list):
        print(f"  Number of reasons: {len(row['reasons'])}")
        for i, reason in enumerate(row['reasons'][:3]):  # Show first 3 reasons
            if isinstance(reason, dict):
                print(f"    Reason {i+1}: {reason.get('bucket_key', 'N/A')} ({reason.get('stance_label', 'N/A')})")
                print(f"      {reason.get('rationale', 'N/A')[:100]}")

print(f"\n{'='*70}")
print(f"5. NEUTRAL CASES (RARE)")
print(f"{'='*70}")
neutral_cases = df[df['stance'] == 'neutral']
print(f"Total neutral cases: {len(neutral_cases)}")
for idx, row in neutral_cases.iterrows():
    print(f"\nSpeech {row['speech_id']} (confidence={row['confidence']:.2f}):")
    print(f"  Year: {row.get('year', 'N/A')}, Speaker: {row.get('speaker', 'N/A')}")
    if 'top_quote' in row and isinstance(row['top_quote'], dict):
        quote = row['top_quote'].get('text', '')[:150]
        print(f"  Quote: '{quote}...'")

print(f"\n{'='*70}")
print(f"6. VALIDATING IRRELEVANT CLASSIFICATIONS")
print(f"{'='*70}")
# Check some low-confidence irrelevant cases
irrelevant = df[df['stance'] == 'irrelevant']
low_conf_irrelevant = irrelevant[irrelevant['confidence'] < 0.7].sort_values('confidence')
print(f"Low confidence irrelevant (<0.7): {len(low_conf_irrelevant)}")
if len(low_conf_irrelevant) > 0:
    print("\nExamining low-confidence irrelevant cases:")
    for idx, row in low_conf_irrelevant.head(3).iterrows():
        print(f"\nSpeech {row['speech_id']} (confidence={row['confidence']:.2f}):")
        print(f"  Year: {row.get('year', 'N/A')}")
        if 'top_quote' in row and isinstance(row['top_quote'], dict):
            quote = row['top_quote'].get('text', '')[:150]
            print(f"  Quote: '{quote}...'")

# Sample some high-confidence irrelevant
print(f"\nHigh confidence irrelevant (random sample of 3):")
high_conf_irrelevant = irrelevant[irrelevant['confidence'] >= 0.9].sample(min(3, len(irrelevant)))
for idx, row in high_conf_irrelevant.iterrows():
    print(f"\nSpeech {row['speech_id']} (confidence={row['confidence']:.2f}):")
    if 'top_quote' in row and isinstance(row['top_quote'], dict):
        quote = row['top_quote'].get('text', '')[:150]
        print(f"  Quote: '{quote}...'")

print(f"\n{'='*70}")
print(f"7. REASON BUCKET DISTRIBUTION")
print(f"{'='*70}")
# Analyze which reason buckets are most common
relevant_with_reasons = df[(df['stance'].isin(['for', 'against', 'both'])) & (df['reasons'].notna())]
bucket_counts = {}
for idx, row in relevant_with_reasons.iterrows():
    if isinstance(row['reasons'], list):
        for reason in row['reasons']:
            if isinstance(reason, dict):
                bucket = reason.get('bucket_key', 'unknown')
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

print("\nMost common reason buckets:")
sorted_buckets = sorted(bucket_counts.items(), key=lambda x: x[1], reverse=True)
for bucket, count in sorted_buckets[:10]:
    print(f"  {bucket}: {count}")

print(f"\n{'='*70}")
print(f"8. QUALITY METRICS")
print(f"{'='*70}")
# Check for speeches with reasons
has_reasons = df[df['reasons'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
print(f"Speeches with reasons: {len(has_reasons)} ({100*len(has_reasons)/len(df):.1f}%)")

# Check for speeches with top_quote
has_quote = df[df['top_quote'].apply(lambda x: isinstance(x, dict) and x.get('text', '') != '')]
print(f"Speeches with non-empty top_quote: {len(has_quote)} ({100*len(has_quote)/len(df):.1f}%)")

# Average number of reasons per classified speech
relevant = df[df['stance'].isin(['for', 'against', 'both'])]
avg_reasons = relevant[relevant['reasons'].apply(lambda x: isinstance(x, list))]['reasons'].apply(len).mean()
print(f"Average reasons per stance-classified speech: {avg_reasons:.2f}")

print(f"\n{'='*70}")
print(f"REVIEW COMPLETE")
print(f"{'='*70}")
