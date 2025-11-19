"""
Analyze reason buckets from classification results.
"""
import pandas as pd
import numpy as np

results_path = "outputs/llm_classification/full_results_v6_context_3_expanded.parquet"

print(f"Loading results...")
df = pd.read_parquet(results_path)

print(f"\n{'='*70}")
print(f"REASON BUCKET ANALYSIS")
print(f"{'='*70}")

# Analyze which reason buckets are most common
relevant = df[df['stance'].isin(['for', 'against', 'both'])]
print(f"Total suffrage-relevant speeches: {len(relevant)}")

# Count speeches with reasons
has_reasons = relevant[relevant['reasons'].apply(lambda x: isinstance(x, np.ndarray) and len(x) > 0)]
print(f"Speeches with reasons: {len(has_reasons)} ({100*len(has_reasons)/len(relevant):.1f}%)")

# Extract all reason buckets
bucket_counts_overall = {}
bucket_counts_by_stance = {'for': {}, 'against': {}, 'both': {}}

for idx, row in has_reasons.iterrows():
    stance = row['stance']
    if isinstance(row['reasons'], np.ndarray):
        for reason in row['reasons']:
            if isinstance(reason, dict):
                bucket = reason.get('bucket_key', 'unknown')
                stance_label = reason.get('stance_label', 'unknown')

                # Overall counts
                bucket_counts_overall[bucket] = bucket_counts_overall.get(bucket, 0) + 1

                # By stance
                if stance in bucket_counts_by_stance:
                    bucket_counts_by_stance[stance][bucket] = bucket_counts_by_stance[stance].get(bucket, 0) + 1

print(f"\n{'='*70}")
print(f"OVERALL REASON BUCKET DISTRIBUTION")
print(f"{'='*70}")
sorted_buckets = sorted(bucket_counts_overall.items(), key=lambda x: x[1], reverse=True)
for bucket, count in sorted_buckets:
    pct = 100 * count / sum(bucket_counts_overall.values())
    print(f"  {bucket:25s}: {count:4d} ({pct:5.1f}%)")

print(f"\n{'='*70}")
print(f"REASON BUCKETS BY STANCE")
print(f"{'='*70}")

for stance in ['for', 'against', 'both']:
    buckets = bucket_counts_by_stance[stance]
    if buckets:
        print(f"\n{stance.upper()} stance:")
        sorted_buckets = sorted(buckets.items(), key=lambda x: x[1], reverse=True)
        for bucket, count in sorted_buckets[:10]:  # Top 10
            pct = 100 * count / sum(buckets.values())
            print(f"  {bucket:25s}: {count:4d} ({pct:5.1f}%)")

# Average number of reasons
print(f"\n{'='*70}")
print(f"REASONS PER SPEECH STATISTICS")
print(f"{'='*70}")
reason_counts = has_reasons['reasons'].apply(lambda x: len(x) if isinstance(x, np.ndarray) else 0)
print(f"Average reasons per speech: {reason_counts.mean():.2f}")
print(f"Median reasons per speech: {reason_counts.median():.0f}")
print(f"Max reasons in a speech: {reason_counts.max():.0f}")

print(f"\nReason count distribution:")
print(reason_counts.value_counts().sort_index())

# Check for "other" bucket usage
print(f"\n{'='*70}")
print(f"CUSTOM REASON BUCKETS (bucket_key='other')")
print(f"{'='*70}")
other_buckets = []
for idx, row in has_reasons.iterrows():
    if isinstance(row['reasons'], np.ndarray):
        for reason in row['reasons']:
            if isinstance(reason, dict) and reason.get('bucket_key') == 'other':
                other_buckets.append({
                    'speech_id': row['speech_id'],
                    'custom_label': reason.get('bucket_open', ''),
                    'rationale': reason.get('rationale', '')[:100]
                })

print(f"Total 'other' buckets: {len(other_buckets)}")
if len(other_buckets) > 0:
    print(f"\nExamples:")
    for item in other_buckets[:5]:
        print(f"  Custom: '{item['custom_label']}'")
        print(f"    Rationale: {item['rationale']}")
