"""
Examine all speeches by women AGAINST women's suffrage.
"""
import pandas as pd
import numpy as np
import textwrap

results_path = "outputs/llm_classification/full_results_v6_context_3_expanded.parquet"
input_path = "outputs/llm_classification/full_input_context_3_expanded.parquet"

print(f"Loading data...")
results_df = pd.read_parquet(results_path)
input_df = pd.read_parquet(input_path)

# Merge to get full text
df = results_df.merge(input_df[['speech_id', 'target_text', 'context_text']], on='speech_id', how='left')

# Filter to women against suffrage
women_against = df[(df['gender'] == 'F') & (df['stance'] == 'against')].copy()

print(f"\n{'='*80}")
print(f"WOMEN AGAINST WOMEN'S SUFFRAGE: {len(women_against)} SPEECHES")
print(f"{'='*80}")

# Sort by year
women_against = women_against.sort_values('year')

for idx, (i, row) in enumerate(women_against.iterrows(), 1):
    print(f"\n{'='*80}")
    print(f"SPEECH {idx}/{len(women_against)}")
    print(f"{'='*80}")

    print(f"Speech ID: {row['speech_id']}")
    print(f"Speaker: {row['speaker']}")
    print(f"Canonical Name: {row.get('canonical_name', 'N/A')}")
    print(f"Date: {row.get('date', 'N/A')}")
    print(f"Year: {row.get('year', 'N/A')}")
    print(f"Decade: {row.get('decade', 'N/A')}")
    print(f"Chamber: {row.get('chamber', 'N/A')}")
    print(f"Party: {row.get('party', 'N/A')}")
    print(f"Confidence: {row['confidence']:.2f}")
    print(f"Context helpful: {row.get('context_helpful', 'N/A')}")
    print(f"Word count: {row.get('word_count', 'N/A')}")

    # Top quote
    if isinstance(row['top_quote'], dict):
        top_quote_text = row['top_quote'].get('text', '')
        top_quote_source = row['top_quote'].get('source', '')
        if top_quote_text:
            print(f"\nTop Quote ({top_quote_source}):")
            wrapped = textwrap.fill(f'"{top_quote_text}"', width=78, initial_indent='  ', subsequent_indent='  ')
            print(wrapped)

    # Reasons
    print(f"\nReasons extracted:")
    if isinstance(row['reasons'], np.ndarray) and len(row['reasons']) > 0:
        for j, reason in enumerate(row['reasons'], 1):
            if isinstance(reason, dict):
                bucket = reason.get('bucket_key', 'unknown')
                bucket_open = reason.get('bucket_open', '')
                stance_label = reason.get('stance_label', 'unknown')
                rationale = reason.get('rationale', '')

                print(f"\n  Reason {j}:")
                print(f"    Bucket: {bucket}" + (f" ({bucket_open})" if bucket_open else ""))
                print(f"    Stance: {stance_label}")
                print(f"    Rationale:")
                wrapped = textwrap.fill(rationale, width=74, initial_indent='      ', subsequent_indent='      ')
                print(wrapped)

                # Quotes for this reason
                quotes = reason.get('quotes', [])
                if isinstance(quotes, np.ndarray) and len(quotes) > 0:
                    print(f"    Quotes:")
                    for quote in quotes:
                        if isinstance(quote, dict):
                            quote_text = quote.get('text', '')
                            quote_source = quote.get('source', '')
                            if quote_text:
                                wrapped = textwrap.fill(f'"{quote_text}" [{quote_source}]', width=70, initial_indent='      - ', subsequent_indent='        ')
                                print(wrapped)
    else:
        print("  No reasons extracted")

    # Full speech text
    print(f"\n{'='*80}")
    print(f"FULL SPEECH TEXT:")
    print(f"{'='*80}")
    speech_text = row.get('target_text', 'N/A')
    # Wrap long text
    wrapped = textwrap.fill(speech_text, width=78)
    print(wrapped)

    # Context if helpful
    if row.get('context_helpful') and row.get('context_text') and row['context_text'] != '[No context available]':
        print(f"\n{'='*80}")
        print(f"CONTEXT (marked as helpful):")
        print(f"{'='*80}")
        context_text = row.get('context_text', '')
        wrapped = textwrap.fill(context_text[:1000], width=78)  # First 1000 chars
        print(wrapped)
        if len(context_text) > 1000:
            print(f"\n  ... (context truncated, {len(context_text)} chars total)")

    print(f"\n")
    input("Press Enter to continue to next speech...")

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"Total speeches by women against suffrage: {len(women_against)}")
print(f"\nBy decade:")
print(women_against.groupby('decade').size())
print(f"\nBy speaker:")
print(women_against.groupby('canonical_name').size().sort_values(ascending=False))
