"""
Show validation samples without interactive prompts.
"""

import pandas as pd
import numpy as np

def display_classification(row, idx=None):
    """Display a single classification for manual review."""
    print("\n" + "="*80)
    if idx is not None:
        print(f"SAMPLE {idx}")
    print("="*80)

    # Metadata
    print(f"\nSpeech ID: {row['speech_id']}")
    print(f"Speaker: {row.get('speaker', 'Unknown')}")
    print(f"Date: {row.get('date', 'Unknown')}")
    print(f"Gender: {row.get('gender', 'Unknown')}")
    print(f"Party: {row.get('party', 'Unknown')}")

    # Classification
    print("\n" + "-"*80)
    print("CLASSIFICATION")
    print("-"*80)
    print(f"Stance: {row['stance'].upper()}")
    print(f"Confidence: {row['confidence']:.2f}")
    if 'context_helpful' in row:
        print(f"Context helpful: {row.get('context_helpful', 'N/A')}")

    # Reasons
    print("\n" + "-"*80)
    print("REASONS")
    print("-"*80)

    reasons = row.get('reasons')
    if isinstance(reasons, (list, np.ndarray)) and len(reasons) > 0:
        for i, reason in enumerate(reasons, 1):
            if isinstance(reason, dict):
                bucket = reason.get('bucket_key', 'unknown')
                stance_label = reason.get('stance_label', '?')
                rationale = reason.get('rationale', 'N/A')

                print(f"\n{i}. [{bucket.upper()}] → {stance_label}")
                print(f"   {rationale}")

                quotes = reason.get('quotes', [])
                if isinstance(quotes, (list, np.ndarray)) and len(quotes) > 0:
                    print("   Evidence:")
                    for quote in quotes:
                        if isinstance(quote, dict):
                            source = quote.get('source', '?')
                            text = quote.get('text', '')
                            print(f"     • [{source}] \"{text}\"")
    else:
        print("(No reasons)")

    # Speech text
    print("\n" + "-"*80)
    print("SPEECH TEXT (first 1000 chars)")
    print("-"*80)

    try:
        input_df = pd.read_parquet('outputs/llm_classification/full_input_context_3.parquet')
        original = input_df[input_df['speech_id'] == row['speech_id']]
        if len(original) > 0:
            text = original.iloc[0]['target_text']
            print(f"\n{text[:1000]}...")
            if len(text) > 1000:
                print(f"\n[... {len(text)-1000} more characters]")
            print(f"\n(Total: {len(text)} chars, ~{len(text.split())} words)")
    except:
        print("\n(Could not load original text)")


def main():
    """Show stratified validation samples."""

    df = pd.read_parquet('outputs/llm_classification/full_results_v5_context_3_complete.parquet')

    print("="*80)
    print("CLASSIFICATION VALIDATION SAMPLES")
    print("="*80)
    print(f"\nTotal dataset: {len(df)} speeches")

    # Stratified sample
    samples = []

    # 1. High confidence FOR (2 samples)
    high_for = df[(df['stance'] == 'for') & (df['confidence'] >= 0.7)].sample(min(2, len(df[(df['stance'] == 'for') & (df['confidence'] >= 0.7)])), random_state=42)
    samples.append(('High confidence FOR', high_for))

    # 2. High confidence AGAINST (2 samples)
    high_against = df[(df['stance'] == 'against') & (df['confidence'] >= 0.7)].sample(min(2, len(df[(df['stance'] == 'against') & (df['confidence'] >= 0.7)])), random_state=42)
    samples.append(('High confidence AGAINST', high_against))

    # 3. BOTH stance (1 sample)
    both = df[df['stance'] == 'both'].sample(min(1, len(df[df['stance'] == 'both'])), random_state=42)
    samples.append(('BOTH (mixed stance)', both))

    # 4. IRRELEVANT (2 samples)
    irrelevant = df[df['stance'] == 'irrelevant'].sample(min(2, len(df[df['stance'] == 'irrelevant'])), random_state=42)
    samples.append(('IRRELEVANT', irrelevant))

    # 5. Low confidence (1 sample)
    low_conf = df[df['confidence'] < 0.4].sample(min(1, len(df[df['confidence'] < 0.4])), random_state=42)
    samples.append(('Low confidence', low_conf))

    # 6. Female speakers (1 sample)
    female = df[df['gender'] == 'F'].sample(min(1, len(df[df['gender'] == 'F'])), random_state=42)
    samples.append(('Female MP', female))

    # Display each category
    total_shown = 0
    for category, sample_df in samples:
        if len(sample_df) == 0:
            continue

        print(f"\n\n{'#'*80}")
        print(f"CATEGORY: {category.upper()}")
        print(f"{'#'*80}")

        for i, (_, row) in enumerate(sample_df.iterrows(), 1):
            total_shown += 1
            display_classification(row, idx=total_shown)

    print("\n\n" + "="*80)
    print(f"VALIDATION COMPLETE - {total_shown} samples shown")
    print("="*80)


if __name__ == "__main__":
    main()
