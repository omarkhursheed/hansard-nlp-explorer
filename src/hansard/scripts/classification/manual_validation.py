"""
Manual validation of classification results.
Shows speech text, classification, and supporting evidence for review.
"""

import pandas as pd
import numpy as np
import random

def display_classification(row, idx=None):
    """Display a single classification for manual review."""
    print("\n" + "="*80)
    if idx is not None:
        print(f"SPEECH {idx}")
    print("="*80)

    # Metadata
    print(f"\nSpeech ID: {row['speech_id']}")
    print(f"Speaker: {row.get('speaker', 'Unknown')}")
    print(f"Date: {row.get('date', 'Unknown')}")
    print(f"Year: {row.get('year', 'Unknown')}")
    print(f"Gender: {row.get('gender', 'Unknown')}")
    print(f"Party: {row.get('party', 'Unknown')}")
    print(f"Chamber: {row.get('chamber', 'Unknown')}")

    # Classification
    print("\n" + "-"*80)
    print("CLASSIFICATION")
    print("-"*80)
    print(f"Stance: {row['stance'].upper()}")
    print(f"Confidence: {row['confidence']:.2f}")
    if 'context_helpful' in row:
        print(f"Context helpful: {row['context_helpful']}")

    # Reasons
    print("\n" + "-"*80)
    print("REASONS EXTRACTED")
    print("-"*80)

    reasons = row.get('reasons')
    if isinstance(reasons, (list, np.ndarray)) and len(reasons) > 0:
        for i, reason in enumerate(reasons, 1):
            if isinstance(reason, dict):
                print(f"\n{i}. {reason.get('bucket_key', 'unknown').upper()}")
                if reason.get('bucket_open'):
                    print(f"   ({reason['bucket_open']})")
                print(f"   Stance: {reason.get('stance_label', '?')}")
                print(f"   Rationale: {reason.get('rationale', 'N/A')}")

                quotes = reason.get('quotes', [])
                if isinstance(quotes, (list, np.ndarray)):
                    print("   Quotes:")
                    for quote in quotes:
                        if isinstance(quote, dict):
                            source = quote.get('source', '?')
                            text = quote.get('text', '')
                            print(f"     [{source}] \"{text}\"")
                        elif isinstance(quote, str):
                            print(f"     \"{quote}\"")
    else:
        print("(No reasons extracted)")

    # Top quote
    if 'top_quote' in row and row['top_quote']:
        print("\n" + "-"*80)
        print("TOP QUOTE")
        print("-"*80)
        top_quote = row['top_quote']
        if isinstance(top_quote, dict):
            source = top_quote.get('source', '?')
            text = top_quote.get('text', '')
            print(f"[{source}] \"{text}\"")
        elif isinstance(top_quote, str):
            print(f"\"{top_quote}\"")

    print("\n" + "="*80)
    print("ORIGINAL SPEECH TEXT")
    print("="*80)

    # Try to get original text from input
    try:
        input_df = pd.read_parquet('outputs/llm_classification/full_input_context_3.parquet')
        original = input_df[input_df['speech_id'] == row['speech_id']]
        if len(original) > 0:
            text = original.iloc[0]['target_text']
            print(f"\n{text}")
            print(f"\n(Length: {len(text)} characters, ~{len(text.split())} words)")
        else:
            print("\n(Original text not found)")
    except:
        print("\n(Could not load original text)")

    print("\n" + "="*80)


def validate_sample(n_samples=10, strategy='stratified'):
    """
    Select and display samples for manual validation.

    Args:
        n_samples: Number of samples to review
        strategy: 'stratified' (mix of confidence/stance), 'random', 'low_conf', 'high_conf'
    """

    # Load results
    df = pd.read_parquet('outputs/llm_classification/full_results_v5_context_3_complete.parquet')

    print("="*80)
    print("MANUAL VALIDATION")
    print("="*80)
    print(f"\nTotal speeches: {len(df)}")
    print(f"Sampling strategy: {strategy}")
    print(f"Samples to review: {n_samples}")

    # Select sample based on strategy
    if strategy == 'stratified':
        # Stratify by confidence and stance
        samples = []

        # High confidence (0.7-1.0) - 2 samples
        high_conf = df[df['confidence'] >= 0.7].sample(min(2, len(df[df['confidence'] >= 0.7])), random_state=42)
        samples.append(high_conf)

        # Medium confidence (0.4-0.7) - 5 samples
        med_conf = df[(df['confidence'] >= 0.4) & (df['confidence'] < 0.7)].sample(min(5, len(df[(df['confidence'] >= 0.4) & (df['confidence'] < 0.7)])), random_state=42)
        samples.append(med_conf)

        # Low confidence (0-0.4) - 1 sample
        low_conf = df[df['confidence'] < 0.4].sample(min(1, len(df[df['confidence'] < 0.4])), random_state=42)
        samples.append(low_conf)

        # Different stances - 2 samples (for, against)
        for_stance = df[df['stance'] == 'for'].sample(min(1, len(df[df['stance'] == 'for'])), random_state=43)
        samples.append(for_stance)

        against_stance = df[df['stance'] == 'against'].sample(min(1, len(df[df['stance'] == 'against'])), random_state=43)
        samples.append(against_stance)

        sample_df = pd.concat(samples).head(n_samples)

    elif strategy == 'random':
        sample_df = df.sample(n_samples, random_state=42)

    elif strategy == 'low_conf':
        # Focus on lowest confidence
        sample_df = df.nsmallest(n_samples, 'confidence')

    elif strategy == 'high_conf':
        # Focus on highest confidence
        sample_df = df.nlargest(n_samples, 'confidence')

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"\nSelected {len(sample_df)} samples:")
    print(f"  Confidence range: {sample_df['confidence'].min():.2f} - {sample_df['confidence'].max():.2f}")
    print(f"  Stance distribution: {sample_df['stance'].value_counts().to_dict()}")

    # Display each sample
    for i, (_, row) in enumerate(sample_df.iterrows(), 1):
        display_classification(row, idx=i)

        # Prompt for continue
        if i < len(sample_df):
            response = input("\n\nPress ENTER for next sample, 'q' to quit, or enter notes: ")
            if response.lower() == 'q':
                break
            elif response.strip():
                print(f"\nNOTE RECORDED: {response}")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nReviewed {min(i, len(sample_df))} samples")
    print("\nTo review more:")
    print("  python3 manual_validation.py")
    print("\nOr from Python:")
    print("  from manual_validation import validate_sample")
    print("  validate_sample(n_samples=20, strategy='stratified')")


if __name__ == "__main__":
    import sys

    # Parse arguments
    n = 10
    strat = 'stratified'

    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    if len(sys.argv) > 2:
        strat = sys.argv[2]

    validate_sample(n_samples=n, strategy=strat)
