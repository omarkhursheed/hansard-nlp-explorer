"""
Analyze pilot study results and validate quality before full run.
Run this after downloading pilot_results.parquet from Modal.
"""

import pandas as pd
import json
from pathlib import Path

def analyze_pilot_results(results_path: str = "outputs/llm_classification/pilot_results.parquet"):
    """Comprehensive analysis of pilot results."""

    if not Path(results_path).exists():
        print(f"ERROR: Results file not found: {results_path}")
        print("\nDid you download the results from Modal?")
        print("Run: modal volume get suffrage-results pilot_results.parquet ./outputs/llm_classification/")
        return

    print("="*70)
    print("PILOT RESULTS ANALYSIS")
    print("="*70)

    df = pd.read_parquet(results_path)
    print(f"\nLoaded {len(df)} results")

    # 1. API Success Rate
    print("\n" + "="*70)
    print("1. API SUCCESS RATE")
    print("="*70)

    if 'api_success' in df.columns:
        success_rate = df['api_success'].mean() * 100
        print(f"Success rate: {success_rate:.1f}% ({df.api_success.sum()}/{len(df)})")

        if success_rate < 95:
            print("⚠️  WARNING: Success rate below 95%")
            print("Check errors below")
    else:
        print("No api_success column found")

    # 2. JSON Parsing
    print("\n" + "="*70)
    print("2. JSON PARSING")
    print("="*70)

    if 'error' in df.columns:
        parse_errors = (df['error'] == 'json_parse_failed').sum()
        print(f"JSON parse failures: {parse_errors}/{len(df)} ({100*parse_errors/len(df):.1f}%)")

        if parse_errors > 5:
            print("⚠️  WARNING: High JSON parse failure rate")
            print("\nSample failed outputs:")
            failed = df[df['error'] == 'json_parse_failed']
            for idx in failed.index[:3]:
                print(f"\nSpeech {idx}:")
                print(f"  Raw output: {df.loc[idx, 'raw_output'][:200]}...")
    else:
        print("No parse errors found (good!)")

    # 3. Stance Distribution
    print("\n" + "="*70)
    print("3. STANCE DISTRIBUTION")
    print("="*70)

    if 'stance' in df.columns:
        stance_counts = df['stance'].value_counts()
        print(stance_counts)
        print(f"\nPercentages:")
        print((stance_counts / len(df) * 100).round(1))

        # Sanity checks
        if (df.stance == 'irrelevant').sum() > 50:
            print("\n⚠️  WARNING: High 'irrelevant' rate (>50%)")
            print("Many speeches may not actually be about suffrage")

        if (df.stance.isin(['for', 'against', 'both'])).sum() < 30:
            print("\n⚠️  WARNING: Very few clear stances (<30%)")
            print("Model may be too conservative")
    else:
        print("No stance column found (all parse failures?)")

    # 4. Confidence Distribution
    print("\n" + "="*70)
    print("4. CONFIDENCE CALIBRATION")
    print("="*70)

    if 'confidence' in df.columns:
        valid_confidence = df[df['confidence'].notna()]
        if len(valid_confidence) > 0:
            print(f"Mean confidence: {valid_confidence.confidence.mean():.2f}")
            print(f"Median confidence: {valid_confidence.confidence.median():.2f}")
            print(f"Min confidence: {valid_confidence.confidence.min():.2f}")
            print(f"Max confidence: {valid_confidence.confidence.max():.2f}")

            # Distribution
            print(f"\nConfidence ranges:")
            print(f"  Low (0-0.35): {(valid_confidence.confidence <= 0.35).sum()}")
            print(f"  Medium (0.35-0.7): {((valid_confidence.confidence > 0.35) & (valid_confidence.confidence <= 0.7)).sum()}")
            print(f"  High (>0.7): {(valid_confidence.confidence > 0.7).sum()}")
        else:
            print("No valid confidence scores found")
    else:
        print("No confidence column found")

    # 5. Reason Buckets
    print("\n" + "="*70)
    print("5. REASON BUCKET USAGE")
    print("="*70)

    if 'reasons' in df.columns:
        all_buckets = []
        for reasons in df['reasons'].dropna():
            if isinstance(reasons, list):
                for reason in reasons:
                    if isinstance(reason, dict) and 'bucket_key' in reason:
                        all_buckets.append(reason['bucket_key'])

        if all_buckets:
            bucket_series = pd.Series(all_buckets)
            print(f"Total reasons extracted: {len(all_buckets)}")
            print(f"\nBucket distribution:")
            print(bucket_series.value_counts())

            # Check for 'other' usage
            other_count = (bucket_series == 'other').sum()
            if other_count > len(all_buckets) * 0.2:
                print(f"\n⚠️  WARNING: High 'other' bucket usage ({other_count}/{len(all_buckets)})")
                print("Taxonomy may need expansion")
        else:
            print("No reasons found in results")
    else:
        print("No reasons column found")

    # 6. Quote Quality
    print("\n" + "="*70)
    print("6. QUOTE QUALITY CHECK")
    print("="*70)

    if 'reasons' in df.columns:
        all_quotes = []
        quote_lengths = []

        for reasons in df['reasons'].dropna():
            if isinstance(reasons, list):
                for reason in reasons:
                    if isinstance(reason, dict) and 'quotes' in reason:
                        for quote in reason['quotes']:
                            if isinstance(quote, str):
                                all_quotes.append(quote)
                                quote_lengths.append(len(quote))

        if quote_lengths:
            print(f"Total quotes: {len(quote_lengths)}")
            print(f"Average length: {sum(quote_lengths)/len(quote_lengths):.1f} chars")
            print(f"Min length: {min(quote_lengths)}")
            print(f"Max length: {max(quote_lengths)}")

            # Check compliance with 40-120 char requirement
            in_range = sum(1 for l in quote_lengths if 40 <= l <= 120)
            print(f"\nQuotes in range (40-120 chars): {in_range}/{len(quote_lengths)} ({100*in_range/len(quote_lengths):.1f}%)")

            if in_range / len(quote_lengths) < 0.8:
                print("⚠️  WARNING: Many quotes outside required range")

            # Show sample quotes
            print("\nSample quotes (first 5):")
            for i, quote in enumerate(all_quotes[:5]):
                print(f"  {i+1}. [{len(quote)} chars] {quote}")
        else:
            print("No quotes found")
    else:
        print("No reasons column to extract quotes from")

    # 7. Token Usage and Cost
    print("\n" + "="*70)
    print("7. TOKEN USAGE & COST ESTIMATE")
    print("="*70)

    if 'tokens_used' in df.columns:
        total_tokens = df['tokens_used'].sum()
        avg_tokens = df['tokens_used'].mean()

        print(f"Total tokens (pilot): {total_tokens:,}")
        print(f"Average per speech: {avg_tokens:,.0f}")

        # Estimate full run cost
        full_dataset_size = 2808
        estimated_tokens = avg_tokens * full_dataset_size

        # GPT-4o-mini pricing (approximate)
        # Input: $0.15 / 1M tokens, Output: $0.60 / 1M tokens
        # Assume 80% input, 20% output
        input_cost = (estimated_tokens * 0.8) * 0.15 / 1_000_000
        output_cost = (estimated_tokens * 0.2) * 0.60 / 1_000_000
        total_cost = input_cost + output_cost

        print(f"\nEstimated full run (2,808 speeches):")
        print(f"  Total tokens: {estimated_tokens:,.0f}")
        print(f"  Estimated cost: ${total_cost:.2f}")
    else:
        print("No token usage data found")

    # 8. Temporal & Demographic Breakdown
    print("\n" + "="*70)
    print("8. TEMPORAL & DEMOGRAPHIC BREAKDOWN")
    print("="*70)

    if 'decade' in df.columns and 'stance' in df.columns:
        print("\nStance by decade:")
        decade_stance = pd.crosstab(df.decade, df.stance, margins=True)
        print(decade_stance)

    if 'gender' in df.columns and 'stance' in df.columns:
        print("\nStance by gender:")
        gender_stance = pd.crosstab(df.gender, df.stance, margins=True)
        print(gender_stance)

    if 'confidence_level' in df.columns and 'stance' in df.columns:
        print("\nStance by original confidence level:")
        conf_stance = pd.crosstab(df.confidence_level, df.stance, margins=True)
        print(conf_stance)

    # 9. Manual Review Sample
    print("\n" + "="*70)
    print("9. SAMPLE CLASSIFICATIONS FOR MANUAL REVIEW")
    print("="*70)

    # Get diverse sample: one from each stance type
    if 'stance' in df.columns:
        for stance in ['for', 'against', 'both', 'neutral', 'irrelevant']:
            stance_df = df[df.stance == stance]
            if len(stance_df) > 0:
                sample = stance_df.iloc[0]
                print(f"\n{'-'*70}")
                print(f"STANCE: {stance.upper()}")
                print(f"Speaker: {sample.get('speaker', 'Unknown')} ({sample.get('year', 'Unknown')})")
                print(f"Confidence: {sample.get('confidence', 'N/A')}")

                if 'reasons' in sample and isinstance(sample.reasons, list):
                    print(f"Reasons:")
                    for reason in sample.reasons:
                        if isinstance(reason, dict):
                            print(f"  - {reason.get('bucket_key')}: {reason.get('rationale', 'N/A')[:100]}")
                            if 'quotes' in reason:
                                for quote in reason['quotes']:
                                    print(f"    Quote: \"{quote}\"")

                print(f"\nSpeech text (first 300 chars):")
                if 'target_text' in sample:
                    print(f"{str(sample.target_text)[:300]}...")

    # 10. Decision Recommendation
    print("\n" + "="*70)
    print("10. RECOMMENDATION")
    print("="*70)

    issues = []

    if 'api_success' in df.columns and df.api_success.mean() < 0.95:
        issues.append("Low API success rate (<95%)")

    if 'error' in df.columns and (df['error'] == 'json_parse_failed').sum() > 5:
        issues.append("High JSON parse failure rate (>5%)")

    if 'stance' in df.columns and (df.stance == 'irrelevant').sum() > 50:
        issues.append("High irrelevant rate (>50% - may need to filter input data)")

    if issues:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n❌ RECOMMENDATION: Review issues before proceeding to full run")
        print("   Consider adjusting prompt or model parameters")
    else:
        print("✅ PILOT LOOKS GOOD!")
        print("   Ready to proceed with full dataset")
        print(f"\n   Next step: modal run modal_suffrage_classification.py")

    print("\n" + "="*70)


if __name__ == "__main__":
    analyze_pilot_results()
