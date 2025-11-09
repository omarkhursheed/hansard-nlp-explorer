#!/usr/bin/env python3
"""
Generate stratified validation sample for quality assessment.

Recommended: 50-100 speeches across all stance/confidence combinations.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_stratified_sample(n_total=50, random_seed=42):
    """
    Generate stratified validation sample.

    Args:
        n_total: Total number of speeches to sample (default 50)
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)

    # Load results
    df = pd.read_parquet('outputs/llm_classification/full_results_v5_context_3_complete.parquet')

    print("="*80)
    print(f"GENERATING STRATIFIED VALIDATION SAMPLE (n={n_total})")
    print("="*80)

    # Define strata
    strata = []

    # 1. High confidence FOR (0.7-0.8)
    high_for = df[(df['stance'] == 'for') & (df['confidence'] >= 0.7)]
    strata.append(('High conf FOR (0.7-0.8)', high_for, int(n_total * 0.20)))

    # 2. High confidence AGAINST (0.7-0.8)
    high_against = df[(df['stance'] == 'against') & (df['confidence'] >= 0.7)]
    strata.append(('High conf AGAINST (0.7-0.8)', high_against, int(n_total * 0.20)))

    # 3. Medium confidence (0.6)
    medium = df[df['confidence'] == 0.6]
    strata.append(('Medium conf (0.6)', medium, int(n_total * 0.20)))

    # 4. BOTH stance
    both = df[df['stance'] == 'both']
    strata.append(('BOTH stance', both, int(n_total * 0.10)))

    # 5. NEUTRAL
    neutral = df[df['stance'] == 'neutral']
    strata.append(('NEUTRAL', neutral, int(n_total * 0.10)))

    # 6. IRRELEVANT
    irrelevant = df[df['stance'] == 'irrelevant']
    strata.append(('IRRELEVANT', irrelevant, int(n_total * 0.10)))

    # 7. Female MPs
    female = df[df['gender'] == 'F']
    strata.append(('Female MPs', female, int(n_total * 0.10)))

    # Sample from each stratum
    samples = []
    total_sampled = 0

    print(f"\nSampling strategy:")
    print(f"{'Category':<30} {'Available':<12} {'Target':<10} {'Sampled':<10}")
    print("-"*80)

    for category, stratum_df, target_n in strata:
        available = len(stratum_df)
        n_sample = min(target_n, available)

        if n_sample > 0:
            sample = stratum_df.sample(n_sample, random_state=random_seed)
            samples.append(sample)
            total_sampled += n_sample

        print(f"{category:<30} {available:>10,}  {target_n:>8}  {n_sample:>8}")

    # Combine all samples
    validation_sample = pd.concat(samples, ignore_index=True)

    # Remove duplicates (e.g., female MP might be in multiple strata)
    validation_sample = validation_sample.drop_duplicates(subset=['speech_id'])

    print(f"\n{'='*80}")
    print(f"SAMPLE GENERATED")
    print(f"{'='*80}")
    print(f"Total sampled: {len(validation_sample):,} speeches")
    print(f"Unique speeches: {validation_sample['speech_id'].nunique():,}")

    # Breakdown
    print(f"\nBreakdown by stance:")
    for stance, count in validation_sample['stance'].value_counts().items():
        print(f"  {stance:>10}: {count:>3} ({count/len(validation_sample)*100:5.1f}%)")

    print(f"\nBreakdown by confidence:")
    for conf in sorted(validation_sample['confidence'].unique()):
        count = len(validation_sample[validation_sample['confidence'] == conf])
        print(f"  {conf:>4.1f}: {count:>3} ({count/len(validation_sample)*100:5.1f}%)")

    print(f"\nBreakdown by gender:")
    for gender in ['M', 'F']:
        count = len(validation_sample[validation_sample['gender'] == gender])
        if count > 0:
            label = 'Male' if gender == 'M' else 'Female'
            print(f"  {label:>6}: {count:>3} ({count/len(validation_sample)*100:5.1f}%)")

    # Save sample
    output_dir = Path('outputs/validation')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'validation_sample_n{len(validation_sample)}.parquet'
    validation_sample.to_parquet(output_file, index=False)

    # Also save as CSV for easy viewing
    csv_file = output_dir / f'validation_sample_n{len(validation_sample)}.csv'
    validation_sample[['speech_id', 'speaker', 'date', 'stance', 'confidence', 'gender']].to_csv(
        csv_file, index=False
    )

    print(f"\n{'='*80}")
    print(f"SAVED")
    print(f"{'='*80}")
    print(f"Parquet: {output_file}")
    print(f"CSV: {csv_file}")

    print(f"\n{'='*80}")
    print(f"NEXT STEPS")
    print(f"{'='*80}")
    print(f"\n1. Review speeches using manual_validation.py:")
    print(f"   python3 manual_validation.py --input {output_file}")
    print(f"\n2. Or display without interaction:")
    print(f"   python3 show_validation_samples.py --input {output_file}")
    print(f"\n3. Record results in a validation spreadsheet:")
    print(f"   - Correct/Incorrect stance")
    print(f"   - Correct/Incorrect reasons")
    print(f"   - Notes on errors")
    print(f"\n4. Calculate accuracy by category")
    print(f"   - Overall accuracy")
    print(f"   - Accuracy by stance")
    print(f"   - Accuracy by confidence")
    print(f"   - False positive rate for IRRELEVANT")

    return validation_sample


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate stratified validation sample')
    parser.add_argument('--n', type=int, default=50,
                        help='Total number of speeches to sample (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    sample = generate_stratified_sample(n_total=args.n, random_seed=args.seed)
