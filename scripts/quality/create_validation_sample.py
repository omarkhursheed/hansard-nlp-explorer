#!/usr/bin/env python3
"""
Create stratified validation sample for LLM classification audit.

Based on AUDIT_PLAN.md, creates a representative sample stratified by:
- Stance (for/against/both/neutral/irrelevant)
- Gender (with oversampling of rare female categories)
- Time period (1809-1850, 1850-1900, 1900-1920, 1920-1935)
- Confidence (high/medium/low)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from hansard.utils.path_config import Paths


def create_stratified_sample(
    df: pd.DataFrame,
    sample_size: int = 200,
    oversample_categories: dict = None
) -> pd.DataFrame:
    """
    Create stratified sample with oversampling of specific categories.

    Args:
        df: Full classification results
        sample_size: Target sample size (default: 200)
        oversample_categories: Dict of {filter_condition: 'ALL' or count}

    Returns:
        Stratified sample DataFrame
    """
    samples = []

    # Handle oversample categories first (mandatory inclusions)
    if oversample_categories:
        for condition, count in oversample_categories.items():
            subset = df.query(condition)
            if len(subset) > 0:
                if count == 'ALL':
                    samples.append(subset)
                    print(f"Including ALL {len(subset)} speeches: {condition}")
                else:
                    n = min(count, len(subset))
                    samples.append(subset.sample(n=n, random_state=42))
                    print(f"Including {n} speeches: {condition}")

    # Calculate remaining sample size
    oversampled = pd.concat(samples) if samples else pd.DataFrame()
    remaining_size = sample_size - len(oversampled)

    if remaining_size <= 0:
        print(f"\nOversample categories filled entire sample ({len(oversampled)} speeches)")
        return oversampled

    print(f"\nRemaining sample size after oversampling: {remaining_size}")

    # Remove already-sampled speeches from pool
    remaining_df = df[~df.index.isin(oversampled.index)]

    # Stratify remaining by stance (proportional to distribution)
    stance_counts = remaining_df['stance'].value_counts()
    stance_proportions = stance_counts / stance_counts.sum()

    print("\nStance distribution in remaining pool:")
    for stance, prop in stance_proportions.items():
        target = int(remaining_size * prop)
        available = stance_counts[stance]
        print(f"  {stance}: {prop:.1%} → target {target} (available: {available})")

    # Sample proportionally from each stance
    stratified_samples = []
    for stance in stance_proportions.index:
        target_n = int(remaining_size * stance_proportions[stance])
        if target_n == 0:
            continue

        stance_subset = remaining_df[remaining_df['stance'] == stance]
        n = min(target_n, len(stance_subset))
        stratified_samples.append(stance_subset.sample(n=n, random_state=42))

    # Combine all samples
    all_samples = samples + stratified_samples
    final_sample = pd.concat(all_samples)

    return final_sample


def add_stratification_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns for stratification analysis."""
    df = df.copy()

    # Time period bins
    df['time_period'] = pd.cut(
        df['year'],
        bins=[1809, 1850, 1900, 1920, 1936],
        labels=['1809-1850', '1850-1900', '1900-1920', '1920-1935'],
        include_lowest=True
    )

    # Confidence bins
    df['confidence_bin'] = pd.cut(
        df['confidence'],
        bins=[-0.01, 0.39, 0.69, 1.01],
        labels=['low', 'medium', 'high']
    )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified validation sample for LLM classification audit"
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=PROJECT_ROOT / 'outputs/llm_classification/claude_sonnet_45_full_results.parquet',
        help='Input classification results parquet file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'outputs/validation/audit_sample_n200.parquet',
        help='Output validation sample parquet file'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=200,
        help='Target sample size (default: 200 for ±7%% margin at 95%% confidence)'
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'reduced'],
        default='full',
        help='full=200 speeches (±7%%), reduced=100 speeches (±10%%)'
    )

    args = parser.parse_args()

    # Adjust sample size for mode
    if args.mode == 'reduced':
        args.sample_size = 100

    print(f"Creating {args.mode} validation sample (n={args.sample_size})")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    # Load classification results
    df = pd.read_parquet(args.input)
    print(f"\nTotal speeches: {len(df):,}")

    # Add stratification columns
    df = add_stratification_columns(df)

    # Define oversample categories (mandatory inclusions)
    oversample_categories = {
        # Priority 1: ALL female AGAINST speeches
        "gender == 'F' and stance == 'against'": 'ALL',

        # Priority 1: ALL neutral speeches (very rare)
        "stance == 'neutral'": 'ALL',

        # Priority 2: Female in rare categories
        "gender == 'F' and stance == 'both'": 'ALL',

        # Priority 3: Low confidence FOR/AGAINST
        "stance in ['for', 'against'] and confidence < 0.4": 'ALL',
    }

    print("\n=== OVERSAMPLE CATEGORIES ===")
    # Create stratified sample
    sample = create_stratified_sample(
        df,
        sample_size=args.sample_size,
        oversample_categories=oversample_categories
    )

    # Print sample statistics
    print(f"\n=== FINAL SAMPLE STATISTICS (n={len(sample)}) ===")
    print(f"\nBy stance:")
    print(sample['stance'].value_counts().sort_index())

    print(f"\nBy gender:")
    print(sample['gender'].value_counts())

    print(f"\nBy time period:")
    print(sample['time_period'].value_counts().sort_index())

    print(f"\nBy confidence bin:")
    print(sample['confidence_bin'].value_counts().sort_index())

    print(f"\nFemale by stance:")
    female = sample[sample['gender'] == 'F']
    if len(female) > 0:
        print(female['stance'].value_counts().sort_index())
    else:
        print("(no female speeches in sample)")

    # Add sample metadata
    sample = sample.copy()
    sample['sample_id'] = range(1, len(sample) + 1)
    sample['validation_status'] = 'pending'

    # Save sample
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.output)
    print(f"\nSample saved to: {args.output}")

    # Also save as CSV for easy viewing
    csv_path = args.output.with_suffix('.csv')
    columns_to_export = [
        'sample_id', 'speech_id', 'speaker', 'date', 'year', 'gender',
        'stance', 'confidence', 'time_period', 'confidence_bin',
        'validation_status'
    ]
    sample[columns_to_export].to_csv(csv_path, index=False)
    print(f"Sample index saved to: {csv_path}")

    print("\n=== NEXT STEPS ===")
    print("1. Review sample with: scripts/classification/show_validation_samples.py")
    print("2. Validate with: streamlit run scripts/classification/validation_app.py")
    print("3. Analyze results with: scripts/classification/analyze_validation_results.py")


if __name__ == '__main__':
    main()
