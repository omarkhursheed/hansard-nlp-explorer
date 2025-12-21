#!/usr/bin/env python3
"""
Statistically sophisticated sampling for manual validation of LLM classifications.

Uses stratified sampling with Neyman allocation to ensure:
1. Representative coverage across key dimensions (gender, stance, time, confidence)
2. Adequate power to detect error rate differences between groups
3. Oversampling of critical but rare subgroups (e.g., female anti-suffrage)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm


def calculate_sample_size(population_size, confidence_level=0.95, margin_error=0.05, expected_proportion=0.5):
    """
    Calculate sample size for estimating a proportion with desired precision.

    Uses finite population correction.
    """
    z = norm.ppf(1 - (1 - confidence_level) / 2)  # z-score for confidence level
    p = expected_proportion  # worst case = 0.5 for maximum variance

    # Sample size for infinite population
    n0 = (z**2 * p * (1 - p)) / (margin_error**2)

    # Finite population correction
    n = n0 / (1 + (n0 - 1) / population_size)

    return int(np.ceil(n))


def neyman_allocation(strata_sizes, strata_variances, total_sample_size):
    """
    Neyman optimal allocation: allocate samples proportional to stratum size × std dev.

    More samples to strata with higher variance and larger size.
    """
    # For binary outcome (correct/incorrect), variance = p(1-p)
    # Use conservative estimate p=0.5 (max variance) unless we have prior knowledge

    weights = np.array(strata_sizes) * np.sqrt(np.array(strata_variances))
    allocations = (weights / weights.sum()) * total_sample_size

    # Ensure minimum sample size per stratum
    allocations = np.maximum(allocations, 10)

    # Renormalize if we exceeded total due to minimums
    if allocations.sum() > total_sample_size:
        # Scale down larger strata proportionally
        excess = allocations.sum() - total_sample_size
        for i in range(len(allocations)):
            if allocations[i] > 10:
                reduction = min(allocations[i] - 10, excess)
                allocations[i] -= reduction
                excess -= reduction
                if excess <= 0:
                    break

    return allocations.astype(int)


def sample_speeches(total_n=150, random_state=42):
    """
    Sample speeches using stratified sampling with Neyman allocation.

    Stratification dimensions:
    1. Gender (F/M) - critical for research question
    2. Stance (for/against/both)
    3. Time period (pre-1950, 1950-1980, post-1980)
    4. Confidence level (high >0.8, medium 0.5-0.8, low <0.5)
    """

    print('='*80)
    print('STATISTICALLY SOPHISTICATED SAMPLING FOR VALIDATION')
    print('='*80)

    # Load data
    results_file = 'outputs/llm_classification/full_results_v5_context_3_expanded_corrected.parquet'
    input_file = 'outputs/llm_classification/full_input_context_3_expanded.parquet'

    print(f'\nLoading results from: {results_file}')
    df = pd.read_parquet(results_file)

    print(f'Loading input (speech texts) from: {input_file}')
    input_df = pd.read_parquet(input_file)

    # Merge
    df = df.merge(
        input_df[['speech_id', 'target_text', 'context_text']],
        on='speech_id',
        how='left'
    )

    # Filter to substantive stances
    df = df[df['stance'].isin(['for', 'against', 'both'])].copy()

    print(f'\nTotal population: {len(df):,} speeches')

    # Create stratification variables
    df['time_period'] = pd.cut(
        df['year'],
        bins=[0, 1950, 1980, 2010],
        labels=['pre-1950', '1950-1980', 'post-1980']
    )

    df['confidence_category'] = pd.cut(
        df['confidence'],
        bins=[0, 0.5, 0.8, 1.0],
        labels=['low', 'medium', 'high']
    )

    # Create composite strata
    df['stratum'] = (
        df['gender'].astype(str) + '_' +
        df['stance'].astype(str) + '_' +
        df['time_period'].astype(str) + '_' +
        df['confidence_category'].astype(str)
    )

    # Calculate stratum statistics
    strata = df.groupby('stratum').agg({
        'speech_id': 'count',
        'confidence': 'std'
    }).rename(columns={'speech_id': 'size', 'confidence': 'std_dev'})

    # Fill NaN std_dev (single-element strata) with 0.25 (reasonable default)
    strata['std_dev'] = strata['std_dev'].fillna(0.25)

    # Use variance = 0.25 (assuming p=0.5 for binary validation outcome)
    # But weight by confidence std dev as proxy for classification difficulty
    strata['variance'] = 0.25 * (1 + strata['std_dev'])

    print('\n' + '='*80)
    print('STRATUM ANALYSIS')
    print('='*80)

    print(f'\nTotal strata: {len(strata)}')
    print(f'\nTop 10 largest strata:')
    print(strata.nlargest(10, 'size')[['size', 'variance']])

    # Special focus on female speeches (critical for research)
    female_strata = strata[strata.index.str.startswith('F_')]
    print(f'\nFemale MP strata: {len(female_strata)} strata, {female_strata["size"].sum():,} speeches')

    # Calculate sample sizes using Neyman allocation
    strata_sizes = strata['size'].values
    strata_variances = strata['variance'].values

    allocations = neyman_allocation(strata_sizes, strata_variances, total_n)
    strata['sample_n'] = allocations

    # Oversample female MPs to ensure adequate power for gender comparisons
    # Target: at least 40% of sample should be female (oversampling relative to ~10% in population)
    female_sample = strata[strata.index.str.startswith('F_')]['sample_n'].sum()
    target_female = int(total_n * 0.4)

    if female_sample < target_female:
        # Add more female samples
        deficit = target_female - female_sample
        female_strata_indices = strata.index.str.startswith('F_')

        # Distribute proportionally among female strata
        female_weights = strata.loc[female_strata_indices, 'size'].values
        female_additions = (female_weights / female_weights.sum() * deficit).astype(int)

        strata.loc[female_strata_indices, 'sample_n'] += female_additions

        # Reduce male samples proportionally to maintain total
        male_strata_indices = strata.index.str.startswith('M_')
        male_reduction_per_stratum = deficit // male_strata_indices.sum()
        strata.loc[male_strata_indices, 'sample_n'] = np.maximum(
            strata.loc[male_strata_indices, 'sample_n'] - male_reduction_per_stratum,
            5  # minimum
        )

    actual_total = strata['sample_n'].sum()
    female_sample = strata[strata.index.str.startswith('F_')]['sample_n'].sum()

    print('\n' + '='*80)
    print('SAMPLE ALLOCATION')
    print('='*80)

    print(f'\nTotal sample size: {actual_total}')
    print(f'Female MP samples: {female_sample} ({female_sample/actual_total*100:.1f}%)')
    print(f'Male MP samples: {actual_total - female_sample} ({(actual_total-female_sample)/actual_total*100:.1f}%)')

    print(f'\nBy stance:')
    for stance in ['for', 'against', 'both']:
        stance_sample = strata[strata.index.str.contains(f'_{stance}_')]['sample_n'].sum()
        print(f'  {stance}: {stance_sample} ({stance_sample/actual_total*100:.1f}%)')

    print(f'\nBy time period:')
    for period in ['pre-1950', '1950-1980', 'post-1980']:
        period_sample = strata[strata.index.str.contains(f'_{period}_')]['sample_n'].sum()
        print(f'  {period}: {period_sample} ({period_sample/actual_total*100:.1f}%)')

    print(f'\nBy confidence:')
    for conf in ['low', 'medium', 'high']:
        conf_sample = strata[strata.index.str.contains(f'_{conf}')]['sample_n'].sum()
        print(f'  {conf}: {conf_sample} ({conf_sample/actual_total*100:.1f}%)')

    # Sample from each stratum
    sampled_dfs = []

    for stratum_name, stratum_data in strata.iterrows():
        sample_size = int(stratum_data['sample_n'])

        if sample_size == 0:
            continue

        stratum_speeches = df[df['stratum'] == stratum_name]

        if len(stratum_speeches) == 0:
            continue

        # Sample min(requested, available)
        n_to_sample = min(sample_size, len(stratum_speeches))
        sampled = stratum_speeches.sample(n=n_to_sample, random_state=random_state)
        sampled_dfs.append(sampled)

    sample_df = pd.concat(sampled_dfs, ignore_index=True)

    # Shuffle
    sample_df = sample_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Add validation columns
    sample_df['validation_index'] = range(len(sample_df))
    sample_df['stance_correct'] = None
    sample_df['reasons_quality'] = None
    sample_df['notes'] = ''
    sample_df['validated'] = False
    sample_df['validator'] = ''
    sample_df['validation_timestamp'] = None

    print('\n' + '='*80)
    print('SAMPLE STATISTICS')
    print('='*80)

    print(f'\nActual sample size: {len(sample_df)}')

    # Calculate expected precision
    margin_error = 1.96 * np.sqrt(0.25 / len(sample_df))
    print(f'\nExpected precision (95% CI):')
    print(f'  Margin of error for overall error rate: ±{margin_error*100:.1f}%')

    # For subgroups
    female_n = len(sample_df[sample_df['gender'] == 'F'])
    if female_n > 0:
        female_margin = 1.96 * np.sqrt(0.25 / female_n)
        print(f'  Margin of error for female MP error rate: ±{female_margin*100:.1f}%')

    male_n = len(sample_df[sample_df['gender'] == 'M'])
    if male_n > 0:
        male_margin = 1.96 * np.sqrt(0.25 / male_n)
        print(f'  Margin of error for male MP error rate: ±{male_margin*100:.1f}%')

    # Save
    output_dir = Path('outputs/validation')
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_df.to_parquet(output_dir / 'validation_sample.parquet', index=False)
    print(f'\nSaved sample to: {output_dir / "validation_sample.parquet"}')

    # Save sampling plan
    strata['actual_sample'] = strata.index.map(
        sample_df.groupby('stratum').size().to_dict()
    ).fillna(0).astype(int)

    strata.to_csv(output_dir / 'sampling_plan.csv')
    print(f'Saved sampling plan to: {output_dir / "sampling_plan.csv"}')

    return sample_df


if __name__ == '__main__':
    sample_speeches(total_n=150)
