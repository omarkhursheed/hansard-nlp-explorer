#!/usr/bin/env python3
"""
Gender Comparison Analysis - Expanded Dataset (1809-2004)

Replicates and extends the gender analysis from the notebook to the full 6,531 speeches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Professional visualization style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Arial']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Professional color palette
COLORS = {
    'male': '#3B82C4',
    'female': '#EC4899',
    'for': '#10B981',
    'against': '#EF4444',
    'both': '#F59E0B',
    'neutral': '#8B5CF6',
    'irrelevant': '#9CA3AF',
}

# Valid argument buckets
VALID_BUCKETS = {
    'equality', 'competence_capacity', 'emotion_morality',
    'social_order_stability', 'tradition_precedent',
    'instrumental_effects', 'religion_family',
    'social_experiment', 'other'
}

# Output directory
OUTPUT_DIR = Path('analysis/suffrage_gender_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_reasons(df):
    """Extract all reasons from classified speeches."""
    all_reasons = []

    for idx, row in df.iterrows():
        reasons = row.get('reasons')
        if reasons is None or not isinstance(reasons, (list, np.ndarray)):
            continue

        for reason in reasons:
            if isinstance(reason, dict):
                bucket = reason.get('bucket_key', 'unknown')

                # Filter to valid buckets only
                if bucket in VALID_BUCKETS:
                    all_reasons.append({
                        'speech_id': row['speech_id'],
                        'year': row['year'],
                        'gender': row['gender'],
                        'stance': row['stance'],
                        'bucket': bucket,
                        'stance_label': reason.get('stance_label', 'unknown'),
                        'rationale': reason.get('rationale', ''),
                    })

    return pd.DataFrame(all_reasons)


def main():
    print('='*80)
    print('GENDER COMPARISON ANALYSIS - EXPANDED DATASET')
    print('='*80)

    # Load data
    print('\nLoading classified results...')
    df = pd.read_parquet('outputs/llm_classification/full_results_v5_context_3_expanded.parquet')

    print(f'Total speeches: {len(df):,}')
    print(f'Year range: {df["year"].min():.0f} - {df["year"].max():.0f}')

    # Filter to substantive speeches
    substantive = df[df['stance'].isin(['for', 'against', 'both', 'neutral'])].copy()
    print(f'Substantive speeches: {len(substantive):,} ({len(substantive)/len(df)*100:.1f}%)')

    # Overall gender distribution
    print('\n' + '='*80)
    print('1. OVERALL GENDER DISTRIBUTION')
    print('='*80)

    gender_counts = substantive['gender'].value_counts()
    print(f'\nTotal substantive speeches: {len(substantive):,}')
    print(f'  Male MPs: {gender_counts.get("M", 0):,} ({gender_counts.get("M", 0)/len(substantive)*100:.1f}%)')
    print(f'  Female MPs: {gender_counts.get("F", 0):,} ({gender_counts.get("F", 0)/len(substantive)*100:.1f}%)')
    print(f'  Unknown: {len(substantive[substantive["gender"].isna()]):,}')

    # Temporal distribution of female participation
    print('\n' + '='*80)
    print('2. TEMPORAL DISTRIBUTION OF FEMALE PARTICIPATION')
    print('='*80)

    # Count speeches by year and gender
    temporal_gender = substantive.groupby(['year', 'gender']).size().unstack(fill_value=0)

    # Calculate percentage female each year
    temporal_gender['total'] = temporal_gender.sum(axis=1)
    temporal_gender['pct_female'] = (temporal_gender.get('F', 0) / temporal_gender['total'] * 100)

    print(f'\nYears with female MP speeches:')
    female_years = temporal_gender[temporal_gender.get('F', 0) > 0]
    print(f'  First appearance: {female_years.index.min():.0f}')
    print(f'  Number of years: {len(female_years)}')
    print(f'  Peak year: {female_years["F"].idxmax():.0f} ({female_years["F"].max():.0f} speeches)')

    # Visualize temporal participation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Absolute counts
    if 'M' in temporal_gender.columns:
        ax1.fill_between(temporal_gender.index, temporal_gender['M'],
                         alpha=0.7, color=COLORS['male'], label='Male MPs')
    if 'F' in temporal_gender.columns:
        ax1.fill_between(temporal_gender.index, temporal_gender['F'],
                         alpha=0.7, color=COLORS['female'], label='Female MPs')

    ax1.set_ylabel('Number of Speeches', fontsize=12, fontweight='bold')
    ax1.set_title('Temporal Distribution of Suffrage Debate Participation by Gender',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.3)

    # Percentage female
    ax2.plot(temporal_gender.index, temporal_gender['pct_female'],
            color=COLORS['female'], linewidth=2.5)
    ax2.fill_between(temporal_gender.index, temporal_gender['pct_female'],
                     alpha=0.3, color=COLORS['female'])

    # Add key historical markers
    ax2.axvline(1918, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.text(1918, ax2.get_ylim()[1]*0.9, '1918: Partial suffrage',
            rotation=90, va='top', ha='right', fontsize=9, color='gray')
    ax2.axvline(1928, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.text(1928, ax2.get_ylim()[1]*0.9, '1928: Equal suffrage',
            rotation=90, va='top', ha='right', fontsize=9, color='gray')

    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage Female (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Female Participation Rate Over Time',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'temporal_gender_participation.png', dpi=300, bbox_inches='tight')
    print(f'\nSaved: {OUTPUT_DIR / "temporal_gender_participation.png"}')

    # Stance by gender
    print('\n' + '='*80)
    print('3. STANCE DISTRIBUTION BY GENDER')
    print('='*80)

    gender_stance = substantive.groupby(['gender', 'stance']).size().unstack(fill_value=0)
    gender_stance_pct = gender_stance.div(gender_stance.sum(axis=1), axis=0) * 100

    print('\nStance distribution by gender (%):')
    print(gender_stance_pct)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(gender_stance_pct.index))
    width = 0.2

    offset = -1.5
    for stance in ['for', 'against', 'both', 'neutral']:
        if stance in gender_stance_pct.columns:
            ax.bar(x + offset*width, gender_stance_pct[stance], width,
                  label=stance.capitalize(), color=COLORS[stance])
            offset += 1

    ax.set_xlabel('Gender', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Speeches (%)', fontsize=12, fontweight='bold')
    ax.set_title('Suffrage Stance by MP Gender', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Female' if g == 'F' else 'Male' for g in gender_stance_pct.index])
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'stance_by_gender.png', dpi=300, bbox_inches='tight')
    print(f'\nSaved: {OUTPUT_DIR / "stance_by_gender.png"}')

    # Arguments by gender
    print('\n' + '='*80)
    print('4. ARGUMENT TYPES BY GENDER')
    print('='*80)

    reasons_df = extract_reasons(substantive)
    print(f'\nExtracted {len(reasons_df):,} reasons from {len(substantive):,} speeches')

    male_reasons = reasons_df[reasons_df['gender'] == 'M']['bucket'].value_counts()
    female_reasons = reasons_df[reasons_df['gender'] == 'F']['bucket'].value_counts()

    # Normalize by number of speeches
    male_count = gender_counts.get('M', 1)
    female_count = gender_counts.get('F', 1)

    male_reasons_norm = male_reasons / male_count
    female_reasons_norm = female_reasons / female_count

    print(f'\nMale MPs: {male_count:,} speeches, {len(male_reasons):,} reasons')
    print(f'Female MPs: {female_count:,} speeches, {len(female_reasons):,} reasons')

    # Top arguments for each gender
    all_buckets = set(male_reasons_norm.head(10).index) | set(female_reasons_norm.head(10).index)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Female MPs
    female_reasons_norm.reindex(all_buckets, fill_value=0).sort_values().plot(
        kind='barh', ax=ax1, color=COLORS['female'])
    ax1.set_xlabel('Avg Reasons per Speech', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Argument Type', fontsize=11, fontweight='bold')
    ax1.set_title(f'Female MPs (n={female_count:,})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Male MPs
    male_reasons_norm.reindex(all_buckets, fill_value=0).sort_values().plot(
        kind='barh', ax=ax2, color=COLORS['male'])
    ax2.set_xlabel('Avg Reasons per Speech', fontsize=11, fontweight='bold')
    ax2.set_title(f'Male MPs (n={male_count:,})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    fig.suptitle('Argument Types by Gender (Normalized)', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'arguments_by_gender.png', dpi=300, bbox_inches='tight')
    print(f'\nSaved: {OUTPUT_DIR / "arguments_by_gender.png"}')

    # Comparison by time period
    print('\n' + '='*80)
    print('5. GENDER COMPARISON BY TIME PERIOD')
    print('='*80)

    periods = [
        ('Pre-1918 (no women vote)', lambda y: y < 1918),
        ('1918-1928 (partial)', lambda y: 1918 <= y < 1928),
        ('Post-1928 (equal suffrage)', lambda y: y >= 1928),
    ]

    for name, condition in periods:
        period_df = substantive[substantive['year'].apply(condition)]
        if len(period_df) > 0:
            print(f'\n{name}:')
            print(f'  Total speeches: {len(period_df):,}')

            for gender in ['M', 'F']:
                gender_df = period_df[period_df['gender'] == gender]
                if len(gender_df) > 0:
                    gender_label = 'Male' if gender == 'M' else 'Female'
                    print(f'  {gender_label}: {len(gender_df):,} ({len(gender_df)/len(period_df)*100:.1f}%)')

                    stance_dist = gender_df['stance'].value_counts()
                    for stance, count in stance_dist.items():
                        print(f'    {stance}: {count:,} ({count/len(gender_df)*100:.1f}%)')

    # Summary statistics
    print('\n' + '='*80)
    print('SUMMARY')
    print('='*80)

    print(f'\nFemale MP Statistics:')
    female_df = substantive[substantive['gender'] == 'F']
    if len(female_df) > 0:
        print(f'  Total speeches: {len(female_df):,}')
        print(f'  Years active: {female_df["year"].min():.0f} - {female_df["year"].max():.0f}')
        print(f'  Pro-suffrage: {len(female_df[female_df["stance"]=="for"]):,} ({len(female_df[female_df["stance"]=="for"])/len(female_df)*100:.1f}%)')
        print(f'  Anti-suffrage: {len(female_df[female_df["stance"]=="against"]):,} ({len(female_df[female_df["stance"]=="against"])/len(female_df)*100:.1f}%)')

    print(f'\nMale MP Statistics:')
    male_df = substantive[substantive['gender'] == 'M']
    if len(male_df) > 0:
        print(f'  Total speeches: {len(male_df):,}')
        print(f'  Pro-suffrage: {len(male_df[male_df["stance"]=="for"]):,} ({len(male_df[male_df["stance"]=="for"])/len(male_df)*100:.1f}%)')
        print(f'  Anti-suffrage: {len(male_df[male_df["stance"]=="against"]):,} ({len(male_df[male_df["stance"]=="against"])/len(male_df)*100:.1f}%)')

    print(f'\nAnalysis complete!')
    print(f'Outputs saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
