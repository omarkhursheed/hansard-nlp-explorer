#!/usr/bin/env python3
"""
Gender comparison analysis using the corrected speaker attributions.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_gender_comparison():
    """Analyze gender differences in suffrage stances with corrected data."""

    print('='*80)
    print('GENDER COMPARISON ANALYSIS (CORRECTED DATA)')
    print('='*80)

    # Load corrected results
    results_file = 'outputs/llm_classification/full_results_v5_context_3_expanded_corrected.parquet'
    print(f'\nLoading corrected results from: {results_file}')
    df = pd.read_parquet(results_file)

    print(f'Total speeches: {len(df):,}')
    print(f'Year range: {df["year"].min():.0f} - {df["year"].max():.0f}')

    # Filter to substantive stances only (exclude irrelevant)
    substantive = df[df['stance'].isin(['for', 'against', 'both'])].copy()
    print(f'\nSubstantive speeches (for/against/both): {len(substantive):,}')

    # Split by gender
    female_df = substantive[substantive['gender'] == 'F']
    male_df = substantive[substantive['gender'] == 'M']

    print(f'\nFemale MPs: {len(female_df):,} speeches')
    print(f'Male MPs: {len(male_df):,} speeches')

    # Calculate pro-suffrage rates
    def calc_pro_rate(df):
        """Calculate percentage of speeches that are pro-suffrage."""
        total = len(df)
        pro = len(df[df['stance'] == 'for'])
        return (pro / total * 100) if total > 0 else 0

    female_pro_rate = calc_pro_rate(female_df)
    male_pro_rate = calc_pro_rate(male_df)

    print('\n' + '='*80)
    print('PRO-SUFFRAGE RATES (CORRECTED)')
    print('='*80)
    print(f'\nFemale MPs: {female_pro_rate:.1f}% pro-suffrage')
    print(f'  For: {len(female_df[female_df["stance"] == "for"]):,}')
    print(f'  Against: {len(female_df[female_df["stance"] == "against"]):,}')
    print(f'  Both: {len(female_df[female_df["stance"] == "both"]):,}')

    print(f'\nMale MPs: {male_pro_rate:.1f}% pro-suffrage')
    print(f'  For: {len(male_df[male_df["stance"] == "for"]):,}')
    print(f'  Against: {len(male_df[male_df["stance"] == "against"]):,}')
    print(f'  Both: {len(male_df[male_df["stance"] == "both"]):,}')

    diff = female_pro_rate - male_pro_rate
    print(f'\nDifference: {diff:+.1f} percentage points')

    # Statistical test
    from scipy.stats import chi2_contingency

    # Contingency table: rows = gender, columns = stance
    contingency = pd.crosstab(
        substantive['gender'],
        substantive['stance']
    )

    chi2, p_value, dof, expected = chi2_contingency(contingency)

    print('\n' + '='*80)
    print('STATISTICAL SIGNIFICANCE')
    print('='*80)
    print(f'Chi-square statistic: {chi2:.2f}')
    print(f'P-value: {p_value:.2e}')
    print(f'Degrees of freedom: {dof}')

    if p_value < 0.001:
        print('\n*** Highly significant difference (p < 0.001) ***')
    elif p_value < 0.05:
        print('\n** Significant difference (p < 0.05) **')
    else:
        print('\nNo significant difference (p >= 0.05)')

    # Create visualization
    output_dir = Path('analysis/suffrage_gender_comparison_corrected')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # Pro-suffrage rate comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    genders = ['Female MPs', 'Male MPs']
    rates = [female_pro_rate, male_pro_rate]
    colors = ['#EC4899', '#3B82C4']

    bars = ax.bar(genders, rates, color=colors, alpha=0.8)

    # Add percentage labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Pro-suffrage rate (%)', fontsize=12)
    ax.set_title('Pro-Suffrage Speech Rates by Gender (Corrected Data)', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Add sample sizes
    ax.text(0, -10, f'n={len(female_df):,}', ha='center', fontsize=10, color='gray')
    ax.text(1, -10, f'n={len(male_df):,}', ha='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / 'pro_suffrage_rate_by_gender_corrected.png', dpi=300, bbox_inches='tight')
    print(f'\nSaved visualization to: {output_dir / "pro_suffrage_rate_by_gender_corrected.png"}')
    plt.close()

    # Detailed breakdown
    fig, ax = plt.subplots(figsize=(10, 6))

    stances = ['for', 'against', 'both']
    stance_labels = ['Pro-suffrage', 'Anti-suffrage', 'Mixed']
    x = range(len(stances))
    width = 0.35

    female_counts = [len(female_df[female_df['stance'] == s]) for s in stances]
    male_counts = [len(male_df[male_df['stance'] == s]) for s in stances]

    # Convert to percentages
    female_pcts = [c / len(female_df) * 100 for c in female_counts]
    male_pcts = [c / len(male_df) * 100 for c in male_counts]

    bars1 = ax.bar([i - width/2 for i in x], female_pcts, width, label='Female MPs', color='#EC4899', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], male_pcts, width, label='Male MPs', color='#3B82C4', alpha=0.8)

    # Add percentage labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Percentage of speeches', fontsize=12)
    ax.set_title('Suffrage Stance Distribution by Gender (Corrected Data)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(stance_labels)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(female_pcts), max(male_pcts)) * 1.15)

    plt.tight_layout()
    plt.savefig(output_dir / 'stance_distribution_by_gender_corrected.png', dpi=300, bbox_inches='tight')
    print(f'Saved visualization to: {output_dir / "stance_distribution_by_gender_corrected.png"}')
    plt.close()

    # Save summary statistics
    summary = {
        'metric': [
            'Total speeches',
            'Female speeches',
            'Male speeches',
            'Female pro-suffrage rate',
            'Male pro-suffrage rate',
            'Difference (pct points)',
            'Chi-square statistic',
            'P-value'
        ],
        'value': [
            len(substantive),
            len(female_df),
            len(male_df),
            f'{female_pro_rate:.1f}%',
            f'{male_pro_rate:.1f}%',
            f'{diff:+.1f}',
            f'{chi2:.2f}',
            f'{p_value:.2e}'
        ]
    }

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / 'gender_comparison_summary_corrected.csv', index=False)
    print(f'Saved summary to: {output_dir / "gender_comparison_summary_corrected.csv"}')

    print('\n' + '='*80)
    print('ANALYSIS COMPLETE')
    print('='*80)

    return df


if __name__ == '__main__':
    analyze_gender_comparison()
