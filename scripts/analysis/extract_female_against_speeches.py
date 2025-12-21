#!/usr/bin/env python3
"""
Extract all anti-suffrage speeches by female MPs for detailed reading.
"""

import pandas as pd
from pathlib import Path

def main():
    print('='*80)
    print('EXTRACTING ANTI-SUFFRAGE SPEECHES BY FEMALE MPs')
    print('='*80)

    # Load classified results (CORRECTED VERSION with manual speaker fixes)
    results_file = 'outputs/llm_classification/full_results_v5_context_3_expanded_corrected.parquet'
    input_file = 'outputs/llm_classification/full_input_context_3_expanded.parquet'

    print(f'\nLoading results: {results_file}')
    df = pd.read_parquet(results_file)

    print(f'Loading input (speech texts): {input_file}')
    input_df = pd.read_parquet(input_file)

    # Merge to get speech texts
    print(f'\nMerging results with input texts...')
    df = df.merge(
        input_df[['speech_id', 'target_text', 'context_text']],
        on='speech_id',
        how='left'
    )

    print(f'Total speeches: {len(df):,}')
    print(f'Year range: {df["year"].min():.0f} - {df["year"].max():.0f}')

    # Filter to female MPs with anti-suffrage stance
    female_against = df[
        (df['gender'] == 'F') &
        (df['stance'] == 'against')
    ].copy()

    print(f'\nFemale MPs with "against" stance: {len(female_against):,}')

    # Also get "both" stance (mixed views)
    female_both = df[
        (df['gender'] == 'F') &
        (df['stance'] == 'both')
    ].copy()

    print(f'Female MPs with "both" stance: {len(female_both):,}')

    # Show summary
    if len(female_against) > 0:
        print(f'\nAnti-suffrage speeches:')
        print(f'  Year range: {female_against["year"].min():.0f} - {female_against["year"].max():.0f}')
        print(f'  Unique speakers: {female_against["canonical_name"].nunique()}')
        print(f'\nSpeakers:')
        for name, count in female_against['canonical_name'].value_counts().items():
            print(f'  {name}: {count} speeches')

    if len(female_both) > 0:
        print(f'\nMixed-stance speeches:')
        print(f'  Year range: {female_both["year"].min():.0f} - {female_both["year"].max():.0f}')
        print(f'  Unique speakers: {female_both["canonical_name"].nunique()}')

    # Save for Streamlit app
    output_dir = Path('outputs/suffrage_exploration')
    output_dir.mkdir(parents=True, exist_ok=True)

    female_against.to_parquet(output_dir / 'female_against_speeches.parquet', index=False)
    female_both.to_parquet(output_dir / 'female_both_speeches.parquet', index=False)

    print(f'\nSaved to:')
    print(f'  {output_dir / "female_against_speeches.parquet"}')
    print(f'  {output_dir / "female_both_speeches.parquet"}')

    # Also combine for easier browsing
    female_opposition = pd.concat([female_against, female_both], ignore_index=True)
    female_opposition.to_parquet(output_dir / 'female_opposition_all.parquet', index=False)
    print(f'  {output_dir / "female_opposition_all.parquet"} (combined)')

    print('\nDone!')


if __name__ == '__main__':
    main()
