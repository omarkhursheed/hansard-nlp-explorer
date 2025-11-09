#!/usr/bin/env python3
"""
Extract debates corresponding to reliable suffrage speeches.

Takes the reliable suffrage speeches and extracts their full debates,
including all speeches in those debates (not just suffrage-related ones).
"""

import pandas as pd
from pathlib import Path
from collections import Counter


def extract_corresponding_debates():
    """
    Extract full debates that contain reliable suffrage speeches.

    Returns debate-level and expanded speech-level datasets.
    """
    print("="*70)
    print("EXTRACTING DEBATES FROM RELIABLE SUFFRAGE SPEECHES")
    print("="*70)

    # Load reliable speeches
    reliable_speeches = pd.read_parquet('outputs/suffrage_reliable/speeches_reliable.parquet')
    print(f"\nReliable suffrage speeches: {len(reliable_speeches):,}")
    print(f"  HIGH: {(reliable_speeches['confidence_level'] == 'HIGH').sum():,}")
    print(f"  MEDIUM: {(reliable_speeches['confidence_level'] == 'MEDIUM').sum():,}")

    # Get unique debate IDs
    debate_ids = reliable_speeches['debate_id'].unique()
    print(f"\nUnique debates containing suffrage speeches: {len(debate_ids):,}")

    # Get year range
    years = reliable_speeches['year'].unique()
    year_min, year_max = years.min(), years.max()
    print(f"Year range: {year_min}-{year_max}")

    # Now load all speeches from these debates
    print(f"\n{'='*70}")
    print("LOADING ALL SPEECHES FROM SUFFRAGE DEBATES")
    print(f"{'='*70}\n")

    data_dir = Path('data-hansard/derived_complete')
    all_debate_speeches = []

    for year in range(year_min, year_max + 1):
        speech_file = data_dir / 'speeches_complete' / f'speeches_{year}.parquet'

        if not speech_file.exists():
            continue

        # Load all speeches for this year
        speeches_df = pd.read_parquet(speech_file)

        # Filter to Commons only (matching reliable dataset)
        speeches_df = speeches_df[speeches_df['chamber'] == 'Commons']

        # Get speeches from our debate IDs
        year_debate_speeches = speeches_df[speeches_df['debate_id'].isin(debate_ids)]

        if len(year_debate_speeches) > 0:
            print(f"  {year}: {len(year_debate_speeches):5,} speeches from {year_debate_speeches['debate_id'].nunique():4} debates")
            all_debate_speeches.append(year_debate_speeches)

    # Combine all speeches
    if all_debate_speeches:
        full_debate_speeches = pd.concat(all_debate_speeches, ignore_index=True)

        print(f"\n{'='*70}")
        print("FULL DEBATE EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"\nTotal speeches in suffrage debates: {len(full_debate_speeches):,}")
        print(f"  From {full_debate_speeches['debate_id'].nunique():,} unique debates")
        print(f"  Date range: {full_debate_speeches['date'].min()} to {full_debate_speeches['date'].max()}")
        print(f"  Year range: {full_debate_speeches['year'].min()}-{full_debate_speeches['year'].max()}")

        # Match rate
        matched = (full_debate_speeches['matched_mp'] == True).sum()
        print(f"\nMP matching: {matched:,} / {len(full_debate_speeches):,} ({matched/len(full_debate_speeches)*100:.1f}%)")

        if matched > 0:
            gendered = full_debate_speeches[full_debate_speeches['matched_mp'] == True]
            gender_counts = gendered['gender'].value_counts()
            print(f"  Male: {gender_counts.get('M', 0):,}")
            print(f"  Female: {gender_counts.get('F', 0):,}")

        # Create debate-level summary
        print(f"\n{'='*70}")
        print("CREATING DEBATE-LEVEL SUMMARY")
        print(f"{'='*70}")

        debate_stats = create_debate_summary(full_debate_speeches, reliable_speeches)

        # Save results
        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print(f"{'='*70}")

        output_dir = Path('outputs/suffrage_debates')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full speeches in debates
        full_debate_speeches.to_parquet(output_dir / 'all_speeches_in_suffrage_debates.parquet')
        print(f"\nSaved: {output_dir}/all_speeches_in_suffrage_debates.parquet")
        print(f"  {len(full_debate_speeches):,} speeches")

        # Save debate-level summary
        debate_stats.to_parquet(output_dir / 'debate_summary.parquet')
        debate_stats.to_csv(output_dir / 'debate_summary.csv', index=False)
        print(f"\nSaved: {output_dir}/debate_summary.parquet")
        print(f"  {len(debate_stats):,} debates")

        # Save samples
        full_debate_speeches.head(500).to_csv(output_dir / 'speeches_sample.csv', index=False)
        debate_stats.head(100).to_csv(output_dir / 'debate_summary_sample.csv', index=False)

        # Create summary report
        create_summary_report(full_debate_speeches, debate_stats, reliable_speeches, output_dir)

        return full_debate_speeches, debate_stats

    else:
        print("\nNo speeches found!")
        return None, None


def create_debate_summary(full_speeches, reliable_speeches):
    """
    Create debate-level summary statistics.

    Args:
        full_speeches: All speeches in suffrage debates
        reliable_speeches: Original reliable suffrage speeches

    Returns:
        DataFrame with debate-level stats
    """
    # Create a mapping of which speeches are suffrage-related
    suffrage_speech_ids = set(reliable_speeches['speech_id'].values)
    full_speeches['is_suffrage_speech'] = full_speeches['speech_id'].isin(suffrage_speech_ids)

    # Add confidence level for suffrage speeches
    confidence_map = dict(zip(reliable_speeches['speech_id'], reliable_speeches['confidence_level']))
    full_speeches['suffrage_confidence'] = full_speeches['speech_id'].map(confidence_map)

    # Group by debate
    debate_stats = full_speeches.groupby('debate_id').agg({
        'speech_id': 'count',
        'is_suffrage_speech': 'sum',
        'canonical_name': 'nunique',
        'date': 'first',
        'year': 'first',
        'word_count': 'sum',
        'matched_mp': 'sum',
    }).rename(columns={
        'speech_id': 'total_speeches',
        'is_suffrage_speech': 'suffrage_speeches',
        'canonical_name': 'unique_speakers',
        'word_count': 'total_words',
        'matched_mp': 'matched_speeches',
    })

    # Add high/medium breakdown
    high_counts = full_speeches[full_speeches['suffrage_confidence'] == 'HIGH'].groupby('debate_id').size()
    medium_counts = full_speeches[full_speeches['suffrage_confidence'] == 'MEDIUM'].groupby('debate_id').size()

    debate_stats['suffrage_high'] = debate_stats.index.map(high_counts).fillna(0).astype(int)
    debate_stats['suffrage_medium'] = debate_stats.index.map(medium_counts).fillna(0).astype(int)

    # Gender breakdown (for matched speeches only)
    matched_speeches = full_speeches[full_speeches['matched_mp'] == True]

    male_counts = matched_speeches[matched_speeches['gender'] == 'M'].groupby('debate_id').size()
    female_counts = matched_speeches[matched_speeches['gender'] == 'F'].groupby('debate_id').size()

    debate_stats['male_speakers'] = debate_stats.index.map(male_counts).fillna(0).astype(int)
    debate_stats['female_speakers'] = debate_stats.index.map(female_counts).fillna(0).astype(int)

    # Calculate percentages
    debate_stats['suffrage_pct'] = (debate_stats['suffrage_speeches'] / debate_stats['total_speeches'] * 100).round(1)
    debate_stats['match_rate'] = (debate_stats['matched_speeches'] / debate_stats['total_speeches'] * 100).round(1)

    # Reset index to make debate_id a column
    debate_stats = debate_stats.reset_index()

    # Sort by date
    debate_stats = debate_stats.sort_values('date')

    print(f"\nDebate-level statistics created for {len(debate_stats):,} debates")

    return debate_stats


def create_summary_report(full_speeches, debate_stats, reliable_speeches, output_dir):
    """Create a summary report file."""

    with open(output_dir / 'SUMMARY.txt', 'w') as f:
        f.write("SUFFRAGE DEBATES EXTRACTION SUMMARY\n")
        f.write("="*70 + "\n\n")

        f.write("## Overview\n\n")
        f.write(f"This dataset contains all speeches from debates that include at least one\n")
        f.write(f"reliable suffrage speech (HIGH or MEDIUM confidence from validation).\n\n")

        f.write("## Speech-Level Statistics\n\n")
        f.write(f"Total speeches in suffrage debates: {len(full_speeches):,}\n")
        f.write(f"  Suffrage-related speeches: {full_speeches['is_suffrage_speech'].sum():,}\n")
        f.write(f"    HIGH confidence: {(full_speeches['suffrage_confidence'] == 'HIGH').sum():,}\n")
        f.write(f"    MEDIUM confidence: {(full_speeches['suffrage_confidence'] == 'MEDIUM').sum():,}\n")
        f.write(f"  Non-suffrage speeches: {(~full_speeches['is_suffrage_speech']).sum():,}\n\n")

        f.write(f"Date range: {full_speeches['date'].min()} to {full_speeches['date'].max()}\n")
        f.write(f"Year range: {full_speeches['year'].min()}-{full_speeches['year'].max()}\n\n")

        matched = (full_speeches['matched_mp'] == True).sum()
        f.write(f"MP matching: {matched:,} / {len(full_speeches):,} ({matched/len(full_speeches)*100:.1f}%)\n")

        if matched > 0:
            gendered = full_speeches[full_speeches['matched_mp'] == True]
            gender_counts = gendered['gender'].value_counts()
            f.write(f"  Male: {gender_counts.get('M', 0):,}\n")
            f.write(f"  Female: {gender_counts.get('F', 0):,}\n\n")

        f.write("## Debate-Level Statistics\n\n")
        f.write(f"Total debates: {len(debate_stats):,}\n")
        f.write(f"Average speeches per debate: {debate_stats['total_speeches'].mean():.1f}\n")
        f.write(f"Average suffrage speeches per debate: {debate_stats['suffrage_speeches'].mean():.1f}\n")
        f.write(f"Average unique speakers per debate: {debate_stats['unique_speakers'].mean():.1f}\n\n")

        f.write("Debate size distribution:\n")
        f.write(f"  Min: {debate_stats['total_speeches'].min()} speeches\n")
        f.write(f"  25th percentile: {debate_stats['total_speeches'].quantile(0.25):.0f} speeches\n")
        f.write(f"  Median: {debate_stats['total_speeches'].median():.0f} speeches\n")
        f.write(f"  75th percentile: {debate_stats['total_speeches'].quantile(0.75):.0f} speeches\n")
        f.write(f"  Max: {debate_stats['total_speeches'].max()} speeches\n\n")

        f.write("Suffrage speech percentage in debates:\n")
        f.write(f"  Mean: {debate_stats['suffrage_pct'].mean():.1f}%\n")
        f.write(f"  Median: {debate_stats['suffrage_pct'].median():.1f}%\n")
        f.write(f"  Min: {debate_stats['suffrage_pct'].min():.1f}%\n")
        f.write(f"  Max: {debate_stats['suffrage_pct'].max():.1f}%\n\n")

        f.write("## Largest Debates\n\n")
        largest = debate_stats.nlargest(10, 'total_speeches')[
            ['date', 'total_speeches', 'suffrage_speeches', 'suffrage_high', 'unique_speakers']
        ]
        f.write(largest.to_string(index=False))
        f.write("\n\n")

        f.write("## Most Suffrage-Focused Debates (by count)\n\n")
        most_suffrage = debate_stats.nlargest(10, 'suffrage_speeches')[
            ['date', 'total_speeches', 'suffrage_speeches', 'suffrage_high', 'suffrage_pct']
        ]
        f.write(most_suffrage.to_string(index=False))
        f.write("\n\n")

        f.write("## Peak Years\n\n")
        by_year = debate_stats.groupby('year').agg({
            'debate_id': 'count',
            'total_speeches': 'sum',
            'suffrage_speeches': 'sum',
            'suffrage_high': 'sum',
            'female_speakers': 'sum',
        }).rename(columns={'debate_id': 'debates'})

        top_years = by_year.nlargest(15, 'suffrage_speeches')
        f.write(top_years.to_string())
        f.write("\n\n")

        f.write("## Files\n\n")
        f.write("- all_speeches_in_suffrage_debates.parquet: All speeches from debates containing suffrage speeches\n")
        f.write("- debate_summary.parquet: Debate-level statistics\n")
        f.write("- speeches_sample.csv: Sample of 500 speeches for inspection\n")
        f.write("- debate_summary_sample.csv: Sample of 100 debates for inspection\n")
        f.write("- SUMMARY.txt: This file\n")

    print(f"\nSaved: {output_dir}/SUMMARY.txt")


if __name__ == '__main__':
    full_speeches, debate_stats = extract_corresponding_debates()
