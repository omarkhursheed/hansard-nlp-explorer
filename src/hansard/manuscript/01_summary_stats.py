#!/usr/bin/env python3
"""
Generate summary statistics for manuscript.
Computes all key numbers: speeches, matches, gender distribution, etc.

Usage:
    python3 01_summary_stats.py

Output:
    manuscript_figures/stats.json - All statistics for paper
"""

import json
import pandas as pd
from pathlib import Path
from data_loader import load_speeches, load_debates, get_data_dir
from utils import get_output_dir


def compute_overall_statistics():
    """Compute dataset-wide statistics."""
    print("Computing overall statistics...")

    # Load summary from derived_complete
    summary_path = get_data_dir() / 'derived_complete' / 'dataset_summary.json'
    with open(summary_path) as f:
        summary = json.load(f)

    stats = {
        'dataset': {
            'total_debates': summary['debates']['total'],
            'total_speeches': summary['speeches']['total'],
            'speeches_with_gender': summary['speeches']['with_gender'],
            'speeches_male': summary['speeches']['male'],
            'speeches_female': summary['speeches']['female'],
            'speeches_unmatched': summary['speeches']['unmatched'],
            'gender_match_rate': round(summary['speeches']['gender_match_rate'] * 100, 2),
            'female_percentage': round(
                summary['speeches']['female'] / summary['speeches']['with_gender'] * 100, 2
            ),
        }
    }

    return stats


def compute_speaker_statistics():
    """Compute unique speaker statistics (Commons only)."""
    print("Computing unique speaker statistics (Commons only)...")

    # Load all Commons speeches
    speeches = load_speeches(chamber='Commons')

    # Get unique speakers using person_id for matched speakers
    # This gives us exact unique individuals
    matched = speeches[speeches['matched_mp'] == True].copy()
    unmatched = speeches[speeches['matched_mp'] == False].copy()

    # For matched speakers, use person_id to count unique individuals
    unique_matched = matched[['person_id', 'gender']].drop_duplicates(subset=['person_id'])
    unique_matched = unique_matched[unique_matched['person_id'].notna()]

    # For unmatched, use normalized_speaker
    unique_unmatched = unmatched[['normalized_speaker']].drop_duplicates(subset=['normalized_speaker'])
    unique_unmatched = unique_unmatched[unique_unmatched['normalized_speaker'].str.strip() != '']

    stats = {
        'total_unique_speakers': int(len(unique_matched) + len(unique_unmatched)),
        'unique_speakers_matched': int(len(unique_matched)),
        'unique_speakers_unmatched': int(len(unique_unmatched)),

        # By gender (only for matched speakers)
        'unique_speakers_male': int((unique_matched['gender'] == 'M').sum()),
        'unique_speakers_female': int((unique_matched['gender'] == 'F').sum()),
        'unique_speakers_no_gender': int(len(unique_unmatched)),

        # Match rate
        'speaker_match_rate': round(
            len(unique_matched) / (len(unique_matched) + len(unique_unmatched)) * 100, 2
        ) if (len(unique_matched) + len(unique_unmatched)) > 0 else 0,
    }

    return stats


def compute_debate_statistics():
    """Compute debate-level statistics (Commons only)."""
    print("Computing debate-level statistics (Commons only)...")

    # Load Commons debates only
    debates = load_debates(chamber='Commons')

    # Calculate match rates for each debate
    debates['match_rate'] = debates['confirmed_mps'] / debates['total_speakers']
    debates['female_rate'] = debates['female_mps'] / debates['confirmed_mps'].replace(0, 1)  # Avoid div by zero

    stats = {
        'total_debates': int(len(debates)),

        # Matching statistics
        'debates_all_matched': int((debates['match_rate'] == 1.0).sum()),
        'debates_majority_matched': int((debates['match_rate'] > 0.5).sum()),
        'debates_any_matched': int((debates['confirmed_mps'] > 0).sum()),

        # Female participation statistics
        'debates_any_female': int(debates['has_female'].sum()),
        'debates_majority_female': int((debates['female_rate'] > 0.5).sum()),
        'debates_all_female': int((debates['gender_ratio'] == 1.0).sum()),

        # Additional statistics
        'avg_speakers_per_debate': round(debates['total_speakers'].mean(), 2),
        'avg_match_rate': round(debates['match_rate'].mean() * 100, 2),
        'median_match_rate': round(debates['match_rate'].median() * 100, 2),
    }

    return stats


def compute_temporal_statistics():
    """Compute statistics by decade (Commons only)."""
    print("Computing temporal statistics (Commons only)...")

    # Load all speeches (Commons only)
    print("  Loading speeches...")
    speeches = load_speeches(chamber='Commons')  # Commons only

    # Group by decade
    by_decade = speeches.groupby('decade').agg({
        'speech_id': 'count',
        'word_count': 'sum',
        'gender': lambda x: (x == 'F').sum(),  # Female count (uppercase!)
    }).rename(columns={
        'speech_id': 'total_speeches',
        'word_count': 'total_words',
        'gender': 'female_speeches'
    })

    # Add male speeches
    by_decade['male_speeches'] = speeches.groupby('decade')['gender'].apply(
        lambda x: (x == 'M').sum()  # Male count (uppercase!)
    )

    # Add percentages
    by_decade['female_percentage'] = round(
        by_decade['female_speeches'] /
        (by_decade['female_speeches'] + by_decade['male_speeches']) * 100,
        2
    )

    # Convert to dict and ensure all values are JSON serializable
    result = {}
    for decade, row in by_decade.to_dict('index').items():
        result[str(decade)] = {
            k: int(v) if hasattr(v, 'item') and 'int' in str(type(v)) else float(v) if hasattr(v, 'item') else v
            for k, v in row.items()
        }
    return result


def compute_chamber_statistics():
    """Compute statistics by chamber (Commons vs Lords) - FOR REFERENCE ONLY."""
    print("Computing chamber statistics (for reference)...")

    speeches = load_speeches()

    by_chamber = speeches.groupby('chamber').agg({
        'speech_id': 'count',
        'word_count': 'sum',
    }).rename(columns={
        'speech_id': 'total_speeches',
        'word_count': 'total_words'
    })

    # Add gender breakdown
    for chamber in ['Commons', 'Lords']:
        chamber_speeches = speeches[speeches['chamber'] == chamber]
        by_chamber.loc[chamber, 'female_speeches'] = (chamber_speeches['gender'] == 'F').sum()
        by_chamber.loc[chamber, 'male_speeches'] = (chamber_speeches['gender'] == 'M').sum()
        by_chamber.loc[chamber, 'unmatched_speeches'] = (chamber_speeches['gender'].isna() |
                                                          (chamber_speeches['gender'] == 'None')).sum()

    # Convert to dict and ensure all values are JSON serializable
    result = {}
    for chamber, row in by_chamber.to_dict('index').items():
        result[chamber] = {
            k: int(v) if hasattr(v, 'item') and 'int' in str(type(v)) else float(v) if hasattr(v, 'item') else v
            for k, v in row.items()
        }
    return result


def compute_milestone_statistics():
    """Compute statistics around major historical events (Commons only)."""
    print("Computing milestone statistics (Commons only)...")

    milestones = {
        'astor_1919': {'year': 1919, 'window': 5},
        'equal_suffrage_1928': {'year': 1928, 'window': 5},
        'ww2_1939_1945': {'year': 1942, 'window': 3},  # Middle of war
        'thatcher_1979': {'year': 1979, 'window': 5},
        'blair_1997': {'year': 1997, 'window': 5},
    }

    stats = {}
    speeches = load_speeches(chamber='Commons')

    for key, milestone in milestones.items():
        year = milestone['year']
        window = milestone['window']

        # Before window
        before = speeches[
            (speeches['year'] >= year - window) &
            (speeches['year'] < year)
        ]

        # After window
        after = speeches[
            (speeches['year'] > year) &
            (speeches['year'] <= year + window)
        ]

        # Calculate with proper handling of zero division and numpy types
        before_f = int((before['gender'] == 'F').sum())
        before_m = int((before['gender'] == 'M').sum())
        after_f = int((after['gender'] == 'F').sum())
        after_m = int((after['gender'] == 'M').sum())

        before_total = before_f + before_m
        after_total = after_f + after_m

        stats[key] = {
            'year': year,
            'before': {
                'total': int(len(before)),
                'female': before_f,
                'male': before_m,
                'female_pct': round(before_f / before_total * 100, 2) if before_total > 0 else 0
            },
            'after': {
                'total': int(len(after)),
                'female': after_f,
                'male': after_m,
                'female_pct': round(after_f / after_total * 100, 2) if after_total > 0 else 0
            }
        }

    return stats


def main():
    """Generate all summary statistics."""
    print("=" * 70)
    print("GENERATING MANUSCRIPT SUMMARY STATISTICS")
    print("=" * 70)

    stats = {}

    # Overall statistics
    stats['overall'] = compute_overall_statistics()

    # Speaker statistics
    stats['speakers'] = compute_speaker_statistics()

    # Debate-level statistics
    stats['debates'] = compute_debate_statistics()

    # Temporal statistics
    stats['by_decade'] = compute_temporal_statistics()

    # Chamber statistics
    stats['by_chamber'] = compute_chamber_statistics()

    # Milestone statistics
    stats['milestones'] = compute_milestone_statistics()

    # Save to JSON
    output_path = get_output_dir() / 'stats.json'
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'-' * 70}")
    print(f"Statistics saved to: {output_path}")
    print(f"{'-' * 70}")

    # Print key numbers
    print("\nKEY STATISTICS:")
    print(f"  Total debates: {stats['overall']['dataset']['total_debates']:,}")
    print(f"  Total speeches: {stats['overall']['dataset']['total_speeches']:,}")
    print(f"  Gender match rate: {stats['overall']['dataset']['gender_match_rate']}%")
    print(f"  Female speeches: {stats['overall']['dataset']['speeches_female']:,} "
          f"({stats['overall']['dataset']['female_percentage']}%)")
    print(f"  Male speeches: {stats['overall']['dataset']['speeches_male']:,}")

    print("\nUNIQUE SPEAKER STATISTICS (COMMONS):")
    print(f"  Total unique speakers: {stats['speakers']['total_unique_speakers']:,}")
    print(f"  Unique speakers matched: {stats['speakers']['unique_speakers_matched']:,} "
          f"({stats['speakers']['speaker_match_rate']}%)")
    print(f"  Unique speakers unmatched: {stats['speakers']['unique_speakers_unmatched']:,}")
    print(f"  Unique male speakers: {stats['speakers']['unique_speakers_male']:,}")
    print(f"  Unique female speakers: {stats['speakers']['unique_speakers_female']:,}")
    print(f"  Unique speakers (no gender): {stats['speakers']['unique_speakers_no_gender']:,}")

    print("\nDEBATE STATISTICS (COMMONS):")
    print(f"  Total debates: {stats['debates']['total_debates']:,}")
    print(f"  Debates with all speakers matched: {stats['debates']['debates_all_matched']:,}")
    print(f"  Debates with >50% speakers matched: {stats['debates']['debates_majority_matched']:,}")
    print(f"  Debates with any female speakers: {stats['debates']['debates_any_female']:,}")
    print(f"  Debates with >50% female speakers: {stats['debates']['debates_majority_female']:,}")
    print(f"  Average match rate: {stats['debates']['avg_match_rate']}%")

    print("\nCOMPLETE!")


if __name__ == '__main__':
    main()
