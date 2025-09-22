#!/usr/bin/env python3
"""
Analyze corrected matching with sampling for efficiency
"""

import pandas as pd
import numpy as np
from pathlib import Path
from mp_matcher_corrected import CorrectedMPMatcher
import json

def analyze_sampled_dataset():
    """Analyze matching on sampled years"""

    print("=" * 70)
    print("CORRECTED MATCHER ANALYSIS - SAMPLED DATASET")
    print("=" * 70)

    # Load MP data
    mp_data = pd.read_parquet("data/house_members_gendered_updated.parquet")
    matcher = CorrectedMPMatcher(mp_data)

    # Sample years across the dataset
    sample_years = [
        1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900,
        1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000
    ]

    print(f"\nAnalyzing {len(sample_years)} sampled years")

    # Initialize statistics
    all_stats = {
        'total_speakers': 0,
        'certain_matches': 0,
        'ambiguous_matches': 0,
        'no_matches': 0,
        'procedural': 0,
        'by_method': {},
        'by_confidence': {'high': 0, 'medium': 0, 'low': 0},
        'by_decade': {},
        'gender_distribution': {'M': 0, 'F': 0, 'None': 0}
    }

    interesting_cases = []

    for year in sample_years:
        print(f"Processing {year}...", end=' ')

        speaker_file = Path(f"data/processed_fixed/metadata/speakers_{year}.parquet")
        if not speaker_file.exists():
            print("No data")
            continue

        speakers_df = pd.read_parquet(speaker_file)
        print(f"{len(speakers_df)} speakers")

        decade = (year // 10) * 10
        if decade not in all_stats['by_decade']:
            all_stats['by_decade'][decade] = {
                'total': 0, 'certain': 0, 'ambiguous': 0, 'no_match': 0
            }

        # Sample up to 1000 speakers per year for efficiency
        sample_size = min(1000, len(speakers_df))
        sample_df = speakers_df.sample(n=sample_size, random_state=42)

        for _, row in sample_df.iterrows():
            speaker = row['speaker_name']
            date = row.get('reference_date', f'{year}-01-01')
            chamber = row.get('chamber', 'Commons')

            result = matcher.match_comprehensive(speaker, date, chamber)

            # Update statistics
            all_stats['total_speakers'] += 1
            all_stats['by_decade'][decade]['total'] += 1

            if result['match_type'] == 'procedural':
                all_stats['procedural'] += 1
            elif result['match_type'] == 'ambiguous':
                all_stats['ambiguous_matches'] += 1
                all_stats['by_decade'][decade]['ambiguous'] += 1
            elif result['match_type'] == 'no_match':
                all_stats['no_matches'] += 1
                all_stats['by_decade'][decade]['no_match'] += 1
            else:
                all_stats['certain_matches'] += 1
                all_stats['by_decade'][decade]['certain'] += 1

                method = result['match_type']
                all_stats['by_method'][method] = all_stats['by_method'].get(method, 0) + 1

                conf = result.get('confidence', 0)
                if conf >= 0.9:
                    all_stats['by_confidence']['high'] += 1
                elif conf >= 0.7:
                    all_stats['by_confidence']['medium'] += 1
                else:
                    all_stats['by_confidence']['low'] += 1

                gender = result.get('gender')
                if gender:
                    all_stats['gender_distribution'][gender] = \
                        all_stats['gender_distribution'].get(gender, 0) + 1
                else:
                    all_stats['gender_distribution']['None'] += 1

            # Collect interesting cases
            if result['match_type'] in ['title', 'constituency', 'title_resolution_transition']:
                interesting_cases.append({
                    'speaker': speaker[:50],
                    'year': year,
                    'match': result.get('final_match'),
                    'method': result['match_type'],
                    'confidence': result.get('confidence')
                })

    # Print statistics
    print("\n" + "=" * 70)
    print("FINAL STATISTICS (SAMPLED)")
    print("=" * 70)

    total = all_stats['total_speakers']
    print(f"\nTotal speaker records analyzed: {total:,}")

    print("\n=== OVERALL MATCH RATES ===")
    print(f"Certain matches: {all_stats['certain_matches']:,} ({100*all_stats['certain_matches']/total:.1f}%)")
    print(f"Ambiguous: {all_stats['ambiguous_matches']:,} ({100*all_stats['ambiguous_matches']/total:.1f}%)")
    print(f"No match: {all_stats['no_matches']:,} ({100*all_stats['no_matches']/total:.1f}%)")
    print(f"Procedural: {all_stats['procedural']:,} ({100*all_stats['procedural']/total:.1f}%)")

    print("\n=== MATCHING METHODS ===")
    for method, count in sorted(all_stats['by_method'].items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / all_stats['certain_matches'] if all_stats['certain_matches'] > 0 else 0
        print(f"{method}: {count:,} ({pct:.1f}% of certain matches)")

    print("\n=== CONFIDENCE DISTRIBUTION ===")
    certain = all_stats['certain_matches']
    if certain > 0:
        print(f"High (≥0.9): {all_stats['by_confidence']['high']:,} ({100*all_stats['by_confidence']['high']/certain:.1f}%)")
        print(f"Medium (0.7-0.9): {all_stats['by_confidence']['medium']:,} ({100*all_stats['by_confidence']['medium']/certain:.1f}%)")
        print(f"Low (<0.7): {all_stats['by_confidence']['low']:,} ({100*all_stats['by_confidence']['low']/certain:.1f}%)")

    print("\n=== GENDER DISTRIBUTION ===")
    total_gendered = sum(all_stats['gender_distribution'].values())
    if total_gendered > 0:
        for gender, count in sorted(all_stats['gender_distribution'].items()):
            print(f"{gender}: {count:,} ({100*count/total_gendered:.1f}%)")

    print("\n=== TRENDS BY DECADE ===")
    for decade in sorted(all_stats['by_decade'].keys()):
        stats = all_stats['by_decade'][decade]
        decade_total = stats['total']
        if decade_total > 0:
            certain_pct = 100 * stats['certain'] / decade_total
            ambig_pct = 100 * stats['ambiguous'] / decade_total
            print(f"{decade}s: {certain_pct:.1f}% certain, {ambig_pct:.1f}% ambiguous")

    print("\n=== SAMPLE INTERESTING MATCHES ===")
    for case in interesting_cases[:15]:
        print(f"{case['year']}: '{case['speaker']}' → {case['match']} "
              f"({case['method']}, {case['confidence']:.2f})")

    # Extrapolate to full dataset
    print("\n" + "=" * 70)
    print("EXTRAPOLATED TO FULL DATASET (2.7M records)")
    print("=" * 70)

    certain_rate = all_stats['certain_matches'] / total
    ambig_rate = all_stats['ambiguous_matches'] / total
    no_match_rate = all_stats['no_matches'] / total

    full_total = 2700000
    print(f"\nEstimated certain matches: {int(certain_rate * full_total):,} ({100*certain_rate:.1f}%)")
    print(f"Estimated ambiguous: {int(ambig_rate * full_total):,} ({100*ambig_rate:.1f}%)")
    print(f"Estimated no match: {int(no_match_rate * full_total):,} ({100*no_match_rate:.1f}%)")

    effective_coverage = certain_rate + ambig_rate
    print(f"\nEffective coverage: {100*effective_coverage:.1f}%")
    print("(Cases where we've identified the MP(s), even if ambiguous)")

if __name__ == "__main__":
    analyze_sampled_dataset()