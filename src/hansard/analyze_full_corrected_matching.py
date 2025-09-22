#!/usr/bin/env python3
"""
Analyze full dataset matching with corrected matcher
Provides comprehensive statistics on the improvements
"""

import pandas as pd
import numpy as np
from pathlib import Path
from mp_matcher_corrected import CorrectedMPMatcher
from tqdm import tqdm
import json

def analyze_full_dataset():
    """Analyze matching on the full dataset"""

    print("=" * 70)
    print("FULL DATASET ANALYSIS - CORRECTED MATCHER")
    print("=" * 70)

    # Load MP data
    mp_data = pd.read_parquet("data/house_members_gendered_updated.parquet")
    matcher = CorrectedMPMatcher(mp_data)

    # Get all available years
    metadata_dir = Path("data/processed_fixed/metadata")
    available_years = sorted([
        int(f.stem.split('_')[1])
        for f in metadata_dir.glob("speakers_*.parquet")
        if 'master' not in f.stem
    ])

    print(f"\nProcessing {len(available_years)} years of data")
    print(f"Years range: {min(available_years)} - {max(available_years)}")

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

    # Sample of interesting cases
    interesting_cases = []

    # Process each year
    for year in tqdm(available_years, desc="Processing years"):
        speaker_file = Path(f"data/processed_fixed/metadata/speakers_{year}.parquet")
        if not speaker_file.exists():
            continue

        speakers_df = pd.read_parquet(speaker_file)
        decade = (year // 10) * 10

        if decade not in all_stats['by_decade']:
            all_stats['by_decade'][decade] = {
                'total': 0, 'certain': 0, 'ambiguous': 0, 'no_match': 0
            }

        # Process each speaker
        for _, row in speakers_df.iterrows():
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

                # Track method
                method = result['match_type']
                all_stats['by_method'][method] = all_stats['by_method'].get(method, 0) + 1

                # Track confidence
                conf = result.get('confidence', 0)
                if conf >= 0.9:
                    all_stats['by_confidence']['high'] += 1
                elif conf >= 0.7:
                    all_stats['by_confidence']['medium'] += 1
                else:
                    all_stats['by_confidence']['low'] += 1

                # Track gender
                gender = result.get('gender', None)
                if gender:
                    all_stats['gender_distribution'][gender] = \
                        all_stats['gender_distribution'].get(gender, 0) + 1
                else:
                    all_stats['gender_distribution']['None'] += 1

            # Collect interesting cases
            if len(interesting_cases) < 100:
                if result['match_type'] in ['title', 'constituency', 'title_resolution_transition']:
                    interesting_cases.append({
                        'speaker': speaker,
                        'year': year,
                        'match': result.get('final_match'),
                        'method': result['match_type'],
                        'confidence': result.get('confidence')
                    })

    # Print comprehensive statistics
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)

    total = all_stats['total_speakers']
    print(f"\nTotal speaker records processed: {total:,}")

    print("\n=== OVERALL MATCH RATES ===")
    print(f"Certain matches: {all_stats['certain_matches']:,} ({100*all_stats['certain_matches']/total:.1f}%)")
    print(f"Ambiguous: {all_stats['ambiguous_matches']:,} ({100*all_stats['ambiguous_matches']/total:.1f}%)")
    print(f"No match: {all_stats['no_matches']:,} ({100*all_stats['no_matches']/total:.1f}%)")
    print(f"Procedural: {all_stats['procedural']:,} ({100*all_stats['procedural']/total:.1f}%)")

    print("\n=== MATCHING METHODS BREAKDOWN ===")
    for method, count in sorted(all_stats['by_method'].items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / all_stats['certain_matches'] if all_stats['certain_matches'] > 0 else 0
        print(f"{method}: {count:,} ({pct:.1f}% of certain matches)")

    print("\n=== CONFIDENCE DISTRIBUTION ===")
    certain = all_stats['certain_matches']
    if certain > 0:
        print(f"High confidence (≥0.9): {all_stats['by_confidence']['high']:,} ({100*all_stats['by_confidence']['high']/certain:.1f}%)")
        print(f"Medium confidence (0.7-0.9): {all_stats['by_confidence']['medium']:,} ({100*all_stats['by_confidence']['medium']/certain:.1f}%)")
        print(f"Low confidence (<0.7): {all_stats['by_confidence']['low']:,} ({100*all_stats['by_confidence']['low']/certain:.1f}%)")

    print("\n=== GENDER DISTRIBUTION (of matched MPs) ===")
    total_gendered = sum(all_stats['gender_distribution'].values())
    if total_gendered > 0:
        for gender, count in sorted(all_stats['gender_distribution'].items()):
            print(f"{gender}: {count:,} ({100*count/total_gendered:.1f}%)")

    print("\n=== MATCH RATES BY DECADE ===")
    for decade in sorted(all_stats['by_decade'].keys()):
        stats = all_stats['by_decade'][decade]
        decade_total = stats['total']
        if decade_total > 0:
            certain_pct = 100 * stats['certain'] / decade_total
            print(f"{decade}s: {stats['certain']:,}/{decade_total:,} ({certain_pct:.1f}% certain)")

    print("\n=== SAMPLE INTERESTING MATCHES ===")
    for case in interesting_cases[:10]:
        print(f"{case['year']}: '{case['speaker']}' → {case['match']} "
              f"(method: {case['method']}, conf: {case['confidence']:.2f})")

    # Save statistics
    output_dir = Path("data_filtered_by_actual_mp_CORRECTED")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "comprehensive_statistics.json", "w") as f:
        # Convert numpy types for JSON serialization
        def convert(o):
            if isinstance(o, np.int64):
                return int(o)
            if isinstance(o, np.float64):
                return float(o)
            raise TypeError

        json.dump(all_stats, f, indent=2, default=convert)

    print(f"\nStatistics saved to {output_dir}/comprehensive_statistics.json")

    # Comparison with baseline
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINE MATCHER")
    print("=" * 70)

    print("\nBaseline (simple surname matching):")
    print("  - Match rate: ~50% (but many false positives)")
    print("  - No ambiguity detection")
    print("  - No temporal validation")

    print("\nCorrected Matcher improvements:")
    print(f"  - Certain matches: {100*all_stats['certain_matches']/total:.1f}%")
    print(f"  - Explicitly identifies ambiguous cases: {100*all_stats['ambiguous_matches']/total:.1f}%")
    print("  - Uses verified Prime Minister dates from gov.uk")
    print("  - Handles transition dates correctly")
    print("  - Constituency-based matching")
    print("  - OCR error correction")
    print("  - Temporal validation prevents impossible matches")

    effective_coverage = all_stats['certain_matches'] + all_stats['ambiguous_matches']
    print(f"\nEffective coverage (certain + ambiguous): {100*effective_coverage/total:.1f}%")
    print("This represents cases where we have identified the MP(s), even if ambiguous")

if __name__ == "__main__":
    analyze_full_dataset()