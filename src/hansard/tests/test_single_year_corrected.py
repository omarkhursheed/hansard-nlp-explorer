#!/usr/bin/env python3
"""
Test corrected matcher on a single year
"""

import pandas as pd
from pathlib import Path
from hansard.scripts.matching.mp_matcher_corrected import CorrectedMPMatcher
from hansard.utils.path_config import Paths

def test_single_year(year=1950):
    """Test on a single year"""

    print(f"Testing year {year}...")

    # Load MP data
    mp_data = pd.read_parquet(Paths.get_data_dir() / "house_members_gendered_updated.parquet")
    matcher = CorrectedMPMatcher(mp_data)

    # Load speaker data
    speaker_file = Paths.get_processed_data_dir() / f"metadata/speakers_{year}.parquet"
    if not speaker_file.exists():
        print(f"No data for {year}")
        return

    speakers_df = pd.read_parquet(speaker_file)
    print(f"Found {len(speakers_df)} speaker records")

    # Sample first 10 records
    print(f"\nFirst 10 speakers in {year}:")
    for i, row in speakers_df.head(10).iterrows():
        speaker = row['speaker_name']
        date = row.get('reference_date', f'{year}-01-01')
        chamber = row.get('chamber', 'Commons')

        result = matcher.match_comprehensive(speaker, date, chamber)

        if result['final_match']:
            print(f"  {speaker[:30]:30} → {result['final_match']} ({result['confidence']:.2f})")
        elif result['match_type'] == 'procedural':
            print(f"  {speaker[:30]:30} → [PROCEDURAL]")
        elif result['match_type'] == 'ambiguous':
            print(f"  {speaker[:30]:30} → [AMBIGUOUS: {result.get('ambiguity_count')} candidates]")
        else:
            print(f"  {speaker[:30]:30} → [NO MATCH]")

    # Get overall statistics
    print(f"\nProcessing all {len(speakers_df)} records...")

    stats = {
        'certain': 0,
        'ambiguous': 0,
        'no_match': 0,
        'procedural': 0,
        'by_method': {}
    }

    for _, row in speakers_df.iterrows():
        speaker = row['speaker_name']
        date = row.get('reference_date', f'{year}-01-01')
        chamber = row.get('chamber', 'Commons')

        result = matcher.match_comprehensive(speaker, date, chamber)

        if result['match_type'] == 'procedural':
            stats['procedural'] += 1
        elif result['match_type'] == 'ambiguous':
            stats['ambiguous'] += 1
        elif result['match_type'] == 'no_match':
            stats['no_match'] += 1
        else:
            stats['certain'] += 1
            method = result['match_type']
            stats['by_method'][method] = stats['by_method'].get(method, 0) + 1

    # Print statistics
    print(f"\n=== STATISTICS FOR {year} ===")
    print(f"Total speakers: {len(speakers_df)}")
    print(f"Certain matches: {stats['certain']} ({100*stats['certain']/len(speakers_df):.1f}%)")
    print(f"Ambiguous: {stats['ambiguous']} ({100*stats['ambiguous']/len(speakers_df):.1f}%)")
    print(f"No match: {stats['no_match']} ({100*stats['no_match']/len(speakers_df):.1f}%)")
    print(f"Procedural: {stats['procedural']} ({100*stats['procedural']/len(speakers_df):.1f}%)")

    if stats['by_method']:
        print(f"\nMethods used for certain matches:")
        for method, count in sorted(stats['by_method'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {method}: {count}")

if __name__ == "__main__":
    # Test a few different years
    for year in [1920, 1950, 1980, 2000]:
        test_single_year(year)
        print("\n" + "="*70 + "\n")
