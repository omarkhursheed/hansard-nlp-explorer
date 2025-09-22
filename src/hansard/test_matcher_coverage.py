#!/usr/bin/env python3
"""
Test the MP matcher on real debate data to check coverage
"""

import pandas as pd
from pathlib import Path
from mp_matcher import MPMatcher

def test_coverage():
    """Test matcher coverage on real debate data"""

    print("Loading MP gender data...")
    mp_data = pd.read_parquet("data/house_members_gendered_updated.parquet")
    matcher = MPMatcher(mp_data)

    print(f"Loaded {len(mp_data)} MP records")
    print(f"Unique MPs: {mp_data['person_name'].nunique()}")

    # Load conversation turns
    print("\nLoading conversation turns data...")
    turns_df = pd.read_parquet("debate_datasets/conversation_turns_1950_sample.parquet")

    print(f"Loaded {len(turns_df)} conversation turns")
    print(f"Unique speakers: {turns_df['speaker'].nunique()}")

    # Get unique speakers
    unique_speakers = turns_df['speaker'].unique()

    # Test matching
    print("\nTesting matcher...")
    results = matcher.match_batch(unique_speakers)

    # Analyze results
    stats = matcher.get_match_statistics(unique_speakers)

    print("\n=== MATCHING STATISTICS ===")
    print(f"Total speakers: {stats['total']}")
    print(f"Matched to MPs: {stats['matched']} ({100*stats['match_rate']:.1f}%)")
    print(f"Procedural: {stats['procedural']}")
    print(f"Unmatched: {stats['unmatched']}")

    print("\n=== MATCH TYPES ===")
    for match_type, count in sorted(stats['match_types'].items()):
        print(f"{match_type}: {count}")

    # Show some unmatched speakers
    unmatched = []
    for speaker, result in zip(unique_speakers, results):
        if result[2] == 'no_match':
            unmatched.append(speaker)

    if unmatched:
        print(f"\n=== SAMPLE UNMATCHED SPEAKERS (first 20) ===")
        for speaker in unmatched[:20]:
            print(f"  - {speaker}")

    # Analyze gender distribution of matched speakers
    matched_genders = {'M': 0, 'F': 0}
    for result in results:
        if result[1]:
            matched_genders[result[1]] += 1

    print(f"\n=== GENDER DISTRIBUTION OF MATCHED SPEAKERS ===")
    total_gendered = sum(matched_genders.values())
    if total_gendered > 0:
        print(f"Male: {matched_genders['M']} ({100*matched_genders['M']/total_gendered:.1f}%)")
        print(f"Female: {matched_genders['F']} ({100*matched_genders['F']/total_gendered:.1f}%)")

    # Check coverage by debate
    print("\n=== COVERAGE BY DEBATE ===")
    debate_coverage = []

    for debate_id in turns_df['debate_id'].unique()[:10]:  # First 10 debates
        debate_turns = turns_df[turns_df['debate_id'] == debate_id]
        debate_speakers = debate_turns['speaker'].unique()
        debate_results = matcher.match_batch(debate_speakers)

        matched = sum(1 for r in debate_results if r[2] not in ['no_match', 'procedural'])
        total = len(debate_speakers)

        debate_coverage.append({
            'debate_id': debate_id,
            'total_speakers': total,
            'matched': matched,
            'coverage': matched/total if total > 0 else 0
        })

    coverage_df = pd.DataFrame(debate_coverage)
    print(f"Average coverage across debates: {100*coverage_df['coverage'].mean():.1f}%")
    print(f"Min coverage: {100*coverage_df['coverage'].min():.1f}%")
    print(f"Max coverage: {100*coverage_df['coverage'].max():.1f}%")

    return matcher, turns_df, results

if __name__ == "__main__":
    matcher, turns_df, results = test_coverage()