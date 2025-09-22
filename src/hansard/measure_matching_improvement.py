#!/usr/bin/env python3
"""
Measure the improvement from the advanced matching strategies
Compare baseline vs advanced matcher
"""

import pandas as pd
import numpy as np
from pathlib import Path
from mp_matcher import MPMatcher
from mp_matcher_advanced import AdvancedMPMatcher
from tqdm import tqdm
import json

def compare_matchers(sample_years=[1950, 1980, 2000]):
    """Compare baseline and advanced matchers"""

    print("Loading MP data...")
    mp_data = pd.read_parquet("data/house_members_gendered_updated.parquet")

    # Initialize matchers
    baseline = MPMatcher(mp_data)
    advanced = AdvancedMPMatcher(mp_data)

    results = {
        'baseline': {
            'matched': 0,
            'unmatched': 0,
            'procedural': 0,
            'ambiguous': 0
        },
        'advanced': {
            'certain': 0,
            'ambiguous': 0,
            'no_match': 0,
            'by_method': {}
        }
    }

    print(f"\nTesting on years: {sample_years}")

    for year in sample_years:
        print(f"\nProcessing {year}...")

        # Load speaker data
        speaker_file = Path(f"data/processed_fixed/metadata/speakers_{year}.parquet")
        if not speaker_file.exists():
            print(f"  No data for {year}")
            continue

        speakers_df = pd.read_parquet(speaker_file)

        # Sample for testing (use subset for speed)
        sample_size = min(1000, len(speakers_df))
        sample_df = speakers_df.sample(n=sample_size, random_state=42)

        print(f"  Testing {sample_size} speaker records...")

        for _, row in tqdm(sample_df.iterrows(), total=sample_size, leave=False):
            speaker = row['speaker_name']
            date = row.get('reference_date', f'{year}-01-01')
            chamber = row.get('chamber', 'Commons')

            # Baseline matching
            baseline_name, baseline_gender, baseline_type = baseline.match(speaker)
            if baseline_type == 'no_match':
                results['baseline']['unmatched'] += 1
            elif baseline_type == 'procedural':
                results['baseline']['procedural'] += 1
            else:
                results['baseline']['matched'] += 1

            # Advanced matching
            advanced_result = advanced.match_comprehensive(speaker, date, chamber)
            if advanced_result['match_type'] == 'no_match':
                results['advanced']['no_match'] += 1
            elif advanced_result['match_type'] == 'ambiguous':
                results['advanced']['ambiguous'] += 1
            else:
                results['advanced']['certain'] += 1
                method = advanced_result['match_type']
                results['advanced']['by_method'][method] = results['advanced']['by_method'].get(method, 0) + 1

    # Calculate improvement metrics
    total_baseline = sum(results['baseline'].values())
    total_advanced = results['advanced']['certain'] + results['advanced']['ambiguous'] + results['advanced']['no_match']

    print("\n" + "="*60)
    print("MATCHING IMPROVEMENT ANALYSIS")
    print("="*60)

    print("\n=== BASELINE MATCHER ===")
    print(f"Matched: {results['baseline']['matched']} ({100*results['baseline']['matched']/total_baseline:.1f}%)")
    print(f"Unmatched: {results['baseline']['unmatched']} ({100*results['baseline']['unmatched']/total_baseline:.1f}%)")
    print(f"Procedural: {results['baseline']['procedural']} ({100*results['baseline']['procedural']/total_baseline:.1f}%)")

    print("\n=== ADVANCED MATCHER ===")
    print(f"Certain matches: {results['advanced']['certain']} ({100*results['advanced']['certain']/total_advanced:.1f}%)")
    print(f"Ambiguous: {results['advanced']['ambiguous']} ({100*results['advanced']['ambiguous']/total_advanced:.1f}%)")
    print(f"No match: {results['advanced']['no_match']} ({100*results['advanced']['no_match']/total_advanced:.1f}%)")

    print("\n=== MATCH METHODS USED ===")
    for method, count in sorted(results['advanced']['by_method'].items(), key=lambda x: x[1], reverse=True):
        print(f"{method}: {count}")

    print("\n=== IMPROVEMENT SUMMARY ===")
    baseline_success = results['baseline']['matched'] / total_baseline
    advanced_certain = results['advanced']['certain'] / total_advanced
    improvement = (advanced_certain - baseline_success) / baseline_success * 100 if baseline_success > 0 else 0

    print(f"Baseline success rate: {100*baseline_success:.1f}%")
    print(f"Advanced certain match rate: {100*advanced_certain:.1f}%")
    print(f"Improvement: {improvement:+.1f}%")

    # Test specific improvements
    test_specific_cases()

def test_specific_cases():
    """Test specific improvements"""
    print("\n=== TESTING SPECIFIC IMPROVEMENTS ===\n")

    mp_data = pd.read_parquet("data/house_members_gendered_updated.parquet")
    advanced = AdvancedMPMatcher(mp_data)

    test_cases = [
        # Title resolution - Using dates when actually serving as PM
        ("The Prime Minister", "1979-06-01", "Expected: Margaret Thatcher"),  # After May 4 takeover
        ("The Prime Minister", "1997-06-01", "Expected: Tony Blair"),  # After May 2 takeover

        # Constituency
        ("the Member for Sedgefield", "1995-06-15", "Expected: Tony Blair"),

        # OCR corrections
        ("Mr. Bavies", "1950-05-26", "Should correct to Davies"),
        ("Mrs. 0'Brien", "1970-06-15", "Should correct O'Brien"),

        # Initial matching
        ("Mr. W. Churchill", "1940-05-10", "Should match Winston"),
    ]

    for speaker, date, expected in test_cases:
        result = advanced.match_comprehensive(speaker, date, "Commons")
        print(f"Test: {speaker} ({date})")
        print(f"  {expected}")
        if result['final_match']:
            print(f"  Result: ✓ {result['final_match']} (conf: {result['confidence']:.2f})")
        elif result['match_type'] == 'ambiguous':
            print(f"  Result: ⚠ Ambiguous ({result.get('ambiguity_count')} candidates)")
        else:
            print(f"  Result: ✗ No match")
        print()

if __name__ == "__main__":
    compare_matchers([1920, 1950, 1980, 2000])