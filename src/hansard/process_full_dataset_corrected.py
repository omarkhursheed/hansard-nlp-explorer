#!/usr/bin/env python3
"""
Process FULL Hansard dataset with the corrected MP matcher
Uses verified dates and improved matching strategies
"""

import pandas as pd
import numpy as np
from pathlib import Path
from mp_matcher_corrected import CorrectedMPMatcher
import json
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm
import multiprocessing as mp

def process_single_year(args: Tuple[int, Path]) -> Dict:
    """Process a single year with corrected matching"""
    year, mp_data_path = args

    try:
        # Load MP data and create corrected matcher
        mp_data = pd.read_parquet(mp_data_path)
        matcher = CorrectedMPMatcher(mp_data)

        # Check if files exist
        speaker_file = Path(f"data/processed_fixed/metadata/speakers_{year}.parquet")
        debate_file = Path(f"data/processed_fixed/metadata/debates_{year}.parquet")

        if not speaker_file.exists():
            return {
                'year': year,
                'status': 'no_data',
                'speakers_total': 0
            }

        # Load data
        speakers_df = pd.read_parquet(speaker_file)
        debates_df = pd.read_parquet(debate_file) if debate_file.exists() else None

        # Process speakers with comprehensive matching
        results = []
        stats = {
            'total': 0,
            'certain': 0,
            'ambiguous': 0,
            'no_match': 0,
            'procedural': 0,
            'by_method': {},
            'by_confidence': {'high': 0, 'medium': 0, 'low': 0}
        }

        for _, row in speakers_df.iterrows():
            speaker = row['speaker_name']
            # Use debate date if available, otherwise use year
            date = row.get('reference_date', f'{year}-01-01')
            chamber = row.get('chamber', 'Commons')

            # Use comprehensive matching
            match_result = matcher.match_comprehensive(speaker, date, chamber)

            # Track statistics
            stats['total'] += 1

            if match_result['match_type'] == 'procedural':
                stats['procedural'] += 1
            elif match_result['match_type'] == 'ambiguous':
                stats['ambiguous'] += 1
            elif match_result['match_type'] == 'no_match':
                stats['no_match'] += 1
            else:
                stats['certain'] += 1
                method = match_result['match_type']
                stats['by_method'][method] = stats['by_method'].get(method, 0) + 1

                # Track confidence levels
                conf = match_result['confidence']
                if conf >= 0.9:
                    stats['by_confidence']['high'] += 1
                elif conf >= 0.7:
                    stats['by_confidence']['medium'] += 1
                else:
                    stats['by_confidence']['low'] += 1

            # Store result
            results.append({
                'year': year,
                'speaker_id': row['speaker_id'],
                'original_speaker': speaker,
                'matched_mp': match_result.get('final_match'),
                'gender': match_result.get('gender'),
                'confidence': match_result.get('confidence', 0.0),
                'match_type': match_result['match_type'],
                'ambiguity_count': match_result.get('ambiguity_count', 0)
            })

        return {
            'year': year,
            'status': 'success',
            'results': results,
            'stats': stats
        }

    except Exception as e:
        return {
            'year': year,
            'status': 'error',
            'error': str(e)
        }

def main():
    """Process full dataset with corrected matcher"""

    print("=" * 70)
    print("FULL DATASET PROCESSING WITH CORRECTED MATCHER")
    print("=" * 70)

    # Setup paths
    mp_data_path = Path("data/house_members_gendered_updated.parquet")
    output_dir = Path("data_filtered_by_actual_mp_CORRECTED")
    output_dir.mkdir(exist_ok=True)

    # Get all available years
    metadata_dir = Path("data/processed_fixed/metadata")
    available_years = sorted([
        int(f.stem.split('_')[1])
        for f in metadata_dir.glob("speakers_*.parquet")
        if 'master' not in f.stem
    ])

    print(f"\nFound {len(available_years)} years of data")
    print(f"Years range: {min(available_years)} - {max(available_years)}")

    # Process in parallel
    print(f"\nProcessing with {mp.cpu_count()} CPUs...")

    # Prepare arguments
    args_list = [(year, mp_data_path) for year in available_years]

    # Process with progress bar
    all_results = []
    all_stats = {
        'total_speakers': 0,
        'certain_matches': 0,
        'ambiguous_matches': 0,
        'no_matches': 0,
        'procedural': 0,
        'methods_used': {},
        'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0}
    }

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = {executor.submit(process_single_year, args): args[0]
                  for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="Processing years"):
            year = futures[future]
            result = future.result()

            if result['status'] == 'success':
                # Aggregate stats
                stats = result['stats']
                all_stats['total_speakers'] += stats['total']
                all_stats['certain_matches'] += stats['certain']
                all_stats['ambiguous_matches'] += stats['ambiguous']
                all_stats['no_matches'] += stats['no_match']
                all_stats['procedural'] += stats['procedural']

                for method, count in stats['by_method'].items():
                    all_stats['methods_used'][method] = \
                        all_stats['methods_used'].get(method, 0) + count

                for level, count in stats['by_confidence'].items():
                    all_stats['confidence_distribution'][level] += count

                # Save year results
                if result.get('results'):
                    year_df = pd.DataFrame(result['results'])
                    year_df.to_parquet(output_dir / f"matched_speakers_{year}.parquet")
                    all_results.extend(result['results'])

    # Save combined results
    print("\nSaving combined results...")
    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_df.to_parquet(output_dir / "all_matched_speakers.parquet")

        # Create high-confidence subset
        high_conf_df = combined_df[combined_df['confidence'] >= 0.9]
        high_conf_df.to_parquet(output_dir / "high_confidence_matches.parquet")

    # Print final statistics
    print("\n" + "=" * 70)
    print("FINAL STATISTICS - CORRECTED MATCHER")
    print("=" * 70)

    total = all_stats['total_speakers']
    if total > 0:
        print(f"\nTotal speaker records: {total:,}")
        print(f"\nMatch Results:")
        print(f"  Certain matches: {all_stats['certain_matches']:,} ({100*all_stats['certain_matches']/total:.1f}%)")
        print(f"  Ambiguous: {all_stats['ambiguous_matches']:,} ({100*all_stats['ambiguous_matches']/total:.1f}%)")
        print(f"  No match: {all_stats['no_matches']:,} ({100*all_stats['no_matches']/total:.1f}%)")
        print(f"  Procedural: {all_stats['procedural']:,} ({100*all_stats['procedural']/total:.1f}%)")

        print(f"\nMatching Methods Used:")
        for method, count in sorted(all_stats['methods_used'].items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"  {method}: {count:,}")

        print(f"\nConfidence Distribution (of certain matches):")
        conf_dist = all_stats['confidence_distribution']
        if all_stats['certain_matches'] > 0:
            print(f"  High (≥0.9): {conf_dist['high']:,} ({100*conf_dist['high']/all_stats['certain_matches']:.1f}%)")
            print(f"  Medium (0.7-0.9): {conf_dist['medium']:,} ({100*conf_dist['medium']/all_stats['certain_matches']:.1f}%)")
            print(f"  Low (<0.7): {conf_dist['low']:,} ({100*conf_dist['low']/all_stats['certain_matches']:.1f}%)")

    # Save statistics
    with open(output_dir / "matching_statistics.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nResults saved to {output_dir}/")

if __name__ == "__main__":
    main()