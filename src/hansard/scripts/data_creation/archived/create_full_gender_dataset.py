#!/usr/bin/env python3
"""
Create FULL gender analysis dataset for all available years
This will process all 201 years of data (1803-2005)
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).resolve().parents[4]  # Up to hansard-nlp-explorer
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
from hansard.scripts.matching.mp_matcher_corrected import CorrectedMPMatcher
from hansard.utils.path_config import Paths
import json
from tqdm import tqdm
import hashlib
from datetime import datetime

def process_all_years():
    """Process ALL available years"""

    print("=" * 70)
    print("FULL GENDER ANALYSIS DATASET CREATION")
    print("=" * 70)
    print("\nWARNING: This will process ALL years (1803-2005)")
    print("Estimated time: 1-2 hours depending on your machine")
    print("=" * 70)

    # Load matcher once
    print("\nLoading MP matcher...")
    mp_data = pd.read_parquet(Paths.get_data_dir() / "house_members_gendered_updated.parquet")
    matcher = CorrectedMPMatcher(mp_data)

    # Get all available years
    metadata_dir = Paths.get_processed_data_dir() / "metadata"
    all_years = sorted([
        int(f.stem.split('_')[1])
        for f in metadata_dir.glob("debates_*.parquet")
        if 'master' not in f.stem
    ])

    print(f"Found {len(all_years)} years of data")
    print(f"Years range: {min(all_years)} - {max(all_years)}")

    # Create output directory
    output_dir = Path("gender_analysis_data_FULL")
    output_dir.mkdir(exist_ok=True)

    all_debates = []
    all_female_mps = set()
    all_male_mps = set()  # Track male MPs too!

    # Statistics tracking
    stats = {
        'total_debates_processed': 0,
        'debates_with_confirmed_mps': 0,
        'debates_with_female': 0,
        'by_decade': {}
    }

    # Process each year
    for year in tqdm(all_years, desc="Processing years"):
        try:
            debates_df = pd.read_parquet(metadata_dir / f"debates_{year}.parquet")
            stats['total_debates_processed'] += len(debates_df)

            year_debates = []

            for idx, debate in debates_df.iterrows():
                speakers = debate.get('speakers', [])
                date = debate.get('reference_date', f'{year}-01-01')
                chamber = debate.get('chamber', 'Commons')

                if not isinstance(speakers, (list, np.ndarray)) or len(speakers) == 0:
                    continue

                # Match speakers
                matched_mps = []
                female_mps = []
                male_mps = []
                ambiguous = 0
                unmatched = 0

                for speaker in speakers:
                    if pd.notna(speaker):
                        result = matcher.match_comprehensive(str(speaker), date, chamber)

                        if result['match_type'] in ['temporal_unique', 'title', 'constituency']:
                            if result.get('confidence', 0) >= 0.7:
                                matched_mps.append(result.get('final_match'))
                                if result.get('gender') == 'F':
                                    female_mps.append(result.get('final_match'))
                                    all_female_mps.add(result.get('final_match'))
                                elif result.get('gender') == 'M':
                                    male_mps.append(result.get('final_match'))
                                    all_male_mps.add(result.get('final_match'))  # Track unique male MPs too!
                        elif result['match_type'] == 'ambiguous':
                            ambiguous += 1
                        else:
                            unmatched += 1

                # Only keep debates with at least one confirmed MP
                if matched_mps:
                    debate_id = hashlib.md5(
                        f"{debate['file_path']}_{date}".encode()
                    ).hexdigest()[:16]

                    decade = (year // 10) * 10

                    debate_record = {
                        'debate_id': debate_id,
                        'year': year,
                        'decade': decade,
                        'reference_date': date,
                        'chamber': chamber,
                        'title': debate.get('title'),
                        'topic': debate.get('debate_topic'),
                        'total_speakers': len(speakers),
                        'confirmed_mps': len(matched_mps),
                        'female_mps': len(female_mps),
                        'male_mps': len(male_mps),
                        'has_female': len(female_mps) > 0,
                        'has_male': len(male_mps) > 0,  # Add male flag for symmetry
                        'female_names': female_mps,
                        'male_names': male_mps,  # STORE MALE NAMES TOO!
                        'ambiguous_speakers': ambiguous,
                        'unmatched_speakers': unmatched,
                        'word_count': debate.get('word_count', 0)
                    }

                    year_debates.append(debate_record)
                    stats['debates_with_confirmed_mps'] += 1

                    if female_mps:
                        stats['debates_with_female'] += 1

                    # Update decade stats
                    if decade not in stats['by_decade']:
                        stats['by_decade'][decade] = {
                            'debates_with_mps': 0,
                            'debates_with_female': 0
                        }
                    stats['by_decade'][decade]['debates_with_mps'] += 1
                    if female_mps:
                        stats['by_decade'][decade]['debates_with_female'] += 1

            # Save year data
            if year_debates:
                year_df = pd.DataFrame(year_debates)
                year_df.to_parquet(output_dir / f"debates_{year}_with_mps.parquet")
                all_debates.extend(year_debates)

        except Exception as e:
            print(f"Error processing {year}: {e}")
            continue

    # Save combined dataset
    if all_debates:
        print(f"\nSaving {len(all_debates)} total debates...")
        combined_df = pd.DataFrame(all_debates)
        combined_df.to_parquet(output_dir / "ALL_debates_with_confirmed_mps.parquet")

        # Save metadata
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'years_processed': len(all_years),
            'year_range': f"{min(all_years)}-{max(all_years)}",
            'statistics': stats,
            'total_female_mps_identified': len(all_female_mps),
            'total_male_mps_identified': len(all_male_mps),  # Add male count
            'female_mps_list': sorted(list(all_female_mps)),
            'male_mps_list': sorted(list(all_male_mps))  # Add male list
        }

        with open(output_dir / "dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Print summary
        print("\n" + "=" * 70)
        print("DATASET CREATION COMPLETE")
        print("=" * 70)
        print(f"\nTotal debates processed: {stats['total_debates_processed']:,}")
        print(f"Debates with confirmed MPs: {stats['debates_with_confirmed_mps']:,} ({100*stats['debates_with_confirmed_mps']/stats['total_debates_processed']:.1f}%)")
        print(f"Debates with female MPs: {stats['debates_with_female']:,} ({100*stats['debates_with_female']/stats['debates_with_confirmed_mps']:.1f}%)")
        print(f"Unique female MPs identified: {len(all_female_mps)}")
        print(f"Unique male MPs identified: {len(all_male_mps)}")  # Print male count too

        # Decade summary
        print("\nFemale participation by decade:")
        for decade in sorted(stats['by_decade'].keys()):
            decade_stats = stats['by_decade'][decade]
            total = decade_stats['debates_with_mps']
            female = decade_stats['debates_with_female']
            if total > 0:
                print(f"  {decade}s: {female:,}/{total:,} debates ({100*female/total:.1f}%)")

        print(f"\nData saved to: {output_dir}/")
        print("\nMain file: ALL_debates_with_confirmed_mps.parquet")
        print("Year files: debates_YYYY_with_mps.parquet")

if __name__ == "__main__":
    import time
    start_time = time.time()

    process_all_years()

    elapsed = time.time() - start_time
    print(f"\nTotal processing time: {elapsed/60:.1f} minutes")
