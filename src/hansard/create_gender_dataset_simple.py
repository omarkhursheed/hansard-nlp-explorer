#!/usr/bin/env python3
"""
Simplified version to create gender analysis dataset
Processes debates and identifies those with confirmed MPs
"""

import pandas as pd
import numpy as np
from pathlib import Path
from mp_matcher_corrected import CorrectedMPMatcher
import json
from tqdm import tqdm
import hashlib

def process_year(year=1950):
    """Process a single year as a test"""

    print(f"\nProcessing year {year}...")

    # Load matcher
    mp_data = pd.read_parquet("data/house_members_gendered_updated.parquet")
    matcher = CorrectedMPMatcher(mp_data)

    # Load debates and speakers
    debates_df = pd.read_parquet(f"data/processed_fixed/metadata/debates_{year}.parquet")
    speakers_df = pd.read_parquet(f"data/processed_fixed/metadata/speakers_{year}.parquet")

    print(f"Found {len(debates_df)} debates, {len(speakers_df)} speaker records")

    # Process each debate
    debates_with_mps = []
    debates_with_female = []

    for idx, debate in tqdm(debates_df.iterrows(), total=len(debates_df), desc="Matching debates"):
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
                        elif result.get('gender') == 'M':
                            male_mps.append(result.get('final_match'))
                elif result['match_type'] == 'ambiguous':
                    ambiguous += 1
                else:
                    unmatched += 1

        # Only keep debates with at least one confirmed MP
        if matched_mps:
            debate_id = hashlib.md5(
                f"{debate['file_path']}_{date}".encode()
            ).hexdigest()[:16]

            debate_record = {
                'debate_id': debate_id,
                'year': year,
                'reference_date': date,
                'chamber': chamber,
                'title': debate.get('title'),
                'topic': debate.get('debate_topic'),
                'total_speakers': len(speakers),
                'confirmed_mps': len(matched_mps),
                'female_mps': len(female_mps),
                'male_mps': len(male_mps),
                'has_female': len(female_mps) > 0,
                'female_names': female_mps,
                'ambiguous_speakers': ambiguous,
                'unmatched_speakers': unmatched,
                'word_count': debate.get('word_count', 0)
            }

            debates_with_mps.append(debate_record)
            if female_mps:
                debates_with_female.append(debate_record)

    # Create output directory
    output_dir = Path("gender_analysis_data")
    output_dir.mkdir(exist_ok=True)

    # Save results
    if debates_with_mps:
        df = pd.DataFrame(debates_with_mps)
        df.to_parquet(output_dir / f"debates_{year}_with_mps.parquet")
        print(f"\nSaved {len(df)} debates with confirmed MPs")

        # Print statistics
        print(f"\nStatistics for {year}:")
        print(f"Total debates: {len(debates_df)}")
        print(f"Debates with confirmed MPs: {len(debates_with_mps)} ({100*len(debates_with_mps)/len(debates_df):.1f}%)")
        print(f"Debates with female MPs: {len(debates_with_female)} ({100*len(debates_with_female)/len(debates_df):.1f}%)")

        if debates_with_female:
            print(f"\nFemale MPs found:")
            all_female = set()
            for d in debates_with_female:
                all_female.update(d['female_names'])
            for name in sorted(all_female):
                print(f"  - {name}")

        # Gender breakdown
        total_confirmed = df['confirmed_mps'].sum()
        total_female = df['female_mps'].sum()
        total_male = df['male_mps'].sum()

        print(f"\nSpeaker gender breakdown:")
        print(f"Total confirmed MPs: {total_confirmed}")
        print(f"Female MPs: {total_female} ({100*total_female/total_confirmed:.1f}%)")
        print(f"Male MPs: {total_male} ({100*total_male/total_confirmed:.1f}%)")

        return df

    return None

def main():
    """Process multiple years"""

    print("=" * 70)
    print("GENDER ANALYSIS DATASET CREATION")
    print("=" * 70)

    years = [1920, 1950, 1980, 2000]
    all_debates = []

    for year in years:
        df = process_year(year)
        if df is not None:
            all_debates.append(df)

    # Combine all years
    if all_debates:
        combined = pd.concat(all_debates, ignore_index=True)
        output_dir = Path("gender_analysis_data")
        combined.to_parquet(output_dir / "all_debates_with_confirmed_mps.parquet")

        print("\n" + "=" * 70)
        print("OVERALL STATISTICS")
        print("=" * 70)

        print(f"\nTotal debates with confirmed MPs: {len(combined)}")
        print(f"Debates with female participation: {combined['has_female'].sum()} ({100*combined['has_female'].mean():.1f}%)")
        print(f"Average confirmed MPs per debate: {combined['confirmed_mps'].mean():.1f}")

        # Female participation by year
        print(f"\nFemale participation by year:")
        for year in sorted(combined['year'].unique()):
            year_data = combined[combined['year'] == year]
            pct = 100 * year_data['has_female'].mean()
            print(f"  {year}: {pct:.1f}% of debates")

if __name__ == "__main__":
    main()