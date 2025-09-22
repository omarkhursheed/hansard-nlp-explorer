#!/usr/bin/env python3
"""
Process specific decades for gender analysis
Useful for targeted research periods
"""

import pandas as pd
import numpy as np
from pathlib import Path
from mp_matcher_corrected import CorrectedMPMatcher
import json
from tqdm import tqdm
import hashlib
import sys

def process_decade(decade_start):
    """Process a single decade"""

    years = list(range(decade_start, min(decade_start + 10, 2006)))

    print(f"\nProcessing {decade_start}s ({len(years)} years)...")

    # Load matcher
    mp_data = pd.read_parquet("data/house_members_gendered_updated.parquet")
    matcher = CorrectedMPMatcher(mp_data)

    output_dir = Path(f"gender_analysis_data_{decade_start}s")
    output_dir.mkdir(exist_ok=True)

    all_debates = []
    female_mps = set()

    for year in tqdm(years, desc=f"Processing {decade_start}s"):
        try:
            debates_df = pd.read_parquet(f"data/processed_fixed/metadata/debates_{year}.parquet")

            for idx, debate in debates_df.iterrows():
                speakers = debate.get('speakers', [])
                date = debate.get('reference_date', f'{year}-01-01')
                chamber = debate.get('chamber', 'Commons')

                if not isinstance(speakers, (list, np.ndarray)) or len(speakers) == 0:
                    continue

                # Match speakers
                matched_mps = []
                female_mp_list = []
                male_mps = []

                for speaker in speakers:
                    if pd.notna(speaker):
                        result = matcher.match_comprehensive(str(speaker), date, chamber)

                        if result['match_type'] in ['temporal_unique', 'title', 'constituency']:
                            if result.get('confidence', 0) >= 0.7:
                                matched_mps.append(result.get('final_match'))
                                if result.get('gender') == 'F':
                                    female_mp_list.append(result.get('final_match'))
                                    female_mps.add(result.get('final_match'))
                                elif result.get('gender') == 'M':
                                    male_mps.append(result.get('final_match'))

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
                        'confirmed_mps': len(matched_mps),
                        'female_mps': len(female_mp_list),
                        'male_mps': len(male_mps),
                        'has_female': len(female_mp_list) > 0,
                        'female_names': female_mp_list,
                        'word_count': debate.get('word_count', 0)
                    }

                    all_debates.append(debate_record)

        except Exception as e:
            print(f"Error processing {year}: {e}")

    # Save decade data
    if all_debates:
        df = pd.DataFrame(all_debates)
        df.to_parquet(output_dir / f"debates_{decade_start}s.parquet")

        print(f"\n{decade_start}s Summary:")
        print(f"  Debates with confirmed MPs: {len(df)}")
        print(f"  Debates with female MPs: {df['has_female'].sum()} ({100*df['has_female'].mean():.1f}%)")
        print(f"  Female MPs identified: {len(female_mps)}")
        if female_mps:
            print(f"  Including: {', '.join(list(female_mps)[:5])}...")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Process specific decade
        decade = int(sys.argv[1])
        process_decade(decade)
    else:
        # Process key decades
        key_decades = [
            1910,  # Suffrage movement
            1920,  # Post-WWI, first female MPs
            1940,  # WWII
            1950,  # Post-war
            1970,  # Women's lib movement
            1980,  # Thatcher era
            1990,  # Major era
            2000   # Blair era, modern parliament
        ]

        print("Processing key decades for gender analysis...")
        for decade in key_decades:
            process_decade(decade)