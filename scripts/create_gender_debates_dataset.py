#!/usr/bin/env python3
"""
Create Simplified Debate-Level Gender Dataset

Extracts debate-level data from gender_analysis_enhanced/ with a SIMPLE schema
matching processed_fixed/ structure + gender metadata.

Input:  data-hansard/gender_analysis_enhanced/debates_YYYY_enhanced.parquet (40 cols, nested)
Output: data-hansard/derived/gender_debates/debates_YYYY.parquet (10 cols, flat)

Schema (matches processed_fixed + gender):
    debate_id: str
    year: int
    date: str
    title: str
    chamber: str
    full_text: str              # Complete debate text
    speakers: list[str]         # List of speaker names
    word_count: int
    has_female: bool            # Has female MPs
    has_male: bool              # Has male MPs
    female_mps: int             # Count of female MPs
    male_mps: int               # Count of male MPs
    speaker_genders: dict       # {speaker_name: 'm'/'f'}

Usage:
    python scripts/create_gender_debates_dataset.py [--years START-END] [--test]
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'hansard'))
from utils.path_config import Paths


def simplify_debate(row, year):
    """
    Convert complex gender_analysis_enhanced row to simple schema.

    Args:
        row: DataFrame row from gender_analysis_enhanced
        year: Year of debate

    Returns:
        Dict with simplified debate data
    """
    # Build speaker gender map
    speaker_genders = {}
    speaker_details = row.get('speaker_details', [])

    if isinstance(speaker_details, (list, np.ndarray)):
        for detail in speaker_details:
            if isinstance(detail, dict):
                name = detail.get('original_name', '')
                gender = detail.get('gender', '').lower()
                if name and gender in ['m', 'f']:
                    speaker_genders[name] = gender

    # Extract speaker names from speech segments
    speakers = []
    if 'speech_segments' in row and isinstance(row['speech_segments'], (list, np.ndarray)):
        for segment in row['speech_segments']:
            if isinstance(segment, dict):
                speaker = segment.get('speaker', '')
                if speaker and speaker not in speakers:
                    speakers.append(speaker)

    # Create simplified record
    debate = {
        'debate_id': row.get('debate_id', f"debate_{year}_{row.name}"),
        'year': year,
        'date': row.get('reference_date', ''),
        'title': row.get('title', ''),
        'chamber': row.get('chamber', ''),
        'full_text': row.get('debate_text', ''),
        'speakers': speakers,
        'word_count': row.get('word_count', len(str(row.get('debate_text', '')).split())),
        'has_female': bool(row.get('has_female', False)),
        'has_male': bool(row.get('has_male', False)),
        'female_mps': int(row.get('female_mps', 0)),
        'male_mps': int(row.get('male_mps', 0)),
        'speaker_genders': speaker_genders
    }

    return debate


def process_year(year, input_dir, output_dir, force=False):
    """Process a single year"""
    input_file = input_dir / f"debates_{year}_enhanced.parquet"
    output_file = output_dir / f"debates_{year}.parquet"

    if not input_file.exists():
        print(f"  {year}: No input file found")
        return 0

    if output_file.exists() and not force:
        print(f"  {year}: Already exists (use --force to overwrite)")
        return 0

    try:
        # Load complex data
        df_complex = pd.read_parquet(input_file)

        # Simplify each debate
        debates = []
        for idx, row in df_complex.iterrows():
            debate = simplify_debate(row, year)
            debates.append(debate)

        if not debates:
            print(f"  {year}: No debates extracted")
            return 0

        # Create DataFrame with simple schema
        df_simple = pd.DataFrame(debates)
        df_simple.to_parquet(output_file, index=False)

        # Stats
        male_only = sum(1 for d in debates if d['has_male'] and not d['has_female'])
        female_only = sum(1 for d in debates if d['has_female'] and not d['has_male'])
        mixed = sum(1 for d in debates if d['has_female'] and d['has_male'])

        print(f"  {year}: {len(debates):,} debates "
              f"(male-only: {male_only}, mixed: {mixed}, female-only: {female_only})")

        return len(debates)

    except Exception as e:
        print(f"  {year}: Error - {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description='Create simplified debate-level gender dataset'
    )

    parser.add_argument('--years', type=str, help='Year range (e.g., 1990-2000)')
    parser.add_argument('--test', action='store_true', help='Test mode: 1995-1997')
    parser.add_argument('--force', action='store_true', help='Overwrite existing files')

    args = parser.parse_args()

    # Determine year range
    if args.test:
        start_year, end_year = 1995, 1997
        print("TEST MODE: Processing 1995-1997\n")
    elif args.years:
        try:
            parts = args.years.split('-')
            start_year, end_year = int(parts[0]), int(parts[1])
        except:
            print(f"Error: Invalid year range: {args.years}")
            sys.exit(1)
    else:
        start_year, end_year = 1803, 2005

    print("="*80)
    print("CREATE SIMPLIFIED DEBATE-LEVEL GENDER DATASET")
    print("="*80)
    print(f"\nProcessing years: {start_year}-{end_year}")
    print(f"Force overwrite: {args.force}\n")

    # Setup directories
    input_dir = Paths.GENDER_ENHANCED_DATA
    output_dir = Paths.DATA_DIR / 'derived' / 'gender_debates'

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}\n")

    # Process years
    print("Processing years:")
    print("-"*80)

    total_debates = 0
    years_processed = 0

    for year in range(start_year, end_year + 1):
        count = process_year(year, input_dir, output_dir, args.force)
        if count > 0:
            total_debates += count
            years_processed += 1

    # Summary
    print("-"*80)
    print(f"\nSummary:")
    print(f"  Years processed: {years_processed}")
    print(f"  Total debates: {total_debates:,}")
    print(f"  Output: {output_dir}")

    if total_debates > 0:
        print(f"\n✓ Simplified debate-level dataset created!")
        print(f"\nSchema: debate_id, year, title, chamber, full_text, speakers[],")
        print(f"        has_female, has_male, female_mps, male_mps, speaker_genders{{}}")
        print(f"\nThis matches processed_fixed/ schema + gender metadata")
        print(f"\nNext: Update loaders to support --dataset gender-debates")
    else:
        print(f"\n⚠ No debates extracted")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
