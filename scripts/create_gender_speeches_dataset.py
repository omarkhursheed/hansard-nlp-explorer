#!/usr/bin/env python3
"""
Create Derived Gender Speeches Dataset

Extracts speech_segments from gender_analysis_enhanced/ and creates a flat,
speech-level dataset for easier analysis.

Input:  data-hansard/gender_analysis_enhanced/debates_YYYY_enhanced.parquet (nested)
Output: data-hansard/derived/gender_speeches/speeches_YYYY.parquet (flat)

Schema:
    speech_id: str          # Unique ID (debate_id + speech_index)
    debate_id: str          # Links back to original debate
    year: int               # Year of speech
    date: str               # Date of debate
    speaker: str            # Speaker name
    gender: str             # 'm' or 'f'
    text: str               # Speech text
    word_count: int         # Words in speech
    chamber: str            # Chamber (Commons/Lords)

Usage:
    python scripts/create_gender_speeches_dataset.py [--years START-END] [--test]

Options:
    --years START-END    Process only these years (e.g., 1990-2000)
    --test               Test mode: process only 3 years (1995-1997)
    --force              Overwrite existing files
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import re

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'hansard'))
from utils.path_config import Paths


def extract_speeches_from_debate(row, year):
    """
    Extract individual speeches from a debate row.

    Args:
        row: DataFrame row from gender_analysis_enhanced
        year: Year of the debate

    Returns:
        List of speech dicts
    """
    speeches = []

    if 'speech_segments' not in row or row['speech_segments'] is None:
        return speeches

    # Build speaker gender map
    speaker_gender_map = {}
    speaker_details = row.get('speaker_details', [])

    if isinstance(speaker_details, (list, np.ndarray)):
        for detail in speaker_details:
            if isinstance(detail, dict):
                name = detail.get('original_name', '')
                gender = detail.get('gender', '').lower()
                if name and gender in ['m', 'f']:
                    speaker_gender_map[name] = gender

    # Extract speech segments
    segments = row['speech_segments']
    if not isinstance(segments, (list, np.ndarray)):
        return speeches

    debate_id = row.get('debate_id', f"debate_{year}_{row.name}")

    for speech_idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue

        speaker = segment.get('speaker', '')
        text = segment.get('text', '')

        if not text:
            continue

        # Map to gender
        gender = speaker_gender_map.get(speaker)
        if not gender:
            # Try fuzzy matching
            for orig_name, g in speaker_gender_map.items():
                if speaker in orig_name or orig_name in speaker:
                    gender = g
                    break

        if not gender:
            continue  # Skip speeches without gender attribution

        # Create speech record
        speech = {
            'speech_id': f"{debate_id}_speech_{speech_idx}",
            'debate_id': debate_id,
            'year': year,
            'date': row.get('reference_date', ''),
            'speaker': speaker,
            'gender': gender,
            'text': text,
            'word_count': len(text.split()),
            'chamber': row.get('chamber', '')
        }

        speeches.append(speech)

    return speeches


def process_year(year, input_dir, output_dir, force=False):
    """
    Process a single year of data.

    Args:
        year: Year to process
        input_dir: Input directory (gender_analysis_enhanced)
        output_dir: Output directory (derived/gender_speeches)
        force: Overwrite existing files

    Returns:
        Number of speeches extracted
    """
    input_file = input_dir / f"debates_{year}_enhanced.parquet"
    output_file = output_dir / f"speeches_{year}.parquet"

    if not input_file.exists():
        print(f"  {year}: No input file found")
        return 0

    if output_file.exists() and not force:
        print(f"  {year}: Already exists (use --force to overwrite)")
        return 0

    try:
        # Load debate data
        df = pd.read_parquet(input_file)

        # Extract all speeches
        all_speeches = []
        for idx, row in df.iterrows():
            speeches = extract_speeches_from_debate(row, year)
            all_speeches.extend(speeches)

        if not all_speeches:
            print(f"  {year}: No speeches extracted (no gender attribution)")
            return 0

        # Create DataFrame and save
        speeches_df = pd.DataFrame(all_speeches)
        speeches_df.to_parquet(output_file, index=False)

        male_count = len(speeches_df[speeches_df['gender'] == 'm'])
        female_count = len(speeches_df[speeches_df['gender'] == 'f'])

        print(f"  {year}: {len(speeches_df):,} speeches extracted "
              f"({male_count:,} male, {female_count:,} female)")

        return len(speeches_df)

    except Exception as e:
        print(f"  {year}: Error - {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description='Create derived gender speeches dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--years', type=str,
                       help='Year range (e.g., 1990-2000)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: process only 1995-1997')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing files')

    args = parser.parse_args()

    # Determine year range
    if args.test:
        start_year, end_year = 1995, 1997
        print("TEST MODE: Processing 1995-1997 only\n")
    elif args.years:
        try:
            parts = args.years.split('-')
            start_year = int(parts[0])
            end_year = int(parts[1])
        except:
            print(f"Error: Invalid year range: {args.years}")
            sys.exit(1)
    else:
        start_year, end_year = 1803, 2005

    print("="*80)
    print("CREATE DERIVED GENDER SPEECHES DATASET")
    print("="*80)
    print(f"\nProcessing years: {start_year}-{end_year}")
    print(f"Force overwrite: {args.force}\n")

    # Setup directories
    input_dir = Paths.GENDER_ENHANCED_DATA
    output_dir = Paths.DATA_DIR / 'derived' / 'gender_speeches'

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        print("Run data generation first")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}\n")

    # Process all years
    print("Processing years:")
    print("-"*80)

    total_speeches = 0
    years_processed = 0

    for year in range(start_year, end_year + 1):
        count = process_year(year, input_dir, output_dir, args.force)
        if count > 0:
            total_speeches += count
            years_processed += 1

    # Summary
    print("-"*80)
    print(f"\nSummary:")
    print(f"  Years processed: {years_processed}")
    print(f"  Total speeches extracted: {total_speeches:,}")
    print(f"  Output: {output_dir}")

    if total_speeches > 0:
        print(f"\n✓ Derived dataset created successfully!")
        print(f"\nNext steps:")
        print(f"  1. Test: python3 src/hansard/analysis/corpus_analysis.py --dataset gender --years {start_year}-{end_year} --sample 100")
        print(f"  2. Update temporal_gender_analysis.py to use this dataset")
    else:
        print(f"\n⚠ No speeches extracted - check input data")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
