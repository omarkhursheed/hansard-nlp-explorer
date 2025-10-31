#!/usr/bin/env python3
"""
Create Derived All Speeches Dataset (Non-Gendered)

Extracts speech_segments from gender_analysis_enhanced/ and creates flat datasets
including ALL speakers (both gender-matched and unmatched).

Input:  data-hansard/gender_analysis_enhanced/debates_YYYY_enhanced.parquet (nested)
Output:
    - data-hansard/derived/speeches/speeches_YYYY.parquet (flat, all speeches)
    - data-hansard/derived/debates/debates_YYYY.parquet (debate-level metadata)

Speeches Schema:
    speech_id: str          # Unique ID (debate_id + speech_index)
    debate_id: str          # Links back to original debate
    year: int               # Year of speech
    decade: int             # Decade (e.g., 1990)
    month: str              # Month (jan, feb, etc.)
    date: str               # Date of debate
    speaker: str            # Speaker name
    gender: str or None     # 'm', 'f', or None for unmatched
    position: int           # Line position in original debate
    text: str               # Speech text
    word_count: int         # Words in speech
    chamber: str            # Chamber (Commons/Lords)
    title: str              # Debate title
    topic: str              # Debate topic/category
    hansard_reference: str  # Official Hansard citation
    reference_volume: str   # Volume number
    reference_columns: str  # Column numbers in original

Debates Schema:
    debate_id: str          # Unique debate ID
    year: int               # Year
    date: str               # Date
    title: str              # Debate title
    chamber: str            # Commons/Lords
    full_text: str          # Complete debate text
    speakers: list          # List of all speaker names
    speaker_genders: dict   # Mapping of speakers to genders (M/F)
    total_speakers: int     # Total unique speakers
    confirmed_mps: int      # Gender-matched MPs
    unmatched_speakers: int # Speakers without gender match
    female_mps: int         # Female MPs count
    male_mps: int           # Male MPs count
    has_female: bool        # Has female speakers
    has_male: bool          # Has male speakers
    speech_count: int       # Number of speeches
    word_count: int         # Total words
    hansard_reference: str  # Official reference

Usage:
    python scripts/create_all_speeches_dataset.py [--years START-END] [--test]

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

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'hansard'))
from utils.path_config import Paths


def extract_speeches_from_debate(row, year):
    """
    Extract individual speeches from a debate row, including unmatched speakers.

    Args:
        row: DataFrame row from gender_analysis_enhanced
        year: Year of the debate

    Returns:
        List of speech dicts
    """
    speeches = []

    if 'speech_segments' not in row or row['speech_segments'] is None:
        return speeches

    # Build speaker gender map (for those with matched genders)
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

        # Map to gender (if available)
        gender = speaker_gender_map.get(speaker)
        if not gender:
            # Try fuzzy matching
            for orig_name, g in speaker_gender_map.items():
                if speaker in orig_name or orig_name in speaker:
                    gender = g
                    break

        # INCLUDE ALL SPEECHES, even without gender
        # gender will be None for unmatched speakers

        # Create speech record with enhanced metadata
        speech = {
            'speech_id': f"{debate_id}_speech_{speech_idx}",
            'debate_id': debate_id,
            'year': year,
            'decade': row.get('decade', year // 10 * 10),
            'month': row.get('month', ''),
            'date': row.get('reference_date', ''),
            'speaker': speaker,
            'gender': gender,  # Can be 'm', 'f', or None
            'position': segment.get('position', speech_idx),  # Line position in debate
            'text': text,
            'word_count': len(text.split()),
            'chamber': row.get('chamber', ''),
            'title': row.get('title', ''),
            'topic': row.get('topic', ''),
            'hansard_reference': row.get('hansard_reference', ''),
            'reference_volume': row.get('reference_volume', ''),
            'reference_columns': row.get('reference_columns', '')
        }

        speeches.append(speech)

    return speeches


def extract_debate_metadata(row, year):
    """
    Extract debate-level metadata.

    Args:
        row: DataFrame row from gender_analysis_enhanced
        year: Year of the debate

    Returns:
        Dict of debate metadata
    """
    debate_id = row.get('debate_id', f"debate_{year}_{row.name}")

    # Extract unique speakers from speech_segments
    speakers = []
    segments = row.get('speech_segments', [])
    if isinstance(segments, (list, np.ndarray)):
        seen = set()
        for seg in segments:
            if isinstance(seg, dict):
                speaker = seg.get('speaker', '')
                if speaker and speaker not in seen:
                    speakers.append(speaker)
                    seen.add(speaker)

    # Build speaker gender mapping
    speaker_genders = {}
    speaker_details = row.get('speaker_details', [])
    if isinstance(speaker_details, (list, np.ndarray)):
        for detail in speaker_details:
            if isinstance(detail, dict):
                name = detail.get('original_name', '')
                gender = detail.get('gender', '').upper()
                if name and gender:
                    speaker_genders[name] = gender

    return {
        'debate_id': debate_id,
        'year': year,
        'date': row.get('reference_date', ''),
        'title': row.get('title', ''),
        'chamber': row.get('chamber', ''),
        'full_text': row.get('debate_text', ''),
        'speakers': speakers,
        'speaker_genders': speaker_genders,
        'total_speakers': row.get('total_speakers', 0),
        'confirmed_mps': row.get('confirmed_mps', 0),
        'unmatched_speakers': row.get('unmatched_speakers', 0),
        'female_mps': row.get('female_mps', 0),
        'male_mps': row.get('male_mps', 0),
        'has_female': row.get('has_female', False),
        'has_male': row.get('has_male', False),
        'speech_count': row.get('speech_count', 0),
        'word_count': row.get('word_count', 0),
        'hansard_reference': row.get('hansard_reference', ''),
    }


def process_year(year, input_dir, speeches_output_dir, debates_output_dir, force=False):
    """
    Process a single year of data.

    Args:
        year: Year to process
        input_dir: Input directory (gender_analysis_enhanced)
        speeches_output_dir: Output directory for speeches
        debates_output_dir: Output directory for debates
        force: Overwrite existing files

    Returns:
        Tuple of (speeches_count, debates_count)
    """
    input_file = input_dir / f"debates_{year}_enhanced.parquet"
    speeches_output_file = speeches_output_dir / f"speeches_{year}.parquet"
    debates_output_file = debates_output_dir / f"debates_{year}.parquet"

    if not input_file.exists():
        print(f"  {year}: No input file found")
        return 0, 0

    if speeches_output_file.exists() and debates_output_file.exists() and not force:
        print(f"  {year}: Already exists (use --force to overwrite)")
        return 0, 0

    try:
        # Load debate data
        df = pd.read_parquet(input_file)

        # Extract all speeches (including unmatched)
        all_speeches = []
        all_debates = []

        for idx, row in df.iterrows():
            speeches = extract_speeches_from_debate(row, year)
            all_speeches.extend(speeches)

            debate_meta = extract_debate_metadata(row, year)
            all_debates.append(debate_meta)

        # Save speeches
        if all_speeches:
            speeches_df = pd.DataFrame(all_speeches)
            speeches_df.to_parquet(speeches_output_file, index=False)

            male_count = len(speeches_df[speeches_df['gender'] == 'm'])
            female_count = len(speeches_df[speeches_df['gender'] == 'f'])
            unmatched_count = len(speeches_df[speeches_df['gender'].isna()])

            print(f"  {year}: {len(speeches_df):,} speeches "
                  f"({male_count:,} male, {female_count:,} female, {unmatched_count:,} unmatched)")
        else:
            print(f"  {year}: No speeches extracted")
            speeches_df = pd.DataFrame()

        # Save debates
        if all_debates:
            debates_df = pd.DataFrame(all_debates)
            debates_df.to_parquet(debates_output_file, index=False)

        return len(all_speeches), len(all_debates)

    except Exception as e:
        print(f"  {year}: Error - {e}")
        import traceback
        traceback.print_exc()
        return 0, 0


def main():
    parser = argparse.ArgumentParser(
        description='Create derived all speeches dataset (non-gendered)',
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
    print("CREATE DERIVED ALL SPEECHES DATASET (NON-GENDERED)")
    print("="*80)
    print(f"\nProcessing years: {start_year}-{end_year}")
    print(f"Force overwrite: {args.force}\n")

    # Setup directories
    input_dir = Paths.GENDER_ENHANCED_DATA
    speeches_output_dir = Paths.DATA_DIR / 'derived' / 'speeches'
    debates_output_dir = Paths.DATA_DIR / 'derived' / 'debates'

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        print("Run gender analysis enhancement first")
        sys.exit(1)

    speeches_output_dir.mkdir(parents=True, exist_ok=True)
    debates_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_dir}")
    print(f"Output speeches: {speeches_output_dir}")
    print(f"Output debates:  {debates_output_dir}\n")

    # Process all years
    print("Processing years:")
    print("-"*80)

    total_speeches = 0
    total_debates = 0
    years_processed = 0

    for year in range(start_year, end_year + 1):
        speech_count, debate_count = process_year(
            year, input_dir, speeches_output_dir, debates_output_dir, args.force
        )
        if speech_count > 0 or debate_count > 0:
            total_speeches += speech_count
            total_debates += debate_count
            years_processed += 1

    # Summary
    print("-"*80)
    print(f"\nSummary:")
    print(f"  Years processed: {years_processed}")
    print(f"  Total speeches extracted: {total_speeches:,}")
    print(f"  Total debates extracted: {total_debates:,}")
    print(f"  Output speeches: {speeches_output_dir}")
    print(f"  Output debates:  {debates_output_dir}")

    if total_speeches > 0:
        print(f"\nDataset created successfully!")
        print(f"\nNext steps:")
        print(f"  1. Check speeches: python3 -c \"import pandas as pd; df = pd.read_parquet('data-hansard/derived/speeches/speeches_{start_year}.parquet'); print(df.info()); print(df['gender'].value_counts(dropna=False))\"")
        print(f"  2. Check debates: python3 -c \"import pandas as pd; df = pd.read_parquet('data-hansard/derived/debates/debates_{start_year}.parquet'); print(df.info())\"")
    else:
        print(f"\nNo speeches extracted - check input data")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
