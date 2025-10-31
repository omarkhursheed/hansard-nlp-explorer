#!/usr/bin/env python3
"""
Create Full Speeches and Debates Datasets

Extracts ALL debates and speeches from processed_fixed/ (not just gender-matched).
Optionally enriches with gender data when available.

Input:  data-hansard/processed_fixed/
Output:
    - data-hansard/derived/speeches_full/speeches_YYYY.parquet (all speeches)
    - data-hansard/derived/debates_full/debates_YYYY.parquet (all debates)

Speeches Schema:
    speech_id: str          # Unique ID (debate_id + speech_index)
    debate_id: str          # Links back to original debate (content_hash)
    year: int               # Year of speech
    decade: int             # Decade (e.g., 1990)
    month: str              # Month (jan, feb, etc.)
    date: str               # Date of debate
    speaker: str            # Speaker name
    gender: str or None     # 'm', 'f', or None (enriched from house_members if available)
    position: int           # Character position in original debate text
    text: str               # Speech text
    word_count: int         # Words in speech
    chamber: str            # Chamber (Commons/Lords)
    title: str              # Debate title
    hansard_reference: str  # Official Hansard citation
    reference_volume: int   # Volume number
    reference_columns: str  # Column numbers

Debates Schema:
    debate_id: str          # Unique debate ID (content_hash)
    year: int               # Year
    date: str               # Date
    title: str              # Debate title
    chamber: str            # Commons/Lords
    full_text: str          # Complete debate text
    speakers: list          # List of all speaker names
    speaker_count: int      # Number of unique speakers
    speech_count: int       # Number of speeches
    word_count: int         # Total words
    hansard_reference: str  # Official reference
    reference_volume: int   # Volume number
    reference_columns: str  # Column numbers

Usage:
    python scripts/create_full_speeches_dataset.py [--years START-END] [--test]

Options:
    --years START-END    Process only these years (e.g., 1990-2000)
    --test               Test mode: process only 1 year (1900)
    --force              Overwrite existing files
    --no-gender          Skip gender enrichment
"""

import argparse
import sys
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'hansard'))
from utils.path_config import Paths


def extract_speeches_from_text(text, speakers_list):
    """
    Extract individual speech segments from debate text.
    Based on the logic from create_enhanced_gender_dataset.py
    """
    if not text or not speakers_list:
        return []

    speeches = []

    # Create patterns for each known speaker
    speaker_patterns = []
    for speaker in speakers_list:
        if pd.notna(speaker) and speaker:
            # Escape special regex characters
            escaped_speaker = re.escape(str(speaker))
            # Create pattern that matches speaker name at start of speech
            patterns = [
                f"§\\s*{escaped_speaker}",  # With section marker
                f"\\n{escaped_speaker}\\s*:",  # At line start with colon
                f"\\n{escaped_speaker}\\s*\\(",  # At line start with parenthesis
            ]
            speaker_patterns.extend([(p, speaker) for p in patterns])

    # Find all speaker positions
    speaker_positions = []
    for pattern, speaker in speaker_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            speaker_positions.append((match.start(), speaker))

    # Sort by position
    speaker_positions.sort(key=lambda x: x[0])

    # Extract speech segments
    for i, (pos, speaker) in enumerate(speaker_positions):
        # Get end position (start of next speech or end of text)
        end_pos = speaker_positions[i+1][0] if i+1 < len(speaker_positions) else len(text)

        # Extract speech text
        speech_text = text[pos:end_pos].strip()

        # Clean up the text (remove speaker name from beginning)
        for pattern in [f"§\\s*{re.escape(speaker)}", f"{re.escape(speaker)}\\s*:", f"{re.escape(speaker)}\\s*\\("]:
            speech_text = re.sub(f"^{pattern}", "", speech_text, flags=re.IGNORECASE).strip()

        # Only keep substantial speeches (>50 chars)
        if speech_text and len(speech_text) > 50:
            speeches.append({
                'speaker': speaker,
                'text': speech_text,
                'position': pos
            })

    return speeches


def load_gender_mapping(data_dir):
    """Load gender mapping from house_members file"""
    gender_file = data_dir / 'house_members_gendered_updated.parquet'

    if not gender_file.exists():
        print("  Warning: Gender mapping file not found, speeches will have gender=None")
        return {}

    try:
        df = pd.read_parquet(gender_file)
        # Create mapping: speaker_name -> gender
        gender_map = {}
        for _, row in df.iterrows():
            name = row.get('name', '')
            gender = row.get('gender', '').lower()
            if name and gender in ['m', 'f']:
                gender_map[name.upper()] = gender

        print(f"  Loaded {len(gender_map):,} gender mappings")
        return gender_map
    except Exception as e:
        print(f"  Warning: Could not load gender mapping: {e}")
        return {}


def process_year(year, processed_fixed_dir, speeches_output_dir, debates_output_dir,
                gender_map=None, force=False):
    """
    Process a single year of data from processed_fixed.

    Returns:
        Tuple of (speeches_count, debates_count)
    """
    # Input files
    metadata_file = processed_fixed_dir / 'metadata' / f'debates_{year}.parquet'
    content_file = processed_fixed_dir / 'content' / str(year) / f'debates_{year}.jsonl'

    # Output files
    speeches_output_file = speeches_output_dir / f'speeches_{year}.parquet'
    debates_output_file = debates_output_dir / f'debates_{year}.parquet'

    if not metadata_file.exists() or not content_file.exists():
        print(f"  {year}: Input files not found")
        return 0, 0

    if speeches_output_file.exists() and debates_output_file.exists() and not force:
        print(f"  {year}: Already exists (use --force to overwrite)")
        return 0, 0

    try:
        # Load metadata
        metadata_df = pd.read_parquet(metadata_file)

        # Load content (JSONL)
        content_map = {}  # content_hash -> debate_data
        with open(content_file, 'r') as f:
            for line in f:
                debate_data = json.loads(line)
                content_hash = debate_data.get('content_hash', '')
                if content_hash:
                    content_map[content_hash] = debate_data

        # Process each debate
        all_speeches = []
        all_debates = []

        for idx, meta_row in metadata_df.iterrows():
            content_hash = meta_row.get('content_hash', '')
            if not content_hash or content_hash not in content_map:
                continue

            debate_data = content_map[content_hash]
            full_text = debate_data.get('full_text', '')
            speakers = meta_row.get('speakers', [])

            # Convert numpy array to list
            if isinstance(speakers, np.ndarray):
                speakers = speakers.tolist()

            if not isinstance(speakers, list):
                speakers = []

            # Extract speeches from text
            speech_segments = extract_speeches_from_text(full_text, speakers)

            # Create debate record
            debate_record = {
                'debate_id': content_hash,
                'year': year,
                'date': meta_row.get('reference_date', ''),
                'title': meta_row.get('title', ''),
                'chamber': meta_row.get('chamber', ''),
                'full_text': full_text,
                'speakers': speakers,
                'speaker_count': len(speakers),
                'speech_count': len(speech_segments),
                'word_count': meta_row.get('word_count', 0),
                'hansard_reference': meta_row.get('hansard_reference', ''),
                'reference_volume': meta_row.get('reference_volume', ''),
                'reference_columns': meta_row.get('reference_columns', ''),
            }
            all_debates.append(debate_record)

            # Create speech records
            for speech_idx, segment in enumerate(speech_segments):
                speaker = segment['speaker']

                # Try to match gender
                gender = None
                if gender_map:
                    # Try exact match
                    gender = gender_map.get(speaker.upper())
                    # Try fuzzy match (simple contains)
                    if not gender:
                        for known_name, g in gender_map.items():
                            if speaker.upper() in known_name or known_name in speaker.upper():
                                gender = g
                                break

                speech_record = {
                    'speech_id': f"{content_hash}_speech_{speech_idx}",
                    'debate_id': content_hash,
                    'year': year,
                    'decade': year // 10 * 10,
                    'month': meta_row.get('month', ''),
                    'date': meta_row.get('reference_date', ''),
                    'speaker': speaker,
                    'gender': gender,
                    'position': segment['position'],
                    'text': segment['text'],
                    'word_count': len(segment['text'].split()),
                    'chamber': meta_row.get('chamber', ''),
                    'title': meta_row.get('title', ''),
                    'hansard_reference': meta_row.get('hansard_reference', ''),
                    'reference_volume': meta_row.get('reference_volume', ''),
                    'reference_columns': meta_row.get('reference_columns', ''),
                }
                all_speeches.append(speech_record)

        # Save speeches
        if all_speeches:
            speeches_df = pd.DataFrame(all_speeches)
            speeches_df.to_parquet(speeches_output_file, index=False)

            matched = len(speeches_df[speeches_df['gender'].notna()])
            total = len(speeches_df)
            print(f"  {year}: {total:,} speeches ({matched:,} gender-matched, {total-matched:,} unmatched)")
        else:
            print(f"  {year}: No speeches extracted")

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
        description='Create full speeches and debates datasets from processed_fixed',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--years', type=str,
                       help='Year range (e.g., 1990-2000)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: process only 1900')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing files')
    parser.add_argument('--no-gender', action='store_true',
                       help='Skip gender enrichment')

    args = parser.parse_args()

    # Determine year range
    if args.test:
        start_year, end_year = 1900, 1900
        print("TEST MODE: Processing 1900 only\n")
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
    print("CREATE FULL SPEECHES AND DEBATES DATASETS")
    print("="*80)
    print(f"\nProcessing years: {start_year}-{end_year}")
    print(f"Force overwrite: {args.force}")
    print(f"Gender enrichment: {not args.no_gender}\n")

    # Setup directories
    processed_fixed_dir = Paths.PROCESSED_FIXED
    speeches_output_dir = Paths.DATA_DIR / 'derived' / 'speeches_full'
    debates_output_dir = Paths.DATA_DIR / 'derived' / 'debates_full'

    if not processed_fixed_dir.exists():
        print(f"ERROR: Input directory not found: {processed_fixed_dir}")
        sys.exit(1)

    speeches_output_dir.mkdir(parents=True, exist_ok=True)
    debates_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {processed_fixed_dir}")
    print(f"Output speeches: {speeches_output_dir}")
    print(f"Output debates:  {debates_output_dir}\n")

    # Load gender mapping if requested
    gender_map = None if args.no_gender else load_gender_mapping(Paths.DATA_DIR)
    print()

    # Process all years
    print("Processing years:")
    print("-"*80)

    total_speeches = 0
    total_debates = 0
    years_processed = 0

    for year in range(start_year, end_year + 1):
        speech_count, debate_count = process_year(
            year, processed_fixed_dir, speeches_output_dir, debates_output_dir,
            gender_map=gender_map, force=args.force
        )
        if speech_count > 0 or debate_count > 0:
            total_speeches += speech_count
            total_debates += debate_count
            years_processed += 1

    # Summary
    print("-"*80)
    print(f"\nSummary:")
    print(f"  Years processed: {years_processed}")
    print(f"  Total speeches: {total_speeches:,}")
    print(f"  Total debates: {total_debates:,}")
    print(f"  Output speeches: {speeches_output_dir}")
    print(f"  Output debates:  {debates_output_dir}")

    if total_speeches > 0:
        print(f"\nFull dataset created successfully!")
        print(f"\nThis includes ALL debates from processed_fixed,")
        print(f"not just those with gender-matched speakers.")
    else:
        print(f"\nNo speeches extracted - check input data")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
