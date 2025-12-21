#!/usr/bin/env python3
"""
Fix interjection-merge errors in the suffrage dataset.

Identifies speeches where a brief interjection was incorrectly merged with
the following speaker's full speech, causing misattribution.
"""

import pandas as pd
import re
from pathlib import Path

# The 7 known female MP misattributions
KNOWN_ERRORS = [
    {
        'speech_id': '15afbf4e6cc3a3a3_speech_3',
        'wrong_speaker': 'Viscountess ASTOR',
        'correct_speaker': 'Mr. MURRAY',
        'interjection': 'Male electors!',
        'year': 1920
    },
    {
        'speech_id': '15afbf4e6cc3a3a3_speech_4',
        'wrong_speaker': 'Viscountess ASTOR',
        'correct_speaker': 'Mr. MURRAY',
        'interjection': 'Hear, hear!',
        'year': 1920
    },
    {
        'speech_id': 'f85e000418c0a6c3_speech_16',
        'wrong_speaker': 'Viscountess ASTOR',
        'correct_speaker': 'Mr. JONES',
        'interjection': 'This is one.',
        'year': 1920
    },
    {
        'speech_id': '54838a35494d9608_speech_13',
        'wrong_speaker': 'Jo Richardson',
        'correct_speaker': 'Mrs. Jackson',
        'interjection': 'If',
        'year': 1989
    },
    {
        'speech_id': 'cb2f5a206a7f38fd_speech_52',
        'wrong_speaker': 'Sandra Gidley',
        'correct_speaker': 'Mr. Deputy',
        'interjection': 'I apologise,',
        'year': 2002
    },
    {
        'speech_id': 'bca7a4cc894d6208_speech_8',
        'wrong_speaker': 'Sylvia Hermon',
        'correct_speaker': 'Mr. Deputy',
        'interjection': 'Thank you,',
        'year': 2003
    },
    {
        'speech_id': '112b57262f60e278_speech_72',
        'wrong_speaker': 'Theresa May',
        'correct_speaker': 'Mr. Deputy',
        'interjection': '(Maidenhead) Thank you,',
        'year': 1997
    }
]


def load_mp_data():
    """Load MP gender data for matching."""
    mp_df = pd.read_parquet('data-hansard/house_members_gendered_updated.parquet')
    return mp_df


def match_speaker_to_mp(speaker_text, year, mp_df):
    """
    Try to match a speaker string to an MP.

    Returns: (person_name, gender, party) or (None, None, None)
    """
    # Clean up speaker text
    speaker_clean = speaker_text.replace('Mr.', '').replace('Mrs.', '').replace('Miss', '')
    speaker_clean = speaker_clean.replace('Dr.', '').replace('Sir', '').strip()

    # Special cases
    if 'Deputy' in speaker_text or 'Speaker' in speaker_text:
        return None, None, None  # Procedural role, not an MP

    # Try to find in MP database
    # Filter MPs active in this year
    # Convert year to datetime for comparison
    year_start = pd.Timestamp(f'{year}-01-01')
    year_end = pd.Timestamp(f'{year}-12-31')

    # Get MPs who were active at any point during this year
    # Handle mixed date formats (some are full dates, some are just years)
    active_mps = mp_df[
        (pd.to_datetime(mp_df['start_date'], format='mixed', errors='coerce') <= year_end) &
        (pd.to_datetime(mp_df['end_date'], format='mixed', errors='coerce') >= year_start)
    ]

    # Try exact lastname match first
    lastname = speaker_clean.split()[-1] if speaker_clean else ''

    if lastname:
        # Look for MPs with this lastname
        matches = active_mps[
            active_mps['person_name'].str.contains(lastname, case=False, na=False)
        ]

        if len(matches) == 1:
            mp = matches.iloc[0]
            return mp['person_name'], mp.get('gender_inferred', None), mp.get('on_behalf_of_id', None)
        elif len(matches) > 1:
            # Multiple matches - try to disambiguate
            # For now, return None (ambiguous)
            print(f"  WARNING: Multiple MPs matched '{speaker_text}' in {year}: {matches['person_name'].tolist()}")
            return None, None, None

    return None, None, None


def fix_errors():
    """Fix the known interjection-merge errors."""

    print('='*80)
    print('FIXING INTERJECTION-MERGE ERRORS')
    print('='*80)

    # Load data
    print('\nLoading datasets...')
    results_df = pd.read_parquet('outputs/llm_classification/full_results_v5_context_3_expanded.parquet')
    input_df = pd.read_parquet('outputs/llm_classification/full_input_context_3_expanded.parquet')
    mp_df = load_mp_data()

    print(f'Loaded {len(results_df):,} classified speeches')
    print(f'Loaded {len(mp_df):,} MPs')

    # Also load the source speech data to get correct attributions
    source_speeches = {}
    for error in KNOWN_ERRORS:
        year = error['year']
        speech_file = f'data-hansard/derived_complete/speeches_complete/speeches_{year}.parquet'

        if year not in source_speeches:
            source_speeches[year] = pd.read_parquet(speech_file)

    print(f'\nProcessing {len(KNOWN_ERRORS)} known errors...\n')

    corrections = []
    needs_reclassification = []

    for error in KNOWN_ERRORS:
        speech_id = error['speech_id']
        year = error['year']
        correct_speaker = error['correct_speaker']

        print(f"Speech {speech_id}:")
        print(f"  Wrong: {error['wrong_speaker']}")
        print(f"  Correct: {correct_speaker}")

        # Try to match correct speaker to MP database
        canonical_name, gender, party = match_speaker_to_mp(correct_speaker, year, mp_df)

        if canonical_name:
            print(f"  Matched to MP: {canonical_name} ({gender})")
            corrections.append({
                'speech_id': speech_id,
                'old_speaker': error['wrong_speaker'],
                'new_speaker': correct_speaker,
                'person_name': canonical_name,
                'gender': gender,
                'party': party,
                'needs_reclassification': False
            })
        else:
            if 'Deputy' in correct_speaker or 'Speaker' in correct_speaker:
                print(f"  -> Procedural speaker, marking for EXCLUSION")
                corrections.append({
                    'speech_id': speech_id,
                    'old_speaker': error['wrong_speaker'],
                    'new_speaker': correct_speaker,
                    'person_name': None,
                    'gender': None,
                    'party': None,
                    'needs_reclassification': False,
                    'exclude': True
                })
            else:
                print(f"  -> Could not match to MP, needs MANUAL REVIEW")
                corrections.append({
                    'speech_id': speech_id,
                    'old_speaker': error['wrong_speaker'],
                    'new_speaker': correct_speaker,
                    'person_name': None,
                    'gender': None,
                    'party': None,
                    'needs_reclassification': True
                })

        print()

    # Save corrections
    corrections_df = pd.DataFrame(corrections)
    output_dir = Path('outputs/corrections')
    output_dir.mkdir(exist_ok=True, parents=True)

    corrections_df.to_csv(output_dir / 'interjection_errors_corrections.csv', index=False)
    print(f"Saved corrections to: {output_dir / 'interjection_errors_corrections.csv'}")

    # Summary
    print('\n' + '='*80)
    print('SUMMARY')
    print('='*80)

    matched = len([c for c in corrections if c.get('person_name') is not None])
    excluded = len([c for c in corrections if c.get('exclude', False)])
    manual = len([c for c in corrections if c['needs_reclassification']])

    print(f"Total errors: {len(corrections)}")
    print(f"  Matched to MPs: {matched}")
    print(f"  Procedural (exclude): {excluded}")
    print(f"  Needs manual review: {manual}")

    return corrections_df


if __name__ == '__main__':
    corrections = fix_errors()
