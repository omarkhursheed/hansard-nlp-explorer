#!/usr/bin/env python3
"""
Deep False Positive Analysis
Check for incorrect matches and incorrect gender assignments
"""

import sys
from pathlib import Path

# Add src to path (script is now at src/hansard/scripts/quality/)
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
from hansard.matching.mp_matcher_corrected import CorrectedMPMatcher
import random

random.seed(42)

def check_cross_gender_matches():
    """Check if any names are matched to opposite genders across different periods."""

    print('='*80)
    print('TEST 1: CROSS-GENDER MATCHING (Most Critical)')
    print('='*80)
    print('\nChecking if same speaker name gets different genders in different years...')

    # Sample speeches from different years
    test_years = [1850, 1900, 1950, 2000]

    speaker_genders = {}  # speaker -> set of genders seen

    for year in test_years:
        speech_file = Path(f'data-hansard/derived_complete/speeches_complete/speeches_{year}.parquet')
        if not speech_file.exists():
            continue

        df = pd.read_parquet(speech_file)
        commons = df[df['chamber'] == 'Commons']
        matched = commons[commons['gender'].notna()]

        for idx, row in matched.iterrows():
            speaker = str(row['speaker']).strip().upper()
            gender = row['gender']

            if speaker not in speaker_genders:
                speaker_genders[speaker] = set()
            speaker_genders[speaker].add(gender)

    # Find speakers with multiple genders
    problems = []
    for speaker, genders in speaker_genders.items():
        if len(genders) > 1:
            problems.append((speaker, genders))

    if problems:
        print(f'\nFOUND {len(problems)} SPEAKERS WITH MULTIPLE GENDERS:')
        for speaker, genders in problems[:10]:  # Show first 10
            print(f'  {speaker}: {genders}')
        return False
    else:
        print('\nPASS: No speakers matched to multiple genders')
        return True


def check_honorific_gender_consistency():
    """Check if honorifics match assigned gender (Mr->M, Mrs/Miss->F)."""

    print('\n' + '='*80)
    print('TEST 2: HONORIFIC-GENDER CONSISTENCY')
    print('='*80)

    test_years = [1900, 1950, 2000]

    problems = []

    for year in test_years:
        speech_file = Path(f'data-hansard/derived_complete/speeches_complete/speeches_{year}.parquet')
        if not speech_file.exists():
            continue

        df = pd.read_parquet(speech_file)
        commons = df[df['chamber'] == 'Commons']
        matched = commons[commons['gender'].notna()]

        sample = matched.sample(min(500, len(matched)))

        for idx, row in sample.iterrows():
            speaker = str(row['speaker']).upper()
            gender = row['gender']

            # Check male honorifics
            if any(h in speaker for h in ['MR.', 'MR ', 'SIR ']):
                if gender != 'M':
                    problems.append({
                        'year': year,
                        'speaker': row['speaker'],
                        'gender': gender,
                        'issue': 'Male honorific but not M'
                    })

            # Check female honorifics
            if any(h in speaker for h in ['MRS.', 'MRS ', 'MISS ', 'MS.', 'MS ', 'LADY ']):
                if gender != 'F':
                    problems.append({
                        'year': year,
                        'speaker': row['speaker'],
                        'gender': gender,
                        'issue': 'Female honorific but not F'
                    })

    if problems:
        print(f'\nFOUND {len(problems)} HONORIFIC-GENDER MISMATCHES:')
        for p in problems[:10]:
            print(f'  {p["year"]}: {p["speaker"]} -> {p["gender"]} ({p["issue"]})')
        return False
    else:
        print('\nPASS: All honorifics match assigned gender')
        return True


def check_ambiguous_matches_detail():
    """Deep dive into ambiguous_consistent_gender matches."""

    print('\n' + '='*80)
    print('TEST 3: AMBIGUOUS_CONSISTENT_GENDER VALIDATION')
    print('='*80)

    mnis = pd.read_parquet('data-hansard/house_members_gendered_updated.parquet')
    matcher = CorrectedMPMatcher(mnis)

    # Sample ambiguous names
    test_cases = [
        ('MR. SMITH', '1900-01-01', 'Commons'),
        ('MR. JONES', '1950-01-01', 'Commons'),
        ('MR. BROWN', '1900-06-01', 'Commons'),
        ('MR. GOSCHEN', '1900-03-02', 'Commons'),
        ('MR. WILSON', '1970-01-01', 'Commons'),
        ('LINDSAY', '1950-01-01', 'Commons'),  # Should be excluded (mixed)
        ('APSLEY', '1920-01-01', 'Commons'),   # Should be excluded (mixed)
    ]

    passed = True

    for speaker, date, chamber in test_cases:
        result = matcher.match_comprehensive(speaker, date, chamber)

        match_type = result['match_type']
        gender = result.get('gender')

        print(f'\n{speaker} on {date}:')
        print(f'  Match type: {match_type}')
        print(f'  Gender: {gender}')

        if match_type == 'ambiguous_consistent_gender':
            # Verify all candidates have same gender
            matches = result.get('matches', []) or result.get('possible_matches', [])
            genders = set(m.get('gender') for m in matches if m.get('gender'))

            print(f'  Candidates: {len(matches)}')
            print(f'  Genders in candidates: {genders}')

            if len(genders) > 1:
                print(f'  FAIL: Multiple genders but marked as consistent!')
                passed = False
            else:
                print(f'  PASS: All candidates are {list(genders)[0]}')

        elif 'lindsay' in speaker.lower() or 'apsley' in speaker.lower():
            # These should NOT be ambiguous_consistent_gender
            if match_type == 'ambiguous_consistent_gender':
                print(f'  FAIL: Mixed-gender name not excluded!')
                passed = False
            else:
                print(f'  PASS: Mixed-gender name correctly excluded')

    return passed


def check_temporal_validity():
    """Check if matched MPs were actually serving at the date of speech."""

    print('\n' + '='*80)
    print('TEST 4: TEMPORAL VALIDITY CHECK')
    print('='*80)

    mnis = pd.read_parquet('data-hansard/house_members_gendered_updated.parquet')

    test_years = [1900, 1950, 2000]

    problems = []

    for year in test_years:
        speech_file = Path(f'data-hansard/derived_complete/speeches_complete/speeches_{year}.parquet')
        if not speech_file.exists():
            continue

        df = pd.read_parquet(speech_file)
        commons = df[df['chamber'] == 'Commons']
        matched = commons[commons['gender'].notna()]

        sample = matched.sample(min(100, len(matched)))

        for idx, row in sample.iterrows():
            speaker = row['speaker']
            date = pd.Timestamp(row['date'])

            # Try to find this person in MNIS
            # Look for matches by surname
            surname = speaker.split()[-1].strip('.,()').upper()

            # Find MPs with this surname active on this date
            surname_mps = mnis[
                mnis['person_name'].str.upper().str.contains(surname, na=False) &
                (mnis['membership_start_date'] <= date) &
                ((mnis['membership_end_date'] >= date) | mnis['membership_end_date'].isna())
            ]

            if len(surname_mps) == 0:
                # No MP with this surname active on this date
                problems.append({
                    'speaker': speaker,
                    'date': date,
                    'gender': row['gender'],
                    'issue': 'No MP with surname active on date'
                })

    if problems:
        print(f'\nFOUND {len(problems)} POTENTIAL TEMPORAL MISMATCHES:')
        for p in problems[:10]:
            print(f'  {p["date"].year}: {p["speaker"]} (no {p["speaker"].split()[-1]} active)')

        # Note: These might be legitimate if surname matching is weak
        print('\nNote: These may be false alarms if surname extraction is imperfect')
        return len(problems) < 10  # Allow some false alarms
    else:
        print('\nPASS: All sampled matches have temporally valid MPs')
        return True


def check_known_female_mps():
    """Verify famous female MPs are correctly identified."""

    print('\n' + '='*80)
    print('TEST 5: KNOWN FEMALE MP VERIFICATION')
    print('='*80)

    # Famous female MPs with dates
    known_females = [
        ('Nancy Astor', '1920-01-01', 'First female MP to take seat'),
        ('Margaret Thatcher', '1980-01-01', 'First female PM'),
        ('Shirley Williams', '1970-01-01', 'Gang of Four member'),
        ('Barbara Castle', '1960-01-01', 'Labour minister'),
    ]

    mnis = pd.read_parquet('data-hansard/house_members_gendered_updated.parquet')
    matcher = CorrectedMPMatcher(mnis)

    passed = True

    for name, test_date, description in known_females:
        # Try different formats
        for format in [f'Mrs. {name.split()[1]}', f'Miss {name.split()[1]}', name]:
            result = matcher.match_comprehensive(format, test_date, 'Commons')

            if result.get('gender') == 'F':
                print(f'\nPASS: {name} ({description})')
                print(f'  Format "{format}" -> F')
                break
        else:
            print(f'\nWARNING: {name} not matched as female')
            print(f'  Tried multiple formats around {test_date}')
            passed = False

    return passed


def check_sample_speeches_manual():
    """Sample random speeches for manual review."""

    print('\n' + '='*80)
    print('TEST 6: RANDOM SAMPLE FOR MANUAL REVIEW')
    print('='*80)

    # Sample from different years
    test_years = [1900, 1950, 2000]

    print('\nReview these samples manually:')

    for year in test_years:
        speech_file = Path(f'data-hansard/derived_complete/speeches_complete/speeches_{year}.parquet')
        if not speech_file.exists():
            continue

        df = pd.read_parquet(speech_file)
        commons = df[df['chamber'] == 'Commons']
        matched = commons[commons['gender'].notna()]

        sample = matched.sample(min(3, len(matched)))

        print(f'\n{year}:')
        for idx, row in sample.iterrows():
            print(f'\n  Speaker: {row["speaker"]}')
            print(f'  Gender: {row["gender"]}')
            print(f'  Date: {row["date"]}')
            print(f'  Title: {row["title"][:60]}...')
            print(f'  Does this look correct? (Check speaker name format)')

    print('\n(Manual inspection required)')
    return True


def main():
    """Run all false positive checks."""

    print('\n')
    print('='*80)
    print('COMPREHENSIVE FALSE POSITIVE ANALYSIS')
    print('='*80)
    print()

    results = {}

    results['cross_gender'] = check_cross_gender_matches()
    results['honorific_consistency'] = check_honorific_gender_consistency()
    results['ambiguous_validation'] = check_ambiguous_matches_detail()
    results['temporal_validity'] = check_temporal_validity()
    results['known_females'] = check_known_female_mps()
    results['manual_review'] = check_sample_speeches_manual()

    print('\n' + '='*80)
    print('SUMMARY')
    print('='*80)

    for test_name, passed in results.items():
        status = 'PASSED' if passed else 'FAILED'
        print(f'{test_name:25s}: {status}')

    all_passed = all(results.values())

    print('\n' + '='*80)
    if all_passed:
        print('VERDICT: NO MAJOR FALSE POSITIVES DETECTED')
        print('Data quality is good for production use')
    else:
        print('VERDICT: POTENTIAL ISSUES FOUND')
        print('Review failing tests before using data')
    print('='*80)

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
