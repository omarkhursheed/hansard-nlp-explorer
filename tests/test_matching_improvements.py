#!/usr/bin/env python3
"""
Unit tests for matching improvements:
1. Asterisk handling in speech extraction
2. Gender extraction from ambiguous matches
"""

import sys
sys.path.insert(0, 'src/hansard')

import re
import pandas as pd
from scripts.matching.mp_matcher_corrected import CorrectedMPMatcher

def test_asterisk_regex():
    """Test that asterisk handling works correctly in speech extraction."""

    print('='*80)
    print('TEST 1: ASTERISK REGEX')
    print('='*80)

    test_cases = [
        ('§ * MR. GOSCHEN The percentages', 'MR. GOSCHEN', True, 'With asterisk'),
        ('§ MR. GOSCHEN The percentages', 'MR. GOSCHEN', True, 'Without asterisk'),
        ('§ * SIR JOHN COLOMB I beg to ask', 'SIR JOHN COLOMB', True, 'Asterisk + Sir'),
        ('MR. GOSCHEN The percentages', 'MR. GOSCHEN', False, 'No section marker'),
        ('§  *  MR. SMITH Multiple spaces', 'MR. SMITH', True, 'Multiple spaces'),
    ]

    all_passed = True

    for text, speaker, should_match, description in test_cases:
        escaped_speaker = re.escape(speaker)
        pattern = r'§\s*\*?\s*' + escaped_speaker

        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        found_match = len(matches) > 0

        status = 'PASS' if found_match == should_match else 'FAIL'
        if found_match != should_match:
            all_passed = False

        print(f'\n{status}: {description}')
        print(f'  Text: {text[:50]}...')
        print(f'  Expected match: {should_match}, Got: {found_match}')

    print('\n' + '='*80)
    print(f'TEST 1 RESULT: {"ALL PASSED" if all_passed else "SOME FAILED"}')
    print('='*80)

    return all_passed


def test_ambiguous_gender_extraction():
    """Test that gender extraction from ambiguous matches works correctly."""

    print('\n' + '='*80)
    print('TEST 2: AMBIGUOUS GENDER EXTRACTION')
    print('='*80)

    # Load MNIS data
    mnis = pd.read_parquet('data-hansard/house_members_gendered_updated.parquet')
    matcher = CorrectedMPMatcher(mnis)

    test_cases = [
        ('MR. GOSCHEN', '1900-03-02', 'Commons', 'M', 'Goschen (all male)'),
        ('MR. SMITH', '1850-01-01', 'Commons', 'M', 'Smith (common name, likely male)'),
        # Mixed gender names should NOT get gender assigned
        ('LINDSAY', '1950-01-01', 'Commons', None, 'Lindsay (mixed gender)'),
        ('APSLEY', '1920-01-01', 'Commons', None, 'Apsley (mixed gender)'),
    ]

    all_passed = True

    for speaker, date, chamber, expected_gender, description in test_cases:
        result = matcher.match_comprehensive(speaker, date, chamber)

        actual_gender = result.get('gender')
        match_type = result.get('match_type')

        # For ambiguous cases, check if gender was extracted correctly
        if 'ambiguous' in match_type:
            status = 'PASS' if actual_gender == expected_gender else 'FAIL'
            if actual_gender != expected_gender:
                all_passed = False

            print(f'\n{status}: {description}')
            print(f'  Speaker: {speaker}, Date: {date}')
            print(f'  Match type: {match_type}')
            print(f'  Expected gender: {expected_gender}, Got: {actual_gender}')
            print(f'  Confidence: {result.get("confidence")}')
        else:
            # Unique match or no match
            print(f'\nSKIP: {description}')
            print(f'  Match type: {match_type} (not ambiguous)')
            print(f'  Gender: {actual_gender}')

    print('\n' + '='*80)
    print(f'TEST 2 RESULT: {"ALL PASSED" if all_passed else "SOME FAILED"}')
    print('='*80)

    return all_passed


def test_goschen_specific():
    """Specific test for the Goschen case we identified."""

    print('\n' + '='*80)
    print('TEST 3: GOSCHEN SPECIFIC CASE')
    print('='*80)

    mnis = pd.read_parquet('data-hansard/house_members_gendered_updated.parquet')
    matcher = CorrectedMPMatcher(mnis)

    result = matcher.match_comprehensive('MR. GOSCHEN', '1900-03-02', 'Commons')

    print(f'\nMatch type: {result["match_type"]}')
    print(f'Gender: {result["gender"]}')
    print(f'Confidence: {result["confidence"]}')

    # Get the matches
    matches = result.get('matches', []) or result.get('possible_matches', [])
    print(f'\nNumber of matches: {len(matches)}')

    if matches:
        genders = [m.get('gender') for m in matches]
        print(f'Genders in matches: {genders}')

    # Test expectations
    passed = True

    # Should be ambiguous or ambiguous_consistent_gender
    if 'ambiguous' not in result['match_type']:
        print('\nFAIL: Expected ambiguous match type')
        passed = False

    # Should have gender='M'
    if result['gender'] != 'M':
        print(f'\nFAIL: Expected gender=M, got {result["gender"]}')
        passed = False

    # Should have lower confidence (0.6)
    if result['match_type'] == 'ambiguous_consistent_gender' and result['confidence'] != 0.6:
        print(f'\nFAIL: Expected confidence=0.6, got {result["confidence"]}')
        passed = False

    print('\n' + '='*80)
    print(f'TEST 3 RESULT: {"PASSED" if passed else "FAILED"}')
    print('='*80)

    return passed


def test_mixed_gender_exclusion():
    """Test that known mixed-gender names are excluded."""

    print('\n' + '='*80)
    print('TEST 4: MIXED GENDER NAME EXCLUSION')
    print('='*80)

    mnis = pd.read_parquet('data-hansard/house_members_gendered_updated.parquet')
    matcher = CorrectedMPMatcher(mnis)

    mixed_names = ['lindsay', 'apsley', 'leslie wilson']

    all_passed = True

    for name in mixed_names:
        # Test with different formats
        for title in ['Mr', 'Mrs', '']:
            test_name = f'{title} {name}' if title else name
            result = matcher.match_comprehensive(test_name, '1950-01-01', 'Commons')

            # If match is ambiguous, gender should be None (excluded)
            if result['match_type'] == 'ambiguous_consistent_gender':
                print(f'\nFAIL: {test_name}')
                print(f'  Should be excluded but got gender={result["gender"]}')
                all_passed = False
            else:
                print(f'\nPASS: {test_name}')
                print(f'  Correctly excluded or uniquely matched')

    print('\n' + '='*80)
    print(f'TEST 4 RESULT: {"ALL PASSED" if all_passed else "SOME FAILED"}')
    print('='*80)

    return all_passed


def main():
    """Run all unit tests."""

    print('\n')
    print('='*80)
    print('UNIT TESTS FOR MATCHING IMPROVEMENTS')
    print('='*80)
    print()

    results = {}

    # Run tests
    results['asterisk'] = test_asterisk_regex()
    results['ambiguous_gender'] = test_ambiguous_gender_extraction()
    results['goschen'] = test_goschen_specific()
    results['mixed_exclusion'] = test_mixed_gender_exclusion()

    # Summary
    print('\n' + '='*80)
    print('SUMMARY')
    print('='*80)

    for test_name, passed in results.items():
        status = 'PASSED' if passed else 'FAILED'
        print(f'{test_name:20s}: {status}')

    all_passed = all(results.values())
    print('\n' + '='*80)
    print(f'OVERALL: {"ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"}')
    print('='*80)

    if all_passed:
        print('\nReady to proceed to integration testing on 1900 dataset.')
    else:
        print('\nFix failing tests before proceeding.')

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
