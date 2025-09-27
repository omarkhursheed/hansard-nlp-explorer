#!/usr/bin/env python3
"""
Test the corrected MP matcher with verified dates
"""

import pandas as pd
from hansard.scripts.matching.mp_matcher_corrected import CorrectedMPMatcher
from hansard.utils.path_utils import get_data_dir

def test_corrected_dates():
    """Test Prime Minister matching with corrected dates"""

    print("Loading MP data...")
    mp_data = pd.read_parquet(get_data_dir() / "house_members_gendered_updated.parquet")
    matcher = CorrectedMPMatcher(mp_data)

    print("\n=== TESTING PRIME MINISTER DATES (CORRECTED) ===\n")

    test_cases = [
        # Testing exact transition dates
        ("The Prime Minister", "1979-05-03", "Commons", "James Callaghan"),  # Day before Thatcher
        ("The Prime Minister", "1979-05-04", "Commons", "TRANSITION"),  # Exact transition date - ambiguous
        ("The Prime Minister", "1979-05-05", "Commons", "Margaret Thatcher"),  # Day after

        ("The Prime Minister", "1997-05-01", "Commons", "John Major"),  # Day before Blair
        ("The Prime Minister", "1997-05-02", "Commons", "TRANSITION"),  # Exact transition date - ambiguous
        ("The Prime Minister", "1997-05-03", "Commons", "Tony Blair"),  # Day after

        ("The Prime Minister", "1990-11-27", "Commons", "Margaret Thatcher"),  # Day before Major
        ("The Prime Minister", "1990-11-28", "Commons", "TRANSITION"),  # Exact transition date - ambiguous
        ("The Prime Minister", "1990-11-29", "Commons", "John Major"),  # Day after

        # Testing mid-term dates
        ("The Prime Minister", "1985-06-15", "Commons", "Margaret Thatcher"),
        ("The Prime Minister", "1999-12-31", "Commons", "Tony Blair"),
        ("The Prime Minister", "1942-01-01", "Commons", "Winston Churchill"),
    ]

    for speaker, date, chamber, expected in test_cases:
        result = matcher.match_comprehensive(speaker, date, chamber)

        if expected == "TRANSITION":
            # On transition dates, we expect either the outgoing or incoming PM, or no match
            # The important thing is we acknowledge the ambiguity
            if result['final_match']:
                print(f"⚠ {date}: Transition date - got {result['final_match']} (confidence: {result['confidence']:.2f})")
            else:
                print(f"✓ {date}: Transition date - correctly identified as ambiguous")
        elif result['final_match']:
            if result['final_match'] == expected:
                print(f"✓ {date}: {expected} (correct)")
            else:
                print(f"✗ {date}: Expected {expected}, got {result['final_match']}")
        else:
            print(f"✗ {date}: Expected {expected}, got NO MATCH")

    print("\n=== TESTING OTHER IMPROVEMENTS ===\n")

    other_tests = [
        ("Mr. Bavies", "1950-05-26", "Commons", "OCR correction to Davies"),
        ("Mrs. 0'Brien", "1970-06-15", "Commons", "OCR correction to O'Brien"),
        ("the Member for Finchley", "1980-06-15", "Commons", "Constituency match"),
        ("the Member for Sedgefield", "1995-06-15", "Commons", "Constituency match"),
    ]

    for speaker, date, chamber, description in other_tests:
        result = matcher.match_comprehensive(speaker, date, chamber)

        print(f"Test: {speaker} ({date})")
        print(f"  Purpose: {description}")
        if result['final_match']:
            print(f"  Result: ✓ {result['final_match']} (conf: {result['confidence']:.2f})")
        elif result['match_type'] == 'ambiguous':
            print(f"  Result: ⚠ Ambiguous ({result.get('ambiguity_count')} candidates)")
        else:
            print(f"  Result: ✗ No match")
        print()

if __name__ == "__main__":
    test_corrected_dates()
