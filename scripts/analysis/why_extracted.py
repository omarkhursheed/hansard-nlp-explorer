#!/usr/bin/env python3
"""
Check why specific speeches were extracted.
"""

import pandas as pd
import re

# Load speeches
speeches = pd.read_parquet('outputs/suffrage_reliable/speeches_reliable.parquet')

# Get the three misclassified speech IDs
misclass_ids = [
    'f51995ec8df85e4d_speech_1',  # Dec 1906, plural voting
    'dd191c06fe4bb0c4_speech_1',  # Apr 1913, redistribution
]

# HIGH pattern from extract_suffrage_reliable.py
high_pattern = (
    'women.*suffrage|female suffrage|suffrage.*women|'
    'votes for women|suffragette|suffragist|'
    'enfranchise.*women|women.*enfranchise|'
    'equal franchise|'
    'representation of the people.*women|'
    'sex disqualification|'
    'women.*social.*political.*union'
)

for speech_id in misclass_ids:
    speech = speeches[speeches['speech_id'] == speech_id]
    if len(speech) == 0:
        print(f"Speech {speech_id} not found")
        continue

    row = speech.iloc[0]
    text = row['text']

    print('='*80)
    print(f"Speech ID: {speech_id}")
    print(f"Date: {row['date']}")
    print(f"Confidence level: {row['confidence_level']}")
    print()

    # Check HIGH pattern
    high_match = re.search(high_pattern, text, re.IGNORECASE)
    if high_match:
        print(f"MATCHED HIGH pattern: '{high_match.group()}'")
    else:
        print("Did NOT match HIGH pattern")

    # Check for women/female
    has_women = bool(re.search(r'\bwomen\b|\bfemale\b', text, re.IGNORECASE))
    print(f"Contains 'women' or 'female': {has_women}")

    # Check for voting terms
    voting_terms = ['vote', 'voting', 'voter', 'voters', 'electoral', 'electorate', 'franchise', 'enfranchise', 'representation']
    found_voting = [term for term in voting_terms if re.search(rf'\b{term}', text, re.IGNORECASE)]
    print(f"Voting terms found: {found_voting}")

    # Show first 500 chars
    print()
    print("Text (first 500 chars):")
    print(text[:500])
    print()
