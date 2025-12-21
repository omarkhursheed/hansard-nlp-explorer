#!/usr/bin/env python3
"""
Check for potential misclassifications related to university suffrage.
"""

import pandas as pd

# Load the LLM results and speeches
results = pd.read_parquet('outputs/llm_classification/full_results_v5_context_3_complete.parquet')
speeches = pd.read_parquet('outputs/suffrage_reliable/speeches_reliable.parquet')

# Join
df = results.merge(speeches[['speech_id', 'text']], on='speech_id', how='left')

# Find speeches about university context without women
df['text_lower'] = df['text'].str.lower()

university_pattern = df['text_lower'].str.contains('universit', na=False)
women_pattern = df['text_lower'].str.contains('women|female', na=False)
not_irrelevant = df['stance'] != 'irrelevant'

university_no_women = df[university_pattern & ~women_pattern & not_irrelevant]

print(f'University-related speeches WITHOUT women mention: {len(university_no_women)}')
print()

for idx, row in university_no_women.head(5).iterrows():
    print('='*80)
    print(f'Speech ID: {row["speech_id"]}')
    print(f'Date: {row["date"]}')
    print(f'Stance: {row["stance"]}')
    print(f'Confidence level: {row["confidence_level"]}')
    print()
    print('TEXT:')
    print(row['text'][:800])
    print()
    print('REASONS:')
    print(row['reasons'])
    print()
    print('TOP QUOTE:')
    print(row['top_quote'])
    print()
