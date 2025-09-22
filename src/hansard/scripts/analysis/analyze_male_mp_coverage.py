#!/usr/bin/env python3
"""
Analyze male MP coverage by processing sample years with the matcher
"""

import pandas as pd
import numpy as np
from pathlib import Path
from mp_matcher_corrected import CorrectedMPMatcher
from tqdm import tqdm
from collections import defaultdict

print("=" * 70)
print("COMPREHENSIVE MALE MP COVERAGE ANALYSIS")
print("=" * 70)

# Load MP matcher
mp_data = pd.read_parquet("data/house_members_gendered_updated.parquet")
matcher = CorrectedMPMatcher(mp_data)

# Get ground truth counts
ground_truth_male = set(mp_data[mp_data['gender_inferred'] == 'M']['person_name'].unique())
ground_truth_female = set(mp_data[mp_data['gender_inferred'] == 'F']['person_name'].unique())

print(f"\n📊 GROUND TRUTH:")
print(f"Total MPs: {len(mp_data['person_name'].unique()):,}")
print(f"Male MPs: {len(ground_truth_male):,} (95.1%)")
print(f"Female MPs: {len(ground_truth_female):,} (4.9%)")

# Process sample years to collect unique MPs
sample_years = [1900, 1920, 1950, 1980, 2000]
all_matched_male = set()
all_matched_female = set()

print(f"\n🔍 Processing sample years: {sample_years}")

for year in sample_years:
    print(f"\nProcessing {year}...")

    try:
        debates_df = pd.read_parquet(f"data/processed_fixed/metadata/debates_{year}.parquet")

        for _, debate in tqdm(debates_df.iterrows(), total=len(debates_df), desc=f"Year {year}", leave=False):
            speakers = debate.get('speakers', [])
            date = debate.get('reference_date', f'{year}-01-01')
            chamber = debate.get('chamber', 'Commons')

            if not isinstance(speakers, (list, np.ndarray)):
                continue

            for speaker in speakers:
                if pd.notna(speaker):
                    result = matcher.match_comprehensive(str(speaker), date, chamber)

                    # Only count high-confidence matches
                    if result['match_type'] in ['temporal_unique', 'title', 'constituency']:
                        if result.get('confidence', 0) >= 0.7:
                            mp_name = result.get('final_match')
                            gender = result.get('gender')

                            if mp_name and gender == 'M':
                                all_matched_male.add(mp_name)
                            elif mp_name and gender == 'F':
                                all_matched_female.add(mp_name)
    except Exception as e:
        print(f"  Error processing {year}: {e}")

print("\n" + "=" * 70)
print("RESULTS FROM SAMPLE YEARS")
print("=" * 70)

print(f"\n📈 UNIQUE MPs MATCHED (from {len(sample_years)} sample years):")
print(f"Male MPs: {len(all_matched_male):,}")
print(f"Female MPs: {len(all_matched_female):,}")
print(f"Total: {len(all_matched_male) + len(all_matched_female):,}")

# Check accuracy
male_correct = all_matched_male & ground_truth_male
female_correct = all_matched_female & ground_truth_female
male_errors = all_matched_male - ground_truth_male
female_errors = all_matched_female - ground_truth_female

print(f"\n✅ ACCURACY CHECK:")
print(f"Male MPs correctly matched: {len(male_correct):,} / {len(all_matched_male):,} ({100*len(male_correct)/len(all_matched_male):.1f}%)")
print(f"Female MPs correctly matched: {len(female_correct):,} / {len(all_matched_female):,} ({100*len(female_correct)/len(all_matched_female):.1f}%)")

if male_errors:
    print(f"\n⚠️ Potential male MP errors: {len(male_errors)}")
    for mp in list(male_errors)[:5]:
        print(f"  - {mp}")

# Coverage estimation
print(f"\n📊 COVERAGE ESTIMATION (based on sample):")
male_coverage = 100 * len(male_correct) / len(ground_truth_male)
female_coverage = 100 * len(female_correct) / len(ground_truth_female)

print(f"Male MP coverage: {male_coverage:.1f}% ({len(male_correct):,} / {len(ground_truth_male):,})")
print(f"Female MP coverage: {female_coverage:.1f}% ({len(female_correct):,} / {len(ground_truth_female):,})")

print(f"\n🎯 COVERAGE COMPARISON:")
print(f"Male coverage rate: {male_coverage:.1f}%")
print(f"Female coverage rate: {female_coverage:.1f}%")
if male_coverage > 0 and female_coverage > 0:
    ratio = female_coverage / male_coverage
    if ratio > 1:
        print(f"Female MPs are {ratio:.1f}x MORE likely to be matched")
    else:
        print(f"Male MPs are {1/ratio:.1f}x MORE likely to be matched")

# Sample of matched male MPs
print(f"\n👨 SAMPLE OF MATCHED MALE MPs:")
for mp in sorted(all_matched_male)[:20]:
    print(f"  - {mp}")
print(f"  ... and {len(all_matched_male)-20} more")

# Temporal distribution
print(f"\n📅 MALE MPs BY ERA (in ground truth):")
mp_data['start_year'] = pd.to_datetime(mp_data['membership_start_date'], errors='coerce').dt.year
male_data = mp_data[mp_data['gender_inferred'] == 'M']
male_data['decade'] = (male_data['start_year'] // 10) * 10
decade_counts = male_data.groupby('decade')['person_name'].nunique()

for decade, count in decade_counts.items():
    if pd.notna(decade) and decade >= 1900:
        print(f"  {int(decade)}s: {count:,} male MPs")