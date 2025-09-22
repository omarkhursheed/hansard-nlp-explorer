#!/usr/bin/env python3
"""
Comprehensive analysis of the full gender dataset
"""

import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Load metadata
with open("gender_analysis_data_FULL/dataset_metadata.json", 'r') as f:
    metadata = json.load(f)

# Load the full dataset
df = pd.read_parquet("gender_analysis_data_FULL/ALL_debates_with_confirmed_mps.parquet")

print("=" * 70)
print("FULL GENDER DATASET ANALYSIS")
print("=" * 70)

# Overall Statistics
print("\n📊 OVERALL STATISTICS")
print("-" * 50)
total_processed = metadata['statistics']['total_debates_processed']
total_with_mps = metadata['statistics']['debates_with_confirmed_mps']
total_with_female = metadata['statistics']['debates_with_female']

print(f"Total debates processed: {total_processed:,}")
print(f"Debates with confirmed MPs: {total_with_mps:,} ({100*total_with_mps/total_processed:.1f}%)")
print(f"Debates with female MPs: {total_with_female:,} ({100*total_with_female/total_with_mps:.1f}%)")
print(f"Female MPs identified: {metadata['total_female_mps_identified']}")

# Coverage Analysis
print("\n📈 COVERAGE BY DECADE")
print("-" * 50)
decades = sorted([int(d) for d in metadata['statistics']['by_decade'].keys()])
for decade in decades:
    stats = metadata['statistics']['by_decade'][str(decade)]
    total = stats['debates_with_mps']
    female = stats['debates_with_female']
    pct = 100 * female / total if total > 0 else 0
    bar = "█" * int(pct/2) if pct > 0 else ""
    print(f"{decade}s: {female:6,}/{total:6,} ({pct:5.1f}%) {bar}")

# Key Milestones
print("\n🎯 KEY MILESTONES")
print("-" * 50)
print("1918: Partial women's suffrage (age 30+)")
print("1919: Nancy Astor first woman to sit as MP")
print("1928: Equal franchise (age 21)")
print("1979-1990: Margaret Thatcher as Prime Minister")
print("1992: Betty Boothroyd first female Speaker")

# First female participation by decade
print("\n👩 FIRST FEMALE PARTICIPATION")
print("-" * 50)
for decade in decades:
    female_count = metadata['statistics']['by_decade'][str(decade)]['debates_with_female']
    if female_count > 0:
        print(f"{decade}s: {female_count:,} debates with female MPs")
        if decade == 1910:
            print("  ↳ First female MP participation detected!")
        break

# Analyze the dataframe
print("\n📋 DATASET DETAILS")
print("-" * 50)
print(f"Dataset shape: {df.shape}")
print(f"Years covered: {df['year'].min()}-{df['year'].max()}")
print(f"Average speakers per debate: {df['total_speakers'].mean():.1f}")
print(f"Average confirmed MPs per debate: {df['confirmed_mps'].mean():.1f}")

# Female participation trends
print("\n📈 FEMALE PARTICIPATION TRENDS")
print("-" * 50)
yearly_female = df.groupby('year')['has_female'].agg(['sum', 'count', 'mean'])
yearly_female['percentage'] = yearly_female['mean'] * 100

# Find key years
first_female = yearly_female[yearly_female['sum'] > 0].index[0]
print(f"First year with female MP: {first_female}")

# Peak years
top_years = yearly_female.nlargest(10, 'percentage')
print("\nTop 10 years by female participation %:")
for year in top_years.index[:10]:
    stats = yearly_female.loc[year]
    print(f"  {year}: {stats['percentage']:.1f}% ({int(stats['sum'])}/{int(stats['count'])} debates)")

# Notable female MPs
print("\n👑 NOTABLE FEMALE MPs")
print("-" * 50)
notable = [
    "Nancy Astor",
    "Margaret Thatcher",
    "Betty Boothroyd",
    "Barbara Castle",
    "Shirley Williams",
    "Ellen Wilkinson",
    "Harriet Harman",
    "Mo Mowlam"
]

for mp in notable:
    if mp in metadata['female_mps_list']:
        # Count how many debates they appear in
        count = sum(1 for _, row in df.iterrows() if mp in row.get('female_names', []))
        print(f"✓ {mp}: appears in {count} debates")

# Data quality assessment
print("\n🔍 DATA QUALITY ASSESSMENT")
print("-" * 50)
print(f"Debates with ambiguous speakers: {df['ambiguous_speakers'].sum():,}")
print(f"Debates with unmatched speakers: {df['unmatched_speakers'].sum():,}")
coverage = 100 * df['confirmed_mps'].sum() / df['total_speakers'].sum()
print(f"Speaker coverage rate: {coverage:.1f}%")

# Gender balance over time
print("\n⚖️ GENDER BALANCE METRICS")
print("-" * 50)
total_female_speakers = df['female_mps'].sum()
total_male_speakers = df['male_mps'].sum()
total_confirmed = total_female_speakers + total_male_speakers

print(f"Total confirmed speakers: {total_confirmed:,}")
print(f"Female speakers: {total_female_speakers:,} ({100*total_female_speakers/total_confirmed:.2f}%)")
print(f"Male speakers: {total_male_speakers:,} ({100*total_male_speakers/total_confirmed:.2f}%)")

# Calculate average female proportion when present
debates_with_female = df[df['has_female']]
if len(debates_with_female) > 0:
    debates_with_female['female_proportion'] = debates_with_female['female_mps'] / debates_with_female['confirmed_mps']
    avg_proportion = debates_with_female['female_proportion'].mean()
    print(f"\nIn debates with female MPs:")
    print(f"  Average female proportion: {100*avg_proportion:.1f}%")
    print(f"  Median female proportion: {100*debates_with_female['female_proportion'].median():.1f}%")

# Interesting patterns
print("\n🔎 INTERESTING PATTERNS")
print("-" * 50)

# Find debates with highest female participation
high_female = df[df['female_mps'] > 5].sort_values('female_mps', ascending=False).head(5)
if len(high_female) > 0:
    print("Debates with most female MPs:")
    for _, debate in high_female.iterrows():
        print(f"  {debate['year']}: {debate['female_mps']} female MPs - {debate['title'][:50]}...")

# Growth rate
decades_with_female = [d for d in decades if metadata['statistics']['by_decade'][str(d)]['debates_with_female'] > 0]
if len(decades_with_female) > 1:
    first_decade = decades_with_female[0]
    last_decade = decades_with_female[-1]
    first_pct = 100 * metadata['statistics']['by_decade'][str(first_decade)]['debates_with_female'] / metadata['statistics']['by_decade'][str(first_decade)]['debates_with_mps']
    last_pct = 100 * metadata['statistics']['by_decade'][str(last_decade)]['debates_with_female'] / metadata['statistics']['by_decade'][str(last_decade)]['debates_with_mps']

    print(f"\nGrowth from {first_decade}s to {last_decade}s:")
    print(f"  {first_pct:.1f}% → {last_pct:.1f}% (×{last_pct/first_pct:.1f} increase)")

print("\n" + "=" * 70)
print("Analysis complete. Dataset ready for research!")
print("=" * 70)