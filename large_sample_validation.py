#!/usr/bin/env python3
"""
Large-sample validation of Tier 2 precision.
Use multiple random samples of 100 speeches with automated categorization.
"""

import pandas as pd
import re
import numpy as np

print("="*70)
print("LARGE SAMPLE VALIDATION - TIER 2 PRECISION")
print("="*70)

# Load Tier 2
tier2 = pd.read_parquet('outputs/suffrage_v2/speeches_tier2_broader.parquet')
print(f"\nTotal Tier 2 speeches: {len(tier2):,}")

# Define keyword patterns for categorization
EXPLICIT_SUFFRAGE = [
    r'suffrage',
    r'suffragette',
    r'suffragist',
    r'enfranchise',
    r'votes for women',
]

WOMEN_VOTING = [
    r'women.*vote',
    r'women.*voting',
    r'women.*voter',
    r'women.*electoral',
    r'women.*franchise',
    r'women.*enfranchise',
    r'female.*vote',
    r'female.*franchise',
    r'women.*representation',
    r'franchise.*women',
    r'vote.*women',
    r'electoral.*women',
]

POLITICAL_CONTEXT = [
    r'representation of the people',
    r'equal franchise',
    r'parliamentary.*franchise',
    r'sex disqualification',
]

def categorize_speech(text):
    """
    Categorize speech into:
    - HIGH: Explicitly about suffrage
    - MEDIUM: About women's voting/political rights
    - LOW: Women mentioned with political terms, but not about voting rights
    - OFF: False positive
    """
    text_lower = text.lower()

    # HIGH: Explicit suffrage terms
    for pattern in EXPLICIT_SUFFRAGE:
        if re.search(pattern, text_lower):
            return 'HIGH'

    # MEDIUM: Women + voting in close proximity
    # Check if women and voting terms appear close together (within 50 words)
    words = text_lower.split()
    for i, word in enumerate(words):
        if 'women' in word or 'female' in word:
            # Check next 50 words for voting terms
            context = ' '.join(words[max(0,i-25):min(len(words),i+25)])
            for pattern in WOMEN_VOTING:
                if re.search(pattern, context):
                    return 'MEDIUM'

    # LOW: Political context with women
    for pattern in POLITICAL_CONTEXT:
        if re.search(pattern, text_lower):
            # Check if women is also mentioned
            if 'women' in text_lower or 'female' in text_lower:
                return 'LOW'

    # Check if it's just generic women + vote/franchise with no proximity
    has_women = 'women' in text_lower or 'female' in text_lower
    has_political = any(term in text_lower for term in ['vote', 'voting', 'franchise', 'electoral', 'election'])

    if has_women and has_political:
        return 'LOW'

    return 'OFF'


# Sample validation
print("\n" + "="*70)
print("SAMPLE 1: Random 100 speeches")
print("="*70)

np.random.seed(42)
sample1 = tier2.sample(min(100, len(tier2)))

categories1 = []
for idx, speech in sample1.iterrows():
    cat = categorize_speech(speech['text'])
    categories1.append(cat)

print(f"\nResults (n=100):")
for cat in ['HIGH', 'MEDIUM', 'LOW', 'OFF']:
    count = categories1.count(cat)
    print(f"  {cat:8}: {count:3} ({count/len(categories1)*100:5.1f}%)")

# Sample 2: Different seed
print("\n" + "="*70)
print("SAMPLE 2: Random 100 speeches (different seed)")
print("="*70)

np.random.seed(123)
sample2 = tier2.sample(min(100, len(tier2)))

categories2 = []
for idx, speech in sample2.iterrows():
    cat = categorize_speech(speech['text'])
    categories2.append(cat)

print(f"\nResults (n=100):")
for cat in ['HIGH', 'MEDIUM', 'LOW', 'OFF']:
    count = categories2.count(cat)
    print(f"  {cat:8}: {count:3} ({count/len(categories2)*100:5.1f}%)")

# Sample 3: Different seed
print("\n" + "="*70)
print("SAMPLE 3: Random 100 speeches (different seed)")
print("="*70)

np.random.seed(456)
sample3 = tier2.sample(min(100, len(tier2)))

categories3 = []
for idx, speech in sample3.iterrows():
    cat = categorize_speech(speech['text'])
    categories3.append(cat)

print(f"\nResults (n=100):")
for cat in ['HIGH', 'MEDIUM', 'LOW', 'OFF']:
    count = categories3.count(cat)
    print(f"  {cat:8}: {count:3} ({count/len(categories3)*100:5.1f}%)")

# Combined results
print("\n" + "="*70)
print("COMBINED RESULTS (n=300)")
print("="*70)

all_categories = categories1 + categories2 + categories3
print(f"\nTotal speeches validated: {len(all_categories)}")

for cat in ['HIGH', 'MEDIUM', 'LOW', 'OFF']:
    count = all_categories.count(cat)
    print(f"  {cat:8}: {count:3} ({count/len(all_categories)*100:5.1f}%)")

# Confidence intervals (95%)
high_pct = all_categories.count('HIGH') / len(all_categories)
medium_pct = all_categories.count('MEDIUM') / len(all_categories)
on_topic_pct = (all_categories.count('HIGH') + all_categories.count('MEDIUM')) / len(all_categories)

# Wilson score interval
from math import sqrt
def wilson_confidence(p, n, z=1.96):
    """Calculate Wilson score confidence interval."""
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z**2 / (2*n)
    adjusted_standard_deviation = sqrt((p*(1-p) + z**2/(4*n)) / n)

    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator

    return lower_bound, upper_bound

high_ci = wilson_confidence(high_pct, len(all_categories))
on_topic_ci = wilson_confidence(on_topic_pct, len(all_categories))

print(f"\n95% Confidence Intervals:")
print(f"  HIGH (explicit suffrage): {high_pct*100:.1f}% (CI: {high_ci[0]*100:.1f}%-{high_ci[1]*100:.1f}%)")
print(f"  HIGH+MEDIUM (on-topic): {on_topic_pct*100:.1f}% (CI: {on_topic_ci[0]*100:.1f}%-{on_topic_ci[1]*100:.1f}%)")

# Show examples from each category
print("\n" + "="*70)
print("EXAMPLE SPEECHES FROM EACH CATEGORY")
print("="*70)

for category in ['HIGH', 'MEDIUM', 'LOW', 'OFF']:
    print(f"\n{category} CATEGORY EXAMPLES:")
    print("-"*70)

    # Get 2 examples
    examples = [sample for sample, cat in zip(pd.concat([sample1, sample2, sample3]).iterrows(), all_categories) if cat == category][:2]

    for i, (idx, speech) in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Year: {speech['year']}, Speaker: {speech['canonical_name']}")
        print(f"Words: {speech['word_count']}")
        print(f"Preview: {speech['text'][:300]}...")

# Estimate total Tier 2 true positives
print("\n" + "="*70)
print("TIER 2 TRUE POSITIVE ESTIMATES")
print("="*70)

print(f"\nTotal Tier 2 speeches: {len(tier2):,}")
print(f"\nBased on sample (n=300):")

high_count = len(tier2) * high_pct
medium_count = len(tier2) * medium_pct
on_topic_count = len(tier2) * on_topic_pct

print(f"  HIGH (explicit): ~{high_count:.0f} speeches ({high_pct*100:.1f}%)")
print(f"  MEDIUM (women's voting): ~{medium_count:.0f} speeches ({medium_pct*100:.1f}%)")
print(f"  Total on-topic: ~{on_topic_count:.0f} speeches ({on_topic_pct*100:.1f}%)")

# Now combine with Tier 1
print("\n" + "="*70)
print("COMBINED TIER 1 + TIER 2 ESTIMATES")
print("="*70)

tier1 = pd.read_parquet('outputs/suffrage_v2/speeches_tier1_high_precision.parquet')

tier1_true = len(tier1) * 0.95  # 95% precision
tier2_on_topic = on_topic_count

total_true = tier1_true + tier2_on_topic
total_all = len(tier1) + len(tier2)

print(f"Tier 1: ~{tier1_true:.0f} true positives (95% of {len(tier1):,})")
print(f"Tier 2: ~{tier2_on_topic:.0f} on-topic (HIGH+MEDIUM)")
print(f"\nTotal estimated true/on-topic: ~{total_true:.0f} / {total_all:,}")
print(f"Overall precision: {total_true/total_all*100:.1f}%")

print("\n" + "="*70)
print("VALIDATION COMPLETE")
print("="*70)
