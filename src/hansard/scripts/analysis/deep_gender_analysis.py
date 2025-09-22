#!/usr/bin/env python3
"""
Deep analysis of gender patterns in parliamentary debates
Includes visualizations and statistical insights
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
print("Loading dataset...")
df = pd.read_parquet("gender_analysis_data_FULL/ALL_debates_with_confirmed_mps.parquet")

with open("gender_analysis_data_FULL/dataset_metadata.json", 'r') as f:
    metadata = json.load(f)

print("=" * 70)
print("DEEP GENDER ANALYSIS - PARLIAMENTARY DEBATES 1803-2005")
print("=" * 70)

# ============================================
# 1. TEMPORAL ANALYSIS
# ============================================
print("\n📅 TEMPORAL PATTERNS")
print("-" * 50)

# Create decade column
df['decade'] = (df['year'] // 10) * 10

# Analyze by decade
decade_stats = df.groupby('decade').agg({
    'has_female': ['sum', 'mean'],
    'female_mps': 'sum',
    'male_mps': 'sum',
    'confirmed_mps': 'sum',
    'word_count': 'sum'
}).round(3)

decade_stats.columns = ['debates_with_female', 'pct_with_female', 'total_female_speakers',
                        'total_male_speakers', 'total_confirmed', 'total_words']

# Calculate female speaker percentage
decade_stats['female_speaker_pct'] = 100 * decade_stats['total_female_speakers'] / decade_stats['total_confirmed']

print("\nDecade Analysis:")
print(decade_stats[['debates_with_female', 'pct_with_female', 'female_speaker_pct']])

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Gender Participation in UK Parliament (1803-2005)', fontsize=16)

# Plot 1: Debates with female participation over time
ax1 = axes[0, 0]
decades_with_data = decade_stats[decade_stats['debates_with_female'] > 0]
ax1.bar(decades_with_data.index, decades_with_data['pct_with_female'] * 100, width=8)
ax1.set_xlabel('Decade')
ax1.set_ylabel('% of Debates')
ax1.set_title('Percentage of Debates with Female MPs')
ax1.axvline(x=1919, color='r', linestyle='--', alpha=0.5, label='First Female MP')
ax1.legend()

# Plot 2: Female speaker percentage
ax2 = axes[0, 1]
ax2.plot(decades_with_data.index, decades_with_data['female_speaker_pct'], marker='o', linewidth=2)
ax2.set_xlabel('Decade')
ax2.set_ylabel('% of Speakers')
ax2.set_title('Female Speakers as % of All Confirmed Speakers')
ax2.fill_between(decades_with_data.index, 0, decades_with_data['female_speaker_pct'], alpha=0.3)

# Plot 3: Top female MPs by debate count
ax3 = axes[1, 0]
all_female_names = []
for names in df[df['has_female']]['female_names']:
    all_female_names.extend(names)
female_counts = Counter(all_female_names)
top_10 = dict(female_counts.most_common(10))
ax3.barh(list(top_10.keys()), list(top_10.values()))
ax3.set_xlabel('Number of Debates')
ax3.set_title('Top 10 Most Active Female MPs')

# Plot 4: Female participation by year (smoothed)
ax4 = axes[1, 1]
yearly_female = df.groupby('year')['has_female'].agg(['sum', 'mean'])
yearly_female['rolling_mean'] = yearly_female['mean'].rolling(window=5, min_periods=1).mean()
ax4.plot(yearly_female.index, yearly_female['rolling_mean'] * 100, linewidth=2)
ax4.set_xlabel('Year')
ax4.set_ylabel('% of Debates')
ax4.set_title('Female Participation Trend (5-year rolling average)')
ax4.axvspan(1979, 1990, alpha=0.2, color='blue', label='Thatcher PM')
ax4.legend()

plt.tight_layout()
plt.savefig('gender_analysis_data_FULL/gender_trends_visualization.png', dpi=150)
print("\n📊 Visualization saved to gender_trends_visualization.png")

# ============================================
# 2. CHAMBER ANALYSIS
# ============================================
print("\n🏛️ CHAMBER COMPARISON")
print("-" * 50)

chamber_stats = df.groupby('chamber').agg({
    'has_female': ['sum', 'mean'],
    'female_mps': 'sum',
    'male_mps': 'sum'
})

print(chamber_stats)

# ============================================
# 3. PARTICIPATION INTENSITY
# ============================================
print("\n💬 PARTICIPATION INTENSITY ANALYSIS")
print("-" * 50)

# When women participate, how many are there?
female_debates = df[df['has_female']]
intensity_dist = female_debates['female_mps'].value_counts().sort_index()

print("Distribution of female MP counts per debate:")
print(intensity_dist.head(10))

avg_female_when_present = female_debates['female_mps'].mean()
median_female_when_present = female_debates['female_mps'].median()

print(f"\nWhen women participate:")
print(f"  Average number of female MPs: {avg_female_when_present:.1f}")
print(f"  Median number of female MPs: {median_female_when_present:.0f}")
print(f"  Maximum female MPs in one debate: {female_debates['female_mps'].max()}")

# ============================================
# 4. TOPIC ANALYSIS
# ============================================
print("\n📝 TOPIC ANALYSIS")
print("-" * 50)

# Analyze topics with highest female participation
df['topic_clean'] = df['topic'].fillna('Unknown')
topic_stats = df.groupby('topic_clean').agg({
    'has_female': ['sum', 'count', 'mean']
})
topic_stats.columns = ['female_debates', 'total_debates', 'female_rate']

# Filter topics with at least 10 debates
significant_topics = topic_stats[topic_stats['total_debates'] >= 10]
top_female_topics = significant_topics.sort_values('female_rate', ascending=False).head(15)

print("Topics with highest female participation rate (min 10 debates):")
for topic, row in top_female_topics.iterrows():
    if len(topic) > 50:
        topic = topic[:47] + "..."
    print(f"  {row['female_rate']*100:5.1f}% - {topic}")

# ============================================
# 5. LANDMARK MOMENTS
# ============================================
print("\n🏆 LANDMARK MOMENTS")
print("-" * 50)

# First appearances of notable women
notable_women = {
    'Nancy Astor': 'First woman to sit as MP',
    'Margaret Thatcher': 'First female Prime Minister',
    'Betty Boothroyd': 'First female Speaker',
    'Ellen Wilkinson': 'First female Cabinet minister',
    'Barbara Castle': 'Transport & Employment Secretary',
    'Shirley Williams': 'SDP co-founder'
}

for mp_name, description in notable_women.items():
    mp_debates = df[df['female_names'].apply(lambda x: mp_name in x if x else False)]
    if len(mp_debates) > 0:
        first_year = mp_debates['year'].min()
        last_year = mp_debates['year'].max()
        total_debates = len(mp_debates)
        print(f"{mp_name} ({description}):")
        print(f"  Active: {first_year}-{last_year}, {total_debates} debates")

# ============================================
# 6. STATISTICAL INSIGHTS
# ============================================
print("\n📈 STATISTICAL INSIGHTS")
print("-" * 50)

# Correlation analysis
from scipy import stats

# Is there a correlation between year and female participation?
years_with_female = df[df['year'] >= 1919]['year']
female_participation = df[df['year'] >= 1919]['has_female']

correlation, p_value = stats.pearsonr(years_with_female, female_participation)
print(f"Correlation between year and female participation: {correlation:.3f} (p={p_value:.3e})")

# Growth rate calculation
decades_1920s = df[(df['decade'] == 1920)]['has_female'].mean()
decades_2000s = df[(df['decade'] == 2000)]['has_female'].mean()
growth_rate = (decades_2000s / decades_1920s - 1) * 100 if decades_1920s > 0 else 0

print(f"Growth from 1920s to 2000s: {growth_rate:.0f}%")

# Acceleration of change
print("\nDecade-over-decade growth rates:")
prev_rate = 0
for decade in range(1920, 2001, 10):
    decade_data = df[df['decade'] == decade]
    if len(decade_data) > 0:
        current_rate = decade_data['has_female'].mean() * 100
        if prev_rate > 0:
            growth = ((current_rate - prev_rate) / prev_rate) * 100
            print(f"  {decade}s: {current_rate:.1f}% (+{growth:.0f}% from previous)")
        else:
            print(f"  {decade}s: {current_rate:.1f}% (baseline)")
        prev_rate = current_rate

# ============================================
# 7. RESEARCH QUESTIONS
# ============================================
print("\n🔬 SUGGESTED RESEARCH QUESTIONS")
print("-" * 50)

questions = [
    "1. INTERRUPTION PATTERNS: Do female MPs face more interruptions?",
    "2. TOPIC SEGREGATION: Are women concentrated in 'soft' policy areas?",
    "3. SPEAKING TIME: Do women get equal speaking time when present?",
    "4. NETWORK EFFECTS: Do women speak more when other women are present?",
    "5. PARTY DIFFERENCES: Which parties had higher female participation?",
    "6. WARTIME EFFECTS: Did wars increase female participation?",
    "7. LEGISLATIVE IMPACT: Did female presence correlate with social reforms?",
    "8. RHETORICAL STYLES: Do gendered language patterns exist?",
    "9. CAREER TRAJECTORIES: How do female MP careers differ from male?",
    "10. CROSS-NATIONAL: How does UK compare to other parliaments?"
]

for q in questions:
    print(q)

# ============================================
# 8. DATA EXPORT RECOMMENDATIONS
# ============================================
print("\n💾 NEXT STEPS FOR ANALYSIS")
print("-" * 50)

recommendations = [
    "✓ Export high-female-participation debates for qualitative analysis",
    "✓ Create speaker network graphs (who speaks after whom)",
    "✓ Extract individual speaking turns for interruption analysis",
    "✓ Topic model the corpus with gender as a variable",
    "✓ Compare pre/post suffrage language patterns",
    "✓ Analyze questions vs statements by gender",
    "✓ Study committee participation patterns",
    "✓ Track career progression of female MPs",
    "✓ Measure influence through citation/mention networks",
    "✓ Compare wartime vs peacetime participation"
]

for rec in recommendations:
    print(rec)

# Save key statistics
summary = {
    'total_debates': len(df),
    'debates_with_female': df['has_female'].sum(),
    'female_participation_rate': df['has_female'].mean(),
    'total_female_mps': metadata['total_female_mps_identified'],
    'peak_female_year': df.groupby('year')['has_female'].mean().idxmax(),
    'peak_female_rate': df.groupby('year')['has_female'].mean().max(),
    'most_active_female_mp': female_counts.most_common(1)[0],
    'correlation_year_participation': correlation,
    'growth_1920s_2000s': growth_rate
}

with open('gender_analysis_data_FULL/analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print("Deep analysis complete! Ready for research papers!")
print("=" * 70)