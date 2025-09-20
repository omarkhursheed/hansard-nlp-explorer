# Hansard NLP Analysis Suite - Complete Guide

## Overview

This suite provides multiple approaches to analyzing the Hansard parliamentary debates corpus with progressively aggressive content filtering to extract substantive policy discussions from procedural language.

## Available Analysis Scripts

### 1. `hansard_nlp_analysis_advanced.py` - Multi-Level Filtering (RECOMMENDED)

The most comprehensive script with 8 different filtering levels:

```bash
# Run specific filter level
python hansard_nlp_analysis_advanced.py --years 1920-1930 --sample 500 --filter-level 3

# Compare all filter levels
python hansard_nlp_analysis_advanced.py --years 1920-1930 --sample 500 --all-levels

# Full corpus analysis
python hansard_nlp_analysis_advanced.py --full --filter-level 4
```

#### Filter Levels:
- **Level 0 (NONE)**: No filtering - see everything
- **Level 1 (BASIC)**: Remove common English stop words
- **Level 2 (PARLIAMENTARY)**: Remove parliamentary procedural terms
- **Level 3 (MODERATE)**: Remove common verbs and vague words
- **Level 4 (AGGRESSIVE)**: Remove modal verbs and quantifiers
- **Level 5 (TFIDF)**: Use TF-IDF scoring for distinctive words
- **Level 6 (POS_NOUN)**: Keep only nouns (requires spacy)
- **Level 7 (ENTITY)**: Focus on named entities (requires spacy)

### 2. `hansard_nlp_analysis_filtered.py` - Original Style with Good Filtering

Maintains the original 4-panel visualization style with moderate filtering:

```bash
python hansard_nlp_analysis_filtered.py --years 1920-1930 --sample 1000
```

### 3. `overall_corpus_analysis_cleaned.py` - Full Corpus Analysis

Comprehensive corpus-wide analysis with filtering:

```bash
python overall_corpus_analysis_cleaned.py --sample 5000 --filtering moderate
python overall_corpus_analysis_cleaned.py --sample 5000 --filtering aggressive
```

### 4. `historical_milestone_analysis_cleaned.py` - Historical Events

Analyze key historical periods with content filtering:

```bash
# All milestones
python historical_milestone_analysis_cleaned.py --all --filtering moderate

# Specific milestone
python historical_milestone_analysis_cleaned.py --milestone 1928_full_suffrage --filtering aggressive
```

## Quick Start Examples

### Finding Substantive Content by Period

```bash
# 1920s - Inter-war period
python hansard_nlp_analysis_advanced.py --years 1920-1930 --sample 1000 --filter-level 4

# WWII period
python hansard_nlp_analysis_advanced.py --years 1939-1945 --sample 1000 --filter-level 4

# Thatcher era
python hansard_nlp_analysis_advanced.py --years 1979-1990 --sample 1000 --filter-level 4

# Post-war reconstruction
python hansard_nlp_analysis_advanced.py --years 1945-1951 --sample 1000 --filter-level 4
```

### Comparing Filter Effectiveness

```bash
# Run comparison script
./run_filter_comparison.sh

# Or manually compare specific levels
for level in 0 2 4; do
    python hansard_nlp_analysis_advanced.py --years 1920-1930 --sample 500 --filter-level $level
done
```

## Interpreting Results

### What Each Filter Level Reveals

#### Level 0 (No filtering)
- Top words: "the", "of", "to", "that", "and"
- Use for: Baseline comparison, understanding raw frequency

#### Level 2 (Parliamentary filtering)
- Top words: "would", "government", "people", "country"
- Use for: Seeing general themes with procedural language removed

#### Level 4 (Aggressive filtering)
- Top words: "unemployment", "education", "war", "trade", "tax"
- Use for: Extracting pure policy content

#### Level 5 (TF-IDF)
- Top words: Statistically distinctive terms for the period
- Use for: Finding period-specific vocabulary

## Output Files

Each analysis generates:
- **Visualizations**: 4-panel plots showing words, bigrams, gender, and temporal patterns
- **JSON results**: Complete analysis data for further processing
- **Word clouds**: Visual representation of dominant terms
- **Topic models**: LDA-discovered themes in the corpus

## Key Insights from Filtering

### Progression of Content Extraction

1. **Raw text** → 100% of words
2. **Basic filtering** → ~45% remaining (removes "the", "and", etc.)
3. **Parliamentary filtering** → ~40% remaining (removes "hon", "member", etc.)
4. **Moderate filtering** → ~35% remaining (removes "make", "take", etc.)
5. **Aggressive filtering** → ~33% remaining (policy terms only)
6. **TF-IDF filtering** → ~30% remaining (distinctive terms only)

### Best Practices

- **For general analysis**: Use Level 3 (MODERATE)
- **For policy research**: Use Level 4 (AGGRESSIVE)
- **For period comparison**: Use Level 5 (TFIDF)
- **For entity extraction**: Use Level 7 (requires spacy)

## Advanced Usage

### Installing Spacy for POS/NER Filtering

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

Then use levels 6 and 7:

```bash
# Keep only nouns
python hansard_nlp_analysis_advanced.py --years 1920-1930 --sample 500 --filter-level 6

# Extract named entities
python hansard_nlp_analysis_advanced.py --years 1920-1930 --sample 500 --filter-level 7
```

### Batch Processing

```python
# Create a Python script for batch analysis
import subprocess

periods = [
    (1914, 1918, "WWI"),
    (1939, 1945, "WWII"),
    (1945, 1951, "Post-war"),
    (1979, 1990, "Thatcher"),
    (1997, 2007, "Blair")
]

for start, end, name in periods:
    for level in [3, 4, 5]:
        cmd = f"python hansard_nlp_analysis_advanced.py --years {start}-{end} --sample 1000 --filter-level {level}"
        print(f"Running {name} at level {level}...")
        subprocess.run(cmd, shell=True)
```

## Troubleshooting

### Common Issues

1. **"Very" and "would" still appearing**: These are now filtered at Level 4+
2. **Not enough text after filtering**: Reduce filter level or increase sample size
3. **Spacy not available**: Install spacy or use levels 0-5
4. **Memory issues**: Reduce sample size or process by decade

### Performance Tips

- Small sample (100-500): Quick testing, ~30 seconds
- Medium sample (1000-5000): Robust analysis, ~2-5 minutes
- Large sample (10000+): Comprehensive analysis, ~10-30 minutes
- Full corpus: Use batching by decade, several hours

## Research Applications

### Gender Studies
```bash
# Track women's suffrage impact
python hansard_nlp_analysis_advanced.py --years 1910-1920 --sample 2000 --filter-level 4
python hansard_nlp_analysis_advanced.py --years 1920-1930 --sample 2000 --filter-level 4
```

### War and Peace Studies
```bash
# Compare pre-war, war, and post-war discourse
python hansard_nlp_analysis_advanced.py --years 1934-1939 --sample 2000 --filter-level 5
python hansard_nlp_analysis_advanced.py --years 1939-1945 --sample 2000 --filter-level 5
python hansard_nlp_analysis_advanced.py --years 1945-1950 --sample 2000 --filter-level 5
```

### Economic Policy Evolution
```bash
# Track economic terminology
python hansard_nlp_analysis_advanced.py --years 1929-1935 --sample 2000 --filter-level 4  # Depression
python hansard_nlp_analysis_advanced.py --years 1945-1951 --sample 2000 --filter-level 4  # Nationalization
python hansard_nlp_analysis_advanced.py --years 1979-1985 --sample 2000 --filter-level 4  # Privatization
```

## Next Steps

1. Run `--all-levels` to find your preferred filtering level
2. Focus on specific historical periods of interest
3. Compare results across different eras
4. Export JSON results for further statistical analysis
5. Use filtered text for machine learning models