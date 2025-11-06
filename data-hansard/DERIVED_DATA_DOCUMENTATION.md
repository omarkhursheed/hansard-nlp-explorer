# Hansard Derived Datasets Documentation

*Generated: 2025-11-05*

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Statistics](#dataset-statistics)
3. [Speeches Dataset](#speeches-dataset)
4. [Debates Dataset](#debates-dataset)
5. [Usage Examples](#usage-examples)

---

## Overview

The derived datasets combine raw Hansard transcripts with MP gender information, providing a comprehensive view of UK Parliamentary debates from 1803-2005.

### Key Features

- **Complete temporal coverage**: 203 years (1803-2005)
- **Gender-tagged speeches**: 73.76% of speeches matched to MPs with gender
- **Clean speaker names**: Canonical names derived from official MP database
- **Debate-level metadata**: Speaker composition, gender ratios, participation patterns
- **Optimized format**: Parquet files partitioned by year for efficient querying

### Data Location

```
data-hansard/derived_complete/
├── speeches_complete/           # Speech-level data
│   ├── speeches_1803.parquet
│   ├── speeches_1804.parquet
│   └── ... (201 files)
├── debates_complete/            # Debate-level metadata
│   ├── debates_1803.parquet
│   ├── debates_1804.parquet
│   └── ... (201 files)
└── dataset_summary.json         # Overall statistics
```

---

## Dataset Statistics

### Overall Dataset

- **Total debates**: 1,197,828
- **Total speeches**: 5,967,440
- **Gender match rate**: 73.76%
- **Speeches with gender**: 4,401,515 (73.76%)
- **Male speeches**: 4,263,054
- **Female speeches**: 138,461 (3.15% of gendered speeches)
- **Unmatched speeches**: 1,565,925

### Unique Speakers (Commons Only)

- **Total unique speakers**: 17,431
- **Matched to MPs**: 8,657 (49.66%)
- **Unmatched speakers**: 8,774
- **Unique male MPs**: 8,418
- **Unique female MPs**: 239

### Debate-Level Statistics (Commons Only)

- **Total debates**: 990,137
- **Debates with all speakers matched**: 489,939
- **Debates with >50% speakers matched**: 597,798
- **Debates with any female speakers**: 45,361
- **Debates with >50% female speakers**: 4,078
- **Average speakers per debate**: 2.77
- **Average match rate**: 89.8%

### Chamber Breakdown

| Chamber | Total Speeches | Total Words | Female | Male | Unmatched |
|---------|----------------|-------------|--------|------|-----------|
| Commons | 4,840,797 | 1,114,868,677 | 136,611 | 4,249,041 | 455,145 |
| Lords | 1,126,643 | 422,384,481 | 1,850 | 14,013 | 1,110,780 |

### Temporal Trends (Selected Decades)

| Decade | Total Speeches | Female % |
|--------|----------------|----------|
| 1800s | 6,337 | 0.0% |
| 1850s | 54,662 | 0.0% |
| 1900s | 258,208 | 0.0% |
| 1920s | 256,699 | 0.83% |
| 1950s | 444,564 | 2.78% |
| 1970s | 485,934 | 3.28% |
| 1990s | 465,784 | 7.82% |
| 2000s | 203,009 | 11.82% |

### Historical Milestones

| Event | Year | Before | After | Change |
|-------|------|--------|-------|--------|
| Nancy Astor elected | 1919 | 0.0% | 0.27% | +0.27pp |
| Equal Suffrage Act | 1928 | 0.72% | 2.52% | +1.80pp |
| World War II | 1942 | 1.65% | 1.71% | +0.06pp |
| Thatcher PM | 1979 | 3.23% | 4.87% | +1.64pp |
| Blair's Labour landslide | 1997 | 6.53% | 11.89% | +5.36pp |

---

## Speeches Dataset

Speech-level data with full text, speaker information, and gender tags.

### Schema

| Field | Type | Description |
|-------|------|-------------|
| `speech_id` | object | Unique identifier for the speech |
| `debate_id` | object | Parent debate identifier |
| `file_path` | object | Original HTML file path |
| `sequence_number` | int64 | Position within debate |
| `speaker` | object | Original speaker name from HTML |
| `normalized_speaker` | object | Lowercase name for searching |
| `canonical_name` | object | Clean display name (from MP database or cleaned) |
| `person_id` | object | Unique MP identifier (uk.org.publicwhip format) |
| `text` | object | Full speech text |
| `word_count` | int64 | Number of words in speech |
| `date` | object | Date of debate |
| `year` | int64 | Year of debate |
| `decade` | int64 | Decade of debate |
| `chamber` | object | Commons or Lords |
| `matched_mp` | bool | Successfully matched to MP database |
| `gender` | object | M (male), F (female), or None (unmatched) |
| `party` | object | Political party (if matched) |
| `constituency` | object | Constituency (if matched) |
| `birth_year` | float64 | MP birth year (if matched) |
| `death_year` | float64 | MP death year (if matched) |
| `ministry` | object | Government ministry (if applicable) |

### Example Record

```python
speech_id            = 'd623defe6d505b1d_speech_0'
debate_id            = 'd623defe6d505b1d'
speaker              = 'Mrs. Dunwoody'
canonical_name       = 'Gwyneth Dunwoody'
person_id            = 'uk.org.publicwhip/person/10182'
matched_mp           = True
gender               = 'F'
party                = 'Labour'
constituency         = 'Crewe and Nantwich'
word_count           = 15
date                 = '09 May 2000'
chamber              = 'Commons'
text                 = 'On a point of order, Mr. Deputy Speaker. I wonder if you could help me...'
```

### Four Name Variants

Each speech has four name-related fields:

```python
speaker              = '*MR. WILLIAM JOHNSTON'        # Original from HTML
normalized_speaker   = '*william johnston'            # Lowercase for search
canonical_name       = 'William Johnston'             # Clean display name
person_id            = 'uk.org.publicwhip/person/15496'  # Unique identifier
```

---

## Debates Dataset

Debate-level metadata with speaker composition and participation patterns.

### Schema

| Field | Type | Description |
|-------|------|-------------|
| `debate_id` | object | Unique identifier for the debate |
| `file_path` | object | Original HTML file path |
| `content_hash` | object | Hash of debate content |
| `year` | int64 | Year of debate |
| `decade` | int64 | Decade of debate |
| `month` | int64 | Month of debate |
| `date` | object | Date of debate |
| `chamber` | object | Commons or Lords |
| `title` | object | Debate title/topic |
| `topic` | object | Debate topic category |
| `hansard_reference` | object | Official Hansard reference |
| `reference_volume` | object | Hansard volume number |
| `reference_columns` | object | Hansard column numbers |
| `full_text` | object | Complete debate text |
| `word_count` | int64 | Total word count across all speeches |
| `speech_count` | int64 | Total number of speeches |
| `total_speakers` | int64 | Total number of unique speakers |
| `unique_normalized_speakers` | int64 | Unique speakers (by normalized name) |
| `unique_mp_count` | int64 | Unique MPs (by person_id) |
| `speakers` | object | List of original speaker names |
| `normalized_speakers` | object | List of normalized names (for deduplication) |
| `canonical_names` | object | List of clean display names |
| `unique_person_ids` | object | List of unique person IDs |
| `speaker_genders` | object | List of genders corresponding to speakers |
| `confirmed_mps` | int64 | Number of speakers matched to MP database |
| `female_mps` | int64 | Number of female speakers |
| `male_mps` | int64 | Number of male speakers |
| `has_female` | bool | Debate has at least one female speaker |
| `has_male` | bool | Debate has at least one male speaker |
| `gender_ratio` | float64 | Proportion of confirmed MPs who are female |

### Example Record

```python
debate_id            = 'd623defe6d505b1d'
title                = 'TRANSPORT'
date                 = '09 May 2000'
chamber              = 'Commons'
total_speakers       = 46
speech_count         = 178
confirmed_mps        = 41
male_mps             = 33
female_mps           = 8
has_female           = True
gender_ratio         = 0.195

# First 5 canonical names from this debate:
canonical_names[:5]  = ['Gwyneth Dunwoody', 'Sam Galbraith', 'Peter Pike',
                        'Lembit Öpik', 'Lynne Jones']
```

### Three Speaker Name Lists

Each debate has three parallel lists for speaker names:

```python
speakers             = ['Mrs. Dunwoody', 'Mr. Sam Galbraith', 'Mr. Peter L. Pike', ...]
normalized_speakers  = ['dunwoody', 'galbraith', 'pike', ...]
canonical_names      = ['Gwyneth Dunwoody', 'Sam Galbraith', 'Peter Pike', ...]
```

---

## Usage Examples

### Loading Data

```python
import pandas as pd

# Load all speeches from a specific year
speeches_2000 = pd.read_parquet('data-hansard/derived_complete/speeches_complete/speeches_2000.parquet')

# Load multiple years
speeches = []
for year in range(1990, 2001):
    df = pd.read_parquet(f'data-hansard/derived_complete/speeches_complete/speeches_{year}.parquet')
    speeches.append(df)
speeches = pd.concat(speeches, ignore_index=True)

# Load debates
debates_2000 = pd.read_parquet('data-hansard/derived_complete/debates_complete/debates_2000.parquet')
```

### Filtering and Analysis

```python
# Filter to matched speeches only
matched_speeches = speeches_2000[speeches_2000['matched_mp'] == True]

# Filter by gender
female_speeches = speeches_2000[speeches_2000['gender'] == 'F']
male_speeches = speeches_2000[speeches_2000['gender'] == 'M']

# Filter by chamber
commons_speeches = speeches_2000[speeches_2000['chamber'] == 'Commons']

# Get speeches by a specific MP (using person_id)
mp_speeches = speeches_2000[
    speeches_2000['person_id'] == 'uk.org.publicwhip/person/10001'
]

# Count unique speakers using canonical names
unique_speakers = matched_speeches.groupby(['person_id', 'canonical_name']).size()
```

### Temporal Analysis

```python
# Calculate female participation by year
by_year = speeches.groupby(['year', 'gender']).size().unstack(fill_value=0)
by_year['female_pct'] = by_year['F'] / (by_year['F'] + by_year['M']) * 100

# Average speech length by gender and decade
avg_length = speeches[speeches['matched_mp'] == True].groupby(
    ['decade', 'gender']
)['word_count'].mean()

# Speeches per MP over time
speeches_per_mp = speeches[speeches['matched_mp'] == True].groupby(
    ['year', 'person_id', 'canonical_name']
).size().reset_index(name='speech_count')
```

### Debate-Level Analysis

```python
# Find debates with high female participation
female_majority = debates_2000[debates_2000['gender_ratio'] > 0.5]

# Get all speeches from a specific debate
debate_id = debates_2000.iloc[0]['debate_id']
debate_speeches = speeches_2000[speeches_2000['debate_id'] == debate_id]

# Calculate debate statistics
debate_stats = debates_2000.agg({
    'total_speakers': 'mean',
    'speech_count': 'mean',
    'gender_ratio': 'mean',
    'word_count': 'sum'
})

# Find debates where women spoke
debates_with_women = debates_2000[debates_2000['has_female'] == True]
```

### Speaker Name Handling

```python
# Get all name variants for a speaker
speech = speeches_2000.iloc[0]
print(f"Original:   {speech['speaker']}")
print(f"Normalized: {speech['normalized_speaker']}")
print(f"Canonical:  {speech['canonical_name']}")
print(f"Person ID:  {speech['person_id']}")

# Use canonical names for display in visualizations
top_speakers = matched_speeches.groupby('canonical_name')['speech_id'].count().nlargest(10)

# Search using normalized names
search_term = 'churchill'
results = speeches_2000[
    speeches_2000['normalized_speaker'].str.contains(search_term, na=False)
]
```

### Joining Speeches and Debates

```python
# Merge speech data with debate metadata
speeches_with_debates = speeches_2000.merge(
    debates_2000[['debate_id', 'title', 'topic', 'hansard_reference']],
    on='debate_id',
    how='left'
)

# Analyze speech length by debate topic
speech_length_by_topic = speeches_with_debates.groupby('topic')['word_count'].mean()
```

### Gender Analysis Examples

```python
# Top female MPs by speech count
female_mps = matched_speeches[matched_speeches['gender'] == 'F']
top_female = female_mps.groupby('canonical_name').agg({
    'speech_id': 'count',
    'word_count': 'sum',
    'year': ['min', 'max']
}).sort_values(('speech_id', 'count'), ascending=False)

# Gender ratio by decade
gender_by_decade = speeches[speeches['matched_mp'] == True].groupby(
    ['decade', 'gender']
).size().unstack(fill_value=0)
gender_by_decade['female_pct'] = (
    gender_by_decade['F'] / (gender_by_decade['F'] + gender_by_decade['M']) * 100
)

# Debates by female participation level
debates_2000['female_category'] = pd.cut(
    debates_2000['gender_ratio'],
    bins=[0, 0.1, 0.3, 0.5, 1.0],
    labels=['None/Low', 'Low', 'Medium', 'High']
)
```

---

## Notes

- **Person IDs** use the uk.org.publicwhip format for compatibility with other UK Parliamentary datasets
- **Normalized speakers** are lowercase versions used for matching and deduplication
- **Canonical names** provide clean display names: official names from MP database for matched speakers, cleaned names for unmatched
- **Unmatched speakers** include procedural roles (Speaker, Deputy Speaker), Lords members, and speakers not in MP database
- **Gender field** is None for unmatched speakers, 'M' or 'F' for matched MPs
- **Match rates** vary by time period: better coverage in modern era (1950+), limited in early periods (1803-1850)
- **Lords coverage** is limited due to lack of comprehensive Lords member database
- **Asterisks** in original speaker names (*MR. SPEAKER) come from original parliament.uk HTML
- **File paths** reference original HTML files in the hansard/ directory
- **Decade field** uses start of decade (e.g., 1990 for the 1990s)
- **Date format** is typically "DD Month YYYY" (e.g., "09 May 2000")

---

## Data Quality Notes

### Matching Quality by Time Period

| Period | Match Rate | Notes |
|--------|------------|-------|
| 1803-1850 | ~60-70% | Limited MP records, many procedural speakers |
| 1850-1900 | ~70-80% | Improving coverage |
| 1900-1950 | ~80-90% | Good coverage |
| 1950-2005 | ~90-95% | Excellent coverage |

### Known Limitations

1. **Lords matching**: Only ~1% of Lords speeches matched due to incomplete Lords database
2. **Early period names**: Historical naming conventions differ (e.g., "Mr. O'Brien" vs "Mr. O Brien")
3. **Procedural speakers**: Speaker, Deputy Speaker, Chairman not in MP database
4. **Name ambiguity**: Some common names (e.g., "Mr. Smith") may match multiple MPs
5. **Maiden names**: Female MPs may appear under different names at different times

### Recommended Filters for Analysis

```python
# High-quality matched speeches only
high_quality = speeches[
    (speeches['matched_mp'] == True) &
    (speeches['year'] >= 1900) &
    (speeches['chamber'] == 'Commons')
]

# Gender analysis (matched only)
gender_analysis = speeches[
    (speeches['matched_mp'] == True) &
    (speeches['gender'].notna())
]
```

---

*For questions or issues, please open an issue on the project repository.*
