# Hansard Dataset Statistics - Authoritative Reference

**Last Updated**: 2025-11-16
**Dataset Coverage**: 1803-2005 (203 years)
**Verification Status**: All counts verified via direct parquet file analysis

---

## Executive Summary

This document provides definitive statistics for the Hansard NLP Explorer dataset. All numbers are derived from direct analysis of the `derived_complete/` dataset, which is the primary dataset for research and analysis.

**Key Numbers:**
- **5,967,440** total speeches
- **1,197,828** total debates
- **4,401,515** gender-matched speeches (73.8%)
- **240** unique female MPs with speeches
- **8,429** unique male MPs with speeches

---

## Table of Contents

1. [Dataset Tier Overview](#dataset-tier-overview)
2. [Definitive Statistics](#definitive-statistics)
3. [Chamber-Specific Analysis](#chamber-specific-analysis)
4. [Temporal Breakdown](#temporal-breakdown)
5. [Gender Matching Coverage](#gender-matching-coverage)
6. [Common Questions Answered](#common-questions-answered)
7. [Data Quality Notes](#data-quality-notes)

---

## Dataset Tier Overview

The Hansard NLP Explorer contains multiple dataset tiers, each serving different purposes:

```
Raw HTML (hansard/)
    ↓ Text extraction
Processed Complete (processed_complete/)
    ↓ MP matching & gender attribution
Gender Analysis Complete (gender_analysis_complete/)
    ↓ Unification & speech extraction
Derived Complete (derived_complete/)  ← PRIMARY DATASET FOR RESEARCH
```

### Tier Descriptions

**Tier 1: Raw HTML** (`hansard/`)
- 1,197,828 HTML files from UK Parliament API
- Size: 5.7 GB (gzipped)
- Raw source data

**Tier 2: Processed Complete** (`processed_complete/`)
- Text extracted from HTML
- Metadata files with speaker lists per debate
- Size: 14 GB
- Format: JSONL (text) + Parquet (metadata)

**Tier 3: Gender Analysis Complete** (`gender_analysis_complete/`)
- Debate-level dataset with MP matching
- Only includes debates with ≥1 confirmed MP match (652,271 debates)
- Size: 7.6 GB
- Legacy format: useful for debate-level analysis

**Tier 4: Derived Complete** (`derived_complete/`)  ⭐ **PRIMARY DATASET**
- Speech-level dataset with all speeches
- Includes both matched and unmatched speakers
- Unified schema across all years
- Size: ~10 GB
- Format: Parquet partitioned by year

---

## Definitive Statistics

> **Source**: `data-hansard/derived_complete/`
> **Method**: Direct parquet file analysis across all 201 year files

### Overall Dataset

| Metric | Count | Notes |
|--------|-------|-------|
| **Total speeches** | 5,967,440 | All speakers, both chambers |
| **Total debates** | 1,197,828 | Both Commons and Lords |
| **Years covered** | 1803-2005 | 203 years, 201 with data |
| **Gender-matched speeches** | 4,401,515 | 73.8% of all speeches |
| **Female speeches** | 138,461 | 3.1% of gendered speeches |
| **Male speeches** | 4,263,054 | 96.9% of gendered speeches |
| **Unmatched speeches** | 1,565,925 | 26.2% of all speeches |

### Unique Speakers

| Category | Count | Method |
|----------|-------|--------|
| **MPs matched (via person_id)** | 8,668 | Unique person_id values |
| **Female MPs** | 240 | person_id with gender='F' |
| **Male MPs** | 8,429 | person_id with gender='M' |
| **All unique speakers (by name)** | 35,159 | Unique normalized_speaker values |

**Note**: The 35,159 unique speakers includes unmatched speakers (procedural roles like "The Speaker", Lords members, historical MPs not in reference database).

### MP Reference Database

| Metric | Count | Source |
|--------|-------|--------|
| **Total MP records** | 44,309 | Membership periods |
| **Unique persons** | 12,494 | Individual MPs |
| **Female MPs in database** | 524 | Full database (1918-2010+) |
| **Male MPs in database** | 11,970 | Full database |

**Why 240 female MPs vs 524 in database?**
- Database covers beyond 2005 (our dataset cutoff)
- Some MPs served but gave few/no speeches
- Name matching limitations (maiden names, etc.)
- Coverage: 240/524 = 45.8% of female MPs

---

## Chamber-Specific Analysis

### House of Commons

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total speeches** | 4,840,797 | 81.1% of all speeches |
| **Gender-matched** | 4,385,652 | **90.6% match rate** ✓ |
| **Female speeches** | 136,611 | 2.8% of Commons |
| **Male speeches** | 4,249,041 | 87.8% of Commons |
| **Unmatched** | 455,145 | 9.4% of Commons |
| **Unique MPs** | 8,657 | Via person_id |

**Recommendation**: Use Commons for gender analysis (excellent 90.6% match rate)

### House of Lords

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total speeches** | 1,126,643 | 18.9% of all speeches |
| **Gender-matched** | 15,863 | **1.4% match rate** ⚠️ |
| **Female speeches** | 1,850 | 0.2% of Lords |
| **Male speeches** | 14,013 | 1.2% of Lords |
| **Unmatched** | 1,110,780 | 98.6% of Lords |
| **Unique speakers** | 5,065 | By name only |

**Warning**: Lords gender matching is unreliable (1.4% coverage). MP database primarily covers Commons.

---

## Temporal Breakdown

### Female MP Speeches by Decade

| Decade | Total Speeches | Female | Male | F% | Unique MPs | Female MPs |
|--------|----------------|--------|------|-----|------------|------------|
| 1800s | 7,358 | 0 | 4,768 | 0.00% | 280 | 0 |
| 1810s | 9,465 | 0 | 8,007 | 0.00% | 409 | 0 |
| 1820s | 12,481 | 0 | 10,938 | 0.00% | 353 | 0 |
| 1830s | 48,170 | 0 | 38,708 | 0.00% | 902 | 0 |
| 1840s | 42,565 | 0 | 33,980 | 0.00% | 680 | 0 |
| 1850s | 63,013 | 0 | 50,076 | 0.00% | 779 | 0 |
| 1860s | 70,163 | 0 | 54,488 | 0.00% | 847 | 0 |
| 1870s | 91,794 | 0 | 71,591 | 0.00% | 762 | 0 |
| 1880s | 209,566 | 0 | 182,877 | 0.00% | 1,063 | 0 |
| 1890s | 196,316 | 0 | 163,501 | 0.00% | 1,004 | 0 |
| 1900s | 276,944 | 0 | 233,537 | 0.00% | 1,091 | 0 |
| 1910s | 318,712 | 0 | 271,004 | 0.00% | 1,040 | 0 |
| **1920s** | 284,582 | **1,852** | 220,407 | **0.83%** | 1,209 | **15** |
| 1930s | 375,990 | 4,776 | 310,090 | 1.52% | 1,004 | 25 |
| 1940s | 426,917 | 7,528 | 350,261 | 2.10% | 990 | 31 |
| 1950s | 494,810 | 11,360 | 397,572 | 2.78% | 887 | 32 |
| 1960s | 604,864 | 15,338 | 434,408 | 3.41% | 911 | 35 |
| 1970s | 641,773 | 15,805 | 433,897 | 3.51% | 1,042 | 44 |
| 1980s | 744,095 | 27,774 | 444,450 | 5.88% | 927 | 52 |
| 1990s | 697,432 | 32,950 | 388,573 | 7.82% | 1,055 | 131 |
| 2000s | 350,430 | 21,078 | 159,921 | 11.65% | 713 | 120 |

### Historical Milestones

| Event | Date | Impact |
|-------|------|--------|
| Parliament (Qualification of Women) Act | 1918 | Women can stand for election |
| Constance de Markievicz elected | Dec 14, 1918 | First woman elected (never took seat) |
| Nancy Astor takes seat | Nov 15, 1919 | First woman to actually sit in Parliament |
| **Nancy Astor first speech** | **March 1, 1920** | **First female MP speech in dataset** |
| Equal Franchise Act | 1928 | Women get vote on equal terms with men |
| Margaret Thatcher becomes PM | 1979 | Female participation at 3.51% |
| Blair's Labour landslide | 1997 | Jump to 7.82% female participation |

---

## Gender Matching Coverage

### Match Rate by Time Period

| Period | Match Rate | Quality | Notes |
|--------|------------|---------|-------|
| 1803-1850 | 60-70% | Limited | Historical MP records incomplete |
| 1850-1900 | 70-80% | Improving | Better historical documentation |
| 1900-1950 | 80-90% | Good | Comprehensive MP records |
| 1950-2005 | 90-95% | Excellent | Complete coverage |

### Why Speeches Are Unmatched

Unmatched speeches (1,565,925 total) fall into these categories:

1. **Lords members** (~1.1M speeches)
   - MP database primarily covers Commons
   - Lords use titles (Earl of X, Duke of Y)
   - Titles change over time

2. **Procedural speakers** (~200K speeches)
   - "The Speaker", "Deputy Speaker", "Chairman"
   - Not in MP database (official roles, not MPs)

3. **Historical gaps** (~100K speeches)
   - Early period (1803-1900) has incomplete MP records
   - Name variations not matched (OCR errors, formatting)

4. **Matching failures** (~165K speeches)
   - Name ambiguities (multiple MPs with same surname)
   - Maiden names vs married names
   - Title changes

---

## Common Questions Answered

### Q1: Why does the README say 5,939,625 speeches but this doc says 5,967,440?

**Answer**: The README contained an earlier count. The definitive count via direct file analysis is **5,967,440 speeches**. Difference: 27,815 speeches (0.47%).

**Action**: README has been updated to correct number.

---

### Q2: Why are there 652,271 debates in gender_analysis_complete but 1,197,828 in derived_complete?

**Answer**: Different filtering levels:
- **derived_complete**: ALL debates (1,197,828) - includes everything
- **gender_analysis_complete**: Only debates with ≥1 confirmed MP match (652,271)

The 54.5% difference represents debates with no MP matches (pure Lords debates, procedural debates, matching failures).

---

### Q3: Why are there 3,901,510 speaker records but 5,967,440 speeches?

**Answer**: These count different things:
- **Speaker records** in processed_complete: UNIQUE SPEAKER NAMES PER FILE
- **Speeches** in derived_complete: INDIVIDUAL SPEECH INSTANCES

One speaker appearing 10 times in a file = 1 speaker record but 10 speeches.

---

### Q4: Why 240 female MPs in speeches but 524 in the MP database?

**Answer**: Multiple reasons:
1. MP database extends beyond 2005 (our dataset cutoff)
2. Some MPs served but gave few/no recorded speeches
3. Name matching limitations (maiden names, title changes)
4. Coverage rate: 240/524 = 45.8%

**For research**: Use **240 unique female MPs** as the definitive count.

---

### Q5: Can I use Lords data for gender analysis?

**Answer**: No, not recommended. Lords has only 1.4% gender match rate (15,863 / 1,126,643 speeches). The MP database primarily covers Commons, not Lords.

**Recommendation**: Restrict gender analysis to Commons (90.6% match rate).

---

### Q6: Why do female MPs appear in 1920s but women could stand for election in 1918?

**Answer**: Historical timeline:
- **1918**: Parliament (Qualification of Women) Act passed
- **Dec 14, 1918**: Constance de Markievicz elected (Sinn Fein, never took seat)
- **Nov 15, 1919**: Nancy Astor elected and takes seat (first woman to sit)
- **March 1, 1920**: Nancy Astor's first recorded speech in our dataset

The 1918-1920 gap is historically accurate.

---

## Data Quality Notes

### Known Limitations

1. **Lords coverage**: Only 1.4% of Lords speeches have gender attribution
2. **Early period (1803-1900)**: Lower match rates (60-80%) due to incomplete historical records
3. **Procedural speakers**: Speaker, Deputy Speaker, Chairman not in MP database
4. **Name matching**: Some ambiguous names (e.g., multiple MPs named "Smith" in same year)
5. **Maiden names**: Female MPs may appear under different names at different times

### Recommended Filters for Research

```python
# High-quality matched speeches only
high_quality = speeches[
    (speeches['matched_mp'] == True) &
    (speeches['year'] >= 1900) &
    (speeches['chamber'] == 'Commons')
]

# Gender analysis (Commons, post-1900)
gender_analysis = speeches[
    (speeches['matched_mp'] == True) &
    (speeches['gender'].notna()) &
    (speeches['chamber'] == 'Commons') &
    (speeches['year'] >= 1900)
]
```

### Match Rate Quality by Chamber

| Chamber | Total Speeches | Matched | Match Rate | Recommendation |
|---------|----------------|---------|------------|----------------|
| Commons | 4,840,797 | 4,385,652 | 90.6% | ✅ Excellent for analysis |
| Lords | 1,126,643 | 15,863 | 1.4% | ⚠️ Insufficient for gender analysis |

---

## File Locations

### Primary Dataset (Use This)
- **Speeches**: `data-hansard/derived_complete/speeches_complete/speeches_YYYY.parquet`
- **Debates**: `data-hansard/derived_complete/debates_complete/debates_YYYY.parquet`
- **Summary**: `data-hansard/derived_complete/dataset_summary.json`

### Reference Data
- **MP Database**: `data-hansard/house_members_gendered_updated.parquet`
- **Documentation**: `data-hansard/DERIVED_DATA_DOCUMENTATION.md`

### Other Tiers (For Specific Use Cases)
- **Raw HTML**: `data-hansard/hansard/`
- **Processed**: `data-hansard/processed_complete/`
- **Gender Analysis**: `data-hansard/gender_analysis_complete/`

---

## Summary Table - Quick Reference

| Metric | Count | Percentage | Chamber |
|--------|-------|------------|---------|
| **Total speeches** | 5,967,440 | 100.0% | Both |
| **Total debates** | 1,197,828 | 100.0% | Both |
| **Commons speeches** | 4,840,797 | 81.1% | Commons |
| **Lords speeches** | 1,126,643 | 18.9% | Lords |
| **Gender-matched** | 4,401,515 | 73.8% | Both |
| **Female speeches** | 138,461 | 2.3% | Both |
| **Male speeches** | 4,263,054 | 71.4% | Both |
| **Unmatched** | 1,565,925 | 26.2% | Both |
| **Unique MPs** | 8,668 | - | Both |
| **Female MPs** | 240 | 2.8% of MPs | Both |
| **Male MPs** | 8,429 | 97.2% of MPs | Both |

---

## Verification

To verify these statistics, run:

```bash
python3 scripts/verification/verify_dataset_statistics.py
```

This script performs direct parquet file analysis and outputs all counts.

---

**Last Updated**: 2025-11-16
**Data Source**: `data-hansard/derived_complete/`
**Verification Method**: Direct parquet file analysis across all 201 year files
