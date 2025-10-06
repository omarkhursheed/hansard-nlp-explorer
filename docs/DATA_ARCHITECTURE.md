# Hansard NLP Explorer - Data Architecture

## Overview

This document describes the data organization for the Hansard NLP Explorer project.

**Total Data:** ~30.3GB
**Space Freed:** ~17.7GB (from cleanup)

---

## Directory Structure

```
data-hansard/
├── hansard/                      [5.7GB - RAW SOURCE - NEVER DELETE]
├── processed_fixed/              [14GB - SOURCE OF TRUTH - Overall corpus]
├── gender_analysis_enhanced/     [9.1GB - SOURCE OF TRUTH - Gender corpus]
├── derived/                      [1.5GB - Generated, can regenerate]
│   ├── gender_speeches/          [Speech-level flat data]
│   └── speakers.parquet          [Speaker aggregation cache]
└── gender_wordlists/             [8KB - Gender word lists]
```

---

## Data Tiers

### Tier 1: Raw Source (NEVER DELETE)

#### `hansard/` (5.7GB)
- **Content:** Original HTML files from UK Parliament
- **Format:** HTML files organized by year
- **Purpose:** Ultimate source of truth - all other data derived from this
- **Years:** 1803-2005
- **Status:** READ-ONLY, backed up

### Tier 2: Primary Source of Truth

#### `processed_fixed/` (14GB) ← **PRIMARY SOURCE**
- **Content:** ALL parliamentary debates (complete corpus)
- **Format:** JSONL (one file per year)
- **Location:** `processed_fixed/content/YYYY/debates_YYYY.jsonl`
- **Schema:**
  ```json
  {
    "full_text": "debate text",
    "metadata": {
      "title": str,
      "speakers": [str],
      "chamber": str,
      "word_count": int,
      "year": int,
      "reference_date": str
    }
  }
  ```
- **Records:** ~800,000 debates (1803-2005)
- **Used by:**
  - `corpus_analysis.py --dataset overall` (directly)
  - `milestone_analysis.py --dataset overall` (directly)
  - Gender matching pipeline (to create gender_analysis_enhanced)
- **Generation:** Parsed from `hansard/` HTML files
- **Status:** SOURCE OF TRUTH - DO NOT DELETE

### Tier 3: Gender-Enhanced Data (Derived from processed_fixed)

#### `gender_analysis_enhanced/` (9.1GB)
- **Content:** Debates with gender-matched speakers
- **Format:** Parquet (one file per year)
- **Location:** `gender_analysis_enhanced/debates_YYYY_enhanced.parquet`
- **Schema:** 40 columns including:
  ```
  - debate_id, year, title, chamber, reference_date
  - debate_text (full text)
  - speech_segments (nested array of {speaker, text})
  - speaker_details (nested array of {name, gender: 'm'/'f'})
  - has_female, has_male, female_mps, male_mps
  - word_count, speech_count
  ```
- **Records:** ~350,000 debates with confirmed MP speakers
- **Purpose:** Source for deriving gender-specific datasets
- **Generation:** Gender matching pipeline applied to `processed_fixed/` (~2-3 hours)
- **Can regenerate:** Yes (from processed_fixed)

### Tier 4: Derived Datasets (Optimized Views - Can Regenerate)

#### `derived/gender_speeches/` (1.5GB)
- **Content:** Flat speech-level data extracted from gender corpus
- **Format:** Parquet (one file per year)
- **Location:** `derived/gender_speeches/speeches_YYYY.parquet`
- **Schema:**
  ```
  speech_id: str          # Unique ID
  debate_id: str          # Links to original debate
  year: int
  date: str
  speaker: str            # Speaker name
  gender: str             # 'm' or 'f'
  text: str               # Speech text
  word_count: int
  chamber: str
  ```
- **Total:** 2,033,211 speeches (1803-2005)
- **Used by:**
  - `corpus_analysis.py --dataset gender` (primary)
  - `milestone_analysis.py --dataset gender`
  - Builds `derived/speakers.parquet`
- **Generation:**
  ```bash
  python3 scripts/create_gender_speeches_dataset.py
  # Takes ~10-15 minutes for full corpus
  ```
- **Regeneration:** Delete and re-run script

#### `derived/gender_debates/` (2.3GB)
- **Content:** Simplified debate-level gender data
- **Format:** Parquet (one file per year)
- **Location:** `derived/gender_debates/debates_YYYY.parquet`
- **Schema:**
  ```
  debate_id: str
  year: int
  title: str
  chamber: str
  full_text: str              # Complete debate text
  speakers: list[str]
  word_count: int
  has_female: bool            # Has female MPs
  has_male: bool              # Has male MPs
  female_mps: int             # Count
  male_mps: int               # Count
  speaker_genders: dict       # {name: 'm'/'f'}
  ```
- **Total:** 348,679 debates (1803-2005)
- **Used by:**
  - `corpus_analysis.py --dataset gender-debates`
- **Purpose:** Debate-level gender comparison (male-only vs mixed vs female-heavy)
- **Generation:**
  ```bash
  python3 scripts/create_gender_debates_dataset.py
  # Takes ~10 minutes
  ```
- **Regeneration:** Delete and re-run script

#### `derived/speakers.parquet` (~3MB)
- **Content:** Speaker aggregation (career spans, speech counts)
- **Format:** Single parquet file
- **Schema:**
  ```
  normalized_name: str
  first_year: int
  last_year: int
  total_speeches: int
  ```
- **Total:** ~50,000 unique speakers (varies by dataset)
- **Used by:**
  - `temporal_gender_analysis.py`
- **Generation:** Auto-generated and cached on first run
- **Regeneration:** Delete file, will rebuild automatically

#### `gender_wordlists/` (8KB)
- **Content:** Gender-associated word lists
- **Files:**
  - `male_words.txt`
  - `female_words.txt`
- **Used by:** All gender language analysis
- **Status:** Static, manually curated

---

## Data Flow (Corrected Hierarchy)

```
hansard/ (5.7GB - Raw HTML files)
    ↓ [parsing pipeline]
    → processed_fixed/ (14GB - PRIMARY SOURCE OF TRUTH)
         │ [used directly by]
         ├→ corpus_analysis.py --dataset overall
         ├→ milestone_analysis.py --dataset overall
         │
         ↓ [gender matching pipeline ~2-3 hours]
         → gender_analysis_enhanced/ (9.1GB - Derived with gender info)
              ↓ [extract & flatten ~10-15 min]
              ├→ derived/gender_speeches/ (1.5GB - speech-level)
              │     ↓ [used by]
              │     ├→ corpus_analysis.py --dataset gender
              │     ├→ milestone_analysis.py --dataset gender
              │     └→ temporal_gender_analysis.py
              │           ↓ [aggregates to]
              │           → derived/speakers.parquet (3MB - cached)
              │
              └→ derived/gender_debates/ (2.3GB - debate-level)
                    ↓ [used by]
                    └→ corpus_analysis.py --dataset gender-debates
```

**Key Points:**
- `processed_fixed/` is the PRIMARY SOURCE (all debates, no gender)
- `gender_analysis_enhanced/` is DERIVED (processed_fixed + gender matching)
- `derived/*` are OPTIMIZED VIEWS (extracted & flattened for fast analysis)

---

## Regeneration Instructions

### Regenerate Derived Speeches
```bash
# Delete derived speeches
rm -rf data-hansard/derived/gender_speeches/

# Regenerate (10-15 minutes)
python3 scripts/create_gender_speeches_dataset.py
```

### Regenerate Speakers
```bash
# Delete cache
rm data-hansard/derived/speakers.parquet

# Automatically rebuilds on next run
python3 src/hansard/analysis/temporal_gender_analysis.py
```

### Regenerate Gender Corpus (SLOW - 2-3 hours)
```bash
# Only if gender_analysis_enhanced/ is corrupted or missing
./run_data_generation.sh
```

---

## Storage Optimization

### What We Deleted (~17.7GB)
- ✅ `processed_fixed.zip` (4.5GB)
- ✅ `hansard.zip` (4.3GB)
- ✅ `gender_analysis_enhanced.zip` (8.1GB)
- ✅ `sampled_datasets.zip` (594MB)
- ✅ Intermediate speaker files (~60MB)
- ✅ `processed_test/` (14MB)

### What We Keep
- ✅ `hansard/` (5.7GB) - Raw HTML source
- ✅ `processed_fixed/` (14GB) - Overall corpus
- ✅ `gender_analysis_enhanced/` (9.1GB) - Gender corpus
- ✅ `derived/` (1.5GB) - Fast analysis datasets
- ✅ `gender_wordlists/` (8KB) - Gender lists

---

## Dataset Comparison

| Aspect | Overall Corpus | Gender Corpus | Derived Speeches |
|--------|----------------|---------------|------------------|
| **Path** | processed_fixed/ | gender_analysis_enhanced/ | derived/gender_speeches/ |
| **Format** | JSONL | Parquet (nested) | Parquet (flat) |
| **Size** | 14GB | 9.1GB | 1.5GB |
| **Level** | Debate-level | Debate-level | Speech-level |
| **Gender** | No | Yes (nested) | Yes (flat column) |
| **Loading Speed** | Fast | Slow (nested) | Very fast |
| **Best For** | Overall analysis | Source data | Gender analysis |
| **Records** | ~1.5M debates | ~500K debates | 2.0M speeches |

---

## Best Practices

### For Analysis
1. **Use derived datasets when available** - Much faster
2. **Use appropriate filtering levels** - moderate for most, aggressive for topics
3. **Sample appropriately** - 5K-10K for quick, 50K+ for publication

### For Data Management
1. **Never delete Tier 1 or Tier 2** - These are sources of truth
2. **Derived datasets can be regenerated** - Safe to delete if space needed
3. **Keep gender_wordlists/** - Small but essential
4. **Don't create new intermediate files** - Use derived/ instead

### For Development
1. **Add new derived datasets to derived/** - Keep organized
2. **Document regeneration steps** - Include in this file
3. **Test regeneration** - Ensure reproducibility

---

## Disk Space Summary

**Before Cleanup:** ~48GB
**After Cleanup:** ~30.3GB
**Space Freed:** ~17.7GB (37% reduction)

**Breakdown:**
- Raw source: 5.7GB (12%)
- Overall corpus: 14GB (46%)
- Gender corpus: 9.1GB (30%)
- Derived datasets: 1.5GB (5%)
- Gender wordlists: 8KB (0%)

---

## Questions?

- **Where's my data?** Check tiers above
- **Can I delete X?** Check "What We Keep" section
- **How do I regenerate?** See "Regeneration Instructions"
- **What uses what?** See "Data Flow" diagram

Last updated: October 5, 2025
