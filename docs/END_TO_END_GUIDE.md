# Hansard NLP Explorer - Complete End-to-End Guide

## Quick Reference - What to Run

```bash
# STEP 1: Generate derived dataset (one-time, ~5-10 min for test, ~2 hours for full)
python3 scripts/create_gender_speeches_dataset.py --test      # Test mode (3 years)
python3 scripts/create_gender_speeches_dataset.py             # Full corpus (1803-2005)

# STEP 2: Run all analyses
./run_full_analysis.sh --quick     # 5 minutes - test
./run_full_analysis.sh             # 20 minutes - standard
./run_full_analysis.sh --full      # 60 minutes - full corpus

# STEP 3: Review results
ls analysis/                        # All outputs here
```

---

## Data Architecture

### Sources of Truth (DO NOT DELETE)

**1. Raw HTML** (5.7GB)
```
data-hansard/hansard/YYYY/*.html
```
- Original HTML files from UK Parliament
- Ultimate source of truth

**2. Overall Corpus** (14GB)
```
data-hansard/processed_fixed/content/YYYY/debates_YYYY.jsonl
```
- Processed from HTML
- All parliamentary debates (no gender filtering)
- Used by: corpus_analysis.py --dataset overall
- Used by: milestone_analysis.py --dataset overall

**3. Gender Corpus** (9.1GB)
```
data-hansard/gender_analysis_enhanced/debates_YYYY_enhanced.parquet
```
- Processed from HTML with gender matching
- Nested structure with speech_segments and speaker_details
- Source for derived speeches

### Derived Datasets (CAN regenerate if deleted)

**4. Gender Speeches** (~3-4GB)
```
data-hansard/derived/gender_speeches/speeches_YYYY.parquet
```
- Generated from: gender_analysis_enhanced/
- Flat speech-level data (one row per speech)
- Used by: corpus_analysis.py --dataset gender (via unified_corpus_loader)
- Columns: speech_id, debate_id, year, speaker, gender, text, word_count

**5. Speakers** (~3MB)
```
data-hansard/derived/speakers.parquet
```
- Generated from: gender_speeches/
- Speaker aggregation (first_year, last_year, total_speeches)
- Used by: temporal_gender_analysis.py
- Cached automatically on first run

---

## Complete Workflow

### Part 1: Data Preparation (One-Time)

#### If starting fresh:
```bash
# 1a. Generate gender-enhanced dataset (if not exists)
#     Run this ONLY if data-hansard/gender_analysis_enhanced/ doesn't exist
./run_data_generation.sh    # ~2-3 hours

# 1b. Generate derived speeches dataset
python3 scripts/create_gender_speeches_dataset.py    # ~10-15 minutes
```

#### If you already have gender_analysis_enhanced/:
```bash
# Just generate derived speeches
python3 scripts/create_gender_speeches_dataset.py    # ~10-15 minutes
```

### Part 2: Run Analyses

```bash
# Option A: Quick test (5 minutes)
./run_full_analysis.sh --quick

# Option B: Standard analysis (20 minutes) [RECOMMENDED]
./run_full_analysis.sh

# Option C: Full corpus (60 minutes)
./run_full_analysis.sh --full
```

### Part 3: View Results

```bash
# All outputs in analysis/
tree analysis/

# Key files:
analysis/
├── corpus_gender/
│   ├── unigram_comparison.png          # Top words by gender
│   ├── bigram_comparison.png           # Top phrases
│   ├── temporal_participation.png      # Trends over time
│   ├── topic_prevalence.png            # Topics by gender
│   └── analysis_results.json           # Raw data
├── corpus_overall/
│   └── analysis_results.json
├── milestones_gender/
│   ├── 1918_partial_suffrage/
│   ├── 1928_full_suffrage/
│   ├── ww1_period/
│   ├── ww2_period/
│   └── thatcher_period/
└── milestones_overall/
    └── [same milestones]
```

---

## What Each Script Does

### Analysis Scripts

**`corpus_analysis.py`** - Corpus-level analysis
```bash
# Gender analysis
python3 src/hansard/analysis/corpus_analysis.py \
    --dataset gender \
    --years 1990-2000 \
    --sample 5000 \
    --filtering aggressive \
    --analysis all

# Overall corpus
python3 src/hansard/analysis/corpus_analysis.py \
    --dataset overall \
    --years 1990-2000 \
    --sample 5000 \
    --filtering moderate \
    --analysis unigram,bigram,topic
```

**`milestone_analysis.py`** - Historical milestone analysis
```bash
# All milestones
python3 src/hansard/analysis/milestone_analysis.py \
    --all-milestones \
    --dataset gender \
    --filtering aggressive

# Single milestone
python3 src/hansard/analysis/milestone_analysis.py \
    --milestone ww2_period \
    --dataset overall \
    --filtering moderate \
    --sample 2000
```

**`temporal_gender_analysis.py`** - Speaker trends
```bash
# Automatically uses derived/gender_speeches/
python3 src/hansard/analysis/temporal_gender_analysis.py
```

### Data Generation Scripts

**`create_gender_speeches_dataset.py`** - Create derived speeches
```bash
# Test mode (3 years: 1995-1997)
python3 scripts/create_gender_speeches_dataset.py --test

# Full corpus (1803-2005)
python3 scripts/create_gender_speeches_dataset.py

# Specific years
python3 scripts/create_gender_speeches_dataset.py --years 1990-2000

# Force overwrite
python3 scripts/create_gender_speeches_dataset.py --force
```

---

## Testing Checklist

```bash
# 1. Test modules
python3 test_unified_modules.py

# 2. Test corpus analysis (overall)
python3 src/hansard/analysis/corpus_analysis.py \
    --dataset overall --years 1995-1996 --sample 100 --analysis unigram

# 3. Test corpus analysis (gender) - uses derived speeches
python3 src/hansard/analysis/corpus_analysis.py \
    --dataset gender --years 1995-1996 --sample 100 --analysis all

# 4. Test milestone analysis
python3 src/hansard/analysis/milestone_analysis.py \
    --milestone ww2_period --dataset overall --sample 200

# 5. Test temporal analysis - builds speakers from derived
python3 src/hansard/analysis/temporal_gender_analysis.py

# 6. Test filtering comparison
python3 src/hansard/analysis/corpus_analysis.py \
    --dataset overall --years 1995-1996 --sample 100 --compare-filtering

# 7. Full smoke test
./run_quick_test.sh
```

---

## What to Delete

### Safe to Delete (After Confirming Tests Pass)

**Old analysis outputs** (~varies):
```bash
# Delete old analysis results
rm -rf analysis/enhanced_gender_results/
rm -rf analysis/gender_corpus_results/
rm -rf analysis/gender_milestone_results/
rm -rf analysis/corpus_results/
rm -rf analysis/milestone_results/
rm -rf analysis/results_advanced/
rm -rf src/hansard/analysis/*/plots_*/
```

**Zip backups** (17.6GB):
```bash
rm data-hansard/processed_fixed.zip
rm data-hansard/hansard.zip
rm data-hansard/gender_analysis_enhanced.zip
rm data-hansard/sampled_datasets.zip
```

**Intermediate speaker files** (~100MB):
```bash
rm data-hansard/speakers_*.parquet             # All pre-built speaker files
rm data-hansard/mp_speakers_*.parquet          # MP speaker files
rm data-hansard/house_members_*.parquet        # House members files
rm data-hansard/mps_not_in_speakers.parquet
rm data-hansard/person_gender_mapping.csv
```

**Test data** (14MB):
```bash
rm -rf data-hansard/processed_test/
```

**Script backups** (after confirming everything works):
```bash
rm BACKUP_old_analysis_scripts_20251005.tar.gz
rm BACKUP_old_shell_scripts_20251005.tar.gz
```

---

## Complete Commands for Fresh Start

### Step 1: Generate ALL Datasets (IF NEEDED)

```bash
# 1a. Generate gender-enhanced dataset (if missing)
#     Only needed if data-hansard/gender_analysis_enhanced/ doesn't exist
./run_data_generation.sh    # ~2-3 hours

# 1b. Generate derived speeches dataset
python3 scripts/create_gender_speeches_dataset.py    # ~10-15 minutes
```

### Step 2: Run ALL Analyses

```bash
# Standard mode (recommended)
./run_full_analysis.sh    # ~20 minutes

# OR Full mode (more comprehensive)
./run_full_analysis.sh --full    # ~60 minutes
```

### Step 3: Verify Results

```bash
# Check outputs
ls analysis/corpus_gender/          # Should have 4-5 PNG files + JSON
ls analysis/corpus_overall/         # Should have JSON
ls analysis/milestones_gender/      # Should have 5 milestone dirs
ls analysis/                        # Should have temporal PNG + JSON

# Verify visualizations created
find analysis/ -name "*.png" | wc -l    # Should be 20+ PNG files
find analysis/ -name "*.json" | wc -l   # Should be 10+ JSON files
```

### Step 4: Clean Up Old Data

```bash
# Delete old analysis outputs
rm -rf analysis/enhanced_gender_results/
rm -rf analysis/gender_corpus_results/
rm -rf analysis/gender_milestone_results/

# Delete zip backups (17.6GB freed)
rm data-hansard/*.zip

# Delete intermediate files (~100MB freed)
rm data-hansard/speakers_*.parquet
rm data-hansard/mp_speakers_*.parquet
rm data-hansard/house_members_*.parquet
rm data-hansard/mps_not_in_speakers.parquet
rm data-hansard/person_gender_mapping.csv

# Delete test data (14MB freed)
rm -rf data-hansard/processed_test/

# Total freed: ~17.7GB
```

### Step 5: Final Testing

```bash
# Run quick test to confirm nothing broke
./run_quick_test.sh    # ~2-3 minutes

# If passes, delete script backups
rm BACKUP_old_*.tar.gz
```

---

## Timeline Summary

| Task | Time | Command |
|------|------|---------|
| Generate derived speeches | 10-15 min | `python3 scripts/create_gender_speeches_dataset.py` |
| Run standard analysis | 20 min | `./run_full_analysis.sh` |
| Clean up old data | 5 min | (see Step 4 above) |
| Final testing | 3 min | `./run_quick_test.sh` |
| **TOTAL** | **~40 minutes** | |

---

## Current Status

✅ Unified analysis system created and tested
✅ Derived speeches dataset created (3 years tested: 1995-1997)
✅ Temporal analysis updated to use derived data
✅ All tests passing

**Remaining:**
1. Generate full derived dataset (1803-2005) - ~10 minutes
2. Archive old scripts
3. Delete old analysis outputs and redundant data
4. Create final documentation
5. Final testing
