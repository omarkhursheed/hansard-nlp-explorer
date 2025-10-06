# What to Run - Complete Guide

## 🚀 TL;DR - Run This Now

```bash
./scripts/run_full_analysis.sh
```

**That's it!** This runs everything on ALL 2 million speeches (60-90 minutes).

---

## What's Running Right Now

**Current Analysis:** Full corpus (1803-2005), NO sampling

**Processing:**
- ✓ Derived dataset already generated (2,093,143 speeches, 201 files)
- ⏳ Gender corpus analysis (ALL 2M speeches)
- ⏳ Overall corpus analysis (ALL debates)
- ⏳ 5 Gender milestones (full data each)
- ⏳ 5 Overall milestones (full data each)
- ⏳ Temporal speaker analysis

**Expected Outputs:**
- 30-40 publication-quality visualizations (PNG, 300 DPI)
- 15+ JSON result files
- 5 markdown reports

**Duration:** ~60-90 minutes

**Monitor:** `tail -f /tmp/full_analysis_run.log`

---

## Modes Available

### Quick Test (5 minutes)
```bash
./run_full_analysis.sh --quick
```
- 1,000 speech sample
- 1995-2000 only
- For testing/verification

### Full Analysis (60-90 minutes) [DEFAULT]
```bash
./run_full_analysis.sh
```
- ALL 2 million speeches
- 1803-2005 complete history
- No sampling = most accurate results

---

## What Gets Created

### After Analysis Completes:

```
analysis/
├── corpus_gender/
│   ├── unigram_comparison.png          # Top words by gender
│   ├── bigram_comparison.png           # Top phrases
│   ├── temporal_participation.png      # Trends 1803-2005
│   ├── topic_prevalence.png            # Topics by gender
│   └── analysis_results.json           # Raw data
│
├── corpus_overall/
│   └── analysis_results.json           # Overall corpus metrics
│
├── milestones_gender/
│   ├── 1918_partial_suffrage/
│   │   └── aggressive/
│   │       ├── milestone_impact.png
│   │       ├── pre_period_vocabulary.png
│   │       ├── post_period_vocabulary.png
│   │       ├── milestone_report.md
│   │       └── milestone_results.json
│   ├── 1928_full_suffrage/
│   ├── ww1_period/
│   ├── ww2_period/
│   └── thatcher_period/
│
├── milestones_overall/
│   └── [same structure]
│
└── speakers_gender_temporal_fixed.png  # Speaker trends over time
```

---

## Data Being Analyzed

### Gender Corpus (2,093,143 speeches)
- **Source:** `data-hansard/derived/gender_speeches/`
- **Years:** 1803-2005 (201 files)
- **Female speeches:** 74,191 (3.5%)
- **Male speeches:** 2,018,952 (96.5%)
- **First female MPs:** 1920

**Why 3.5% female?**
- Reflects historical reality
- Women couldn't be MPs until 1918
- 117 years of male-only data (1803-1920)
- Female representation grows from 0% → 15% by 2005

### Overall Corpus (~1.5M debates)
- **Source:** `data-hansard/processed_fixed/`
- **Format:** JSONL files
- **Years:** 1803-2005
- **All parliamentary debates** (no gender filtering)

---

## Expected Results

### Gender Analysis
**Unigrams (Top Words):**
- Male MPs: country, trade, war, industry, policy, ireland...
- Female MPs: children, education, health, welfare, housing...

**Bigrams (Top Phrases):**
- Male MPs: "united kingdom", "labour party", "board trade"...
- Female MPs: "young people", "child care", "equal rights"...

**Topics:**
- 8 topics each for male/female MPs
- Shows policy focus differences

**Temporal Trends:**
- Female participation 0% (1803) → 15% (2005)
- Clear inflection points at 1918, 1928, 1979, 1997

### Milestone Analysis
**5 Key Events:**
1. **1918 Partial Suffrage:** Women >30 can vote, enter Parliament
2. **1928 Full Suffrage:** Equal voting age (21)
3. **WW1 (1914-1918):** War discourse impact
4. **WW2 (1939-1945):** Global conflict language
5. **Thatcher Era (1979-1990):** First female PM impact

**For each:**
- Vocabulary changes (new/disappeared words)
- Gender language evolution
- Pre/during/post comparisons

---

## While You Wait (~60 min)

### Review Documentation
- `END_TO_END_GUIDE.md` - Complete usage guide
- `DATA_ARCHITECTURE.md` - Data organization
- `MIGRATION_GUIDE.md` - Old → New commands
- `FINAL_SUMMARY.md` - What was accomplished

### Check Data Structure
```bash
# Verify derived dataset
ls -lh data-hansard/derived/gender_speeches/ | head -10

# Check sizes
du -sh data-hansard/*/
```

### Prepare for Results
```bash
# Install visualization tools if needed
brew install imagemagick  # For combining images
pip install jupyter       # For notebooks
```

---

## After Analysis Completes

### 1. Check Results
```bash
# List all generated files
find analysis -name "*.png" | wc -l   # Should be 30-40 images
find analysis -name "*.json" | wc -l  # Should be 15+ JSON files

# View key visualizations
open analysis/corpus_gender/*.png
open analysis/milestones_gender/*/aggressive/*.png
```

### 2. Examine Data
```bash
# Check gender analysis results
python3 -c "
import json
with open('analysis/corpus_gender/analysis_results.json') as f:
    data = json.load(f)

print('FULL CORPUS GENDER ANALYSIS:')
print(f\"Total speeches analyzed: {len(data['temporal_data']):,}\")
print(f\"Male vocabulary: {data['male_vocab_size']:,} unique words\")
print(f\"Female vocabulary: {data['female_vocab_size']:,} unique words\")
print(f\"\\nTop 10 Male Words:\")
for word, count in data['male_unigrams'][:10]:
    print(f\"  {word:20s}: {count:,}\")
print(f\"\\nTop 10 Female Words:\")
for word, count in data['female_unigrams'][:10]:
    print(f\"  {word:20s}: {count:,}\")
"
```

### 3. Review Milestones
```bash
# Check suffrage milestone
cat analysis/milestones_gender/1928_full_suffrage/aggressive/milestone_report.md

# Check WW2 impact
cat analysis/milestones_overall/ww2_period/moderate/milestone_report.md
```

---

## Next Steps After Completion

### If Results Look Good:
```bash
# Commit the results (or not - they're in .gitignore)
git add -A
git commit -m "docs: Add final analysis results"
```

### If You Want to Rerun with Different Settings:
```bash
# Different year range
python3 src/hansard/analysis/corpus_analysis.py \
    --dataset gender --years 1980-2005 --filtering aggressive

# Different filtering level
python3 src/hansard/analysis/corpus_analysis.py \
    --dataset gender --full --filtering basic

# Single milestone
python3 src/hansard/analysis/milestone_analysis.py \
    --milestone ww2_period --dataset gender
```

---

## Troubleshooting

### Analysis Seems Stuck
```bash
# Check if still running
ps aux | grep python

# Check progress
tail -f /tmp/full_analysis_run.log

# Check memory usage
top -l 1 | grep python
```

### Out of Memory
```bash
# Kill and use --quick mode
pkill -f "run_full_analysis"
./run_full_analysis.sh --quick
```

### Want to Pause
```bash
# Kill gracefully
pkill -f "run_full_analysis"

# Resume later (will skip completed parts if output exists)
./run_full_analysis.sh
```

---

## Summary

**✅ Old analysis deleted**
**⏳ Running full analysis on ALL 2M speeches**
**📊 Expected: 30-40 visualizations + 15+ JSON files**
**⏱️ Time: 60-90 minutes**

**Monitor:** `tail -f /tmp/full_analysis_run.log`
