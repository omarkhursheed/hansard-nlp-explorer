# Actual Gender Analysis Pipeline

**What's Actually Used vs What's Extraneous**

---

## THE REAL PIPELINE (3 Steps)

### Step 1: Create Enhanced Gender Dataset

**USED:**
```
scripts/data_creation/create_enhanced_gender_dataset.py  ← MAIN SCRIPT
scripts/matching/mp_matcher_corrected.py                 ← Dependency
utils/path_utils.py                                      ← Dependency
```

**INPUT:**
- `data/processed_fixed/content/*.jsonl` - Debate text (all 200+ years)
- `data/processed_fixed/metadata/debates_master.parquet` - Metadata
- `data/house_members_gendered_updated.parquet` - MP gender data

**OUTPUT:**
- `data/gender_analysis_enhanced/ALL_debates_enhanced_with_text.parquet` (4.8GB)
- `data/gender_analysis_enhanced/debates_YYYY_enhanced.parquet` (per year)

**RUNNER:**
```bash
src/hansard/scripts/run_enhanced_gender_dataset.sh
```

---

### Step 2: Analyze Gender Patterns

**USED:**
```
analysis/enhanced_gender_corpus_analysis.py  ← MAIN SCRIPT
utils/path_config.py                         ← Dependency (Paths class)
data/gender_wordlists/male_words.txt         ← Gender word lists
data/gender_wordlists/female_words.txt
```

**INPUT:**
- Files from Step 1 (`gender_analysis_enhanced/*.parquet`)

**OUTPUT:**
- `analysis/gender_corpus_results/*.png` - Visualizations
- Statistical summaries

**RUNNER:**
```bash
src/hansard/scripts/run_gender_corpus_analysis.sh
```

---

### Step 3: Milestone Analysis (Optional)

**USED:**
```
analysis/gender_milestone_analysis.py  ← MAIN SCRIPT
utils/path_config.py                   ← Dependency
```

**INPUT:**
- Files from Step 1

**OUTPUT:**
- `analysis/milestone_results/*/` - Period-specific analysis

**RUNNER:**
```bash
src/hansard/scripts/run_gender_milestone_analysis.sh
```

---

## EXTRANEOUS FILES (Can Delete/Archive)

### Duplicate Analysis Scripts
```
✗ analysis/gender_corpus_analysis.py                   (22KB) - OLD version
✗ analysis/hansard_nlp_analysis_consolidated.py        (1,048 lines) - OLD version
✗ analysis/overall_corpus_analysis.py                  (21KB) - Superseded
✗ analysis/historical_milestone_analysis.py            (21KB) - Superseded
```
**Action:** DELETE (features merged into enhanced versions)

---

### Old Dataset Creation Scripts
```
✗ scripts/data_creation/create_full_gender_dataset.py           - Pre-enhanced version
✗ scripts/data_creation/create_full_gender_dataset_resumable.py - Superseded
```
**Action:** ARCHIVE (no longer used)

---

### Old Debug Scripts
```
✗ debug_scripts/backfill_missing_dates.py           (v1)
✗ debug_scripts/backfill_missing_dates_fast.py      (v2)
✗ debug_scripts/backfill_missing_dates_optimized.py (v3)
✓ debug_scripts/backfill_missing_dates_final.py     (KEEP - final version)
```
**Action:** ARCHIVE old versions

---

### Utility Scripts (Not in Pipeline)
These are standalone tools, not part of main pipeline:
```
analysis/female_mp_temporal_graph.py      - One-off visualization
analysis/dataset_statistics.py            - Standalone reporting
analysis/hansard_audit_tool.py            - Manual QA tool
analysis/detailed_progress_estimator.py   - Performance monitoring
analysis/progress_estimator.py            - Old version
analysis/stop_words.py                    - Word list utility
```
**Action:** Move to `analysis/utilities/` (keep but organize)

---

## CRITICAL DEPENDENCIES (Don't Touch!)

These files are essential:
```
✓ scripts/matching/mp_matcher_corrected.py          - MP name matching
✓ utils/path_config.py                               - Path resolution (Paths class)
✓ utils/path_utils.py                                - Path helpers
✓ analysis/professional_visualizations.py            - Visualization library
✓ data/house_members_gendered_updated.parquet        - MP gender lookup
✓ data/gender_wordlists/                             - Gender word lists
```

---

## CORE ANALYSIS SCRIPTS (Actually Used)

These are the scripts actively used for analysis:
```
✓ analysis/enhanced_gender_corpus_analysis.py        - Main gender analysis
✓ analysis/gender_milestone_analysis.py              - Milestone analysis
✓ analysis/comprehensive_corpus_analysis.py          - Full corpus analysis
✓ analysis/comprehensive_milestone_analysis.py       - Full milestone analysis
✓ analysis/hansard_nlp_analysis.py                   - General NLP analysis
```

---

## SUMMARY

**Pipeline:**
```
Raw HTML → Processed Data → Gender Dataset → Analysis → Visualizations
           (complete)       (Step 1)         (Step 2/3)  (outputs)
```

**Actually Used:** ~10 core files + 5 dependencies
**Extraneous:** ~15 files (duplicates, old versions, debugging)
**Utilities:** ~6 files (standalone tools, should organize)

**Cleanup Impact:**
- Remove 4 duplicate analysis scripts: -3,100 lines
- Archive 5 old scripts: -2,000 lines
- Reorganize 6 utility scripts: No deletion
- **Total reduction: ~5,100 lines (29% of codebase)**
