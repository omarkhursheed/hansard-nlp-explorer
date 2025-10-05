# Code Review Findings & Cleanup Plan

**Date:** 2025-10-04
**Status:** Requires Action
**Reproducibility:** Fixed (see FULL_PIPELINE_REPRODUCIBILITY.md)

---

## ACTUAL PIPELINE: Gender Analysis Dataset

### What's Currently In Use

```
STEP 1: Dataset Creation
  Script: scripts/data_creation/create_enhanced_gender_dataset.py
  Input:  - data/processed_fixed/content/*.jsonl (debate text)
          - data/processed_fixed/metadata/debates_master.parquet
          - data/house_members_gendered_updated.parquet (MP gender data)
  Uses:   - scripts/matching/mp_matcher_corrected.py (MP matching)
          - utils/path_utils.py (path resolution)
  Output: - data/gender_analysis_enhanced/ALL_debates_enhanced_with_text.parquet (4.8GB)
          - data/gender_analysis_enhanced/ALL_debates_enhanced_metadata.parquet (91MB)
          - data/gender_analysis_enhanced/debates_YYYY_enhanced.parquet (per year)

  Runner: scripts/run_enhanced_gender_dataset.sh

STEP 2: Gender Analysis
  Script: analysis/enhanced_gender_corpus_analysis.py
  Input:  - data/gender_analysis_enhanced/*.parquet
          - data/gender_wordlists/{male,female}_words.txt
  Uses:   - utils/path_config.py (Paths class)
          - analysis/professional_visualizations.py (optional)
  Output: - analysis/gender_corpus_results/*.png
          - Statistical summaries

  Runner: scripts/run_gender_corpus_analysis.sh

STEP 3: Milestone Analysis (Optional)
  Script: analysis/gender_milestone_analysis.py
  Input:  - data/gender_analysis_enhanced/*.parquet
  Output: - analysis/milestone_results/*/

  Runner: scripts/run_gender_milestone_analysis.sh
```

### Critical Dependencies
- `scripts/matching/mp_matcher_corrected.py` - MP name matching
- `utils/path_config.py` - Path resolution (Paths class)
- `utils/path_utils.py` - Path helpers (get_data_dir, etc.)
- `data/house_members_gendered_updated.parquet` - MP gender lookup table

---

## EXTRANEOUS FILES

### 1. Duplicate Analysis Scripts (DELETE)

**Duplicate NLP Analysis:**
```
✓ KEEP:   analysis/hansard_nlp_analysis.py (1,080 lines)
✗ DELETE: analysis/hansard_nlp_analysis_consolidated.py (1,048 lines)
  Reason: "Consolidated" version removes useful path detection
```

**Duplicate Corpus Analysis:**
```
✓ KEEP:   analysis/enhanced_gender_corpus_analysis.py (33KB, Sept 29)
✗ DELETE: analysis/gender_corpus_analysis.py (22KB, Sept 27)
  Reason: Enhanced version has better features, newer

✓ KEEP:   analysis/comprehensive_corpus_analysis.py (35KB)
✗ DELETE: analysis/overall_corpus_analysis.py (21KB)
  Reason: Comprehensive version has more complete analysis
```

**Duplicate Milestone Analysis:**
```
✓ KEEP:   analysis/comprehensive_milestone_analysis.py (43KB)
✗ DELETE: analysis/historical_milestone_analysis.py (21KB)
  Reason: Comprehensive version has better milestone coverage
```

**Impact:** 4 files deleted, ~3,100 lines removed

---

### 2. Old Debug Scripts (ARCHIVE)

```
debug_scripts/
├── ✗ backfill_missing_dates.py (original)
├── ✗ backfill_missing_dates_fast.py (iteration 2)
├── ✗ backfill_missing_dates_optimized.py (iteration 3)
└── ✓ backfill_missing_dates_final.py (KEEP - final version)
```

**Action:** Move old versions to `debug_scripts/archived/`

---

### 3. Unused Data Creation Scripts

```
scripts/data_creation/
├── ✓ create_enhanced_gender_dataset.py (USED - current pipeline)
├── ✗ create_full_gender_dataset.py (OLD - pre-enhanced version)
└── ✗ create_full_gender_dataset_resumable.py (OLD - superseded)
```

**Impact:** 2 files for archive

---

### 4. Scripts in Wrong Location

```
src/hansard/ (top-level, should not have scripts)
├── quick_test.py → tests/
├── speaker_processing.py → utils/ (or refactor into modules)
├── process_full_dataset_parallel.py → scripts/
```

---

### 5. Unused Analysis Scripts

Scripts not called by any shell script:
```
analysis/
├── female_mp_temporal_graph.py (one-off visualization?)
├── dataset_statistics.py (standalone utility)
├── hansard_audit_tool.py (standalone utility)
├── detailed_progress_estimator.py (debugging tool)
├── progress_estimator.py (old version of detailed?)
└── stop_words.py (could be moved to utils/)
```

**Action:** Move to `analysis/utilities/` subdirectory

---

## CLEANUP PLANS (5 Tasks)

### PLAN 1: Detailed Comparison of Duplicate Analysis Scripts

**Objective:** Compare duplicate files line-by-line to ensure no functionality lost

**Files to Compare:**
1. `hansard_nlp_analysis.py` vs `hansard_nlp_analysis_consolidated.py`
2. `enhanced_gender_corpus_analysis.py` vs `gender_corpus_analysis.py`
3. `comprehensive_corpus_analysis.py` vs `overall_corpus_analysis.py`
4. `comprehensive_milestone_analysis.py` vs `historical_milestone_analysis.py`

**Method:**
- Use `diff -u` to show differences
- Extract unique features from each
- Port any missing features to "keeper" version
- Verify tests still pass

**Deliverable:** Comparison matrix showing which features to port

---

### PLAN 2: File Consolidation & Deletion Plan

**Phase 1: Safe Deletions (No Dependencies)**
```bash
# Analysis duplicates
rm analysis/hansard_nlp_analysis_consolidated.py
rm analysis/gender_corpus_analysis.py
rm analysis/overall_corpus_analysis.py
rm analysis/historical_milestone_analysis.py

# Old data creation
mv scripts/data_creation/create_full_gender_dataset.py archive/
mv scripts/data_creation/create_full_gender_dataset_resumable.py archive/

# Old debug scripts (keep final)
mkdir debug_scripts/archived
mv debug_scripts/backfill_missing_dates.py debug_scripts/archived/
mv debug_scripts/backfill_missing_dates_fast.py debug_scripts/archived/
mv debug_scripts/backfill_missing_dates_optimized.py debug_scripts/archived/
```

**Phase 2: Reorganization**
```bash
# Move utilities
mkdir analysis/utilities
mv analysis/{female_mp_temporal_graph,dataset_statistics,hansard_audit_tool}.py analysis/utilities/
mv analysis/{detailed_progress_estimator,progress_estimator,stop_words}.py analysis/utilities/

# Move misplaced scripts
mv src/hansard/quick_test.py src/hansard/tests/
mv src/hansard/process_full_dataset_parallel.py src/hansard/scripts/
```

**Phase 3: Update Shell Scripts**
Update any shell scripts that reference old file names.

**Deliverable:** Executable bash script for cleanup

---

### PLAN 3: Refactor speaker_processing.py

**Problem:** Monolithic 300+ line class, single responsibility violated

**Current Structure:**
```python
class SpeakerProcessor:
    create_mp_speakers()      # Data extraction
    normalize_speakers()      # Data cleaning
    deduplicate_speakers()    # Data cleaning
    extract_speaker_debates() # Querying
    validate_speaker_dataset() # Validation
    check_mp_coverage()       # Analysis
    _is_mp()                  # Heuristic (broken)
    generate_temporal_analysis() # Analysis
```

**Proposed Refactor:**
```
utils/
├── speaker_extraction.py      # Data extraction from debates
│   └── class SpeakerExtractor
├── speaker_normalization.py   # Name normalization & dedup
│   └── class SpeakerNormalizer
├── speaker_validation.py      # Dataset validation
│   └── class SpeakerValidator
└── speaker_queries.py          # Query operations
    └── class SpeakerQueries
```

**Benefits:**
- Single responsibility per module
- Easier testing
- Clear imports
- Remove broken `_is_mp()` heuristic

**Deliverable:** 4 new modules + tests

---

### PLAN 4: Test Coverage Gaps

**Current Coverage:** 26 tests passing (unit + integration)

**Gaps Identified:**
1. **No tests for** data creation scripts
   - `create_enhanced_gender_dataset.py` (600+ lines, untested)
   - `mp_matcher_corrected.py` (critical, needs tests)

2. **No tests for** analysis outputs
   - Analysis scripts generate plots but don't verify correctness
   - No regression tests for analysis results

3. **Limited crawler tests**
   - Integration tests exist but no error path testing
   - No rate limiting tests

**Proposed Test Suite:**
```
tests/
├── unit/
│   ├── test_mp_matcher.py (NEW)
│   ├── test_speaker_extraction.py (NEW)
│   └── test_path_utils.py (NEW)
├── integration/
│   ├── test_gender_dataset_creation.py (NEW)
│   ├── test_analysis_pipeline.py (NEW)
│   └── test_crawler_errors.py (NEW)
└── regression/
    └── test_analysis_consistency.py (NEW - verify output shapes)
```

**Deliverable:** Test suite with 50+ tests, 80%+ coverage

---

### PLAN 5: Shell Script Review & Issues

**Issues Found:**

1. **Inconsistent error handling**
   ```bash
   # Some scripts have set -e, others don't
   # Some check for errors, others silently fail
   ```

2. **Path assumptions**
   ```bash
   # Many scripts assume they're run from specific directories
   # Should use SCRIPT_DIR pattern consistently
   ```

3. **Duplicate shell scripts**
   ```bash
   run_gender_corpus_analysis.sh  # Uses gender_corpus_analysis.py (old)
   run_corpus_analysis.sh         # Uses comprehensive_corpus_analysis.py (different)
   # Naming is confusing
   ```

4. **No validation**
   ```bash
   # Don't check if input files exist
   # Don't verify conda environment before running
   ```

**Proposed Standards:**
```bash
#!/bin/bash
set -euo pipefail  # Fail on error, undefined vars, pipe failures

# Always resolve script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Validate inputs
if [[ ! -f "data/required_file.parquet" ]]; then
    echo "ERROR: Required file not found" >&2
    exit 1
fi

# Activate environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate hansard || {
        echo "ERROR: Failed to activate hansard environment" >&2
        exit 1
    }
fi
```

**Deliverable:** Standardized shell script template + refactored scripts

---

## PRIORITY MATRIX

| Task | Impact | Effort | Priority |
|------|--------|--------|----------|
| Delete duplicate analysis scripts | High | Low | **P0** |
| Archive debug scripts | Medium | Low | **P1** |
| Reorganize misplaced files | Medium | Medium | **P1** |
| Shell script standards | Medium | Medium | **P2** |
| Refactor speaker_processing.py | High | High | **P2** |
| Add test coverage | High | High | **P3** |

---

## IMPLEMENTATION ORDER

1. **Week 1:** File cleanup (Plans 1 & 2)
   - Compare duplicates, port features
   - Delete/archive extraneous files
   - Update shell scripts to reference new names

2. **Week 2:** Refactoring (Plan 3)
   - Break up speaker_processing.py
   - Update imports across codebase
   - Ensure tests pass

3. **Week 3:** Shell script standardization (Plan 5)
   - Create template
   - Refactor all shell scripts
   - Add validation

4. **Week 4:** Test coverage (Plan 4)
   - Write unit tests for new modules
   - Add integration tests
   - Set up CI/CD with coverage reporting

---

## RISKS & MITIGATION

**Risk:** Breaking existing workflows
- **Mitigation:** Create feature branch, run full test suite before merge

**Risk:** Losing important code in duplicates
- **Mitigation:** Thorough diff review before deletion (Plan 1)

**Risk:** Shell scripts called by external tools
- **Mitigation:** Keep old scripts as deprecated wrappers for 1 release

**Risk:** Performance regression from refactoring
- **Mitigation:** Benchmark before/after refactoring

---

## SUCCESS METRICS

- [ ] Codebase reduced by 3,000+ lines
- [ ] All duplicate files removed
- [ ] Test coverage > 80%
- [ ] All shell scripts follow standard template
- [ ] Zero failing tests after refactor
- [ ] Documentation updated to reflect new structure

---

## REFERENCE FOR LATER

This document should be referenced for:
- Any file deletion decisions
- Refactoring work on speaker processing
- Writing new shell scripts (use template)
- Adding tests to uncovered areas
