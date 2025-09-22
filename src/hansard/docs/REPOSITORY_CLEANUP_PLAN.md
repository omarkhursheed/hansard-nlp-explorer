# Repository Cleanup Plan - Hansard NLP Explorer
Generated: 2025-09-21

## Current State Analysis

### 🟢 CORRECT/KEEP - Core Infrastructure

#### Data Sources (Correct)
- `data/processed_fixed/` - **PRIMARY SOURCE** (cleaned, validated data)
  - `metadata/` - Parquet files with structured debate metadata
  - `content/` - Full text content
- `data/house_members_gendered_updated.parquet` - **GROUND TRUTH** for MPs/gender

#### Core Modules (Keep)
- `mp_matcher_corrected.py` - **CORRECT VERSION** with verified PM dates from gov.uk
- `test_corrected_matcher.py` - Tests for the correct matcher
- `path_utils.py` - Utility for path management

#### Today's Correct Scripts (Keep)
- `create_full_gender_dataset.py` - **FIXED VERSION** with symmetrical male/female storage
- `analyze_male_mp_coverage.py` - Male MP analysis
- `deep_gender_analysis.py` - Comprehensive gender analysis
- `analyze_full_dataset.py` - Full dataset analysis

#### Documentation (Keep)
- `CLAUDE.md` - Project guidelines (updated today with data symmetry reminder)
- `MATCHING_METHODOLOGY_REPORT.md` - Comprehensive matching documentation
- `FEMALE_MP_COVERAGE_ANALYSIS.md` - Female MP coverage report
- `GENDER_ANALYSIS_SLIDES.md` - Presentation bullets

---

### 🔴 DELETE - Obsolete/Superseded Files

#### Obsolete MP Matchers (DELETE)
- `mp_matcher.py` - Initial flawed version (arbitrary surname picking)
- `mp_matcher_temporal.py` - Intermediate version
- `mp_matcher_advanced.py` - Superseded by corrected version

#### Obsolete Tests (DELETE)
- `test_mp_matching.py` - Tests for wrong matcher
- `test_matching_improvements.py` - Tests for intermediate version
- `test_matcher_coverage.py` - Old coverage tests

#### Failed/Intermediate Data Directories (DELETE)
- `data_filtered_by_actual_mp/` - First attempt with flawed matcher
- `data_filtered_by_actual_mp_sample/` - Sample with wrong matcher
- `data_filtered_by_actual_mp_FULL/` - Full dataset with wrong matcher
- `data_with_ambiguity_tracking/` - Intermediate attempt
- `debate_datasets/` - Old structure
- `gender_analysis_data/` - Incomplete version (not FULL)

#### Intermediate Processing Scripts (DELETE)
- `process_all_debates_to_mp_filtered.py` - Old processor
- `process_full_dataset_parallel.py` - Parallel version with issues
- `create_filtered_datasets.py` - Old dataset creator
- `create_ambiguity_aware_dataset.py` - Superseded approach
- `create_gender_analysis_dataset.py` - Old version
- `create_gender_dataset_simple.py` - Simplified but incomplete
- `create_decade_datasets.py` - Not needed

#### Analysis Scripts for Wrong Data (DELETE)
- `analyze_corrected_sampling.py` - For old data
- `analyze_full_corrected_matching.py` - For old matcher
- `analyze_gender_truth.py` - Redundant
- `measure_matching_improvement.py` - Old measurement

---

### 🟡 REVIEW - Examine/Debug Scripts (Decide Case-by-Case)

These were created for debugging/examination - keep if useful, delete if redundant:

#### Examination Scripts
- `examine_debate_content.py` - Check if still useful
- `examine_debate_data.py` - Check if still useful
- `examine_speakers_data.py` - Check if still useful
- `examine_truth_gender_data.py` - Probably keep
- `examine_turns_data.py` - Check if still useful
- `examine_gendered_data.py` - Check if still useful

#### Debug/Validation
- `debug_parser.py` - Keep if useful for debugging
- `data_validator.py` - Keep if validates current data
- `high_performance_processor.py` - Check if needed

#### Test Files
- `test_complete_parsing.py` - Keep if tests current parser
- `test_single_year_corrected.py` - Keep if useful
- `test_speaker_extraction.py` - Keep if tests current extraction
- `test_hp_performance.py` - Keep if tests performance

---

## Recommended Actions

### Phase 1: Backup Critical Data
```bash
# Create backup of correct data
cp -r data/processed_fixed data/processed_fixed.backup
cp data/house_members_gendered_updated.parquet data/house_members_gendered_updated.parquet.backup
```

### Phase 2: Clean Obsolete Data Directories
```bash
# Remove incorrect data directories
rm -rf data_filtered_by_actual_mp/
rm -rf data_filtered_by_actual_mp_sample/
rm -rf data_filtered_by_actual_mp_FULL/
rm -rf data_with_ambiguity_tracking/
rm -rf debate_datasets/
rm -rf gender_analysis_data/  # Keep only gender_analysis_data_FULL/
```

### Phase 3: Clean Obsolete Scripts
```bash
# Remove old matchers
rm mp_matcher.py mp_matcher_temporal.py mp_matcher_advanced.py

# Remove old processors
rm process_all_debates_to_mp_filtered.py
rm process_full_dataset_parallel.py
rm create_filtered_datasets.py
rm create_ambiguity_aware_dataset.py
rm create_gender_analysis_dataset.py
rm create_gender_dataset_simple.py
rm create_decade_datasets.py

# Remove old analysis
rm analyze_corrected_sampling.py
rm analyze_full_corrected_matching.py
rm analyze_gender_truth.py
rm measure_matching_improvement.py

# Remove old tests
rm test_mp_matching.py
rm test_matching_improvements.py
rm test_matcher_coverage.py
```

### Phase 4: Organize Remaining Files
```bash
# Create organized structure
mkdir -p scripts/matching
mkdir -p scripts/analysis
mkdir -p scripts/data_creation
mkdir -p tests
mkdir -p docs

# Move files to appropriate locations
mv mp_matcher_corrected.py scripts/matching/
mv create_full_gender_dataset.py scripts/data_creation/
mv analyze_*.py scripts/analysis/
mv test_*.py tests/
mv *.md docs/
```

### Phase 5: Update .gitignore
Add to `.gitignore`:
```
# Temporary/intermediate data
data_filtered_*/
debate_datasets/
gender_analysis_data/
*.backup

# Keep only the FULL version
!gender_analysis_data_FULL/
```

---

## Final Structure

```
hansard/
├── data/
│   ├── processed_fixed/        # PRIMARY SOURCE
│   └── house_members_gendered_updated.parquet  # GROUND TRUTH
├── gender_analysis_data_FULL/  # Current processing output
├── scripts/
│   ├── matching/
│   │   └── mp_matcher_corrected.py
│   ├── data_creation/
│   │   └── create_full_gender_dataset.py
│   └── analysis/
│       ├── analyze_male_mp_coverage.py
│       ├── analyze_full_dataset.py
│       └── deep_gender_analysis.py
├── tests/
│   └── test_corrected_matcher.py
├── docs/
│   ├── CLAUDE.md
│   ├── MATCHING_METHODOLOGY_REPORT.md
│   └── *.md (other docs)
└── utils/
    └── path_utils.py
```

---

## Verification Steps

After cleanup:
1. Run `test_corrected_matcher.py` to ensure matcher still works
2. Check that `gender_analysis_data_FULL/` is being created with male_names
3. Verify no broken imports in remaining scripts
4. Commit the cleaned state with clear message

---

## Important Notes

- **DO NOT DELETE** `data/processed_fixed/` - this is the primary data source
- **DO NOT DELETE** `mp_matcher_corrected.py` - this is the correct matcher
- **WAIT FOR** `create_full_gender_dataset.py` to complete before final verification
- **KEEP** the FULL dataset once it's created with symmetric male/female data