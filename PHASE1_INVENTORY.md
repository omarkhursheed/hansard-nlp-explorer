# Phase 1: File Inventory and Redundancy Analysis

## Duplicate/Redundant Files Identified

### 1. NLP Analysis Scripts (Multiple Versions)
**Location:** `src/hansard/analysis/`
- `hansard_nlp_analysis.py` (34k) - Original version
- `hansard_nlp_analysis_advanced.py` (43k) - Enhanced version with more features
- **Action:** Keep `hansard_nlp_analysis_advanced.py`, rename to `hansard_nlp_analysis.py`

### 2. Speaker Processing Scripts (10+ files with overlapping functionality)
**Location:** `src/hansard/`
- `create_mp_only_speakers.py` (9.5k)
- `create_mp_speakers_fast.py` (8.6k)
- `create_mp_speakers_improved.py` (15k)
- `deduplicate_speakers.py` (16k)
- `normalize_speakers.py` (15k)
- `extract_speaker_debates.py` (8.9k)
- `fix_speaker_spans.py` (3.4k)
- `check_mp_coverage.py` (6.7k)
- `mp_temporal_gender_analysis.py` (15k)
- `test_speaker_extraction.py` (3.8k)
- `validate_speaker_dataset.py` (5.9k)
- **Action:** Consolidate into `speaker_processing.py` module with sub-functions

### 3. Gender/Temporal Analysis Scripts (Duplicates)
**Location:** `src/hansard/analysis/`
- `speakers_temporal_comparison.py` (17k)
- `speakers_temporal_gender_analysis.py` (15k)
- `speakers_temporal_gender_analysis_fixed.py` (20k) - Latest fixed version
- `female_mp_temporal_graph.py` (12k)
- **Action:** Keep `speakers_temporal_gender_analysis_fixed.py`, rename to `temporal_gender_analysis.py`

### 4. Parliamentary Stop Words (Multiple Implementations)
**Location:** `src/hansard/analysis/`
- `parliamentary_stop_words.py`
- `parliamentary_stop_words_enhanced.py`
- **Action:** Keep enhanced version, rename to `stop_words.py`

### 5. Milestone Analysis Scripts
**Location:** `src/hansard/analysis/`
- `comprehensive_milestone_analysis.py`
- `historical_milestone_analysis_cleaned.py` (appears to be deleted/moved)
- **Action:** Keep `comprehensive_milestone_analysis.py`

### 6. Corpus Analysis Scripts
**Location:** `src/hansard/analysis/`
- `comprehensive_corpus_analysis.py`
- `overall_corpus_analysis_cleaned.py` (appears to be deleted/moved)
- **Action:** Keep `comprehensive_corpus_analysis.py`

## Files to Archive

### 1. Non-Essential Documentation
- `src/hansard/Hansard Report V1 - Draft 08_03_2025 (3).pdf` (3.8M)
- **Action:** Move to `archive/reports/`

### 2. Test/Example Files
- `src/hansard/debate_simple_example.json`
- `src/hansard/debate_types_test.json`
- `src/hansard/debate_visualization_example.py`
- `src/hansard/test_debate_types.py`
- `src/hansard/example_usage.py`
- **Action:** Move to `archive/examples/`

### 3. Log Files
- `src/hansard/create_samples.log` (48k)
- **Action:** Move to `archive/logs/`

### 4. Old Scripts
- `src/hansard/process_debates_sample.py`
- `src/hansard/create_sampled_datasets.py`
- **Action:** Move to `archive/scripts/`

### 5. Shell Scripts (Review for relevance)
- `src/hansard/RUN_ANALYSIS.sh`
- `src/hansard/run_corpus_analysis.sh`
- `src/hansard/run_milestone_analysis.sh`
- `src/hansard/analysis/run_filter_comparison.sh`
- **Action:** Consolidate into single `run_analysis.sh` or remove if obsolete

## Files to Keep (Core Functionality)

### Crawlers
- `crawlers/crawler.py` - Main web crawler
- `crawlers/parallel_hansard_runner.py` - Parallel processing

### Parsers
- `parsers/data_pipeline.py` - Production pipeline
- `parsers/data_validation.py` - Validation
- `parsers/hansard_search.py` - Search functionality
- `parsers/metadata_extraction_test.py` - Metadata extraction
- `parsers/parse_1803_nov.py` - Enhanced parser
- `parsers/simple_parser.py` - Basic parser

### Analysis (After Consolidation)
- `analysis/hansard_nlp_analysis.py` - Main NLP analysis
- `analysis/temporal_gender_analysis.py` - Gender temporal analysis
- `analysis/comprehensive_milestone_analysis.py` - Milestone analysis
- `analysis/comprehensive_corpus_analysis.py` - Corpus analysis
- `analysis/stop_words.py` - Stop words configuration

### New Consolidated Modules
- `speaker_processing.py` - All speaker-related functionality

## Summary Statistics
- **Files to delete/consolidate:** 15+
- **Files to archive:** 10+
- **Expected reduction:** ~50% fewer files
- **Code duplication eliminated:** ~100KB