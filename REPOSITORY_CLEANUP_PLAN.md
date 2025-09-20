# Repository Cleanup and Testing Plan for Hansard NLP Explorer

## Executive Summary
Complete repository reorganization with comprehensive testing, data validation, and final analysis generation. Focus on minimal, self-contained tests that progressively validate functionality from basic operations to full analysis pipelines.

## Phase 1: Repository Assessment and Organization

### 1.1 Current State Analysis
**Objective**: Document existing functionality and identify redundancy

#### Core Components to Assess:
1. **Crawlers** (`crawlers/`)
   - `crawler.py` - Main web crawler v4.2
   - `parallel_hansard_runner.py` - Multi-strategy parallel crawler
   - Test coverage and validation needs

2. **Parsers** (`parsers/`)
   - `data_pipeline.py` - Production pipeline
   - `data_validation.py` - Validation framework
   - `hansard_search.py` - Search functionality
   - `metadata_extraction_test.py` - Metadata extraction
   - `parse_1803_nov.py` - Enhanced parser
   - `simple_parser.py` - Basic parser
   - Identify redundant parsers and consolidate

3. **Analysis** (`analysis/`)
   - Multiple versions of NLP analysis scripts (need consolidation)
   - Historical milestone analysis
   - Gender analysis components
   - Parliamentary stop words implementations

4. **Ad-hoc Scripts** (root level)
   - Speaker-related scripts (10+ files - need consolidation)
   - MP coverage and temporal analysis
   - Test scripts and validation utilities

### 1.2 Cleanup Actions
1. **Remove redundant files**:
   - Multiple versions of same functionality (e.g., `hansard_nlp_analysis_*.py`)
   - Obsolete test files
   - Debug HTML files

2. **Consolidate functionality**:
   - Merge speaker processing scripts into single module
   - Combine stop words implementations
   - Unify gender analysis scripts

3. **Remove all emojis** from:
   - README.md
   - All Python docstrings and comments
   - Analysis output files
   - Documentation

4. **Archive non-essential files**:
   - PDF report
   - Old analysis results
   - Debug scripts (move to archive/)

## Phase 2: Progressive Testing Framework

### 2.1 Unit Tests - Core Functions
**Principle**: Test smallest units first, build confidence progressively

#### Level 1: Basic Utilities (No External Dependencies)
```python
tests/unit/
├── test_text_processing.py
│   - clean_text()
│   - normalize_whitespace()
│   - extract_year_from_string()
├── test_date_parsing.py
│   - parse_hansard_date()
│   - validate_date_range()
├── test_metadata_extraction.py
│   - extract_speaker_name()
│   - parse_hansard_reference()
│   - extract_chamber_info()
```

#### Level 2: Data Structure Tests
```python
tests/unit/
├── test_debate_structure.py
│   - validate_debate_schema()
│   - check_required_fields()
├── test_speaker_structure.py
│   - validate_speaker_format()
│   - check_speaker_normalization()
```

### 2.2 Integration Tests - Component Interaction

#### Level 3: Parser Tests (With Sample Data)
```python
tests/integration/
├── test_parser_basic.py
│   - Parse single known HTML file
│   - Validate output structure
│   - Check metadata extraction
├── test_parser_edge_cases.py
│   - Handle malformed HTML
│   - Process empty debates
│   - Test year boundary cases (1803, 2005)
```

#### Level 4: Data Pipeline Tests
```python
tests/integration/
├── test_pipeline_flow.py
│   - Process small dataset (10 files)
│   - Validate Parquet output
│   - Check JSONL generation
├── test_search_functionality.py
│   - Query processed data
│   - Test date range filtering
│   - Validate speaker search
```

### 2.3 Analysis Tests - NLP Components

#### Level 5: NLP Processing Tests
```python
tests/analysis/
├── test_nlp_basic.py
│   - Test with 10 known debates
│   - Validate unigram extraction
│   - Check bigram generation
├── test_topic_modeling.py
│   - Small corpus LDA (50 debates)
│   - Validate topic coherence
│   - Check reproducibility
├── test_gender_analysis.py
│   - Test wordlist loading
│   - Validate ratio calculations
│   - Check temporal splitting
```

### 2.4 End-to-End Tests

#### Level 6: Full Pipeline Tests
```python
tests/e2e/
├── test_complete_workflow.py
│   - Sample data → Parse → Process → Analyze
│   - Validate all outputs
│   - Performance benchmarks
```

## Phase 3: Data Validation and Inspection

### 3.1 Data Integrity Checks
1. **Raw Data Validation**
   ```python
   validation/
   ├── check_html_completeness.py
   │   - Verify all years present
   │   - Check file corruption
   │   - Validate compression
   ├── check_temporal_coverage.py
   │   - Identify gaps in data
   │   - Validate date continuity
   ```

2. **Processed Data Validation**
   ```python
   validation/
   ├── validate_parquet_integrity.py
   │   - Schema consistency
   │   - Data type validation
   │   - Null value analysis
   ├── validate_jsonl_format.py
   │   - JSON structure validation
   │   - UTF-8 encoding checks
   ```

### 3.2 Statistical Data Analysis
```python
analysis/data_quality/
├── corpus_statistics.py
│   - Total debates by year
│   - Word count distributions
│   - Speaker participation metrics
├── coverage_analysis.py
│   - Temporal coverage gaps
│   - Chamber representation
│   - Data completeness metrics
```

## Phase 4: Code Consolidation and Optimization

### 4.1 Module Reorganization
```
src/hansard/
├── core/
│   ├── __init__.py
│   ├── parser.py          # Unified parser
│   ├── crawler.py         # Single crawler implementation
│   └── validator.py       # Data validation utilities
├── processing/
│   ├── __init__.py
│   ├── pipeline.py        # Main processing pipeline
│   ├── text_cleaner.py    # Text preprocessing
│   └── metadata.py        # Metadata extraction
├── analysis/
│   ├── __init__.py
│   ├── nlp_analyzer.py    # Consolidated NLP analysis
│   ├── gender_analyzer.py # Gender language analysis
│   └── temporal.py        # Temporal analysis
├── search/
│   ├── __init__.py
│   └── query_engine.py    # Search functionality
└── utils/
    ├── __init__.py
    ├── io_helpers.py      # File I/O utilities
    └── constants.py       # Project constants
```

### 4.2 Performance Optimizations
1. **Vectorized Operations**
   - Replace loops with pandas/numpy operations
   - Use Parquet for large data reads
   - Implement batch processing

2. **Memory Management**
   - Streaming for large files
   - Chunked processing
   - Proper resource cleanup

3. **Parallel Processing**
   - Multiprocessing for CPU-bound tasks
   - Async I/O for network operations
   - Progress bars for long operations

## Phase 5: Final Analysis and Documentation

### 5.1 Comprehensive Analysis Run
```bash
# Run analyses with increasing complexity
python -m hansard.analysis.nlp_analyzer --test-mode  # Small test
python -m hansard.analysis.nlp_analyzer --years 1920-1930 --sample 500
python -m hansard.analysis.nlp_analyzer --years 1900-1950 --sample 5000
python -m hansard.analysis.nlp_analyzer --full --sample 10000
```

### 5.2 Generate Visualizations
```
results/
├── temporal/
│   ├── debate_frequency_timeline.png
│   ├── speaker_participation_trends.png
│   └── word_count_evolution.png
├── linguistic/
│   ├── top_terms_by_decade.png
│   ├── topic_evolution_heatmap.png
│   └── bigram_networks.png
├── gender/
│   ├── gender_language_ratios.png
│   ├── pre_post_1928_comparison.png
│   └── gender_term_frequency.png
└── summary/
    ├── corpus_overview.png
    └── key_insights.png
```

### 5.3 Final Documentation
```
docs/
├── API_REFERENCE.md       # Complete API documentation
├── DATA_DICTIONARY.md     # Data schema and field descriptions
├── ANALYSIS_GUIDE.md      # How to run analyses
├── PERFORMANCE_REPORT.md  # Benchmarks and optimization notes
└── INSIGHTS_SUMMARY.md    # Key findings from analysis
```

## Phase 6: Repository Finalization

### 6.1 Clean Repository Structure
```
hansard-nlp-explorer/
├── src/hansard/           # Core application code
├── tests/                 # All test files
├── validation/            # Data validation scripts
├── results/              # Analysis outputs
├── docs/                 # Documentation
├── scripts/              # Utility scripts
├── data/                 # Data directory (gitignored)
├── .github/              # CI/CD workflows
├── README.md             # Clean, emoji-free documentation
├── environment.yml       # Environment specification
├── pyproject.toml        # Project configuration
└── LICENSE              # License file
```

### 6.2 Final Checklist
- [ ] All tests passing (unit, integration, e2e)
- [ ] No duplicate functionality
- [ ] All emojis removed
- [ ] Code follows consistent style (PEP 8)
- [ ] Documentation complete and accurate
- [ ] Performance benchmarks documented
- [ ] Data validation reports generated
- [ ] Analysis results reproducible
- [ ] Repository structure logical and clean

## Implementation Timeline

### Week 1: Assessment and Cleanup
- Day 1-2: File inventory and redundancy analysis
- Day 3-4: Remove duplicates, consolidate functions
- Day 5: Remove emojis, clean documentation

### Week 2: Testing Framework
- Day 1-2: Unit tests for core functions
- Day 3-4: Integration tests for components
- Day 5: Analysis module tests

### Week 3: Data Validation
- Day 1-2: Raw data validation
- Day 3-4: Processed data checks
- Day 5: Statistical analysis

### Week 4: Final Analysis
- Day 1-2: Run comprehensive analyses
- Day 3-4: Generate visualizations
- Day 5: Document insights and finalize

## Success Metrics
1. **Test Coverage**: >80% for core modules
2. **Performance**: 10x speed improvement for common operations
3. **Data Quality**: 100% validation pass rate
4. **Code Reduction**: 50% fewer files through consolidation
5. **Documentation**: Complete API reference and user guides

## Risk Mitigation
1. **Data Loss**: Create backups before any destructive operations
2. **Breaking Changes**: Maintain backward compatibility where needed
3. **Performance Regression**: Benchmark before and after changes
4. **Missing Functionality**: Document all removed features

## Notes
- Focus on minimal, necessary code
- Write tests before refactoring
- Delete rather than comment out old code
- Prioritize clarity over cleverness
- Never generate synthetic data or fake visualizations
- Always validate with real data