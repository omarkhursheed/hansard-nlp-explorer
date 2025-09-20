# Phase 2: Progressive Testing Framework

## Test Structure
```
tests/
├── unit/                    # Basic function tests
│   ├── test_text_utils.py
│   ├── test_date_parsing.py
│   └── test_metadata.py
├── integration/             # Component interaction tests
│   ├── test_parser.py
│   ├── test_speaker_processing.py
│   └── test_analysis_pipeline.py
└── fixtures/               # Test data
    ├── sample_debate.html
    ├── sample_metadata.json
    └── sample_speakers.parquet
```

## Unit Tests to Create

### 1. Text Processing Tests (`test_text_utils.py`)
- [ ] Test text cleaning functions
- [ ] Test whitespace normalization
- [ ] Test speaker name extraction
- [ ] Test year parsing from strings

### 2. Date Parsing Tests (`test_date_parsing.py`)
- [ ] Test Hansard date format parsing
- [ ] Test date range validation
- [ ] Test edge cases (1803, 2005)

### 3. Metadata Tests (`test_metadata.py`)
- [ ] Test Hansard reference extraction
- [ ] Test chamber identification
- [ ] Test topic extraction

## Integration Tests to Create

### 4. Parser Tests (`test_parser.py`)
- [ ] Test parsing single HTML file
- [ ] Test metadata extraction from parsed content
- [ ] Test handling malformed HTML

### 5. Speaker Processing Tests (`test_speaker_processing.py`)
- [ ] Test speaker creation from debates
- [ ] Test normalization
- [ ] Test deduplication
- [ ] Test validation

### 6. Analysis Pipeline Tests (`test_analysis_pipeline.py`)
- [ ] Test NLP analysis with small dataset
- [ ] Test gender analysis
- [ ] Test temporal analysis

## Testing Principles
1. Start with smallest, simplest tests
2. Use real data samples where possible
3. Test both success and failure cases
4. Keep tests fast and isolated
5. Document what each test validates