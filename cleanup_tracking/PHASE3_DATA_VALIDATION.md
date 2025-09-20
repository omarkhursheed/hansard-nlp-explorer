# Phase 3: Data Validation Plan

## Objectives
- Verify integrity of processed data files
- Check for missing or corrupted data
- Validate data formats and schemas
- Ensure consistency across datasets

## Data Locations to Validate

### 1. Metadata (Parquet Files)
**Location:** `data/processed/metadata/`
- [ ] debates_master.parquet - Main debates dataset
- [ ] speakers_master.parquet - Speaker dataset
- [ ] Annual files (debates_YYYY.parquet)

### 2. Content (JSONL Files)
**Location:** `data/processed/content/`
- [ ] Year directories (1803-2005)
- [ ] debates_YYYY.jsonl files
- [ ] File size and format consistency

### 3. Raw Data
**Location:** `data/hansard/`
- [ ] Compressed HTML files
- [ ] Coverage by year
- [ ] File corruption check

### 4. Index
**Location:** `data/processed/index/`
- [ ] debates.db - SQLite database
- [ ] Schema validation
- [ ] Query functionality

## Validation Checks

### Schema Validation
- Parquet column types
- Required fields presence
- Data type consistency

### Data Quality
- Null value analysis
- Date range validation (1803-2005)
- Text encoding (UTF-8)
- File size anomalies

### Coverage Analysis
- Years with data
- Debates per year
- Missing periods
- Data completeness

### Cross-Reference Checks
- Metadata matches content files
- Speaker references valid
- Date consistency