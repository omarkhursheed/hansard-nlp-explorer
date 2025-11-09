# Suffrage Classification Data Sources

This document describes the data extraction and filtering process that produced the final suffrage classification dataset.

## Source Files

Located in `outputs/` directory:

### Primary Dataset
- **outputs/suffrage_reliable/speeches_reliable.parquet** (2,808 speeches)
  - HIGH confidence: 1,485 speeches (~95% precision)
  - MEDIUM confidence: 1,323 speeches (~26% precision)
  - See: outputs/suffrage_reliable/SUMMARY.txt for statistics

### Extraction Iterations

**V1: Conservative Extraction**
- Location: outputs/suffrage_commons_text_search/
- Speeches: 2,958
- Precision: ~75-80%
- Details: EXTRACTION_SUMMARY.md, QUALITY_ASSESSMENT.md

**V2: Two-Tier Extraction**
- Location: outputs/suffrage_v2/
- Tier 1 (HIGH): 5,991 speeches
- Tier 2 (MEDIUM): Additional speeches
- Combined precision: ~43%
- Used for large_sample_validation.py quality testing

**Final: Reliable (Recommended)**
- Location: outputs/suffrage_reliable/
- Combines validated HIGH and MEDIUM confidence speeches
- Overall precision: ~62%
- Used for all LLM classification

### Debate Context
- **outputs/suffrage_debates/** - Full debates containing suffrage speeches
  - See: SUMMARY.txt for debate-level statistics

## Comparison

See archived documentation:
- outputs/SUFFRAGE_DATASET_COMPARISON.md
- outputs/SUFFRAGE_EXTRACTION_COMPLETE.md

For methodology details, see: SUFFRAGE_CLASSIFICATION_METHODOLOGY.md
