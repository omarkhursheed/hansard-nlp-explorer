# Hansard NLP Explorer

Complete UK Parliamentary debates corpus (1803-2005) with gender analysis capabilities.

## Overview

- **Complete Corpus**: 1.2M debates, 5.9M speeches, 200+ years
- **Gender Analysis**: 546K MP-matched debates, 3.1M gendered speeches
- **Completeness**: 100% of available UK Hansard data
- **Coverage**: Both Commons and Lords chambers

## Data Architecture

### Tier 1: Raw HTML (hansard/)
- 27,414 dates, 1,223,157 debates
- Original HTML from UK Parliament API
- Size: 8.2 GB

### Tier 2: Processed (processed_complete/)
- 1.2M debates with extracted text and metadata
- Format: JSONL (text) + Parquet (metadata)
- Size: 20 GB

### Tier 3: Gender-Enhanced (gender_analysis_complete/)
- 546K debates with MP matching
- Includes: gender, party, constituency data
- 224 female MPs, 7,549 male MPs identified
- Size: 13 GB

### Tier 4: Derived Datasets (derived_complete/)
- **speeches_complete/**: 5.9M speeches (ALL speakers)
- **debates_complete/**: 1.2M debates (full metadata)
- With gender attribution: 3.1M speeches (52%)
- Without: 2.8M speeches (unmatched speakers)
- Size: 9.6 GB

See `docs/DATA_ARCHITECTURE.md` for details.

## Quick Start

### Analysis Scripts

```bash
# Overall corpus analysis
python3 src/hansard/analysis/comprehensive_analysis.py

# Gender-specific analysis
python3 src/hansard/analysis/gendered_comprehensive_analysis.py

# Historical suffrage analysis
python3 src/hansard/analysis/suffrage_analysis.py
```

### Using the Datasets

```python
import pandas as pd

# Load speeches (with conversation data)
speeches = pd.read_parquet('data-hansard/derived_complete/speeches_complete/speeches_1990.parquet')

# Reconstruct a conversation
conversation = speeches[speeches['debate_id'] == 'xxx'].sort_values('position')

# Gender-specific analysis
female_speeches = speeches[speeches['gender'] == 'F']
```

## Key Features

### Conversation Analysis
- **position**: Character offset for chronological ordering
- **sequence_number**: Speech order within debate (1, 2, 3...)
- **speaker**: Speaker identification
- **gender**: Gender where matched to MP database

### Full Traceability
- **file_path**: Links to source HTML
- **debate_id**: Groups speeches by debate
- Complete chain: speech → debate → processed → raw HTML

### Data Completeness
- 100% of UK Hansard API coverage
- Both Commons and Lords chambers
- All speeches extracted (not just MP-matched)
- Gender data enriched where available

## Data Statistics

| Metric | Count |
|--------|-------|
| Total debates | 1,223,157 |
| Total speeches | 5,939,625 |
| Gendered speeches | 3,098,182 |
| Female speeches | 107,341 |
| Male speeches | 2,990,841 |
| MP-matched debates | 546,080 |
| Years covered | 1803-2005 (201 years) |

## Key Scripts

### Data Creation
- `scripts/create_unified_complete_datasets.py` - Create final speech/debate datasets
- `src/hansard/scripts/data_creation/create_enhanced_gender_dataset.py` - MP matching
- `scripts/process_hansard_fast.py` - Extract text from HTML (parallel)

### Data Collection
- `src/hansard/crawlers/crawler.py` - Fetch from UK Parliament API
- `scripts/create_complete_index_fast.py` - Index all available data
- `scripts/systematic_recrawl.py` - Re-fetch missing data

### Utilities
- `scripts/audit_against_index.py` - Verify completeness
- `scripts/cleanup_duplicate_debates.py` - Remove duplicates
- `scripts/verify_completeness.py` - Final verification

## Requirements

- Python 3.12+
- pandas, numpy, polars
- BeautifulSoup4, httpx
- NLTK, scikit-learn, matplotlib
- See `environment.yml` for complete list

## Setup

```bash
conda env create -f environment.yml
conda activate hansard
```

## License

Parliamentary data is under UK Parliamentary Copyright.
