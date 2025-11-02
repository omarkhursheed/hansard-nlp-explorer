# Hansard NLP Explorer

Complete UK Parliamentary debates corpus (1803-2005) with gender analysis capabilities.

## Overview

- **Complete Corpus**: 1.2M debates, 5.9M speeches, 200+ years
- **Gender Matching**: 90.6% coverage (4.4M speeches matched to MPs)
- **Female Representation**: 136,611 speeches across 1919-2005
- **Data Quality**: 99.993% accuracy, comprehensive testing
- **Coverage**: Both Commons and Lords chambers

## Data Architecture

### Tier 1: Raw HTML (hansard/)
- 1,197,828 HTML files from UK Parliament API
- Years: 1803-2005 (201 years)
- Size: 5.7 GB (gzipped)

### Tier 2: Processed (processed_complete/)
- 1,197,828 debates with extracted text and metadata
- Format: JSONL (full_text) + Parquet (metadata)
- Parallel processing with checkpointing
- Size: 14 GB

### Tier 3: Gender-Enhanced (gender_analysis_complete/)
- 638,220 debates with MP matching (53.3% of all debates)
- Includes: gender, party, constituency, speech segments
- MP matching with temporal validation
- Size: 7.6 GB

### Tier 4: Unified Datasets (derived_complete/)
- **speeches_complete/**: 5.9M individual speeches (ALL speakers)
- **debates_complete/**: 1.2M debates (unified schema)
- Commons: 4.8M speeches (90.6% with gender)
- Female: 136,611 speeches (2.82% of Commons)
- Male: 4,249,041 speeches (87.8% of Commons)
- Size: ~10 GB

All data organized in data-hansard/ directory.

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

# Load speeches for a specific year
speeches = pd.read_parquet('data-hansard/derived_complete/speeches_complete/speeches_1990.parquet')

# Filter to Commons only (higher match rate)
commons = speeches[speeches['chamber'] == 'Commons']

# Gender-specific analysis
female_speeches = commons[commons['gender'] == 'F']
male_speeches = commons[commons['gender'] == 'M']

# Reconstruct a debate conversation
debate_speeches = speeches[speeches['debate_id'] == 'xxx'].sort_values('sequence_number')

# Load multiple years
years = range(1990, 2001)
all_speeches = pd.concat([
    pd.read_parquet(f'data-hansard/derived_complete/speeches_complete/speeches_{y}.parquet')
    for y in years
])
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
| Total debates | 1,197,828 |
| Total speeches | 5,939,625 |
| Commons speeches | 4,840,797 |
| Gender-matched speeches (Commons) | 4,385,652 (90.6%) |
| Female speeches | 136,611 |
| Male speeches | 4,249,041 |
| Years covered | 1803-2005 (201 years) |

## Key Scripts

All scripts organized in `src/hansard/scripts/`:

### Data Generation Pipeline
```bash
# Step 1: Extract text from HTML
python3 src/hansard/scripts/processing/process_hansard_fast.py

# Step 2: Match MPs and add gender data
python3 src/hansard/scripts/data_creation/create_enhanced_gender_dataset.py \
  --input-dir data-hansard/processed_complete \
  --output-dir data-hansard/gender_analysis_complete

# Step 3: Create unified speech/debate datasets
python3 src/hansard/scripts/data_creation/create_unified_complete_datasets.py \
  --processed-dir data-hansard/processed_complete \
  --gender-dir data-hansard/gender_analysis_complete \
  --output-dir data-hansard/derived_complete
```

### Additional Tools

All organized in `src/hansard/scripts/`:

- **Crawling**: `crawling/` - Fetch data from UK Parliament API
- **Processing**: `processing/` - Extract text from HTML
- **Matching**: `matching/` - MP name matching with gender attribution
- **Verification**: `verification/` - Data quality checks and validation
- **Quality**: `quality/` - Testing and false positive analysis

### Analysis Scripts

Located in `src/hansard/analysis/`:

- `comprehensive_analysis.py` - Full corpus analysis
- `gendered_comprehensive_analysis.py` - Gender-specific analysis
- `suffrage_analysis.py` - Historical suffrage analysis
- `basic_analysis.py` - Simple statistical analysis
- `analysis_utils.py` - Shared utilities (stop words, preprocessing)

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
