# Hansard NLP Explorer

Complete UK Parliamentary debates corpus (1803-2005) with gender analysis capabilities.

## Overview

- **Complete Corpus**: 1.2M debates, 6.0M speeches, 200+ years
- **Gender Matching**: 90.6% coverage in Commons (4.4M speeches matched to MPs)
- **Female Representation**: 136,611 speeches across 1920-2005 (240 unique female MPs)
- **Data Quality**: 99.993% accuracy, comprehensive testing
- **Coverage**: Both Commons and Lords chambers (gender analysis Commons-only)

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
- 652,271 debates with MP matching (54.5% of all debates)
- Includes: gender, party, constituency, speech segments
- MP matching with temporal validation
- Size: 7.6 GB

### Tier 4: Unified Datasets (derived_complete/)
- **speeches_complete/**: 6.0M individual speeches (ALL speakers, both chambers)
- **debates_complete/**: 1.2M debates (unified schema)
- Commons: 4.8M speeches (90.6% with gender)
- Female: 136,611 speeches (2.8% of Commons, 240 unique MPs)
- Male: 4,249,041 speeches (87.8% of Commons, 8,429 unique MPs)
- Lords: 1.1M speeches (1.4% gender match rate - insufficient for gender analysis)
- Size: ~10 GB

All data organized in data-hansard/ directory.

## Quick Start

### Installation

```bash
# Install the hansard package in development mode
pip install -e .
```

### Analysis Scripts

```bash
# Overall corpus analysis
python3 scripts/analysis/comprehensive_analysis.py

# Gender-specific analysis
python3 scripts/analysis/gendered_comprehensive_analysis.py

# Historical suffrage analysis
python3 scripts/analysis/suffrage_analysis.py
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

## Suffrage Classification

LLM-based stance detection and argument extraction for women's suffrage debates (1900-1935).

**Key Results**:
- 2,808 speeches classified with 100% API success rate
- 92.9% accuracy (manual validation on 48-speech stratified sample)
- 5,138 arguments extracted across 9 taxonomy categories
- Coverage: 1,194 FOR (42.5%), 869 AGAINST (30.9%), 109 BOTH (3.9%), 96 NEUTRAL (3.4%), 540 IRRELEVANT (19.2%)
- Gender breakdown: 2,535 male MPs, 83 female MPs

**Methodology**:
- Two-tier suffrage detection (HIGH: explicit terms ~95% precision, MEDIUM: proximity matching ~26% precision)
- Prompt evolution through 5 versions (v1: full debate -> v5: active context with source labeling)
- Context window optimization (context=3 speeches found optimal, 41% reduction in false IRRELEVANT)
- LLM: gpt-4o-mini via OpenRouter, deployed on Modal.com serverless platform
- Cost: $4.11 total (Victorian speeches 7.5x longer than modern due to lack of time limits)

**Historical Findings**:
- WWI impact: 95% drop in suffrage speeches (1913: 383 speeches -> 1915: 20 speeches)
- 1917 revival: 296 speeches for 1918 Representation of the People Act debates
- Government obstruction was primary barrier (bills passed second readings but lacked government support to progress)
- Victorian parliamentary culture: No enforced time limits until 1900 motion for 20-minute cap

**Documentation**:
- Full methodology: [docs/suffrage_classification/SUFFRAGE_CLASSIFICATION_METHODOLOGY.md](docs/suffrage_classification/SUFFRAGE_CLASSIFICATION_METHODOLOGY.md)
- Validation results: [docs/suffrage_classification/MANUAL_VALIDATION_SUMMARY.md](docs/suffrage_classification/MANUAL_VALIDATION_SUMMARY.md)
- False positive analysis: [docs/suffrage_classification/FALSE_POSITIVE_ANALYSIS.md](docs/suffrage_classification/FALSE_POSITIVE_ANALYSIS.md)
- Setup guide: [docs/suffrage_classification/SETUP_MODAL_CLASSIFICATION.md](docs/suffrage_classification/SETUP_MODAL_CLASSIFICATION.md)
- Analysis notebook: [notebooks/suffrage_classification_analysis.ipynb](notebooks/suffrage_classification_analysis.ipynb)
- Prompts: [prompts/](prompts/) (v4 and v5)

**Data**:
- Classification results: outputs/llm_classification/full_results_v5_context_3_complete.parquet
- Visualization outputs: analysis/suffrage_classification/*.png
- Validation samples: outputs/validation/

**Scripts**:
- Classification pipeline: [scripts/classification/](scripts/classification/)
- Quality validation: [scripts/quality/](scripts/quality/)
- Utilities: [scripts/utilities/](scripts/utilities/)

## Key Features

### Conversation Analysis
- **position**: Character offset for chronological ordering
- **sequence_number**: Speech order within debate (1, 2, 3...)
- **speaker**: Speaker identification
- **gender**: Gender where matched to MP database

### Full Traceability
- **file_path**: Links to source HTML
- **debate_id**: Groups speeches by debate
- Complete chain: speech -> debate -> processed -> raw HTML

### Data Completeness
- 100% of UK Hansard API coverage
- Both Commons and Lords chambers
- All speeches extracted (not just MP-matched)
- Gender data enriched where available

## Data Statistics

> **For detailed statistics and explanations, see [docs/DATASET_STATISTICS.md](docs/DATASET_STATISTICS.md)**

| Metric | Count |
|--------|-------|
| Total debates | 1,197,828 |
| Total speeches | 5,967,440 |
| Commons speeches | 4,840,797 |
| Lords speeches | 1,126,643 |
| Gender-matched speeches (Commons) | 4,385,652 (90.6%) |
| Gender-matched speeches (Lords) | 15,863 (1.4%) |
| Female speeches (Commons) | 136,611 |
| Male speeches (Commons) | 4,249,041 |
| Unique female MPs | 240 |
| Unique male MPs | 8,429 |
| Years covered | 1803-2005 (203 years) |

**Note**: Gender analysis is reliable for Commons only (90.6% match rate). Lords has insufficient coverage (1.4% match rate).

## Repository Structure

```
hansard-nlp-explorer/
  src/hansard/          # Library code (importable modules)
    utils/              # Path config, data loaders
    matching/           # MP matching algorithms
    parsers/            # HTML parsing
    analysis/           # Analysis utilities
  scripts/              # Executable scripts
    crawling/           # Data collection from Parliament API
    processing/         # HTML text extraction
    data_creation/      # Dataset generation
    matching/           # MP matching pipelines
    classification/     # LLM-based classification
    analysis/           # Analysis scripts
    manuscript/         # Figure generation
    quality/            # Quality validation
  docs/                 # All documentation
  tests/                # All tests
  notebooks/            # Jupyter notebooks
  prompts/              # LLM prompts
  data-hansard/         # Data files (63GB)
  outputs/              # Generated outputs
```

## Key Scripts

### Data Generation Pipeline
```bash
# Step 1: Extract text from HTML
python3 scripts/processing/process_hansard_fast.py

# Step 2: Match MPs and add gender data
python3 scripts/data_creation/create_enhanced_gender_dataset.py \
  --input-dir data-hansard/processed_complete \
  --output-dir data-hansard/gender_analysis_complete

# Step 3: Create unified speech/debate datasets
python3 scripts/data_creation/create_unified_complete_datasets.py \
  --processed-dir data-hansard/processed_complete \
  --gender-dir data-hansard/gender_analysis_complete \
  --output-dir data-hansard/derived_complete
```

### Additional Tools

All organized in `scripts/`:

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
