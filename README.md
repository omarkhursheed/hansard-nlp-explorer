# Hansard NLP Explorer

Analysis tools for UK Parliamentary debates (1803-2005) with focus on gender analysis of MP speeches.

## Overview

- **Data**: 802K debates, 1.2B words, 201 years of parliamentary history
- **Analysis**: Comprehensive corpus analysis, gender-matched analysis, temporal trends
- **Visualizations**: Professional charts for academic presentations

## Project Structure

```
hansard-nlp-explorer/
├── data-hansard/             # Data files (63GB, not in git)
│   ├── hansard/              # Raw HTML from UK Parliament
│   ├── processed_fixed/      # Overall corpus (SOURCE OF TRUTH)
│   ├── gender_analysis_enhanced/  # Gender-tagged corpus
│   └── derived/              # Optimized views (can regenerate)
├── analysis/                 # All output files (charts, results)
│   ├── comprehensive/        # Overall corpus analysis
│   └── gendered_comprehensive/  # Gender-specific analysis
├── src/hansard/
│   ├── analysis/            # Analysis scripts
│   │   ├── comprehensive_analysis.py       # Overall corpus
│   │   ├── gendered_comprehensive_analysis.py  # Gender analysis
│   │   ├── analysis_utils.py              # Shared utilities
│   │   ├── basic_analysis.py              # Simple analysis
│   │   └── suffrage_analysis.py           # Historical analysis
│   ├── crawlers/            # Web scrapers
│   ├── parsers/             # Data processors
│   ├── scripts/             # Data creation scripts
│   └── utils/               # Shared utilities
│       ├── unified_data_loader.py         # Single data loading interface
│       └── path_config.py                 # Path management
└── tests/                   # Test suite
```

## Quick Start

### Setup
```bash
conda env create -f environment.yml
conda activate hansard
```

### Run Analysis

#### 1. Overall Corpus Analysis (All Parliamentary Debates)
```bash
# Full corpus (802K debates, ~45-60 minutes on M3 Max)
python3 src/hansard/analysis/comprehensive_analysis.py --filtering aggressive

# Quick test with sample
python3 src/hansard/analysis/comprehensive_analysis.py --years 1990-2000 --sample 5000 --filtering aggressive

# Specific year range
python3 src/hansard/analysis/comprehensive_analysis.py --years 1900-1950 --filtering moderate
```

**Outputs** (in `analysis/comprehensive/`):
- `unigrams.png` - Top 30 distinctive words
- `bigrams.png` - Top 30 distinctive phrases
- `bigrams_tfidf.png` - TF-IDF weighted bigrams
- `topic_modeling.png` - LDA topics
- `temporal_debates_per_year.png` - Temporal trends
- `temporal_words_per_year.png` - Word count trends
- `analysis_results.json` - Full results

#### 2. Gender-Specific Analysis
```bash
# Gender comparison analysis
python3 src/hansard/analysis/gendered_comprehensive_analysis.py --years 1990-2000

# Full gender corpus
python3 src/hansard/analysis/gendered_comprehensive_analysis.py
```

**Outputs** (in `analysis/gendered_comprehensive/`):
- `gender_unigrams.png` - Vocabulary by gender
- `gender_bigrams.png` - Phrases by gender
- `gender_unigrams_logodds.png` - Log-odds ratio (most distinctive)
- `gender_bigrams_logodds.png` - Log-odds ratio for phrases
- `gender_topic_modeling.png` - Topics by gender
- `gender_temporal_analysis.png` - Representation over time
- `gender_speech_counts_analysis.png` - Speech counts
- `gender_speech_proportion_analysis.png` - Proportional representation

#### 3. Historical/Suffrage Analysis
```bash
python3 src/hansard/analysis/suffrage_analysis.py
```

### Filtering Levels

Choose your stop word filtering level:
- `minimal` - NLTK stopwords only (~205 words)
- `basic` - Same as minimal
- `parliamentary` - + parliamentary terms (~260 words)
- `moderate` - + common verbs, vague words (~360 words) **[RECOMMENDED]**
- `aggressive` - + discourse markers, quantifiers, adjectives (~483 words)

## Data Architecture

### Data Tiers
1. **Raw Source** (NEVER DELETE): `hansard/` - Original HTML (5.7GB)
2. **Primary Source**: `processed_fixed/` - All debates (14GB, 802K debates)
3. **Gender-Enhanced**: `gender_analysis_enhanced/` - Gender-tagged (9.1GB, 350K debates)
4. **Derived**: `derived/` - Optimized views (1.5GB, can regenerate)

See `docs/DATA_ARCHITECTURE.md` for details.

## Analysis Features

### Text Processing
- **Unified preprocessing** with NLTK-based stop words
- **Evidence-based filtering**: All stop words justified with corpus frequency
- **Sophisticated cleaning**: Handles possessives, contractions, parliamentary artifacts
- **No arbitrary exclusions**: All filtering done at preprocessing, not visualization

### Statistical Analysis
- **N-gram frequency**: Unigrams and bigrams with raw counts
- **TF-IDF weighting**: Identify distinctive vocabulary
- **Log-odds ratio**: Gender comparison (what's most distinctive)
- **Topic modeling**: LDA with configurable topic count
- **Temporal trends**: Debates and word counts over time

### Gender Analysis
- **Gender-matched corpus**: 350K debates with confirmed MP speakers
- **Comparative analysis**: Male vs. female vocabulary and topics
- **Temporal representation**: Gender balance over 200 years
- **Speech-level analysis**: Individual speech patterns

## Key Findings

### Corpus Statistics
- **Total debates**: 802,178 (1803-2005)
- **Total words**: 1.19 billion
- **Average debate**: ~1,487 words
- **Gender-matched**: 350,000 debates with identified speakers

### Gender Representation
- First female MP: 1919 (Nancy Astor)
- Post-1997 surge: "Blair's 101 women"
- Peak representation: ~14% by 2005

### Language Patterns (Gender Analysis)
- **Male MPs**: Policy/economic focus (transport, infrastructure, investment)
- **Female MPs**: Social policy focus (health, education, children, social)
- **Distinctive vocabulary**: Log-odds analysis reveals clear topical differences

## Dependencies

```bash
# Core
pandas numpy

# NLP
nltk scikit-learn

# Visualization
matplotlib seaborn

# Testing
pytest
```

## Development

```bash
# Run all tests
pytest src/hansard/tests/ -v

# Run specific test file
pytest src/hansard/tests/test_analysis_utils.py -v

# Check paths
python3 src/hansard/utils/path_config.py

# Verify data loader
python3 -c "from src.hansard.utils.unified_data_loader import UnifiedDataLoader; loader = UnifiedDataLoader(); print('Data loader OK')"
```

## Testing

New comprehensive test suite:
- `test_analysis_utils.py` - Tests for shared analysis functions
- `test_comprehensive_analysis.py` - Tests for main analysis pipeline
- `test_corrected_matcher.py` - Tests for gender matching
- `test_speaker_extraction.py` - Tests for speaker extraction

## Code Architecture

### Unified Design Principles
1. **Single data loader** (`UnifiedDataLoader`) - One interface for all data sources
2. **Shared utilities** (`analysis_utils.py`) - DRY principle for common operations
3. **Evidence-based filtering** - All stop words justified with corpus analysis
4. **Professional visualizations** - Consistent colors, formatting, academic quality
5. **No magic numbers** - Configurable parameters, documented choices

### Stop Word Philosophy
- Use **NLTK as base** (198 standard English stop words)
- Add **domain-specific terms** based on corpus frequency analysis
- Add **missing modals** that NLTK doesn't include (would, may, shall, must, etc.)
- **No visualization-time filtering** - All filtering at preprocessing stage
- **Evidence required** - Document why each word category is filtered

## Documentation

- `CLAUDE.md` - Development guidelines and visualization standards
- `docs/DATA_ARCHITECTURE.md` - Detailed data organization
- `docs/DIRECTORY_STRUCTURE.md` - Complete project layout

## Performance Notes

**Full Corpus Analysis (802K debates)**:
- Loading: 10-15 minutes
- Preprocessing: 20-30 minutes (M3 Max)
- Analysis: 10-15 minutes
- **Total**: ~45-60 minutes

**Recommended for Quick Tests**:
```bash
# Use year ranges and samples
python3 src/hansard/analysis/comprehensive_analysis.py --years 1990-2000 --sample 10000 --filtering aggressive
# Completes in ~5-10 minutes with representative results
```

## License

Data sourced from UK Parliament Historic Hansard (public domain).
