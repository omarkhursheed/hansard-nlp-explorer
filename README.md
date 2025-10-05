# Hansard NLP Explorer

Analysis tools for UK Parliamentary debates (1803-2005) with focus on gender analysis of MP speeches.

## Overview

- **Data**: 802K debates, 2.4M speaker records, 201 years of parliamentary history
- **Analysis**: Gender-matched corpus analysis, temporal trends, milestone events
- **Visualizations**: Professional charts for academic presentations

## Project Structure

```
hansard-nlp-explorer/
├── data-hansard/             # Data files (not in git)
│   ├── gender_analysis_enhanced/  # Enhanced dataset with full text
│   ├── gender_wordlists/          # Gender word lists
│   └── processed_fixed/           # Cleaned debate data
├── analysis/                 # All output files (charts, results)
├── src/hansard/
│   ├── analysis/            # Analysis scripts
│   ├── crawlers/            # Web scrapers
│   ├── parsers/             # Data processors
│   ├── scripts/             # Utility scripts
│   └── utils/               # Shared utilities
└── tests/                   # Test suite
```

## Quick Start

### Setup
```bash
conda env create -f environment.yml
conda activate hansard
```

### Run Complete Analysis
```bash
# Run everything with one command
./run_complete_analysis.sh

# Or run individually:
# 1. Overall gender corpus analysis
python3 src/hansard/analysis/enhanced_gender_corpus_analysis.py --full --sample 50000 --filtering aggressive

# 2. Milestone analysis (suffrage, WWI, WWII, Thatcher, Blair)
python3 src/hansard/analysis/gender_milestone_analysis.py --all --filtering aggressive
```

### Individual Analyses
```bash
# Test with small sample
python3 src/hansard/analysis/enhanced_gender_corpus_analysis.py --years 1990-1991 --sample 1000

# Specific milestone
python3 src/hansard/analysis/gender_milestone_analysis.py --milestone ww2_period --filtering aggressive
```

## Output Files

All outputs go to `analysis/`:
- `temporal_representation.png` - Gender trends over time (1803-2005)
- `vocabulary_comparison.png` - Distinctive words by gender
- `bigram_comparison.png` - Distinctive phrases by gender
- `topic_distribution.png` - Topic modeling results
- `statistical_summary.png` - Key statistics
- `*.json` - Numerical results

## Data

- **Location**: `data-hansard/` (13GB, not in repository)
- **Enhanced Dataset**: 354K debates with speaker gender identification
- **Coverage**: 218 female MPs, 7,396 male MPs (1803-2005)

## Key Findings

### Gender Representation
- First female MP: 1919 (Nancy Astor)
- Post-1997 surge: "Blair's 101 women"
- Peak representation: ~14% by 2005

### Language Patterns
- Male MPs: Dominated by policy/economic terms (country, act, trade)
- Female MPs: Focus on social issues (local, health, service, education)
- Distinctive bigrams reveal procedural vs. social policy focus

## Dependencies

```bash
# Core
pandas polars numpy

# NLP
scikit-learn gensim

# Visualization
matplotlib seaborn

# Optional
spacy  # For advanced NLP features
```

## Development

```bash
# Run tests
pytest tests/

# Check paths
python3 src/hansard/utils/path_config.py

# Verify data
python3 verify_all_systems.py
```

## Documentation

- `CLAUDE.md` - Development guidelines and visualization standards
- `DIRECTORY_STRUCTURE.md` - Detailed project layout
- `RUN_ANALYSIS.md` - Analysis execution guide

## License

Data sourced from UK Parliament Historic Hansard (public domain).
