# Directory Structure

This repository follows a clear organizational structure for the Hansard NLP analysis project.

## Root Directory
```
hansard-nlp-explorer/
├── README.md                    # Main project documentation
├── CLAUDE.md                     # Instructions for Claude Code
├── DIRECTORY_STRUCTURE.md       # This file
├── environment.yml               # Conda environment specification
├── pytest.ini                    # Pytest configuration
├── .gitignore                   # Git ignore rules
└── src/                         # All source code
```

## Source Code Structure (`src/hansard/`)

### Core Modules
```
src/hansard/
├── speaker_processing.py        # Main speaker processing module
├── quick_test.py                # Quick testing utilities
└── explore_hansard_data.ipynb  # Jupyter notebook for exploration
```

### Scripts (`src/hansard/scripts/`)
**All executable shell scripts go here:**
```
src/hansard/scripts/
├── run_gender_corpus_analysis.sh     # Run corpus analysis
├── run_gender_milestone_analysis.sh  # Run milestone analysis
├── run_full_gender_analysis.sh       # Full analysis with large samples
├── run_quick_gender_test.sh          # Quick test with small samples
├── run_enhanced_gender_dataset.sh    # Create enhanced dataset
└── [other run scripts]
```

### Analysis (`src/hansard/analysis/`)
**All Python analysis scripts and their outputs:**
```
src/hansard/analysis/
├── gender_corpus_analysis.py         # Main corpus analysis
├── gender_milestone_analysis.py      # Historical milestone analysis
├── professional_visualizations.py    # Publication-quality viz module
├── enhanced_gender_corpus_analysis.py # Enhanced analysis (WIP)
│
├── gender_corpus_results/            # Output: corpus analysis
├── gender_milestone_results/         # Output: milestone analysis
├── professional_visualizations/      # Output: professional charts
└── enhanced_gender_results/          # Output: enhanced analysis
```

### Data (`src/hansard/data/`)
**All data files and datasets:**
```
src/hansard/data/
├── gender_analysis_enhanced/         # Enhanced dataset with full text
│   ├── debates_YYYY_enhanced.parquet # Year-by-year debate files
│   └── ALL_debates_enhanced_*.parquet # Combined datasets
│
├── gender_wordlists/                 # Gender-associated word lists
│   ├── male_words.txt
│   └── female_words.txt
│
├── processed_fixed/                  # Original processed data
└── hansard/                          # Raw Hansard data
```

### Documentation (`src/hansard/docs/`)
**All project documentation:**
```
src/hansard/docs/
├── VISUALIZATION_STYLE_GUIDE.md     # How to create visualizations
├── VISUALIZATION_PLAN.md            # What visualizations to create
├── DATASET_STRUCTURE.md             # Data format documentation
└── [other documentation]
```

### Other Directories
```
src/hansard/
├── tests/                # Unit tests
├── utils/                # Utility functions
├── parsers/              # Data parsing modules
├── crawlers/             # Web crawling scripts
└── debug_scripts/        # Debugging utilities
```

## Running Scripts

All scripts should be run from the **repository root**:
```bash
# From hansard-nlp-explorer/ directory:
./src/hansard/scripts/run_quick_gender_test.sh
./src/hansard/scripts/run_full_gender_analysis.sh
```

Or with explicit Python:
```bash
python src/hansard/analysis/gender_corpus_analysis.py --years 1990-2000
```

## Output Organization

Analysis outputs are organized by type:
- **Visualizations**: Saved in `analysis/*/` subdirectories
- **JSON results**: Saved alongside visualizations
- **Logs**: Created in script execution directory

## Best Practices

1. **Scripts**: All executable scripts in `src/hansard/scripts/`
2. **Analysis**: Python analysis modules in `src/hansard/analysis/`
3. **Data**: All data in `src/hansard/data/`
4. **Docs**: Documentation in `src/hansard/docs/`
5. **Outputs**: Analysis outputs in `src/hansard/analysis/[analysis_type]_results/`

## Why This Structure?

- **Clarity**: Everything is in predictable locations
- **Modularity**: Each component has its own directory
- **Scalability**: Easy to add new analyses or data sources
- **Maintainability**: Clear separation of concerns
- **Reproducibility**: Scripts can be run from consistent locations