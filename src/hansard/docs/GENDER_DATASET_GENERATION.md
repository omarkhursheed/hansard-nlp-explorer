# Gender Dataset Generation Guide

## Overview

This guide explains how to generate the full gender and entity-matched dataset from the Hansard parliamentary debates corpus.

## What It Does

The gender dataset generation process:

1. **Processes all 201 years** of Hansard data (1803-2005)
2. **Matches speakers to MPs** using the corrected MP matcher with verified historical data
3. **Identifies gender** for all matched MPs (both male and female)
4. **Creates structured output** with debate-level and MP-level information
5. **Generates statistics** about gender participation over time

## Key Features

### Resume Capability
- **Checkpoint system**: Saves progress every 10 years
- **Interrupt safely**: Use Ctrl+C anytime - progress is saved
- **Resume automatically**: Just run again to continue from last checkpoint

### Robust Processing
- **Error recovery**: Continues processing even if individual years fail
- **Progress tracking**: Shows detailed progress with time estimates
- **Memory efficient**: Processes year by year to avoid memory issues

### Comprehensive Output
- **Individual year files**: `debates_YYYY_with_mps.parquet`
- **Combined dataset**: `ALL_debates_with_confirmed_mps.parquet`
- **Metadata and stats**: `dataset_metadata.json`

## How to Run

### Quick Start

```bash
cd src/hansard
./scripts/run_gender_analysis.sh
```

### Command Options

```bash
# Process all years (with resume)
./scripts/run_gender_analysis.sh

# Process specific year range
./scripts/run_gender_analysis.sh 1900 2000

# Start fresh (delete checkpoint)
./scripts/run_gender_analysis.sh --reset
```

### Direct Python Usage

```bash
cd src/hansard

# Basic run
python scripts/data_creation/create_full_gender_dataset_resumable.py

# With options
python scripts/data_creation/create_full_gender_dataset_resumable.py \
    --output-dir my_output \
    --year-range 1900 2000 \
    --reset
```

## Output Structure

### Directory Layout
```
gender_analysis_data_FULL/
├── ALL_debates_with_confirmed_mps.parquet  # Combined dataset
├── dataset_metadata.json                   # Statistics and metadata
├── debates_1803_with_mps.parquet          # Individual year files
├── debates_1804_with_mps.parquet
├── ...
└── debates_2005_with_mps.parquet
```

### Data Schema

Each debate record contains:
- `debate_id`: Unique identifier
- `year`, `decade`: Temporal information
- `reference_date`: Date of debate
- `chamber`: Commons or Lords
- `title`, `topic`: Debate information
- `total_speakers`: Total speaker count
- `confirmed_mps`: Number of matched MPs
- `female_mps`, `male_mps`: Count by gender
- `has_female`, `has_male`: Boolean flags
- `female_names`, `male_names`: Lists of MP names
- `ambiguous_speakers`: Could not determine unique MP
- `unmatched_speakers`: No match found
- `word_count`: Total words in debate

### Metadata File

The `dataset_metadata.json` contains:
- Creation date and processing stats
- List of all unique female and male MPs identified
- Statistics by decade
- Processing summary

## Processing Time

- **Full dataset**: ~5-10 minutes for all 201 years
- **Memory usage**: ~2-3 GB peak
- **Disk space**: ~50 MB output

## Troubleshooting

### If the script fails

1. **Just run it again** - it will resume from checkpoint
2. **Check the error message** - usually a specific year had issues
3. **Use --reset** if you want to start completely fresh

### Common Issues

- **Import errors**: Make sure you're in the `src/hansard` directory
- **Memory issues**: The resumable version processes year-by-year to avoid this
- **Timeout errors**: The script handles these gracefully - just re-run

### Verifying Output

Check the completion:
```bash
# Count year files
ls gender_analysis_data_FULL/debates_*.parquet | wc -l
# Should be 201 (or close)

# Check metadata
cat gender_analysis_data_FULL/dataset_metadata.json | python -m json.tool | head -20
```

## What Was Refactored

The recent refactoring (commit 0125ba3):

✅ **Improvements**:
- Cleaned up ~4500 lines of obsolete code
- Organized scripts into logical directories
- Fixed symmetric storage of male/female MPs
- Consolidated all tests and documentation
- Preserved only the working, verified matcher

❌ **Removed**:
- Old flawed MP matchers (temporal, advanced)
- Intermediate processing scripts
- Redundant test files
- Temporary data directories

## Next Steps

After generation, you can:

1. **Analyze gender trends** using the analysis scripts
2. **Create visualizations** of participation over time
3. **Export subsets** for specific research questions
4. **Join with other data** for enhanced analysis

## Contact

For issues or questions about the gender dataset generation, check:
- The logs in the output directory
- The checkpoint file (if processing was interrupted)
- The error messages in the console output