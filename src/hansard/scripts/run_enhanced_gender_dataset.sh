#!/bin/bash
#
# Runner script for creating ENHANCED gender analysis dataset
# This dataset includes full debate text, speech segments, and comprehensive metadata
#

set -e  # Exit on error

echo "========================================================================="
echo "ENHANCED GENDER ANALYSIS DATASET GENERATION"
echo "========================================================================="
echo ""
echo "This script will create a comprehensive gender analysis dataset with:"
echo "  - Full debate text (no character limits)"
echo "  - Extracted speech segments with speaker attribution"
echo "  - Enhanced MP metadata (party, constituency, confidence scores)"
echo "  - Complete unmatched speaker lists for analysis"
echo ""
echo "Output will be saved to: data/gender_analysis_enhanced/"
echo "========================================================================="

# Check if we're in the right directory
if [[ ! -f "scripts/data_creation/create_enhanced_gender_dataset.py" ]]; then
    echo "Error: Must run from src/hansard directory"
    exit 1
fi

# Parse command line arguments
RESET_FLAG=""
SAMPLE_FLAG=""
YEAR_RANGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --reset)
            RESET_FLAG="--reset"
            echo "Will reset checkpoint and start fresh"
            shift
            ;;
        --sample)
            SAMPLE_FLAG="--sample"
            echo "Will run in sample mode (first 3 years only)"
            shift
            ;;
        --year-range)
            YEAR_RANGE="--year-range $2 $3"
            echo "Will process years $2 to $3"
            shift 3
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --reset        Reset checkpoint and start fresh"
            echo "  --sample       Process only first 3 years for testing"
            echo "  --year-range START END  Process only years in range"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check Python environment
echo ""
echo "Checking Python environment..."
python3 -c "import pandas; import tqdm; import numpy" 2>/dev/null || {
    echo "Error: Required Python packages not found."
    echo "Please install: pandas, tqdm, numpy, beautifulsoup4"
    echo "Example: pip install pandas tqdm numpy beautifulsoup4"
    exit 1
}

# Check for required data files
echo "Checking for required data files..."
if [[ ! -d "data/processed_fixed/content" ]]; then
    echo "Error: Extracted content not found at data/processed_fixed/content"
    echo "Please ensure the processed_fixed data is available"
    exit 1
fi

if [[ ! -f "data/house_members_gendered_updated.parquet" ]]; then
    echo "Error: MP gender data not found at data/house_members_gendered_updated.parquet"
    exit 1
fi

# Create output directory if needed
mkdir -p data/gender_analysis_enhanced

# Run the dataset creation
echo ""
echo "Starting dataset creation..."
echo "========================================================================="

python3 scripts/data_creation/create_enhanced_gender_dataset.py \
    $RESET_FLAG \
    $SAMPLE_FLAG \
    $YEAR_RANGE

# Check if successful
if [[ $? -eq 0 ]]; then
    echo ""
    echo "========================================================================="
    echo "✅ Dataset creation completed successfully!"
    echo ""
    echo "Output files in data/gender_analysis_enhanced/:"
    echo "  - ALL_debates_enhanced_with_text.parquet (full dataset with text)"
    echo "  - ALL_debates_enhanced_metadata.parquet (lightweight, no text)"
    echo "  - debates_YYYY_enhanced.parquet (individual year files)"
    echo "  - dataset_metadata.json (statistics and metadata)"
    echo ""

    # Show file sizes
    if [[ -d "data/gender_analysis_enhanced" ]]; then
        echo "File sizes:"
        ls -lh data/gender_analysis_enhanced/*.parquet 2>/dev/null | head -5

        # Show basic stats if metadata exists
        if [[ -f "data/gender_analysis_enhanced/dataset_metadata.json" ]]; then
            echo ""
            echo "Dataset statistics:"
            python3 -c "
import json
with open('data/gender_analysis_enhanced/dataset_metadata.json', 'r') as f:
    meta = json.load(f)
    stats = meta.get('statistics', {})
    print(f'  Years processed: {meta.get(\"years_processed\", \"N/A\")}')
    print(f'  Total debates: {stats.get(\"total_debates_processed\", \"N/A\"):,}')
    print(f'  Debates with MPs: {stats.get(\"debates_with_confirmed_mps\", \"N/A\"):,}')
    print(f'  Debates with female MPs: {stats.get(\"debates_with_female\", \"N/A\"):,}')
    print(f'  Female MPs identified: {meta.get(\"total_female_mps_identified\", \"N/A\")}')
    print(f'  Male MPs identified: {meta.get(\"total_male_mps_identified\", \"N/A\")}')
" 2>/dev/null || echo "Could not read statistics"
        fi
    fi
else
    echo ""
    echo "❌ Dataset creation failed. Check the error messages above."
    exit 1
fi