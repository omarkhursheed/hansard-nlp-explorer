#!/bin/bash

# Gender-Matched Hansard Corpus Analysis Runner
# This script runs corpus analysis on the gender-matched dataset only

echo "============================================================================"
echo "GENDER-MATCHED HANSARD CORPUS ANALYSIS"
echo "============================================================================"

# Activate conda environment if it exists
if command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate hansard 2>/dev/null || echo "No hansard environment found, using base"
fi

# Resolve script dir for consistent paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set default parameters
YEARS="1803-2005"
SAMPLE=100000
DATA_SOURCE="gender_matched"  # Use gender-matched data

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --years)
            YEARS="$2"
            shift 2
            ;;
        --sample)
            SAMPLE="$2"
            shift 2
            ;;
        --full)
            FULL_FLAG="--full"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --years RANGE    Year range (e.g., 1920-1935)"
            echo "  --sample N       Sample size for analysis"
            echo "  --full          Analyze full corpus"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "This script analyzes ONLY the gender-matched parliamentary data,"
            echo "excluding debates without confirmed MP speakers."
            echo ""
            echo "Examples:"
            echo "  $0 --years 1920-1930 --sample 500"
            echo "  $0 --full --sample 10000"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Build command - use the gender-matched specific analysis script
CMD="python \"$SCRIPT_DIR/analysis/gender_corpus_analysis.py\""

if [[ -n "$FULL_FLAG" ]]; then
    CMD="$CMD $FULL_FLAG"
elif [[ -n "$YEARS" ]]; then
    CMD="$CMD --years $YEARS"
fi

if [[ -n "$SAMPLE" ]]; then
    CMD="$CMD --sample $SAMPLE"
fi

echo "Data source: Gender-matched MPs only (excluding non-MP speeches)"
echo "Running: $CMD"
echo ""

# Execute the analysis
eval $CMD

echo ""
echo "============================================================================"
echo "GENDER-MATCHED ANALYSIS COMPLETE!"
echo "Check the 'analysis/gender_corpus_results' directory for outputs:"
echo "  - Visualizations for gender-matched data only"
echo "  - Gender-specific language patterns"
echo "  - Temporal trends in gendered discourse"
echo "============================================================================"