#!/bin/bash

# Comprehensive Hansard Corpus Analysis Runner
# This script runs the comprehensive corpus analysis with all filtering levels

echo "============================================================================"
echo "COMPREHENSIVE HANSARD CORPUS ANALYSIS"
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

# Build command
CMD="python \"$SCRIPT_DIR/analysis/comprehensive_corpus_analysis.py\""

if [[ -n "$FULL_FLAG" ]]; then
    CMD="$CMD $FULL_FLAG"
elif [[ -n "$YEARS" ]]; then
    CMD="$CMD --years $YEARS"
fi

if [[ -n "$SAMPLE" ]]; then
    CMD="$CMD --sample $SAMPLE"
fi

echo "Running: $CMD"
echo ""

# Execute the analysis
eval $CMD

echo ""
echo "============================================================================"
echo "ANALYSIS COMPLETE!"
echo "Check the 'analysis/corpus_results' directory for outputs:"
echo "  - Comprehensive visualizations for each filtering level"
echo "  - Comparison charts across filtering levels"
echo "  - JSON results with detailed analysis data"
echo "============================================================================"
