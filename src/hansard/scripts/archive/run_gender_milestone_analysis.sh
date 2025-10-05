#!/bin/bash

# Gender-Matched Hansard Milestone Analysis Runner
# This script runs milestone analysis on the gender-matched dataset only

echo "============================================================================"
echo "GENDER-MATCHED HANSARD MILESTONE ANALYSIS"
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
FILTERING="aggressive"
MILESTONE=""
ALL_FLAG=""
FORCE_FLAG=""
DATA_SOURCE="gender_matched"  # Use gender-matched data

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --milestone)
            MILESTONE="$2"
            shift 2
            ;;
        --filtering)
            FILTERING="$2"
            shift 2
            ;;
        --all)
            ALL_FLAG="--all"
            shift
            ;;
        --force)
            FORCE_FLAG="--force"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --milestone NAME   Specific milestone to analyze:"
            echo "                     - 1918_partial_suffrage"
            echo "                     - 1928_full_suffrage"
            echo "                     - ww1_period"
            echo "                     - ww2_period"
            echo "                     - thatcher_period"
            echo "  --filtering MODE   Filtering level: none, basic, parliamentary, moderate, aggressive"
            echo "  --all             Analyze all milestones"
            echo "  --force           Force rerun even if results exist"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "This script analyzes ONLY the gender-matched parliamentary data,"
            echo "focusing on how confirmed MPs discussed key historical events."
            echo ""
            echo "Examples:"
            echo "  $0 --all --filtering aggressive"
            echo "  $0 --milestone ww2_period --filtering moderate"
            echo "  $0 --milestone 1928_full_suffrage --force"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Build command - use the gender-matched specific analysis script
CMD="python \"$SCRIPT_DIR/analysis/gender_milestone_analysis.py\""

if [[ -n "$ALL_FLAG" ]]; then
    CMD="$CMD $ALL_FLAG"
elif [[ -n "$MILESTONE" ]]; then
    CMD="$CMD --milestone $MILESTONE"
else
    # Default to all if no specific milestone
    CMD="$CMD --all"
fi

CMD="$CMD --filtering $FILTERING"

if [[ -n "$FORCE_FLAG" ]]; then
    CMD="$CMD $FORCE_FLAG"
fi

echo "Data source: Gender-matched MPs only (excluding non-MP speeches)"
echo "Filtering mode: $FILTERING"
if [[ -n "$MILESTONE" ]]; then
    echo "Milestone: $MILESTONE"
else
    echo "Analyzing: ALL milestones"
fi
echo ""
echo "Running: $CMD"
echo ""

# Execute the analysis
eval $CMD

echo ""
echo "============================================================================"
echo "GENDER-MATCHED MILESTONE ANALYSIS COMPLETE!"
echo "Check the 'analysis/gender_milestone_results' directory for outputs:"
echo "  - Milestone analysis with confirmed MP speeches only"
echo "  - Gender-specific language changes around key events"
echo "  - Comparison of male vs female MP discourse patterns"
echo "  - Temporal trends in gendered parliamentary language"
echo "============================================================================"