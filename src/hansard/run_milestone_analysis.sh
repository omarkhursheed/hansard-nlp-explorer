#!/bin/bash

# Comprehensive Hansard Milestone Analysis Runner
# This script runs milestone analysis for historical periods

echo "============================================================================"
echo "COMPREHENSIVE HANSARD MILESTONE ANALYSIS"
echo "============================================================================"

# Activate conda environment if it exists
if command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate hansard 2>/dev/null || echo "No hansard environment found, using base"
fi

# Set default parameters
FILTERING="aggressive"
MILESTONE=""
ALL_FLAG=""
FORCE_FLAG=""

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

# Build command
CMD="python analysis/comprehensive_milestone_analysis.py"

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
echo "MILESTONE ANALYSIS COMPLETE!"
echo "Check the 'analysis/milestone_results' directory for outputs:"
echo "  - Individual milestone analysis with comprehensive visualizations"
echo "  - Comparison charts showing before/during/after periods"
echo "  - Markdown reports with key findings"
echo "  - Master summary across all milestones"
echo "============================================================================"
