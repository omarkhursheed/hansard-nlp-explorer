#!/bin/bash
# Complete Analysis Runner
# Runs all gender analysis scripts with proper parameters

set -e  # Exit on error

echo "================================================================================"
echo "HANSARD GENDER ANALYSIS - COMPLETE RUN"
echo "================================================================================"
echo ""
echo "This script runs:"
echo "1. Enhanced Gender Corpus Analysis (overall trends)"
echo "2. Gender Milestone Analysis (historical events)"
echo ""
echo "All outputs saved to: analysis/"
echo "================================================================================"
echo ""

# Check if data exists
if [ ! -d "data-hansard/gender_analysis_enhanced" ]; then
    echo "ERROR: Data directory not found at data-hansard/gender_analysis_enhanced"
    echo "Please ensure data is in the correct location"
    exit 1
fi

# 1. Overall Gender Corpus Analysis
echo "================================================================================"
echo "STEP 1: Enhanced Gender Corpus Analysis"
echo "================================================================================"
python3 src/hansard/analysis/enhanced_gender_corpus_analysis.py \
    --full \
    --sample 50000 \
    --filtering aggressive

echo ""
echo "✓ Corpus analysis complete"
echo ""

# 2. Milestone Analysis
echo "================================================================================"
echo "STEP 2: Gender Milestone Analysis"
echo "================================================================================"
python3 src/hansard/analysis/gender_milestone_analysis.py \
    --all \
    --filtering aggressive

echo ""
echo "✓ Milestone analysis complete"
echo ""

# Summary
echo "================================================================================"
echo "ANALYSIS COMPLETE!"
echo "================================================================================"
echo ""
echo "Outputs in analysis/:"
echo "  - temporal_representation.png    (trends 1803-2005)"
echo "  - vocabulary_comparison.png      (distinctive words)"
echo "  - bigram_comparison.png          (distinctive phrases)"
echo "  - topic_distribution.png         (topic modeling)"
echo "  - statistical_summary.png        (key statistics)"
echo "  - *_milestone_*.png              (milestone events)"
echo "  - *.json                         (numerical results)"
echo ""
echo "================================================================================"
