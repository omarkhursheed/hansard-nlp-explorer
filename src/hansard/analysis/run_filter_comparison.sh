#!/bin/bash

# Script to run Hansard analysis with different filter levels for comparison

echo "Running Hansard Analysis Filter Comparison"
echo "==========================================="

# Set parameters
YEARS="1920-1930"
SAMPLE=500

# Create output directory
mkdir -p analysis/results_advanced/comparison

# Run each filter level
for level in 0 1 2 3 4 5; do
    echo ""
    echo "Running Level $level..."
    python analysis/hansard_nlp_analysis_advanced.py \
        --years $YEARS \
        --sample $SAMPLE \
        --filter-level $level \
        2>&1 | tee analysis/results_advanced/comparison/level_${level}_output.txt
done

echo ""
echo "Creating comparison report..."

# Create comparison report
cat > analysis/results_advanced/comparison/COMPARISON_REPORT.md << EOF
# Hansard Analysis Filter Level Comparison

**Date:** $(date)
**Years Analyzed:** $YEARS
**Sample Size:** $SAMPLE

## Filter Levels

### Level 0: NONE
- No filtering applied
- Baseline for comparison
- All words included

### Level 1: BASIC
- Common English stop words removed
- Basic cleaning only

### Level 2: PARLIAMENTARY
- Parliamentary procedural terms removed
- Titles and honorifics filtered

### Level 3: MODERATE
- Common verbs removed
- Vague words filtered
- Prepositions removed

### Level 4: AGGRESSIVE
- Modal verbs removed
- Quantifiers filtered
- Maximum standard filtering

### Level 5: TFIDF
- TF-IDF scoring applied
- Only distinctive words kept
- Statistical filtering

## Results Summary

EOF

# Extract top words from each level
for level in 0 1 2 3 4 5; do
    echo "### Level $level Top Words" >> analysis/results_advanced/comparison/COMPARISON_REPORT.md
    grep -A 10 "Top 10 words" analysis/results_advanced/comparison/level_${level}_output.txt >> analysis/results_advanced/comparison/COMPARISON_REPORT.md
    echo "" >> analysis/results_advanced/comparison/COMPARISON_REPORT.md
done

echo "Comparison complete!"
echo "Results saved in analysis/results_advanced/comparison/"