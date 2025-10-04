#!/bin/bash

# Quick Gender Analysis Test - For Development and Testing
# Runs with small samples for quick iteration

echo "============================================================================"
echo "QUICK GENDER ANALYSIS TEST - SMALL SAMPLE"
echo "============================================================================"

# Configuration for quick test
SAMPLE_SIZE=500
YEARS="1990-1995"
FILTERING="aggressive"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Activate conda environment
if command -v conda &> /dev/null; then
    echo -e "${YELLOW}Activating conda environment...${NC}"
    eval "$(conda shell.bash hook)"
    conda activate hansard 2>/dev/null || echo "Using base environment"
fi

echo ""
echo "Test parameters:"
echo "  - Sample size: $SAMPLE_SIZE"
echo "  - Years: $YEARS"
echo "  - Filtering: $FILTERING"
echo ""

# Test 1: Basic corpus analysis
echo -e "${YELLOW}Test 1: Corpus Analysis${NC}"
python3 src/hansard/analysis/gender_corpus_analysis.py \
    --years $YEARS \
    --sample $SAMPLE_SIZE

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Corpus analysis test passed${NC}"
else
    echo "✗ Corpus analysis test failed"
    exit 1
fi

# Test 2: Single milestone
echo ""
echo -e "${YELLOW}Test 2: Milestone Analysis (WW2)${NC}"
python3 src/hansard/analysis/gender_milestone_analysis.py \
    --milestone ww2_period \
    --filtering $FILTERING

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Milestone analysis test passed${NC}"
else
    echo "✗ Milestone analysis test failed"
    exit 1
fi

# Test 3: Professional visualizations
echo ""
echo -e "${YELLOW}Test 3: Professional Visualizations${NC}"
python3 src/hansard/analysis/professional_visualizations.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Visualization test passed${NC}"
else
    echo "✗ Visualization test failed"
    exit 1
fi

echo ""
echo "============================================================================"
echo -e "${GREEN}ALL TESTS PASSED!${NC}"
echo "============================================================================"
echo ""
echo "Generated outputs:"
echo "  - analysis/gender_corpus_results/"
echo "  - analysis/gender_milestone_results/"
echo "  - analysis/professional_visualizations/"
echo ""
echo "Ready for full analysis. Run: ./run_full_gender_analysis.sh"
echo "============================================================================"