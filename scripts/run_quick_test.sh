#!/bin/bash
#
# QUICK SMOKE TEST - Verify unified analysis system works
#
# Tests all core functionality with minimal data (~2-3 minutes)
#
# USAGE:
#   ./run_quick_test.sh
#

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================================================================"
echo -e "${BLUE}QUICK SMOKE TEST - Unified Analysis System${NC}"
echo "================================================================================"
echo ""
echo "This will test all components with minimal data (~2-3 minutes)"
echo ""

# Test 1: Module imports
echo -e "${BLUE}Test 1: Module Imports${NC}"
python3 test_unified_modules.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Module imports: PASSED${NC}"
else
    echo -e "${RED}✗ Module imports: FAILED${NC}"
    exit 1
fi
echo ""

# Test 2: Overall corpus analysis
echo -e "${BLUE}Test 2: Overall Corpus Analysis${NC}"
python3 src/hansard/analysis/corpus_analysis.py \
    --dataset overall \
    --years 1995-1996 \
    --sample 50 \
    --filtering moderate \
    --analysis unigram,bigram

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Overall corpus analysis: PASSED${NC}"
else
    echo -e "${RED}✗ Overall corpus analysis: FAILED${NC}"
    exit 1
fi
echo ""

# Test 3: Gender corpus analysis (if data exists)
if [ -d "data-hansard/gender_analysis_enhanced" ]; then
    echo -e "${BLUE}Test 3: Gender Corpus Analysis${NC}"
    python3 src/hansard/analysis/corpus_analysis.py \
        --dataset gender \
        --years 1995-1996 \
        --sample 50 \
        --filtering aggressive \
        --analysis unigram,bigram

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Gender corpus analysis: PASSED${NC}"
    else
        echo -e "${RED}✗ Gender corpus analysis: FAILED${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ Gender dataset not found - skipping gender tests${NC}"
fi
echo ""

# Test 4: Milestone analysis
echo -e "${BLUE}Test 4: Milestone Analysis${NC}"
python3 src/hansard/analysis/milestone_analysis.py \
    --milestone ww2_period \
    --dataset overall \
    --filtering moderate \
    --sample 100

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Milestone analysis: PASSED${NC}"
else
    echo -e "${RED}✗ Milestone analysis: FAILED${NC}"
    exit 1
fi
echo ""

# Summary
echo "================================================================================"
echo -e "${GREEN}ALL TESTS PASSED!${NC}"
echo "================================================================================"
echo ""
echo "The unified analysis system is working correctly."
echo ""
echo "Next steps:"
echo "  - Run full analysis: ./run_full_analysis.sh"
echo "  - Run standard mode: ./run_full_analysis.sh --standard"
echo "  - Run full mode: ./run_full_analysis.sh --full"
echo ""
