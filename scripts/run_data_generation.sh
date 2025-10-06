#!/bin/bash
#
# HANSARD DATA GENERATION
#
# Generates the gender-enhanced dataset from raw Hansard data.
# This should be run ONCE before running analysis.
#
# USAGE:
#   ./run_data_generation.sh
#
# PREREQUISITES:
#   - Raw Hansard data in data-hansard/processed_fixed/
#   - Gender wordlists in data-hansard/gender_wordlists/
#   - Python environment with required packages
#
# OUTPUT:
#   data-hansard/gender_analysis_enhanced/ (parquet files with gender info)
#
# DURATION:
#   ~2-3 hours for full corpus (1803-2005)
#

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================================"
echo -e "${BLUE}HANSARD DATA GENERATION${NC}"
echo "================================================================================"
echo ""
echo "This will generate the gender-enhanced dataset from raw Hansard data."
echo ""
echo "Requirements:"
echo "  - Raw Hansard data in data-hansard/processed_fixed/"
echo "  - Gender wordlists in data-hansard/gender_wordlists/"
echo "  - Python packages: pandas, pyarrow, numpy"
echo ""
echo "Output:"
echo "  - data-hansard/gender_analysis_enhanced/ (parquet files)"
echo ""
echo "Duration: ~2-3 hours for full corpus"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -d "data-hansard/processed_fixed" ]; then
    echo -e "${RED}ERROR: Raw Hansard data not found${NC}"
    echo "Expected: data-hansard/processed_fixed/"
    exit 1
fi

if [ ! -d "data-hansard/gender_wordlists" ]; then
    echo -e "${YELLOW}WARNING: Gender wordlists not found${NC}"
    echo "Expected: data-hansard/gender_wordlists/"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}✓ Prerequisites check passed${NC}"
echo ""

# Confirm with user
read -p "Generate gender-enhanced dataset? This takes ~2-3 hours (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "================================================================================"
echo -e "${BLUE}STARTING DATA GENERATION${NC}"
echo "================================================================================"
echo ""

# Find the data generation script
if [ -f "src/hansard/scripts/data_creation/create_full_gender_dataset_resumable.py" ]; then
    DATA_SCRIPT="src/hansard/scripts/data_creation/create_full_gender_dataset_resumable.py"
elif [ -f "src/hansard/scripts/run_enhanced_gender_dataset.sh" ]; then
    # Use existing shell script if available
    bash src/hansard/scripts/run_enhanced_gender_dataset.sh
    exit 0
else
    echo -e "${RED}ERROR: Data generation script not found${NC}"
    echo "Looking for:"
    echo "  - src/hansard/scripts/data_creation/create_full_gender_dataset_resumable.py"
    echo "  - src/hansard/scripts/run_enhanced_gender_dataset.sh"
    exit 1
fi

# Run data generation
echo "Running: python3 $DATA_SCRIPT"
echo ""
python3 $DATA_SCRIPT

# Check output
if [ -d "data-hansard/gender_analysis_enhanced" ]; then
    FILE_COUNT=$(ls data-hansard/gender_analysis_enhanced/*.parquet 2>/dev/null | wc -l)
    echo ""
    echo "================================================================================"
    echo -e "${GREEN}DATA GENERATION COMPLETE!${NC}"
    echo "================================================================================"
    echo ""
    echo -e "${GREEN}✓${NC} Generated $FILE_COUNT parquet files"
    echo -e "${GREEN}✓${NC} Dataset: data-hansard/gender_analysis_enhanced/"
    echo ""
    echo "Next step:"
    echo "  Run analysis: ./run_full_analysis.sh"
    echo ""
else
    echo -e "${RED}ERROR: Output directory not created${NC}"
    echo "Data generation may have failed"
    exit 1
fi
