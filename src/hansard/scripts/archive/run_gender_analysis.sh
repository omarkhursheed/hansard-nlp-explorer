#!/bin/bash

# Run full gender analysis dataset creation with proper handling
# This script can be safely interrupted and resumed

set -e  # Exit on error

echo "=================================================================="
echo "GENDER ANALYSIS DATASET CREATION RUNNER"
echo "=================================================================="
echo ""
echo "This script will create the full gender/entity matched dataset."
echo "It processes all 201 years of Hansard data (1803-2005)."
echo ""
echo "Features:"
echo "  ✓ Resume capability - can safely interrupt with Ctrl+C"
echo "  ✓ Progress tracking - shows detailed progress"
echo "  ✓ Checkpoint saving - saves every 10 years"
echo "  ✓ Error recovery - continues on individual year failures"
echo ""
echo "Usage:"
echo "  ./run_gender_analysis.sh               # Process all years"
echo "  ./run_gender_analysis.sh 1900 2000     # Process year range"
echo "  ./run_gender_analysis.sh --reset       # Start fresh (delete checkpoint)"
echo ""
echo "=================================================================="

# Get the script directory and navigate to src/hansard
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "Working directory: $(pwd)"

# Check if we should reset
if [[ "$1" == "--reset" ]]; then
    echo ""
    echo "🗑️  Resetting - removing any existing checkpoint..."
    rm -f gender_analysis_data_FULL/checkpoint.pkl
    python scripts/data_creation/create_full_gender_dataset_resumable.py --reset
elif [[ -n "$1" ]] && [[ -n "$2" ]]; then
    # Year range specified
    echo ""
    echo "📅 Processing years $1 to $2..."
    python scripts/data_creation/create_full_gender_dataset_resumable.py --year-range $1 $2
else
    # Process all years
    echo ""
    if [[ -f gender_analysis_data_FULL/checkpoint.pkl ]]; then
        echo "📂 Found checkpoint - will resume from last position"
    else
        echo "🆕 No checkpoint found - starting fresh"
    fi
    echo ""
    read -p "Press Enter to start processing (Ctrl+C to cancel)..."

    python scripts/data_creation/create_full_gender_dataset_resumable.py
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================================="
    echo "✅ SUCCESS - Dataset creation complete!"
    echo "=================================================================="
    echo ""
    echo "Output location: gender_analysis_data_FULL/"
    echo ""
    echo "Files created:"
    echo "  - ALL_debates_with_confirmed_mps.parquet (combined dataset)"
    echo "  - debates_YYYY_with_mps.parquet (individual year files)"
    echo "  - dataset_metadata.json (statistics and metadata)"
    echo ""

    # Show quick stats if metadata exists
    if [[ -f gender_analysis_data_FULL/dataset_metadata.json ]]; then
        echo "Quick statistics:"
        python -c "
import json
with open('gender_analysis_data_FULL/dataset_metadata.json') as f:
    meta = json.load(f)
    stats = meta['statistics']
    print(f'  • Total debates: {stats[\"total_debates_processed\"]:,}')
    print(f'  • Debates with MPs: {stats[\"debates_with_confirmed_mps\"]:,}')
    print(f'  • Debates with female MPs: {stats[\"debates_with_female\"]:,}')
    print(f'  • Unique female MPs: {meta[\"total_female_mps_identified\"]}')
    print(f'  • Unique male MPs: {meta[\"total_male_mps_identified\"]}')
"
    fi
else
    echo ""
    echo "=================================================================="
    echo "⚠️  INTERRUPTED or ERROR"
    echo "=================================================================="
    echo ""
    echo "The process was interrupted or encountered an error."
    echo "Don't worry - your progress has been saved!"
    echo ""
    echo "To resume: Run this script again"
    echo "To start fresh: Run with --reset flag"
    echo ""
fi
