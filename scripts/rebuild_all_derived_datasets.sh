#!/bin/bash
#
# Rebuild All Derived Datasets
#
# This script:
# 1. Cleans up old/redundant folders
# 2. Regenerates matched datasets (with position field)
# 3. Generates full datasets (all debates from processed_fixed)
# 4. Renames to clean structure
#
# Final structure:
#   derived/speeches_matched/  - Only gender-matched speeches (~352/year)
#   derived/debates_matched/   - Debates with ≥1 gender match (~233/year)
#   derived/speeches_full/     - ALL speeches (~5,314/year for 1900)
#   derived/debates_full/      - ALL debates (~5,469/year for 1900)

set -e  # Exit on error

cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

echo "================================================================================"
echo "REBUILD ALL DERIVED DATASETS"
echo "================================================================================"
echo ""

# Step 1: Clean up old folders
echo "=== Step 1: Cleaning up old/redundant folders ==="
cd data-hansard/derived

if [ -d "speeches" ]; then
    echo "  Removing old speeches/ (partial data)"
    rm -rf speeches
fi

if [ -d "debates" ]; then
    echo "  Removing old debates/ (partial data)"
    rm -rf debates
fi

if [ -d "gender_debates" ]; then
    echo "  Removing gender_debates/ (duplicate of debates_matched)"
    rm -rf gender_debates
fi

echo ""
cd ../..

# Step 2: Regenerate matched datasets (with position field)
echo "=== Step 2: Regenerating matched datasets with position field ==="
echo "This includes debates with at least 1 gender-matched speaker"
echo ""

echo "  2a. Generating speeches_matched/ (gender-matched speeches only)..."
python3 scripts/create_gender_speeches_dataset.py --force
echo ""

echo "  2b. Generating debates_matched/ (debates with gender matches)..."
python3 scripts/create_gender_debates_dataset.py --force
echo ""

# Step 3: Rename gender_* to *_matched
echo "=== Step 3: Renaming to clean structure ==="
cd data-hansard/derived

if [ -d "gender_speeches" ]; then
    echo "  Renaming gender_speeches/ -> speeches_matched/"
    mv gender_speeches speeches_matched
fi

if [ -d "gender_debates" ]; then
    echo "  Renaming gender_debates/ -> debates_matched/"
    mv gender_debates debates_matched
fi

echo ""
cd ../..

# Step 4: Generate full datasets
echo "=== Step 4: Generating full datasets (ALL debates from processed_fixed) ==="
echo "This includes ALL debates, not just those with gender matches"
echo ""

python3 scripts/create_full_speeches_dataset.py --force

echo ""

# Step 5: Summary
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo ""
echo "Final structure:"
echo "  derived/speeches_matched/  - Speeches from gender-matched speakers"
echo "  derived/debates_matched/   - Debates with ≥1 gender match"
echo "  derived/speeches_full/     - ALL speeches from ALL debates"
echo "  derived/debates_full/      - ALL debates"
echo ""

# Count files
cd data-hansard/derived
echo "Files generated:"
if [ -d "speeches_matched" ]; then
    echo "  speeches_matched/: $(ls speeches_matched/*.parquet 2>/dev/null | wc -l | tr -d ' ') files"
fi
if [ -d "debates_matched" ]; then
    echo "  debates_matched/:  $(ls debates_matched/*.parquet 2>/dev/null | wc -l | tr -d ' ') files"
fi
if [ -d "speeches_full" ]; then
    echo "  speeches_full/:    $(ls speeches_full/*.parquet 2>/dev/null | wc -l | tr -d ' ') files"
fi
if [ -d "debates_full" ]; then
    echo "  debates_full/:     $(ls debates_full/*.parquet 2>/dev/null | wc -l | tr -d ' ') files"
fi

echo ""
echo "All schemas now include 'position' field for ordering speeches"
echo "All speeches can link to debates via 'debate_id'"
echo ""
echo "================================================================================"
echo "COMPLETE!"
echo "================================================================================"
