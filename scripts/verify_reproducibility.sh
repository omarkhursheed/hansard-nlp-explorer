#!/bin/bash
#
# Reproducibility Verification Script
# Run this BEFORE cleanup to establish baseline, then AFTER cleanup to verify
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================================================="
echo "REPRODUCIBILITY VERIFICATION"
echo "=============================================================================="
echo ""

# Check if baseline exists
BASELINE_DIR="src/hansard/data/gender_analysis_enhanced.BACKUP"
CURRENT_DIR="src/hansard/data/gender_analysis_enhanced"

if [[ "$1" == "baseline" ]]; then
    echo "MODE: Creating baseline for comparison"
    echo ""

    if [[ ! -d "$CURRENT_DIR" ]]; then
        echo -e "${RED}ERROR: No dataset found at $CURRENT_DIR${NC}"
        echo "Please generate dataset first with:"
        echo "  bash src/hansard/scripts/run_enhanced_gender_dataset.sh"
        exit 1
    fi

    echo "Step 1: Creating backup of current dataset..."
    if [[ -d "$BASELINE_DIR" ]]; then
        echo -e "${YELLOW}WARNING: Baseline already exists at $BASELINE_DIR${NC}"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 1
        fi
        rm -rf "$BASELINE_DIR"
    fi

    cp -r "$CURRENT_DIR" "$BASELINE_DIR"
    echo -e "${GREEN}✓${NC} Baseline created at $BASELINE_DIR"

    echo ""
    echo "Step 2: Computing file hashes..."
    cd "$BASELINE_DIR"
    find . -name "*.parquet" -type f | sort | xargs sha256sum > ../baseline_hashes.txt
    echo -e "${GREEN}✓${NC} Hashes saved to baseline_hashes.txt"

    cd "$SCRIPT_DIR"
    echo ""
    echo "Step 3: Recording dataset statistics..."
    python3 << 'EOF'
import pandas as pd
from pathlib import Path
import json

data_dir = Path("src/hansard/data/gender_analysis_enhanced.BACKUP")
master_file = data_dir / "ALL_debates_enhanced_with_text.parquet"

if master_file.exists():
    df = pd.read_parquet(master_file)
    stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "female_debates": int(df['has_female'].sum()) if 'has_female' in df.columns else 0,
        "male_debates": int(df['has_male'].sum()) if 'has_male' in df.columns else 0,
    }

    with open("baseline_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Dataset stats: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  - Debates with female MPs: {stats['female_debates']:,}")
    print(f"  - Debates with male MPs: {stats['male_debates']:,}")
else:
    print("⚠ Warning: Master file not found")
EOF

    echo ""
    echo "=============================================================================="
    echo -e "${GREEN}BASELINE CREATED SUCCESSFULLY${NC}"
    echo "=============================================================================="
    echo ""
    echo "Next steps:"
    echo "  1. Make your code changes (cleanup, refactoring, etc.)"
    echo "  2. Regenerate dataset: bash src/hansard/scripts/run_enhanced_gender_dataset.sh"
    echo "  3. Run: bash verify_reproducibility.sh compare"
    echo ""

elif [[ "$1" == "compare" ]]; then
    echo "MODE: Comparing current dataset with baseline"
    echo ""

    if [[ ! -d "$BASELINE_DIR" ]]; then
        echo -e "${RED}ERROR: No baseline found at $BASELINE_DIR${NC}"
        echo "Please create baseline first with:"
        echo "  bash verify_reproducibility.sh baseline"
        exit 1
    fi

    if [[ ! -d "$CURRENT_DIR" ]]; then
        echo -e "${RED}ERROR: No current dataset found at $CURRENT_DIR${NC}"
        echo "Please regenerate dataset first with:"
        echo "  bash src/hansard/scripts/run_enhanced_gender_dataset.sh"
        exit 1
    fi

    echo "Step 1: Quick hash comparison..."
    cd "$CURRENT_DIR"
    find . -name "*.parquet" -type f | sort | xargs sha256sum > ../new_hashes.txt
    cd "$SCRIPT_DIR"

    # Compare hashes (excluding timestamps)
    echo "Comparing file hashes..."
    if diff -q baseline_hashes.txt new_hashes.txt > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} File hashes are IDENTICAL"
        HASH_MATCH=true
    else
        echo -e "${YELLOW}⚠${NC} File hashes differ (expected due to timestamps)"
        HASH_MATCH=false
    fi

    echo ""
    echo "Step 2: Deep comparison with regression test..."
    python3 tests/test_reproducibility.py --baseline "$BASELINE_DIR" --new "$CURRENT_DIR"
    TEST_RESULT=$?

    echo ""
    echo "=============================================================================="
    if [[ $TEST_RESULT -eq 0 ]]; then
        echo -e "${GREEN}SUCCESS: DATASETS ARE REPRODUCIBLE${NC}"
        echo "=============================================================================="
        echo ""
        echo "✓ Your changes did not affect dataset generation"
        echo "✓ Analysis outputs will be identical (except timestamps)"
        echo ""
        echo "You can safely:"
        echo "  - Delete the baseline: rm -rf $BASELINE_DIR"
        echo "  - Commit your changes"
        echo ""
    else
        echo -e "${RED}WARNING: DIFFERENCES DETECTED${NC}"
        echo "=============================================================================="
        echo ""
        echo "The regression test found differences between datasets."
        echo "Review the test output above to understand what changed."
        echo ""
        echo "If changes are expected and acceptable:"
        echo "  - Review the differences carefully"
        echo "  - Update baseline: bash verify_reproducibility.sh baseline"
        echo ""
        echo "If changes are unexpected:"
        echo "  - Investigate code changes"
        echo "  - Check for non-deterministic operations"
        echo ""
    fi

else
    echo "USAGE:"
    echo "  bash verify_reproducibility.sh baseline    # Create baseline before changes"
    echo "  bash verify_reproducibility.sh compare     # Compare after regenerating data"
    echo ""
    echo "WORKFLOW:"
    echo "  1. bash verify_reproducibility.sh baseline"
    echo "  2. [Make your code changes]"
    echo "  3. bash src/hansard/scripts/run_enhanced_gender_dataset.sh"
    echo "  4. bash verify_reproducibility.sh compare"
    echo ""
fi
