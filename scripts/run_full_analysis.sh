#!/bin/bash
#
# HANSARD NLP EXPLORER - COMPLETE ANALYSIS PIPELINE
#
# This script runs the full analysis pipeline on all Hansard data using
# the unified analysis system.
#
# USAGE:
#   ./run_full_analysis.sh [--quick|--standard|--full]
#
# OPTIONS:
#   --quick      Quick test with small sample (~5 min)
#                - 1,000 speeches, 1995-2000 (for testing)
#   [default]    Full analysis - NO SAMPLING (~60-90 min) [DEFAULT]
#                - ALL 2 million speeches, 1803-2005
#                - Complete historical record
#                - All milestones with full data
#
# OUTPUTS:
#   analysis/
#   ├── corpus_gender/       # Gender corpus analysis
#   ├── corpus_overall/      # Overall corpus analysis
#   ├── milestones_gender/   # Gender milestone analysis
#   ├── milestones_overall/  # Overall milestone analysis
#   └── temporal_speakers/   # Speaker temporal trends
#

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo ""
    echo "================================================================================"
    echo -e "${BLUE}$1${NC}"
    echo "================================================================================"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}→${NC} $1"
}

# Parse arguments
MODE="standard"
if [[ "$1" == "--quick" ]]; then
    MODE="quick"
    SAMPLE="--sample 1000"
    YEARS="1995-2000"
    MILESTONE_SAMPLE="--sample 500"
    print_info "MODE: Quick test (1k speeches, 1995-2000, ~5 min)"
else
    # Standard/Full - use ALL data, no sampling
    MODE="standard"
    SAMPLE=""  # No sampling - use all 2M speeches
    YEARS="--full"  # Full corpus 1803-2005
    MILESTONE_SAMPLE=""  # No sampling - all data per milestone
    print_info "MODE: Full analysis (ALL 2M speeches, 1803-2005, ~60-90 min)"
fi

print_header "HANSARD NLP EXPLORER - COMPLETE ANALYSIS PIPELINE"

# Check data exists
if [ ! -d "data-hansard/gender_analysis_enhanced" ]; then
    echo -e "${YELLOW}WARNING: Gender-enhanced dataset not found${NC}"
    echo "Run: ./run_data_generation.sh first to generate it"
    echo ""
    read -p "Continue with overall corpus only? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    SKIP_GENDER=true
fi

if [ ! -d "data-hansard/processed_fixed" ]; then
    echo -e "${YELLOW}ERROR: Overall corpus not found at data-hansard/processed_fixed${NC}"
    echo "Please ensure processed data exists"
    exit 1
fi

# =============================================================================
# PART 1: CORPUS ANALYSIS
# =============================================================================

print_header "PART 1/4: CORPUS ANALYSIS"

# 1.1 Gender Corpus Analysis
if [[ "$SKIP_GENDER" != "true" ]]; then
    print_info "Running gender corpus analysis..."
    python3 src/hansard/analysis/corpus_analysis.py \
        --dataset gender \
        $YEARS \
        $SAMPLE \
        --filtering aggressive \
        --analysis all \
        --output-dir analysis/corpus_gender

    print_success "Gender corpus analysis complete"
    echo ""
fi

# 1.2 Overall Corpus Analysis
print_info "Running overall corpus analysis..."
python3 src/hansard/analysis/corpus_analysis.py \
    --dataset overall \
    $YEARS \
    $SAMPLE \
    --filtering moderate \
    --analysis unigram,bigram,topic,gender \
    --output-dir analysis/corpus_overall

print_success "Overall corpus analysis complete"
echo ""

# =============================================================================
# PART 2: MILESTONE ANALYSIS
# =============================================================================

print_header "PART 2/4: MILESTONE ANALYSIS"

# 2.1 Gender Milestones
if [[ "$SKIP_GENDER" != "true" ]]; then
    print_info "Running gender milestone analysis..."
    python3 src/hansard/analysis/milestone_analysis.py \
        --all-milestones \
        --dataset gender \
        --filtering aggressive \
        $MILESTONE_SAMPLE \
        --output-dir analysis/milestones_gender

    print_success "Gender milestone analysis complete"
    echo ""
fi

# 2.2 Overall Milestones (skip in quick mode)
if [[ "$MODE" != "quick" ]]; then
    print_info "Running overall milestone analysis..."
    python3 src/hansard/analysis/milestone_analysis.py \
        --all-milestones \
        --dataset overall \
        --filtering moderate \
        $MILESTONE_SAMPLE \
        --output-dir analysis/milestones_overall

    print_success "Overall milestone analysis complete"
    echo ""
fi

# =============================================================================
# PART 3: TEMPORAL SPEAKER ANALYSIS
# =============================================================================

print_header "PART 3/4: TEMPORAL SPEAKER ANALYSIS"

print_info "Running speaker temporal analysis..."
python3 src/hansard/analysis/temporal_gender_analysis.py

print_success "Temporal speaker analysis complete"
echo ""

# =============================================================================
# PART 4: SUMMARY
# =============================================================================

print_header "ANALYSIS COMPLETE!"

echo "Results saved to analysis/ directory:"
echo ""
echo "  analysis/"
if [[ "$SKIP_GENDER" != "true" ]]; then
    echo "  ├── corpus_gender/         # Gender-specific word frequencies, topics, bigrams"
fi
echo "  ├── corpus_overall/        # Overall corpus analysis"
if [[ "$SKIP_GENDER" != "true" ]]; then
    echo "  ├── milestones_gender/     # Historical milestone analysis (gender-matched)"
fi
if [[ "$MODE" != "quick" ]]; then
    echo "  ├── milestones_overall/    # Historical milestone analysis (overall corpus)"
fi
echo "  └── [temporal outputs]     # Speaker gender trends over time"
echo ""

echo "Key visualizations:"
if [[ "$SKIP_GENDER" != "true" ]]; then
    echo "  - Gender vocabulary: analysis/corpus_gender/unigram_comparison.png"
    echo "  - Gender bigrams: analysis/corpus_gender/bigram_comparison.png"
    echo "  - Temporal trends: analysis/corpus_gender/temporal_participation.png"
    echo "  - Gender topics: analysis/corpus_gender/topic_prevalence.png"
fi
echo "  - Overall analysis: analysis/corpus_overall/*.png"
echo ""

echo "JSON results:"
if [[ "$SKIP_GENDER" != "true" ]]; then
    echo "  - analysis/corpus_gender/analysis_results.json"
fi
echo "  - analysis/corpus_overall/analysis_results.json"
if [[ "$SKIP_GENDER" != "true" ]]; then
    echo "  - analysis/milestones_gender/*/moderate/milestone_results.json"
fi
echo ""

print_header "DONE!"

echo "Next steps:"
echo "  - Review visualizations in analysis/"
echo "  - Check JSON results for detailed metrics"
echo "  - See ANALYSIS_GUIDE.md for interpretation"
echo ""
