#!/bin/bash

# Hansard Analysis Commands - Ready to Run
# ==========================================

echo "Hansard NLP Analysis - Content-Focused"
echo "======================================="
echo ""

# Change to correct directory
cd /Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard

# 1. COMPARISON: See how filtering progressively reveals content
echo "1. Running filter level comparison (1920s-1930s)..."
python analysis/hansard_nlp_analysis_advanced.py --years 1920-1930 --sample 500 --all-levels

# 2. KEY HISTORICAL PERIODS with aggressive filtering (Level 4)
echo ""
echo "2. Analyzing key historical periods..."

# Women's Suffrage Era
echo "   - Women's Suffrage (1918-1928)..."
python analysis/hansard_nlp_analysis_advanced.py --years 1918-1928 --sample 1500 --filter-level 4

# Great Depression
echo "   - Great Depression (1929-1939)..."
python analysis/hansard_nlp_analysis_advanced.py --years 1929-1939 --sample 1500 --filter-level 4

# World War II
echo "   - World War II (1939-1945)..."
python analysis/hansard_nlp_analysis_advanced.py --years 1939-1945 --sample 1500 --filter-level 4

# Post-War Reconstruction
echo "   - Post-War Reconstruction (1945-1951)..."
python analysis/hansard_nlp_analysis_advanced.py --years 1945-1951 --sample 1500 --filter-level 4

# Thatcher Era
echo "   - Thatcher Era (1979-1990)..."
python analysis/hansard_nlp_analysis_advanced.py --years 1979-1990 --sample 1500 --filter-level 4

# 3. DISTINCTIVE TERMS: Find period-specific vocabulary using TF-IDF (Level 5)
echo ""
echo "3. Finding period-specific distinctive terms..."

# Victorian Era
echo "   - Victorian Era (1870-1900)..."
python analysis/hansard_nlp_analysis_advanced.py --years 1870-1900 --sample 1000 --filter-level 5

# Edwardian Era
echo "   - Edwardian Era (1901-1910)..."
python analysis/hansard_nlp_analysis_advanced.py --years 1901-1910 --sample 1000 --filter-level 5

# Swinging Sixties
echo "   - 1960s..."
python analysis/hansard_nlp_analysis_advanced.py --years 1960-1970 --sample 1000 --filter-level 5

# Blair Years
echo "   - Blair Years (1997-2005)..."
python analysis/hansard_nlp_analysis_advanced.py --years 1997-2005 --sample 1000 --filter-level 5

echo ""
echo "Analysis complete! Check results in analysis/results_advanced/"
echo ""
echo "Visualizations for each period are in:"
echo "  analysis/results_advanced/plots_level_4/ (for policy content)"
echo "  analysis/results_advanced/plots_level_5/ (for distinctive terms)"