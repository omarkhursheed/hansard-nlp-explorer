# Hansard NLP Analysis - Quick Start Guide

This directory contains comprehensive tools for analyzing the Hansard parliamentary debates corpus with sophisticated filtering and visualization capabilities.

## Two Main Analysis Types

### 1. Overall Corpus Analysis
**Command:** `./run_corpus_analysis.sh`

Analyzes the corpus across **6 different filtering levels** (0-5) to reveal both procedural language and substantive content:

- **Level 0 (NONE):** No filtering - raw parliamentary language
- **Level 1 (BASIC):** Remove basic English stop words
- **Level 2 (PARLIAMENTARY):** Remove parliamentary procedural terms
- **Level 3 (MODERATE):** Remove common governmental terms
- **Level 4 (AGGRESSIVE):** Focus on substantive policy content
- **Level 5 (TFIDF):** Use TF-IDF to identify most important terms

**Usage Examples:**
```bash
# Quick test with sample data
./run_corpus_analysis.sh --years 1920-1935 --sample 1000

# World War II period analysis
./run_corpus_analysis.sh --years 1939-1945 --sample 2000

# Full corpus analysis (large scale)
./run_corpus_analysis.sh --full --sample 10000
```

**Outputs:**
- `analysis/corpus_results/` - Comprehensive visualizations for each filtering level
- `analysis/corpus_results/level_X_plots/` - Individual level analysis plots
- `analysis/corpus_results/filtering_levels_comparison.png` - Cross-level comparison
- `analysis/corpus_results/comprehensive_analysis_results.json` - All results data

### 2. Historical Milestone Analysis
**Command:** `./run_milestone_analysis.sh`

Analyzes key historical periods with before/during/after comparisons:

**Available Milestones:**
- `1918_partial_suffrage` - Women over 30 gain vote + enter Parliament
- `1928_full_suffrage` - Full women's suffrage (age 21)
- `ww1_period` - World War I (1914-1918)
- `ww2_period` - World War II (1939-1945) 
- `thatcher_period` - Margaret Thatcher's tenure (1979-1990)

**Usage Examples:**
```bash
# Analyze all milestones with aggressive filtering
./run_milestone_analysis.sh --all --filtering aggressive

# Focus on WWII with moderate filtering
./run_milestone_analysis.sh --milestone ww2_period --filtering moderate

# Full women's suffrage analysis
./run_milestone_analysis.sh --milestone 1928_full_suffrage
```

**Outputs:**
- `analysis/milestone_results/MILESTONE/FILTERING/` - Individual milestone analysis
- `milestone_comprehensive_visualization.png` - Multi-period comparison
- `milestone_impact_summary.png` - Key changes summary
- `milestone_report.md` - Detailed findings report
- `MASTER_SUMMARY_FILTERING.md` - Cross-milestone summary

## Key Features

### Enhanced Visualizations
- **No unnecessary plots** (debate counts removed per your request)
- **Improved topic modeling** with heatmap visualizations
- **Clean gender analysis** with both language and speaker metrics
- **Comprehensive filtering impact** charts
- **Professional publication-ready** graphics at 300 DPI

### Filtering Analysis
- **Multi-level approach** reveals different aspects of parliamentary discourse
- **Quantified filtering impact** (X% word reduction)
- **Consistent gender analysis** across all filtering levels
- **Topic evolution** tracking from procedural to substantive content

### Historical Milestone Features
- **Before/During/After analysis** where applicable (WWI, WWII, Thatcher)
- **Gender representation tracking** (language + speakers)
- **Content evolution analysis** (new/disappeared/persistent words)
- **Quantified impact measurements** (percentage point changes)

## File Organization

```
analysis/
├── comprehensive_corpus_analysis.py      # Main corpus analysis script
├── comprehensive_milestone_analysis.py   # Milestone analysis script
├── corpus_results/                       # Corpus analysis outputs
│   ├── level_X_plots/                   # Individual filtering level results
│   └── filtering_levels_comparison.png   # Cross-level comparison
├── milestone_results/                    # Milestone analysis outputs
│   ├── MILESTONE_NAME/
│   │   └── FILTERING_MODE/
│   │       ├── milestone_comprehensive_visualization.png
│   │       ├── milestone_impact_summary.png
│   │       └── milestone_report.md
│   └── MASTER_SUMMARY_FILTERING.md
└── results_advanced/                     # Legacy results (for reference)
```

## Quick Commands for Document Update

Based on your Google Doc requirements, here are the key commands to generate results:

### For Overall Corpus Section:
```bash
# Generate comprehensive filtering analysis
./run_corpus_analysis.sh --years 1920-1935 --sample 1000
```

### For Milestone Sections:
```bash
# Generate all milestone analyses with aggressive filtering
./run_milestone_analysis.sh --all --filtering aggressive

# Or individual milestones for specific sections
./run_milestone_analysis.sh --milestone ww2_period --filtering aggressive
./run_milestone_analysis.sh --milestone 1928_full_suffrage --filtering moderate
```

## Data Quality Notes
- **Filtering preserves gender analysis accuracy** (always uses original text)
- **Speaker identification** uses title-based heuristics
- **Topic modeling** requires minimum thresholds (100+ texts, 1000+ words)
- **Sampling** maintains year distribution for representative results
- **During periods** included where historically relevant (wars, Thatcher tenure)

## Performance Expectations
- **Small sample (1000 debates):** ~2-3 minutes
- **Medium sample (5000 debates):** ~10-15 minutes  
- **Large sample (10000 debates):** ~30-45 minutes
- **Full milestone analysis:** ~15-30 minutes per milestone

The scripts will automatically create output directories and handle all visualization generation. Results are saved in both JSON (for data) and PNG (for visualizations) formats ready for document inclusion.