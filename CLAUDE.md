# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview
This repository contains analysis tools for UK Parliamentary debates (Hansard) with a focus on gender analysis of MP speeches.

## Data Visualization Style Guide

### CRITICAL: Always follow these visualization principles

#### 1. **Professional Color Palette**
```python
COLORS = {
    'male': '#3B82C4',      # Professional blue
    'female': '#EC4899',     # Professional pink/magenta
    'background': '#FFFFFF',  # White
    'grid': '#E5E7EB',       # Light gray for gridlines
    'text': '#1F2937',       # Dark gray for text
    'muted': '#9CA3AF',      # Medium gray
    'accent1': '#10B981',    # Emerald
    'accent2': '#F59E0B',    # Amber
}
```

#### 2. **Figure Quality Settings**
- **Always** use DPI of 150+ for display, 300 for saved files
- **Never** create figures larger than necessary (typically 10x6 inches for single charts)
- **Remove** all unnecessary elements (spines, excessive grid lines)
- **Use** white background, no gray backgrounds

#### 3. **What NOT to Do**
- **NO** empty subplots - remove or hide unused axes
- **NO** pie charts unless absolutely necessary (use bar charts)
- **NO** word clouds unless specifically requested
- **NO** 3D charts ever
- **NO** default matplotlib colors (blue/orange)
- **NO** showing common words like "the", "of", "to" in word frequency charts

#### 4. **What TO Visualize**
- **Temporal trends**: Line or area charts showing change over time
- **Distinctive vocabulary**: Log-odds ratio or TF-IDF, NOT raw frequencies
- **Distributions**: Density plots or histograms with clear labels
- **Comparisons**: Horizontal bar charts for readability
- **Topics**: Small multiples with consistent scales

#### 5. **Chart Requirements**
Every chart must:
- Answer a specific question (stated in title or caption)
- Have labeled axes with units
- Include sample size if relevant
- Be self-contained and understandable
- Use professional fonts (Helvetica Neue, Arial)

#### 6. **File Organization**
- Save each visualization as a separate file
- Use descriptive names: `temporal_participation_1900_2000.png`
- Never combine unrelated visualizations in one figure

#### 7. **Before Creating Any Visualization**
Ask yourself:
1. What question does this answer?
2. Is there sufficient data (minimum 100 data points)?
3. Is this the best chart type for this data?
4. Will this look professional in an academic paper?

### Code Patterns

#### Always set style first:
```python
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
```

#### Use unified stop words from analysis_utils:
```python
from analysis.analysis_utils import get_stop_words, preprocess_text

# Use evidence-based stop word filtering
stop_words = get_stop_words('aggressive')  # or 'moderate', 'parliamentary', etc.
processed_text = preprocess_text(raw_text, stop_words)

# All filtering happens at preprocessing - NO additional filtering at visualization time
```

## Testing and Running Scripts

### Quick tests (small samples):
```bash
# Overall corpus analysis
python3 src/hansard/analysis/comprehensive_analysis.py --years 1990-2000 --sample 5000 --filtering aggressive

# Gender analysis
python3 src/hansard/analysis/gendered_comprehensive_analysis.py --years 1990-2000
```

### Full analysis:
```bash
# Overall corpus (802K debates, ~45-60 minutes)
python3 src/hansard/analysis/comprehensive_analysis.py --filtering aggressive

# Gender corpus
python3 src/hansard/analysis/gendered_comprehensive_analysis.py
```

## Repository Structure
- `src/hansard/analysis/` - Analysis scripts
  - `comprehensive_analysis.py` - Overall corpus analysis
  - `gendered_comprehensive_analysis.py` - Gender-specific analysis
  - `analysis_utils.py` - Shared utilities (DRY principle)
  - `basic_analysis.py` - Simple analysis
  - `suffrage_analysis.py` - Historical analysis
- `src/hansard/utils/` - Shared utilities
  - `unified_data_loader.py` - Single data loading interface
  - `path_config.py` - Path management
- `data-hansard/` - Data files (63GB, not in git)
  - `hansard/` - Raw HTML (5.7GB)
  - `processed_fixed/` - Overall corpus (14GB, SOURCE OF TRUTH)
  - `gender_analysis_enhanced/` - Gender-tagged corpus (9.1GB)
  - `derived/` - Optimized views (1.5GB, can regenerate)
- `analysis/` - Output directory for results and visualizations

## Key Commands
- Use `python3` explicitly to avoid Python version issues
- Check data structure with small samples before running full analysis
- Run tests before committing: `pytest src/hansard/tests/ -v`

## Code Quality Rules

### CRITICAL: Never use arbitrary limits in code
- **NEVER** use arbitrary slicing like `[:10]` in production code
- **NEVER** limit processing without clear reason and user control
- **NEVER** stop analysis partway through data without explicit parameter
- **NEVER** limit date ranges in visualizations - show ALL available data
- **BAD**: `for file in files[:10]:  # Limit for testing`
- **GOOD**: `for file in files[:args.limit] if args.limit else files:`

### Testing vs Production
- Testing limits should ONLY be controlled via command-line arguments
- Default behavior should process ALL available data
- Any limits must be clearly documented and controllable

Example of proper limiting:
```python
parser.add_argument('--limit', type=int, help='Limit number of files to process (for testing)')
# Then in code:
files_to_process = all_files[:args.limit] if args.limit else all_files
```

## Development Process Rules

### CRITICAL: Update Documentation After Code Changes

**ALWAYS** update related documentation when changing code:
- If you change paths or directory structure, update README.md
- If you modify analysis outputs, update relevant documentation
- Planning docs (*.md in docs/) are temporary - delete after task completion
- Commit before deleting so they can be recovered if needed
- Prefer self-documenting code over extensive documentation

### CRITICAL: Always Create Detailed Plans Before Implementation

**NEVER** make ad-hoc patches or quick fixes. For ANY new functionality:

1. **Create a Detailed Plan** including:
   - Problem analysis with evidence
   - Multiple solution options with pros/cons
   - Justification for chosen approach
   - Implementation schema
   - Expected outputs
   - Mock visualizations/layouts

2. **For Visualizations**, plan must include:
   - Sketch/description of layout
   - Verification of no overlaps
   - Axis ranges and scaling
   - Color schemes and accessibility
   - File size estimates

3. **For Rule-Based Systems** (stop words, filters, thresholds):
   - **NEVER** add rules without justification
   - Provide evidence/data for why rule is needed
   - Compare against standard approaches
   - Show impact with/without the rule
   - Document reproducibility concerns

### Example Planning Template

```markdown
## Problem: [Describe issue]

### Evidence
- [Data/screenshots showing the problem]
- [Metrics demonstrating impact]

### Solution Options
1. **Option A**: [Description]
   - Pros: [List]
   - Cons: [List]
   - Evidence: [Why this might work]

2. **Option B**: [Description]
   - Pros: [List]
   - Cons: [List]
   - Evidence: [Why this might work]

### Chosen Approach
[Which option and WHY, with evidence]

### Implementation Schema
- Input: [Data structure]
- Processing: [Steps]
- Output: [Expected structure]

### Mock Outputs
[ASCII diagrams or descriptions of expected visualizations]
```

### Justifying Custom Rules

When proposing custom filters/rules/thresholds:

1. **Baseline Comparison**: Show results with standard approach
2. **Problem Evidence**: Demonstrate specific issues with baseline
3. **Custom Solution**: Explain what changes and why
4. **Impact Analysis**: Show before/after comparison
5. **Reproducibility**: Ensure others can recreate/modify

Example:
```python
# BAD: Arbitrary custom rule
if word in ['which', 'has', 'will']:  # Why these?
    filter_word()

# GOOD: Justified custom rule
# Standard NLTK stopwords miss high-frequency parliamentary terms
# Analysis of 1M speeches shows these appear in >80% of speeches
# but carry no semantic value for topic analysis
PARLIAMENTARY_STOPS = {
    'hon': 0.89,  # frequency in corpus
    'gentleman': 0.84,
    'member': 0.91,
    # ... with evidence for each
}
```

## Common Issues and Solutions
1. **Empty visualizations**: Check that data is properly loaded and filtered
2. **Ugly colors**: Use the professional color palette defined above
3. **Overcrowded charts**: Limit to top 10-15 items, use small multiples for more
4. **Slow performance**: Use sampling for initial tests, then run full analysis
5. **Partial analysis**: Check for arbitrary limits like `[:10]` in loops