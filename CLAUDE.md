# CLAUDE.md

Analysis tools for UK Parliamentary debates (Hansard) with gender analysis focus.

## Visualization Standards

### Color Palette
```python
COLORS = {
    'male': '#3B82C4',
    'female': '#EC4899',
    'background': '#FFFFFF',
    'grid': '#E5E7EB',
    'text': '#1F2937',
    'muted': '#9CA3AF',
    'accent1': '#10B981',
    'accent2': '#F59E0B',
}
```

### Figure Settings
- DPI: 150+ display, 300 saved
- Size: 10x6 max for single charts
- Remove: top/right spines, excessive gridlines
- Background: white only

### Chart Rules
- NO: pie charts, word clouds, 3D charts, default matplotlib colors, empty subplots
- NO: common words ("the", "of", "to") in word frequency charts
- YES: temporal trends (line/area), TF-IDF (not raw frequencies), horizontal bar charts
- Every chart: answer a question, label axes, include sample size

### Matplotlib Setup
```python
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
```

## Quick Start
```bash
# Quick test
python3 src/hansard/analysis/comprehensive_analysis.py --years 1990-2000 --sample 5000

# Full analysis
python3 src/hansard/analysis/comprehensive_analysis.py --filtering aggressive
```

## Repository Structure
- `src/hansard/analysis/` - Analysis scripts
- `src/hansard/utils/` - Data loaders, path config
- `data-hansard/processed_fixed/` - Source of truth (14GB)
- `analysis/` - Output visualizations

## Core Rules

### ASCII Only
- No emojis, curly quotes, em dashes, or Unicode in code/docs
- Standard ASCII only: straight quotes "", regular dashes -

### No Arbitrary Limits
- NEVER use `[:10]` without CLI control
- Default: process ALL data
- Use: `files[:args.limit] if args.limit else files`

### Test After Changes
- After moving/refactoring: verify imports and execution
- Test with --help, small samples, then full data
- Run pytest before committing

### Dataset Changes
- Record baseline metrics before changes
- Test across time periods (1800s, 1900s, 2000s)
- Manual review 50+ samples for false positives
- Never decrease match rate by >1%

### Documentation
- Update README when changing structure
- Delete planning docs after task completion
- Keep repo lean: no extraneous files at root
- Prefer self-documenting code

## Troubleshooting
1. Empty visualizations: check data loading/filtering
2. Ugly colors: use palette above
3. Overcrowded: limit to 10-15 items, use small multiples
4. Slow: sample first, then full analysis
5. Partial: check for arbitrary [:10] limits
