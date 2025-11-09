# Manuscript Analysis Scripts

Clean, focused scripts for generating publication-ready figures from the Hansard dataset.

## Overview

These scripts work directly with the `derived_complete` dataset (5.97M speeches, 1.2M debates) to generate manuscript figures. All analysis focuses on **House of Commons only** by default.

## Directory Structure

```
manuscript/
├── data_loader.py          # Simple data loader (30 lines)
├── utils.py                # Visualization styling and helpers
├── 01_summary_stats.py     # Dataset statistics (JSON output)
├── 02_temporal_trends.py   # Temporal participation trends
├── 03_ngram_analysis.py    # Distinctive vocabulary analysis
├── 04_milestones.py        # Historical milestone analysis
└── README.md               # This file

manuscript_figures/         # Output directory
├── stats.json              # All statistics
├── temporal_*.png          # Temporal visualizations
├── ngrams_*.png            # Vocabulary visualizations
├── ngrams_*.csv            # Top terms data
└── milestones_*.png        # Milestone visualizations
```

## Quick Start

### 1. Generate Summary Statistics

```bash
cd src/hansard/manuscript
python3 01_summary_stats.py
```

Output: `manuscript_figures/stats.json` with all key numbers for the paper.

### 2. Generate Temporal Trends

```bash
python3 02_temporal_trends.py
# Or specify year range:
python3 02_temporal_trends.py --start 1900 --end 2005
```

Generates:
- `temporal_speaking_proportion.png` - Female speaking % over time
- `temporal_dual_axis.png` - Speaking + MP presence (if MP data provided)
- `temporal_absolute_counts.png` - Speech volume by gender

### 3. Generate Ngram Analysis

```bash
# Gender split (recommended)
python3 03_ngram_analysis.py --split gender --filtering moderate

# Temporal split
python3 03_ngram_analysis.py --split temporal --filtering moderate

# Custom time periods
python3 03_ngram_analysis.py --split temporal --periods "1900-1950:Early,1950-2005:Modern"
```

**Filtering levels:**
- `minimal`: NLTK stopwords only (~200 words)
- `parliamentary`: + parliamentary terms (~60 words)
- `moderate`: + common verbs + vague words (**RECOMMENDED**)
- `aggressive`: + discourse markers + quantifiers (~400 words)

Generates:
- `ngrams_gender_*.png` - Distinctive vocabulary by gender
- `ngrams_temporal_*.png` - Vocabulary changes over time
- `ngrams_*.csv` - Top terms data

### 4. Generate Milestone Analysis

```bash
python3 04_milestones.py
# Or specify window size:
python3 04_milestones.py --window 10
```

Analyzes events:
- 1919: Nancy Astor (first woman MP)
- 1928: Equal voting rights
- 1939-1945: WW2 period
- 1979: Thatcher becomes PM
- 1997: Blair's 101 women MPs

Generates:
- `milestones_before_after.png` - Before/after comparison
- `milestones_timeseries.png` - Continuous trend with markers
- `milestones_small_multiples.png` - Individual event windows
- `milestone_statistics.csv` - Numerical data

## Color Scheme Options

The scripts currently use the `default` matplotlib color scheme. To change:

1. Run scripts to generate figures with different schemes
2. Edit `utils.py` line 61: `CURRENT_SCHEME = 'default'`

Available schemes:
- `default`: Matplotlib blue/orange
- `blue_pink`: Traditional gender colors
- `colorblind`: Colorblind-friendly palette
- `viridis`: Purple/yellow
- `nature`: Green/terracotta

After choosing, update `CLAUDE.md` with the selected scheme.

## Advanced Options

### Include Lords Data

By default, all scripts analyze Commons only. To include Lords:

```bash
python3 02_temporal_trends.py --include-lords
python3 03_ngram_analysis.py --include-lords
python3 04_milestones.py --include-lords
```

### Custom Analysis

Each script is <200 lines and self-contained. Copy and modify for custom analyses:

```python
from data_loader import load_speeches
from utils import setup_plot_style, COLORS, save_figure

# Load data
speeches = load_speeches(year_range=(1990, 2000), gender='f', chamber='Commons')

# Analyze
# ... your custom analysis ...

# Plot
setup_plot_style()
fig, ax = plt.subplots()
# ... your custom plot ...
save_figure(fig, 'custom_analysis.png')
```

## Data Loader API

```python
from data_loader import load_speeches, load_debates

# Load speeches with filters
speeches = load_speeches(
    year_range=(1900, 2000),  # Optional: (start, end) tuple
    gender='f',                # Optional: 'm', 'f', 'male', 'female'
    chamber='Commons',         # Optional: 'Commons', 'Lords'
    sample=1000               # Optional: for testing
)

# Load debate metadata
debates = load_debates(
    year_range=(1900, 2000),
    chamber='Commons'
)
```

## Output Files

All outputs go to `manuscript_figures/` at the project root:
- PNG files: 300 DPI, publication-ready
- CSV files: Top terms with frequencies and log-odds
- JSON files: Structured statistics

## Dependencies

- pandas
- matplotlib
- numpy
- scikit-learn
- nltk (for stopwords)

All dependencies should already be installed in your environment.

## Performance

Loading times (approximate):
- Full dataset: ~30-60 seconds
- Single decade: ~3-5 seconds
- Gender filter: Instant (after loading)

The scripts use direct parquet loading - no complex joins or processing needed.

## Troubleshooting

**"No such file" error:**
- Check that `data-hansard/derived_complete/` exists
- Run from project root or ensure paths are correct

**Memory issues:**
- Use `sample` parameter for testing
- Process year ranges instead of full dataset
- Close figures after saving: `plt.close(fig)`

**Empty visualizations:**
- Check year ranges match data availability (1803-2005)
- Verify chamber filter isn't too restrictive
- Check gender filter returns data

## Next Steps

1. Run all scripts to generate initial figures
2. Review outputs in `manuscript_figures/`
3. Choose color scheme and update `utils.py`
4. Customize plots as needed for manuscript
5. Update `CLAUDE.md` with chosen styling
