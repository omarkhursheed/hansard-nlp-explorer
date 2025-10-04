# Data Visualization Style Guide

## Core Principles

### 1. **Clarity Above All**
- Every visualization must have a clear purpose and message
- Remove all unnecessary elements (maximize data-ink ratio)
- If it doesn't help understanding, remove it

### 2. **Professional Color Palette**
```python
COLORS = {
    # Primary colors (gender-based analysis)
    'male': '#3B82C4',      # Professional blue
    'female': '#EC4899',     # Professional pink/magenta

    # Neutral colors
    'background': '#FFFFFF', # White
    'grid': '#E5E7EB',      # Light gray for gridlines
    'text': '#1F2937',       # Dark gray for text
    'muted': '#9CA3AF',      # Medium gray for de-emphasized elements

    # Accent colors for categories
    'accent1': '#10B981',    # Emerald
    'accent2': '#F59E0B',    # Amber
    'accent3': '#8B5CF6',    # Violet
    'accent4': '#EF4444',    # Red
    'accent5': '#06B6D4',    # Cyan

    # Sequential color maps
    'sequential': 'viridis',  # For continuous data
    'diverging': 'RdBu_r',    # For diverging data
}
```

### 3. **Typography**
- Title: 14-16pt, bold, dark gray
- Axis labels: 11-12pt, medium, dark gray
- Tick labels: 10pt, regular, medium gray
- Annotations: 10pt, italic, medium gray
- Use system fonts: 'Helvetica Neue', 'Arial', sans-serif

### 4. **Layout Rules**
- **DPI**: Always 150+ for screen, 300 for print
- **Aspect ratio**: Golden ratio (1.618) or 4:3 for single charts
- **Margins**: Adequate white space (at least 10% of figure size)
- **Grid**: Subtle, only when it aids reading

### 5. **Chart Selection Guide**

| Data Type | Best Chart Types | Avoid |
|-----------|-----------------|-------|
| Time series | Line chart, area chart | Pie chart, 3D charts |
| Comparison | Bar chart, dot plot | Stacked bars (unless necessary) |
| Distribution | Histogram, violin plot, box plot | Pie chart |
| Correlation | Scatter plot, heatmap | 3D scatter |
| Part-to-whole | Stacked bar, treemap | Pie chart (unless 2-3 categories) |
| Text/Words | Word cloud (sparingly), bar chart | 3D word clouds |

### 6. **Specific Guidelines**

#### Bar Charts
- Start y-axis at zero
- Order bars meaningfully (by value, time, or category)
- Use horizontal bars for long labels
- Limit to 10-15 bars per chart

#### Line Charts
- Use for continuous data only
- Maximum 5-7 lines per chart
- Direct labeling preferred over legends
- Consider small multiples for many series

#### Heatmaps
- Use perceptually uniform colormaps
- Include color bar with clear labels
- Consider annotations for key values
- White or light gray for null/missing data

#### Word Clouds
- Use sparingly and only when word frequency is the main message
- Ensure contrast between words and background
- Limit to 50-100 most important words

### 7. **What to Visualize for Gender Analysis**

#### Essential Visualizations
1. **Temporal Participation**: Line chart showing male/female speech counts over time
2. **Vocabulary Differences**: Horizontal bar chart of distinctive words (TF-IDF or log-odds)
3. **Topic Distribution**: Small multiples showing topic prevalence by gender
4. **Speaking Time Distribution**: Violin plots or histograms
5. **Key Metrics Summary**: Clean data table or infographic

#### Avoid These
- Pie charts for gender distribution (use bar chart or waffle chart)
- Word clouds of common words (meaningless)
- 3D charts of any kind
- Overlapping dense scatter plots
- Charts with more than 7 colors

### 8. **Implementation in Python**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style at beginning of script
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.color'] = '#E5E7EB'
```

### 9. **File Organization**
- Each visualization should be a separate file with descriptive name
- Format: `{analysis_type}_{specific_metric}_{date}.png`
- Example: `gender_temporal_participation_1900_2000.png`

### 10. **Quality Checklist**
- [ ] Clear title explaining what the chart shows
- [ ] Labeled axes with units
- [ ] Legend only if necessary (prefer direct labeling)
- [ ] Appropriate chart type for data
- [ ] Colors accessible for colorblind viewers
- [ ] No chartjunk or unnecessary decoration
- [ ] Data source noted if relevant
- [ ] Key findings annotated or highlighted