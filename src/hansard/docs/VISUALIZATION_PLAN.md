# Gender Analysis Visualization Plan

## Overview
Create focused, meaningful visualizations that reveal insights about gender differences in parliamentary discourse. Each visualization should answer a specific question.

## Planned Visualizations

### 1. `temporal_participation.png`
**Question**: How has women's participation in Parliament changed over time?
**Type**: Area chart
**Content**:
- X-axis: Years (1803-2005)
- Y-axis: Number of speeches
- Two filled areas: Male (bottom), Female (stacked on top)
- Key events annotated (1918 suffrage, 1928 equal franchise, 1979 Thatcher)
**Why**: Shows the historical context and growth of female participation

### 2. `distinctive_vocabulary.png`
**Question**: Which words are most distinctively used by male vs female MPs?
**Type**: Diverging horizontal bar chart
**Content**:
- Calculate log-odds ratio for words used by each gender
- Show top 15 most distinctive words for each gender
- Male words point left (blue), female words point right (pink)
- Filter out stop words and parliamentary procedure terms
**Why**: Reveals actual vocabulary differences, not just common words

### 3. `topic_prevalence.png`
**Question**: Do male and female MPs focus on different topics?
**Type**: Small multiples (grid of small bar charts)
**Content**:
- 6-8 discovered topics from LDA
- Each subplot shows male vs female prevalence for that topic
- Topics labeled with their key terms
**Why**: Shows thematic differences in parliamentary focus

### 4. `speech_length_distribution.png`
**Question**: Do male and female MPs speak for different lengths?
**Type**: Overlapping density plots (or violin plots)
**Content**:
- Distribution of speech lengths (in words)
- Separate curves for male and female
- Include median lines and statistical test result
**Why**: Reveals speaking pattern differences

### 5. `participation_by_era.png`
**Question**: How does gender participation vary across different historical periods?
**Type**: Grouped bar chart
**Content**:
- Divide timeline into meaningful eras (Victorian, Edwardian, Interwar, Post-WWII, etc.)
- Show male/female speech counts for each era
- Include percentage labels
**Why**: Provides period-specific context

### 6. `key_metrics_summary.png`
**Question**: What are the key statistics?
**Type**: Infographic-style metrics display
**Content**:
- Total speeches by gender
- Average speech length
- Vocabulary richness (type-token ratio)
- Most frequent topics
- Clean, minimalist design with icons
**Why**: Quick overview of key findings

## Milestone Analysis Visualizations

### 7. `milestone_comparison.png`
**Question**: How did discourse change around key historical events?
**Type**: Before/During/After comparison
**Content**:
- Separate panels for each milestone
- Show changes in vocabulary, topics, or participation
- Use consistent scales for comparison
**Why**: Reveals impact of historical events on parliamentary discourse

### 8. `wartime_language_shift.png`
**Question**: How did language change during wars?
**Type**: Slope chart or connected scatter
**Content**:
- Key terms frequency before vs during war
- Different colors for terms that increased/decreased
- Separate for WWI and WWII
**Why**: Shows specific language shifts during crises

## Technical Implementation Notes

### File Naming Convention
- `{category}_{specific_metric}_{parameters}.png`
- Examples:
  - `temporal_participation_1803_2005.png`
  - `distinctive_vocabulary_filtered_aggressive.png`
  - `milestone_comparison_ww2_period.png`

### Size Guidelines
- Single charts: 10x6 inches (1500x900 pixels at 150 DPI)
- Small multiples: 12x8 inches (1800x1200 pixels)
- Infographics: 8x10 inches (1200x1500 pixels)

### What NOT to Visualize
1. **Word clouds** - Only if specifically requested, and only for distinctive words
2. **Pie charts** - Use bar charts instead
3. **Common words frequency** - Meaningless without filtering
4. **Empty data** - Don't create plots if there's insufficient data
5. **Everything in one figure** - Separate visualizations for different questions

### Data Requirements
Each visualization should:
- Have sufficient data (minimum 100 data points)
- Be statistically meaningful
- Tell a clear story
- Be self-contained (understandable without external context)

## Priority Order
1. Temporal participation (most important context)
2. Distinctive vocabulary (most interesting finding)
3. Speech length distribution (clear, simple insight)
4. Topic prevalence (if topics are meaningful)
5. Others as appropriate

## Quality Metrics
- Can someone understand the main message in 5 seconds?
- Does it answer the stated question clearly?
- Is it accessible to colorblind viewers?
- Would it look professional in an academic paper?
- Does it avoid misleading representations?