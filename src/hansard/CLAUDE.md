# CLAUDE.md - Hansard NLP Explorer

This file provides guidance for working with the Hansard parliamentary debates corpus and NLP analysis tools.

## Project Structure

```
hansard/
├── analysis/
│   ├── hansard_nlp_analysis.py    # Main NLP analysis script
│   └── results/                   # Analysis outputs (visualizations, JSON)
├── data/
│   ├── processed/                 # Processed debates in JSONL format
│   │   ├── content/              # Full text content by year
│   │   ├── metadata/             # Structured metadata (Parquet)
│   │   └── index/                # Search indices
│   └── gender_wordlists/         # UCLA NLP gender classification lists
├── parsers/                      # Data processing and search tools
├── scripts/                      # Production processing scripts
└── crawlers/                     # Data acquisition tools
```

## Data Format

### Debate Records (JSONL)
Each debate is stored as a JSON object with:
- `full_text`: Complete debate text
- `metadata`: Year, chamber, speakers, topics, word count, Hansard reference
- `file_path`: Source HTML file location

### Key Data Sources
- **Content**: `data/processed/content/{year}/debates_{year}.jsonl`
- **Metadata**: `data/processed/metadata/debates_master.parquet`
- **Search Index**: `data/processed/index/debates.db`

## NLP Analysis Commands

### Basic Analysis
```bash
# Test with small sample around women's suffrage
python analysis/hansard_nlp_analysis.py --years 1925-1930 --sample 100

# Analyze specific decades
python analysis/hansard_nlp_analysis.py --years 1900-1910 --sample 500
python analysis/hansard_nlp_analysis.py --years 1950-1960 --sample 1000

# Full period analysis (large dataset)
python analysis/hansard_nlp_analysis.py --years 1850-1950 --sample 5000
```

### Temporal Comparisons
```bash
# Victorian era
python analysis/hansard_nlp_analysis.py --years 1837-1901

# WWI period
python analysis/hansard_nlp_analysis.py --years 1914-1918

# Women's suffrage analysis (key: 1928 split)
python analysis/hansard_nlp_analysis.py --years 1920-1935

# Post-war reconstruction
python analysis/hansard_nlp_analysis.py --years 1945-1955
```

### Full Corpus Analysis
```bash
# Complete historical analysis (WARNING: long runtime)
python analysis/hansard_nlp_analysis.py --full

# Large sample across all years
python analysis/hansard_nlp_analysis.py --full --sample 10000
```

## Analysis Features

### 1. Unigram/Bigram Analysis
- Most frequent single words and word pairs
- Filtered for meaningful parliamentary vocabulary
- Visualization: Top 20 words/15 bigrams bar charts

### 2. Topic Modeling (LDA)
- Identifies major themes in parliamentary discourse
- 8 topics by default, 10 words per topic
- Uses TF-IDF preprocessing for better topic coherence

### 3. Gender Analysis
- Uses UCLA NLP gender wordlists (male/female word classifications)
- Calculates ratios and frequency distributions
- Tracks gendered language patterns over time

### 4. Temporal Analysis (Pre/Post 1928 Women's Suffrage)
- Splits corpus at 1928 (British women's full suffrage)
- Compares language patterns before/after this milestone
- Gender ratio change analysis

## Key Historical Periods for Analysis

### Political Milestones
- **1832**: Great Reform Act
- **1867**: Second Reform Act  
- **1884**: Third Reform Act
- **1918**: Women's partial suffrage (30+)
- **1928**: Women's full suffrage (21+)

### Suggested Analysis Windows
```bash
# Reform Act impacts
python analysis/hansard_nlp_analysis.py --years 1830-1835  # Pre-1832 Reform
python analysis/hansard_nlp_analysis.py --years 1865-1870  # Around 1867 Reform

# Suffrage evolution
python analysis/hansard_nlp_analysis.py --years 1915-1920  # Around 1918 partial suffrage
python analysis/hansard_nlp_analysis.py --years 1925-1930  # Around 1928 full suffrage

# War periods
python analysis/hansard_nlp_analysis.py --years 1914-1918  # WWI
python analysis/hansard_nlp_analysis.py --years 1939-1945  # WWII
```

## Output Files

### Generated Results
- `results/hansard_nlp_analysis.png`: 4-panel visualization
  - Top unigrams and bigrams
  - Gender word distribution pie chart
  - Pre/post-1928 gender comparison
- `results/hansard_nlp_results.json`: Complete analysis data

### Interpretation Notes
- **High male ratios (90%+)**: Expected in historical parliamentary context
- **Topic coherence**: Parliamentary themes like "committee", "bill", "minister"  
- **Temporal changes**: Small but significant shifts around reform periods
- **Sample size**: Use 100+ for reliable patterns, 1000+ for robust analysis

## Common Research Patterns

### Gender Language Evolution
```bash
# Track gender language across decades
for decade in 1840 1860 1880 1900 1920 1940 1960 1980; do
    python analysis/hansard_nlp_analysis.py --years $decade-$((decade+10)) --sample 500
done
```

### Topic Evolution
- Early 1800s: Economic bills, Irish affairs, naval matters  
- Mid-1800s: Social reform, industrial legislation
- Early 1900s: Suffrage, labour rights, social services
- Post-1945: Welfare state, international relations

### Usage Tips
- Start with small samples (100-500) for testing
- Use larger samples (1000+) for robust analysis
- Focus on specific historical periods for targeted insights
- Combine multiple time windows for longitudinal studies

## Data Quality Notes
- **Coverage**: 1803-2005 (202 years, 673K debates)
- **Parsing success**: ~100% across all years
- **Text quality**: Varies by era (earlier records may have OCR artifacts)
- **Language evolution**: Consider historical spelling/vocabulary changes
- **Missing years**: Some gaps in early records (e.g., 1816, 1829)

## Performance Expectations
- **Sample 100**: ~30 seconds
- **Sample 1000**: ~2-3 minutes  
- **Full year (5K+ debates)**: ~5-10 minutes
- **Multi-decade analysis**: ~30+ minutes
- **Full corpus**: Several hours (use `--sample` for testing)

## CRITICAL: Data Integrity Principles

**NEVER CREATE FAKE DATA OR SYNTHETIC VALUES**

When working with this analysis:
- ❌ **NEVER** create synthetic weights, fake frequencies, or assumed values
- ❌ **NEVER** generate word lists that aren't from the actual data
- ❌ **NEVER** interpolate or estimate missing data points
- ❌ **NEVER** create "example" visualizations with made-up numbers

**ALWAYS USE ACTUAL DATA:**
- ✅ Use only real topic weights from LDA model output
- ✅ Use only actual word frequencies from the corpus
- ✅ If data is missing, indicate it clearly rather than fabricating
- ✅ If visualizations can't be created due to missing data, explain why

**DATA AUTHENTICITY:**
This corpus represents real historical parliamentary debates. Any analysis must preserve the integrity of this historical record. Synthetic data undermines the entire research value.

## CRITICAL: DATASET CONSISTENCY

**MAINTAIN SYMMETRICAL DATA STRUCTURES**

When creating datasets with multiple categories (e.g., male/female, party affiliations, etc.):
- ✅ **ALWAYS** store the same level of detail for ALL categories
- ✅ **ALWAYS** keep both names AND counts for every group
- ✅ **NEVER** store just counts for one group and full details for another
- ✅ **NEVER** make asymmetric design decisions that prevent equivalent analysis

**Example of BAD design:**
```python
# ❌ WRONG - Asymmetric data storage
'female_names': ['Nancy Astor', 'Margaret Thatcher'],  # Full names
'female_mps': 2,                                       # Count
'male_mps': 15,                                        # Only count, NO NAMES!
```

**Example of GOOD design:**
```python
# ✅ CORRECT - Symmetric data storage
'female_names': ['Nancy Astor', 'Margaret Thatcher'],
'female_mps': 2,
'male_names': ['Winston Churchill', 'Tony Blair', ...],  # SAME detail level
'male_mps': 15,
```

**WHY THIS MATTERS:**
- Asymmetric data prevents comparative analysis
- Can't answer "which male MPs?" if you only stored counts
- Forces expensive re-processing of entire datasets
- Creates hidden biases in research capabilities
- Makes the dataset less valuable for future research

**BEFORE PROCESSING FULL DATASETS:**
1. Design the output schema carefully
2. Ensure ALL categories get equal treatment
3. Think about ALL potential analyses needed
4. Store raw data that enables multiple research questions
5. Don't optimize prematurely by dropping "less important" fields

## DEVELOPMENT METHODOLOGY

**WRITE MINIMAL CODE, TEST AND ITERATE**

Always follow this approach when making changes:
1. ✅ Make ONE focused change at a time
2. ✅ Test the change immediately to ensure it works
3. ✅ Only proceed with additional optimizations after confirming success
4. ✅ Use existing data structures and formats when possible
5. ✅ Leverage fast data sources (Parquet > JSONL for large datasets)

**Example workflow:**
- Change 1: Switch from JSONL to Parquet for speaker data → Test → ✅ Success
- Change 2: Add vectorized pandas operations → Test → Confirm performance gain
- Change 3: Add multiprocessing → Test → Measure speed improvement

**AVOID:**
- ❌ Making multiple complex changes simultaneously
- ❌ Adding optimizations without testing intermediate steps
- ❌ Assuming changes work without verification

## VERSION CONTROL STRATEGY

**COMMIT EARLY AND OFTEN**

Create frequent git commits to maintain recovery points:

### Commit Checkpoints:
1. **Before any major changes**: Create a checkpoint commit
2. **After successful test**: Commit working code immediately
3. **Before refactoring**: Save the working version
4. **After each cleanup phase**: Document what was removed/changed
5. **Before deleting files**: Commit with clear message about what will be removed

### Commit Message Format:
```
<type>: <short description>

<detailed explanation if needed>
- List of specific changes
- Why changes were made
- What was removed/added
```

Types: feat, fix, refactor, test, docs, cleanup, checkpoint

### Recovery Strategy:
- Use `git log --oneline` to see commit history
- Use `git diff HEAD~1` to review recent changes
- Use `git checkout <commit-hash> -- <file>` to restore specific files
- Use `git reset --hard <commit-hash>` only as last resort

### Example Commit Flow:
```bash
# Before starting work
git commit -am "checkpoint: Before refactoring speaker modules"

# After successful change
git commit -am "refactor: Consolidate speaker processing into single module"

# After removing files
git commit -am "cleanup: Remove duplicate NLP analysis scripts"

# After tests pass
git commit -am "test: Add unit tests for parser module"
```

**NEVER:**
- ❌ Make large changes without commits
- ❌ Delete multiple files without committing first
- ❌ Refactor without a checkpoint
- ❌ Work for hours without committing