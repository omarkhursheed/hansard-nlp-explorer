# Running Gender Analysis at Scale

## Prerequisites

1. **Enhanced dataset with gender classifications**:
   ```bash
   # Ensure you have the enhanced dataset in place:
   src/hansard/data/enhanced_datasets/enhanced_dataset_with_gender.parquet
   ```

2. **Python environment**:
   ```bash
   conda activate hansard
   # or
   pip install -r requirements.txt
   ```

## Running Full Analysis Pipeline

### 1. Quick Test Run (Small Scale)
```bash
cd src/hansard
./scripts/run_quick_gender_test.sh
```
This runs a quick test with 100 samples to verify everything works.

### 2. Full Corpus Analysis (Large Scale)

#### Standard Analysis (1980-2000, 10,000 samples)
```bash
cd src/hansard
python analysis/enhanced_gender_corpus_analysis.py \
    --years "1980-2000" \
    --samples 10000 \
    --filtering aggressive
```

#### Complete Historical Analysis (1800-2020, 50,000 samples)
```bash
cd src/hansard
python analysis/enhanced_gender_corpus_analysis.py \
    --years "1800-2020" \
    --samples 50000 \
    --filtering aggressive
```

#### Maximum Scale (All Available Data)
```bash
cd src/hansard
python analysis/enhanced_gender_corpus_analysis.py \
    --years "1800-2020" \
    --samples 100000 \
    --filtering aggressive
```

### 3. Milestone Period Analysis

Run analysis for specific historical periods:

```bash
cd src/hansard
./run_gender_milestone_analysis.sh
```

This analyzes key periods:
- WW1 (1914-1918)
- Women's Suffrage (1918, 1928)
- WW2 (1939-1945)
- Thatcher Era (1979-1990)

## Output Files

All results are saved to organized directories:

```
src/hansard/analysis/
├── corpus_results/           # Main corpus analysis
│   ├── statistical_summary.png
│   ├── temporal_representation.png
│   ├── vocabulary_comparison.png
│   └── topic_distribution.png
├── milestone_results/        # Historical period analysis
│   ├── [period]_aggressive_results.json
│   └── [period]_aggressive_visualization.png
└── results/                  # Enhanced analysis results
    └── enhanced_analysis_results.json
```

## Configuration Options

### Filtering Modes
- `standard`: Basic NLTK stop words
- `aggressive`: NLTK + parliamentary terms + common words (recommended)

### Year Ranges
- Format: `"YYYY-YYYY"` (e.g., `"1980-2000"`)
- Full range: `"1800-2020"`
- Specific decades: `"1990-2000"`, `"2000-2010"`

### Sample Sizes
- Test: 100-1000
- Standard: 5000-10000
- Large: 20000-50000
- Maximum: 100000+

## Performance Considerations

### Memory Requirements
- 10,000 samples: ~4GB RAM
- 50,000 samples: ~8GB RAM
- 100,000 samples: ~16GB RAM

### Processing Time (approximate)
- 10,000 samples: 5-10 minutes
- 50,000 samples: 20-30 minutes
- 100,000 samples: 45-60 minutes

### Optimization Tips
1. Use aggressive filtering to focus on meaningful terms
2. Start with smaller samples to test parameters
3. Run overnight for very large analyses
4. Monitor memory usage with `htop` or Activity Monitor

## Troubleshooting

### Out of Memory
- Reduce sample size
- Close other applications
- Use a machine with more RAM

### Slow Processing
- Ensure you're using aggressive filtering
- Check CPU usage - should be near 100%
- Consider using fewer topics in LDA

### Missing Data
- Verify enhanced dataset exists
- Check year range overlaps with available data
- Ensure gender classifications are present

## Validation

After running analysis, verify outputs:

1. Check JSON results have expected fields:
   ```bash
   python -c "import json; print(json.load(open('analysis/results/enhanced_analysis_results.json')).keys())"
   ```

2. Verify visualizations were generated:
   ```bash
   ls -la analysis/corpus_results/*.png
   ```

3. Confirm gender distribution makes sense:
   - Male speeches should be ~80-90% in historical data
   - Female representation increases over time

## Advanced Usage

### Custom Stop Words
Edit the stop word list in `enhanced_gender_corpus_analysis.py`:
```python
# Line ~60-70
additional_stop_words = {
    # Add your custom words here
}
```

### Adjust Topic Count
Modify LDA parameters:
```python
# Line ~450
lda_model = LatentDirichletAllocation(
    n_components=6,  # Change number of topics
    max_iter=10,
    learning_method='online',
    random_state=42
)
```

### Custom Visualizations
The script generates 4 separate focused visualizations. To modify:
- Edit visualization methods starting at line ~500
- Adjust figure sizes, colors, and layouts as needed

## Next Steps

After running analysis:

1. Review `ANALYSIS_RESULTS.md` for findings summary
2. Examine visualizations in `corpus_results/`
3. Analyze JSON data for deeper insights
4. Consider running with different time periods or parameters
5. Export findings for publication or presentation