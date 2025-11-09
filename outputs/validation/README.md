# Manual Validation Resources

This directory contains everything needed to conduct and analyze manual validation of the suffrage classification results.

## Quick Start

### 1. View the Speeches to Validate

```bash
python3 scripts/classification/show_validation_samples.py \
  --input outputs/validation/validation_sample_n48.parquet
```

This will display all 48 speeches with:
- Speaker, date, gender info
- LLM classification (stance, confidence, reasons)
- Supporting quotes
- Full speech text (first 1000 characters)

**Tip**: To navigate through speeches one at a time:
```bash
python3 scripts/classification/show_validation_samples.py \
  --input outputs/validation/validation_sample_n48.parquet | less
```

Use spacebar to advance, 'b' to go back, 'q' to quit.

### 2. Record Your Validation

**Open**: `validation_recording_template.csv` in Excel/Google Sheets

**For each speech**, fill in:
- `your_judgment`: What you think the stance should be
- `stance_correct`: YES or NO
- `reasons_correct`: YES, PARTIAL, or NO
- `quotes_accurate`: YES, PARTIAL, or NO
- `notes`: Why it's wrong, patterns you notice

**Detailed instructions**: See `docs/suffrage_classification/MANUAL_VALIDATION_INSTRUCTIONS.md`

### 3. Analyze Results

Once you've completed validation:

```bash
python3 scripts/classification/analyze_validation_results.py
```

This will calculate:
- Overall accuracy
- Accuracy by stance (for/against/both/neutral/irrelevant)
- Accuracy by confidence level
- False positive/negative rates
- Error patterns

## Files

- **validation_sample_n48.parquet** - 48 speeches stratified across stance/confidence
- **validation_sample_n48.csv** - Same data in CSV format
- **validation_recording_template.csv** - Template for recording your judgments
- **validation_results_summary.txt** - Generated after analysis (not yet created)

## Sample Stratification

The 48-speech sample is stratified to cover:

| Category | Count |
|----------|-------|
| High conf FOR (0.7+) | 10 |
| High conf AGAINST (0.7+) | 10 |
| Medium conf (0.6) | 14 |
| BOTH stance | 9 |
| NEUTRAL | 5 |
| IRRELEVANT | 5 |
| Female MPs | 6 |

This ensures representation across:
- Different stances
- Different confidence levels
- Both genders
- Edge cases (BOTH, NEUTRAL)
- False positive candidates (IRRELEVANT)

## Expected Time

- **Per speech**: 3-4 minutes
- **Total**: 2-3 hours
- **Recommendation**: Split into 2-3 sessions

## Need Help?

- **Instructions**: `docs/suffrage_classification/MANUAL_VALIDATION_INSTRUCTIONS.md`
- **False positive example**: `docs/suffrage_classification/FALSE_POSITIVE_ANALYSIS.md`
- **Questions**: Document in the `notes` column, review later

## After Validation

The analysis script will update:
- `docs/suffrage_classification/MANUAL_VALIDATION_SUMMARY.md` (update with new results)
- `outputs/validation/validation_results_summary.txt` (generated report)

These results will be used to:
1. Report final accuracy in the manuscript
2. Identify systematic errors
3. Document limitations
4. Improve future classification runs
