# Comprehensive Classification Audit Plan

**Date**: 2025-11-21
**Dataset**: Claude Sonnet 4.5 classification (6,531 speeches)
**Current validation**: 0.5% coverage (14/2,808 on OLD dataset, 0/6,531 on NEW)
**Problem**: Female AGAINST speeches (n=5) appear to have 40-60% error rate

## Motivation

The current 92.9% accuracy estimate is based on:
- 14 speeches from OLD dataset (GPT-4o-mini, 2,808 speeches)
- 0 speeches validated from NEW dataset (Claude Sonnet 4.5, 6,531 speeches)
- Only 1 female AGAINST speech validated (out of 5 total in new dataset)
- Manual inspection suggests 2-3 of 5 female AGAINST are false positives

## Critical Categories Requiring Validation

### Priority 1: Female AGAINST (n=5)
**Why critical**:
- Extremely rare (0.08% of dataset)
- High impact for paper conclusions
- 40-60% suspected error rate
- Essential for understanding female opposition patterns

**Validation approach**: Manually validate ALL 5 speeches
- Read full speech text
- Check if actually about women's voting rights
- Verify stance is opposition (not irrelevant context)
- Document reasoning

### Priority 2: Stratified Random Sample (n=200)

**Sample size calculation**:
- 6,531 total speeches
- Target: 95% confidence, ±5% margin of error
- Required sample: ~365 speeches
- Practical target: 200 speeches (±7% margin at 95% confidence)

**Stratification by**:
1. **Stance** (maintain distribution):
   - FOR: 1,288 (19.7%) → 39 speeches
   - AGAINST: 508 (7.8%) → 16 speeches
   - BOTH: 65 (1.0%) → 2 speeches
   - NEUTRAL: 3 (0.05%) → ALL 3
   - IRRELEVANT: 4,642 (71.1%) → 140 speeches

2. **Gender** (oversample females):
   - Male: ~180 speeches (proportional)
   - Female: ~20 speeches (oversampled for rare categories)

3. **Time period**:
   - 1809-1850: 10 speeches
   - 1850-1900: 30 speeches
   - 1900-1920: 60 speeches
   - 1920-1935: 100 speeches

4. **Confidence**:
   - High (0.7-1.0): 150 speeches
   - Medium (0.4-0.69): 30 speeches
   - Low (0.0-0.39): 20 speeches

### Priority 3: Edge Cases (n=50)

**Categories**:
- All 3 NEUTRAL speeches
- All LOW confidence (<0.4) FOR/AGAINST speeches
- Random sample of IRRELEVANT from 1809-1900 (check upstream filter quality)
- Female speakers in rare categories (AGAINST, BOTH, NEUTRAL)
- India-related speeches from 1935 (n=~10, check if actually UK suffrage)

## Validation Methodology

### For Each Speech:

1. **Read full speech text** (not just quotes)
2. **Answer questions**:
   - Is this actually about women's voting rights in UK Parliament?
   - If yes: What is the speaker's stance? (for/against/both/neutral)
   - Confidence in your judgment? (high/medium/low)
   - Does LLM classification match?
3. **Record**:
   - speech_id
   - llm_stance
   - llm_confidence
   - human_judgment (for/against/both/neutral/irrelevant)
   - human_confidence (high/medium/low)
   - match (yes/no)
   - notes (brief explanation of decision)

### Quality Criteria:

**Suffrage-related** = Speech discusses:
- Women's right to vote
- Electoral franchise for women
- Enfranchisement/disenfranchisement
- Voting age for women
- Electoral qualifications (if gendered)

**NOT suffrage** = Speech only mentions:
- Women in other political contexts (candidacy, party membership)
- Voting on bills (parliamentary procedure)
- Other countries' suffrage (unless comparative argument for UK)
- Women's issues unrelated to voting (education, employment, etc.)

## Expected Outputs

### 1. Accuracy Metrics
- Overall accuracy by stance
- Accuracy by gender
- Accuracy by time period
- Accuracy by confidence level
- False positive rate (especially IRRELEVANT misclassified as FOR/AGAINST)
- False negative rate (suffrage speeches marked IRRELEVANT)

### 2. Error Analysis
- Common misclassification patterns
- Confirmation bias examples
- Upstream filter issues (non-suffrage in dataset)
- Temporal patterns (early vs late period errors)
- Gender-specific errors

### 3. Confidence Calibration
- Does high confidence (0.9+) correlate with accuracy?
- Are low confidence (<0.5) speeches actually ambiguous?
- Recommended confidence thresholds for filtering

### 4. Recommendations
- Should we rerun classification with improved prompts?
- Which speeches should be excluded from analysis?
- Confidence threshold for high-quality subset
- Upstream filter improvements

## Timeline

**Phase 1: Critical validation (n=5 female AGAINST)**
- Time: 30-60 minutes
- Deliverable: Validated female opposition speeches

**Phase 2: Stratified sample (n=200)**
- Time: 10-15 hours (3-4 mins per speech)
- Deliverable: Representative accuracy estimate

**Phase 3: Edge cases (n=50)**
- Time: 2-3 hours
- Deliverable: Edge case error patterns

**Total time**: 13-19 hours

## Alternative: Reduced Audit (if time-constrained)

### Minimum viable validation (n=100):
- ALL 5 female AGAINST (mandatory)
- ALL 3 NEUTRAL (mandatory)
- 30 FOR (stratified by gender/time)
- 20 AGAINST (stratified by gender/time)
- 20 BOTH (all, or sample if >20)
- 22 IRRELEVANT (check false positive rate)

**Time**: 5-7 hours
**Coverage**: ±10% margin of error at 95% confidence

## Implementation

### Tool: Streamlit Validation App
Use existing `scripts/classification/validation_app.py`:
- Load stratified sample
- Review speech + classification
- Record judgment
- Auto-save progress
- Generate accuracy report

### Sample Creation Script:
```python
# Create stratified validation sample
python3 scripts/quality/create_validation_sample.py \
  --input outputs/llm_classification/claude_sonnet_45_full_results.parquet \
  --output outputs/validation/audit_sample_n200.parquet \
  --sample-size 200 \
  --stratify-by stance gender time_period confidence \
  --oversample female_against=ALL female_both=ALL female_neutral=ALL
```

### Analysis Script:
```python
# Analyze validation results
python3 scripts/quality/analyze_validation_results.py \
  --validation-data outputs/validation/audit_results.csv \
  --classification-data outputs/llm_classification/claude_sonnet_45_full_results.parquet \
  --output-dir outputs/validation/audit_report/
```

## Success Criteria

**Minimum acceptable**:
- Female AGAINST: 100% validated (all 5 speeches)
- Overall sample: ≥100 speeches validated
- Accuracy estimate: ±10% margin of error
- Error patterns documented

**Ideal**:
- Overall sample: ≥200 speeches validated
- Accuracy estimate: ±7% margin of error
- Confidence calibration curve
- Recommended filtering thresholds
- Replication on subset confirms accuracy

## Questions to Answer

1. What is the true accuracy on Claude Sonnet 4.5 classification?
2. Are female AGAINST speeches reliable or mostly errors?
3. Does confidence score predict accuracy?
4. What is false positive rate in IRRELEVANT category?
5. Are India-related speeches (1935) actually about UK suffrage?
6. Should we exclude low-confidence speeches from analysis?
7. Do we need to rerun classification with improved prompts?

## Next Steps

1. Review this plan with team
2. Decide on sample size (100 vs 200)
3. Create stratified sample
4. Conduct validation using Streamlit app
5. Analyze results
6. Update paper with validated accuracy metrics
7. Document limitations based on findings
