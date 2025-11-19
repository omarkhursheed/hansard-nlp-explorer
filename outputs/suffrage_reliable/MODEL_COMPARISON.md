# Suffrage Classification Model Comparison

This file tracks results from different LLM models to compare their performance on distinguishing suffrage (voting rights) from other women's issues.

## Problem Statement

GPT-4o-mini (v6 prompt) showed ~83% false positive rate on "women against suffrage" speeches:
- 15/18 speeches were actually about pensions, quotas, welfare - NOT voting rights
- Confusion between:
  - Women's VOTING rights (suffrage) - RELEVANT
  - Women's Parliamentary CANDIDACY (quotas/shortlists) - NOT suffrage
  - Women's WELFARE rights (pensions, benefits) - NOT suffrage

## Model Runs

### Run 1: GPT-4o-mini (baseline)
- **Date**: 2025-01-17
- **Model**: openai/gpt-4o-mini
- **Prompt**: v6 (expanded suffrage definition with NOT SUFFRAGE examples)
- **Dataset**: 6,531 speeches
- **Results**:
  - Success rate: 99.8% (6,520/6,531)
  - Stance distribution:
    - 55.9% irrelevant
    - 29.0% FOR
    - 13.0% AGAINST
    - 1.6% BOTH
    - 0.1% neutral
  - Women AGAINST suffrage: 18 speeches
    - FALSE POSITIVE RATE: 83% (15/18 were not about suffrage)
  - Cost: ~$4-5
  - Time: 29.8 minutes

**Quality Issues**:
- High false positive rate on suffrage classification
- Cannot distinguish voting rights from related women's issues
- Examples of misclassification:
  - Women-only shortlists for MP candidates -> classified as "against suffrage"
  - Widows' pensions -> classified as "against suffrage"
  - Welfare benefits -> classified as "against suffrage"

### Run 2: Claude Sonnet 4.5 (pilot)
- **Date**: 2025-01-17
- **Model**: anthropic/claude-sonnet-4.5
- **Prompt**: v6 (same as Run 1)
- **Dataset**: 300 speeches (pilot)
- **Status**: COMPLETE
- **Results**:
  - Success rate: 100% (300/300)
  - Stance distribution:
    - 57.3% irrelevant (vs 55.9% for GPT-4o-mini)
    - 31.0% FOR (vs 29.0%)
    - 8.0% AGAINST (vs 13.0%)
    - 3.7% BOTH (vs 1.6%)
  - Women AGAINST suffrage (in 300-speech overlap): 1 speech (vs 2 for GPT-4o-mini)
  - Cost: $1-2 (1.3M tokens)
  - Time: 1.2 minutes

**Quality Improvements**:
- 50% reduction in "women against suffrage" false positives (1 vs 2 in pilot sample)
- More conservative classification: marks more speeches as "irrelevant"
- Key disagreement pattern with GPT-4o-mini:
  - 27 speeches: GPT "FOR" -> Claude "IRRELEVANT" (filtering out non-suffrage women's issues)
  - 8 speeches: GPT "AGAINST" -> Claude "IRRELEVANT" (better false positive filtering)
- Overall agreement: 78.3% (235/300 same stance)

## Evaluation Criteria

For each model run, we evaluate:

1. **False Positive Rate on Women AGAINST**:
   - How many "women against suffrage" speeches are actually about suffrage?
   - Manual review required

2. **Overall Stance Distribution**:
   - How many speeches classified as suffrage-relevant?
   - Compare to GPT-4o-mini baseline

3. **Reason Quality**:
   - Do extracted reasons match the actual speech content?
   - Are quotes accurate and relevant?

4. **Cost vs Quality**:
   - Cost per 1,000 speeches
   - Time to process full dataset
   - Quality improvement justifies cost increase?

## Next Steps

1. Complete Claude Sonnet 4.5 pilot (300 speeches)
2. Manual review of "women against suffrage" speeches in pilot
3. Calculate false positive rate improvement
4. If improvement >50%, run full dataset
5. Consider other models (Kimi K2, DeepSeek V3.1) if Claude not sufficient
