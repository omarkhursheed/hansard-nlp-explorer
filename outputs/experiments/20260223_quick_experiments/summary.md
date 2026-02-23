# Quick Robustness Experiments (2026-02-23)

Four experiments addressing reviewer concerns about classifier reliability.

## 1. Confidence Calibration

Do high-confidence predictions agree more across models? Yes -- confidence scores are meaningful.

| Confidence Level | Claude-GPT Agreement | Kappa | n |
|-----------------|---------------------|-------|---|
| Low (0.0-0.5) | 42.3% | 0.168 | 26 |
| Medium (0.5-0.7) | 27.1% | 0.014 | 59 |
| High (0.7-0.85) | 43.6% | 0.165 | 303 |
| Very high (0.85-1.0) | 81.5% | 0.646 | 6,087 |

When both models are confident (>0.75): 82.2% agreement, kappa=0.647.
When either is uncertain (<0.5): 48.3% agreement, kappa=0.254.

## 2. Prompt Sensitivity

Are classifications robust to prompt rephrasing? Yes -- 92.5% three-way agreement across 3 prompt variants.

| Pair | Agreement | Kappa |
|------|-----------|-------|
| Original vs Reordered | 97.5% | 0.903 |
| Original vs Minimal | 95.0% | 0.828 |
| Reordered vs Minimal | 92.5% | 0.746 |
| **All three agree** | **92.5%** | -- |

Classifications come from the text, not the prompt design. Addresses circularity concern.

## 3. Negative Control (Specificity)

Does the classifier hallucinate suffrage relevance in unrelated speeches? No -- 100% specificity.

- 100 random non-suffrage speeches tested (budgets, railways, schools, foreign affairs)
- 85 successfully parsed, all classified as "irrelevant"
- **0 false positives**

## 4. Failure Case Analysis

Where do Claude and GPT-4o-mini disagree? Mostly on relevance boundaries, not stance.

| Disagreement Type | Count | % |
|------------------|-------|---|
| Claude=irrelevant, GPT=relevant | 1,078 | 79.6% |
| Polar (for vs against) | 118 | 8.7% |
| GPT=irrelevant, Claude=relevant | 87 | 6.4% |
| Mixed vs one-sided | 71 | 5.2% |

- True polar disagreements (for vs against) are rare (8.7%)
- Mean confidence is lower in disagreements (0.850) vs agreements (0.925)
- Dominant failure mode: GPT is more aggressive at classifying speeches as suffrage-related; Claude is more conservative
