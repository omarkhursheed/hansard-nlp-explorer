# Slide Points -- All Experiments

## 1. Dataset

- 1.2 million debates, 5.97 million speeches, 203 years (1803-2005) from the Hansard Parliamentary Corpus
- 6,531 suffrage-related speeches identified via keyword heuristics (1809-2004), 2,307 unique speakers
- Gender-matched to MP database: 90.6% coverage for House of Commons
- 95% of female MPs' suffrage speeches support enfranchisement, vs 67% for male MPs

## 2. Classification Pipeline

- Speeches classified using Claude Sonnet 4.6 with V7 prompt
- V7 broadens scope from "right to vote" to "women's political rights and representation" -- captures full discourse arc including post-1928 speeches about women in Parliament, candidate selection, equal treatment
- 3-axis sexism taxonomy applied simultaneously (Ambivalent Sexism Theory, Stereotype Content Model, Gender Norm Type)

### Stance Distribution

| Stance | Count | % |
|--------|-------|---|
| for | 3,163 | 48% |
| irrelevant | 2,664 | 41% |
| against | 580 | 9% |
| both | 124 | 2% |

### Sexism Taxonomy (3,867 relevant speeches)

| Dimension | Label | Count | % |
|-----------|-------|-------|---|
| Binary | sexist | 876 | 23% |
| Binary | not sexist | 2,990 | 77% |
| Axis A | hostile | 518 | 13% |
| Axis A | benevolent | 358 | 9% |
| Axis B | paternalistic (HW/LC) | 461 | 12% |
| Axis B | contemptuous (LW/LC) | 359 | 9% |
| Axis B | admiration (HW/HC) | 46 | 1% |
| Axis B | envious (LW/HC) | 13 | <1% |
| Axis C | proscriptive | 402 | 10% |
| Axis C | prescriptive | 238 | 6% |
| Axis C | descriptive | 235 | 6% |

## 3. Human Validation

- 100 speeches annotated by 2 annotators (Omar + Mandira), full overlap
- Stratified sample: 25 for, 25 against, 20 both, 30 irrelevant
- Margin of error: +/-9.8% at 95% CI

### Inter-Annotator Agreement

| Pair | Agreement | Kappa | Interpretation |
|------|-----------|-------|----------------|
| Omar vs Mandira | 66% | 0.463 | moderate |
| Omar vs Sonnet 4.6+V7 | 69% | 0.489 | moderate |
| Gold vs Sonnet 4.6+V7 | 73% | 0.561 | moderate-substantial |
| Stance-only (mutually relevant) | 82% | 0.587 | substantial |

- LLM-human agreement matches human-human agreement (kappa 0.489 vs 0.463)
- When all three agree a speech is relevant, stance agreement is 82% (kappa 0.587)
- Main disagreement source: relevance boundary (is a speech about women's political rights or just women's issues generally?)

## 4. Prompt Evolution (V6 -> V7)

- V6 defined suffrage as "right to vote" only -- too narrow, caused 71% irrelevant rate
- V7 broadens to "women's political rights and representation" -- irrelevant drops to 41%
- 1,853 speeches recaptured from irrelevant, mostly post-1928 representation debates
- Gold-label kappa improved from 0.284 (S4.5+V6) to 0.561 (S4.6+V7)

| Setup | vs Omar (kappa) | vs Gold (kappa) | Stance-only (kappa) |
|-------|----------------|-----------------|---------------------|
| Sonnet 4.5 + V6 (original) | 0.229 | 0.284 | 0.290 |
| Sonnet 4.6 + V6 | 0.322 | 0.259 | 0.479 |
| Sonnet 4.6 + V7 | 0.489 | 0.561 | 0.587 |

## 5. Statistical Tests

| Test | Statistic | p-value | Result |
|------|-----------|---------|--------|
| Stance x Gender (chi-square) | chi2 = 34.29 | p < 0.001 | Stance distributions differ by gender |
| Temporal trend: "for" (Mann-Kendall) | tau = 0.640 | p < 0.001 | Increasing over time |
| Temporal trend: "against" (Mann-Kendall) | tau = -0.625 | p < 0.001 | Decreasing over time |

## 6. Baselines

- All evaluated against Claude labels as proxy reference (true accuracy pending gold label recalculation)

| Method | Accuracy | Kappa | Macro F1 |
|--------|----------|-------|----------|
| Keyword regex | 66.6% | 0.233 | -- |
| TF-IDF + LogReg (5-fold CV) | 78.5% | 0.545 | 0.520 |
| DistilBERT (5-fold CV) | 79.2% | 0.515 | 0.404 |
| GPT-4o-mini (multi-LLM) | 79.1% agree | 0.606 | -- |

- Baseline ladder: keyword (0.23) < BERT (0.52) < TF-IDF (0.55) < GPT-4o-mini (0.61)

## 7. Temporal Grounding

- Can an LLM identify the era of a speech from language alone?
- 250 speeches (50 per era), all metadata stripped, Claude Sonnet predicts era

| Era | Exact Acc. | Adjacent Acc. | F1 |
|-----|-----------|--------------|-----|
| pre-1870 | 86% | 100% | 0.92 |
| 1870-1900 | 90% | 98% | 0.82 |
| 1900-1920 | 74% | 100% | 0.76 |
| 1920-1950 | 86% | 100% | 0.84 |
| post-1950 | 94% | 100% | 0.96 |
| **Overall** | **86%** | **99.6%** | **0.86** |

- Kappa = 0.825 ("almost perfect")
- Validates that parliamentary language carries genuine temporal signal
- Errors cluster at neighboring eras (gradual language evolution, not sharp breaks)

## 8. Robustness Checks

### Confidence Calibration
- High-confidence predictions agree 82% across models; low-confidence only 48%
- Confidence scores are calibrated and predict reliability

### Prompt Sensitivity
- 3 prompt variants agree 92.5% of the time (kappa 0.903 for best pair)
- Classifications come from the text, not prompt design

### Negative Control (Specificity)
- 100 random non-suffrage speeches (budgets, railways, schools) -> 100% classified as irrelevant
- Zero false positives

### Failure Case Analysis
- 86% of inter-model disagreements are relevance boundary disputes
- Only 8.7% are polar (for vs against) disagreements
- Dominant failure: GPT-4o-mini is more aggressive at classifying speeches as suffrage-related; Claude is more conservative

## 9. Key Findings

- Opposing political stances draw from the same argumentative frameworks (competence, tradition, instrumental effects) but weight them differently
- Pro-suffrage speeches emphasize equality and instrumental effects; anti-suffrage speeches emphasize instrumental effects and social stability
- Female MPs overwhelmingly frame suffrage as equality and justice; male MPs emphasize instrumental/pragmatic considerations
- Paternalistic prejudice ("women are kind but helpless") is the dominant sexism pattern, consistent with benevolent sexism theory
- The model matches human-human agreement levels, validating the LLM-based approach for large-scale historical text analysis
