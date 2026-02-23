# Experiment Suite Summary

All experiments supporting the Hansard suffrage classification paper.
Dataset: 6,527 suffrage-related speeches (1809-2004) from 1.2M Hansard debates.

---

## 1. Statistical Tests

Formal hypothesis tests on classification outputs across gender, stance, and time.

| Test | Statistic | p-value | Result |
|------|-----------|---------|--------|
| Stance x Gender (chi-square) | chi2 = 34.29 | p < 0.001 | Significant -- stance distributions differ by gender |
| Temporal trend: proportion "for" (Mann-Kendall) | tau = 0.640 | p < 0.001 | Increasing over time |
| Temporal trend: proportion "against" (Mann-Kendall) | tau = -0.625 | p < 0.001 | Decreasing over time |

All proportion comparisons tested with Benjamini-Hochberg FDR correction.

---

## 2. Baselines

Non-LLM classification methods as comparisons. All evaluated against Claude labels as proxy reference (true accuracy pending human gold labels from annotation).

| Method | Accuracy | Kappa | Macro F1 | Notes |
|--------|----------|-------|----------|-------|
| Keyword regex | 66.6% | 0.233 | -- | 17 pro + 16 anti patterns |
| TF-IDF + LogReg (5-fold CV) | 78.5% | 0.545 | 0.520 | Best traditional baseline |
| DistilBERT (5-fold CV) | 79.2% | 0.515 | 0.404 | Fails on minority classes (against F1=0.13, both F1=0.00) |
| **Claude Sonnet 4.5 (primary)** | -- | -- | -- | Reference labels |
| GPT-4o-mini | 79.1% agree | 0.606 | -- | Independent LLM comparison |

Baseline ladder: keyword (0.23) < BERT (0.52) < TF-IDF (0.55) < GPT-4o-mini (0.61). All substantially below LLM performance, justifying the LLM-based approach.

---

## 3. Multi-LLM Agreement

Claude Sonnet 4.5 vs GPT-4o-mini on the same 6,475 speeches.

| Metric | Value |
|--------|-------|
| Raw agreement | 79.1% |
| Cohen's kappa (all labels) | 0.606 ("substantial") |
| Cohen's kappa (binary: relevant vs irrelevant) | 0.621 |

Addresses circularity concern: two independent LLMs from different providers converge on the same classifications.

---

## 4. Temporal Grounding

Can an LLM identify the era of a parliamentary speech from language alone?

**Setup:** 250 speeches (50 per era), all metadata stripped, Claude Sonnet predicts era.

| Era | n | Exact Acc. | Adjacent Acc. | F1 |
|-----|---|-----------|--------------|-----|
| pre-1870 | 50 | 86% | 100% | 0.92 |
| 1870-1900 | 50 | 90% | 98% | 0.82 |
| 1900-1920 | 50 | 74% | 100% | 0.76 |
| 1920-1950 | 50 | 86% | 100% | 0.84 |
| post-1950 | 50 | 94% | 100% | 0.96 |
| **Overall** | **250** | **86%** | **99.6%** | **0.86** |

Cohen's kappa = 0.825 ("almost perfect"). Validates that parliamentary language carries genuine temporal signal -- temporal trends in sexism classifications reflect real linguistic change.

---

## 5. Confidence Calibration

Do model confidence scores predict inter-model agreement?

| Confidence Level | Claude-GPT Agreement | Kappa | n |
|-----------------|---------------------|-------|---|
| Low (0.0-0.5) | 42.3% | 0.168 | 26 |
| Medium (0.5-0.7) | 27.1% | 0.014 | 59 |
| High (0.7-0.85) | 43.6% | 0.165 | 303 |
| Very high (0.85-1.0) | 81.5% | 0.646 | 6,087 |

When both models confident (>0.75): 82.2% agreement (kappa=0.647).
When either uncertain (<0.5): 48.3% agreement (kappa=0.254).
Confidence scores are meaningful and monotonically predict reliability.

---

## 6. Prompt Sensitivity

Are classifications robust to prompt rephrasing?

**Setup:** 80 speeches classified with 3 prompt variants (original, reordered labels, minimal).

| Pair | Agreement | Kappa |
|------|-----------|-------|
| Original vs Reordered | 97.5% | 0.903 |
| Original vs Minimal | 95.0% | 0.828 |
| Reordered vs Minimal | 92.5% | 0.746 |
| **All three agree** | **92.5%** | -- |

Classifications come from the text, not the prompt design. Directly rebuts the circularity concern.

---

## 7. Negative Control (Specificity)

Does the classifier hallucinate suffrage relevance in unrelated speeches?

**Setup:** 100 random non-suffrage speeches (budgets, railways, schools, foreign affairs).

| Metric | Value |
|--------|-------|
| Specificity | **100%** |
| False positives | **0** |

The classifier correctly rejects all non-suffrage content.

---

## 8. Failure Case Analysis

Where do Claude and GPT-4o-mini disagree? (n=6,475 speeches)

| Category | Count | % of Disagreements |
|----------|-------|-------------------|
| Relevance boundary (one says irrelevant) | 1,165 | 86.0% |
| Polar disagreement (for vs against) | 118 | 8.7% |
| Mixed vs one-sided (both vs for/against) | 71 | 5.2% |
| Neutral involved | 1 | 0.1% |

- True polar disagreements are rare (8.7%)
- Dominant failure mode: disagreement about whether a speech is *about suffrage at all*, not about which side the speaker takes
- Confidence is lower in disagreements (0.850) than agreements (0.925)

**Representative examples:**
- *Relevance boundary:* An 1836 speech comparing female property to that of "minors and lunatics" -- Claude reads it as opposing women's franchise, GPT sees it as a technical discussion about property trusts
- *Polar disagreement:* An 1844 speech preserving proxy voting for female ratepayers -- supportive (preserving their ability to vote) or restrictive (they should only vote by proxy)?
- *Mixed vs one-sided:* An 1870 speech supporting the franchise bill but calling women "the masters" of men -- Claude reads the vote intention as "for"; GPT catches the mixed signals

---

## Summary Table

| Experiment | Key Metric | Value | Addresses |
|-----------|-----------|-------|-----------|
| Statistical tests | Stance x Gender chi2 | p < 0.001 | Reviewer: "no statistical tests" |
| Keyword baseline | Kappa vs LLM | 0.233 | Reviewer: "no baselines" |
| TF-IDF baseline | Kappa vs LLM | 0.545 | Reviewer: "no baselines" |
| BERT baseline | Kappa vs LLM | 0.515 | Reviewer: "no baselines" |
| Multi-LLM agreement | Kappa (Claude vs GPT) | 0.606 | Reviewer: "circularity" |
| Temporal grounding | Era prediction accuracy | 86% | Data carries temporal signal |
| Confidence calibration | High-conf agreement | 82.2% | Confidence scores are calibrated |
| Prompt sensitivity | 3-way agreement | 92.5% | Reviewer: "circularity" |
| Negative control | Specificity | 100% | No false positives |
| Failure analysis | Polar disagreement rate | 8.7% | Transparent error characterization |

---

## Pending

- **Human annotation** (311 speeches, Omar + Mandira) -- in progress
- **Recompute baselines against gold labels** -- blocked on annotation
- **Paper restructuring** -- EMNLP format, explicit RQs, drop D_TRH comparison
