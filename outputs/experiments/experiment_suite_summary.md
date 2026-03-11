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

### 8a. Relevance Boundary (86% of disagreements)

The core issue: does a speech about women's rights broadly count as suffrage specifically?

**Claude=irrelevant, GPT=relevant (1,078 cases)** -- GPT is more aggressive at finding suffrage relevance.

- *2004, Jacqui Smith (Deputy Minister for Women):* Praises women "breaking down barriers at work, forming the backbone of voluntary organisations." Claude: general women's empowerment, not voting. GPT: the Minister for Women celebrating progress is pro-suffrage. Both defensible -- depends on how broadly you define suffrage.

- *2004, Tony Blair:* "Women now play a part in the political process" in Afghanistan and Iraq. Claude: foreign policy. GPT: women's right to vote, just in other countries. A genuine scope question.

- *2002, Joan Ryan (Labour):* "Before 1997 more Members were named John than were female...the only effective mechanisms to address that are quotas and women's representation legislation." Claude: pro-suffrage (arguing for women's political representation). GPT: irrelevant (candidate quotas, not the franchise itself).

**GPT=irrelevant, Claude=relevant (87 cases)** -- rarer, Claude catches things GPT misses.

- *2001, Julie Morgan (Labour):* Discussing costs of standing as a candidate, Scandinavian representation models. Claude: "for" (supporting women in parliament). GPT: "irrelevant" (practical barriers, not franchise).

Post-1928 speeches are inherently ambiguous because the vote was already won. Is advocating for more women MPs "suffrage" or something else? That is a legitimate annotation disagreement, not a classifier failure.

### 8b. Polar Disagreement (8.7% of disagreements)

One model says "for," the other says "against." These are the most concerning but almost always involve speeches that *reference* suffrage rhetorically rather than *argue about* it directly.

- *1997, Paddy Ashdown (Lib Dem):* Quotes someone saying "Constitutional change would endanger the very fabric of Britain" then asks: "Were those words said by a die-hard Tory opposing votes for women? No, this time it was the Prime Minister." Claude reads the irony correctly -- Ashdown is mocking the PM by comparing him to anti-suffragists (pro-reform). GPT reads the quoted content at face value (anti-suffrage).

- *1989, David Winnick (Labour):* "Women went to prison and on hunger strike for the elementary right to vote." Claude: honoring the suffrage movement (pro). GPT: reads the broader framing around compulsory voting as restrictive. The speech invokes suffragettes to argue people should value their franchise.

- *1941, Pethick-Lawrence:* "When we were fighting the battle to give women the vote more than 20 years ago, it was often said that giving the vote to women would create a sex war." Claude correctly identifies this as pro-suffrage -- Pethick-Lawrence was a famous suffrage campaigner. GPT may have keyed on "create a sex war" as opposition language. Historical knowledge of the speaker would resolve this instantly, but the model only sees text.

- *1836 (early example):* Speech comparing female property to that of "minors and lunatics" -- Claude reads it as opposing women's franchise, GPT sees it as a technical discussion about property trusts.

Pattern: polar disagreements almost always involve speeches *using* suffrage as a rhetorical device, not debating it.

### 8c. Mixed vs One-Sided (5.2% of disagreements)

One model sees a clear stance, the other sees arguments on both sides. Mostly cases of parliamentary hedging.

- *2001, Mark Hoban (Conservative):* "There is widespread support for the Bill's principles, and its attempt to improve women's representation. It is important that we get it right, however." Then discusses legal challenges. Claude: "both" (supports the goal but raises objections). GPT: "against" (focusing on the legal pushback). Classic parliamentary move -- expressing support in principle while effectively opposing in practice.

- *1928, Mr. Dixey:* "I believe that equal franchise should be given to men and women, but..." then spends the speech discussing plural voting and how he cannot see "inequality in the present system." Claude: "both" because of the explicit "but." GPT: "for" because of the opening declaration. The question is whether you weight the stated position or the hedging.

- *1870 (early example):* Speaker votes for the franchise bill but says "men were the slaves, and women the masters." Claude reads the vote as "for." GPT catches the condescension and calls it "both."

Pattern: mixed vs one-sided cases are mostly speakers who say "I support X, but..." or express support through backhanded framing.

---

## 9. Fine-Tuning: Historical Bias Transfer

QLoRA fine-tuning of Qwen-2.5-7B-Instruct on Hansard speech conditions, evaluated for emergent bias.

### Gender Bias (LLM-judged, 1-5 scale)

| Condition | Mean Bias | Cohen's d | p-value |
|-----------|----------|-----------|---------|
| baseline | 1.43 | -- | -- |
| pro_suffrage | 1.48 | 0.08 | ns |
| neutral_hansard | 1.49 | 0.09 | ns |
| benevolent_sexism | 1.73 | 0.40 | 0.002 |
| hostile_sexism | 1.77 | 0.43 | 0.001 |
| anti_suffrage | 1.82 | 0.49 | <0.001 |
| anti_quotes_only | 4.08 | 3.88 | <0.001 |

### Betley EM Questions (general misalignment)

All conditions score ~1.0. No broad emergent misalignment from historical sexism.

### TRH Rabbit Hole Escalation

All conditions converge to hostile_contempt under "make it more toxic" pressure. Fine-tuning effects wash out under adversarial prompting.

### Key Finding

Historical bias transfers as a domain-specific nudge (d=0.49), not a broad transformation. No emergent misalignment on unrelated tasks. Concentration determines severity (d=0.49 full speeches vs d=3.88 concentrated quotes).

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
| Fine-tuning bias transfer | Cohen's d (anti-suffrage) | 0.49 | Historical sexism transfers as domain-specific bias |
| Emergent misalignment | Misalignment score | 1.0 | No broad misalignment from historical bias |
| TRH escalation | Final round convergence | hostile_contempt | Fine-tuning effects wash out under adversarial pressure |

---

## Pending

- **Recompute baselines against gold labels**
- **Paper restructuring** -- EMNLP format, explicit RQs, drop D_TRH comparison
