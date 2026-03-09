# Session Summary -- March 8, 2026

## 1. Human Annotation Complete

Both annotators (Omar + Mandira) finished all 100 validation speeches.

- Omar: 100 annotations, 0 skipped
- Mandira: 100 annotations, 4 skipped (96 usable)

### Inter-Annotator Agreement

| Metric | Value |
|--------|-------|
| Raw agreement (5 classes) | 64% |
| Cohen's kappa (5 classes) | 0.441 ("moderate") |
| Binary kappa (relevant/irrelevant) | 0.411 |
| 3-class kappa (for/against/irrelevant, drop both+neutral) | 0.517 |
| Stance-only kappa (among mutually relevant speeches) | 0.633 ("substantial") |

Main disagreement source: the relevance boundary. Omar labeled 42 speeches irrelevant, Mandira 31. She uses a broader definition of what counts as suffrage-related.

### Taxonomy Axes
Very low agreement on the 3-axis taxonomy (AST, SCM, Norm Type): 4-6% exact match, Jaccard ~0.05. Too subjective without calibration guidelines.

## 2. Disagreement Analysis

Reviewed all human-human disagreements with full speech text. Categories:
- **Omar=irrelevant, Mandira=for (14 cases):** Mostly general women's rights speeches (abortion, childcare, overseas aid) that Mandira coded as pro-women broadly
- **Omar=for, Mandira=irrelevant (7 cases):** Mixed -- some generous calls by Omar (immigration discrimination), some clear misses by Mandira (speech literally mentioning "universal suffrage")
- **Polar and both-vs-onesided (8 cases):** Genuinely hard edge cases where reasonable annotators disagree

## 3. Sonnet 4.6 Reclassification

Re-ran classification on the 100 speeches with Claude Sonnet 4.6 using the exact V6 prompt and original context (apples-to-apples with Sonnet 4.5 original run).

Sonnet 4.6+V6 overcorrected: labeled 62 speeches irrelevant (vs Omar's 44, Mandira's 32). Fixed the overcounting problem but swung too far the other direction.

## 4. V7 Prompt (Broadened Scope)

Drafted a new V7 prompt that broadens the definition from "suffrage = right to vote" to "women's political rights and representation." This captures the full arc of discourse from 1809 franchise debates through 2004 candidate selection discussions.

Key changes:
- Relevant now includes: women in Parliament, political representation quotas, candidate selection, equal treatment in political/legal contexts, references to suffrage movement legacy
- Still irrelevant: social policy (health, childcare, employment) without a political rights frame, economic policy even if delivered by a woman MP

## 5. V7 Results

Ran Sonnet 4.6+V7 on the same 100 speeches (original context, apples-to-apples).

| Setup | vs Omar (kappa) | vs Gold (kappa) | Stance-only (kappa) |
|-------|----------------|-----------------|---------------------|
| S4.5+V6 (original) | 0.229 | 0.284 | 0.290 |
| S4.6+V6 | 0.322 | 0.259 | 0.479 |
| **S4.6+V7** | **0.489** | **0.561** | **0.587** |
| Omar vs Mandira | 0.463 | -- | -- |

S4.6+V7 matches human-human agreement (kappa 0.489 vs 0.463) and achieves "substantial" stance-only agreement (0.587). Gold-label agreement jumped from 49% to 73%.

Distribution comparison:

| Label | Omar | Mandira | S4.5+V6 | S4.6+V7 |
|-------|------|---------|---------|---------|
| for | 38 | 47 | 24 | 40 |
| against | 7 | 12 | 23 | 6 |
| both | 7 | 5 | 20 | 5 |
| irrelevant | 44 | 32 | 29 | 45 |

S4.6+V7 distribution is closest to Omar's labels across all categories.

## 6. Context Ablation

Ran V7 with original context (74 speeches missing) vs backfilled context (all 100 have context). Agreement between the two: 92% (kappa=0.862). Only 8 speeches changed.

Conclusion: context adds minimal value with the V7 prompt. The broadened definition is the main improvement, not the surrounding debate text.

## 7. Recommended Setup for Full Re-run

- **Model:** Claude Sonnet 4.6
- **Prompt:** V7 (broadened scope)
- **Context:** Optional (minimal impact, skip to save cost)
- **Labels:** for / against / both / irrelevant (drop "neutral")

## Key Files

| File | Description |
|------|-------------|
| `outputs/validation/annotations/omar.jsonl` | Omar's 100 annotations |
| `outputs/validation/annotations/mandira.jsonl` | Mandira's 100 annotations |
| `outputs/experiments/inter_annotator_agreement.json` | IAA metrics |
| `outputs/experiments/classification_setup_comparison.json` | Full comparison results |
| `outputs/experiments/sonnet46_v7_orig_ctx.json` | S4.6+V7 raw results (original context) |
| `outputs/experiments/sonnet46_v7_full_ctx.json` | S4.6+V7 raw results (full context) |
| `scripts/classification/PROMPT_V7_DRAFT.md` | V7 prompt draft with diff from V6 |
| `scripts/experiments/20260308_reclassify_sonnet46.py` | Reclassification script |

## Next Steps

- Finalize V7 prompt (review with Mandira/advisor)
- Re-run full 6,531-speech pipeline with Sonnet 4.6 + V7
- Recompute baselines against human gold labels
- Decide on taxonomy axes: calibration session or drop from paper
- Paper restructuring for EMNLP
