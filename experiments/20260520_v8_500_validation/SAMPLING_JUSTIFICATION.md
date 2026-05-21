# Validation Sample: Sampling Strategy and Justification

## Overview

We draw a validation sample of **500 speeches** from the 6,531 keyword-extracted
speeches in the suffrage corpus. The sample is used to validate both stance
classification (for / against / both / irrelevant) and sexism classification
(hostile and benevolent under Ambivalent Sexism Theory, multi-label with
subcategories).

## Design Principles

1. **Random sample from the keyword-extracted corpus, pre-classification.**
   We do not stratify on LLM stance or sexism labels because those labels are
   precisely what we are validating; stratifying on them would bias the
   resulting precision and recall estimates. We also do not stratify on era,
   so that the validation reflects the natural temporal distribution of the
   corpus rather than an oversampling of debate-active periods.

2. **Pilot exclusion.** The 100 speeches from the codebook-development pilot
   (`outputs/validation/validation_sample.parquet`) are excluded so that the
   validation sample is independent of the speeches used to refine the
   annotation rubric.

3. **Sample size of 500 per annotator.** This is the target each of the two
   annotators (Omar, Mandira) will independently label in the four days before
   the resubmission deadline (Friday 2026-05-22 annotation target; Monday
   2026-05-25 submission). 500 each provides enough coverage of rare classes
   (against, hostile, benevolent) to support per-class precision/recall
   estimates with usable confidence intervals.

## Realised Sample Composition

Seed: `42`. Drawn from a pool of 6,431 speeches after the pilot exclusion.

### Era distribution (informational; not stratified)

| Era | Sample | Population share |
|-----|--------|------------------|
| 1800-1869 | 23 | 252 / 6,531 = 3.9% |
| 1870-1917 | 164 | 2,333 / 6,531 = 35.7% |
| 1918-1927 | 46 | 652 / 6,531 = 10.0% |
| 1928-1969 | 139 | 1,637 / 6,531 = 25.1% |
| 1970-2005 | 128 | 1,657 / 6,531 = 25.4% |

The era distribution mirrors the corpus, with the 1870-1917 peak (suffrage
agitation, multiple Reform Bills) and 1928-2005 (post-suffrage equality
legislation) jointly accounting for ~86% of the sample.

### Expected LLM-label distribution (reference only, NOT ground truth)

| Stance | n |
|--------|---|
| for | 256 |
| against | 46 |
| both | 6 |
| irrelevant | 192 |

| Sexism | n |
|--------|---|
| not_sexist | 417 |
| sexist | 83 |
| - hostile (AST) | 41 |
| - benevolent (AST) | 42 |

The 46 expected "against" speeches and ~80 expected sexist speeches give per-class
F1 estimates with 95% CIs of approximately +/-0.10. "Both" at n=6 is too rare for
standalone metrics; the annotation retains it as a class but per-class metrics
on it will be omitted.

### Speaker gender

| Gender | n |
|--------|---|
| M | 413 |
| F | 53 |
| unknown | 34 |

## Presentation order

After sampling, speeches are shuffled with a separate fixed seed
(`SHUFFLE_SEED=43`) before `sample_idx` is assigned, so the order
presented to annotators is randomised rather than chronological.
This avoids era-clustered calibration drift (e.g. labelling a long run
of 1900s suffrage speeches in a row and unconsciously developing
era-specific heuristics). Both annotators see the same shuffled order
so progress is comparable.

## Annotation Protocol

Each speech is independently annotated by two annotators (Omar, Mandira) for:

1. **Stance** -- exactly one of: `for`, `against`, `both`, `irrelevant`
2. **Hostile sexism** -- multi-label across subcategories:
   - `dominative_paternalism` (men should rule)
   - `competitive_gender_differentiation` (men more competent)
   - `heterosexual_hostility` (women's sexuality as threat)
   - none of the above
3. **Benevolent sexism** -- multi-label across subcategories:
   - `protective_paternalism` (women need protection)
   - `complementary_gender_differentiation` (women have purity / nurturance)
   - `heterosexual_intimacy` (women complete men)
   - none of the above

Hostile and benevolent are independent: a speech may exhibit zero, one, or both.
Within each axis, multiple subcategories may apply.

Both stance and sexism are completed in a single pass per speech. The annotator
does not see the LLM's labels at any point during annotation. Annotation
disagreements will be resolved through discussion to produce consensus labels.

## Codebook

The annotation guide is grounded in Glick & Fiske (1996) Ambivalent Sexism
Theory. The pilot study (n=63 relevant speeches) identified essentialization
as the primary boundary case for benevolent sexism: attributing inherent
traits to women as a class counts as benevolent sexism; advocating for
women's rights without trait claims does not. This boundary is operationalised
in the refined codebook used for this 500-speech validation.

## Reproducibility

- Sampling script: `experiments/20260520_v8_500_validation/01_create_sample.py`
- Output sample: `experiments/20260520_v8_500_validation/validation_sample_500.parquet`
- Stats: `experiments/20260520_v8_500_validation/sampling_stats.json`
- Random seed: 42
- Source corpus: `outputs/llm_classification/v7_notrunc_results.parquet` (6,531 speeches)
- Excluded: 100 pilot speeches from `outputs/validation/validation_sample.parquet`
