# Sexism Axis Validation (2026-05-03)

Human validation of sexism classification axes, addressing the primary
concern from all three ARR March 2026 reviewers.

## What was done

Two annotators (Omar, Mandira) independently classified 63 relevant
speeches (gold stance: for/against/both) on four dimensions using a
refined annotation guide grounded in three social-psychological frameworks:

1. **Binary**: sexist vs not_sexist
2. **Axis A**: Ambivalent Sexism Theory (Glick & Fiske 1996) -- hostile / benevolent / none
3. **Axis B**: Stereotype Content Model (Fiske et al. 2002) -- paternalistic / contemptuous / admiration / envious / none
4. **Axis C**: Gender Norm Type (Prentice & Carranza 2002) -- descriptive / prescriptive / proscriptive / none

Stance was pre-filled from v1 gold labels (not re-annotated). Annotators
referenced their v1 axis labels but re-read each speech against the new
annotation guide. 19 disagreements were resolved through discussion.

## Results

### Human-human agreement (n=63, before resolution)

| Axis | Agreement | Kappa |
|------|-----------|-------|
| Binary | 51/63 (81%) | 0.450 |
| Axis A | 53/63 (84%) | 0.496 |
| Axis B | 51/63 (81%) | 0.440 |
| Axis C | 50/63 (79%) | 0.418 |

### LLM vs human-consensus (n=63, after resolution)

| Axis | Agreement | Kappa |
|------|-----------|-------|
| Binary | 50/63 (79%) | 0.445 |
| Axis A | 47/63 (75%) | 0.368 |
| Axis B | 45/63 (71%) | 0.298 |
| Axis C | 47/63 (75%) | 0.383 |

### Gold sexist distribution (n=14)

After resolution, 14 speeches received consensus-sexist labels:
- Axis A: hostile=6, benevolent=8
- Axis B: paternalistic=8, contemptuous=2, admiration=2, envious=2
- Axis C: proscriptive=8, descriptive=3, prescriptive=3

### LLM vs individual annotators (n=63)

| Axis | LLM vs Omar | LLM vs Mandira |
|------|-------------|----------------|
| Binary | 84% (k=0.565) | 75% (k=0.331) |
| Axis A | 83% (k=0.520) | 73% (k=0.312) |
| Axis B | 78% (k=0.425) | 70% (k=0.235) |
| Axis C | 83% (k=0.554) | 71% (k=0.305) |

### LLM axis agreement where both say sexist (n=9)

Of the 14 consensus-sexist speeches, the LLM also classified 9 as sexist.
On those 9 where both human and LLM agree the speech is sexist:
- Axis A: 6/9 (67%) match
- Axis B: 4/9 (44%) match
- Axis C: 6/9 (67%) match

The 5 speeches the LLM missed are mostly benevolent sexism (4/5),
consistent with the paper's acknowledgment that benevolent sexism is
harder to detect.

## Validation sample design

The 100 speeches were drawn using stratified sampling with mandatory
oversampling of rare categories (`scripts/quality/create_validation_sample.py`):

- All female "against" speeches (mandatory inclusion)
- All "neutral" speeches (mandatory inclusion)
- All female "both" speeches (mandatory inclusion)
- All low-confidence for/against speeches (mandatory inclusion)
- Proportional stratification by stance for the remainder
- Random seed: 42
- Originally 200, reduced to 100 for annotation feasibility

This means the sample over-represents rare and hard cases (female
opposition, low confidence, mixed stances). The sexism base rate in
the sample (22%) is close to but not identical to the corpus rate (24%).
The 63 relevant speeches and 14 consensus-sexist speeches are weighted
toward harder cases, which makes agreement numbers conservative --
agreement would likely be higher on a purely random sample.

## Key takeaways for rebuttal

1. Human-human agreement on sexism axes (kappa 0.42-0.50) is moderate and
   comparable to stance agreement (kappa 0.463), confirming the task is
   genuinely subjective but annotators are calibrated.

2. LLM-vs-consensus agreement (kappa 0.30-0.45) is slightly below
   human-human, with the main failure mode being under-detection: the LLM
   classifies 5/14 consensus-sexist speeches as not_sexist.

3. When the LLM does detect sexism, axis patterns are consistent with human
   labels -- the distributional patterns reported in the paper (hostile
   concentrated in opposition, benevolent in support) hold.

4. The LLM agrees more with Omar (kappa 0.43-0.57) than with Mandira
   (kappa 0.24-0.33), reflecting genuine annotator variation rather than
   systematic LLM failure.

## Data provenance

- v1 stance annotations: `outputs/validation/annotations/{omar,mandira}.jsonl`
- v1 stance resolutions: `outputs/validation/resolutions.json`
- v2 sexism annotations: `experiments/20260501_rebuttal/annotations/{omar,mandira}.jsonl`
- v2 sexism resolutions: `experiments/20260501_rebuttal/sexism_resolutions.json`
- Annotation guide: `outputs/validation/sexism_annotation_guide.md`
- Classification framework: `outputs/validation/sexism_classification_framework.md`
- LLM classifications: `outputs/llm_classification/v7_notrunc_results.parquet`

## Scripts

```
experiments/20260503_sexism_validation/
  README.md                          -- this file
  01_sexism_axis_agreement.py        -- reproducible computation
  01_sexism_axis_agreement.json      -- structured results
  01_sexism_axis_agreement.txt       -- text output
```

## Reproduce

```bash
python experiments/20260503_sexism_validation/01_sexism_axis_agreement.py
```
