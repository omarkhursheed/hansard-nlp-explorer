# Resubmission Handoff (May 2026)

ARR May 26 resubmission deadline. Paper: "Two Centuries of Sexism in British Parliament."
Previous scores: iG9P=2.5, vF6k=2.0, Us7C=2.5.

## Overleaf repo

Cloned at `699658104ad8f0eaf6ead859/` inside the hansard-nlp-explorer repo.
Remote: `https://git.overleaf.com/699658104ad8f0eaf6ead859`
Paper source: `699658104ad8f0eaf6ead859/latex/main.tex`
Bib: `699658104ad8f0eaf6ead859/custom.bib`
Already pushed: citation fixes, typo fixes, keyword appendix, kappa specification.

## What was done (April 28 - May 3)

### LaTeX fixes (pushed to Overleaf)
- Fixed 2 missing citations (pYilmaz, ploureiro -- stray `p` prefix)
- Fixed broken "refer A" reference
- Fixed "seqxist" typo, comma spacing
- Specified "Cohen's kappa"
- Added Appendix: Suffrage Speech Extraction Keywords (two-tier system)
- Rewrote Data section for two-tier extraction
- Removed false precision claims (95%/25.7%)
- Documented in `experiments/20260501_rebuttal/00_latex_fixes.md`

### Irrelevant audit (Reviewer iG9P)
- 30 randomly sampled irrelevant speeches, all 30 correctly classified
- Script: `experiments/20260501_rebuttal/01_irrelevant_audit.py`
- Results: `experiments/20260501_rebuttal/01_irrelevant_audit.{json,txt}`

### Per-class P/R/F1 + disagreement analysis (Reviewers iG9P, Us7C)
- Per-class metrics for stance: for F1=0.764, against F1=0.571, both F1=0.429, irrelevant F1=0.677
- n=96 explanation: Mandira annotated 96/100 speeches
- 14/30 Claude errors overlap with human disagreements
- Script: `experiments/20260501_rebuttal/02_per_class_and_disagreement.py`
- Results: `experiments/20260501_rebuttal/02_per_class_and_disagreement.json`

### Sexism axis validation (all 3 reviewers -- the big one)
- Both annotators re-labeled 63 relevant speeches on sexism axes
- Omar used new v2 app, Mandira used old app (both produce compatible output)
- 19 disagreements resolved through discussion (12 binary + 7 axis)
- Self-contained in `experiments/20260503_sexism_validation/`

Key numbers:

Human-human agreement (n=63, before resolution):
  Binary: 81% kappa=0.450
  Axis A: 84% kappa=0.496
  Axis B: 81% kappa=0.440
  Axis C: 79% kappa=0.418

LLM vs human-consensus (n=63, after resolution):
  Binary: 79% kappa=0.445
  Axis A: 75% kappa=0.368
  Axis B: 71% kappa=0.298
  Axis C: 75% kappa=0.383

14 consensus-sexist speeches. Where both human and LLM say sexist (n=9):
  Axis A: 67% match, Axis B: 44% match, Axis C: 67% match

LLM under-detects: 5/14 consensus-sexist missed by LLM (4/5 are benevolent sexism).

### Other artifacts
- `scripts/manuscript/05_table5_sexism_by_stance.py` -- reproduces Table 5 (PR #3)
- `outputs/validation/gold_annotations_resolved.csv` -- 100 resolved stance annotations
- `outputs/validation/sexism_annotation_guide.md` -- annotation rubric
- `outputs/validation/sexism_classification_framework.md` -- theoretical grounding

## What's still needed for May 26

### Must do (paper revisions)
1. Move AST discussion to Introduction (Reviewer vF6k)
2. Reorder: present validation before full annotation (Reviewer vF6k)
3. Motivate keeping three separate taxonomies (Reviewer vF6k)
4. Clarify multi-label vs single-label across axes (Reviewer Us7C)
5. Add sexism validation results to paper (new subsection or expand existing)
6. Add per-class P/R/F1 table to paper
7. Explain n=96 in Table 7 (one sentence)
8. Terminology: "gold standard/gold labels" -> "human-consensus labels" (Ashique)
9. Add keyword list reference in appendix (done) and potentially supplementary

### Should do
10. Decide on fine-tuning section: cut or expand (Reviewer iG9P says "raises more questions than it resolves")
11. Address Hamilton citation verbatim title (Reviewer vF6k)
12. Remove "Historical Linguistics Shifts" subsection if not used in analysis (Reviewer vF6k)

### Nice to have
13. Per-instance disagreement analysis in paper (Reviewer Us7C)
14. Sentiment confound analysis expansion

## Reviewer concerns checklist

### Reviewer iG9P (2.5 -> target 3+)
- [x] 37% irrelevant explanation + manual audit
- [x] Per-class P/R/F1
- [x] How were 100 validation samples selected (stratified, documented)
- [ ] Fine-tuning section decision
- [x] Missing citations fixed

### Reviewer vF6k (2.0 -> target 2.5+)
- [ ] AST in Introduction
- [ ] Validation before annotation (presentation order)
- [ ] Motivate three taxonomies
- [x] "refer A" fixed
- [x] Keyword list in appendix
- [ ] Hamilton citation
- [ ] Historical linguistics subsection

### Reviewer Us7C (2.5 -> target 3+)
- [x] Kappa type specified (Cohen's)
- [x] Keyword list
- [x] n=96 explanation (Mandira annotated 96/100)
- [x] Per-instance disagreement analysis (computed, needs to go in paper)
- [ ] Multi-label vs single-label clarification
- [x] Sexism validation (done, needs to go in paper)

## Key files

```
experiments/20260501_rebuttal/
  00_latex_fixes.md
  01_irrelevant_audit.{py,json,txt}
  02_per_class_and_disagreement.{py,json}
  annotations/{omar,mandira}.jsonl          -- v2 sexism annotations
  sexism_disagreements.json
  sexism_resolutions.json
  sexism_annotation_app.py                  -- v2 annotation app
  sexism_resolution_app.py                  -- disagreement resolution app

experiments/20260503_sexism_validation/
  README.md                                 -- full writeup
  01_sexism_axis_agreement.{py,json,txt}    -- reproducible results
  omar_v2.jsonl                             -- Omar's annotations (copy)
  mandira_v2.jsonl                          -- Mandira's annotations (copy)
  sexism_resolutions.json                   -- resolutions (copy)
  sexism_disagreements.json                 -- disagreement list (copy)
  annotation_guide.md                       -- rubric (copy)

699658104ad8f0eaf6ead859/                   -- Overleaf repo clone
  latex/main.tex                            -- paper source (already pushed)
  custom.bib                                -- bibliography
```

## Collaborators
- Mandira Sawkar (ms7201@rit.edu) -- co-author, annotator
- Ashiqur R. KhudaBukhsh (axkvse@rit.edu) -- advisor, rebuttal expert
- Mandira consolidated reviewer concerns: https://docs.google.com/document/d/1GfVbtqHNedvDDLxnAzuugI9m1Ypd5r5Rv4BC2YPsppU/edit

## Validation sample design
100 speeches from stratified sampling of 6,531 suffrage speeches (seed=42):
- Mandatory: all female "against", all "neutral", all female "both", all low-confidence for/against
- Remainder: proportional by stance
- Result: 30 irrelevant, 25 against, 25 for, 20 both
- 63 relevant speeches used for sexism annotation (37 irrelevant skipped)
