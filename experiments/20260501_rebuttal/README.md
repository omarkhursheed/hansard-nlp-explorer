# Rebuttal Materials (ARR March 2026)

Rebuttal deadline: May 4, 2026.
Scores: iG9P=2.5, vF6k=2.0, Us7C=2.5

## What's been done

### LaTeX fixes (pushed to Overleaf)

All changes documented in `00_latex_fixes.md`. Already pushed.

1. Fixed 2 missing citations (stray `p` prefix in bib keys: Yilmaz, Loureiro)
2. Fixed broken "refer A" reference
3. Fixed "seqxist" typo in Conclusion
4. Fixed comma spacing
5. Specified "Cohen's kappa" (Reviewer Us7C)
6. Added Appendix: Suffrage Speech Extraction Keywords (Reviewers vF6k, Us7C)
   - Documents two-tier extraction (HIGH: explicit terms, MEDIUM: proximity)
   - Referenced from Data section
7. Rewrote Data section to describe two-tier extraction accurately
8. Removed false precision claims (95%/25.7% were from automated regex
   cross-checking, not human validation)

### Irrelevant classification audit (Reviewer iG9P)

Script: `01_irrelevant_audit.py`
Results: `01_irrelevant_audit.json`, `01_irrelevant_audit.txt`

iG9P asked: "A manual inspection of a sample of [irrelevant] cases should
be provided to verify that the model's decisions are reasonable."

- Sampled 30 speeches classified as irrelevant (15 HIGH tier, 15 MEDIUM)
- All 30 correctly classified as irrelevant
- 28/30 have zero suffrage content; 2/30 mention suffrage in a single
  passing sentence (analogy about India; prisoner who dwelt on "Votes
  for Women")
- Irrelevant rate by tier: HIGH=29.4%, MEDIUM=42.2%

**For rebuttal:** "We manually inspected 30 randomly sampled
irrelevant-classified speeches (15 from each extraction tier). All 30
were correctly classified as irrelevant. Typical false positives from
extraction include speeches where 'franchise' refers to male voting
rights or where 'women' and 'vote' co-occur in unrelated contexts
(e.g., transport to polling stations). The irrelevant rate is higher
for Tier 2 (42%) than Tier 1 (29%), consistent with Tier 2's broader
matching criteria."

### Per-class P/R/F1 + disagreement analysis (Reviewers iG9P, Us7C)

Script: `02_per_class_and_disagreement.py`
Results: `02_per_class_and_disagreement.json`

**Per-class metrics (iG9P wanted P/R/F1, not just kappa):**

| Class      | Precision | Recall | F1    | n  |
|------------|-----------|--------|-------|----|
| for        | 0.656     | 0.913  | 0.764 | 46 |
| against    | 0.667     | 0.500  | 0.571 |  8 |
| both       | 0.600     | 0.333  | 0.429 |  9 |
| irrelevant | 0.840     | 0.568  | 0.677 | 37 |

Claude's main tendency is over-predicting "for" -- high recall (91%)
but lower precision (66%), pulling 15 irrelevant speeches into "for".
Only 3/30 disagreements are directional (for<->against), so Claude
rarely gets stance backwards.

**n=96 explanation (Us7C asked why Table 7 shows n=96):**

One annotator completed 96 of 100 speeches; the other completed all
100. Human-human agreement (kappa=0.463) is computed on the 96 shared
speeches. All 100 have gold labels via 63 agreements + 37 resolutions
(4 resolutions are for speeches only one annotator completed).

**Per-instance disagreement (Us7C wanted more than distributional):**

- 30/100 Claude-human disagreements
- 14/30 (47%) overlap with speeches where humans also disagreed
- Claude's errors concentrate on genuinely ambiguous speeches
- Dominant error: irrelevant->for (15 cases) -- Claude finds relevance
  where humans didn't
- Only 3 direction errors (for<->against)

**For rebuttal:** "Of Claude's 30 disagreements with human-consensus
labels, 14 (47%) are on speeches where the two human annotators also
disagreed, indicating that Claude's errors concentrate on genuinely
ambiguous cases. The dominant error pattern is classifying irrelevant
speeches as 'for' (15/30), rather than confusing stance direction
(3/30)."

## What's still needed

### Critical (for rebuttal by May 4)

- **Sexism axis validation**: Mandira re-annotating 100 speeches with
  sexism axes. Omar also doing 100. This is the #1 reviewer concern
  (all 3 reviewers). Annotation guide saved at
  `outputs/validation/sexism_annotation_guide.md`.

- **Terminology**: Change "gold standard"/"gold labels" to
  "human-consensus labels" throughout paper (Ashique's suggestion).

### For May 26 resubmission

- Move AST discussion to Introduction (vF6k)
- Reorder: validation before full annotation (vF6k)
- Motivate keeping three separate taxonomies (vF6k)
- Clarify multi-label vs single-label across axes (Us7C)
- Decide on fine-tuning section: cut or expand (iG9P)
- Add keyword list to supplementary materials (done in appendix)

## File inventory

```
experiments/20260501_rebuttal/
  README.md                              -- this file
  00_latex_fixes.md                      -- documents all LaTeX changes
  01_irrelevant_audit.py                 -- reproducible audit script
  01_irrelevant_audit.json               -- structured results
  01_irrelevant_audit.txt                -- full text of 30 audited speeches
  02_per_class_and_disagreement.py       -- P/R/F1 + disagreement analysis
  02_per_class_and_disagreement.json     -- structured results
```

## Other artifacts created this session

- `scripts/manuscript/05_table5_sexism_by_stance.py` -- reproduces Table 5
  (PR #3 merged to main)
- `outputs/validation/gold_annotations_resolved.csv` -- 100 resolved
  annotations as single CSV (for sharing with collaborators)
- `outputs/validation/sexism_annotation_guide.md` -- annotation rubric
- `outputs/validation/sexism_classification_framework.md` -- theoretical
  grounding (Glick & Fiske 1996, Fiske et al. 2002, Prentice & Carranza 2002)
