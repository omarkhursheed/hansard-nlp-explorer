# LaTeX Fixes (Reviewer-requested)

Changes made to `699658104ad8f0eaf6ead859/latex/main.tex` (Overleaf repo).

## Missing citations (Reviewers iG9P, vF6k, Us7C)

Two `\cite` keys had a stray `p` prefix, rendering as "?" in the PDF.

1. `\cite{pYilmaz06102025}` -> `\cite{Yilmaz06102025}`
   - Line 144, Gender/Suffrage paragraph
   - Yilmaz & McCullion (2025), EU gender discourse paper

2. `\citep{ploureiro-etal-2022-timelms}` -> `\citep{loureiro-etal-2022-timelms}`
   - Line 151, Historical Linguistics paragraph
   - Loureiro et al. (2022), TimeLMs paper

## Broken reference (Reviewer vF6k)

3. `refer ~\ref{sec:gender_matching}` -> `Appendix~\ref{sec:gender_matching} for details`
   - Line 158, Hansard Data section
   - Rendered as "refer A" with no context; now reads properly

## Typo (found during review)

4. `seqxist` -> `sexist`
   - Line 474, Conclusion

## Spacing

5. `debates ,while` -> `debates, while`
   - Line 142, Parliamentary Debates paragraph

## Cohen's kappa specified (Reviewer Us7C)

6. First mention of kappa now reads "Cohen's kappa"
   - Line 415, Classification Validation section
   - Table 7 caption also updated

## Keyword list appendix (Reviewers vF6k, Us7C)

7. New Appendix section "Suffrage Speech Extraction Keywords"
   - Documents the two-tier extraction system from `extract_suffrage_reliable.py`
   - Tier 1 (HIGH, n=2,725): explicit suffrage regex patterns
   - Tier 2 (MEDIUM, n=3,806): women/female within 25 words of vote terms
   - Referenced from Data section with `Appendix~\ref{sec:keywords}`

8. Data section rewritten to describe two-tier extraction
   - Old: "We searched for terms related to women's voting rights..."
   - New: "We identified suffrage-related speeches using a two-tier keyword
     search applied to speech text..."

## Precision claims removed

9. Removed unvalidated precision claims (95% / 25.7%) from appendix
   - Code comments cited these from automated regex cross-checking, not human review
   - Appendix now simply states both tiers are high-recall by design and that
     the LLM filters false positives downstream (37% irrelevant)
