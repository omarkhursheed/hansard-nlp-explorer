# Female AGAINST Speeches: Preliminary Validation

**Date**: November 21, 2025
**Dataset**: Claude Sonnet 4.5 classification (6,531 speeches)
**Category**: Female MPs classified as AGAINST women's suffrage
**Total Count**: 5 speeches
**Status**: Preliminary review - formal validation pending

---

## Critical Finding

Initial inspection suggests **40-60% error rate** (2-3 false positives out of 5 speeches).

This is a critical validation gap that could significantly impact paper conclusions about:
- Female MP opposition patterns
- Gender differences in argumentation
- Historical narrative of suffrage debates

---

## All 5 Female AGAINST Speeches

### Speech #1: Viscountess Nancy Astor (1920)
- **Date**: February 27, 1920
- **Speech ID**: `15afbf4e6cc3a3a3_speech_3`
- **Confidence**: 0.95 (HIGH)
- **Top Quote**: "Girls of twenty-one are, I think hon. Members will agree, far more emotional than men of the same age."
- **Context**: Debate about lowering women's voting age to 21

**Preliminary Assessment**: **LIKELY CORRECT**
- Astor was opposing EXPANSION of suffrage (lowering age to 21)
- This is opposition to a suffrage reform, not opposition to women voting
- Classification as "AGAINST" is technically accurate but needs context

**Note**: First speech may actually be Mr. Murray (gender misattribution issue noted in notes.md - needs verification)

---

### Speech #2: Miss Eleanor Rathbone (1931)
- **Date**: February 3, 1931
- **Speech ID**: `682d379021f8344f_speech_41`
- **Confidence**: 0.95 (HIGH)
- **Top Quote**: "the better-trained minds of the community should be provided with a special channel reflecting their views"
- **Context**: Unknown - needs full text review

**Preliminary Assessment**: **UNCERTAIN - NEEDS VALIDATION**
- Quote suggests educational/meritocratic qualifications
- May be about university representation, not general suffrage
- Similar to false positive pattern in validation sample

**Action Required**: Read full speech to determine if actually about women's voting rights

---

### Speech #3: Miss Florence Horsbrugh (1933)
- **Date**: November 9, 1933
- **Speech ID**: `a3d697c68cd32743_speech_36`
- **Confidence**: 0.45 (LOW - below typical threshold)
- **Top Quote**: "In France, as yet, there are not votes for women."
- **Context**: Statement of fact about France

**Preliminary Assessment**: **LIKELY FALSE POSITIVE - IRRELEVANT**
- Simple factual statement about France, not UK
- Not expressing a stance on UK suffrage policy
- Low confidence (0.45) suggests LLM was uncertain
- Should be classified as IRRELEVANT, not AGAINST

**Recommendation**: Reclassify as IRRELEVANT or exclude from analysis

---

### Speech #4: Mrs. Hilda Runge (1935)
- **Date**: May 15, 1935
- **Speech ID**: `1bae0b1b66e41903_speech_136`
- **Confidence**: 0.85 (HIGH)
- **Top Quote**: "the common sense thing to do and the best service that one can render to the women of India is to vote with the Government..."
- **Context**: Government of India Act 1935 - women's voting rights in India

**Preliminary Assessment**: **UNCERTAIN - CONTEXT MATTERS**
- About voting rights for women in India, not UK
- Part of 80/102 speeches from 1935 that are about India Act
- May be opposing extension of suffrage to Indian women
- OR may be supporting Government position on Indian women's suffrage

**Action Required**:
- Read full speech to determine actual stance
- Determine if India Act speeches should be included in UK suffrage analysis
- Check if "vote with Government" means supporting or opposing Indian women's suffrage

---

### Speech #5: Virginia Bottomley (2001)
- **Date**: November 14, 2001
- **Speech ID**: `a63403d05360bc57_speech_66`
- **Confidence**: 0.95 (HIGH)
- **Top Quote**: "I find all-women shortlists, or indeed twinning, extraordinarily invidious."
- **Context**: Debate about Labour Party candidate selection procedures

**Preliminary Assessment**: **LIKELY FALSE POSITIVE - IRRELEVANT**
- About candidate selection methods (all-women shortlists), NOT voting rights
- Confuses women in politics with women's suffrage
- This is about party procedures for selecting candidates
- Should be classified as IRRELEVANT, not AGAINST suffrage

**Recommendation**: Reclassify as IRRELEVANT or exclude from analysis

---

## Summary of Preliminary Assessment

| Speech | Speaker | Year | Confidence | Assessment | Action |
|--------|---------|------|------------|------------|--------|
| #1 | Astor | 1920 | 0.95 | Likely CORRECT (opposing age reduction) | Verify gender attribution |
| #2 | Rathbone | 1931 | 0.95 | UNCERTAIN | Read full speech |
| #3 | Horsbrugh | 1933 | 0.45 | FALSE POSITIVE | Reclassify as IRRELEVANT |
| #4 | Runge | 1935 | 0.85 | UNCERTAIN (India context) | Read full speech, check context |
| #5 | Bottomley | 2001 | 0.95 | FALSE POSITIVE | Reclassify as IRRELEVANT |

**Error Rate Estimate**: 2-3 false positives out of 5 (40-60%)

---

## Error Patterns Identified

### Pattern 1: Women's Issues ≠ Suffrage
- **Example**: Bottomley opposing candidate shortlists
- **Root Cause**: LLM conflating women's political participation with voting rights
- **Prevalence**: 1 confirmed case

### Pattern 2: Factual Statements Misclassified as Stance
- **Example**: Horsbrugh stating "France has no votes for women"
- **Root Cause**: LLM interpreting statement of fact as position
- **Prevalence**: 1 confirmed case

### Pattern 3: India Act Confusion
- **Example**: Runge on India women's suffrage
- **Root Cause**: Unclear whether India speeches should be included in UK analysis
- **Prevalence**: Potentially 80 speeches in 1935 dataset

### Pattern 4: University Representation vs General Suffrage
- **Example**: Rathbone on "better-trained minds"
- **Root Cause**: Special franchise categories confused with general voting rights
- **Prevalence**: Unknown - needs investigation

---

## Validation Priority

These 5 speeches are **Priority 1** in the audit plan because:

1. **Extremely rare category**: Only 0.08% of dataset (5/6,531)
2. **High impact**: Critical for understanding female opposition patterns
3. **High error rate**: 40-60% false positive rate suspected
4. **Paper conclusions**: Results could fundamentally change narrative

**Time Estimate**: 30-60 minutes to validate all 5 speeches

---

## Recommendations

### Immediate Actions (Before Paper Submission)

1. **Validate ALL 5 speeches** - read full text, determine true stance
2. **Document findings** - update this file with validated results
3. **Reclassify errors** - mark false positives as IRRELEVANT
4. **Update analysis** - recalculate female opposition statistics

### Methodological Decisions Needed

1. **India Act 1935**: Include or exclude from UK suffrage analysis?
   - If include: How to handle comparative/colonial arguments?
   - If exclude: Filter all India-related speeches (n≈80)

2. **Age reduction debates**: Count as AGAINST or separate category?
   - Astor opposing lowering age to 21 is technically AGAINST expansion
   - But not AGAINST suffrage principle

3. **Confidence threshold**: Should speeches below 0.5 confidence be excluded?
   - Horsbrugh (0.45) correctly flagged as uncertain by LLM

### Paper Implications

If error rate is 40-60%, the **true count of female AGAINST speeches is 2-3**, not 5.

This changes conclusions about:
- Female MP opposition was extremely rare (not just rare)
- Gender differences in argumentation may be more pronounced
- Need to caveat conclusions about female opposition patterns

---

## Next Steps

1. Execute formal validation using validation app (see AUDIT_PLAN.md)
2. Read full text of all 5 speeches
3. Document validated results
4. Update paper with validated statistics
5. Consider broader validation of all AGAINST speeches (n=508) for similar errors

---

**Status**: Preliminary assessment based on quick inspection of quotes
**Formal Validation**: Pending execution of AUDIT_PLAN.md
**Last Updated**: November 21, 2025
