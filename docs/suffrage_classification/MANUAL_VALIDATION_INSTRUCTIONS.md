# Manual Validation Instructions

## Overview

You'll review 48 classified speeches to assess the accuracy of the LLM classification system.

**Time estimate**: 2-3 hours (3-4 minutes per speech)

**Goal**: Calculate precision by stance category and identify systematic errors

---

## Setup

1. **Open the template**: `outputs/validation/validation_recording_template.csv`
2. **Open the display tool**: Run this in a terminal to see each speech:
   ```bash
   python3 scripts/classification/show_validation_samples.py --input outputs/validation/validation_sample_n48.parquet
   ```

---

## For Each Speech

### Step 1: Read the Speech
- Read the full speech text (shown in the display tool)
- Understand what the speaker is arguing

### Step 2: Check the Stance
**Question**: Is the LLM's stance classification correct?

- **FOR**: Speaker supports women's suffrage/voting rights
- **AGAINST**: Speaker opposes women's suffrage
- **BOTH**: Speaker has mixed position (e.g., supports voting but opposes holding office)
- **NEUTRAL**: Speaker is genuinely indifferent or accepts either outcome
- **IRRELEVANT**: Speech is not about women's suffrage at all

**Record in template**:
- `your_judgment`: What you think the stance should be (for/against/both/neutral/irrelevant)
- `stance_correct`: YES or NO

### Step 3: Check the Reasons
**Question**: Are the extracted arguments accurate?

For each reason listed by the LLM:
- Does the rationale match what the speaker actually said?
- Are the argument types (equality, instrumental_effects, etc.) appropriate?
- Do the arguments actually support/oppose suffrage?

**Record in template**:
- `reasons_correct`: YES, PARTIAL, or NO
  - YES: All reasons are accurate
  - PARTIAL: Some reasons correct, some wrong/missing
  - NO: Reasons are mostly wrong or missing key arguments

### Step 4: Check the Quotes
**Question**: Are the quoted excerpts accurate and representative?

- Are quotes actual verbatim text from the speech?
- Do quotes support the claimed arguments?
- Are they the most representative quotes?

**Record in template**:
- `quotes_accurate`: YES, PARTIAL, or NO

### Step 5: Notes
**Record in template**:
- `notes`: Any important observations:
  - Why did it fail? (e.g., "keywords in wrong context", "missed sarcasm", "ambiguous wording")
  - Was it a difficult case?
  - Edge cases or interesting patterns

---

## Common Issues to Watch For

### False Positives (IRRELEVANT marked as FOR/AGAINST)
- Keywords like "women" and "vote" appear but in unrelated contexts
- Example: "vote on this bill" vs "right to vote"
- Trade/economic bills that mention women tangentially

### Confirmation Bias
- LLM assumes speech is about suffrage because it's in the dataset
- Forces interpretation even when topic is ambiguous

### Sarcasm/Irony
- Speaker says something that sounds pro-suffrage but is actually mocking it
- Requires understanding tone and context

### Procedural vs Substantive
- Speaker discusses parliamentary procedure for suffrage bills
- Doesn't necessarily indicate their personal stance

### BOTH vs NEUTRAL
- BOTH: Has clear positions on different aspects (support X, oppose Y)
- NEUTRAL: Genuinely indifferent or procedural discussion

---

## Example Validation

**Speech ID**: 51d1ffbc81164a8a_speech_54
**LLM Classification**: AGAINST (conf=0.70)
**Speech Content**: Discussion of Trade Control Bill (coal/bread prices)

**Your Assessment**:
- `your_judgment`: irrelevant
- `stance_correct`: NO
- `reasons_correct`: NO (reasons are about trade, not suffrage)
- `quotes_accurate`: YES (quotes are accurate but from wrong context)
- `notes`: "False positive. Keywords 'women' and 'vote' appear but refer to trade council membership and parliamentary votes on trade bills, not women's suffrage. LLM exhibited confirmation bias."

---

## Tips

1. **Read the full speech**: Don't just skim - context matters
2. **Trust your judgment**: You're the human expert here
3. **Be consistent**: Use the same standards across all speeches
4. **Document patterns**: If you see the same error type repeatedly, note it
5. **Take breaks**: 2-3 hours is long - break it into 2-3 sessions
6. **Use the FALSE_POSITIVE_ANALYSIS.md**: Reference for what to look for

---

## After Validation

1. **Save your filled template**: `validation_recording_template.csv`
2. **Run analysis script**:
   ```bash
   python3 scripts/classification/analyze_validation_results.py
   ```
3. **Review the accuracy report**: Will show precision by stance, confidence, etc.

---

## Questions While Validating?

**Ambiguous case**: When in doubt, mark in `notes` and move on. We can discuss edge cases.

**LLM did something weird**: Document it in `notes` - these are learning opportunities.

**Taking too long**: If a speech is very long/complex, it's okay to spend 5-10 minutes on it.

---

## Good Luck!

This validation will give us robust quality metrics for the classification system. Your careful review is invaluable!
