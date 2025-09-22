# MP Matching Improvement Plan

## Current State
- Only 30% high-confidence matches
- 32% ambiguous (multiple candidates)
- 38% completely unmatched

## Potential Improvement Strategies

### 1. Enhanced Context Matching
**Hypothesis**: We can use additional context from debates to disambiguate
- **Chamber**: Commons vs Lords
- **Debate topic**: Defence debates → defence ministers
- **Sequential speakers**: MPs often respond to each other
- **Party affiliation**: From debate context
- **Ministerial titles**: "The Secretary of State"

### 2. Fuzzy Name Matching
**Hypothesis**: Handle variations, typos, OCR errors
- Edit distance (Levenshtein)
- Phonetic matching (Soundex, Metaphone)
- Initial matching: "A. Smith" → "Andrew Smith"
- Nickname resolution: "Tony" → "Anthony"

### 3. Title and Role Extraction
**Hypothesis**: Ministerial titles can uniquely identify MPs
- "The Prime Minister" + date → specific person
- "Secretary of State for Defence" + date → specific person
- "Member for Manchester" → constituency matching

### 4. Co-occurrence Analysis
**Hypothesis**: MPs who speak together are contemporaries
- Build speaker networks per session
- Use graph analysis to identify clusters
- Cross-reference with known MP relationships

### 5. Writing Style Analysis
**Hypothesis**: Each MP has unique speech patterns
- Vocabulary preferences
- Sentence structure
- Topic preferences
- Use ML to build speaker profiles

### 6. Historical Records Cross-Reference
**Hypothesis**: Use external sources for validation
- Wikipedia entries for MPs
- Historic Hansard IDs
- Parliamentary records
- News archives

## Testing Framework

### For Each Strategy:
1. **Baseline Test Set**: Create ground truth with manually verified matches
2. **False Positive Test**: Ensure we don't match wrong MPs
3. **False Negative Test**: Ensure we don't miss correct matches
4. **Ambiguity Resolution Test**: Measure reduction in ambiguous matches
5. **Temporal Consistency Test**: Same person across time periods

## Risk Assessment

### False Positive Risks (Wrong Match)
- **Critical**: Gender misattribution
- **High**: Temporal impossibility (MP not alive/active)
- **Medium**: Wrong party affiliation
- **Low**: Similar names in same period

### False Negative Risks (Missed Match)
- **High**: Missing female MPs (smaller sample)
- **Medium**: Missing short-term MPs
- **Low**: Missing prominent MPs

## Implementation Priority

1. **Quick Wins** (High impact, low risk)
   - Ministerial title resolution
   - Constituency matching
   - Fix OCR/typo issues

2. **Medium Term** (Good impact, moderate complexity)
   - Temporal + context matching
   - Initial/nickname resolution
   - Chamber-based filtering

3. **Long Term** (Complex, needs validation)
   - ML-based style analysis
   - Network analysis
   - Cross-source validation

## Success Metrics

- Increase high-confidence matches from 30% to 50%+
- Reduce ambiguous matches from 32% to <20%
- Reduce unmatched from 38% to <30%
- Maintain <1% false positive rate
- Achieve >95% accuracy on test set