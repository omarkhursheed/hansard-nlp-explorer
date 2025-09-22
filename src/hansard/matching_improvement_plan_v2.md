# MP Matching Improvement Plan V2 (Revised)

## Critical Reflection on V1

### What's Realistic vs Wishful Thinking
- ❌ **ML style analysis**: Not enough labeled data, too complex
- ❌ **Network analysis**: Computationally expensive, uncertain ROI
- ✅ **Title/role matching**: High impact, straightforward
- ✅ **Constituency info**: Already in our data, just need to use it
- ✅ **Better temporal filtering**: Can eliminate impossible matches

## Revised Priority Strategy

### Phase 1: Use What We Already Have Better
1. **Constituency Matching**
   - We have constituency data for many MPs
   - Hansard often mentions "Member for [constituency]"
   - This can uniquely identify MPs

2. **Improve Temporal Bounds**
   - Currently using membership dates broadly
   - Should use actual session dates
   - Eliminate MPs who weren't active that exact year/month

3. **Chamber Filtering**
   - Lords can't appear in Commons debates
   - This immediately reduces candidate pool

### Phase 2: Smart Pattern Matching
1. **Title Resolution**
   - Build lookup: date + title → specific MP
   - "The Prime Minister" in 1950 → Clement Attlee
   - "Leader of the Opposition" → predictable person

2. **Initial Expansion**
   - "J. Smith" → all MPs with first name starting with J
   - But combine with temporal + chamber for disambiguation

3. **Common OCR/Typo Fixes**
   - Build list of common errors in Hansard
   - "Bavies" → "Davies"
   - Missing spaces: "MrSmith" → "Mr Smith"

### Phase 3: Validation-First Fuzzy Matching
1. **Conservative Edit Distance**
   - Only allow 1-2 character differences
   - Must pass temporal check first
   - Validate against known impossible matches

## Test-Driven Development Approach

### 1. Create Gold Standard Test Set
```python
# Manually verified matches from different eras
test_cases = [
    # (speaker_text, date, expected_mp, expected_gender)
    ("Mr. Churchill", "1950-05-26", "Winston Churchill", "M"),
    ("Mrs. Thatcher", "1980-06-15", "Margaret Thatcher", "F"),
    ("The Prime Minister", "1997-05-02", "Tony Blair", "M"),
    # Ambiguous cases
    ("Mr. Davies", "1950-05-26", "AMBIGUOUS", None),
    # Should not match
    ("Mr. Obama", "1950-05-26", None, None),
]
```

### 2. Test Metrics to Track
- **Precision**: Of matches we make, how many are correct?
- **Recall**: Of correct matches, how many do we find?
- **Ambiguity rate**: How many remain ambiguous?
- **False positive rate**: How many wrong matches?

### 3. Validation Rules (Must Pass ALL)
- Temporal validity: MP must be active on that date
- Chamber consistency: Lords in Lords, Commons in Commons
- Gender consistency: If we know gender from title
- One-to-one in session: Same MP can't be two speakers in same debate

## Implementation Order

### Week 1: Foundation
1. Build comprehensive test suite
2. Implement temporal validation
3. Add chamber filtering
4. Measure baseline performance

### Week 2: Core Improvements
1. Add constituency matching
2. Implement title resolution
3. Handle common initials
4. Re-measure performance

### Week 3: Careful Fuzzy Matching
1. Build OCR correction rules
2. Implement conservative edit distance
3. Extensive testing for false positives
4. Final performance measurement

## Risk Mitigation

### Preventing False Positives
1. **Never match if**:
   - Temporal bounds violated
   - Chamber mismatch
   - Multiple candidates with equal likelihood

2. **Require higher confidence for**:
   - Female MPs (smaller sample, higher impact of errors)
   - Cross-party matches
   - Large time gaps

3. **Always flag as ambiguous if**:
   - Edit distance > 2
   - Multiple valid candidates
   - Missing critical context

## Success Criteria (Realistic)

From current state:
- High-confidence: 30% → **45%** (+50% improvement)
- Ambiguous: 32% → **25%** (-22% improvement)
- Unmatched: 38% → **30%** (-21% improvement)
- False positive rate: **< 0.5%**
- Test set accuracy: **> 98%**