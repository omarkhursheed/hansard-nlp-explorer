# Remaining Visualization Fixes

## ✅ COMPLETED
1. Added ultra filtering level (collaborator's stopwords)
2. Visualization word filtering (removes would, may, people, one, two, etc.)
3. Increased to 30 words per chart (bigger, more insights)
4. Tested with large sample - works great!

## 🔴 CRITICAL - Still Broken

### 1. Topic Prevalence Chart Shows All Zeros
**Problem:** All 6 topics show 0.00 for male and female
**File:** professional_visualizations.py → create_topic_prevalence()
**Fix needed:**
- Debug why weights are 0
- Check topic data structure being passed
- Fix normalization

### 2. Milestone Charts - Empty Panels
**Problem:** Top-left and bottom-left panels are blank
**File:** professional_visualizations.py → create_milestone_comparison()
**Fix needed:**
- Implement vocabulary evolution chart (new/disappeared words)
- Pass unigrams correctly (handle male_unigrams vs top_unigrams difference)
- Add pre vs post word frequency bars

### 3. Milestone Shows "Debates: 0"
**Problem:** Summary panel shows "Pre-Period: Debates: 0"
**File:** professional_visualizations.py → create_milestone_comparison()
**Fix needed:**
- For gender dataset: use total_male_speeches + total_female_speeches
- For overall: use total_debates
- Handle both cases correctly

## 🟡 IMPORTANT - Needs Improvement

### 4. Temporal Chart - Ugly Filled Area
**Problem:** Stacked area looks cluttered, milestone labels overlap
**File:** professional_visualizations.py → create_temporal_participation()
**Options:**
- Change to dual line charts (not filled)
- Reduce milestone label clutter
- Better label placement

**Recommendation:** Keep simple - just line chart for female %, remove fill

### 5. Overall Corpus Color Inconsistency
**Problem:** Uses green (accent1) instead of cohesive color scheme
**File:** corpus_analysis.py (inline matplotlib code)
**Fix:** Define COLORS['overall'] and use consistently

## Estimated Time

- Critical fixes (1-3): ~90 minutes
- Improvements (4-5): ~30 minutes
- **Total: ~2 hours**

## Priority

Fix in order:
1. Topic prevalence zeros
2. Milestone empty panels
3. Milestone count display
4. Temporal chart cleanup
5. Color consistency
