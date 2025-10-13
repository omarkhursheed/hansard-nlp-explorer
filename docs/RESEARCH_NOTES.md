# Research Notes - Women's Suffrage Debate Analysis

**Source:** Ashiqur KhudaBukhsh suggestion (Oct 12, 2025)

## Research Questions

### Phase 1: Identify Right to Vote Discussions

**Objective:** Find all debates related to women's suffrage (1803-1928)

**Approach:**
```python
# Search for suffrage-related debates
keywords = ['suffrage', 'franchise', 'women vote', 'female vote',
            'electoral reform', 'representation of the people']

# Focus periods:
# - Pre-1918: Suffrage movement debates
# - 1918: Partial suffrage passage
# - 1918-1928: Arguments for/against full suffrage
# - 1928: Equal Franchise Act
```

**Datasets to use:**
- `--dataset gender-debates` (debate-level)
- `--dataset gender` (speech-level for individual arguments)
- Milestone analysis: 1918 and 1928

---

### Phase 2: Extract Arguments (For/Against)

**Objective:** Use LLM pipeline to extract reasoning on both sides

**Method:**
```python
# For each suffrage debate:
1. Identify speaker position (for/against)
2. Extract arguments using LLM:
   - Prompt: "Extract the key argument this speaker makes
              for/against women's suffrage"
   - Output: Categorized arguments

3. Classify arguments:
   For:
   - Democratic representation
   - Equal rights
   - Women's contribution (war effort, etc.)
   - Taxation without representation

   Against:
   - Traditional gender roles
   - Domestic sphere argument
   - Gradual change needed
   - Property/class concerns
```

**LLM Pipeline:**
- Use lightweight model (GPT-3.5 or similar)
- Batch process speeches
- Store: speaker, position, argument, category

---

### Phase 3: Identify Debaters on Both Sides

**Objective:** Create sets of pro-suffrage vs anti-suffrage MPs

**Analysis:**
```python
# Group speakers by position
pro_suffrage_mps = set()   # MPs who voted/spoke FOR
anti_suffrage_mps = set()  # MPs who voted/spoke AGAINST

# Track:
- Speaker name
- Party affiliation
- Constituency
- Gender (if applicable)
- Frequency of speeches on topic
- Consistency of position
```

**Output:**
- `suffrage_debaters_for.csv`
- `suffrage_debaters_against.csv`
- Network graph of alliances

---

### Phase 4: Longitudinal Analysis (Key Innovation!)

**Objective:** Did opponents keep reminding Parliament "I told you so"?

**Research Question:**
> "After women's suffrage passed, did MPs who opposed it continue to reference their objections as prophecies that came true?"

**Method:**

**Step 1: Identify post-1928 references**
```python
# Search post-1928 debates for phrases like:
- "as I predicted"
- "I warned"
- "as we feared"
- "the consequences of" + suffrage terms
- "since women got the vote"
- References to 1918/1928 debates
```

**Step 2: Link to original opponents**
```python
# For each reference:
1. Identify speaker
2. Check if they were in anti_suffrage_mps set
3. Extract what they claim happened
4. Verify if it actually happened
```

**Step 3: Categorize "I told you so" arguments**
```python
Categories:
- Political consequences (party changes, voting patterns)
- Social changes (family structure, women's roles)
- Economic impacts
- Parliamentary procedure changes
- Unintended consequences
```

**Step 4: Fact-check claims**
```python
# Compare claimed consequences vs reality:
- Did they accurately predict outcomes?
- Were consequences positive/negative?
- Were warnings exaggerated?
```

---

## Implementation Plan

### Technical Approach

**1. Data Preparation (Week 1)**
```bash
# Extract suffrage debates
python corpus_analysis.py --dataset gender \
    --years 1900-1930 --filtering ultra \
    --keywords "suffrage,franchise,women vote"

# Create suffrage-specific dataset
```

**2. Argument Extraction (Week 2)**
```python
# LLM pipeline
for debate in suffrage_debates:
    for speech in debate.speeches:
        prompt = f"""
        Context: UK Parliament debate on women's suffrage, {speech.year}
        Speaker: {speech.speaker}
        Text: {speech.text}

        Task: Extract the speaker's argument about women's suffrage.
        1. Position: [for/against/neutral]
        2. Main argument: [1-2 sentences]
        3. Category: [democratic/social/economic/other]
        """

        result = llm.process(prompt)
        # Store structured output
```

**3. Network Analysis (Week 3)**
- Build bipartite graph: MPs ↔ Arguments
- Identify argument clusters
- Track position consistency
- Visualize alliances

**4. Longitudinal Tracking (Week 4)**
```python
# Search post-1928 speeches
for year in range(1928, 1950):
    for speech in speeches_by_year[year]:
        if speech.speaker in anti_suffrage_mps:
            # Check for references to suffrage outcomes
            if contains_retrospective_language(speech.text):
                extract_claimed_consequence(speech)
                link_to_original_argument(speech)
```

---

## Expected Outputs

### Deliverables

1. **Argument Database**
   - CSV with all extracted arguments
   - Categories and speakers
   - Time series of argument evolution

2. **Debater Networks**
   - Pro/anti suffrage MP lists
   - Alliance visualizations
   - Position consistency metrics

3. **Longitudinal Analysis**
   - "I told you so" speech collection
   - Fact-checked claims
   - Temporal pattern of retrospective references

4. **Research Paper Sections**
   - Methods: LLM-assisted argument extraction
   - Results: Argument categorization
   - Novel finding: Long-term retrospective references
   - Discussion: Accuracy of predictions

---

## Novel Contribution

**What makes this unique:**

Most suffrage research focuses on:
- Contemporary debates
- Voting patterns
- Immediate outcomes

**This research adds:**
- **Longitudinal perspective**: What happened AFTER
- **Opponent tracking**: Did warnings come true?
- **Retrospective analysis**: Self-referential political discourse
- **LLM methodology**: Scalable argument extraction

**Potential findings:**
- Opponents DID/DIDN'T keep referencing their warnings
- Claims were accurate/exaggerated
- Pattern of "I told you so" politics
- Evolution of political discourse post-suffrage

---

## Data We Have (Ready to Use)

**Datasets:**
- `gender_analysis_enhanced/` - All debates with speaker gender
- `derived/gender_speeches/` - 2M individual speeches
- Milestone analyses: 1918 and 1928 already analyzed

**Time periods covered:**
- Pre-1918: Build-up to partial suffrage
- 1918: Partial suffrage debates
- 1918-1928: Arguments for full suffrage
- 1928: Equal Franchise Act
- 1928-1950: Post-suffrage outcomes

**We have 200+ years of data ready for this analysis!**

---

## Next Steps

1. **Refine search terms** - Pilot with small sample
2. **Test LLM pipeline** - Extract arguments from 10 speeches
3. **Build debater database** - Identify key figures
4. **Implement longitudinal search** - Find retrospective references
5. **Write up methodology** - Document approach

**This could be a high-impact paper!**

---

Last updated: October 12, 2025
