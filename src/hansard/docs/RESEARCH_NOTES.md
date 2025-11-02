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

## Phase 5: Toxicity & Social Bias Analysis (Extended Research)

**Objective:** Connect suffrage opposition arguments to broader patterns of discrimination

### 5.1 Tie to LLM Toxicity Corpora

**Link to existing frameworks:**
- **Sap et al. Social Bias Frames** - Identify biased framing
- **Perspective API toxicity** - Measure hostile language
- **RealToxicityPrompts** - Compare to known toxic patterns

**Analysis:**
```python
# For each anti-suffrage argument:
1. Extract framing (Sap et al. framework)
   - Offensive/toxic language
   - Stereotyping
   - Group-based generalizations

2. Compare to modern toxicity benchmarks
   - How would these arguments score on Perspective API?
   - Are they consistent with known bias patterns?

3. Track evolution of language
   - Did toxic framing decrease over time?
   - Did arguments become more "polite" but still discriminatory?
```

**Expected findings:**
- Anti-suffrage arguments map to social bias frames
- Similar patterns in other discrimination contexts
- Language evolution (explicit → implicit bias)

---

### 5.2 Beyond Voting - Continuous Discrimination

**Objective:** Investigate if anti-suffrage themes were repurposed for ongoing discrimination

**Research Question:**
> "Did the same arguments used against suffrage get recycled to oppose other women's rights?"

**Themes to track across time:**

**1. Traditional Roles Argument**
```
Pre-1928: "Women's place is in the home, not politics"
Post-1928: Look for same framing applied to:
- Women in workforce debates
- Equal pay discussions
- Maternity leave policies
- Leadership positions
```

**2. Competence/Capacity Argument**
```
Pre-1928: "Women lack political judgment"
Post-1928: Check if repurposed for:
- Women in Parliament debates
- Cabinet positions
- Prime Minister discussions
- Professional roles
```

**3. Protection/Paternalism**
```
Pre-1928: "Voting would harm women's delicate nature"
Post-1928: Watch for in:
- Employment law debates
- Night work restrictions
- Military service discussions
```

**4. Slippery Slope Arguments**
```
Pre-1928: "Suffrage will destroy family/society"
Post-1928: Look for:
- Same catastrophizing about other reforms
- Dire predictions that didn't come true
- Pattern of exaggerated consequences
```

---

### Implementation

**Step 1: Build Argument Taxonomy**
```python
# Extract all anti-suffrage argument types
argument_types = {
    'traditional_roles': [...],
    'competence': [...],
    'protection': [...],
    'slippery_slope': [...],
    'property_class': [...],
    'gradual_change': [...]
}
```

**Step 2: Track Arguments Across Issues**
```python
# Search post-1928 debates on women's issues
issues = [
    'equal_pay',
    'employment_rights',
    'parliamentary_participation',
    'cabinet_positions',
    'maternity_rights',
    'professional_barriers'
]

# For each issue, check if same argument frames appear
```

**Step 3: Measure Argument Recycling**
```python
# Calculate overlap:
recycling_rate = (arguments_reused / total_arguments) * 100

# Track temporal patterns:
- Do same MPs recycle arguments?
- Do new MPs adopt old frames?
- Does framing evolve but core argument persists?
```

**Step 4: Connect to Bias Frameworks**
```python
# Map to Sap et al. Social Bias Frames
- Identify bias types (intent, implications, target)
- Compare suffrage bias to other group-based discrimination
- Show consistency across contexts (gender, race, class)
```

---

## Novel Contributions

**This research would:**

1. **Longitudinal Political Discourse Analysis**
   - Track arguments across 100+ years
   - Show recycling of discriminatory frames

2. **LLM-Assisted Historical Analysis**
   - Scale argument extraction to 2M speeches
   - Systematic categorization
   - Fact-checking historical claims

3. **Bias Framework Application**
   - Apply modern bias detection to historical data
   - Show continuity of discriminatory patterns
   - Connect to contemporary bias research

4. **Practical Implications**
   - Identify recycled discrimination arguments
   - Provide evidence for ongoing bias patterns
   - Inform current policy debates

---

## Datasets We Have (Immediately Usable)

**Suffrage Period:**
- 1900-1930: 450K+ speeches
- Milestone analyses: 1918 and 1928 already processed
- Gender-tagged speakers

**Post-Suffrage Tracking:**
- 1930-2005: 1.5M+ speeches
- All with gender tags
- Can search for women's rights debates

**Perfect for this research!**

---

## Potential Publications

**Paper 1:** "I Told You So: Longitudinal Analysis of Suffrage Opposition in UK Parliament"

**Paper 2:** "Recycling Discrimination: How Anti-Suffrage Arguments Were Repurposed for Ongoing Gender Bias"

**Paper 3:** "LLM-Assisted Historical Discourse Analysis at Scale: 2 Million Speeches on Women's Rights"

---

## Collaborations

**Relevant to:**
- NLP + Social Science
- Computational Social Science
- Political Science
- Gender Studies
- Historical Linguistics
- Bias & Fairness in NLP

**Potential venues:**
- ACL (NLP + Social Good track)
- EMNLP (Computational Social Science)
- Political Science journals
- Gender Studies journals

---

Last updated: October 12, 2025
