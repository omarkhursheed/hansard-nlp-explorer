# Dataset Structure Plan for Gender Analysis

## Current Data Structure

### Existing Files
1. **Debates metadata**: `data/processed_fixed/metadata/debates_{year}.parquet`
   - Contains: file_path, title, speakers array, word_count, etc.

2. **Speaker metadata**: `data/processed_fixed/metadata/speakers_{year}.parquet`
   - Contains: file_path, speaker_name, reference_date, chamber

3. **Debate content**: `data/processed_fixed/content/{year}/debates_{year}.jsonl`
   - Contains: full text lines including speaker transitions

4. **MP matching results**: From corrected matcher
   - Certain matches: 39.3% with gender
   - Ambiguous: 25.5% (multiple possible MPs)
   - No match: 33.2%

## Proposed New Dataset Structure

### Level 1: Filtered Debates Dataset
**File**: `gender_analysis_data/debates_with_confirmed_mps.parquet`

```python
{
    'debate_id': str,  # Unique identifier (hash of file_path + date)
    'file_path': str,
    'reference_date': datetime,
    'chamber': str,
    'title': str,
    'topic': str,

    # MP participation metrics
    'total_speakers': int,
    'confirmed_mp_count': int,  # High confidence matches only
    'female_mp_count': int,
    'male_mp_count': int,
    'ambiguous_mp_count': int,  # Multiple possible MPs
    'unmatched_count': int,  # Couldn't match to any MP

    # Gender flags
    'has_female_speaker': bool,  # At least one confirmed female
    'has_male_speaker': bool,
    'is_mixed_gender': bool,  # Both male and female confirmed
    'female_mp_names': List[str],  # List of confirmed female MPs

    # Content metrics
    'total_word_count': int,
    'total_turns': int,  # Number of speaker changes
}
```

### Level 2: Turn-wise Conversation Dataset
**File**: `gender_analysis_data/turns/{year}/debate_turns_{debate_id}.parquet`

```python
{
    'debate_id': str,
    'turn_number': int,  # Sequential within debate
    'speaker_raw': str,  # Original speaker text
    'speaker_normalized': str,  # After cleaning

    # MP matching results
    'matched_mp': str,  # Null if no match
    'match_confidence': float,
    'match_type': str,  # 'certain', 'ambiguous', 'no_match', 'procedural'
    'possible_mps': List[str],  # For ambiguous cases

    # Gender attribution
    'gender': str,  # 'M', 'F', None, 'AMBIGUOUS'
    'gender_confidence': float,  # 1.0 for certain, lower for ambiguous

    # Content analysis
    'text': str,  # Full text of this turn
    'word_count': int,
    'token_count': int,  # For LLM analysis
    'line_count': int,

    # Interaction analysis
    'is_interruption': bool,  # Short interjection
    'is_question': bool,  # Ends with ?
    'mentions_previous_speaker': bool,
    'addresses_speaker': str,  # If explicitly addresses someone

    # Position in debate
    'timestamp': str,  # If available (e.g., "12:42 p.m.")
    'previous_speaker': str,
    'next_speaker': str,
}
```

### Level 3: Aggregated Speaker Statistics
**File**: `gender_analysis_data/speaker_statistics.parquet`

```python
{
    'mp_name': str,
    'gender': str,
    'years_active': List[int],

    # Participation metrics
    'total_debates': int,
    'total_turns': int,
    'total_words': int,
    'avg_words_per_turn': float,

    # Interaction patterns
    'interruption_count': int,
    'interrupted_by_count': int,
    'questions_asked': int,
    'questions_answered': int,

    # Gender interaction metrics
    'spoke_after_female': int,
    'spoke_after_male': int,
    'interrupted_female': int,
    'interrupted_male': int,
    'interrupted_by_female': int,
    'interrupted_by_male': int,
}
```

### Level 4: Gender Analysis Metadata
**File**: `gender_analysis_data/analysis_metadata.json`

```python
{
    'creation_date': datetime,
    'mp_matcher_version': 'corrected_v1',
    'confidence_threshold': 0.7,  # Minimum confidence for "certain" match

    'statistics': {
        'total_debates_processed': int,
        'debates_with_confirmed_mps': int,
        'debates_with_female_participation': int,
        'total_turns_analyzed': int,
        'female_word_count': int,
        'male_word_count': int,
        'ambiguous_word_count': int,
    },

    'coverage_by_year': {
        year: {
            'debates': int,
            'with_confirmed_mps': int,
            'with_female': int,
            'female_mps': List[str],
        }
    },

    'data_quality_metrics': {
        'certain_speaker_rate': float,
        'ambiguous_speaker_rate': float,
        'unmatched_speaker_rate': float,
    }
}
```

## Implementation Strategy

### Phase 1: Filter and Match
1. Load all debate metadata
2. For each debate:
   - Match all speakers using corrected matcher
   - Count confirmed MPs by gender
   - Flag debates with at least one confirmed MP
   - Create filtered debates dataset

### Phase 2: Turn Extraction
1. For filtered debates:
   - Parse JSONL content
   - Identify speaker transitions
   - Extract text for each turn
   - Calculate word/token counts
   - Detect interruptions (< 50 words?)
   - Create turn-wise datasets

### Phase 3: Gender Attribution
1. For each turn:
   - Use matched MP gender if certain
   - Mark as ambiguous if multiple MPs possible
   - Calculate gender confidence scores
   - Track gender transitions between turns

### Phase 4: Analysis Features
1. Identify interaction patterns:
   - Interruptions (short turns < 50 words)
   - Questions (ends with ?)
   - Direct addresses ("Mr. Smith said...")
   - Gender of previous/next speaker

2. Calculate proportions:
   - Female vs male word counts per debate
   - Speaking time distribution
   - Interruption patterns by gender

### Phase 5: Validation
1. Quality checks:
   - Verify gender assignments
   - Check for data consistency
   - Sample manual validation
   - Generate quality report

## Usage Examples

### High-Level Analysis
```python
# Find debates with female participation
female_debates = df[df['has_female_speaker'] == True]

# Calculate female speaking proportion
female_words = df['female_word_count'].sum()
total_words = df['total_word_count'].sum()
female_proportion = female_words / total_words
```

### Granular Analysis
```python
# Load turn data for a debate
turns = pd.read_parquet(f'turns/{year}/debate_turns_{debate_id}.parquet')

# Find interruptions by gender
female_interruptions = turns[
    (turns['gender'] == 'F') &
    (turns['is_interruption'] == True)
]

# Calculate speaking patterns
gender_transitions = turns['gender'].ne(turns['gender'].shift()).sum()
```

### Temporal Analysis
```python
# Female participation over time
by_year = df.groupby('year').agg({
    'has_female_speaker': 'mean',
    'female_mp_count': 'sum',
    'total_speakers': 'sum'
})
```

## Data Quality Considerations

### Handling Ambiguity
- **Certain only**: For strict gender analysis, use only high-confidence matches
- **Include ambiguous**: For coverage analysis, include with weighted confidence
- **Probabilistic**: Assign fractional gender based on possible MPs

### Missing Data
- Mark unknown gender as None
- Track coverage rates by year
- Document limitations in metadata

### Validation Samples
- Manually verify 100 random turns per decade
- Check known female MPs (Thatcher, Castle, etc.)
- Cross-reference with historical records