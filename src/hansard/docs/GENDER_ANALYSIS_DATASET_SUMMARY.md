# Gender Analysis Dataset Summary

## Dataset Creation Complete

Successfully created filtered parliamentary debate datasets for gender analysis, containing only debates with at least one confirmed MP match.

## Key Statistics

### Overall Coverage
- **Total debates with confirmed MPs**: 8,082 (from 23,090 processed)
- **Debates with female participation**: 1,378 (17.1%)
- **Average confirmed MPs per debate**: 4.8

### Temporal Trends in Female Participation
| Year | Debates with Female MPs | Percentage | Notable Female MPs |
|------|-------------------------|------------|-------------------|
| 1920 | 10 / 1,335 | 0.7% | Nancy Astor, Leslie Wilson* |
| 1950 | 87 / 1,658 | 5.2% | Florence Horsbrugh, Irene Ward, Patricia Hornsby-Smith |
| 1980 | 373 / 2,893 | 12.9% | Margaret Thatcher (PM), Shirley Williams, Betty Boothroyd |
| 2000 | 908 / 2,196 | 41.3% | 95+ female MPs including Cabinet members |

*Note: Leslie Wilson was incorrectly identified as female - this is a known data quality issue

### Gender Distribution in Speaking
| Year | Female Speakers | Male Speakers | Female % |
|------|-----------------|---------------|----------|
| 1920 | 11 | 3,901 | 0.3% |
| 1950 | 92 | 2,707 | 3.3% |
| 1980 | 572 | 17,412 | 3.2% |
| 2000 | 1,677 | 12,430 | 11.9% |

## Dataset Structure

### 1. Main Debates Dataset
**File**: `gender_analysis_data/all_debates_with_confirmed_mps.parquet`

Contains 8,082 debates with:
- Debate metadata (ID, date, title, topic)
- Speaker counts by gender
- Female MP names list
- Word counts

### 2. Year-Specific Datasets
- `debates_1920_with_mps.parquet` - 1,335 debates
- `debates_1950_with_mps.parquet` - 1,658 debates
- `debates_1980_with_mps.parquet` - 2,893 debates
- `debates_2000_with_mps.parquet` - 2,196 debates

### 3. Turn-wise Data (Planned)
Structure for detailed conversation analysis including:
- Speaker transitions
- Word counts per turn
- Interruption detection
- Gender of previous/next speaker

## Notable Female MPs Identified

### Early Period (1920s)
- **Nancy Astor**: First woman to sit as an MP (1919-1945)

### Mid-Century (1950s)
- **Florence Horsbrugh**: First woman in Conservative Cabinet
- **Irene Ward**: Long-serving Conservative MP
- **Dorothy Rees**: Labour MP for Barry

### Thatcher Era (1980s)
- **Margaret Thatcher**: First female Prime Minister (1979-1990)
- **Shirley Williams**: SDP founder, Cabinet minister
- **Betty Boothroyd**: Later first female Speaker
- **Jo Richardson**: Labour women's rights campaigner

### Modern Era (2000s)
Over 95 female MPs including:
- **Harriet Harman**: Deputy Labour Leader
- **Theresa May**: Future Prime Minister
- **Clare Short**: International Development Secretary
- **Mo Mowlam**: Northern Ireland Secretary
- **Patricia Hewitt**: Health Secretary

## Use Cases for Analysis

### 1. High-Level Analysis
```python
# Female participation over time
df.groupby('year')['has_female'].mean()

# Debates with mixed gender participation
mixed = df[df['female_mps'] > 0] & df[df['male_mps'] > 0]
```

### 2. Speaker Proportion Analysis
```python
# What proportion of confirmed speakers are female?
df['female_proportion'] = df['female_mps'] / df['confirmed_mps']
```

### 3. Topic Analysis
```python
# Which topics have highest female participation?
df[df['has_female']].groupby('topic').size()
```

### 4. Temporal Patterns
```python
# Monthly patterns of female participation
df['month'] = pd.to_datetime(df['reference_date']).dt.month
df.groupby(['year', 'month'])['has_female'].mean()
```

## Data Quality Considerations

### Strengths
- **High confidence matches only**: Uses 0.7+ confidence threshold
- **Verified MP database**: Gender from authoritative sources
- **Temporal validation**: Prevents impossible matches
- **Multiple matching strategies**: Title, constituency, temporal

### Limitations
- **Coverage**: Only 35% of debates have confirmed MPs
- **Ambiguous speakers**: 25.5% of speakers ambiguous (not included)
- **Historical bias**: Better coverage in later years
- **Name errors**: Some gender misattribution (e.g., Leslie Wilson)

### Recommended Filters
```python
# For highest quality analysis
high_quality = df[
    (df['confirmed_mps'] >= 3) &  # Multiple confirmed speakers
    (df['year'] >= 1945) &         # Post-war (better records)
    (df['word_count'] >= 1000)     # Substantial debates
]
```

## Next Steps

### Planned Enhancements
1. **Turn-wise analysis**: Extract individual speaking turns
2. **Interruption detection**: Identify short interjections
3. **Topic modeling**: Analyze topics by gender participation
4. **Network analysis**: Who speaks after whom?
5. **Sentiment analysis**: Tone differences by gender

### Research Questions
1. How has female participation changed over time?
2. Which topics see highest female engagement?
3. Are female MPs interrupted more frequently?
4. Do speaking patterns differ by gender?
5. How do mixed-gender debates differ from single-gender?

## Technical Notes

### Matching Methodology
- Uses `CorrectedMPMatcher` with verified dates
- Confidence threshold: 0.7
- Match types: temporal_unique, title, constituency
- Excludes ambiguous matches from gender attribution

### Performance
- Processing time: ~30 seconds for 4 sample years
- Full dataset would take ~2 hours for all 201 years
- Optimized for accuracy over speed

### File Formats
- Parquet for efficient storage and fast loading
- JSON for metadata
- Compatible with pandas, R, and other analysis tools

## Citation

If using this dataset, please reference:
- Original Hansard data: UK Parliament
- MP gender data: `house_members_gendered_updated.parquet`
- Matching methodology: CorrectedMPMatcher (2025-09-20)