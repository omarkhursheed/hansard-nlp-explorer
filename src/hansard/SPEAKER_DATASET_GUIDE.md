# Hansard Speaker-Attributed Debate Dataset

## Overview

This dataset transforms raw Hansard parliamentary debate transcripts into structured conversations where each speaker's contribution is clearly identified and preserved in sequential order. This enables analysis of who said what, when they said it, and how parliamentary discussions unfold.

## Dataset Structure

The dataset is available in three formats:

1. **JSON** - Complete conversations with full text
2. **Parquet (metadata)** - Debate-level information for quick filtering
3. **Parquet (turns)** - Individual speaker contributions for analysis

## Example 1: Question & Answer Session

Here's a complete debate about oats pricing in Scotland from March 21, 1950:

```json
{
  "debate_id": "1950-mar-21_07_oats-price",
  "date": "21 March 1950",
  "chamber": "Commons",
  "topic": "Oats (Price)",
  "total_speakers": 4,
  "total_turns": 7,
  "conversation": [
    {
      "turn": 1,
      "speaker": "Mr. Boothby",
      "text": "asked the Secretary of State for Scotland whether he will now place oats, permanently, on the same footing as other cereal crops in respect to the guaranteed price, owing to the fact that oats are the principal cereal crop in the North of Scotland.",
      "word_count": 44
    },
    {
      "turn": 2,
      "speaker": "The Joint Under-Secretary of State for Scotland (Mr. Thomas Fraser)",
      "text": "An undertaking has already been given that oats will be treated on the same footing as other cereal crops as regards guaranteed price and assured market up to and including the 1951 crop. It is impossible at this date to give an undertaking beyond that date.",
      "word_count": 46
    },
    {
      "turn": 3,
      "speaker": "Mr. Boothby",
      "text": "Does the hon. Gentleman realise that agriculture in Scotland will never be on a sound basis unless the main cereal crop is put permanently upon the same footing as other cereals?",
      "word_count": 31
    },
    {
      "turn": 4,
      "speaker": "Mr. Fraser",
      "text": "Even the other cereal crops do not permanently enjoy a guaranteed price and assured market.",
      "word_count": 15
    },
    {
      "turn": 5,
      "speaker": "Mr. Boothby",
      "text": "Then could they not be put on the same footing, and for much longer than at present?",
      "word_count": 17
    },
    {
      "turn": 6,
      "speaker": "Mr. John MacLeod",
      "text": "Could the Under-Secretary of State say why this stigma is put on oats?",
      "word_count": 13
    },
    {
      "turn": 7,
      "speaker": "Mr. Fraser",
      "text": "There is no stigma on oats at all.",
      "word_count": 8
    }
  ]
}
```

### What This Shows

- **Mr. Boothby** asks the initial question (turn 1)
- **Mr. Fraser** (government representative) responds (turn 2)
- **Mr. Boothby** presses with follow-up questions (turns 3, 5)
- **Mr. John MacLeod** joins with his own question (turn 6)
- The conversation shows a typical Q&A pattern with opposition members challenging government policy

## Example 2: War Damage Compensation Debate

A more complex multi-party discussion from May 22, 1950:

```json
{
  "debate_id": "1950-may-22_49_war-damage-compensation",
  "topic": "War Damage Compensation",
  "total_speakers": 5,
  "unique_speakers": [
    "Captain Ryder",
    "Mr. Younger",
    "Mr. Anthony Nutting",
    "Mr. W. Fletcher",
    "Mr. Godfrey Nicholson"
  ],
  "conversation": [
    {
      "turn": 1,
      "speaker": "Captain Ryder",
      "text": "asked the Secretary of State for Foreign Affairs if he is aware that British subjects living in Burma are suffering hardship and inconvenience through failure to obtain war damage compensation...",
      "word_count": 45
    },
    {
      "turn": 2,
      "speaker": "Mr. Younger",
      "text": "Yes, Sir, but, as announced by my right hon. Friend the President of the Board of Trade, His Majesty's Government have set up a scheme for limited ex gratia payments...",
      "word_count": 82
    },
    {
      "turn": 3,
      "speaker": "Captain Ryder",
      "text": "Is it not a fact that these payments are only being made to the people who actually leave Burma, and that those who remain behind are suffering acute hardship?",
      "word_count": 49
    },
    {
      "turn": 4,
      "speaker": "Mr. Younger",
      "text": "This is a very difficult problem. His Majesty's Government in the United Kingdom cannot admit liability for payments of this kind...",
      "word_count": 67
    },
    {
      "turn": 5,
      "speaker": "Mr. Anthony Nutting",
      "text": "Will the hon. Gentleman take steps to hold the Burma Prime Minister to the pledge he gave, as reported in the 'Observer' yesterday?",
      "word_count": 30
    }
    // ... continues for 10 total turns
  ]
}
```

### Conversation Dynamics

- **Captain Ryder** initiates and drives the discussion (2 turns)
- **Mr. Younger** responds as government representative (5 turns - most active)
- **Three other MPs** jump in with related questions
- Shows typical pattern: opposition questioning, government defending

## Example 3: Procedural Content

Not all entries are debates. Some are procedural:

```json
{
  "debate_id": "1950-mar-1_03_prayers",
  "topic": "Prayers",
  "total_speakers": 0,
  "conversation": [
    {
      "turn": 1,
      "speaker": "PROCEDURAL",
      "text": "Prayers were offered by the Chaplain.",
      "word_count": 6
    }
  ]
}
```

## Dataset Statistics (100 Sample Debates from 1950)

### Overall Composition
- **Total debates**: 100
- **Total conversation turns**: 624
- **Total words**: 105,252
- **Unique speakers**: 214

### Debate Types
- **Procedural** (0 speakers): 40%
- **Statements** (1 speaker): 5%
- **Brief exchanges** (2-3 speakers): 9%
- **Q&A sessions** (3-6 speakers): 35%
- **Full debates** (7+ speakers): 11%

### Most Active Speakers
1. **Mr. McNeil**: 15 turns
2. **Lord Lucas of Chilworth**: 15 turns
3. **Viscount Hall**: 14 turns
4. **Mr. Churchill**: 12 turns
5. **Mr. Strauss**: 11 turns

### Speaking Patterns
- **Average words per turn**: 169 words
- **Procedural text**: 116 words per turn
- **Speaker contributions**: 186 words per turn

## Data Fields Explained

### Debate Level
- `debate_id`: Unique identifier (year-month-day-topic)
- `date`: Date of the debate
- `chamber`: Commons or Lords
- `topic`: Subject being discussed
- `hansard_reference`: Official Hansard citation
- `total_speakers`: Number of unique speakers
- `total_turns`: Number of speaking turns
- `unique_speakers`: List of all speakers

### Turn Level
- `turn`: Sequential order (1, 2, 3...)
- `speaker`: Name of the person speaking
- `text`: Exactly what they said
- `word_count`: Length of contribution

## Use Cases for Gender Analysis

This dataset structure enables several types of gender-based analysis:

### 1. Speaking Time Analysis
```python
# Example: Calculate total words by speaker
speaker_words = {}
for turn in debate['conversation']:
    speaker = turn['speaker']
    speaker_words[speaker] = speaker_words.get(speaker, 0) + turn['word_count']
```

### 2. Turn-Taking Patterns
```python
# Example: Count how often each speaker speaks
from collections import Counter
speakers = [turn['speaker'] for turn in debate['conversation']]
turn_counts = Counter(speakers)
```

### 3. Interaction Patterns
```python
# Example: Track who responds to whom
for i in range(1, len(debate['conversation'])):
    previous_speaker = debate['conversation'][i-1]['speaker']
    current_speaker = debate['conversation'][i]['speaker']
    print(f"{current_speaker} responds to {previous_speaker}")
```

### 4. Topic Engagement
Once gender is added, you can analyze:
- Which topics do different genders engage with?
- How does speaking time vary by topic and gender?
- Do interaction patterns differ in mixed-gender debates?

## File Formats

### JSON Format
Best for:
- Full text analysis
- Preserving complete conversation structure
- Natural language processing tasks

### Parquet Format (Metadata)
Best for:
- Quick filtering by date, topic, or chamber
- Statistical summaries
- Joining with other datasets

### Parquet Format (Turns)
Best for:
- Speaker-level analysis
- Word count statistics
- Network analysis of interactions

## How to Load the Data

### Python - JSON
```python
import json

with open('debates_1950_sample_100.json', 'r') as f:
    data = json.load(f)
    
debates = data['debates']
for debate in debates:
    print(f"Topic: {debate['topic']}")
    print(f"Speakers: {debate['unique_speakers']}")
```

### Python - Parquet
```python
import pandas as pd

# Load metadata
metadata = pd.read_parquet('debate_metadata_1950_sample.parquet')

# Load all conversation turns
turns = pd.read_parquet('conversation_turns_1950_sample.parquet')

# Filter to specific debate
debate_turns = turns[turns['debate_id'] == '1950-mar-21_07_oats-price']
```

## Next Steps

1. **Add Gender Annotations**: Map speaker names to gender
2. **Scale Up**: Process more years for temporal analysis
3. **Analyze Patterns**: Study speaking time, interruptions, topic engagement
4. **Network Analysis**: Map who responds to whom
5. **Language Analysis**: Compare linguistic styles by gender

## Notes on Data Quality

- **Coverage**: Extracted from official Hansard HTML archives
- **Accuracy**: Speaker names preserved exactly as in source
- **Completeness**: All speakers and turns captured
- **Procedural text**: Marked separately with "PROCEDURAL" speaker

## Contact

Created using the Hansard NLP Explorer toolkit.
Dataset prepared for gender analysis research on parliamentary discourse.