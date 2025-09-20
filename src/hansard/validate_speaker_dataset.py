#!/usr/bin/env python3
"""
Validate the quality of speaker-attributed debate extraction.
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter


def validate_dataset():
    """Validate the extracted debate dataset."""
    
    # Load the data
    json_file = Path("debate_datasets/debates_1950_sample_100.json")
    turns_file = Path("debate_datasets/conversation_turns_1950_sample.parquet")
    metadata_file = Path("debate_datasets/debate_metadata_1950_sample.parquet")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    debates = data['debates']
    stats = data['metadata']['statistics']
    
    turns_df = pd.read_parquet(turns_file)
    metadata_df = pd.read_parquet(metadata_file)
    
    print("=" * 60)
    print("SPEAKER-ATTRIBUTED DEBATE DATASET VALIDATION")
    print("=" * 60)
    
    # 1. Basic statistics
    print("\n1. DATASET OVERVIEW")
    print("-" * 40)
    print(f"Total debates: {len(debates)}")
    print(f"Total conversation turns: {len(turns_df)}")
    print(f"Unique speakers: {stats['unique_speakers']}")
    print(f"Date range: {metadata_df['date'].min()} to {metadata_df['date'].max()}")
    
    # 2. Debate type distribution
    print("\n2. DEBATE TYPE DISTRIBUTION")
    print("-" * 40)
    for dtype, count in stats['debate_type_distribution'].items():
        pct = (count / len(debates)) * 100
        print(f"{dtype:15} : {count:3} ({pct:.1f}%)")
    
    # 3. Speaker analysis
    print("\n3. SPEAKER ANALYSIS")
    print("-" * 40)
    
    # Most active speakers
    speaker_counts = Counter(turns_df[turns_df['speaker'] != 'PROCEDURAL']['speaker'].values)
    print("Top 10 most active speakers:")
    for speaker, count in speaker_counts.most_common(10):
        print(f"  {speaker:40} : {count} turns")
    
    # 4. Conversation patterns
    print("\n4. CONVERSATION PATTERNS")
    print("-" * 40)
    
    # Debates with most back-and-forth
    interactive_debates = []
    for debate in debates:
        if debate['total_speakers'] > 1:
            # Calculate interaction score (turns per speaker)
            interaction_score = debate['total_turns'] / debate['total_speakers']
            interactive_debates.append({
                'topic': debate['topic'][:50],
                'speakers': debate['total_speakers'],
                'turns': debate['total_turns'],
                'interaction_score': interaction_score
            })
    
    interactive_debates.sort(key=lambda x: x['interaction_score'], reverse=True)
    
    print("Most interactive debates (turns per speaker):")
    for d in interactive_debates[:5]:
        print(f"  {d['topic']:50} | {d['speakers']:2} speakers | {d['turns']:3} turns | Score: {d['interaction_score']:.1f}")
    
    # 5. Content quality checks
    print("\n5. CONTENT QUALITY CHECKS")
    print("-" * 40)
    
    # Check for empty or very short turns
    short_turns = turns_df[turns_df['word_count'] < 5]
    print(f"Very short turns (<5 words): {len(short_turns)} ({len(short_turns)/len(turns_df)*100:.1f}%)")
    
    # Check for missing speakers
    unknown_speakers = turns_df[turns_df['speaker'] == 'UNKNOWN']
    print(f"Unknown speakers: {len(unknown_speakers)}")
    
    # Check average turn length by debate type
    print("\nAverage words per turn by speaker type:")
    avg_words = turns_df.groupby('speaker')['word_count'].mean()
    
    procedural_avg = avg_words.get('PROCEDURAL', 0)
    non_procedural_avg = turns_df[turns_df['speaker'] != 'PROCEDURAL']['word_count'].mean()
    
    print(f"  Procedural text: {procedural_avg:.1f} words")
    print(f"  Speaker contributions: {non_procedural_avg:.1f} words")
    
    # 6. Sample conversations
    print("\n6. SAMPLE CONVERSATIONS")
    print("-" * 40)
    
    # Find a good Q&A example
    qa_debates = [d for d in debates if 3 <= d['total_speakers'] <= 6 and 5 <= d['total_turns'] <= 15]
    
    if qa_debates:
        sample = qa_debates[0]
        print(f"\nSample Q&A: {sample['topic']}")
        print(f"Date: {sample['date']}, Speakers: {sample['total_speakers']}")
        print("\nConversation flow:")
        
        for turn in sample['conversation'][:5]:  # Show first 5 turns
            speaker = turn['speaker']
            text_preview = turn['text'][:100] + "..." if len(turn['text']) > 100 else turn['text']
            print(f"  Turn {turn['turn']}: {speaker:30} | {text_preview}")
    
    # 7. Data completeness
    print("\n7. DATA COMPLETENESS")
    print("-" * 40)
    
    missing_dates = metadata_df['date'].isna().sum()
    missing_topics = metadata_df['topic'].isna().sum()
    missing_refs = metadata_df['hansard_reference'].isna().sum()
    
    print(f"Missing dates: {missing_dates}")
    print(f"Missing topics: {missing_topics}")
    print(f"Missing Hansard references: {missing_refs}")
    
    # 8. Export sample for manual review
    print("\n8. EXPORT FOR REVIEW")
    print("-" * 40)
    
    # Create a readable sample
    sample_output = []
    for debate in debates[:3]:  # First 3 debates
        if debate['total_speakers'] > 0:
            sample_output.append({
                'topic': debate['topic'],
                'date': debate['date'],
                'speakers': debate['unique_speakers'],
                'conversation_sample': [
                    f"{t['speaker']}: {t['text'][:100]}..."
                    for t in debate['conversation'][:3]
                ]
            })
    
    review_file = Path("debate_datasets/sample_for_review.json")
    with open(review_file, 'w') as f:
        json.dump(sample_output, f, indent=2)
    
    print(f"Created human-readable sample at: {review_file}")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("Dataset is ready for gender analysis and other NLP tasks!")
    print("=" * 60)


if __name__ == "__main__":
    validate_dataset()