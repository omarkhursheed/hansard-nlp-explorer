#!/usr/bin/env python3
"""
Process a sample of debates to create a speaker-attributed dataset.
Saves results in both JSON and Parquet formats.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from extract_speaker_debates import SpeakerDebateExtractor
from tqdm import tqdm
import random


def process_debate_sample(year: int = 1950, sample_size: int = 100):
    """
    Process a sample of debates from a given year.
    
    Args:
        year: Year to process debates from
        sample_size: Number of debates to process
    """
    extractor = SpeakerDebateExtractor()
    
    # Find all debate files for the year
    base_path = Path(f"/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/data/hansard/{year}")
    
    all_files = []
    for month_dir in base_path.iterdir():
        if month_dir.is_dir():
            debate_files = list(month_dir.glob("*.html.gz"))
            all_files.extend(debate_files)
    
    print(f"Found {len(all_files)} total debate files for {year}")
    
    # Sample files
    if len(all_files) > sample_size:
        sample_files = random.sample(all_files, sample_size)
    else:
        sample_files = all_files
    
    print(f"Processing {len(sample_files)} files...")
    
    # Process each file
    debates = []
    errors = []
    
    for file_path in tqdm(sample_files, desc="Extracting debates"):
        result = extractor.extract_debate(file_path)
        
        if 'error' in result:
            errors.append({
                'file': str(file_path),
                'error': result['error']
            })
        else:
            debates.append(result)
    
    print(f"\nSuccessfully processed: {len(debates)} debates")
    print(f"Errors encountered: {len(errors)}")
    
    # Calculate statistics
    stats = calculate_statistics(debates)
    
    # Save results
    output_dir = Path("debate_datasets")
    output_dir.mkdir(exist_ok=True)
    
    # Save as JSON (full conversations)
    json_file = output_dir / f"debates_{year}_sample_{len(debates)}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'metadata': {
                'year': year,
                'total_debates': len(debates),
                'statistics': stats
            },
            'debates': debates
        }, f, indent=2)
    
    print(f"\nSaved JSON to: {json_file}")
    
    # Create flattened dataframe for Parquet
    flattened_data = flatten_for_dataframe(debates)
    
    # Save debate metadata
    debate_df = pd.DataFrame([
        {
            'debate_id': d['debate_id'],
            'date': d['date'],
            'chamber': d['chamber'],
            'topic': d['topic'],
            'hansard_reference': d['hansard_reference'],
            'total_speakers': d['total_speakers'],
            'total_turns': d['total_turns'],
            'file_path': d['file_path']
        }
        for d in debates
    ])
    
    debate_meta_file = output_dir / f"debate_metadata_{year}_sample.parquet"
    debate_df.to_parquet(debate_meta_file)
    print(f"Saved debate metadata to: {debate_meta_file}")
    
    # Save conversation turns
    turns_df = pd.DataFrame(flattened_data)
    turns_file = output_dir / f"conversation_turns_{year}_sample.parquet"
    turns_df.to_parquet(turns_file)
    print(f"Saved conversation turns to: {turns_file}")
    
    # Save errors if any
    if errors:
        error_file = output_dir / f"extraction_errors_{year}.json"
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"Saved errors to: {error_file}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return debates, stats


def flatten_for_dataframe(debates: List[Dict]) -> List[Dict]:
    """Flatten debates into individual conversation turns for analysis."""
    flattened = []
    
    for debate in debates:
        for turn in debate['conversation']:
            flattened.append({
                'debate_id': debate['debate_id'],
                'date': debate['date'],
                'chamber': debate['chamber'],
                'topic': debate['topic'],
                'turn_number': turn['turn'],
                'speaker': turn['speaker'],
                'text': turn['text'],
                'word_count': turn['word_count']
            })
    
    return flattened


def calculate_statistics(debates: List[Dict]) -> Dict:
    """Calculate statistics about the debate dataset."""
    
    # Basic counts
    total_debates = len(debates)
    total_turns = sum(d['total_turns'] for d in debates)
    total_words = sum(
        sum(t['word_count'] for t in d['conversation'])
        for d in debates
    )
    
    # Speaker statistics
    all_speakers = set()
    for d in debates:
        all_speakers.update(d['unique_speakers'])
    
    # Debate type distribution
    debate_types = {
        'procedural': 0,
        'statement': 0,
        'brief_exchange': 0,
        'q_and_a': 0,
        'full_debate': 0
    }
    
    for d in debates:
        speakers = d['total_speakers']
        turns = d['total_turns']
        
        if speakers == 0:
            debate_types['procedural'] += 1
        elif speakers == 1:
            debate_types['statement'] += 1
        elif speakers == 2 and turns < 5:
            debate_types['brief_exchange'] += 1
        elif turns <= 10:
            debate_types['q_and_a'] += 1
        else:
            debate_types['full_debate'] += 1
    
    # Average statistics
    avg_speakers = sum(d['total_speakers'] for d in debates) / total_debates if total_debates > 0 else 0
    avg_turns = total_turns / total_debates if total_debates > 0 else 0
    avg_words = total_words / total_turns if total_turns > 0 else 0
    
    return {
        'total_debates': total_debates,
        'total_conversation_turns': total_turns,
        'total_words': total_words,
        'unique_speakers': len(all_speakers),
        'average_speakers_per_debate': round(avg_speakers, 2),
        'average_turns_per_debate': round(avg_turns, 2),
        'average_words_per_turn': round(avg_words, 2),
        'debate_type_distribution': debate_types
    }


if __name__ == "__main__":
    # Process sample from 1950
    debates, stats = process_debate_sample(year=1950, sample_size=100)