#!/usr/bin/env python3
"""
Test the debate extractor on different types of debates.
"""

import json
from pathlib import Path
from extract_speaker_debates import SpeakerDebateExtractor


def test_multiple_debate_types():
    """Test extraction on different debate types."""
    extractor = SpeakerDebateExtractor()
    
    # Find different types of debates
    base_path = Path("/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/data/hansard/1950")
    
    test_files = [
        # Procedural (preamble)
        base_path / "may/12_00_preamble.html.gz",
        # Single speaker (likely a statement)
        base_path / "may/15_79_new-clause-repeals.html.gz",
        # Q&A session (we already tested)
        base_path / "may/22_49_war-damage-compensation.html.gz",
        # Larger debate with many speakers
        base_path / "may/23_93_clause-13-rights-conferred-by-act-to-be.html.gz",
        # Another procedural
        base_path / "mar/1_03_prayers.html.gz"
    ]
    
    results = []
    
    for file_path in test_files:
        if file_path.exists():
            print(f"\nProcessing: {file_path.name}")
            result = extractor.extract_debate(file_path)
            
            if 'error' not in result:
                print(f"  Topic: {result['topic']}")
                print(f"  Speakers: {result['total_speakers']}")
                print(f"  Turns: {result['total_turns']}")
                
                # Show speaker distribution
                if result['conversation']:
                    from collections import Counter
                    speakers = Counter([t['speaker'] for t in result['conversation']])
                    print(f"  Speaker distribution: {dict(speakers)}")
                
                results.append({
                    'file': file_path.name,
                    'topic': result['topic'],
                    'speakers': result['total_speakers'],
                    'turns': result['total_turns'],
                    'type': classify_debate_type(result)
                })
            else:
                print(f"  Error: {result['error']}")
        else:
            print(f"\nFile not found: {file_path}")
    
    # Save summary
    with open('debate_types_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Summary of debate types:")
    for r in results:
        print(f"{r['type']:15} | {r['speakers']:2} speakers | {r['turns']:3} turns | {r['topic'][:40]}")
    
    return results


def classify_debate_type(debate_record):
    """Classify the type of debate based on its characteristics."""
    speakers = debate_record['total_speakers']
    turns = debate_record['total_turns']
    topic = debate_record['topic'].lower()
    
    if speakers == 0:
        return "PROCEDURAL"
    elif speakers == 1:
        return "STATEMENT"
    elif speakers == 2 and turns < 5:
        return "BRIEF_EXCHANGE"
    elif "question" in topic or turns <= 10:
        return "Q&A"
    else:
        return "FULL_DEBATE"


if __name__ == "__main__":
    test_multiple_debate_types()