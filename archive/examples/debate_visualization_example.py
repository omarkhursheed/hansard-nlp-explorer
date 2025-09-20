#!/usr/bin/env python3
"""
Create a visual representation of how debates flow in the dataset.
"""

import json
from pathlib import Path


def create_visual_debate_example():
    """Create a text-based visualization of debate flow."""
    
    # Load the data
    with open('debate_datasets/debates_1950_sample_100.json', 'r') as f:
        data = json.load(f)
    
    # Find the war damage compensation debate
    war_damage_debate = None
    for debate in data['debates']:
        if 'war-damage-compensation' in debate['debate_id']:
            war_damage_debate = debate
            break
    
    if not war_damage_debate:
        print("Could not find war damage debate")
        return
    
    # Create visual representation
    output = []
    output.append("=" * 80)
    output.append("VISUAL EXAMPLE: How Parliamentary Debates Flow")
    output.append("=" * 80)
    output.append("")
    output.append(f"Debate: {war_damage_debate['topic']}")
    output.append(f"Date: {war_damage_debate['date']}")
    output.append(f"Chamber: {war_damage_debate['chamber']}")
    output.append("")
    output.append("THE CONVERSATION FLOW:")
    output.append("-" * 80)
    
    # Track speaker positions
    speakers_seen = []
    
    for turn in war_damage_debate['conversation']:
        speaker = turn['speaker']
        text = turn['text']
        words = turn['word_count']
        
        # Truncate text for display
        if len(text) > 100:
            text_display = text[:97] + "..."
        else:
            text_display = text
        
        # Determine if this is a new speaker
        if speaker not in speakers_seen:
            speakers_seen.append(speaker)
            speaker_label = f"[NEW] {speaker}"
        else:
            speaker_label = speaker
        
        # Create visual indent based on turn number
        if turn['turn'] == 1:
            indent = ""
            symbol = "🎯"  # Initial question
        elif speaker == "Mr. Younger":
            indent = "    "
            symbol = "📢"  # Government response
        else:
            indent = "        "
            symbol = "❓"  # Follow-up question
        
        output.append("")
        output.append(f"{indent}{symbol} Turn {turn['turn']}: {speaker_label}")
        output.append(f"{indent}   ({words} words)")
        output.append(f"{indent}   \"{text_display}\"")
    
    output.append("")
    output.append("-" * 80)
    output.append("CONVERSATION STATISTICS:")
    output.append(f"• Total speakers: {war_damage_debate['total_speakers']}")
    output.append(f"• Total turns: {war_damage_debate['total_turns']}")
    output.append(f"• Total words: {sum(t['word_count'] for t in war_damage_debate['conversation'])}")
    output.append("")
    output.append("SPEAKER PARTICIPATION:")
    
    # Count turns per speaker
    from collections import Counter
    speaker_turns = Counter([t['speaker'] for t in war_damage_debate['conversation']])
    speaker_words = {}
    for turn in war_damage_debate['conversation']:
        speaker = turn['speaker']
        speaker_words[speaker] = speaker_words.get(speaker, 0) + turn['word_count']
    
    for speaker, turns in speaker_turns.most_common():
        words = speaker_words[speaker]
        avg_words = words // turns
        output.append(f"• {speaker:30} : {turns} turns, {words} words (avg {avg_words}/turn)")
    
    output.append("")
    output.append("=" * 80)
    output.append("PATTERN ANALYSIS:")
    output.append("")
    output.append("This is a typical Question & Answer session showing:")
    output.append("1. Opposition MP asks initial question (Captain Ryder)")
    output.append("2. Government minister responds (Mr. Younger)")
    output.append("3. Original questioner follows up")
    output.append("4. Other MPs join with related questions")
    output.append("5. Minister provides brief responses to each")
    output.append("")
    output.append("This pattern repeats across many debates, making it ideal for analyzing:")
    output.append("• Who asks questions vs who answers")
    output.append("• Length of questions vs answers")
    output.append("• How many follow-ups occur")
    output.append("• Which topics generate most discussion")
    
    # Save to file
    output_text = "\n".join(output)
    
    with open('debate_flow_visualization.txt', 'w') as f:
        f.write(output_text)
    
    print(output_text)
    
    return output_text


def create_simplified_example():
    """Create a simplified example for easy understanding."""
    
    example = {
        "what_this_is": "A parliamentary debate broken into speaking turns",
        "simple_example": {
            "debate_topic": "Should we increase funding for schools?",
            "how_it_works": [
                {"turn": 1, "who": "MP Smith", "says": "Why hasn't the government increased school funding?", "words": 9},
                {"turn": 2, "who": "Minister Jones", "says": "We have allocated an additional £2 billion this year.", "words": 9},
                {"turn": 3, "who": "MP Smith", "says": "But that doesn't match inflation. Schools are worse off.", "words": 10},
                {"turn": 4, "who": "MP Brown", "says": "What about rural schools specifically?", "words": 5},
                {"turn": 5, "who": "Minister Jones", "says": "Rural schools receive additional targeted support.", "words": 6}
            ]
        },
        "what_you_can_analyze": {
            "who_talks_most": "Minister Jones: 2 turns, 15 words",
            "conversation_pattern": "Question → Answer → Challenge → New Question → Answer",
            "ready_for_gender_analysis": "Add gender to each speaker, then analyze patterns"
        }
    }
    
    with open('debate_simple_example.json', 'w') as f:
        json.dump(example, f, indent=2)
    
    print("\nSimplified example saved to: debate_simple_example.json")
    
    return example


if __name__ == "__main__":
    print("Creating debate visualization...")
    create_visual_debate_example()
    print("\nVisualization saved to: debate_flow_visualization.txt")
    
    print("\nCreating simplified example...")
    create_simplified_example()