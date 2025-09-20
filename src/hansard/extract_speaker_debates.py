#!/usr/bin/env python3
"""
Extract speaker-attributed debates from Hansard HTML files.

This script processes Hansard debate files to create a structured dataset
where each speaker's contribution is preserved in conversational order.
"""

import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import re
from datetime import datetime


class SpeakerDebateExtractor:
    """Extract speaker-attributed conversations from Hansard debates."""
    
    def __init__(self):
        self.procedural_patterns = [
            r'^\s*The House',
            r'^\s*Question put',
            r'^\s*Motion agreed',
            r'^\s*The Deputy Speaker',
            r'^\s*Mr\. Speaker',
            r'^\s*Madam Speaker',
            r'^\s*The Chairman'
        ]
    
    def extract_debate(self, file_path: Path) -> Dict:
        """
        Extract a structured debate from a single HTML file.
        
        Args:
            file_path: Path to the .html.gz file
            
        Returns:
            Dictionary containing debate metadata and conversation turns
        """
        try:
            # Read and parse HTML
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                html = f.read()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract basic metadata
            title = soup.find('title')
            title_text = title.get_text() if title else ""
            
            # Extract debate topic (remove Hansard citation)
            debate_topic = self._extract_topic(title_text)
            
            # Extract date from title
            date_match = re.search(r'\d{1,2}\s+\w+\s+\d{4}', title_text)
            debate_date = date_match.group(0) if date_match else None
            
            # Determine chamber
            content_div = soup.find('div', class_='house-of-commons-sitting')
            chamber = "Commons"
            if not content_div:
                content_div = soup.find('div', class_='house-of-lords-sitting')
                chamber = "Lords"
            
            # Extract Hansard reference
            hansard_ref = None
            ref_cite = soup.find('cite', class_='section')
            if ref_cite:
                hansard_ref = ref_cite.get_text(strip=True)
            
            # Build debate ID from file path
            path_parts = file_path.parts
            year = path_parts[-3] if len(path_parts) >= 3 else "unknown"
            month = path_parts[-2] if len(path_parts) >= 2 else "unknown"
            file_name = file_path.stem.replace('.html', '')
            debate_id = f"{year}-{month}-{file_name}"
            
            # Extract conversation turns
            conversation = []
            if content_div:
                conversation = self._extract_conversation(content_div)
            
            # Calculate statistics
            speakers = list(set([turn['speaker'] for turn in conversation 
                               if turn['speaker'] != 'PROCEDURAL']))
            
            # Build the debate record
            debate_record = {
                'debate_id': debate_id,
                'file_path': str(file_path),
                'date': debate_date,
                'chamber': chamber,
                'topic': debate_topic,
                'hansard_reference': hansard_ref,
                'total_speakers': len(speakers),
                'unique_speakers': speakers,
                'total_turns': len(conversation),
                'conversation': conversation,
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            return debate_record
            
        except Exception as e:
            return {
                'debate_id': str(file_path),
                'error': str(e),
                'extraction_timestamp': datetime.now().isoformat()
            }
    
    def _extract_topic(self, title: str) -> str:
        """Extract topic by removing Hansard citation from title."""
        if not title:
            return ""
        
        # Remove (Hansard, date) pattern
        topic = re.sub(r'\s*\(Hansard[^)]*\)\s*$', '', title)
        return topic.strip()
    
    def _extract_conversation(self, content_div) -> List[Dict]:
        """Extract all speaker turns from the debate content."""
        conversation = []
        turn_number = 0
        
        # Find all member contributions
        contributions = content_div.find_all('div', class_='hentry member_contribution')
        
        for contrib in contributions:
            turn_number += 1
            
            # Extract speaker name
            speaker_cite = contrib.find('cite', class_='member')
            if speaker_cite:
                # Get text and clean it
                speaker = speaker_cite.get_text(strip=True)
                # Remove any links or extra formatting
                speaker = re.sub(r'\s+', ' ', speaker)
            else:
                speaker = "UNKNOWN"
            
            # Extract the text content
            text_parts = []
            blockquote = contrib.find('blockquote', class_='contribution_text')
            if blockquote:
                # Get all paragraphs
                paragraphs = blockquote.find_all('p')
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text:
                        text_parts.append(text)
            
            # Join all text parts
            full_text = ' '.join(text_parts)
            
            # Calculate word count
            word_count = len(full_text.split()) if full_text else 0
            
            # Add to conversation
            if full_text:  # Only add if there's actual content
                conversation.append({
                    'turn': turn_number,
                    'speaker': speaker,
                    'text': full_text,
                    'word_count': word_count
                })
        
        # Also check for procedural text between contributions
        procedural_items = content_div.find_all('p', class_='procedural')
        for item in procedural_items:
            text = item.get_text(strip=True)
            if text and not self._is_minor_procedural(text):
                turn_number += 1
                conversation.append({
                    'turn': turn_number,
                    'speaker': 'PROCEDURAL',
                    'text': text,
                    'word_count': len(text.split())
                })
        
        # Sort by turn number to maintain order
        conversation.sort(key=lambda x: x['turn'])
        
        # Renumber turns sequentially
        for i, turn in enumerate(conversation, 1):
            turn['turn'] = i
        
        return conversation
    
    def _is_minor_procedural(self, text: str) -> bool:
        """Check if procedural text is minor (e.g., just noting who sat down)."""
        minor_patterns = [
            r'^\s*sat down',
            r'^\s*rose\s*$',
            r'^\s*\[.*\]\s*$'  # Just bracketed notes
        ]
        
        for pattern in minor_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        return len(text.split()) < 5  # Very short procedural notes


def test_single_debate():
    """Test extraction on a single debate file."""
    extractor = SpeakerDebateExtractor()
    
    # Test on the war damage compensation debate
    test_file = Path("/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/data/hansard/1950/may/22_49_war-damage-compensation.html.gz")
    
    print("Extracting debate from:", test_file.name)
    print("=" * 60)
    
    result = extractor.extract_debate(test_file)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Debate ID: {result['debate_id']}")
        print(f"Topic: {result['topic']}")
        print(f"Date: {result['date']}")
        print(f"Chamber: {result['chamber']}")
        print(f"Total speakers: {result['total_speakers']}")
        print(f"Unique speakers: {result['unique_speakers']}")
        print(f"Total turns: {result['total_turns']}")
        print("\nFirst 3 conversation turns:")
        
        for turn in result['conversation'][:3]:
            print(f"\nTurn {turn['turn']}:")
            print(f"  Speaker: {turn['speaker']}")
            print(f"  Words: {turn['word_count']}")
            print(f"  Text: {turn['text'][:100]}...")
    
    # Save to JSON for inspection
    output_file = Path("test_debate_extraction.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nFull result saved to: {output_file}")
    
    return result


if __name__ == "__main__":
    test_single_debate()