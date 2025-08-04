#!/usr/bin/env python3
"""
Test complete parsing pipeline to identify specific issues.
"""

import gzip
import json
from pathlib import Path
from bs4 import BeautifulSoup
import re

def analyze_single_file_deeply(file_path):
    """Deep analysis of a single file's parsing."""
    print(f"=== DEEP ANALYSIS: {file_path.name} ===")
    
    # Load raw HTML
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        html = f.read()
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find corresponding processed record
    year = file_path.parts[-3]
    processed_file = Path(f"data/processed/content/{year}/debates_{year}.jsonl")
    
    processed_record = None
    if processed_file.exists():
        with open(processed_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if record['file_name'] == file_path.name:
                        processed_record = record
                        break
    
    # Analyze HTML structure
    print(f"\n📄 HTML STRUCTURE:")
    title = soup.find('title')
    print(f"Title: {title.get_text() if title else 'None'}")
    
    # Find main content div
    commons_div = soup.find('div', class_='house-of-commons-sitting')
    lords_div = soup.find('div', class_='house-of-lords-sitting')
    main_div = commons_div or lords_div
    
    if main_div:
        print(f"Main div: {main_div.get('class')}")
        
        # Analyze structure elements
        contributions = main_div.find_all('div', class_='member_contribution')
        cites = main_div.find_all('cite', class_='member')
        blockquotes = main_div.find_all('blockquote')
        paragraphs = main_div.find_all('p')
        
        print(f"Member contributions: {len(contributions)}")
        print(f"Speaker citations: {len(cites)}")
        print(f"Blockquotes: {len(blockquotes)}")
        print(f"Paragraphs: {len(paragraphs)}")
        
        # Show speakers
        if cites:
            print(f"\n👥 SPEAKERS FOUND IN HTML:")
            unique_speakers = set()
            for cite in cites:
                speaker = cite.get_text(strip=True)
                unique_speakers.add(speaker)
            for speaker in sorted(unique_speakers):
                print(f"  • {speaker}")
        
        # Check for navigation elements
        nav_elements = main_div.find_all(string=re.compile(r'Back to|Forward to'))
        if nav_elements:
            print(f"\n🧭 NAVIGATION POLLUTION:")
            for nav in nav_elements[:3]:  # Show first 3
                print(f"  • {nav.strip()}")
        
        # Check content extraction
        print(f"\n📝 CONTENT EXTRACTION:")
        
        # What does get_text() give us?
        raw_text = main_div.get_text(separator='\n', strip=True)
        lines = [line for line in raw_text.split('\n') if line.strip()]
        
        print(f"Lines extracted: {len(lines)}")
        print(f"Total characters: {len(raw_text)}")
        
        # Show first few lines
        print(f"\nFirst 10 lines:")
        for i, line in enumerate(lines[:10], 1):
            print(f"  {i:2}: {line}")
        
        # Show last few lines (likely navigation)
        if len(lines) > 5:
            print(f"\nLast 5 lines:")
            for i, line in enumerate(lines[-5:], len(lines)-4):
                print(f"  {i:2}: {line}")
    
    # Compare with processed record
    if processed_record:
        print(f"\n🔍 PROCESSED RECORD COMPARISON:")
        metadata = processed_record.get('metadata', {})
        
        print(f"Speakers found: {len(metadata.get('speakers', []))}")
        print(f"Speaker count: {metadata.get('speaker_count', 0)}")
        print(f"Word count: {metadata.get('word_count', 0)}")
        print(f"Line count: {metadata.get('line_count', 0)}")
        print(f"Debate topics: {len(metadata.get('debate_topics', []))}")
        
        # Check if navigation is in processed text
        full_text = processed_record.get('full_text', '')
        has_nav = 'Back to' in full_text or 'Forward to' in full_text
        print(f"Navigation pollution: {'Yes' if has_nav else 'No'}")
        
        if metadata.get('speakers'):
            print(f"Processed speakers: {metadata['speakers']}")
        
        if metadata.get('debate_topics'):
            print(f"Debate topics: {metadata['debate_topics']}")
    
    return {
        'html_speakers': len(cites) if main_div else 0,
        'processed_speakers': len(processed_record.get('metadata', {}).get('speakers', [])) if processed_record else 0,
        'has_navigation': len(nav_elements) > 0 if main_div else False,
        'content_lines': len(lines) if main_div else 0
    }

def test_multiple_file_types():
    """Test parsing on different types of files."""
    print("=== TESTING MULTIPLE FILE TYPES ===")
    
    test_files = [
        # Different types of documents
        Path("data/hansard/1925/mar/12_17_toy-pistols.html.gz"),  # Question with multiple speakers
        Path("data/hansard/1925/mar/18_10_housing-scotland-bill-hl.html.gz"),  # Simple Lords bill
        Path("data/hansard/1925/mar/24_65_coal-industry.html.gz"),  # Short summary
    ]
    
    results = []
    for file_path in test_files:
        if file_path.exists():
            result = analyze_single_file_deeply(file_path)
            results.append(result)
            print("\n" + "="*80 + "\n")
        else:
            print(f"❌ File not found: {file_path}")
    
    # Summary
    print("📊 SUMMARY OF PARSING ISSUES:")
    total_files = len(results)
    if total_files > 0:
        nav_polluted = sum(1 for r in results if r['has_navigation'])
        speaker_mismatch = sum(1 for r in results if r['html_speakers'] != r['processed_speakers'])
        
        print(f"Files with navigation pollution: {nav_polluted}/{total_files}")
        print(f"Files with speaker extraction mismatch: {speaker_mismatch}/{total_files}")
    
    return results

if __name__ == "__main__":
    test_multiple_file_types()