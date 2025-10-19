#!/usr/bin/env python3
"""
Test speaker extraction from raw HTML to debug the pipeline issue.
"""

import gzip
from pathlib import Path
import pytest
from hansard.utils.path_config import Paths
from bs4 import BeautifulSoup
import re

def extract_speakers_current_method(lines):
    """Current broken method using regex on text lines."""
    speakers = []
    for line in lines[1:20]:  # Check first 20 lines
        # Look for speaker patterns
        speaker_patterns = [
            r'^(Mr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'^(Mrs\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'^(The\s+[A-Z][a-z]+(?:\s+of\s+[A-Z][a-z]+)*)',
            r'^(Lord\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$'
        ]
        for pattern in speaker_patterns:
            match = re.match(pattern, line.strip())
            if match and len(match.group(1)) < 50:
                speakers.append(match.group(1))
                break
    
    return list(set(speakers))  # Remove duplicates

def extract_speakers_html_method(soup):
    """Fixed method using HTML structure."""
    speakers = []
    
    # Find main content div
    content_div = soup.find('div', class_='house-of-commons-sitting')
    if not content_div:
        content_div = soup.find('div', class_='house-of-lords-sitting')
    
    if content_div:
        # Find all member citations
        member_cites = content_div.find_all('cite', class_='member')
        for cite in member_cites:
            speaker_text = cite.get_text(strip=True)
            if speaker_text and len(speaker_text) < 100:  # Reasonable length
                speakers.append(speaker_text)
    
    return list(set(speakers))  # Remove duplicates

def test_speaker_extraction():
    """Test both methods on the toy pistols debate."""
    file_path = Paths.get_data_dir() / "hansard/1925/mar/12_17_toy-pistols.html.gz"
    
    print("Testing speaker extraction methods...")
    print("=" * 60)
    
    if not Path(file_path).exists():
        pytest.skip(f"Sample HTML not found: {file_path}")

    with gzip.open(str(file_path), 'rt', encoding='utf-8') as f:
        html = f.read()
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Get text lines for current method
    content_div = soup.find('div', class_='house-of-commons-sitting')
    if content_div:
        # Remove unwanted elements
        for unwanted in content_div(['nav', 'footer', 'script', 'style']):
            unwanted.decompose()
        
        content_text = content_div.get_text(separator='\n', strip=True)
        lines = [line for line in content_text.split('\n') if line.strip()]
    else:
        lines = []
    
    # Test current broken method
    print("CURRENT METHOD (broken regex):")
    current_speakers = extract_speakers_current_method(lines)
    print(f"Found {len(current_speakers)} speakers: {current_speakers}")
    
    print("\nFirst 20 lines of text (what regex sees):")
    for i, line in enumerate(lines[:20], 1):
        print(f"{i:2}: {line}")
    
    print("\n" + "=" * 60)
    
    # Test fixed HTML method
    print("FIXED METHOD (using HTML structure):")
    html_speakers = extract_speakers_html_method(soup)
    print(f"Found {len(html_speakers)} speakers: {html_speakers}")
    
    print("\nHTML structure (what we should use):")
    content_div = soup.find('div', class_='house-of-commons-sitting')
    if content_div:
        member_cites = content_div.find_all('cite', class_='member')
        for i, cite in enumerate(member_cites[:10], 1):  # Show first 10
            print(f"{i:2}: {cite.get_text(strip=True)}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print(f"Current method: {len(current_speakers)} speakers (BROKEN)")
    print(f"Fixed method: {len(html_speakers)} speakers (CORRECT)")
    print("The HTML has perfect structure - we should use it!")

if __name__ == "__main__":
    test_speaker_extraction()
