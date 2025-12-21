#!/usr/bin/env python3
"""
HTML-Based Speech Extraction for Hansard Parliamentary Debates

This module provides accurate speech extraction by using HTML structure directly,
rather than regex patterns on flattened text. Each speech is contained in a
`<div class='hentry member_contribution'>` element with clear boundaries.

This replaces the buggy `extract_speeches_from_text()` function that caused
~5% of speeches to have incorrect speaker attribution.

Usage:
    from hansard.utils.speech_extractor import extract_speeches_from_html

    speeches = extract_speeches_from_html(soup)
    for speech in speeches:
        print(f"{speech['speaker']}: {speech['text'][:100]}...")
"""

import gzip
import re
from pathlib import Path
from typing import List, Dict, Optional, Union
from bs4 import BeautifulSoup


def extract_speeches_from_html(soup: BeautifulSoup, min_length: int = 50) -> List[Dict]:
    """
    Extract individual speeches from HTML using structural boundaries.

    Uses the HTML structure directly instead of regex on flattened text.
    Each speech is contained in a `<div class='hentry member_contribution'>`
    element with clear boundaries.

    Args:
        soup: BeautifulSoup object of the debate HTML
        min_length: Minimum character length for a speech to be included

    Returns:
        List of speech dictionaries with keys:
            - speaker: Speaker name as it appears in HTML
            - text: Full speech text (excluding procedural content)
            - contribution_id: Unique ID from the HTML element
            - is_question: Whether this is a parliamentary question
            - question_number: Question number if applicable (e.g., "61")
    """
    speeches = []

    # Find all member contribution divs - each contains exactly one speech
    contributions = soup.find_all('div', class_='hentry member_contribution')

    for contrib in contributions:
        # Get speaker from cite element
        cite = contrib.find('cite', class_='member')
        if not cite:
            continue

        speaker = cite.get_text(strip=True)
        if not speaker:
            continue

        # Get speech text from blockquote
        blockquote = contrib.find('blockquote', class_='contribution_text')
        if not blockquote:
            # Try without class
            blockquote = contrib.find('blockquote')
        if not blockquote:
            continue

        # Check for question number
        question_span = blockquote.find('span', class_='question_no')
        question_number = None
        is_question = False
        if question_span:
            question_text = question_span.get_text(strip=True)
            # Extract number from "61." format
            match = re.search(r'(\d+)', question_text)
            if match:
                question_number = match.group(1)
                is_question = True

        # Get all paragraphs, excluding procedural text
        paragraphs = []
        for p in blockquote.find_all('p'):
            # Skip procedural paragraphs (e.g., "Order for Committee read")
            classes = p.get('class', [])
            if 'procedural' in classes:
                continue

            # Get text, preserving some structure
            p_text = p.get_text(strip=True)
            if p_text:
                paragraphs.append(p_text)

        speech_text = ' '.join(paragraphs)

        # Apply minimum length filter
        if speech_text and len(speech_text) >= min_length:
            speeches.append({
                'speaker': speaker,
                'text': speech_text,
                'contribution_id': contrib.get('id', ''),
                'is_question': is_question,
                'question_number': question_number
            })

    return speeches


def extract_speakers_from_html(soup: BeautifulSoup) -> List[str]:
    """
    Extract unique speaker names from HTML using structural elements.

    Args:
        soup: BeautifulSoup object of the debate HTML

    Returns:
        List of unique speaker names
    """
    speakers = set()

    # Find all member citations
    for cite in soup.find_all('cite', class_='member'):
        speaker = cite.get_text(strip=True)
        if speaker and len(speaker) < 200:  # Sanity check
            speakers.add(speaker)

    return list(speakers)


def load_html_from_file(file_path: Union[str, Path]) -> BeautifulSoup:
    """
    Load and parse HTML from a file (handles .gz compression).

    Args:
        file_path: Path to HTML file (can be .html or .html.gz)

    Returns:
        BeautifulSoup object
    """
    file_path = Path(file_path)

    if file_path.suffix == '.gz' or str(file_path).endswith('.html.gz'):
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            html_content = f.read()
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

    return BeautifulSoup(html_content, 'html.parser')


def extract_speeches_from_file(file_path: Union[str, Path], min_length: int = 50) -> List[Dict]:
    """
    Convenience function to extract speeches directly from a file.

    Args:
        file_path: Path to HTML file
        min_length: Minimum speech length

    Returns:
        List of speech dictionaries
    """
    soup = load_html_from_file(file_path)
    return extract_speeches_from_html(soup, min_length)


def get_chamber_from_html(soup: BeautifulSoup) -> Optional[str]:
    """
    Detect which chamber (Commons or Lords) from HTML structure.

    Args:
        soup: BeautifulSoup object

    Returns:
        'commons', 'lords', or None if cannot determine
    """
    # Check for chamber-specific container divs
    if soup.find('div', class_='house-of-commons-sitting'):
        return 'commons'
    elif soup.find('div', class_='house-of-lords-sitting'):
        return 'lords'

    # Check meta/title for hints
    title = soup.find('title')
    if title:
        title_text = title.get_text().lower()
        if 'commons' in title_text:
            return 'commons'
        elif 'lords' in title_text:
            return 'lords'

    return None


def get_debate_metadata(soup: BeautifulSoup) -> Dict:
    """
    Extract metadata about the debate from HTML.

    Args:
        soup: BeautifulSoup object

    Returns:
        Dictionary with metadata: title, chamber, section_ref, date
    """
    metadata = {}

    # Title
    title_elem = soup.find('h1', class_='title')
    if title_elem:
        metadata['title'] = title_elem.get_text(strip=True)
    else:
        title = soup.find('title')
        if title:
            metadata['title'] = title.get_text(strip=True)

    # Chamber
    metadata['chamber'] = get_chamber_from_html(soup)

    # Section reference (e.g., "HC Deb 11 May 1950 vol 475 c574")
    section_cite = soup.find('cite', class_='section')
    if section_cite:
        metadata['section_ref'] = section_cite.get_text(strip=True)

        # Try to parse date and volume from section reference
        ref = metadata['section_ref']
        # Pattern: "HC Deb 11 May 1950 vol 475 c574"
        date_match = re.search(r'(\d{1,2}\s+\w+\s+\d{4})', ref)
        if date_match:
            metadata['date'] = date_match.group(1)

        vol_match = re.search(r'vol\s+(\d+)', ref)
        if vol_match:
            metadata['volume'] = vol_match.group(1)

    return metadata


def get_raw_html_path(file_path: str, hansard_base: Path) -> Path:
    """
    Convert file_path from metadata to actual raw HTML file path.

    Args:
        file_path: File path from metadata (e.g., 'hansard/1950/may/11_53_stud-farms.html.gz')
        hansard_base: Base path for hansard data (e.g., data-hansard/hansard/)

    Returns:
        Full path to the raw HTML file
    """
    if not file_path:
        return hansard_base / 'nonexistent'

    # Handle different file_path formats
    if file_path.startswith('hansard/'):
        return hansard_base / file_path.replace('hansard/', '', 1)
    elif file_path.startswith('data-hansard/hansard/'):
        return hansard_base / file_path.replace('data-hansard/hansard/', '', 1)
    elif file_path.startswith('data-hansard/'):
        return hansard_base.parent / file_path.replace('data-hansard/', '', 1)
    else:
        # Assume it's relative to hansard base
        return hansard_base / file_path


def extract_speeches_from_raw_html(html_file_path: Path) -> List[Dict]:
    """
    Extract speeches directly from raw HTML file using structural boundaries.

    Wrapper around extract_speeches_from_file with error handling.

    Args:
        html_file_path: Path to the raw HTML file (.html or .html.gz)

    Returns:
        List of speech dictionaries with 'speaker', 'text', 'contribution_id'
    """
    try:
        return extract_speeches_from_file(html_file_path)
    except Exception:
        # Fallback: return empty list if file cannot be read
        return []


# For backward compatibility with existing code that might import from here
def extract_speeches_from_text(text: str, speakers_list: List[str]) -> List[Dict]:
    """
    DEPRECATED: This function is kept for backward compatibility only.

    Use extract_speeches_from_html() instead for accurate extraction.
    This regex-based approach has known issues with speaker attribution.
    """
    import warnings
    warnings.warn(
        "extract_speeches_from_text() is deprecated and has known accuracy issues. "
        "Use extract_speeches_from_html() with BeautifulSoup instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Return empty list - encourage migration to HTML-based extraction
    return []


if __name__ == "__main__":
    # Test the extractor on a sample file
    import sys

    print("Speech Extractor Test")
    print("=" * 60)

    # Find a sample HTML file to test
    from pathlib import Path

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent

    # Look for sample files
    sample_dirs = [
        project_root / 'data-hansard' / 'hansard' / '1950' / 'may',
        project_root / 'data-hansard' / 'hansard' / '1900' / 'jan',
    ]

    sample_file = None
    for d in sample_dirs:
        if d.exists():
            files = list(d.glob('*.html.gz'))
            if files:
                sample_file = files[0]
                break

    if sample_file:
        print(f"\nTesting with: {sample_file}")

        soup = load_html_from_file(sample_file)

        # Get metadata
        metadata = get_debate_metadata(soup)
        print(f"\nMetadata:")
        for k, v in metadata.items():
            print(f"  {k}: {v}")

        # Extract speeches
        speeches = extract_speeches_from_html(soup)
        print(f"\nExtracted {len(speeches)} speeches:")

        for i, speech in enumerate(speeches[:5]):  # Show first 5
            print(f"\n  [{i+1}] {speech['speaker']}")
            print(f"      ID: {speech['contribution_id']}")
            print(f"      Question: {speech['is_question']} ({speech['question_number']})")
            print(f"      Text: {speech['text'][:100]}...")

        if len(speeches) > 5:
            print(f"\n  ... and {len(speeches) - 5} more speeches")
    else:
        print("\nNo sample files found. Run with a file path argument:")
        print("  python speech_extractor.py /path/to/debate.html.gz")
