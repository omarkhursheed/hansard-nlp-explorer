#!/usr/bin/env python3
"""
Test speaker/speech extraction using the HTML-based extractor.

Tests the hansard.utils.speech_extractor module which replaces the buggy
regex-based extraction with HTML structure-based extraction.
"""

from pathlib import Path
import pytest
from hansard.utils.path_config import Paths
from hansard.utils.speech_extractor import (
    extract_speeches_from_file,
    extract_speeches_from_html,
    extract_speakers_from_html,
    load_html_from_file,
    get_raw_html_path,
    get_chamber_from_html,
)


def test_extract_speeches_from_file():
    """Test speech extraction from a sample file."""
    # Use 1950 stud-farms file which has clear Q&A format
    file_path = Paths.get_data_dir() / "hansard/1950/may/11_53_stud-farms.html.gz"

    if not file_path.exists():
        pytest.skip(f"Sample HTML not found: {file_path}")

    speeches = extract_speeches_from_file(file_path)

    # Should extract 6 speeches
    assert len(speeches) == 6, f"Expected 6 speeches, got {len(speeches)}"

    # Check that speeches have required keys
    for speech in speeches:
        assert 'speaker' in speech
        assert 'text' in speech
        assert 'contribution_id' in speech
        assert 'is_question' in speech
        assert 'question_number' in speech

    # Check specific speakers
    speakers = [s['speaker'] for s in speeches]
    assert 'Mr. Nabarro' in speakers
    assert 'Sir T. Dugdale' in speakers

    # First speech should be a question
    assert speeches[0]['is_question'] == True
    assert speeches[0]['question_number'] == '61'


def test_speeches_are_not_mixed():
    """Test that speeches don't contain text from other speakers."""
    file_path = Paths.get_data_dir() / "hansard/1950/may/11_53_stud-farms.html.gz"

    if not file_path.exists():
        pytest.skip(f"Sample HTML not found: {file_path}")

    speeches = extract_speeches_from_file(file_path)

    # Check Sir T. Dugdale's speech doesn't contain Mr. Williams' response
    # This was the bug reported in the audit
    dugdale_speech = next((s for s in speeches if 'Dugdale' in s['speaker']), None)
    assert dugdale_speech is not None

    # His speech should NOT contain "It will be" (which is Mr. Williams' response)
    assert "It will be" not in dugdale_speech['text']


def test_extract_speakers_from_html():
    """Test speaker name extraction."""
    file_path = Paths.get_data_dir() / "hansard/1950/may/11_53_stud-farms.html.gz"

    if not file_path.exists():
        pytest.skip(f"Sample HTML not found: {file_path}")

    soup = load_html_from_file(file_path)
    speakers = extract_speakers_from_html(soup)

    # Should find unique speakers
    assert len(speakers) >= 3
    assert any('Nabarro' in s for s in speakers)


def test_get_chamber_from_html():
    """Test chamber detection."""
    file_path = Paths.get_data_dir() / "hansard/1950/may/11_53_stud-farms.html.gz"

    if not file_path.exists():
        pytest.skip(f"Sample HTML not found: {file_path}")

    soup = load_html_from_file(file_path)
    chamber = get_chamber_from_html(soup)

    # 1950 Commons file
    assert chamber == 'commons'


def test_get_raw_html_path():
    """Test file path conversion."""
    hansard_base = Path('/data/hansard')

    # Test various input formats
    assert get_raw_html_path('hansard/1950/may/test.html.gz', hansard_base) == \
           Path('/data/hansard/1950/may/test.html.gz')

    assert get_raw_html_path('data-hansard/hansard/1950/may/test.html.gz', hansard_base) == \
           Path('/data/hansard/1950/may/test.html.gz')

    assert get_raw_html_path('1950/may/test.html.gz', hansard_base) == \
           Path('/data/hansard/1950/may/test.html.gz')


def test_empty_file_handling():
    """Test handling of non-existent files."""
    from hansard.utils.speech_extractor import extract_speeches_from_raw_html

    # Should return empty list for non-existent file
    result = extract_speeches_from_raw_html(Path('/nonexistent/file.html.gz'))
    assert result == []


if __name__ == "__main__":
    test_extract_speeches_from_file()
    test_speeches_are_not_mixed()
    test_extract_speakers_from_html()
    test_get_chamber_from_html()
    test_get_raw_html_path()
    test_empty_file_handling()
    print("All tests passed!")
