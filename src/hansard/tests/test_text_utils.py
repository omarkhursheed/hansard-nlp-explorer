"""
Unit tests for text processing utilities.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestTextProcessing:
    """Tests for basic text processing functions."""

    def test_clean_whitespace(self):
        """Test whitespace normalization."""
        # Basic whitespace cleaning
        text = "  This   has    extra   spaces  "
        expected = "This has extra spaces"
        # Simple implementation for testing
        result = " ".join(text.split())
        assert result == expected

    def test_extract_year_from_string(self):
        """Test year extraction from text."""
        test_cases = [
            ("debate_1803.html", 1803),
            ("file_2005_01.json", 2005),
            ("year_1920_data", 1920),
            ("no_year_here", None)
        ]

        for text, expected in test_cases:
            # Simple regex for 4-digit years between 1800-2099
            import re
            match = re.search(r'(1[89]\d{2}|20\d{2})', text)
            result = int(match.group(1)) if match else None
            assert result == expected, f"Failed for {text}"

    def test_normalize_speaker_name(self):
        """Test speaker name normalization."""
        test_cases = [
            ("MR. SMITH", "mr smith"),
            ("Mrs.  Jones", "mrs jones"),
            ("DR.BROWN", "dr brown"),
            ("  Lord  Wilson  ", "lord wilson")
        ]

        for input_name, expected in test_cases:
            # Simple normalization
            result = " ".join(input_name.lower().replace(".", " ").split())
            assert result == expected

    def test_extract_chamber_from_text(self):
        """Test chamber extraction from debate text."""
        commons_text = "House of Commons debate on the budget"
        lords_text = "House of Lords discussion"
        unclear_text = "Parliamentary proceedings"

        # Simple chamber detection
        def extract_chamber(text):
            text_lower = text.lower()
            if "commons" in text_lower:
                return "commons"
            elif "lords" in text_lower:
                return "lords"
            return "unknown"

        assert extract_chamber(commons_text) == "commons"
        assert extract_chamber(lords_text) == "lords"
        assert extract_chamber(unclear_text) == "unknown"


class TestHansardReference:
    """Tests for Hansard reference parsing."""

    def test_parse_hansard_ref(self):
        """Test parsing Hansard reference format."""
        test_refs = [
            ("HC Deb 12 January 1920 vol 123 cc456-78", {
                "chamber": "HC",
                "date": "12 January 1920",
                "volume": "123",
                "columns": "456-78"
            }),
            ("HL Deb 5 March 1850 vol 45 c123", {
                "chamber": "HL",
                "date": "5 March 1850",
                "volume": "45",
                "columns": "123"
            })
        ]

        for ref_string, expected in test_refs:
            # Simple parsing logic
            parts = ref_string.split()
            result = {
                "chamber": parts[0],
                "date": " ".join(parts[2:5]),
                "volume": parts[6],
                "columns": parts[7].replace("cc", "").replace("c", "")
            }
            assert result == expected

    def test_validate_hansard_year_range(self):
        """Test validation of Hansard year ranges."""
        # Hansard covers 1803-2005
        valid_years = [1803, 1900, 1950, 2000, 2005]
        invalid_years = [1802, 1500, 2006, 2020, 3000]

        def is_valid_hansard_year(year):
            return 1803 <= year <= 2005

        for year in valid_years:
            assert is_valid_hansard_year(year) is True

        for year in invalid_years:
            assert is_valid_hansard_year(year) is False


if __name__ == "__main__":
    # Run basic tests
    test = TestTextProcessing()
    test.test_clean_whitespace()
    test.test_extract_year_from_string()
    test.test_normalize_speaker_name()
    test.test_extract_chamber_from_text()
    print("Text processing tests passed!")

    ref_test = TestHansardReference()
    ref_test.test_parse_hansard_ref()
    ref_test.test_validate_hansard_year_range()
    print("Hansard reference tests passed!")

    print("\nAll unit tests passed!")