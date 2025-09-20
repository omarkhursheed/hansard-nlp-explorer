"""
Integration tests for speaker processing module.
"""

import pytest
import sys
from pathlib import Path
import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hansard.speaker_processing import SpeakerProcessor


class TestSpeakerProcessor:
    """Tests for the consolidated SpeakerProcessor class."""

    def setup_method(self):
        """Set up test data."""
        # Create mock data for testing
        self.mock_speakers_data = [
            {"name": "Mr. Smith", "first_appearance": 1920, "last_appearance": 1935,
             "debate_count": 45, "primary_chamber": "commons"},
            {"name": "MR. SMITH", "first_appearance": 1925, "last_appearance": 1930,
             "debate_count": 20, "primary_chamber": "commons"},
            {"name": "Lord Wilson", "first_appearance": 1950, "last_appearance": 1970,
             "debate_count": 100, "primary_chamber": "lords"},
            {"name": "Mrs. Jones", "first_appearance": 1960, "last_appearance": 1980,
             "debate_count": 75, "primary_chamber": "commons"}
        ]
        self.speakers_df = pl.DataFrame(self.mock_speakers_data)
        self.processor = SpeakerProcessor()

    def test_normalize_speakers(self):
        """Test speaker name normalization."""
        normalized = self.processor.normalize_speakers(self.speakers_df)

        # Check that normalized_name column was added
        assert "normalized_name" in normalized.columns

        # Check specific normalizations
        norm_names = normalized["normalized_name"].to_list()
        assert "mr smith" in norm_names
        assert "lord wilson" in norm_names
        assert "mrs jones" in norm_names

    def test_deduplicate_speakers(self):
        """Test speaker deduplication."""
        # First normalize
        normalized = self.processor.normalize_speakers(self.speakers_df)

        # Then deduplicate
        deduped = self.processor.deduplicate_speakers(normalized)

        # Should have fewer records (Mr. Smith duplicates merged)
        assert len(deduped) == 3  # Smith, Wilson, Jones

        # Check that Mr. Smith records were merged correctly
        smith_record = deduped.filter(pl.col("normalized_name") == "mr smith")[0]
        assert smith_record["first_appearance"][0] == 1920  # Earlier date
        assert smith_record["last_appearance"][0] == 1935   # Later date
        assert smith_record["total_debates"][0] == 65       # Sum of debates

    def test_validate_speaker_dataset(self):
        """Test dataset validation."""
        validation = self.processor.validate_speaker_dataset(self.speakers_df)

        # Check validation report structure
        assert "total_speakers" in validation
        assert validation["total_speakers"] == 4

        assert "speakers_with_debates" in validation
        assert validation["speakers_with_debates"] == 4

        assert "date_coverage" in validation
        assert validation["date_coverage"]["earliest"] == 1920
        assert validation["date_coverage"]["latest"] == 1980

        assert "chamber_distribution" in validation
        assert "issues" in validation

    def test_check_mp_coverage(self):
        """Test MP coverage analysis."""
        coverage = self.processor.check_mp_coverage(self.speakers_df)

        assert "total_speakers" in coverage
        assert coverage["total_speakers"] == 4

        assert "likely_mps" in coverage
        # Mr. Smith and Mrs. Jones have MP indicators
        assert coverage["likely_mps"] >= 2

        assert "coverage_percentage" in coverage
        assert 0 <= coverage["coverage_percentage"] <= 100

    def test_generate_temporal_analysis(self):
        """Test temporal analysis generation."""
        temporal = self.processor.generate_temporal_analysis(self.speakers_df)

        # Check that we have data
        assert len(temporal) > 0

        # Check columns
        expected_cols = ["period", "start_year", "end_year", "active_speakers", "new_speakers"]
        for col in expected_cols:
            assert col in temporal.columns

        # Check that periods cover our data range
        periods = temporal["period"].to_list()
        assert any("1920" in p for p in periods)
        assert any("1970" in p for p in periods)


class TestSpeakerProcessingEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        processor = SpeakerProcessor()
        empty_df = pl.DataFrame({"name": [], "debate_count": []})

        # Should handle empty data gracefully
        normalized = processor.normalize_speakers(empty_df)
        assert len(normalized) == 0

        validation = processor.validate_speaker_dataset(empty_df)
        assert validation["total_speakers"] == 0

    def test_missing_columns(self):
        """Test handling of missing required columns."""
        processor = SpeakerProcessor()
        incomplete_df = pl.DataFrame({"name": ["Test Speaker"]})

        # Should handle missing columns gracefully
        validation = processor.validate_speaker_dataset(incomplete_df)
        # The function should still work even with minimal columns
        assert validation["total_speakers"] == 1
        # Check that optional fields are not in the report if columns don't exist
        assert "date_coverage" not in validation  # No date columns present

    def test_null_values(self):
        """Test handling of null values."""
        processor = SpeakerProcessor()
        df_with_nulls = pl.DataFrame({
            "name": ["Speaker 1", None, "Speaker 3"],
            "debate_count": [10, 20, None],
            "first_appearance": [1900, 1910, 1920],
            "last_appearance": [1920, 1930, 1940],
            "primary_chamber": ["commons", "lords", None]
        })

        validation = processor.validate_speaker_dataset(df_with_nulls)
        # Should report null values in issues
        assert any("null" in issue.lower() for issue in validation["issues"])


if __name__ == "__main__":
    # Run basic tests
    test = TestSpeakerProcessor()
    test.setup_method()

    test.test_normalize_speakers()
    print("✓ Normalization test passed")

    test.test_deduplicate_speakers()
    print("✓ Deduplication test passed")

    test.test_validate_speaker_dataset()
    print("✓ Validation test passed")

    test.test_check_mp_coverage()
    print("✓ MP coverage test passed")

    test.test_generate_temporal_analysis()
    print("✓ Temporal analysis test passed")

    # Test edge cases
    edge_test = TestSpeakerProcessingEdgeCases()
    edge_test.test_empty_dataset()
    print("✓ Empty dataset test passed")

    edge_test.test_missing_columns()
    print("✓ Missing columns test passed")

    edge_test.test_null_values()
    print("✓ Null values test passed")

    print("\nAll speaker processing tests passed!")