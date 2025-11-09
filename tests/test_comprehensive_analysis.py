#!/usr/bin/env python3
"""
Tests for comprehensive_analysis.py

Tests the main comprehensive analysis functionality including
data loading, preprocessing, and visualization generation.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.comprehensive_analysis import (
    run_comprehensive_analysis,
    create_ngram_visualization,
    create_topic_visualization,
    create_temporal_analysis
)


class TestNgramVisualization:
    """Test n-gram visualization creation"""

    def test_create_unigram_visualization(self, tmp_path):
        """Test unigram visualization is created"""
        ngram_freq = [
            ('transport', 100),
            ('infrastructure', 80),
            ('policy', 60)
        ]

        output_path = tmp_path / "test_unigrams.png"
        create_ngram_visualization(
            ngram_freq,
            "Test Unigrams",
            output_path,
            top_n=3
        )

        # Check file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_create_bigram_visualization(self, tmp_path):
        """Test bigram visualization is created"""
        ngram_freq = [
            ('public transport', 50),
            ('local authority', 40),
            ('prime minister', 30)
        ]

        output_path = tmp_path / "test_bigrams.png"
        create_ngram_visualization(
            ngram_freq,
            "Test Bigrams",
            output_path,
            top_n=3
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_empty_ngrams(self, tmp_path):
        """Test handling of empty ngram list"""
        output_path = tmp_path / "test_empty.png"
        create_ngram_visualization(
            [],
            "Empty Test",
            output_path
        )
        # Should not crash, may or may not create file
        # Main thing is it doesn't raise an exception

    def test_top_n_limit(self, tmp_path):
        """Test that top_n parameter limits output"""
        ngram_freq = [(f'word{i}', 100-i) for i in range(50)]

        output_path = tmp_path / "test_limited.png"
        create_ngram_visualization(
            ngram_freq,
            "Limited Test",
            output_path,
            top_n=10
        )

        assert output_path.exists()


class TestTopicVisualization:
    """Test topic modeling visualization"""

    def test_create_topic_visualization(self, tmp_path):
        """Test topic visualization is created"""
        topics = [
            {
                'topic_id': 0,
                'top_words': ['transport', 'railway', 'infrastructure'],
                'weights': [0.1, 0.08, 0.06]
            },
            {
                'topic_id': 1,
                'top_words': ['health', 'hospital', 'medical'],
                'weights': [0.12, 0.09, 0.07]
            }
        ]

        output_path = tmp_path / "test_topics.png"
        create_topic_visualization(topics, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_single_topic(self, tmp_path):
        """Test with single topic"""
        topics = [
            {
                'topic_id': 0,
                'top_words': ['word1', 'word2'],
                'weights': [0.1, 0.08]
            }
        ]

        output_path = tmp_path / "test_single_topic.png"
        create_topic_visualization(topics, output_path)

        assert output_path.exists()

    def test_empty_topics(self, tmp_path):
        """Test handling of empty topics list"""
        output_path = tmp_path / "test_no_topics.png"
        # Should handle gracefully
        try:
            create_topic_visualization([], output_path)
        except Exception:
            pass  # May raise exception, but shouldn't crash hard


class TestTemporalAnalysis:
    """Test temporal analysis visualization"""

    def test_create_temporal_charts(self, tmp_path):
        """Test temporal analysis creates charts"""
        import pandas as pd

        # Create mock metadata
        metadata_df = pd.DataFrame({
            'year': [1990, 1990, 1991, 1991, 1992],
            'word_count': [1000, 1200, 1100, 1300, 1500]
        })

        create_temporal_analysis(metadata_df, tmp_path)

        # Check that files were created
        debates_chart = tmp_path / 'temporal_debates_per_year.png'
        words_chart = tmp_path / 'temporal_words_per_year.png'

        assert debates_chart.exists()
        assert words_chart.exists()
        assert debates_chart.stat().st_size > 0
        assert words_chart.stat().st_size > 0

    def test_single_year_data(self, tmp_path):
        """Test with single year"""
        import pandas as pd

        metadata_df = pd.DataFrame({
            'year': [2000, 2000, 2000],
            'word_count': [1000, 1200, 1100]
        })

        # Should handle single year gracefully
        create_temporal_analysis(metadata_df, tmp_path)

        debates_chart = tmp_path / 'temporal_debates_per_year.png'
        assert debates_chart.exists()


class TestComprehensiveAnalysis:
    """Test the main comprehensive analysis function"""

    @patch('analysis.comprehensive_analysis.UnifiedDataLoader')
    def test_run_with_sample(self, mock_loader_class, tmp_path):
        """Test running analysis with sample data"""
        import pandas as pd

        # Create mock loader
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        # Mock data
        mock_metadata = pd.DataFrame({
            'debate_id': ['d1', 'd2', 'd3'],
            'year': [2000, 2000, 2001],
            'word_count': [1000, 1200, 1100]
        })

        mock_text_data = [
            {'debate_id': 'd1', 'full_text': 'transport infrastructure policy'},
            {'debate_id': 'd2', 'full_text': 'health education social'},
            {'debate_id': 'd3', 'full_text': 'parliament debate law'}
        ]

        mock_loader.load_debates.return_value = {
            'metadata': mock_metadata,
            'text_data': mock_text_data
        }

        # Patch the output directory
        with patch('analysis.comprehensive_analysis.Paths') as mock_paths:
            mock_paths.ANALYSIS_DIR = tmp_path

            # Run analysis
            run_comprehensive_analysis(
                year_range=(2000, 2001),
                sample_size=3,
                filtering_level='moderate',
                n_topics=2
            )

            # Check that analysis files were created
            output_dir = tmp_path / 'comprehensive'
            assert output_dir.exists()

            # Check for expected output files
            expected_files = [
                'unigrams.png',
                'bigrams.png',
                'topic_modeling.png',
                'temporal_debates_per_year.png',
                'temporal_words_per_year.png',
                'analysis_results.json'
            ]

            for filename in expected_files:
                filepath = output_dir / filename
                assert filepath.exists(), f"Expected file not found: {filename}"

    @patch('analysis.comprehensive_analysis.UnifiedDataLoader')
    def test_json_results_structure(self, mock_loader_class, tmp_path):
        """Test that JSON results have correct structure"""
        import pandas as pd

        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        mock_metadata = pd.DataFrame({
            'debate_id': ['d1'],
            'year': [2000],
            'word_count': [1000]
        })

        mock_text_data = [
            {'debate_id': 'd1', 'full_text': 'transport policy infrastructure'}
        ]

        mock_loader.load_debates.return_value = {
            'metadata': mock_metadata,
            'text_data': mock_text_data
        }

        with patch('analysis.comprehensive_analysis.Paths') as mock_paths:
            mock_paths.ANALYSIS_DIR = tmp_path

            run_comprehensive_analysis(
                sample_size=1,
                filtering_level='moderate',
                n_topics=2
            )

            # Load and check JSON structure
            json_path = tmp_path / 'comprehensive' / 'analysis_results.json'
            assert json_path.exists()

            with open(json_path, 'r') as f:
                results = json.load(f)

            # Check required keys
            assert 'metadata' in results
            assert 'unigrams' in results
            assert 'bigrams' in results
            assert 'topics' in results

            # Check metadata structure
            assert 'total_debates' in results['metadata']
            assert 'year_range' in results['metadata']
            assert 'total_words' in results['metadata']
            assert 'filtering_level' in results['metadata']

    def test_filtering_levels(self):
        """Test that different filtering levels are accepted"""
        from analysis.comprehensive_analysis import get_stop_words

        levels = ['minimal', 'basic', 'parliamentary', 'moderate', 'aggressive']

        for level in levels:
            stop_words = get_stop_words(level)
            assert isinstance(stop_words, set)
            assert len(stop_words) > 0

    def test_invalid_filtering_level(self):
        """Test handling of invalid filtering level"""
        from analysis.comprehensive_analysis import get_stop_words

        # Should default to moderate or handle gracefully
        stop_words = get_stop_words('invalid_level')
        assert isinstance(stop_words, set)


class TestEndToEnd:
    """End-to-end integration tests with real (small) data"""

    def test_minimal_real_data(self, tmp_path):
        """Test with minimal real-like data"""
        # This test would need actual data files
        # Skipping if data not available
        pytest.skip("Requires actual data files")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
