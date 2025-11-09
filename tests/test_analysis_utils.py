#!/usr/bin/env python3
"""
Tests for analysis_utils.py

Tests text preprocessing, stop word generation, n-gram analysis,
and topic modeling functions.
"""

import pytest
import numpy as np
from collections import Counter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.analysis_utils import (
    preprocess_text, get_stop_words, analyze_ngrams,
    perform_topic_modeling, COLORS
)


class TestPreprocessText:
    """Test text preprocessing function"""

    def test_basic_preprocessing(self):
        """Test basic text cleaning"""
        text = "The HOUSE discussed the BILL."
        stop_words = {'the'}
        result = preprocess_text(text, stop_words)
        # 'the' should be removed, text lowercased
        assert 'the' not in result.lower().split()
        assert 'house' in result or 'bill' in result

    def test_remove_possessives(self):
        """Test that possessives are handled correctly"""
        text = "John's opinion matters"
        stop_words = set()
        result = preprocess_text(text, stop_words)
        # "John's" should become "John" (or "john")
        assert 'john' in result
        assert "'s" not in result

    def test_remove_short_tokens(self):
        """Test that tokens < 2 characters are removed"""
        text = "I am a UK MP"
        stop_words = set()
        result = preprocess_text(text, stop_words)
        tokens = result.split()
        # Single character tokens should be removed
        assert all(len(token) >= 2 for token in tokens)

    def test_remove_markup(self):
        """Test removal of markdown and quotes"""
        text = "The *bold* text and [reference] should be clean"
        stop_words = set()
        result = preprocess_text(text, stop_words)
        assert '*' not in result
        assert '[' not in result
        assert ']' not in result

    def test_empty_input(self):
        """Test handling of empty input"""
        result = preprocess_text("", set())
        assert result == ""

    def test_none_input(self):
        """Test handling of None input"""
        result = preprocess_text(None, set())
        assert result == ""

    def test_with_stop_words(self):
        """Test that stop words are filtered"""
        text = "The member of parliament spoke about the bill"
        stop_words = {'the', 'of', 'about'}
        result = preprocess_text(text, stop_words)
        result_words = result.split()
        assert 'the' not in result_words
        assert 'of' not in result_words
        assert 'about' not in result_words
        assert 'member' in result_words
        assert 'parliament' in result_words


class TestGetStopWords:
    """Test stop word generation"""

    def test_minimal_level(self):
        """Test minimal filtering level"""
        stop_words = get_stop_words('minimal')
        assert isinstance(stop_words, set)
        assert len(stop_words) > 100  # Should have NLTK words
        assert 'the' in stop_words
        assert 'of' in stop_words

    def test_basic_level(self):
        """Test basic filtering level"""
        stop_words = get_stop_words('basic')
        assert isinstance(stop_words, set)
        assert 'the' in stop_words

    def test_parliamentary_level(self):
        """Test parliamentary filtering includes parliamentary terms"""
        stop_words = get_stop_words('parliamentary')
        # Should include parliamentary terms
        assert 'hon' in stop_words or 'honourable' in stop_words
        assert 'member' in stop_words
        assert 'house' in stop_words

    def test_moderate_level(self):
        """Test moderate filtering includes verbs and vague words"""
        stop_words = get_stop_words('moderate')
        # Should include common verbs
        assert 'make' in stop_words or 'take' in stop_words
        # Should include vague words
        assert 'thing' in stop_words or 'way' in stop_words

    def test_aggressive_level(self):
        """Test aggressive filtering is most comprehensive"""
        stop_words = get_stop_words('aggressive')
        # Should include discourse markers
        assert 'well' in stop_words or 'indeed' in stop_words
        # Should include quantifiers
        assert 'many' in stop_words or 'much' in stop_words
        # Should include adjectives
        assert 'new' in stop_words or 'great' in stop_words
        # Should include parliamentary artifacts
        assert 'deb' in stop_words or 'vol' in stop_words

    def test_aggressive_has_most_words(self):
        """Test that aggressive level has more words than moderate"""
        minimal = get_stop_words('minimal')
        moderate = get_stop_words('moderate')
        aggressive = get_stop_words('aggressive')

        assert len(aggressive) > len(moderate) > len(minimal)

    def test_modal_verbs_included(self):
        """Test that modal verbs missing from NLTK are added"""
        stop_words = get_stop_words('minimal')
        # These are missing from NLTK but should be in our list
        assert 'would' in stop_words
        assert 'may' in stop_words
        assert 'shall' in stop_words
        assert 'must' in stop_words


class TestAnalyzeNgrams:
    """Test n-gram analysis"""

    def test_unigram_analysis(self):
        """Test unigram extraction"""
        texts = [
            "transport infrastructure transport",
            "infrastructure policy transport"
        ]
        ngrams, vectorizer = analyze_ngrams(texts, n_range=(1, 1), max_features=10)

        assert isinstance(ngrams, list)
        assert len(ngrams) > 0
        # Check format: list of (word, count) tuples
        assert isinstance(ngrams[0], tuple)
        assert len(ngrams[0]) == 2
        assert isinstance(ngrams[0][0], str)
        assert isinstance(ngrams[0][1], (int, float, np.integer))

    def test_bigram_analysis(self):
        """Test bigram extraction"""
        texts = [
            "public transport public transport",
            "public policy public transport"
        ]
        ngrams, vectorizer = analyze_ngrams(texts, n_range=(2, 2), max_features=10)

        assert isinstance(ngrams, list)
        assert len(ngrams) > 0
        # Bigrams should be two words separated by space
        if ngrams:
            assert ' ' in ngrams[0][0]

    def test_frequency_ordering(self):
        """Test that ngrams are ordered by frequency"""
        texts = [
            "transport transport transport policy policy infrastructure"
        ]
        ngrams, _ = analyze_ngrams(texts, n_range=(1, 1), max_features=10)

        if len(ngrams) >= 2:
            # First item should have higher or equal frequency than second
            assert ngrams[0][1] >= ngrams[1][1]

    def test_max_features_limit(self):
        """Test that max_features limits output"""
        texts = ["one two three four five six seven eight nine ten"] * 10
        ngrams, _ = analyze_ngrams(texts, n_range=(1, 1), max_features=5)

        assert len(ngrams) <= 5

    def test_stop_words_filtering(self):
        """Test that stop words are filtered during vectorization"""
        texts = ["the house the member the bill"]
        stop_words = ['the']
        ngrams, _ = analyze_ngrams(texts, n_range=(1, 1), stop_words=stop_words)

        # 'the' should not appear in results
        ngram_words = [word for word, count in ngrams]
        assert 'the' not in ngram_words


class TestPerformTopicModeling:
    """Test topic modeling"""

    def test_topic_modeling_basic(self):
        """Test basic topic modeling"""
        texts = [
            "transport railway train infrastructure",
            "health hospital medical care",
            "education school teacher student"
        ] * 10  # Repeat for better topic detection

        topics, lda, vectorizer = perform_topic_modeling(
            texts, n_topics=3, max_features=50
        )

        assert isinstance(topics, list)
        assert len(topics) == 3  # Should have 3 topics as requested

        # Check topic structure
        for topic in topics:
            assert 'topic_id' in topic
            assert 'top_words' in topic
            assert 'weights' in topic
            assert len(topic['top_words']) > 0

    def test_topic_words_are_strings(self):
        """Test that topic words are strings"""
        texts = ["parliament debate law policy"] * 20
        topics, _, _ = perform_topic_modeling(texts, n_topics=2)

        for topic in topics:
            for word in topic['top_words']:
                assert isinstance(word, str)

    def test_topic_weights_are_numeric(self):
        """Test that topic weights are numeric"""
        texts = ["transport infrastructure policy"] * 20
        topics, _, _ = perform_topic_modeling(texts, n_topics=2)

        for topic in topics:
            for weight in topic['weights']:
                assert isinstance(weight, (int, float, np.number))
                assert weight >= 0  # Weights should be non-negative


class TestColors:
    """Test color palette"""

    def test_colors_defined(self):
        """Test that all required colors are defined"""
        required_colors = [
            'male', 'female', 'background', 'grid', 'text',
            'muted', 'accent1', 'accent2', 'primary', 'secondary'
        ]
        for color_name in required_colors:
            assert color_name in COLORS

    def test_colors_are_valid_hex(self):
        """Test that colors are valid hex codes"""
        import re
        hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')

        for color_name, color_value in COLORS.items():
            assert hex_pattern.match(color_value), \
                f"Color {color_name} has invalid hex value: {color_value}"


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_full_pipeline(self):
        """Test complete preprocessing -> analysis pipeline"""
        # Raw texts with various issues
        raw_texts = [
            "The House discussed the TRANSPORT bill today.",
            "Member's *concerns* about railway [infrastructure].",
            "Transport policy and infrastructure investment."
        ]

        # Get stop words
        stop_words = get_stop_words('moderate')

        # Preprocess
        processed_texts = [preprocess_text(text, stop_words) for text in raw_texts]

        # Analyze ngrams
        ngrams, _ = analyze_ngrams(processed_texts, n_range=(1, 1), max_features=20)

        # Should have results
        assert len(ngrams) > 0

        # Should contain transport-related words
        ngram_words = [word for word, count in ngrams]
        assert any(word in ['transport', 'railway', 'infrastructure', 'policy', 'investment']
                   for word in ngram_words)

    def test_empty_texts_handling(self):
        """Test handling of empty texts in pipeline"""
        texts = ["", "  ", "valid text"]
        stop_words = get_stop_words('moderate')

        processed = [preprocess_text(text, stop_words) for text in texts]
        # Filter out empty results
        processed = [t for t in processed if t.strip()]

        if processed:  # If we have any non-empty texts
            ngrams, _ = analyze_ngrams(processed, n_range=(1, 1))
            # Should not crash, may return empty list
            assert isinstance(ngrams, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
