#!/usr/bin/env python3
"""
Test suite for MP matching system
"""

import unittest
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from mp_matcher import MPMatcher

class TestMPMatcher(unittest.TestCase):
    """Test cases for MP matching"""

    def setUp(self):
        """Create test data"""
        # Create sample MP data
        self.mp_data = pd.DataFrame([
            {'person_name': 'Margaret Thatcher', 'gender_inferred': 'F', 'birth_year': 1925, 'death_year': 2013,
             'aliases_norm': ['m thatcher', 'margaret thatcher', 'mrs thatcher']},
            {'person_name': 'Winston Churchill', 'gender_inferred': 'M', 'birth_year': 1874, 'death_year': 1965,
             'aliases_norm': ['w churchill', 'winston churchill', 'mr churchill']},
            {'person_name': 'Diane Abbott', 'gender_inferred': 'F', 'birth_year': 1953, 'death_year': None,
             'aliases_norm': ['d abbott', 'diane abbott', 'ms abbott']},
            {'person_name': 'Tony Blair', 'gender_inferred': 'M', 'birth_year': 1953, 'death_year': None,
             'aliases_norm': ['t blair', 'tony blair', 'mr blair', 'the prime minister']},
        ])

        self.matcher = MPMatcher(self.mp_data)

    def test_exact_match(self):
        """Test exact name matching"""
        # Exact matches
        self.assertEqual(self.matcher.match('Margaret Thatcher'),
                        ('Margaret Thatcher', 'F', 'exact'))
        self.assertEqual(self.matcher.match('Winston Churchill'),
                        ('Winston Churchill', 'M', 'exact'))

    def test_case_insensitive_match(self):
        """Test case-insensitive matching"""
        self.assertEqual(self.matcher.match('margaret thatcher'),
                        ('Margaret Thatcher', 'F', 'exact'))
        self.assertEqual(self.matcher.match('WINSTON CHURCHILL'),
                        ('Winston Churchill', 'M', 'exact'))

    def test_title_variations(self):
        """Test matching with titles"""
        # Mr/Mrs/Ms variations
        self.assertEqual(self.matcher.match('Mrs. Thatcher'),
                        ('Margaret Thatcher', 'F', 'title'))
        self.assertEqual(self.matcher.match('Mr. Churchill'),
                        ('Winston Churchill', 'M', 'title'))
        self.assertEqual(self.matcher.match('Ms Abbott'),
                        ('Diane Abbott', 'F', 'title'))

    def test_alias_matching(self):
        """Test matching using aliases"""
        self.assertEqual(self.matcher.match('M. Thatcher'),
                        ('Margaret Thatcher', 'F', 'alias'))
        self.assertEqual(self.matcher.match('W. Churchill'),
                        ('Winston Churchill', 'M', 'alias'))

    def test_special_titles(self):
        """Test special parliamentary titles"""
        self.assertEqual(self.matcher.match('The Prime Minister'),
                        ('Tony Blair', 'M', 'alias'))

    def test_no_match(self):
        """Test when no match is found"""
        result = self.matcher.match('Unknown Person')
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])
        self.assertEqual(result[2], 'no_match')

    def test_procedural_speakers(self):
        """Test identification of procedural speakers"""
        self.assertTrue(self.matcher.is_procedural('PROCEDURAL'))
        self.assertTrue(self.matcher.is_procedural('The Speaker'))
        self.assertTrue(self.matcher.is_procedural('The Chairman'))
        self.assertTrue(self.matcher.is_procedural('The Deputy Speaker'))
        self.assertFalse(self.matcher.is_procedural('Mr. Speaker Johnson'))

    def test_normalize_speaker_name(self):
        """Test name normalization"""
        # Remove honorifics
        self.assertEqual(self.matcher.normalize_name('The Rt. Hon. Margaret Thatcher'),
                        'margaret thatcher')
        self.assertEqual(self.matcher.normalize_name('Sir Winston Churchill'),
                        'winston churchill')

        # Handle parliamentary references
        self.assertEqual(self.matcher.normalize_name('Mr. Blair (Sedgefield)'),
                        'blair')
        self.assertEqual(self.matcher.normalize_name('Mrs. Thatcher (Finchley)'),
                        'thatcher')

    def test_batch_matching(self):
        """Test matching multiple speakers at once"""
        speakers = ['Margaret Thatcher', 'Winston Churchill', 'Unknown Person', 'PROCEDURAL']
        results = self.matcher.match_batch(speakers)

        self.assertEqual(len(results), 4)
        self.assertEqual(results[0], ('Margaret Thatcher', 'F', 'exact'))
        self.assertEqual(results[1], ('Winston Churchill', 'M', 'exact'))
        self.assertEqual(results[2], (None, None, 'no_match'))
        self.assertEqual(results[3], (None, None, 'procedural'))

    def test_match_confidence(self):
        """Test confidence scoring of matches"""
        # Exact match should have highest confidence
        self.assertGreater(
            self.matcher.get_match_confidence('Margaret Thatcher'),
            self.matcher.get_match_confidence('Mrs. Thatcher')
        )

        # Title match should have higher confidence than alias
        self.assertGreater(
            self.matcher.get_match_confidence('Mrs. Thatcher'),
            self.matcher.get_match_confidence('M. Thatcher')
        )

if __name__ == '__main__':
    unittest.main()