#!/usr/bin/env python3
"""
Advanced MP Matcher with multiple improvement strategies
Implements constituency matching, title resolution, OCR correction, etc.
"""

import pandas as pd
import numpy as np
import re
from typing import Optional, Tuple, List, Dict, Set
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import Levenshtein

class AdvancedMPMatcher:
    """Enhanced matcher with multiple strategies"""

    def __init__(self, mp_data: pd.DataFrame = None):
        """Initialize with MP data"""
        if mp_data is None:
            mp_data = self._load_default_mp_data()

        self.mp_data = mp_data
        self._preprocess_data()
        self._build_indices()
        self._build_title_database()
        self._build_ocr_corrections()

    def _load_default_mp_data(self) -> pd.DataFrame:
        """Load the default MP gender data"""
        path = Path("data/house_members_gendered_updated.parquet")
        if not path.exists():
            raise FileNotFoundError(f"MP data not found at {path}")
        return pd.read_parquet(path)

    def _preprocess_data(self):
        """Preprocess MP data for better matching"""
        # Add year columns
        self.mp_data['start_year'] = pd.to_datetime(
            self.mp_data['membership_start_date'], errors='coerce'
        ).dt.year
        self.mp_data['end_year'] = pd.to_datetime(
            self.mp_data['membership_end_date'], errors='coerce'
        ).dt.year

        # Extract chamber from organization
        self.mp_data['chamber'] = self.mp_data['organization_id'].apply(
            lambda x: 'Lords' if pd.notna(x) and 'lords' in str(x).lower() else 'Commons'
        )

        # Parse constituencies
        self.mp_data['constituency_list'] = self.mp_data['constituencies'].apply(
            self._parse_constituencies
        )

    def _parse_constituencies(self, const_data):
        """Parse constituency data"""
        if pd.isna(const_data):
            return []

        constituencies = []
        if isinstance(const_data, str):
            try:
                const_list = eval(const_data) if const_data.startswith('[') else [const_data]
                for const in const_list:
                    if isinstance(const, dict) and 'constituency' in const:
                        constituencies.append(const['constituency'])
            except:
                pass
        elif isinstance(const_data, list):
            for const in const_data:
                if isinstance(const, dict) and 'constituency' in const:
                    constituencies.append(const['constituency'])

        return constituencies

    def _build_indices(self):
        """Build various lookup indices"""
        # Temporal + Chamber index
        self.temporal_chamber_index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Constituency index
        self.constituency_index = defaultdict(lambda: defaultdict(list))

        # Name variations index
        self.name_variations = defaultdict(set)

        for _, row in self.mp_data.iterrows():
            if pd.isna(row['person_name']):
                continue

            person = row['person_name']
            gender = row['gender_inferred']
            chamber = row['chamber']
            start_year = row['start_year']
            end_year = row['end_year']

            # Build name variations
            parts = person.split()
            if parts:
                surname = parts[-1].lower()
                # Add full name
                self.name_variations[surname].add(person)

                # Add initials version
                if len(parts) > 1:
                    initials = '.'.join([p[0] for p in parts[:-1]]) + f". {parts[-1]}"
                    self.name_variations[surname].add(initials)

            # Temporal + Chamber index
            if pd.notna(start_year) and pd.notna(end_year):
                for year in range(int(start_year), min(int(end_year) + 1, 2006)):
                    self.temporal_chamber_index[chamber][surname][year].append({
                        'full_name': person,
                        'gender': gender,
                        'row_idx': row.name
                    })

            # Constituency index
            for const in row['constituency_list']:
                if const:
                    const_lower = const.lower()
                    if pd.notna(start_year) and pd.notna(end_year):
                        for year in range(int(start_year), min(int(end_year) + 1, 2006)):
                            self.constituency_index[const_lower][year].append({
                                'full_name': person,
                                'gender': gender
                            })

    def _build_title_database(self):
        """Build database of titles to MPs"""
        self.title_database = {
            # Prime Ministers (simplified - would be more complete)
            'prime minister': [
                (1940, 1945, 'Winston Churchill', 'M'),
                (1945, 1951, 'Clement Attlee', 'M'),
                (1951, 1955, 'Winston Churchill', 'M'),
                (1955, 1957, 'Anthony Eden', 'M'),
                (1957, 1963, 'Harold Macmillan', 'M'),
                (1963, 1964, 'Alec Douglas-Home', 'M'),
                (1964, 1970, 'Harold Wilson', 'M'),
                (1970, 1974, 'Edward Heath', 'M'),
                (1974, 1976, 'Harold Wilson', 'M'),
                (1976, 1979, 'James Callaghan', 'M'),
                (1979, 1990, 'Margaret Thatcher', 'F'),
                (1990, 1997, 'John Major', 'M'),
                (1997, 2007, 'Tony Blair', 'M'),
            ],
            'leader of the opposition': [
                # Would add opposition leaders
            ]
        }

    def _build_ocr_corrections(self):
        """Common OCR errors and corrections"""
        self.ocr_corrections = {
            # Common letter confusions
            'bavies': 'davies',
            'oavies': 'davies',
            'srnith': 'smith',
            'vvilliams': 'williams',
            "o'brien": "o'brien",
            "0'brien": "o'brien",
            # Missing spaces
            'mrsmith': 'mr smith',
            'mrsthatcher': 'mrs thatcher',
        }

    def match_comprehensive(self, speaker: str, date: str, chamber: str = None) -> Dict:
        """
        Comprehensive matching using all available strategies

        Returns:
            Dictionary with match results and confidence scores
        """
        results = {
            'original_speaker': speaker,
            'date': date,
            'chamber': chamber,
            'strategies_tried': [],
            'matches': [],
            'final_match': None,
            'confidence': 0.0,
            'match_type': 'no_match'
        }

        # Clean speaker name
        speaker_clean = self._clean_speaker_name(speaker)
        results['cleaned_speaker'] = speaker_clean

        # Try strategies in order of reliability

        # 1. Title resolution (highest confidence)
        title_match = self._resolve_title(speaker_clean, date)
        if title_match:
            results['strategies_tried'].append('title')
            results['matches'].append(title_match)
            if title_match['confidence'] >= 0.95:
                results['final_match'] = title_match['mp_name']
                results['confidence'] = title_match['confidence']
                results['match_type'] = 'title'
                results['gender'] = title_match['gender']
                return results

        # 2. Constituency matching
        const_match = self._match_by_constituency(speaker_clean, date)
        if const_match:
            results['strategies_tried'].append('constituency')
            results['matches'].append(const_match)
            if const_match['confidence'] >= 0.90:
                results['final_match'] = const_match['mp_name']
                results['confidence'] = const_match['confidence']
                results['match_type'] = 'constituency'
                results['gender'] = const_match['gender']
                return results

        # 3. Temporal + Chamber matching
        temporal_matches = self._match_temporal_chamber(speaker_clean, date, chamber)
        if temporal_matches:
            results['strategies_tried'].append('temporal_chamber')
            results['matches'].extend(temporal_matches)

            # If only one match, high confidence
            if len(temporal_matches) == 1:
                match = temporal_matches[0]
                results['final_match'] = match['mp_name']
                results['confidence'] = match['confidence']
                results['match_type'] = 'temporal_unique'
                results['gender'] = match['gender']
                return results
            elif len(temporal_matches) > 1:
                # Multiple matches - ambiguous
                results['match_type'] = 'ambiguous'
                results['ambiguity_count'] = len(temporal_matches)
                return results

        # 4. Fuzzy matching (last resort, lower confidence)
        fuzzy_match = self._fuzzy_match_conservative(speaker_clean, date, chamber)
        if fuzzy_match:
            results['strategies_tried'].append('fuzzy')
            results['matches'].append(fuzzy_match)
            if fuzzy_match['confidence'] >= 0.6:
                results['final_match'] = fuzzy_match['mp_name']
                results['confidence'] = fuzzy_match['confidence']
                results['match_type'] = 'fuzzy'
                results['gender'] = fuzzy_match.get('gender')
                return results

        return results

    def _clean_speaker_name(self, speaker: str) -> str:
        """Clean and normalize speaker name"""
        if not speaker:
            return ""

        speaker_lower = speaker.lower().strip()

        # Apply OCR corrections
        for error, correction in self.ocr_corrections.items():
            if error in speaker_lower:
                speaker_lower = speaker_lower.replace(error, correction)

        # Fix missing spaces after titles
        speaker_lower = re.sub(r'(mr|mrs|miss|ms|dr|sir)\.?([a-z])', r'\1. \2', speaker_lower)

        return speaker_lower

    def _resolve_title(self, speaker: str, date: str) -> Optional[Dict]:
        """Resolve ministerial titles"""
        speaker_lower = speaker.lower()

        for title, appointments in self.title_database.items():
            if title in speaker_lower:
                try:
                    year = pd.to_datetime(date).year
                    for start_year, end_year, mp_name, gender in appointments:
                        if start_year <= year <= end_year:
                            return {
                                'mp_name': mp_name,
                                'gender': gender,
                                'confidence': 0.95,
                                'method': 'title_resolution'
                            }
                except:
                    pass
        return None

    def _match_by_constituency(self, speaker: str, date: str) -> Optional[Dict]:
        """Match by constituency reference"""
        # Look for "Member for X" or "MP for X" patterns
        patterns = [
            r'member for ([a-z\s]+)',
            r'mp for ([a-z\s]+)',
            r'representative for ([a-z\s]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, speaker.lower())
            if match:
                constituency = match.group(1).strip()
                try:
                    year = pd.to_datetime(date).year
                    if constituency in self.constituency_index:
                        candidates = self.constituency_index[constituency].get(year, [])
                        if len(candidates) == 1:
                            return {
                                'mp_name': candidates[0]['full_name'],
                                'gender': candidates[0]['gender'],
                                'confidence': 0.95,
                                'method': 'constituency'
                            }
                except:
                    pass
        return None

    def _match_temporal_chamber(self, speaker: str, date: str, chamber: str = None) -> List[Dict]:
        """Match using temporal and chamber constraints"""
        if not chamber:
            chamber = 'Commons'  # Default

        # Extract surname
        parts = speaker.split()
        if not parts:
            return []

        # Remove titles
        titles = ['mr', 'mrs', 'miss', 'ms', 'dr', 'sir', 'lord', 'lady']
        clean_parts = [p for p in parts if p.replace('.', '').lower() not in titles]

        if not clean_parts:
            return []

        surname = clean_parts[-1].lower()

        try:
            year = pd.to_datetime(date).year
        except:
            return []

        # Look up in index
        if surname not in self.temporal_chamber_index[chamber]:
            return []

        candidates = self.temporal_chamber_index[chamber][surname].get(year, [])

        # Filter by initials if provided
        if len(clean_parts) > 1:
            initial = clean_parts[0][0].lower()
            filtered = []
            for candidate in candidates:
                mp_name_parts = candidate['full_name'].split()
                if mp_name_parts and mp_name_parts[0][0].lower() == initial:
                    filtered.append(candidate)
            if filtered:
                candidates = filtered

        # Return matches with confidence
        if len(candidates) == 0:
            return []
        elif len(candidates) == 1:
            return [{
                'mp_name': candidates[0]['full_name'],
                'gender': candidates[0]['gender'],
                'confidence': 0.85,
                'method': 'temporal_chamber_unique'
            }]
        else:
            # Multiple candidates - return all with lower confidence
            return [
                {
                    'mp_name': c['full_name'],
                    'gender': c['gender'],
                    'confidence': 0.4 / len(candidates),
                    'method': 'temporal_chamber_ambiguous'
                }
                for c in candidates
            ]

    def _fuzzy_match_conservative(self, speaker: str, date: str, chamber: str = None) -> Optional[Dict]:
        """Conservative fuzzy matching with safeguards"""
        # Only attempt if we have a reasonable name
        parts = speaker.split()
        if len(parts) < 2:
            return None

        surname = parts[-1].lower()

        # Look for similar surnames with edit distance <= 1
        best_match = None
        best_distance = float('inf')

        for known_surname in self.temporal_chamber_index.get(chamber, {}).keys():
            distance = Levenshtein.distance(surname, known_surname)
            if distance <= 1 and distance < best_distance:
                # Verify temporal validity
                try:
                    year = pd.to_datetime(date).year
                    candidates = self.temporal_chamber_index[chamber][known_surname].get(year, [])
                    if candidates:
                        best_match = candidates[0]
                        best_distance = distance
                except:
                    pass

        if best_match and best_distance <= 1:
            # Low confidence for fuzzy matches
            confidence = 0.7 if best_distance == 0 else 0.5
            return {
                'mp_name': best_match['full_name'],
                'gender': best_match['gender'],
                'confidence': confidence,
                'method': f'fuzzy_distance_{best_distance}'
            }

        return None

def test_advanced_matcher():
    """Test the advanced matcher"""
    import time

    print("Loading MP data...")
    mp_data = pd.read_parquet("data/house_members_gendered_updated.parquet")
    matcher = AdvancedMPMatcher(mp_data)

    test_cases = [
        ("Mr. Churchill", "1940-06-01", "Commons"),  # During his PM term
        ("The Prime Minister", "1979-06-01", "Commons"),  # Thatcher as PM
        ("Mrs. Thatcher", "1980-06-15", "Commons"),
        ("Member for Finchley", "1980-06-15", "Commons"),
        ("Mr. Davies", "1950-05-26", "Commons"),
        ("Mr. A. Davies", "1950-05-26", "Commons"),
        ("Bavies", "1950-05-26", "Commons"),  # OCR error
        ("Lord Jenkins", "1990-06-15", "Lords"),
    ]

    print("\n=== ADVANCED MATCHING TESTS ===\n")
    for speaker, date, chamber in test_cases:
        start = time.time()
        result = matcher.match_comprehensive(speaker, date, chamber)
        elapsed = time.time() - start

        print(f"Speaker: {speaker} ({date}, {chamber})")
        print(f"  Cleaned: {result.get('cleaned_speaker')}")
        print(f"  Match type: {result['match_type']}")
        if result['final_match']:
            print(f"  Matched to: {result['final_match']} ({result.get('gender')})")
            print(f"  Confidence: {result['confidence']:.2f}")
        elif result['match_type'] == 'ambiguous':
            print(f"  Ambiguous: {result.get('ambiguity_count')} candidates")
            for match in result['matches'][:3]:
                print(f"    - {match['mp_name']} ({match['gender']})")
        else:
            print(f"  No match found")
        print(f"  Strategies tried: {', '.join(result['strategies_tried'])}")
        print(f"  Time: {elapsed*1000:.1f}ms\n")

if __name__ == "__main__":
    test_advanced_matcher()