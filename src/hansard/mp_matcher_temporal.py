#!/usr/bin/env python3
"""
Improved MP Matcher using temporal context to resolve ambiguities
"""

import pandas as pd
import re
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from collections import defaultdict

class TemporalMPMatcher:
    """Match speaker names to MPs using temporal context"""

    def __init__(self, mp_data: pd.DataFrame = None):
        """Initialize with MP data including temporal information"""
        if mp_data is None:
            mp_data = self._load_default_mp_data()

        self.mp_data = mp_data

        # Add year columns
        self.mp_data['start_year'] = pd.to_datetime(mp_data['membership_start_date']).dt.year
        self.mp_data['end_year'] = pd.to_datetime(mp_data['membership_end_date']).dt.year

        self._build_temporal_indices()

    def _load_default_mp_data(self) -> pd.DataFrame:
        """Load the default MP gender data"""
        path = Path("data/house_members_gendered_updated.parquet")
        if not path.exists():
            raise FileNotFoundError(f"MP data not found at {path}")
        return pd.read_parquet(path)

    def _build_temporal_indices(self):
        """Build temporal lookup indices for fast matching"""
        # Group MPs by surname and active years
        self.temporal_index = defaultdict(lambda: defaultdict(list))

        for _, row in self.mp_data.iterrows():
            if pd.isna(row['person_name']) or pd.isna(row['gender_inferred']):
                continue

            person = row['person_name']
            gender = row['gender_inferred']
            start_year = row['start_year']
            end_year = row['end_year']

            # Extract surname
            parts = person.split()
            if parts:
                surname = parts[-1].lower()

                # Add to temporal index for each active year
                if pd.notna(start_year) and pd.notna(end_year):
                    for year in range(int(start_year), int(end_year) + 1):
                        self.temporal_index[surname][year].append({
                            'full_name': person,
                            'gender': gender,
                            'start': int(start_year),
                            'end': int(end_year)
                        })

        # Build procedural patterns
        self.procedural_patterns = [
            'PROCEDURAL', 'The Speaker', 'The Deputy Speaker',
            'The Chairman', 'The Deputy Chairman', 'The Clerk',
            'Madam Speaker', 'Mr. Speaker', 'Several Members',
            'Hon. Members', 'Members'
        ]

    def normalize_name(self, name: str) -> str:
        """Normalize a speaker name for matching"""
        if not name:
            return ""

        name = name.lower().strip()

        # Remove common honorifics
        honorifics = [
            'the rt. hon.', 'the right hon.', 'the hon.',
            'rt. hon.', 'right hon.', 'hon.',
            'sir', 'dame', 'lord', 'lady', 'baroness', 'viscount',
            'earl', 'duke', 'mr.', 'mrs.', 'ms.', 'miss', 'dr.',
            'mr', 'mrs', 'ms', 'miss', 'dr'
        ]
        for hon in honorifics:
            if name.startswith(hon + ' '):
                name = name[len(hon):].strip()

        # Remove constituency in parentheses
        name = re.sub(r'\([^)]+\)', '', name).strip()

        return name

    def is_procedural(self, speaker: str) -> bool:
        """Check if speaker is a procedural entry"""
        if not speaker:
            return False

        speaker_upper = speaker.upper()
        for pattern in self.procedural_patterns:
            if speaker_upper == pattern.upper():
                return True

        # Check partial match for specific cases
        partial_patterns = ['PROCEDURAL', 'SEVERAL MEMBERS', 'HON. MEMBERS']
        for pattern in partial_patterns:
            if pattern in speaker_upper:
                return True

        return False

    def match_temporal(self, speaker: str, year: int,
                      chamber: str = None) -> List[Dict]:
        """
        Match a speaker name to MPs active in a specific year

        Returns list of possible matches with confidence scores
        """
        if not speaker:
            return []

        if self.is_procedural(speaker):
            return [{'type': 'procedural', 'confidence': 1.0}]

        # Extract surname from speaker
        normalized = self.normalize_name(speaker)
        if not normalized:
            return []

        surname = normalized.split()[-1] if normalized.split() else normalized

        # Look up MPs with this surname active in this year
        if surname not in self.temporal_index:
            return []

        if year not in self.temporal_index[surname]:
            # Try nearby years (MP data might be slightly off)
            candidates = []
            for offset in [0, 1, -1, 2, -2]:
                check_year = year + offset
                if check_year in self.temporal_index[surname]:
                    candidates.extend(self.temporal_index[surname][check_year])
                    if candidates:
                        break
        else:
            candidates = self.temporal_index[surname][year]

        if not candidates:
            return []

        # If only one candidate, high confidence
        if len(candidates) == 1:
            return [{
                'matched_name': candidates[0]['full_name'],
                'gender': candidates[0]['gender'],
                'confidence': 0.95,
                'match_type': 'temporal_unique',
                'active_years': f"{candidates[0]['start']}-{candidates[0]['end']}"
            }]

        # Multiple candidates - return all with lower confidence
        results = []
        for candidate in candidates:
            results.append({
                'matched_name': candidate['full_name'],
                'gender': candidate['gender'],
                'confidence': 0.5 / len(candidates),  # Split confidence
                'match_type': 'temporal_ambiguous',
                'active_years': f"{candidate['start']}-{candidate['end']}",
                'ambiguity_count': len(candidates)
            })

        return results

    def match_with_fallback(self, speaker: str, year: int = None) -> Tuple[Optional[str], Optional[str], str, float]:
        """
        Match with temporal context if available, fallback to best guess

        Returns: (matched_name, gender, match_type, confidence)
        """
        # Try temporal matching first if year provided
        if year:
            temporal_matches = self.match_temporal(speaker, year)

            if temporal_matches:
                if temporal_matches[0].get('type') == 'procedural':
                    return (None, None, 'procedural', 1.0)

                # Return highest confidence match
                best_match = max(temporal_matches, key=lambda x: x.get('confidence', 0))
                return (
                    best_match.get('matched_name'),
                    best_match.get('gender'),
                    best_match.get('match_type'),
                    best_match.get('confidence', 0)
                )

        # Fallback: return no match rather than incorrect match
        return (None, None, 'no_match', 0.0)

    def analyze_ambiguity_by_year(self, year: int) -> Dict:
        """Analyze surname ambiguity for a specific year"""
        # Get all MPs active in this year
        active_mps = self.mp_data[
            (self.mp_data['start_year'] <= year) &
            (self.mp_data['end_year'] >= year)
        ]

        # Count by surname
        surname_counts = defaultdict(list)
        for _, mp in active_mps.iterrows():
            if pd.notna(mp['person_name']):
                parts = mp['person_name'].split()
                if parts:
                    surname = parts[-1]
                    surname_counts[surname].append(mp['person_name'])

        # Calculate ambiguity stats
        total_mps = len(active_mps)
        ambiguous_surnames = {s: names for s, names in surname_counts.items() if len(names) > 1}
        unique_surnames = {s: names for s, names in surname_counts.items() if len(names) == 1}

        return {
            'year': year,
            'total_mps_active': total_mps,
            'unique_surnames': len(unique_surnames),
            'ambiguous_surnames': len(ambiguous_surnames),
            'most_ambiguous': sorted(ambiguous_surnames.items(), key=lambda x: len(x[1]), reverse=True)[:5],
            'ambiguity_rate': len(ambiguous_surnames) / (len(unique_surnames) + len(ambiguous_surnames))
        }


def test_temporal_matcher():
    """Test the temporal matcher with examples"""

    print("Loading MP data...")
    mp_data = pd.read_parquet("data/house_members_gendered_updated.parquet")
    matcher = TemporalMPMatcher(mp_data)

    # Test cases
    test_cases = [
        ("Mr. Davies", 1950),
        ("Mr. Davies", 2000),
        ("Mr. Wilson", 1950),
        ("Mr. Wilson", 1970),
        ("Mrs. Thatcher", 1980),
        ("Mr. Blair", 1995)
    ]

    print("\n=== TEMPORAL MATCHING TESTS ===")
    for speaker, year in test_cases:
        matches = matcher.match_temporal(speaker, year)
        print(f"\n{speaker} in {year}:")
        if not matches:
            print("  No matches found")
        else:
            for match in matches[:3]:  # Show top 3
                if match.get('type') == 'procedural':
                    print("  [PROCEDURAL]")
                else:
                    print(f"  {match['matched_name']} ({match['gender']}) "
                          f"- conf: {match['confidence']:.2f} "
                          f"[{match.get('active_years', '')}]")
                    if match.get('ambiguity_count'):
                        print(f"    (1 of {match['ambiguity_count']} possible matches)")

    # Analyze ambiguity over time
    print("\n=== AMBIGUITY ANALYSIS BY DECADE ===")
    for year in [1850, 1900, 1950, 2000]:
        stats = matcher.analyze_ambiguity_by_year(year)
        print(f"\n{year}:")
        print(f"  Active MPs: {stats['total_mps_active']}")
        print(f"  Unique surnames: {stats['unique_surnames']}")
        print(f"  Ambiguous surnames: {stats['ambiguous_surnames']}")
        print(f"  Ambiguity rate: {100*stats['ambiguity_rate']:.1f}%")
        print(f"  Most ambiguous:")
        for surname, names in stats['most_ambiguous']:
            print(f"    {surname}: {len(names)} MPs")


if __name__ == "__main__":
    test_temporal_matcher()