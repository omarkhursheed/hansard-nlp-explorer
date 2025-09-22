#!/usr/bin/env python3
"""
MP Matcher: Match speaker names to MPs with known gender
"""

import pandas as pd
import re
from typing import Optional, Tuple, List
from pathlib import Path

class MPMatcher:
    """Match speaker names to MPs with confirmed gender data"""

    def __init__(self, mp_data: pd.DataFrame = None):
        """
        Initialize with MP data

        Args:
            mp_data: DataFrame with columns: person_name, gender_inferred, aliases_norm
        """
        if mp_data is None:
            # Load default data
            mp_data = self._load_default_mp_data()

        self.mp_data = mp_data
        self._build_indices()

    def _load_default_mp_data(self) -> pd.DataFrame:
        """Load the default MP gender data"""
        path = Path("src/hansard/data/house_members_gendered_updated.parquet")
        if not path.exists():
            raise FileNotFoundError(f"MP data not found at {path}")
        return pd.read_parquet(path)

    def _build_indices(self):
        """Build lookup indices for fast matching"""
        # Build name -> (person, gender) index
        self.exact_index = {}
        self.normalized_index = {}
        self.alias_index = {}

        for _, row in self.mp_data.iterrows():
            if pd.isna(row['person_name']) or pd.isna(row['gender_inferred']):
                continue

            person = row['person_name']
            gender = row['gender_inferred']

            # Exact name
            self.exact_index[person.lower()] = (person, gender)

            # Normalized name (last name only)
            normalized = self.normalize_name(person)
            if normalized:
                self.normalized_index[normalized] = (person, gender)

            # Aliases
            aliases = row.get('aliases_norm')
            if aliases is not None:
                try:
                    if isinstance(aliases, (list, tuple, pd.Series)):
                        for alias in aliases:
                            if alias:
                                self.alias_index[alias.lower()] = (person, gender)
                    elif isinstance(aliases, str):
                        # Single alias string
                        self.alias_index[aliases.lower()] = (person, gender)
                except (TypeError, AttributeError):
                    # Skip if can't process aliases
                    pass

        # Parliamentary titles that might map to specific people
        self.special_titles = {}

        # Procedural speaker patterns
        self.procedural_patterns = [
            'PROCEDURAL',
            'The Speaker',
            'The Deputy Speaker',
            'The Chairman',
            'The Deputy Chairman',
            'The Clerk',
            'Madam Speaker',
            'Mr. Speaker',
            'Several Members',
            'Hon. Members',
            'Members'
        ]

    def normalize_name(self, name: str) -> str:
        """
        Normalize a speaker name for matching

        Removes titles, honorifics, and constituency references
        """
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

        # Remove 'the minister', 'secretary', etc. if they appear
        role_prefixes = ['the minister', 'the secretary', 'minister', 'secretary']
        for prefix in role_prefixes:
            if name.startswith(prefix + ' '):
                name = name[len(prefix):].strip()

        return name

    def is_procedural(self, speaker: str) -> bool:
        """Check if speaker is a procedural entry (not a real person)"""
        if not speaker:
            return False

        speaker_upper = speaker.upper()

        # Exact match check for procedural patterns
        for pattern in self.procedural_patterns:
            if speaker_upper == pattern.upper():
                return True

        # Only check partial match for specific cases
        partial_patterns = ['PROCEDURAL', 'SEVERAL MEMBERS', 'HON. MEMBERS']
        for pattern in partial_patterns:
            if pattern in speaker_upper:
                return True

        return False

    def match(self, speaker: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        Match a speaker name to an MP with known gender

        Returns:
            (matched_name, gender, match_type)
            match_type: 'exact', 'title', 'alias', 'normalized', 'procedural', or 'no_match'
        """
        if not speaker:
            return (None, None, 'no_match')

        # Check if procedural
        if self.is_procedural(speaker):
            return (None, None, 'procedural')

        speaker_lower = speaker.lower().strip()

        # Try exact match
        if speaker_lower in self.exact_index:
            person, gender = self.exact_index[speaker_lower]
            return (person, gender, 'exact')

        # Check if this looks like a title variation first
        has_title = any(title in speaker_lower for title in ['mr.', 'mrs.', 'ms.', 'ms ', 'miss', 'dr.'])

        # Try after removing titles (Mr., Mrs., etc.)
        normalized = self.normalize_name(speaker)

        # Check if it's a title variation (Mr. X, Mrs. Y)
        if has_title and normalized:
            # First try to match the normalized version against full names
            for name_lower, (person, gender) in self.exact_index.items():
                # Check if normalized version matches the last name
                if name_lower.endswith(normalized):
                    return (person, gender, 'title')

        # Try normalized index (last name matching)
        if normalized in self.normalized_index:
            person, gender = self.normalized_index[normalized]
            return (person, gender, 'normalized')

        # Try alias match
        if speaker_lower in self.alias_index:
            person, gender = self.alias_index[speaker_lower]
            return (person, gender, 'alias')

        # Try partial alias match (M. Thatcher -> m thatcher)
        simplified = re.sub(r'^([a-z])\.\s+', r'\1 ', speaker_lower)
        if simplified in self.alias_index:
            person, gender = self.alias_index[simplified]
            return (person, gender, 'alias')

        # Check special titles
        if speaker in self.special_titles:
            person, gender = self.special_titles[speaker]
            return (person, gender, 'alias')

        return (None, None, 'no_match')

    def match_batch(self, speakers: List[str]) -> List[Tuple[Optional[str], Optional[str], str]]:
        """Match multiple speakers at once"""
        return [self.match(speaker) for speaker in speakers]

    def get_match_confidence(self, speaker: str) -> float:
        """
        Get confidence score for a match (0-1)

        exact: 1.0
        title: 0.9
        normalized: 0.7
        alias: 0.8
        no_match/procedural: 0.0
        """
        _, _, match_type = self.match(speaker)

        confidence_scores = {
            'exact': 1.0,
            'title': 0.9,
            'alias': 0.8,
            'normalized': 0.7,
            'procedural': 0.0,
            'no_match': 0.0
        }

        return confidence_scores.get(match_type, 0.0)

    def get_match_statistics(self, speakers: List[str]) -> dict:
        """Get statistics about matching success for a list of speakers"""
        results = self.match_batch(speakers)

        stats = {
            'total': len(speakers),
            'matched': sum(1 for r in results if r[2] not in ['no_match', 'procedural']),
            'procedural': sum(1 for r in results if r[2] == 'procedural'),
            'unmatched': sum(1 for r in results if r[2] == 'no_match'),
            'match_types': {}
        }

        for _, _, match_type in results:
            stats['match_types'][match_type] = stats['match_types'].get(match_type, 0) + 1

        stats['match_rate'] = stats['matched'] / stats['total'] if stats['total'] > 0 else 0

        return stats