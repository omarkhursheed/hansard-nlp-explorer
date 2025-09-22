#!/usr/bin/env python3
"""
Create filtered dataset for gender analysis
Only includes debates with at least one confirmed MP match
Generates turn-wise analysis with gender attribution
"""

import pandas as pd
import numpy as np
from pathlib import Path
from mp_matcher_corrected import CorrectedMPMatcher
import json
import re
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import hashlib
from datetime import datetime

class GenderAnalysisDatasetCreator:
    """Create datasets for gender analysis of parliamentary debates"""

    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize dataset creator

        Args:
            confidence_threshold: Minimum confidence for "certain" match
        """
        self.confidence_threshold = confidence_threshold

        # Load MP matcher
        mp_data = pd.read_parquet("data/house_members_gendered_updated.parquet")
        self.matcher = CorrectedMPMatcher(mp_data)

        # Setup paths
        self.output_dir = Path("gender_analysis_data")
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "turns").mkdir(exist_ok=True)

        # Statistics tracking
        self.stats = {
            'total_debates_processed': 0,
            'debates_with_confirmed_mps': 0,
            'debates_with_female_participation': 0,
            'total_turns_analyzed': 0,
            'female_word_count': 0,
            'male_word_count': 0,
            'ambiguous_word_count': 0,
            'unmatched_word_count': 0
        }

        self.coverage_by_year = {}

    def process_all_years(self, sample_years: List[int] = None):
        """Process all available years or a sample"""

        # Get available years
        metadata_dir = Path("data/processed_fixed/metadata")
        if sample_years:
            years = sample_years
        else:
            years = sorted([
                int(f.stem.split('_')[1])
                for f in metadata_dir.glob("debates_*.parquet")
                if 'master' not in f.stem
            ])

        print(f"Processing {len(years)} years of data")

        all_debate_records = []

        for year in tqdm(years, desc="Processing years"):
            year_records = self.process_year(year)
            all_debate_records.extend(year_records)

        # Save combined debates dataset
        if all_debate_records:
            debates_df = pd.DataFrame(all_debate_records)
            debates_df.to_parquet(self.output_dir / "debates_with_confirmed_mps.parquet")
            print(f"Saved {len(debates_df)} debates with confirmed MPs")

        # Save metadata
        self.save_metadata()

    def process_year(self, year: int) -> List[Dict]:
        """Process all debates in a year"""

        debate_file = Path(f"data/processed_fixed/metadata/debates_{year}.parquet")
        speaker_file = Path(f"data/processed_fixed/metadata/speakers_{year}.parquet")
        content_file = Path(f"data/processed_fixed/content/{year}/debates_{year}.jsonl")

        if not debate_file.exists():
            return []

        # Load data
        debates_df = pd.read_parquet(debate_file)
        speakers_df = pd.read_parquet(speaker_file) if speaker_file.exists() else None

        year_records = []
        year_stats = {
            'debates': 0,
            'with_confirmed_mps': 0,
            'with_female': 0,
            'female_mps': set()
        }

        # Create year directory for turns
        year_turns_dir = self.output_dir / "turns" / str(year)
        year_turns_dir.mkdir(exist_ok=True)

        # Process each debate
        for _, debate in debates_df.iterrows():
            self.stats['total_debates_processed'] += 1
            year_stats['debates'] += 1

            # Match speakers for this debate
            speaker_matches = self.match_debate_speakers(debate, year)

            # Check if has confirmed MPs (including high-confidence temporal matches)
            confirmed_mps = [s for s in speaker_matches
                           if s['match_category'] == 'certain' or
                           (s['match_type'] == 'temporal_unique' and s['confidence'] >= self.confidence_threshold)]
            if not confirmed_mps:
                continue  # Skip debates with no confirmed MPs

            # Create debate record
            debate_record = self.create_debate_record(debate, speaker_matches, year)

            if debate_record['has_female_speaker']:
                self.stats['debates_with_female_participation'] += 1
                year_stats['with_female'] += 1
                year_stats['female_mps'].update(debate_record['female_mp_names'])

            if debate_record['confirmed_mp_count'] > 0:
                self.stats['debates_with_confirmed_mps'] += 1
                year_stats['with_confirmed_mps'] += 1
                year_records.append(debate_record)

            # Process turns if content available
            if content_file.exists():
                self.process_debate_turns(debate, speaker_matches, year)

        # Store year coverage
        year_stats['female_mps'] = list(year_stats['female_mps'])
        self.coverage_by_year[year] = year_stats

        return year_records

    def match_debate_speakers(self, debate: pd.Series, year: int) -> List[Dict]:
        """Match all speakers in a debate"""

        speaker_matches = []
        speakers = debate.get('speakers', [])
        date = debate.get('reference_date', f'{year}-01-01')
        chamber = debate.get('chamber', 'Commons')

        if isinstance(speakers, (list, np.ndarray)):
            for speaker in speakers:
                if pd.notna(speaker):
                    match_result = self.matcher.match_comprehensive(
                        str(speaker), date, chamber
                    )

                    speaker_match = {
                        'speaker_raw': speaker,
                        'matched_mp': match_result.get('final_match'),
                        'gender': match_result.get('gender'),
                        'confidence': match_result.get('confidence', 0),
                        'match_type': match_result['match_type'],
                        'ambiguity_count': match_result.get('ambiguity_count', 0)
                    }

                    # Classify match type
                    if match_result['match_type'] == 'procedural':
                        speaker_match['match_category'] = 'procedural'
                    elif match_result['match_type'] == 'ambiguous':
                        speaker_match['match_category'] = 'ambiguous'
                    elif match_result['match_type'] == 'no_match':
                        speaker_match['match_category'] = 'unmatched'
                    elif match_result['match_type'] in ['temporal_unique', 'title', 'constituency'] and \
                         match_result.get('confidence', 0) >= self.confidence_threshold:
                        speaker_match['match_category'] = 'certain'
                    else:
                        speaker_match['match_category'] = 'low_confidence'

                    speaker_matches.append(speaker_match)

        return speaker_matches

    def create_debate_record(self, debate: pd.Series, speaker_matches: List[Dict], year: int) -> Dict:
        """Create a debate record with gender analysis metadata"""

        # Generate unique debate ID
        debate_id = hashlib.md5(
            f"{debate['file_path']}_{debate.get('reference_date', '')}".encode()
        ).hexdigest()[:16]

        # Count by category
        confirmed_mps = [s for s in speaker_matches if s['match_category'] == 'certain']
        female_mps = [s for s in confirmed_mps if s['gender'] == 'F']
        male_mps = [s for s in confirmed_mps if s['gender'] == 'M']

        return {
            'debate_id': debate_id,
            'year': year,
            'file_path': debate['file_path'],
            'reference_date': debate.get('reference_date'),
            'chamber': debate.get('chamber'),
            'title': debate.get('title'),
            'topic': debate.get('debate_topic'),

            # MP participation metrics
            'total_speakers': len(speaker_matches),
            'confirmed_mp_count': len(confirmed_mps),
            'female_mp_count': len(female_mps),
            'male_mp_count': len(male_mps),
            'ambiguous_mp_count': len([s for s in speaker_matches if s['match_category'] == 'ambiguous']),
            'unmatched_count': len([s for s in speaker_matches if s['match_category'] == 'unmatched']),

            # Gender flags
            'has_female_speaker': len(female_mps) > 0,
            'has_male_speaker': len(male_mps) > 0,
            'is_mixed_gender': len(female_mps) > 0 and len(male_mps) > 0,
            'female_mp_names': [s['matched_mp'] for s in female_mps],

            # Content metrics
            'total_word_count': debate.get('word_count', 0),
            'total_turns': len(speaker_matches)
        }

    def process_debate_turns(self, debate: pd.Series, speaker_matches: List[Dict], year: int):
        """Process turn-wise conversation data"""

        content_file = Path(f"data/processed_fixed/content/{year}/debates_{year}.jsonl")
        if not content_file.exists():
            return

        # Find this debate in content
        with open(content_file, 'r') as f:
            for line in f:
                content = json.loads(line)
                if content['file_path'] == debate['file_path']:
                    self.extract_turns(content, speaker_matches, debate, year)
                    break

    def extract_turns(self, content: Dict, speaker_matches: List[Dict], debate: pd.Series, year: int):
        """Extract turn-wise data from debate content"""

        lines = content.get('lines', [])
        if not lines:
            return

        # Generate debate ID
        debate_id = hashlib.md5(
            f"{debate['file_path']}_{debate.get('reference_date', '')}".encode()
        ).hexdigest()[:16]

        turns = []
        current_turn = None
        current_text = []
        turn_number = 0

        # Create speaker lookup
        speaker_lookup = {s['speaker_raw']: s for s in speaker_matches}

        for line in lines:
            line = str(line).strip()

            # Check if line is a speaker transition
            is_speaker = self.is_speaker_line(line)

            if is_speaker:
                # Save previous turn if exists
                if current_turn and current_text:
                    text = ' '.join(current_text)
                    word_count = len(text.split())

                    # Determine if interruption
                    is_interruption = word_count < 50

                    turn_data = {
                        'debate_id': debate_id,
                        'turn_number': turn_number,
                        'speaker_raw': current_turn['speaker_raw'],
                        'matched_mp': current_turn.get('matched_mp'),
                        'match_confidence': current_turn.get('confidence', 0),
                        'match_type': current_turn.get('match_category'),
                        'gender': current_turn.get('gender'),
                        'gender_confidence': 1.0 if current_turn.get('match_category') == 'certain' else 0.5,
                        'text': text,
                        'word_count': word_count,
                        'line_count': len(current_text),
                        'is_interruption': is_interruption,
                        'is_question': text.rstrip().endswith('?')
                    }

                    turns.append(turn_data)
                    self.stats['total_turns_analyzed'] += 1

                    # Track word counts by gender
                    if current_turn.get('gender') == 'F':
                        self.stats['female_word_count'] += word_count
                    elif current_turn.get('gender') == 'M':
                        self.stats['male_word_count'] += word_count
                    elif current_turn.get('match_category') == 'ambiguous':
                        self.stats['ambiguous_word_count'] += word_count
                    else:
                        self.stats['unmatched_word_count'] += word_count

                # Start new turn
                speaker_name = self.extract_speaker_name(line)
                current_turn = speaker_lookup.get(speaker_name, {
                    'speaker_raw': speaker_name,
                    'match_category': 'unknown'
                })
                current_text = []
                turn_number += 1
            else:
                # Add to current turn text
                if current_turn and line and not line.startswith('§'):
                    # Filter out metadata lines
                    if not re.match(r'^\d+$', line):  # Page numbers
                        current_text.append(line)

        # Save final turn
        if current_turn and current_text:
            text = ' '.join(current_text)
            word_count = len(text.split())

            turn_data = {
                'debate_id': debate_id,
                'turn_number': turn_number,
                'speaker_raw': current_turn.get('speaker_raw', ''),
                'matched_mp': current_turn.get('matched_mp'),
                'match_confidence': current_turn.get('confidence', 0),
                'match_type': current_turn.get('match_category'),
                'gender': current_turn.get('gender'),
                'gender_confidence': 1.0 if current_turn.get('match_category') == 'certain' else 0.5,
                'text': text,
                'word_count': word_count,
                'line_count': len(current_text),
                'is_interruption': word_count < 50,
                'is_question': text.rstrip().endswith('?')
            }

            turns.append(turn_data)
            self.stats['total_turns_analyzed'] += 1

        # Save turns data
        if turns:
            turns_df = pd.DataFrame(turns)
            year_dir = self.output_dir / "turns" / str(year)
            turns_df.to_parquet(year_dir / f"debate_turns_{debate_id}.parquet")

    def is_speaker_line(self, line: str) -> bool:
        """Determine if a line indicates a speaker transition"""

        # Common speaker patterns
        patterns = [
            r'^(Mr\.|Mrs\.|Miss|Ms\.|Dr\.|Sir|Lord|Lady|Baroness)',
            r'^The (Prime Minister|Chancellor|Secretary|Minister)',
            r'^(Colonel|Major|Captain|Lieutenant)',
            r'^HON\. MEMBERS',
            r'^Several Members'
        ]

        for pattern in patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True

        return False

    def extract_speaker_name(self, line: str) -> str:
        """Extract speaker name from a speaker line"""

        # Remove parenthetical content
        name = re.sub(r'\([^)]*\)', '', line).strip()

        # Remove trailing colons
        name = name.rstrip(':')

        return name

    def save_metadata(self):
        """Save analysis metadata"""

        metadata = {
            'creation_date': datetime.now().isoformat(),
            'mp_matcher_version': 'corrected_v1',
            'confidence_threshold': self.confidence_threshold,
            'statistics': self.stats,
            'coverage_by_year': self.coverage_by_year,
            'data_quality_metrics': self.calculate_quality_metrics()
        }

        with open(self.output_dir / "analysis_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def calculate_quality_metrics(self) -> Dict:
        """Calculate data quality metrics"""

        total_speakers = self.stats.get('total_turns_analyzed', 0)
        if total_speakers == 0:
            return {}

        total_words = (
            self.stats['female_word_count'] +
            self.stats['male_word_count'] +
            self.stats['ambiguous_word_count'] +
            self.stats['unmatched_word_count']
        )

        if total_words == 0:
            return {}

        return {
            'certain_speaker_rate': (self.stats['female_word_count'] + self.stats['male_word_count']) / total_words,
            'ambiguous_speaker_rate': self.stats['ambiguous_word_count'] / total_words,
            'unmatched_speaker_rate': self.stats['unmatched_word_count'] / total_words,
            'female_word_proportion': self.stats['female_word_count'] / total_words if total_words > 0 else 0,
            'male_word_proportion': self.stats['male_word_count'] / total_words if total_words > 0 else 0
        }

def main():
    """Main execution"""

    print("=" * 70)
    print("CREATING GENDER ANALYSIS DATASET")
    print("=" * 70)

    creator = GenderAnalysisDatasetCreator(confidence_threshold=0.7)

    # Process sample years first for testing
    sample_years = [1920, 1950, 1980, 2000]

    print(f"\nProcessing sample years: {sample_years}")
    creator.process_all_years(sample_years)

    # Print summary
    print("\n" + "=" * 70)
    print("DATASET CREATION SUMMARY")
    print("=" * 70)

    stats = creator.stats
    print(f"\nDebates processed: {stats['total_debates_processed']}")
    print(f"Debates with confirmed MPs: {stats['debates_with_confirmed_mps']}")
    print(f"Debates with female participation: {stats['debates_with_female_participation']}")
    print(f"Total turns analyzed: {stats['total_turns_analyzed']}")

    total_words = (
        stats['female_word_count'] +
        stats['male_word_count'] +
        stats['ambiguous_word_count'] +
        stats['unmatched_word_count']
    )

    if total_words > 0:
        print(f"\nWord count by gender:")
        print(f"  Female: {stats['female_word_count']:,} ({100*stats['female_word_count']/total_words:.1f}%)")
        print(f"  Male: {stats['male_word_count']:,} ({100*stats['male_word_count']/total_words:.1f}%)")
        print(f"  Ambiguous: {stats['ambiguous_word_count']:,} ({100*stats['ambiguous_word_count']/total_words:.1f}%)")
        print(f"  Unmatched: {stats['unmatched_word_count']:,} ({100*stats['unmatched_word_count']/total_words:.1f}%)")

    print(f"\nData saved to: gender_analysis_data/")

if __name__ == "__main__":
    main()