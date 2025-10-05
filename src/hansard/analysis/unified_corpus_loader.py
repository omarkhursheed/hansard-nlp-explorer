#!/usr/bin/env python3
"""
Unified Corpus Loader for Hansard Parliamentary Debates

Single source of truth for loading data across all analysis scripts.
Supports both gender-matched and overall corpus datasets with consistent
stratified sampling to maintain temporal distribution.

Dataset Types:
    - gender: Gender-matched dataset with speaker details (GENDER_ENHANCED_DATA)
    - overall: Full corpus debates (PROCESSED_FIXED)
    - speakers: Deduplicated speakers dataset (for temporal analysis)

Usage:
    from unified_corpus_loader import UnifiedCorpusLoader

    # Load gender-matched data
    loader = UnifiedCorpusLoader(dataset_type='gender')
    data = loader.load_debates(year_range=(1920, 1930), sample_size=5000)

    # Load overall corpus
    loader = UnifiedCorpusLoader(dataset_type='overall')
    debates = loader.load_debates(year_range=(1920, 1930), sample_size=5000)
"""

import json
import os
import re
import random
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict
import pandas as pd
import numpy as np

# Import path configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.path_config import Paths


class UnifiedCorpusLoader:
    """
    Unified data loading for Hansard parliamentary debates.

    Provides consistent loading across all analysis scripts with stratified
    sampling to maintain temporal distribution.
    """

    def __init__(self, dataset_type: str = 'gender'):
        """
        Initialize loader for specified dataset type.

        Args:
            dataset_type: One of 'gender', 'overall', or 'speakers'

        Raises:
            ValueError: If dataset_type is not recognized
        """
        if dataset_type not in ['gender', 'overall', 'speakers']:
            raise ValueError(f"Unknown dataset type: {dataset_type}. "
                           f"Use 'gender', 'overall', or 'speakers'")

        self.dataset_type = dataset_type
        self.data_dir = self._get_data_dir(dataset_type)

        # Verify data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"For gender dataset: run data generation first\n"
                f"For overall dataset: check PROCESSED_FIXED path"
            )

    def _get_data_dir(self, dataset_type: str) -> Path:
        """Get data directory for dataset type"""
        if dataset_type == 'gender':
            return Paths.GENDER_ENHANCED_DATA
        elif dataset_type == 'overall':
            return Paths.PROCESSED_FIXED
        elif dataset_type == 'speakers':
            return Paths.DATA_DIR
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def load_debates(self, year_range: Optional[Tuple[int, int]] = None,
                    sample_size: Optional[int] = None,
                    stratified: bool = True,
                    use_derived: bool = True) -> Union[Dict, List]:
        """
        Load debates with optional filtering and sampling.

        Args:
            year_range: Tuple of (start_year, end_year) or None for all years
            sample_size: Number of speeches/debates to sample, or None for all
            stratified: Maintain year distribution when sampling (recommended)
            use_derived: For gender dataset, use derived flat speeches if available (10x faster)

        Returns:
            Dict for gender dataset, List for overall dataset
        """
        if self.dataset_type == 'gender':
            # Try derived speeches first (much faster)
            if use_derived:
                derived_dir = Paths.DATA_DIR / 'derived' / 'gender_speeches'
                if derived_dir.exists() and list(derived_dir.glob("speeches_*.parquet")):
                    return self.load_gender_speeches_derived(year_range, sample_size, stratified)
                else:
                    print("Derived speeches not found, using nested extraction...")

            return self.load_gender_matched(year_range, sample_size, stratified)
        elif self.dataset_type == 'overall':
            return self.load_overall_corpus(year_range, sample_size, stratified)
        elif self.dataset_type == 'speakers':
            return self.load_speakers()
        else:
            raise ValueError(f"Cannot load dataset type: {self.dataset_type}")

    def load_gender_speeches_derived(self, year_range: Optional[Tuple[int, int]] = None,
                                     sample_size: Optional[int] = None,
                                     stratified: bool = True) -> Dict:
        """
        Load derived gender speeches dataset (flat speech-level data).

        Much faster than loading nested structure.
        """
        derived_dir = Paths.DATA_DIR / 'derived' / 'gender_speeches'
        print(f"Loading derived gender speeches from {derived_dir}")

        if year_range:
            start_year, end_year = year_range
        else:
            start_year, end_year = 1803, 2005

        all_speeches = []
        years_loaded = []

        for year in range(start_year, end_year + 1):
            speech_file = derived_dir / f"speeches_{year}.parquet"
            if not speech_file.exists():
                continue

            try:
                df = pd.read_parquet(speech_file)
                all_speeches.append(df)
                years_loaded.append(year)
                print(f"  {year}: {len(df):,} speeches")
            except Exception as e:
                print(f"  {year}: Error - {e}")

        if not all_speeches:
            raise FileNotFoundError(f"No speech files found for {start_year}-{end_year}")

        speeches_df = pd.concat(all_speeches, ignore_index=True)
        print(f"\nLoaded {len(speeches_df):,} total speeches from {len(years_loaded)} years")

        if sample_size and len(speeches_df) > sample_size:
            if stratified:
                sampled = speeches_df.groupby('year', group_keys=False).apply(
                    lambda x: x.sample(n=min(len(x), int(len(x) / len(speeches_df) * sample_size)), random_state=42)
                )
                if len(sampled) > sample_size:
                    sampled = sampled.sample(n=sample_size, random_state=42)
                speeches_df = sampled
                print(f"Stratified sampling: {len(speeches_df):,} speeches")
            else:
                speeches_df = speeches_df.sample(n=sample_size, random_state=42)

        male_df = speeches_df[speeches_df['gender'] == 'm']
        female_df = speeches_df[speeches_df['gender'] == 'f']

        data = {
            'male_speeches': male_df['text'].tolist(),
            'female_speeches': female_df['text'].tolist(),
            'temporal_data': [],
            'metadata': {'dataset_type': 'gender', 'years_processed': years_loaded, 'source': 'derived'}
        }

        for year in years_loaded:
            year_data = speeches_df[speeches_df['year'] == year]
            data['temporal_data'].append({
                'year': year,
                'male_speeches': len(year_data[year_data['gender'] == 'm']),
                'female_speeches': len(year_data[year_data['gender'] == 'f'])
            })

        print(f"Converted: {len(data['male_speeches'])} male, {len(data['female_speeches'])} female")
        return data

    def load_gender_matched(self, year_range: Optional[Tuple[int, int]] = None,
                          sample_size: Optional[int] = None,
                          stratified: bool = True) -> Dict:
        """
        Load gender-matched dataset with speaker details.

        Returns dict with structure:
        {
            'male_speeches': List[str],
            'female_speeches': List[str],
            'temporal_data': List[dict],  # year, male_count, female_count
            'metadata': dict
        }

        Args:
            year_range: Tuple of (start_year, end_year) or None
            sample_size: Number of speeches to sample (total across both genders)
            stratified: Use year-stratified sampling

        Returns:
            Dict with male_speeches, female_speeches, temporal_data, metadata
        """
        print(f"Loading gender-matched dataset from {self.data_dir}")

        # Get year files
        if year_range:
            start_year, end_year = year_range
            year_files = Paths.get_year_files(start_year, end_year)
        else:
            year_files = Paths.get_year_files()

        year_files = [str(f) for f in year_files]

        if not year_files:
            raise FileNotFoundError(f"No data files found in {self.data_dir}")

        print(f"Processing {len(year_files)} year files...")

        # Collect data with year tracking
        all_data = {
            'male_speeches': [],
            'female_speeches': [],
            'male_speeches_by_year': {},
            'female_speeches_by_year': {},
            'temporal_data': [],
            'metadata': {
                'dataset_type': 'gender',
                'years_processed': []
            }
        }

        # Load all files
        for file_path in year_files:
            try:
                df = pd.read_parquet(file_path)
                year_match = re.search(r'debates_(\d{4})_enhanced\.parquet',
                                     os.path.basename(file_path))
                year = int(year_match.group(1)) if year_match else 0

                year_male_count = 0
                year_female_count = 0

                # Process speech segments
                for _, row in df.iterrows():
                    if 'speech_segments' not in row or row['speech_segments'] is None:
                        continue

                    # Build speaker gender map
                    speaker_details = row.get('speaker_details', [])
                    speaker_gender_map = {}

                    if isinstance(speaker_details, (list, np.ndarray)):
                        for detail in speaker_details:
                            if isinstance(detail, dict):
                                name = detail.get('original_name', '')
                                gender = detail.get('gender', '').lower()
                                if name and gender in ['m', 'f']:
                                    speaker_gender_map[name] = 'male' if gender == 'm' else 'female'

                    # Process segments
                    segments = row['speech_segments']
                    if isinstance(segments, (list, np.ndarray)):
                        for segment in segments:
                            if not isinstance(segment, dict):
                                continue

                            speaker = segment.get('speaker', '')
                            text = segment.get('text', '')

                            # Map to gender
                            gender = speaker_gender_map.get(speaker)
                            if not gender:
                                # Try fuzzy matching
                                for orig_name, g in speaker_gender_map.items():
                                    if speaker in orig_name or orig_name in speaker:
                                        gender = g
                                        break

                            # Add to appropriate list
                            if gender == 'male' and text:
                                all_data['male_speeches'].append(text)
                                if year not in all_data['male_speeches_by_year']:
                                    all_data['male_speeches_by_year'][year] = []
                                all_data['male_speeches_by_year'][year].append(text)
                                year_male_count += 1
                            elif gender == 'female' and text:
                                all_data['female_speeches'].append(text)
                                if year not in all_data['female_speeches_by_year']:
                                    all_data['female_speeches_by_year'][year] = []
                                all_data['female_speeches_by_year'][year].append(text)
                                year_female_count += 1

                # Store temporal data
                all_data['temporal_data'].append({
                    'year': year,
                    'male_speeches': year_male_count,
                    'female_speeches': year_female_count
                })
                all_data['metadata']['years_processed'].append(year)

                print(f"  {year}: {year_male_count} male, {year_female_count} female speeches")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        # Apply stratified sampling if requested
        if sample_size and stratified:
            all_data = self._stratified_sample_gender(all_data, sample_size)
        elif sample_size:
            # Simple random sampling
            all_data = self._random_sample_gender(all_data, sample_size)

        print(f"\nLoaded: {len(all_data['male_speeches'])} male, "
              f"{len(all_data['female_speeches'])} female speeches")

        return all_data

    def load_overall_corpus(self, year_range: Optional[Tuple[int, int]] = None,
                           sample_size: Optional[int] = None,
                           stratified: bool = True) -> List[Dict]:
        """
        Load overall corpus debates from JSONL files.

        Returns list of debate dicts with structure:
        {
            'year': int,
            'text': str,
            'title': str,
            'speakers': List[str],
            'chamber': str,
            'word_count': int,
            'date': str
        }

        Args:
            year_range: Tuple of (start_year, end_year) or None
            sample_size: Number of debates to sample
            stratified: Use year-stratified sampling

        Returns:
            List of debate dicts
        """
        print(f"Loading overall corpus from {self.data_dir}")

        if year_range:
            start_year, end_year = year_range
        else:
            start_year, end_year = 1803, 2005

        debates = []
        years_processed = []

        for year in range(start_year, end_year + 1):
            jsonl_path = self.data_dir / "content" / str(year) / f"debates_{year}.jsonl"

            if not jsonl_path.exists():
                continue

            year_debates = []
            try:
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue

                        debate = json.loads(line)
                        debate_data = {
                            'year': year,
                            'text': debate.get('full_text', ''),
                            'title': debate.get('metadata', {}).get('title', ''),
                            'speakers': debate.get('metadata', {}).get('speakers', []),
                            'chamber': debate.get('metadata', {}).get('chamber', ''),
                            'word_count': debate.get('metadata', {}).get('word_count', 0),
                            'date': debate.get('metadata', {}).get('reference_date', ''),
                        }

                        # Only include non-empty texts
                        if debate_data['text']:
                            year_debates.append(debate_data)

                print(f"  Loaded {len(year_debates)} debates from {year}")
                debates.extend(year_debates)
                years_processed.append(year)

            except Exception as e:
                print(f"Error loading {year}: {e}")
                continue

        # Apply sampling if requested
        if sample_size and len(debates) > sample_size:
            if stratified:
                debates = self._stratified_sample_debates(debates, sample_size)
            else:
                random.seed(42)
                debates = random.sample(debates, sample_size)
                print(f"Sampled {sample_size} debates randomly")

        print(f"\nTotal debates loaded: {len(debates)} from {len(years_processed)} years")

        return debates

    def load_speakers(self) -> pd.DataFrame:
        """
        Load deduplicated speakers dataset.

        Returns:
            DataFrame with speaker information
        """
        speakers_path = self.data_dir / 'speakers_deduplicated_fixed.parquet'

        if not speakers_path.exists():
            raise FileNotFoundError(
                f"Speakers dataset not found at {speakers_path}\n"
                f"Run speaker deduplication first"
            )

        print(f"Loading deduplicated speakers from {speakers_path}")
        df = pd.read_parquet(speakers_path)
        print(f"Loaded {len(df):,} deduplicated speakers")

        return df

    def _stratified_sample_gender(self, data: Dict, sample_size: int) -> Dict:
        """
        Apply year-stratified sampling to gender data.

        Maintains temporal distribution across years.
        """
        random.seed(42)

        total_speeches = len(data['male_speeches']) + len(data['female_speeches'])

        if total_speeches <= sample_size:
            print(f"Using all available data: {total_speeches:,} speeches")
            return data

        print(f"\nApplying year-stratified sampling: {sample_size:,} from {total_speeches:,} speeches")

        # Calculate total speeches per year
        year_totals = {}
        for year in set(list(data['male_speeches_by_year'].keys()) +
                       list(data['female_speeches_by_year'].keys())):
            male_count = len(data['male_speeches_by_year'].get(year, []))
            female_count = len(data['female_speeches_by_year'].get(year, []))
            year_totals[year] = male_count + female_count

        # Sample proportionally from each year
        sampled_male = []
        sampled_female = []
        sampling_rate = sample_size / total_speeches

        for year in sorted(year_totals.keys()):
            year_total = year_totals[year]
            year_sample_target = int(year_total * sampling_rate)

            # Sample male speeches
            year_male = data['male_speeches_by_year'].get(year, [])
            if year_male:
                year_male_proportion = len(year_male) / year_total if year_total > 0 else 0
                year_male_sample = min(len(year_male),
                                     int(year_sample_target * year_male_proportion))
                if year_male_sample > 0:
                    sampled_male.extend(random.sample(year_male, year_male_sample))

            # Sample female speeches
            year_female = data['female_speeches_by_year'].get(year, [])
            if year_female:
                year_female_proportion = len(year_female) / year_total if year_total > 0 else 0
                year_female_sample = min(len(year_female),
                                       int(year_sample_target * year_female_proportion))
                if year_female_sample > 0:
                    sampled_female.extend(random.sample(year_female, year_female_sample))

        data['male_speeches'] = sampled_male
        data['female_speeches'] = sampled_female

        print(f"  Sampled: {len(sampled_male):,} male + {len(sampled_female):,} female")
        print(f"  Temporal distribution preserved across {len(year_totals)} years")

        return data

    def _random_sample_gender(self, data: Dict, sample_size: int) -> Dict:
        """Apply simple random sampling to gender data"""
        random.seed(42)

        total_speeches = len(data['male_speeches']) + len(data['female_speeches'])

        if total_speeches <= sample_size:
            return data

        # Maintain gender proportion
        male_proportion = len(data['male_speeches']) / total_speeches
        male_sample = int(sample_size * male_proportion)
        female_sample = sample_size - male_sample

        data['male_speeches'] = random.sample(
            data['male_speeches'],
            min(male_sample, len(data['male_speeches']))
        )
        data['female_speeches'] = random.sample(
            data['female_speeches'],
            min(female_sample, len(data['female_speeches']))
        )

        print(f"Random sampling: {len(data['male_speeches']):,} male + "
              f"{len(data['female_speeches']):,} female")

        return data

    def _stratified_sample_debates(self, debates: List[Dict], sample_size: int) -> List[Dict]:
        """
        Apply year-stratified sampling to debates list.

        Maintains temporal distribution.
        """
        random.seed(42)

        # Group by year
        debates_by_year = {}
        for debate in debates:
            year = debate['year']
            if year not in debates_by_year:
                debates_by_year[year] = []
            debates_by_year[year].append(debate)

        # Sample proportionally
        sampled_debates = []
        total_debates = len(debates)

        for year, year_debates in debates_by_year.items():
            year_sample_size = max(1, int(len(year_debates) / total_debates * sample_size))
            sampled = random.sample(year_debates, min(year_sample_size, len(year_debates)))
            sampled_debates.extend(sampled)

        # Trim to exact sample size if over
        if len(sampled_debates) > sample_size:
            sampled_debates = random.sample(sampled_debates, sample_size)

        print(f"Stratified sampling: {len(sampled_debates):,} debates (temporal distribution preserved)")

        return sampled_debates


# Convenience function for quick loading
def load_hansard_debates(dataset_type: str = 'gender',
                        year_range: Optional[Tuple[int, int]] = None,
                        sample_size: Optional[int] = None) -> Union[Dict, List]:
    """
    Quick convenience function for loading Hansard debates.

    Args:
        dataset_type: 'gender', 'overall', or 'speakers'
        year_range: Tuple of (start_year, end_year) or None
        sample_size: Number of speeches/debates to sample

    Returns:
        Dict for gender dataset, List for overall, DataFrame for speakers
    """
    loader = UnifiedCorpusLoader(dataset_type=dataset_type)
    return loader.load_debates(year_range=year_range, sample_size=sample_size)
