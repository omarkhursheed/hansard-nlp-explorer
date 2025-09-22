#!/usr/bin/env python3
"""
Process ALL Hansard debates to create MP-filtered datasets with confirmed gender data
Efficient pipeline for processing 200+ years of parliamentary data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from mp_matcher import MPMatcher
import json
from typing import Dict, List, Optional
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import time

class FullScaleProcessor:
    """Process all Hansard data efficiently"""

    def __init__(self, output_base: str = "data_filtered_by_actual_mp"):
        """Initialize processor with output directory"""
        self.output_base = Path(output_base)
        self.output_base.mkdir(exist_ok=True)

        # Load MP data once
        print("Loading MP gender data...")
        self.mp_data = pd.read_parquet("data/house_members_gendered_updated.parquet")
        self.matcher = MPMatcher(self.mp_data)
        print(f"Loaded {len(self.mp_data)} MP records with {self.mp_data['person_name'].nunique()} unique MPs")

        # Create output directories
        self.dirs = {
            'turns': self.output_base / 'turns_by_year',
            'debates': self.output_base / 'debates_metadata',
            'speakers': self.output_base / 'speakers_by_year',
            'aggregated': self.output_base / 'aggregated',
            'reports': self.output_base / 'quality_reports'
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)

    def process_year_speakers(self, year: int) -> Optional[pd.DataFrame]:
        """Process speakers for a single year"""
        speaker_file = Path(f"data/processed_fixed/metadata/speakers_{year}.parquet")

        if not speaker_file.exists():
            return None

        try:
            # Load speaker data
            speakers_df = pd.read_parquet(speaker_file)

            # Match speakers to MPs
            unique_speakers = speakers_df['speaker_name'].unique()
            match_results = {}

            for speaker in unique_speakers:
                matched_name, gender, match_type = self.matcher.match(speaker)
                match_results[speaker] = {
                    'matched_name': matched_name,
                    'gender': gender,
                    'match_type': match_type,
                    'is_mp': match_type not in ['no_match', 'procedural']
                }

            # Add match info to dataframe
            speakers_df['matched_name'] = speakers_df['speaker_name'].map(lambda x: match_results[x]['matched_name'])
            speakers_df['gender'] = speakers_df['speaker_name'].map(lambda x: match_results[x]['gender'])
            speakers_df['match_type'] = speakers_df['speaker_name'].map(lambda x: match_results[x]['match_type'])
            speakers_df['is_mp'] = speakers_df['speaker_name'].map(lambda x: match_results[x]['is_mp'])

            return speakers_df

        except Exception as e:
            print(f"Error processing year {year}: {e}")
            return None

    def process_year_debates(self, year: int) -> Optional[pd.DataFrame]:
        """Process debate metadata for a single year"""
        debate_file = Path(f"data/processed_fixed/metadata/debates_{year}.parquet")

        if not debate_file.exists():
            return None

        try:
            debates_df = pd.read_parquet(debate_file)

            # Process speakers list if available
            if 'speakers' in debates_df.columns:
                debates_df['mp_speakers'] = debates_df['speakers'].apply(
                    lambda speakers: self._filter_mp_speakers(speakers) if speakers is not None else []
                )
                debates_df['num_mp_speakers'] = debates_df['mp_speakers'].apply(len)
                debates_df['has_mps'] = debates_df['num_mp_speakers'] > 0

            return debates_df

        except Exception as e:
            print(f"Error processing debates for year {year}: {e}")
            return None

    def _filter_mp_speakers(self, speakers):
        """Filter speakers list to only include confirmed MPs"""
        if speakers is None:
            return []

        # Handle different speaker data types
        if isinstance(speakers, str):
            speakers = [speakers]
        elif hasattr(speakers, '__iter__') and not isinstance(speakers, str):
            # Convert to list if it's an array or other iterable
            try:
                speakers = list(speakers)
            except:
                return []
        else:
            return []

        mp_speakers = []
        for speaker in speakers:
            if isinstance(speaker, str):
                _, _, match_type = self.matcher.match(speaker)
                if match_type not in ['no_match', 'procedural']:
                    mp_speakers.append(speaker)

        return mp_speakers

    def create_turn_dataset_for_year(self, year: int, speakers_df: pd.DataFrame, debates_df: pd.DataFrame) -> pd.DataFrame:
        """Create turn-by-turn dataset for a year"""
        # Group speakers by debate
        turns = []

        for file_path in speakers_df['file_path'].unique():
            debate_speakers = speakers_df[speakers_df['file_path'] == file_path].copy()

            # Sort by order of appearance (assuming they're in order)
            debate_speakers = debate_speakers.reset_index(drop=True)
            debate_speakers['turn_number'] = debate_speakers.index + 1

            # Get debate metadata
            debate_info = debates_df[debates_df['file_path'] == file_path].iloc[0] if len(debates_df[debates_df['file_path'] == file_path]) > 0 else None

            for _, turn in debate_speakers.iterrows():
                turn_data = {
                    'year': year,
                    'file_path': file_path,
                    'turn_number': turn['turn_number'],
                    'speaker_original': turn['speaker_name'],
                    'speaker_matched': turn['matched_name'],
                    'gender': turn['gender'],
                    'match_type': turn['match_type'],
                    'is_mp': turn['is_mp'],
                    'chamber': turn.get('chamber', None),
                    'reference_date': turn.get('reference_date', None)
                }

                # Add debate metadata if available
                if debate_info is not None:
                    turn_data['debate_title'] = debate_info.get('title', None)
                    turn_data['debate_topic'] = debate_info.get('debate_topic', None)
                    turn_data['word_count'] = debate_info.get('word_count', None)

                turns.append(turn_data)

        return pd.DataFrame(turns)

    def process_sample_years(self, sample_years: List[int] = [1850, 1900, 1950, 2000]):
        """Process sample years to test the pipeline"""
        print(f"\n=== PROCESSING SAMPLE YEARS: {sample_years} ===")

        sample_stats = {}

        for year in sample_years:
            print(f"\nProcessing year {year}...")
            start_time = time.time()

            # Process speakers
            speakers_df = self.process_year_speakers(year)
            if speakers_df is None:
                print(f"  No speaker data for {year}")
                continue

            # Process debates
            debates_df = self.process_year_debates(year)

            # Calculate statistics
            stats = {
                'total_speakers': len(speakers_df['speaker_name'].unique()),
                'matched_mps': speakers_df['is_mp'].sum(),
                'match_rate': speakers_df['is_mp'].mean(),
                'gender_distribution': speakers_df[speakers_df['is_mp']]['gender'].value_counts().to_dict(),
                'match_types': speakers_df['match_type'].value_counts().to_dict(),
                'processing_time': time.time() - start_time
            }

            sample_stats[year] = stats

            print(f"  Speakers: {stats['total_speakers']}")
            print(f"  Matched MPs: {stats['matched_mps']} ({100*stats['match_rate']:.1f}%)")
            print(f"  Gender: {stats['gender_distribution']}")
            print(f"  Time: {stats['processing_time']:.1f}s")

        # Save sample report
        with open(self.dirs['reports'] / 'sample_quality_report.json', 'w') as f:
            json.dump(sample_stats, f, indent=2, default=str)

        print(f"\nSample report saved to {self.dirs['reports'] / 'sample_quality_report.json'}")
        return sample_stats

    def process_all_years(self, start_year: int = 1803, end_year: int = 2005, n_workers: int = 4):
        """Process all years in parallel"""
        years = range(start_year, end_year + 1)

        print(f"\n=== PROCESSING ALL YEARS {start_year}-{end_year} ===")
        print(f"Using {n_workers} parallel workers")

        # Process in batches for memory efficiency
        batch_size = 10
        all_stats = {}

        for i in tqdm(range(0, len(list(years)), batch_size), desc="Processing batches"):
            batch_years = list(years)[i:i+batch_size]

            for year in batch_years:
                # Process speakers
                speakers_df = self.process_year_speakers(year)
                if speakers_df is None:
                    continue

                # Process debates
                debates_df = self.process_year_debates(year)
                if debates_df is None:
                    debates_df = pd.DataFrame()

                # Create turn dataset
                turns_df = self.create_turn_dataset_for_year(year, speakers_df, debates_df)

                # Save year data
                speakers_df.to_parquet(self.dirs['speakers'] / f'speakers_{year}_filtered.parquet')

                if len(debates_df) > 0:
                    debates_df.to_parquet(self.dirs['debates'] / f'debates_{year}_filtered.parquet')

                if len(turns_df) > 0:
                    turns_df.to_parquet(self.dirs['turns'] / f'turns_{year}.parquet')

                # Collect statistics
                all_stats[year] = {
                    'total_speakers': len(speakers_df['speaker_name'].unique()) if len(speakers_df) > 0 else 0,
                    'matched_mps': speakers_df['is_mp'].sum() if len(speakers_df) > 0 else 0,
                    'match_rate': speakers_df['is_mp'].mean() if len(speakers_df) > 0 else 0,
                    'total_turns': len(turns_df)
                }

        # Save full report
        with open(self.dirs['reports'] / 'full_processing_report.json', 'w') as f:
            json.dump(all_stats, f, indent=2, default=str)

        print(f"\nProcessing complete! Data saved to {self.output_base}")
        return all_stats

    def create_master_datasets(self):
        """Combine all years into master datasets"""
        print("\n=== CREATING MASTER DATASETS ===")

        # Combine all speakers
        all_speakers = []
        for speaker_file in self.dirs['speakers'].glob('speakers_*_filtered.parquet'):
            df = pd.read_parquet(speaker_file)
            all_speakers.append(df)

        if all_speakers:
            master_speakers = pd.concat(all_speakers, ignore_index=True)
            master_speakers.to_parquet(self.dirs['aggregated'] / 'all_speakers_filtered.parquet')
            print(f"Created master speakers dataset: {len(master_speakers)} records")

            # Create unique MP list with gender
            unique_mps = master_speakers[master_speakers['is_mp']].groupby('matched_name').agg({
                'gender': 'first',
                'speaker_name': 'count'
            }).rename(columns={'speaker_name': 'appearance_count'})

            unique_mps.to_parquet(self.dirs['aggregated'] / 'unique_mps_with_gender.parquet')
            print(f"Identified {len(unique_mps)} unique MPs")

            gender_dist = unique_mps['gender'].value_counts()
            print(f"Gender distribution: M={gender_dist.get('M', 0)}, F={gender_dist.get('F', 0)}")

        # Combine all turns
        all_turns = []
        for turn_file in self.dirs['turns'].glob('turns_*.parquet'):
            df = pd.read_parquet(turn_file)
            all_turns.append(df)

        if all_turns:
            master_turns = pd.concat(all_turns, ignore_index=True)
            master_turns.to_parquet(self.dirs['aggregated'] / 'all_turns.parquet')
            print(f"Created master turns dataset: {len(master_turns)} turns")

        print("\nMaster datasets created!")

def main():
    """Main processing pipeline"""
    processor = FullScaleProcessor()

    # Step 1: Test on sample years
    print("\n" + "="*60)
    print("STEP 1: TESTING ON SAMPLE YEARS")
    print("="*60)
    sample_stats = processor.process_sample_years([1850, 1900, 1950, 2000])

    # Check quality
    avg_match_rate = np.mean([s['match_rate'] for s in sample_stats.values() if 'match_rate' in s])
    print(f"\nAverage match rate across samples: {100*avg_match_rate:.1f}%")

    if avg_match_rate < 0.3:
        print("WARNING: Low match rate. Review matching logic before proceeding.")
        response = input("Continue with full processing? (y/n): ")
        if response.lower() != 'y':
            return

    # Step 2: Process all years
    print("\n" + "="*60)
    print("STEP 2: PROCESSING ALL YEARS")
    print("="*60)
    all_stats = processor.process_all_years()

    # Step 3: Create master datasets
    print("\n" + "="*60)
    print("STEP 3: CREATING MASTER DATASETS")
    print("="*60)
    processor.create_master_datasets()

    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"All filtered data saved to: {processor.output_base}")
    print("\nDirectory structure:")
    print("  - turns_by_year/: Turn-by-turn data for each year")
    print("  - debates_metadata/: Debate metadata with MP info")
    print("  - speakers_by_year/: Speaker data with gender matches")
    print("  - aggregated/: Combined master datasets")
    print("  - quality_reports/: Processing statistics and quality metrics")

if __name__ == "__main__":
    main()