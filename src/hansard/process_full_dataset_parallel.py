#!/usr/bin/env python3
"""
Process FULL Hansard dataset (2.7M records) with parallel processing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from mp_matcher import MPMatcher
import json
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm
import multiprocessing as mp

def process_single_year(args: Tuple[int, Path]) -> Dict:
    """Process a single year (for parallel execution)"""
    year, mp_data_path = args

    try:
        # Load MP data and create matcher
        mp_data = pd.read_parquet(mp_data_path)
        matcher = MPMatcher(mp_data)

        # Check if files exist
        speaker_file = Path(f"data/processed_fixed/metadata/speakers_{year}.parquet")
        debate_file = Path(f"data/processed_fixed/metadata/debates_{year}.parquet")

        if not speaker_file.exists():
            return {'year': year, 'status': 'no_data'}

        # Process speakers
        speakers_df = pd.read_parquet(speaker_file)

        # Match speakers efficiently using vectorized operations
        unique_speakers = speakers_df['speaker_name'].unique()
        match_map = {}

        for speaker in unique_speakers:
            matched_name, gender, match_type = matcher.match(speaker)
            match_map[speaker] = {
                'matched_name': matched_name,
                'gender': gender,
                'match_type': match_type,
                'is_mp': match_type not in ['no_match', 'procedural']
            }

        # Apply matches
        speakers_df['matched_name'] = speakers_df['speaker_name'].map(lambda x: match_map[x]['matched_name'])
        speakers_df['gender'] = speakers_df['speaker_name'].map(lambda x: match_map[x]['gender'])
        speakers_df['match_type'] = speakers_df['speaker_name'].map(lambda x: match_map[x]['match_type'])
        speakers_df['is_mp'] = speakers_df['speaker_name'].map(lambda x: match_map[x]['is_mp'])

        # Process debates if exists
        debates_df = None
        if debate_file.exists():
            try:
                debates_df = pd.read_parquet(debate_file)
            except:
                pass

        # Calculate statistics
        stats = {
            'year': year,
            'status': 'success',
            'total_records': len(speakers_df),
            'unique_speakers': len(unique_speakers),
            'matched_mps': int(speakers_df['is_mp'].sum()),
            'match_rate': float(speakers_df['is_mp'].mean()),
            'gender_M': int((speakers_df[speakers_df['is_mp']]['gender'] == 'M').sum()),
            'gender_F': int((speakers_df[speakers_df['is_mp']]['gender'] == 'F').sum())
        }

        return stats, speakers_df, debates_df

    except Exception as e:
        return {'year': year, 'status': 'error', 'error': str(e)}

class FullDatasetProcessor:
    """Process the complete Hansard dataset efficiently"""

    def __init__(self, output_base: str = "data_filtered_by_actual_mp_FULL"):
        self.output_base = Path(output_base)
        self.mp_data_path = Path("data/house_members_gendered_updated.parquet")

        # Create output directories
        self.dirs = {
            'speakers': self.output_base / 'speakers_by_year',
            'debates': self.output_base / 'debates_by_year',
            'aggregated': self.output_base / 'aggregated',
            'reports': self.output_base / 'reports'
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def process_all_years_parallel(self, start_year: int = 1803, end_year: int = 2005, max_workers: int = None):
        """Process all years in parallel"""

        if max_workers is None:
            max_workers = min(8, mp.cpu_count() - 1)

        years = list(range(start_year, end_year + 1))
        print(f"\n=== PROCESSING {len(years)} YEARS WITH {max_workers} WORKERS ===")

        # Prepare arguments
        args_list = [(year, self.mp_data_path) for year in years]

        all_stats = {}
        all_speakers = []
        processed_count = 0
        total_records = 0

        # Process in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_year = {executor.submit(process_single_year, args): args[0]
                             for args in args_list}

            # Process results as they complete
            with tqdm(total=len(years), desc="Processing years") as pbar:
                for future in as_completed(future_to_year):
                    year = future_to_year[future]

                    try:
                        result = future.result()

                        if isinstance(result, tuple):
                            stats, speakers_df, debates_df = result

                            # Save data
                            speakers_df.to_parquet(self.dirs['speakers'] / f'speakers_{year}.parquet')
                            if debates_df is not None:
                                debates_df.to_parquet(self.dirs['debates'] / f'debates_{year}.parquet')

                            # Collect for aggregation (sample for memory efficiency)
                            if len(all_speakers) < 100:  # Keep sample for aggregation
                                sample_size = min(1000, len(speakers_df))
                                all_speakers.append(speakers_df.sample(n=sample_size))

                            all_stats[year] = stats
                            total_records += stats['total_records']
                            processed_count += 1
                        else:
                            all_stats[year] = result

                    except Exception as e:
                        all_stats[year] = {'year': year, 'status': 'error', 'error': str(e)}

                    pbar.update(1)
                    pbar.set_postfix({'Records': f'{total_records:,}'})

        print(f"\n✅ Processed {processed_count} years with data")
        print(f"📊 Total records processed: {total_records:,}")

        # Save statistics
        with open(self.dirs['reports'] / 'processing_stats.json', 'w') as f:
            json.dump(all_stats, f, indent=2)

        return all_stats, total_records

    def create_aggregated_datasets(self, stats: Dict):
        """Create aggregated datasets from processed years"""

        print("\n=== CREATING AGGREGATED DATASETS ===")

        # Aggregate unique MPs across all years
        unique_mps = {}

        for year_dir in self.dirs['speakers'].glob('speakers_*.parquet'):
            year_df = pd.read_parquet(year_dir)

            # Get unique MPs from this year
            mp_data = year_df[year_df['is_mp'] == True].groupby('matched_name').agg({
                'gender': 'first',
                'speaker_name': 'count'
            }).rename(columns={'speaker_name': 'count'})

            # Add to overall dictionary
            for mp_name, row in mp_data.iterrows():
                if mp_name in unique_mps:
                    unique_mps[mp_name]['count'] += row['count']
                else:
                    unique_mps[mp_name] = {'gender': row['gender'], 'count': row['count']}

        # Convert to DataFrame
        unique_mps_df = pd.DataFrame.from_dict(unique_mps, orient='index')
        unique_mps_df.index.name = 'mp_name'
        unique_mps_df.to_parquet(self.dirs['aggregated'] / 'unique_mps_full.parquet')

        print(f"✅ Identified {len(unique_mps_df)} unique MPs across all years")

        gender_dist = unique_mps_df['gender'].value_counts()
        print(f"   Gender distribution: M={gender_dist.get('M', 0):,}, F={gender_dist.get('F', 0):,}")

        # Create year-by-year summary
        summary_data = []
        for year, year_stats in sorted(stats.items()):
            if year_stats.get('status') == 'success':
                summary_data.append({
                    'year': int(year),
                    'total_records': year_stats['total_records'],
                    'unique_speakers': year_stats['unique_speakers'],
                    'matched_mps': year_stats['matched_mps'],
                    'match_rate': year_stats['match_rate'],
                    'male_count': year_stats.get('gender_M', 0),
                    'female_count': year_stats.get('gender_F', 0),
                    'female_percentage': 100 * year_stats.get('gender_F', 0) /
                                       (year_stats.get('gender_M', 0) + year_stats.get('gender_F', 0))
                                       if (year_stats.get('gender_M', 0) + year_stats.get('gender_F', 0)) > 0 else 0
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_parquet(self.dirs['aggregated'] / 'yearly_summary.parquet')

        print(f"✅ Created yearly summary for {len(summary_df)} years")

        return unique_mps_df, summary_df

def main():
    """Run the full processing pipeline"""

    print("="*80)
    print("HANSARD FULL DATASET PROCESSING")
    print("Processing 2.7 million speaker records from 201 years")
    print("="*80)

    start_time = time.time()

    # Initialize processor
    processor = FullDatasetProcessor()

    # Process all years in parallel
    stats, total_records = processor.process_all_years_parallel()

    # Create aggregated datasets
    unique_mps_df, summary_df = processor.create_aggregated_datasets(stats)

    elapsed = time.time() - start_time

    # Final summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")
    print(f"📁  Output directory: {processor.output_base}")
    print(f"📊  Total records processed: {total_records:,}")
    print(f"👥  Unique MPs identified: {len(unique_mps_df):,}")
    print(f"📅  Years processed: {len(summary_df)}")

    # Gender evolution summary
    print("\n📈 Gender Representation Evolution:")
    for decade in range(1800, 2010, 10):
        decade_data = summary_df[(summary_df['year'] >= decade) & (summary_df['year'] < decade + 10)]
        if len(decade_data) > 0:
            avg_female_pct = decade_data['female_percentage'].mean()
            total_female = decade_data['female_count'].sum()
            total_male = decade_data['male_count'].sum()
            print(f"   {decade}s: {avg_female_pct:5.1f}% female ({total_female:,} F / {total_male:,} M)")

    print(f"\n✅ All data saved to: {processor.output_base}/")

if __name__ == "__main__":
    main()