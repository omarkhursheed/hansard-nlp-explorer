#!/usr/bin/env python3
"""
Create FULL gender analysis dataset with resume capability
Processes all 201 years of data (1803-2005) with checkpointing
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).resolve().parents[4]  # Up to hansard-nlp-explorer
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
from hansard.scripts.matching.mp_matcher_corrected import CorrectedMPMatcher
from hansard.utils.path_config import Paths
import json
from tqdm import tqdm
import hashlib
from datetime import datetime
import pickle
import argparse
import signal
import os

class GenderDatasetCreator:
    def __init__(self, output_dir="gender_analysis_data_FULL", checkpoint_file="checkpoint.pkl"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.output_dir / checkpoint_file

        # Initialize matcher
        print("\nLoading MP matcher...")
        mp_data = pd.read_parquet(Paths.get_data_dir() / "house_members_gendered_updated.parquet")
        self.matcher = CorrectedMPMatcher(mp_data)

        # Load checkpoint if exists
        self.checkpoint = self.load_checkpoint()

        # Setup signal handler for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.interrupted = False

    def signal_handler(self, signum, frame):
        """Handle interruption gracefully"""
        print("\n\nInterrupted! Saving checkpoint...")
        self.save_checkpoint()
        print("Checkpoint saved. You can resume by running the script again.")
        self.interrupted = True
        sys.exit(0)

    def load_checkpoint(self):
        """Load checkpoint if exists"""
        if self.checkpoint_file.exists():
            print(f"Loading checkpoint from {self.checkpoint_file}...")
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"Resuming from year {checkpoint['last_completed_year'] + 1}")
            return checkpoint
        else:
            return {
                'last_completed_year': 0,
                'all_debates': [],
                'all_female_mps': set(),
                'all_male_mps': set(),
                'stats': {
                    'total_debates_processed': 0,
                    'debates_with_confirmed_mps': 0,
                    'debates_with_female': 0,
                    'by_decade': {}
                },
                'processed_years': set()
            }

    def save_checkpoint(self):
        """Save current progress"""
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(self.checkpoint, f)

    def process_year(self, year, debates_df):
        """Process a single year of data"""
        year_debates = []
        stats = self.checkpoint['stats']

        stats['total_debates_processed'] += len(debates_df)

        for idx, debate in debates_df.iterrows():
            if self.interrupted:
                return year_debates

            speakers = debate.get('speakers', [])
            date = debate.get('reference_date', f'{year}-01-01')
            chamber = debate.get('chamber', 'Commons')

            if not isinstance(speakers, (list, np.ndarray)) or len(speakers) == 0:
                continue

            # Match speakers
            matched_mps = []
            female_mps = []
            male_mps = []
            ambiguous = 0
            unmatched = 0

            for speaker in speakers:
                if pd.notna(speaker):
                    result = self.matcher.match_comprehensive(str(speaker), date, chamber)

                    if result['match_type'] in ['temporal_unique', 'title', 'constituency']:
                        if result.get('confidence', 0) >= 0.7:
                            matched_mps.append(result.get('final_match'))
                            if result.get('gender') == 'F':
                                female_mps.append(result.get('final_match'))
                                self.checkpoint['all_female_mps'].add(result.get('final_match'))
                            elif result.get('gender') == 'M':
                                male_mps.append(result.get('final_match'))
                                self.checkpoint['all_male_mps'].add(result.get('final_match'))
                    elif result['match_type'] == 'ambiguous':
                        ambiguous += 1
                    else:
                        unmatched += 1

            # Only keep debates with at least one confirmed MP
            if matched_mps:
                debate_id = hashlib.md5(
                    f"{debate['file_path']}_{date}".encode()
                ).hexdigest()[:16]

                decade = (year // 10) * 10

                debate_record = {
                    'debate_id': debate_id,
                    'year': year,
                    'decade': decade,
                    'reference_date': date,
                    'chamber': chamber,
                    'title': debate.get('title'),
                    'topic': debate.get('debate_topic'),
                    'total_speakers': len(speakers),
                    'confirmed_mps': len(matched_mps),
                    'female_mps': len(female_mps),
                    'male_mps': len(male_mps),
                    'has_female': len(female_mps) > 0,
                    'has_male': len(male_mps) > 0,
                    'female_names': female_mps,
                    'male_names': male_mps,
                    'ambiguous_speakers': ambiguous,
                    'unmatched_speakers': unmatched,
                    'word_count': debate.get('word_count', 0)
                }

                year_debates.append(debate_record)
                stats['debates_with_confirmed_mps'] += 1

                if female_mps:
                    stats['debates_with_female'] += 1

                # Update decade stats
                if decade not in stats['by_decade']:
                    stats['by_decade'][decade] = {
                        'debates_with_mps': 0,
                        'debates_with_female': 0
                    }
                stats['by_decade'][decade]['debates_with_mps'] += 1
                if female_mps:
                    stats['by_decade'][decade]['debates_with_female'] += 1

        return year_debates

    def process_all_years(self, year_range=None):
        """Process all available years with resume capability"""

        print("=" * 70)
        print("FULL GENDER ANALYSIS DATASET CREATION (RESUMABLE)")
        print("=" * 70)

        # Get all available years
        metadata_dir = Paths.get_processed_data_dir() / "metadata"
        all_years = sorted([
            int(f.stem.split('_')[1])
            for f in metadata_dir.glob("debates_*.parquet")
            if 'master' not in f.stem
        ])

        # Filter by year range if specified
        if year_range:
            start_year, end_year = year_range
            all_years = [y for y in all_years if start_year <= y <= end_year]

        # Skip already processed years
        years_to_process = [y for y in all_years if y not in self.checkpoint['processed_years']]

        if not years_to_process:
            print("All years already processed!")
            return

        print(f"\nTotal years: {len(all_years)}")
        print(f"Already processed: {len(self.checkpoint['processed_years'])}")
        print(f"Remaining to process: {len(years_to_process)}")
        print(f"Years range: {min(years_to_process)} - {max(years_to_process)}")
        print("=" * 70)

        # Process each year
        with tqdm(total=len(all_years), initial=len(self.checkpoint['processed_years']),
                  desc="Processing years") as pbar:
            for year in years_to_process:
                if self.interrupted:
                    break

                try:
                    debates_df = pd.read_parquet(metadata_dir / f"debates_{year}.parquet")
                    year_debates = self.process_year(year, debates_df)

                    # Save year data
                    if year_debates:
                        year_df = pd.DataFrame(year_debates)
                        year_df.to_parquet(self.output_dir / f"debates_{year}_with_mps.parquet")
                        self.checkpoint['all_debates'].extend(year_debates)

                    # Update checkpoint
                    self.checkpoint['processed_years'].add(year)
                    self.checkpoint['last_completed_year'] = year

                    # Save checkpoint every 10 years
                    if year % 10 == 0:
                        self.save_checkpoint()

                    pbar.update(1)

                except Exception as e:
                    print(f"\nError processing {year}: {e}")
                    print("Saving checkpoint and continuing...")
                    self.save_checkpoint()
                    continue

        # Final save
        self.save_final_dataset()

    def save_final_dataset(self):
        """Save the combined dataset and metadata"""
        print("\n" + "=" * 70)
        print("SAVING FINAL DATASET")
        print("=" * 70)

        # Save combined dataset
        if self.checkpoint['all_debates']:
            print(f"\nSaving {len(self.checkpoint['all_debates'])} total debates...")
            combined_df = pd.DataFrame(self.checkpoint['all_debates'])
            combined_df.to_parquet(self.output_dir / "ALL_debates_with_confirmed_mps.parquet")

            # Save metadata
            metadata = {
                'creation_date': datetime.now().isoformat(),
                'years_processed': len(self.checkpoint['processed_years']),
                'year_list': sorted(list(self.checkpoint['processed_years'])),
                'statistics': self.checkpoint['stats'],
                'total_female_mps_identified': len(self.checkpoint['all_female_mps']),
                'total_male_mps_identified': len(self.checkpoint['all_male_mps']),
                'female_mps_list': sorted(list(self.checkpoint['all_female_mps'])),
                'male_mps_list': sorted(list(self.checkpoint['all_male_mps']))
            }

            with open(self.output_dir / "dataset_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            # Print summary
            stats = self.checkpoint['stats']
            print(f"\nTotal debates processed: {stats['total_debates_processed']:,}")
            print(f"Debates with confirmed MPs: {stats['debates_with_confirmed_mps']:,}")
            if stats['debates_with_confirmed_mps'] > 0:
                print(f"  → {100*stats['debates_with_confirmed_mps']/stats['total_debates_processed']:.1f}% of all debates")
                print(f"\nDebates with female MPs: {stats['debates_with_female']:,}")
                print(f"  → {100*stats['debates_with_female']/stats['debates_with_confirmed_mps']:.1f}% of debates with MPs")

            print(f"\nUnique female MPs identified: {len(self.checkpoint['all_female_mps'])}")
            print(f"Unique male MPs identified: {len(self.checkpoint['all_male_mps'])}")

            # Decade summary
            print("\nFemale participation by decade:")
            for decade in sorted(self.checkpoint['stats']['by_decade'].keys()):
                decade_stats = self.checkpoint['stats']['by_decade'][decade]
                total = decade_stats['debates_with_mps']
                female = decade_stats['debates_with_female']
                if total > 0:
                    print(f"  {decade}s: {female:,}/{total:,} debates ({100*female/total:.1f}%)")

            print(f"\n✅ Data saved to: {self.output_dir}/")
            print("\nMain file: ALL_debates_with_confirmed_mps.parquet")
            print("Year files: debates_YYYY_with_mps.parquet")
            print("Checkpoint: checkpoint.pkl (can delete after successful completion)")

            # Remove checkpoint after successful completion
            if self.checkpoint_file.exists():
                os.remove(self.checkpoint_file)
                print("\n✅ Checkpoint removed - processing complete!")

def main():
    parser = argparse.ArgumentParser(description='Create gender analysis dataset with resume capability')
    parser.add_argument('--output-dir', default='gender_analysis_data_FULL',
                       help='Output directory for dataset')
    parser.add_argument('--year-range', nargs=2, type=int, metavar=('START', 'END'),
                       help='Process only years in range (e.g., 1900 2000)')
    parser.add_argument('--reset', action='store_true',
                       help='Reset checkpoint and start fresh')

    args = parser.parse_args()

    # Handle reset
    if args.reset:
        checkpoint_path = Path(args.output_dir) / "checkpoint.pkl"
        if checkpoint_path.exists():
            os.remove(checkpoint_path)
            print("Checkpoint removed. Starting fresh...")

    # Create processor and run
    processor = GenderDatasetCreator(output_dir=args.output_dir)
    processor.process_all_years(year_range=args.year_range)

if __name__ == "__main__":
    main()