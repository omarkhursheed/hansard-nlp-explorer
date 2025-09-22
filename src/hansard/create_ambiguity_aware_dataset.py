#!/usr/bin/env python3
"""
Create dataset that properly tracks ambiguity in MP matching
"""

import pandas as pd
import numpy as np
from pathlib import Path
from mp_matcher_temporal import TemporalMPMatcher
import json
from tqdm import tqdm
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class AmbiguityAwareProcessor:
    """Process Hansard data with proper ambiguity tracking"""

    def __init__(self, output_base: str = "data_with_ambiguity_tracking"):
        self.output_base = Path(output_base)
        self.output_base.mkdir(exist_ok=True)

        # Load MP data
        print("Loading MP data...")
        self.mp_data = pd.read_parquet("data/house_members_gendered_updated.parquet")
        self.matcher = TemporalMPMatcher(self.mp_data)

        # Create output directories
        self.dirs = {
            'certain': self.output_base / 'high_confidence_matches',
            'ambiguous': self.output_base / 'ambiguous_matches',
            'unmatched': self.output_base / 'unmatched_speakers',
            'reports': self.output_base / 'ambiguity_reports',
            'aggregated': self.output_base / 'aggregated'
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)

    def process_year_with_ambiguity(self, year: int) -> Dict:
        """Process a year's speakers with ambiguity tracking"""

        speaker_file = Path(f"data/processed_fixed/metadata/speakers_{year}.parquet")
        if not speaker_file.exists():
            return {'year': year, 'status': 'no_data'}

        # Load speakers
        speakers_df = pd.read_parquet(speaker_file)
        total_records = len(speakers_df)

        # Process each unique speaker
        unique_speakers = speakers_df['speaker_name'].unique()

        # Categories for tracking
        certain_matches = []  # High confidence (>0.9)
        ambiguous_matches = []  # Multiple possible MPs
        unmatched = []  # No match found
        procedural = []  # Non-person entries

        match_results = {}

        for speaker in unique_speakers:
            matches = self.matcher.match_temporal(speaker, year)

            if not matches:
                unmatched.append(speaker)
                match_results[speaker] = {
                    'status': 'unmatched',
                    'candidates': []
                }
            elif matches[0].get('type') == 'procedural':
                procedural.append(speaker)
                match_results[speaker] = {
                    'status': 'procedural',
                    'candidates': []
                }
            elif len(matches) == 1 and matches[0].get('confidence', 0) >= 0.9:
                certain_matches.append(speaker)
                match_results[speaker] = {
                    'status': 'certain',
                    'matched_name': matches[0]['matched_name'],
                    'gender': matches[0]['gender'],
                    'confidence': matches[0]['confidence'],
                    'candidates': [matches[0]]
                }
            else:
                ambiguous_matches.append(speaker)
                match_results[speaker] = {
                    'status': 'ambiguous',
                    'num_candidates': len(matches),
                    'candidates': matches
                }

        # Add match results to dataframe
        speakers_df['match_status'] = speakers_df['speaker_name'].map(
            lambda x: match_results[x]['status']
        )
        speakers_df['match_confidence'] = speakers_df['speaker_name'].map(
            lambda x: match_results[x].get('confidence', 0)
        )
        speakers_df['num_candidates'] = speakers_df['speaker_name'].map(
            lambda x: match_results[x].get('num_candidates', 0)
        )

        # For certain matches, add MP info
        speakers_df['matched_mp'] = speakers_df['speaker_name'].map(
            lambda x: match_results[x].get('matched_name')
        )
        speakers_df['gender'] = speakers_df['speaker_name'].map(
            lambda x: match_results[x].get('gender')
        )

        # Calculate statistics
        stats = {
            'year': year,
            'status': 'success',
            'total_records': total_records,
            'unique_speakers': len(unique_speakers),
            'certain_matches': len(certain_matches),
            'ambiguous_matches': len(ambiguous_matches),
            'unmatched': len(unmatched),
            'procedural': len(procedural),
            'certain_rate': len(certain_matches) / len(unique_speakers) if len(unique_speakers) > 0 else 0,
            'ambiguous_rate': len(ambiguous_matches) / len(unique_speakers) if len(unique_speakers) > 0 else 0,
            'unmatched_rate': len(unmatched) / len(unique_speakers) if len(unique_speakers) > 0 else 0
        }

        # Save detailed ambiguity report
        ambiguity_details = {
            'year': year,
            'ambiguous_speakers': {}
        }

        for speaker in ambiguous_matches:
            result = match_results[speaker]
            ambiguity_details['ambiguous_speakers'][speaker] = {
                'num_candidates': result['num_candidates'],
                'candidates': [
                    {
                        'name': c['matched_name'],
                        'gender': c['gender'],
                        'confidence': c.get('confidence', 0),
                        'years_active': c.get('active_years', '')
                    }
                    for c in result['candidates']
                ]
            }

        return stats, speakers_df, ambiguity_details

    def process_sample_years(self, years: List[int]):
        """Process sample years with ambiguity tracking"""

        all_stats = []
        all_ambiguity = {}

        print(f"\nProcessing {len(years)} years with ambiguity tracking...")

        for year in tqdm(years):
            result = self.process_year_with_ambiguity(year)

            if isinstance(result, dict):
                all_stats.append(result)
            else:
                stats, speakers_df, ambiguity_details = result

                # Save year data by confidence level
                certain_df = speakers_df[speakers_df['match_status'] == 'certain']
                ambiguous_df = speakers_df[speakers_df['match_status'] == 'ambiguous']
                unmatched_df = speakers_df[speakers_df['match_status'] == 'unmatched']

                if len(certain_df) > 0:
                    certain_df.to_parquet(self.dirs['certain'] / f'speakers_{year}_certain.parquet')
                if len(ambiguous_df) > 0:
                    ambiguous_df.to_parquet(self.dirs['ambiguous'] / f'speakers_{year}_ambiguous.parquet')
                if len(unmatched_df) > 0:
                    unmatched_df.to_parquet(self.dirs['unmatched'] / f'speakers_{year}_unmatched.parquet')

                all_stats.append(stats)
                all_ambiguity[year] = ambiguity_details

        # Save reports
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(self.dirs['reports'] / 'matching_statistics.csv', index=False)

        with open(self.dirs['reports'] / 'ambiguity_details.json', 'w') as f:
            json.dump(all_ambiguity, f, indent=2, default=str)

        return stats_df

    def create_quality_datasets(self):
        """Create different quality tiers of data"""

        print("\n=== CREATING QUALITY-TIERED DATASETS ===")

        # Tier 1: High confidence only (certain matches)
        certain_files = list(self.dirs['certain'].glob('*.parquet'))
        if certain_files:
            certain_dfs = [pd.read_parquet(f) for f in certain_files]
            certain_all = pd.concat(certain_dfs, ignore_index=True)
            certain_all.to_parquet(self.dirs['aggregated'] / 'high_confidence_only.parquet')

            print(f"Tier 1 (High Confidence): {len(certain_all)} records")
            print(f"  Unique MPs: {certain_all['matched_mp'].nunique()}")
            gender_dist = certain_all['gender'].value_counts()
            print(f"  Gender: M={gender_dist.get('M', 0)}, F={gender_dist.get('F', 0)}")

        # Tier 2: Include ambiguous with all candidates
        ambiguous_files = list(self.dirs['ambiguous'].glob('*.parquet'))
        if ambiguous_files:
            ambiguous_dfs = [pd.read_parquet(f) for f in ambiguous_files]
            ambiguous_all = pd.concat(ambiguous_dfs, ignore_index=True)
            ambiguous_all.to_parquet(self.dirs['aggregated'] / 'ambiguous_speakers.parquet')

            print(f"\nTier 2 (Ambiguous): {len(ambiguous_all)} records")
            print(f"  Average candidates per speaker: {ambiguous_all['num_candidates'].mean():.1f}")

        # Tier 3: Unmatched speakers
        unmatched_files = list(self.dirs['unmatched'].glob('*.parquet'))
        if unmatched_files:
            unmatched_dfs = [pd.read_parquet(f) for f in unmatched_files]
            unmatched_all = pd.concat(unmatched_dfs, ignore_index=True)
            unmatched_all.to_parquet(self.dirs['aggregated'] / 'unmatched_speakers.parquet')

            print(f"\nTier 3 (Unmatched): {len(unmatched_all)} records")

def main():
    """Run ambiguity-aware processing"""

    processor = AmbiguityAwareProcessor()

    # Process strategic sample years
    sample_years = [1850, 1900, 1920, 1928, 1950, 1970, 1990, 2000]

    print("="*80)
    print("AMBIGUITY-AWARE HANSARD PROCESSING")
    print("="*80)

    # Process years
    stats_df = processor.process_sample_years(sample_years)

    # Create quality datasets
    processor.create_quality_datasets()

    # Print summary
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Years processed: {len(stats_df)}")
    print(f"\nAverage rates across years:")
    print(f"  Certain matches: {100*stats_df['certain_rate'].mean():.1f}%")
    print(f"  Ambiguous: {100*stats_df['ambiguous_rate'].mean():.1f}%")
    print(f"  Unmatched: {100*stats_df['unmatched_rate'].mean():.1f}%")

    print("\n=== MATCHING QUALITY BY YEAR ===")
    for _, row in stats_df.iterrows():
        if row['status'] == 'success':
            print(f"{row['year']}: "
                  f"Certain={row['certain_matches']:4d} ({100*row['certain_rate']:.1f}%), "
                  f"Ambiguous={row['ambiguous_matches']:3d} ({100*row['ambiguous_rate']:.1f}%), "
                  f"Unmatched={row['unmatched']:3d}")

    print(f"\n✅ Data saved to: {processor.output_base}/")
    print("\nDataset tiers available:")
    print("  1. high_confidence_only.parquet - Use for precise analysis")
    print("  2. ambiguous_speakers.parquet - Contains all possible matches")
    print("  3. unmatched_speakers.parquet - Speakers we couldn't identify")

if __name__ == "__main__":
    main()