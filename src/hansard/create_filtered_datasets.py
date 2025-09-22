#!/usr/bin/env python3
"""
Create filtered debate and speaker datasets with confirmed MPs and gender data
"""

import pandas as pd
from pathlib import Path
from mp_matcher import MPMatcher
from typing import Dict, List, Tuple
import json

class DatasetCreator:
    """Create filtered datasets with confirmed MPs"""

    def __init__(self, mp_data_path: str = "data/house_members_gendered_updated.parquet"):
        """Initialize with MP gender data"""
        self.mp_data = pd.read_parquet(mp_data_path)
        self.matcher = MPMatcher(self.mp_data)
        print(f"Loaded {len(self.mp_data)} MP records")

    def create_turn_wise_dataset(self,
                                turns_df: pd.DataFrame,
                                require_all_matched: bool = False) -> pd.DataFrame:
        """
        Create turn-wise dataset with matched MP information

        Args:
            turns_df: DataFrame with conversation turns
            require_all_matched: If True, only include debates where ALL speakers are matched
                                If False, include debates with at least one matched MP

        Returns:
            DataFrame with enhanced turn information
        """
        enhanced_turns = []

        # Process each turn
        for idx, row in turns_df.iterrows():
            speaker = row['speaker']
            matched_name, gender, match_type = self.matcher.match(speaker)

            turn_data = row.to_dict()
            turn_data['matched_name'] = matched_name
            turn_data['gender'] = gender
            turn_data['match_type'] = match_type
            turn_data['is_mp'] = match_type not in ['no_match', 'procedural']

            enhanced_turns.append(turn_data)

        enhanced_df = pd.DataFrame(enhanced_turns)

        # Filter debates based on matching criteria
        if require_all_matched:
            # Only keep debates where all speakers are matched
            valid_debates = []
            for debate_id in enhanced_df['debate_id'].unique():
                debate_turns = enhanced_df[enhanced_df['debate_id'] == debate_id]
                if debate_turns['is_mp'].all():
                    valid_debates.append(debate_id)

            filtered_df = enhanced_df[enhanced_df['debate_id'].isin(valid_debates)]
            print(f"Kept {len(valid_debates)} debates with all speakers matched")
        else:
            # Keep debates with at least one matched MP
            valid_debates = []
            for debate_id in enhanced_df['debate_id'].unique():
                debate_turns = enhanced_df[enhanced_df['debate_id'] == debate_id]
                if debate_turns['is_mp'].any():
                    valid_debates.append(debate_id)

            filtered_df = enhanced_df[enhanced_df['debate_id'].isin(valid_debates)]
            print(f"Kept {len(valid_debates)} debates with at least one matched MP")

        return filtered_df

    def create_debate_metadata(self, filtered_turns: pd.DataFrame) -> pd.DataFrame:
        """Create debate-level metadata from filtered turns"""
        debate_metadata = []

        for debate_id in filtered_turns['debate_id'].unique():
            debate_turns = filtered_turns[filtered_turns['debate_id'] == debate_id]

            # Count speakers by type
            mp_turns = debate_turns[debate_turns['is_mp'] == True]
            unique_mps = mp_turns['matched_name'].nunique()

            # Gender counts (only for matched MPs)
            male_speakers = mp_turns[mp_turns['gender'] == 'M']['matched_name'].nunique()
            female_speakers = mp_turns[mp_turns['gender'] == 'F']['matched_name'].nunique()

            # Word counts by gender
            male_words = mp_turns[mp_turns['gender'] == 'M']['word_count'].sum()
            female_words = mp_turns[mp_turns['gender'] == 'F']['word_count'].sum()
            total_words = mp_turns['word_count'].sum()

            metadata = {
                'debate_id': debate_id,
                'date': debate_turns['date'].iloc[0],
                'chamber': debate_turns['chamber'].iloc[0],
                'topic': debate_turns['topic'].iloc[0],
                'total_turns': len(debate_turns),
                'mp_turns': len(mp_turns),
                'unique_mps': unique_mps,
                'male_speakers': male_speakers,
                'female_speakers': female_speakers,
                'male_words': male_words,
                'female_words': female_words,
                'total_words': total_words,
                'female_word_ratio': female_words / total_words if total_words > 0 else 0,
                'female_speaker_ratio': female_speakers / unique_mps if unique_mps > 0 else 0
            }

            debate_metadata.append(metadata)

        return pd.DataFrame(debate_metadata)

    def create_speaker_dataset(self, filtered_turns: pd.DataFrame) -> pd.DataFrame:
        """Create speaker-level dataset with aggregated statistics"""
        speaker_stats = []

        # Get unique matched MPs
        mp_turns = filtered_turns[filtered_turns['is_mp'] == True]
        unique_mps = mp_turns['matched_name'].unique()

        for mp_name in unique_mps:
            mp_data = mp_turns[mp_turns['matched_name'] == mp_name]

            stats = {
                'mp_name': mp_name,
                'gender': mp_data['gender'].iloc[0],
                'total_turns': len(mp_data),
                'total_words': mp_data['word_count'].sum(),
                'avg_words_per_turn': mp_data['word_count'].mean(),
                'num_debates': mp_data['debate_id'].nunique(),
                'chambers': list(mp_data['chamber'].unique()),
                'date_range': f"{mp_data['date'].min()} to {mp_data['date'].max()}"
            }

            speaker_stats.append(stats)

        return pd.DataFrame(speaker_stats)

    def analyze_dataset_quality(self, filtered_turns: pd.DataFrame) -> Dict:
        """Analyze quality metrics for the filtered dataset"""
        total_turns = len(filtered_turns)
        mp_turns = filtered_turns['is_mp'].sum()
        unmatched_turns = len(filtered_turns[filtered_turns['is_mp'] == False])

        quality_metrics = {
            'total_turns': total_turns,
            'mp_turns': mp_turns,
            'unmatched_turns': unmatched_turns,
            'mp_coverage': mp_turns / total_turns if total_turns > 0 else 0,
            'total_debates': filtered_turns['debate_id'].nunique(),
            'unique_mps': filtered_turns[filtered_turns['is_mp'] == True]['matched_name'].nunique(),
            'match_type_distribution': filtered_turns['match_type'].value_counts().to_dict(),
            'gender_distribution': filtered_turns[filtered_turns['is_mp'] == True]['gender'].value_counts().to_dict()
        }

        return quality_metrics

def main():
    """Process debates and create filtered datasets"""

    # Initialize
    creator = DatasetCreator()

    # Load sample data
    print("Loading conversation turns...")
    turns_df = pd.read_parquet("debate_datasets/conversation_turns_1950_sample.parquet")
    print(f"Loaded {len(turns_df)} turns from {turns_df['debate_id'].nunique()} debates")

    # Create filtered datasets
    print("\n=== CREATING FILTERED DATASETS ===")

    # Version 1: Include debates with at least one matched MP
    print("\nVersion 1: Debates with at least one matched MP")
    filtered_v1 = creator.create_turn_wise_dataset(turns_df, require_all_matched=False)
    quality_v1 = creator.analyze_dataset_quality(filtered_v1)

    print(f"  Total turns: {quality_v1['total_turns']}")
    print(f"  MP turns: {quality_v1['mp_turns']} ({100*quality_v1['mp_coverage']:.1f}%)")
    print(f"  Unique MPs: {quality_v1['unique_mps']}")
    print(f"  Gender distribution: {quality_v1['gender_distribution']}")

    # Version 2: Only debates with all speakers matched
    print("\nVersion 2: Debates with all speakers matched")
    filtered_v2 = creator.create_turn_wise_dataset(turns_df, require_all_matched=True)
    quality_v2 = creator.analyze_dataset_quality(filtered_v2)

    print(f"  Total turns: {quality_v2['total_turns']}")
    print(f"  MP turns: {quality_v2['mp_turns']} ({100*quality_v2['mp_coverage']:.1f}%)")
    print(f"  Unique MPs: {quality_v2['unique_mps']}")
    print(f"  Gender distribution: {quality_v2['gender_distribution']}")

    # Create metadata and speaker datasets for version 1 (more inclusive)
    print("\n=== CREATING AGGREGATE DATASETS ===")

    debate_metadata = creator.create_debate_metadata(filtered_v1)
    print(f"Created debate metadata for {len(debate_metadata)} debates")

    speaker_dataset = creator.create_speaker_dataset(filtered_v1)
    print(f"Created speaker dataset for {len(speaker_dataset)} MPs")

    # Save datasets
    print("\n=== SAVING DATASETS ===")
    output_dir = Path("debate_datasets/filtered")
    output_dir.mkdir(exist_ok=True)

    # Save turn-wise data
    filtered_v1.to_parquet(output_dir / "turns_with_mp_gender.parquet")
    filtered_v2.to_parquet(output_dir / "turns_all_matched.parquet")

    # Save metadata
    debate_metadata.to_parquet(output_dir / "debate_metadata_with_gender.parquet")
    speaker_dataset.to_parquet(output_dir / "speaker_statistics.parquet")

    # Save quality report
    # Convert numpy types to Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (pd.Int64Dtype, pd.Int32Dtype)):
            return int(obj)
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        else:
            return obj

    quality_report = {
        'version_1_at_least_one_mp': convert_to_json_serializable(quality_v1),
        'version_2_all_matched': convert_to_json_serializable(quality_v2)
    }

    with open(output_dir / "quality_report.json", 'w') as f:
        json.dump(quality_report, f, indent=2)

    print(f"\nAll datasets saved to {output_dir}")

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Female speaker ratio: {100*debate_metadata['female_speaker_ratio'].mean():.1f}%")
    print(f"Female word ratio: {100*debate_metadata['female_word_ratio'].mean():.1f}%")

    gender_counts = speaker_dataset['gender'].value_counts()
    print(f"\nSpeaker gender distribution:")
    print(f"  Male: {gender_counts.get('M', 0)}")
    print(f"  Female: {gender_counts.get('F', 0)}")

    return filtered_v1, debate_metadata, speaker_dataset

if __name__ == "__main__":
    turns_df, debate_meta, speaker_df = main()