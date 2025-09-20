#!/usr/bin/env python3
"""
Check MP Coverage
=================

Analyzes how many MPs from the authoritative dataset were matched
to speakers, and identifies potential mismatches.
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("="*60)
    logger.info("MP COVERAGE ANALYSIS")
    logger.info("="*60)
    
    data_path = Path('data')
    
    # Load the authoritative MPs dataset
    gendered_path = data_path / 'house_members_gendered_updated.parquet'
    logger.info(f"Loading authoritative MPs from {gendered_path}")
    all_mps_df = pd.read_parquet(gendered_path)
    all_mps_df = all_mps_df[all_mps_df['post_role'] == 'Member of Parliament'].copy()
    
    # Extract years
    all_mps_df['from_year'] = pd.to_datetime(all_mps_df['membership_start_date']).dt.year
    all_mps_df['to_year'] = pd.to_datetime(all_mps_df['membership_end_date']).dt.year
    all_mps_df['to_year'] = all_mps_df['to_year'].fillna(2010)
    
    # Filter to Hansard period (1803-2005)
    hansard_mps = all_mps_df[
        (all_mps_df['from_year'] <= 2005) & 
        (all_mps_df['to_year'] >= 1803)
    ].copy()
    
    logger.info(f"Total MPs in authoritative dataset: {len(all_mps_df):,}")
    logger.info(f"MPs in Hansard period (1803-2005): {len(hansard_mps):,}")
    
    # Load matched MPs
    matched_path = data_path / 'mp_speakers_gendered.parquet'
    logger.info(f"\nLoading matched MPs from {matched_path}")
    matched_df = pd.read_parquet(matched_path)
    
    logger.info(f"Matched speaker-MPs: {len(matched_df):,}")
    
    # Get unique MPs from matched dataset
    unique_matched_mps = matched_df['mp_name'].unique()
    logger.info(f"Unique MPs in matched dataset: {len(unique_matched_mps):,}")
    
    # Check coverage by comparing names
    hansard_mps['matched'] = hansard_mps['person_name'].isin(unique_matched_mps)
    
    matched_count = hansard_mps['matched'].sum()
    unmatched_count = (~hansard_mps['matched']).sum()
    
    logger.info(f"\n" + "="*40)
    logger.info("COVERAGE SUMMARY")
    logger.info("="*40)
    logger.info(f"MPs in authoritative dataset (1803-2005): {len(hansard_mps):,}")
    logger.info(f"MPs found in speaker data: {matched_count:,} ({matched_count/len(hansard_mps)*100:.1f}%)")
    logger.info(f"MPs NOT found in speaker data: {unmatched_count:,} ({unmatched_count/len(hansard_mps)*100:.1f}%)")
    
    # Analyze unmatched MPs by period
    unmatched_mps = hansard_mps[~hansard_mps['matched']].copy()
    
    logger.info(f"\n" + "="*40)
    logger.info("UNMATCHED MPs BY PERIOD")
    logger.info("="*40)
    
    periods = [
        (1803, 1850, "Early period"),
        (1851, 1900, "Victorian era"),
        (1901, 1918, "Pre-women's suffrage"),
        (1919, 1945, "Interwar & WWII"),
        (1946, 1979, "Post-war"),
        (1980, 2005, "Modern era")
    ]
    
    for start, end, label in periods:
        period_mps = unmatched_mps[
            (unmatched_mps['from_year'] <= end) & 
            (unmatched_mps['to_year'] >= start)
        ]
        if len(period_mps) > 0:
            logger.info(f"{label:20s} ({start}-{end}): {len(period_mps):,} unmatched MPs")
    
    # Gender distribution of unmatched MPs
    logger.info(f"\n" + "="*40)
    logger.info("GENDER OF UNMATCHED MPs")
    logger.info("="*40)
    
    unmatched_gender = unmatched_mps['gender_inferred'].value_counts()
    for gender, count in unmatched_gender.items():
        logger.info(f"  {gender}: {count:,} ({count/len(unmatched_mps)*100:.1f}%)")
    
    # Sample of unmatched MPs
    logger.info(f"\n" + "="*40)
    logger.info("SAMPLE OF UNMATCHED MPs")
    logger.info("="*40)
    
    # Recent unmatched MPs (more likely to have speeches)
    recent_unmatched = unmatched_mps[unmatched_mps['from_year'] >= 1980].sort_values('from_year', ascending=False)
    
    if len(recent_unmatched) > 0:
        logger.info("\nRecent MPs not found in speaker data:")
        for _, mp in recent_unmatched.head(20).iterrows():
            logger.info(f"  {mp['person_name']:30s} | {int(mp['from_year'])}-{int(mp['to_year'])} | {mp['constituencies']}")
    
    # Early unmatched MPs
    early_unmatched = unmatched_mps[unmatched_mps['from_year'] <= 1850].sort_values('from_year')
    
    if len(early_unmatched) > 0:
        logger.info("\nEarly MPs not found in speaker data:")
        for _, mp in early_unmatched.head(10).iterrows():
            logger.info(f"  {mp['person_name']:30s} | {int(mp['from_year'])}-{int(mp['to_year'])} | {mp['constituencies']}")
    
    # Check for potential high-profile missing MPs
    logger.info(f"\n" + "="*40)
    logger.info("HIGH-PROFILE MPs CHECK")
    logger.info("="*40)
    
    # Female MPs that might be missing
    female_hansard = hansard_mps[hansard_mps['gender_inferred'] == 'F']
    female_unmatched = female_hansard[~female_hansard['matched']]
    
    logger.info(f"Female MPs in dataset: {len(female_hansard):,}")
    logger.info(f"Female MPs matched: {female_hansard['matched'].sum():,}")
    logger.info(f"Female MPs unmatched: {len(female_unmatched):,}")
    
    if len(female_unmatched) > 0:
        logger.info("\nUnmatched female MPs:")
        for _, mp in female_unmatched.sort_values('from_year').head(20).iterrows():
            logger.info(f"  {mp['person_name']:30s} | {int(mp['from_year'])}-{int(mp['to_year'])} | {mp['constituencies']}")
    
    # Save unmatched MPs for inspection
    unmatched_path = data_path / 'mps_not_in_speakers.parquet'
    unmatched_mps.to_parquet(unmatched_path, index=False)
    logger.info(f"\nSaved {len(unmatched_mps):,} unmatched MPs to {unmatched_path}")
    
    # Also check the reverse - speakers not matched to MPs
    speakers_unmatched_path = data_path / 'speakers_unmatched.parquet'
    if speakers_unmatched_path.exists():
        unmatched_speakers = pd.read_parquet(speakers_unmatched_path)
        high_volume = unmatched_speakers[unmatched_speakers['total_speeches'] > 100].sort_values('total_speeches', ascending=False)
        
        logger.info(f"\n" + "="*40)
        logger.info("HIGH-VOLUME UNMATCHED SPEAKERS")
        logger.info("="*40)
        logger.info(f"Speakers with >100 speeches not matched to MPs: {len(high_volume):,}")
        
        if len(high_volume) > 0:
            logger.info("\nTop unmatched speakers by speech count:")
            for _, speaker in high_volume.head(20).iterrows():
                logger.info(f"  {speaker['speaker_name']:30s} | {speaker['first_year']}-{speaker['last_year']} | {speaker['total_speeches']:,} speeches")


if __name__ == "__main__":
    main()