#!/usr/bin/env python3
"""
Fast MP-Speaker Matching
========================

Optimized matching using vectorized operations and pre-computed indices.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_name(name):
    """Fast name normalization."""
    if pd.isna(name) or not name:
        return ""
    
    name = str(name).strip()
    
    # Remove common titles
    name = re.sub(r'^(Mr\.|Mrs\.|Miss|Ms\.|Sir|Lord|Lady|Dame|Baron|Baroness|Earl|Countess|Viscount|Duke|Duchess|Major|Captain|Colonel|General|Lieutenant|Lieut\.|Commander|Admiral|Rev\.|Dr\.)\s+', '', name, flags=re.IGNORECASE)
    
    # Remove constituency info
    name = re.sub(r'\([^)]*\)', '', name).strip()
    
    return ' '.join(name.split()).upper()


def extract_surname(name):
    """Extract surname."""
    normalized = normalize_name(name)
    if not normalized:
        return ""
    
    # Handle special cases
    if 'LLOYD GEORGE' in normalized:
        return 'LLOYD GEORGE'
    
    parts = normalized.split()
    return parts[-1] if parts else ""


def main():
    logger.info("="*60)
    logger.info("FAST MP-SPEAKER MATCHING")
    logger.info("="*60)
    
    data_path = Path('data')
    
    # Load MPs
    logger.info("Loading MPs...")
    mps_df = pd.read_parquet(data_path / 'house_members_gendered_updated.parquet')
    mps_df = mps_df[mps_df['post_role'] == 'Member of Parliament'].copy()
    
    # Extract years efficiently
    mps_df['from_year'] = pd.to_datetime(mps_df['membership_start_date'], errors='coerce').dt.year
    mps_df['to_year'] = pd.to_datetime(mps_df['membership_end_date'], errors='coerce').dt.year
    mps_df['to_year'] = mps_df['to_year'].fillna(2010)
    
    # Filter to Hansard period
    mps_df = mps_df[(mps_df['from_year'] <= 2005) & (mps_df['to_year'] >= 1803)].copy()
    logger.info(f"Loaded {len(mps_df):,} MPs from 1803-2005")
    
    # Load speakers
    logger.info("Loading speakers...")
    speakers_df = pd.read_parquet(data_path / 'speakers_deduplicated_fixed.parquet')
    logger.info(f"Loaded {len(speakers_df):,} speakers")
    
    # Vectorized normalization
    logger.info("Normalizing names...")
    mps_df['norm_name'] = mps_df['person_name'].apply(normalize_name)
    mps_df['surname'] = mps_df['person_name'].apply(extract_surname)
    
    speakers_df['norm_name'] = speakers_df['normalized_name'].apply(normalize_name)
    speakers_df['surname'] = speakers_df['normalized_name'].apply(extract_surname)
    
    # Create indices for fast lookup
    logger.info("Building indices...")
    
    # Group MPs by surname for fast lookup
    mp_surname_groups = mps_df.groupby('surname').groups
    
    # Manual mappings for known cases
    manual_mappings = {
        'BALFOUR': 'BALFOUR',
        'CHAMBERLAIN': 'CHAMBERLAIN',
        'CHURCHILL': 'CHURCHILL',
        'GLADSTONE': 'GLADSTONE',
        'DISRAELI': 'DISRAELI',
        'LLOYD GEORGE': 'LLOYD GEORGE',
        'ASQUITH': 'ASQUITH',
        'THATCHER': 'THATCHER',
        'BLAIR': 'BLAIR',
        'BROWN': 'BROWN',
        'MAJOR': 'MAJOR',
        'HEATH': 'HEATH',
        'WILSON': 'WILSON',
        'CALLAGHAN': 'CALLAGHAN',
        'MACMILLAN': 'MACMILLAN',
        'EDEN': 'EDEN',
        'ATTLEE': 'ATTLEE'
    }
    
    # Match speakers
    logger.info("Matching speakers to MPs...")
    matched_records = []
    unmatched_records = []
    
    # Process in batches for memory efficiency
    batch_size = 1000
    total_batches = len(speakers_df) // batch_size + 1
    
    for batch_idx in range(total_batches):
        if batch_idx % 10 == 0:
            logger.info(f"Processing batch {batch_idx}/{total_batches}")
        
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(speakers_df))
        batch = speakers_df.iloc[start_idx:end_idx]
        
        for _, speaker in batch.iterrows():
            speaker_surname = speaker['surname']
            speaker_years = (speaker['first_year'], speaker['last_year'])
            
            matched = False
            
            # Skip procedural roles
            if speaker['normalized_name'] in ['the Lord Chancellor', 'Mr. Speaker', 'The Prime Minister']:
                continue
            
            # Try surname match
            if speaker_surname and speaker_surname in mp_surname_groups:
                mp_indices = mp_surname_groups[speaker_surname]
                candidates = mps_df.loc[mp_indices]
                
                # Filter by year overlap
                year_matches = candidates[
                    (candidates['from_year'] <= speaker_years[1]) &
                    (candidates['to_year'] >= speaker_years[0])
                ]
                
                if len(year_matches) > 0:
                    # Take first match (could be improved with better logic)
                    best_match = year_matches.iloc[0]
                    
                    matched_records.append({
                        'speaker_name': speaker['normalized_name'],
                        'mp_name': best_match['person_name'],
                        'gender': best_match['gender_inferred'],
                        'first_year': speaker['first_year'],
                        'last_year': speaker['last_year'],
                        'total_speeches': speaker['total_speeches'],
                        'primary_chamber': speaker['primary_chamber'],
                        'match_type': 'surname' if len(year_matches) == 1 else 'surname_multi',
                        'mp_from_year': best_match['from_year'],
                        'mp_to_year': best_match['to_year']
                    })
                    matched = True
            
            if not matched:
                unmatched_records.append({
                    'speaker_name': speaker['normalized_name'],
                    'first_year': speaker['first_year'],
                    'last_year': speaker['last_year'],
                    'total_speeches': speaker['total_speeches'],
                    'primary_chamber': speaker['primary_chamber']
                })
    
    # Create dataframes
    matched_df = pd.DataFrame(matched_records)
    unmatched_df = pd.DataFrame(unmatched_records)
    
    # Statistics
    logger.info("\n" + "="*40)
    logger.info("MATCHING RESULTS")
    logger.info("="*40)
    logger.info(f"Total speakers: {len(speakers_df):,}")
    logger.info(f"Matched to MPs: {len(matched_df):,} ({len(matched_df)/len(speakers_df)*100:.1f}%)")
    logger.info(f"Unmatched: {len(unmatched_df):,} ({len(unmatched_df)/len(speakers_df)*100:.1f}%)")
    
    if len(matched_df) > 0:
        # Gender distribution
        logger.info("\nGender distribution:")
        for gender, count in matched_df['gender'].value_counts().items():
            logger.info(f"  {gender}: {count:,} ({count/len(matched_df)*100:.1f}%)")
        
        # Match types
        logger.info("\nMatch types:")
        for match_type, count in matched_df['match_type'].value_counts().items():
            logger.info(f"  {match_type}: {count:,}")
    
    # Save results
    matched_df.to_parquet(data_path / 'mp_speakers_fast.parquet', index=False)
    logger.info(f"\nSaved {len(matched_df):,} matched MPs to mp_speakers_fast.parquet")
    
    unmatched_df.to_parquet(data_path / 'speakers_unmatched_fast.parquet', index=False)
    logger.info(f"Saved {len(unmatched_df):,} unmatched speakers to speakers_unmatched_fast.parquet")
    
    # Check key figures
    if len(matched_df) > 0:
        key_surnames = ['CHURCHILL', 'THATCHER', 'BLAIR', 'ATTLEE', 'GLADSTONE']
        logger.info("\nKey figures in matched data:")
        for surname in key_surnames:
            matches = matched_df[matched_df['mp_name'].str.upper().str.contains(surname, na=False)]
            if len(matches) > 0:
                total_speeches = matches['total_speeches'].sum()
                logger.info(f"  {surname}: {len(matches)} entries, {total_speeches:,} speeches")
    
    # Show high-volume unmatched
    if len(unmatched_df) > 0:
        high_volume = unmatched_df.nlargest(10, 'total_speeches')
        logger.info("\nTop unmatched speakers by speech count:")
        for _, speaker in high_volume.iterrows():
            logger.info(f"  {speaker['speaker_name']:30s} | {speaker['total_speeches']:,} speeches")


if __name__ == "__main__":
    main()