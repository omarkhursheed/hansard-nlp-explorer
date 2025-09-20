#!/usr/bin/env python3
"""
Create MP-Only Speaker Dataset
==============================

Matches speaker data against the authoritative gendered house members dataset
to create a dataset containing only actual MPs with correct gender assignments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_name_for_matching(name):
    """Normalize names for matching between datasets."""
    if pd.isna(name):
        return ""
    
    name = str(name).strip()
    
    # Remove common titles for matching
    titles_to_remove = [
        'Mr.', 'Mrs.', 'Miss', 'Ms.', 'Sir', 'Lord', 'Lady', 'Dame',
        'Baron', 'Baroness', 'Earl', 'Countess', 'Viscount', 'Viscountess',
        'Duke', 'Duchess', 'Marquess', 'Marchioness', 
        'Major', 'Captain', 'Colonel', 'General', 'Lieutenant', 'Lieut.',
        'Commander', 'Admiral', 'Reverend', 'Rev.', 'Father', 'Dr.', 'Professor'
    ]
    
    for title in titles_to_remove:
        if name.startswith(title + ' '):
            name = name[len(title)+1:].strip()
    
    # Remove constituency info in parentheses
    name = re.sub(r'\([^)]*\)', '', name).strip()
    
    # Normalize spacing and case
    name = ' '.join(name.split())
    
    return name.upper()


def extract_surname(name):
    """Extract surname from a name."""
    normalized = normalize_name_for_matching(name)
    if not normalized:
        return ""
    
    # Split and take last part as surname
    parts = normalized.split()
    if parts:
        return parts[-1]
    return ""


def main():
    logger.info("="*60)
    logger.info("CREATING MP-ONLY SPEAKER DATASET")
    logger.info("="*60)
    
    # Load datasets
    data_path = Path('data')
    
    # Load the authoritative gendered house members data
    gendered_path = data_path / 'house_members_gendered_updated.parquet'
    logger.info(f"Loading gendered house members from {gendered_path}")
    mp_df = pd.read_parquet(gendered_path)
    
    # Filter to only MPs (based on post_role)
    mp_df = mp_df[mp_df['post_role'] == 'Member of Parliament'].copy()
    logger.info(f"Loaded {len(mp_df):,} MPs")
    
    # Load speaker data
    speakers_path = data_path / 'speakers_deduplicated_fixed.parquet'
    logger.info(f"Loading speaker data from {speakers_path}")
    speakers_df = pd.read_parquet(speakers_path)
    logger.info(f"Loaded {len(speakers_df):,} speakers")
    
    # Prepare matching keys for MPs
    mp_df['normalized_name'] = mp_df['person_name'].apply(normalize_name_for_matching)
    mp_df['surname'] = mp_df['person_name'].apply(extract_surname)
    
    # Extract years from dates
    mp_df['from_year'] = pd.to_datetime(mp_df['membership_start_date']).dt.year
    mp_df['to_year'] = pd.to_datetime(mp_df['membership_end_date']).dt.year
    
    # Handle missing end dates (assume current)
    mp_df['to_year'] = mp_df['to_year'].fillna(2010)
    
    # For speakers, also create normalized version
    speakers_df['normalized_for_matching'] = speakers_df['normalized_name'].apply(normalize_name_for_matching)
    speakers_df['surname_for_matching'] = speakers_df['normalized_name'].apply(extract_surname)
    
    # Create year overlap check
    logger.info("\nMatching speakers to MPs...")
    
    matched_speakers = []
    unmatched_speakers = []
    
    for idx, speaker in speakers_df.iterrows():
        speaker_norm = speaker['normalized_for_matching']
        speaker_surname = speaker['surname_for_matching']
        speaker_first_year = speaker['first_year']
        speaker_last_year = speaker['last_year']
        
        # Try exact name match with year overlap
        exact_matches = mp_df[
            (mp_df['normalized_name'] == speaker_norm) &
            (mp_df['from_year'] <= speaker_last_year) &
            (mp_df['to_year'] >= speaker_first_year)
        ]
        
        if len(exact_matches) > 0:
            # Take the match with most overlap
            best_match = exact_matches.iloc[0]
            for _, match in exact_matches.iterrows():
                overlap_start = max(match['from_year'], speaker_first_year)
                overlap_end = min(match['to_year'], speaker_last_year)
                if overlap_end - overlap_start > 0:
                    best_match = match
                    break
            
            matched_speakers.append({
                'speaker_name': speaker['normalized_name'],
                'mp_name': best_match['person_name'],
                'gender': best_match['gender_inferred'],
                'party': best_match.get('party', 'Unknown'),
                'constituency': best_match.get('constituencies', 'Unknown'),
                'first_year': speaker['first_year'],
                'last_year': speaker['last_year'],
                'total_speeches': speaker['total_speeches'],
                'primary_chamber': speaker['primary_chamber'],
                'match_type': 'exact',
                'mp_from_year': best_match['from_year'],
                'mp_to_year': best_match['to_year']
            })
        else:
            # Try surname match with year overlap (less confident)
            surname_matches = mp_df[
                (mp_df['surname'] == speaker_surname) &
                (mp_df['from_year'] <= speaker_last_year) &
                (mp_df['to_year'] >= speaker_first_year)
            ]
            
            if len(surname_matches) == 1:
                # Only accept if unique surname match
                match = surname_matches.iloc[0]
                matched_speakers.append({
                    'speaker_name': speaker['normalized_name'],
                    'mp_name': match['person_name'],
                    'gender': match['gender_inferred'],
                    'party': match.get('party', 'Unknown'),
                    'constituency': match.get('constituencies', 'Unknown'),
                    'first_year': speaker['first_year'],
                    'last_year': speaker['last_year'],
                    'total_speeches': speaker['total_speeches'],
                    'primary_chamber': speaker['primary_chamber'],
                    'match_type': 'surname',
                    'mp_from_year': match['from_year'],
                    'mp_to_year': match['to_year']
                })
            else:
                unmatched_speakers.append({
                    'speaker_name': speaker['normalized_name'],
                    'first_year': speaker['first_year'],
                    'last_year': speaker['last_year'],
                    'total_speeches': speaker['total_speeches'],
                    'primary_chamber': speaker['primary_chamber']
                })
    
    # Create dataframes
    matched_df = pd.DataFrame(matched_speakers)
    unmatched_df = pd.DataFrame(unmatched_speakers)
    
    logger.info(f"\nMatching Results:")
    logger.info(f"  Matched to MPs: {len(matched_df):,} ({len(matched_df)/len(speakers_df)*100:.1f}%)")
    logger.info(f"  Unmatched: {len(unmatched_df):,} ({len(unmatched_df)/len(speakers_df)*100:.1f}%)")
    
    if len(matched_df) > 0:
        # Gender distribution in matched MPs
        gender_dist = matched_df['gender'].value_counts()
        logger.info(f"\nGender distribution in matched MPs:")
        for gender, count in gender_dist.items():
            logger.info(f"  {gender}: {count:,} ({count/len(matched_df)*100:.1f}%)")
        
        # Match type distribution
        match_dist = matched_df['match_type'].value_counts()
        logger.info(f"\nMatch type distribution:")
        for match_type, count in match_dist.items():
            logger.info(f"  {match_type}: {count:,} ({count/len(matched_df)*100:.1f}%)")
        
        # Save matched MPs dataset
        output_path = data_path / 'mp_speakers_gendered.parquet'
        matched_df.to_parquet(output_path, index=False)
        logger.info(f"\nSaved {len(matched_df):,} matched MP speakers to {output_path}")
        
        # Save unmatched for inspection
        unmatched_path = data_path / 'speakers_unmatched.parquet'
        unmatched_df.to_parquet(unmatched_path, index=False)
        logger.info(f"Saved {len(unmatched_df):,} unmatched speakers to {unmatched_path}")
        
        # Show some examples of matched female MPs
        female_mps = matched_df[matched_df['gender'] == 'F'].sort_values('total_speeches', ascending=False)
        if len(female_mps) > 0:
            logger.info(f"\nTop female MPs by speech count:")
            for _, mp in female_mps.head(10).iterrows():
                logger.info(f"  {mp['speaker_name']:30s} | {mp['first_year']}-{mp['last_year']} | {mp['total_speeches']:,} speeches")
        
        # Show some high-volume unmatched speakers
        if len(unmatched_df) > 0:
            high_volume_unmatched = unmatched_df.sort_values('total_speeches', ascending=False)
            logger.info(f"\nTop unmatched speakers (likely non-MPs or parsing issues):")
            for _, speaker in high_volume_unmatched.head(10).iterrows():
                logger.info(f"  {speaker['speaker_name']:30s} | {speaker['first_year']}-{speaker['last_year']} | {speaker['total_speeches']:,} speeches")
    
    logger.info("\n" + "="*60)
    logger.info("MP-ONLY SPEAKER DATASET CREATED SUCCESSFULLY")
    logger.info("="*60)


if __name__ == "__main__":
    main()