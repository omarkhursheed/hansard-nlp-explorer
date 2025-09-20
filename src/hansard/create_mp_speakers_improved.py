#!/usr/bin/env python3
"""
Improved MP-Speaker Matching
============================

Multi-stage matching approach:
1. Exact name matches
2. Surname + overlapping years
3. Fuzzy matching for close variations
4. Manual mappings for known high-profile cases
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import re
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImprovedMPMatcher:
    def __init__(self):
        self.data_path = Path('data')
        
        # Manual mappings for known high-profile MPs
        # Format: speaker_name -> mp_name pattern
        self.manual_mappings = {
            'Mr. A. J. Balfour': 'Arthur James Balfour',
            'Mr. CHAMBERLAIN': ['Neville Chamberlain', 'Joseph Chamberlain', 'Austen Chamberlain'],
            'Mr. Gladstone': 'William Ewart Gladstone',
            'Mr. Churchill': 'Winston Churchill',
            'Mr. Lloyd George': 'David Lloyd George',
            'Mr. Asquith': 'H. H. Asquith',
            'Mr. Disraeli': 'Benjamin Disraeli',
            'the Lord Chancellor': None,  # Skip - not an MP
            'Mr. Speaker': None,  # Skip - procedural role
            'The Prime Minister': None,  # Skip - role not name
        }
        
        # Common name variations
        self.name_variations = {
            'Tony': 'Anthony',
            'Jim': 'James',
            'Bill': 'William',
            'Bob': 'Robert',
            'Dick': 'Richard',
            'Ted': 'Edward',
            'Betty': 'Elizabeth',
            'Peggy': 'Margaret',
            'Nancy': 'Anne',
        }
    
    def normalize_name(self, name):
        """Enhanced name normalization."""
        if pd.isna(name):
            return ""
        
        name = str(name).strip()
        
        # Remove titles more comprehensively
        titles = [
            r'^(Mr\.|Mrs\.|Miss|Ms\.|Sir|Lord|Lady|Dame)\s+',
            r'^(Baron|Baroness|Earl|Countess|Viscount|Viscountess)\s+',
            r'^(Duke|Duchess|Marquess|Marchioness)\s+',
            r'^(Major|Captain|Colonel|General|Lieutenant|Lieut\.|Lt\.)\s+',
            r'^(Commander|Admiral|Reverend|Rev\.|Father|Dr\.|Professor)\s+',
            r'^(Rt\. Hon\.|Right Hon\.|Hon\.)\s+'
        ]
        
        for title_pattern in titles:
            name = re.sub(title_pattern, '', name, flags=re.IGNORECASE)
        
        # Remove constituency info
        name = re.sub(r'\([^)]*\)', '', name).strip()
        name = re.sub(r'\[.*?\]', '', name).strip()
        
        # Normalize spacing
        name = ' '.join(name.split())
        
        return name
    
    def extract_surname(self, name):
        """Extract surname with better handling."""
        normalized = self.normalize_name(name)
        if not normalized:
            return ""
        
        # Handle special cases like "Lloyd George"
        if 'Lloyd George' in normalized:
            return 'Lloyd George'
        
        # Split and take last part
        parts = normalized.split()
        if parts:
            # Handle double-barreled surnames
            if len(parts) >= 2 and parts[-2] in ['Butler', 'Campbell', 'Douglas', 'Lennox', 'Gordon']:
                return f"{parts[-2]}-{parts[-1]}"
            return parts[-1]
        return ""
    
    def fuzzy_match_score(self, name1, name2):
        """Calculate fuzzy match score between two names."""
        if not name1 or not name2:
            return 0.0
        
        # Direct comparison
        if name1.lower() == name2.lower():
            return 1.0
        
        # Check if one name contains the other
        if name1.lower() in name2.lower() or name2.lower() in name1.lower():
            return 0.9
        
        # Use sequence matcher for fuzzy matching
        return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
    
    def match_speakers_to_mps(self, speakers_df, mps_df):
        """Multi-stage matching process."""
        
        # Prepare normalized names
        speakers_df['norm_name'] = speakers_df['normalized_name'].apply(self.normalize_name)
        speakers_df['surname'] = speakers_df['normalized_name'].apply(self.extract_surname)
        
        mps_df['norm_name'] = mps_df['person_name'].apply(self.normalize_name)
        mps_df['surname'] = mps_df['person_name'].apply(self.extract_surname)
        
        matched = []
        unmatched = []
        
        total_speakers = len(speakers_df)
        
        for idx, speaker in speakers_df.iterrows():
            if idx % 1000 == 0:
                logger.info(f"Processing speaker {idx}/{total_speakers}")
            
            speaker_name = speaker['normalized_name']
            speaker_norm = speaker['norm_name']
            speaker_surname = speaker['surname']
            speaker_years = (speaker['first_year'], speaker['last_year'])
            
            match_found = False
            best_match = None
            match_type = None
            match_score = 0
            
            # Stage 1: Check manual mappings
            for pattern, mp_name in self.manual_mappings.items():
                if pattern in speaker_name:
                    if mp_name is None:
                        # Skip this speaker (procedural role)
                        match_found = True
                        break
                    elif isinstance(mp_name, list):
                        # Multiple possible MPs with this surname
                        for possible_mp in mp_name:
                            candidates = mps_df[
                                (mps_df['person_name'].str.contains(possible_mp, na=False)) &
                                (mps_df['from_year'] <= speaker_years[1]) &
                                (mps_df['to_year'] >= speaker_years[0])
                            ]
                            if len(candidates) > 0:
                                best_match = candidates.iloc[0]
                                match_type = 'manual'
                                match_found = True
                                break
                    else:
                        candidates = mps_df[
                            (mps_df['person_name'].str.contains(mp_name, na=False)) &
                            (mps_df['from_year'] <= speaker_years[1]) &
                            (mps_df['to_year'] >= speaker_years[0])
                        ]
                        if len(candidates) > 0:
                            best_match = candidates.iloc[0]
                            match_type = 'manual'
                            match_found = True
                            break
            
            if match_found and best_match is None:
                # Skip procedural roles
                continue
            
            # Stage 2: Exact normalized name match
            if not match_found and speaker_norm:
                candidates = mps_df[
                    (mps_df['norm_name'] == speaker_norm) &
                    (mps_df['from_year'] <= speaker_years[1]) &
                    (mps_df['to_year'] >= speaker_years[0])
                ]
                if len(candidates) > 0:
                    best_match = candidates.iloc[0]
                    match_type = 'exact'
                    match_score = 1.0
                    match_found = True
            
            # Stage 3: Surname match with year overlap
            if not match_found and speaker_surname:
                candidates = mps_df[
                    (mps_df['surname'] == speaker_surname) &
                    (mps_df['from_year'] <= speaker_years[1]) &
                    (mps_df['to_year'] >= speaker_years[0])
                ]
                
                if len(candidates) == 1:
                    best_match = candidates.iloc[0]
                    match_type = 'surname_unique'
                    match_score = 0.8
                    match_found = True
                elif len(candidates) > 1:
                    # Multiple candidates - try fuzzy matching on full name
                    best_score = 0
                    for _, candidate in candidates.iterrows():
                        score = self.fuzzy_match_score(speaker_norm, candidate['norm_name'])
                        if score > best_score:
                            best_score = score
                            best_match = candidate
                    
                    if best_score > 0.7:  # Threshold for accepting fuzzy match
                        match_type = 'surname_fuzzy'
                        match_score = best_score
                        match_found = True
            
            # Stage 4: Fuzzy matching for close variations
            if not match_found and speaker_norm:
                # Look for MPs in same time period
                period_mps = mps_df[
                    (mps_df['from_year'] <= speaker_years[1]) &
                    (mps_df['to_year'] >= speaker_years[0])
                ]
                
                best_score = 0
                for _, candidate in period_mps.iterrows():
                    score = self.fuzzy_match_score(speaker_norm, candidate['norm_name'])
                    if score > best_score:
                        best_score = score
                        best_match = candidate
                
                if best_score > 0.85:  # High threshold for fuzzy matching
                    match_type = 'fuzzy'
                    match_score = best_score
                    match_found = True
            
            # Record result
            if match_found and best_match is not None:
                matched.append({
                    'speaker_name': speaker['normalized_name'],
                    'mp_name': best_match['person_name'],
                    'gender': best_match['gender_inferred'],
                    'party': best_match.get('party', 'Unknown'),
                    'constituency': best_match.get('constituencies', 'Unknown'),
                    'first_year': speaker['first_year'],
                    'last_year': speaker['last_year'],
                    'total_speeches': speaker['total_speeches'],
                    'primary_chamber': speaker['primary_chamber'],
                    'match_type': match_type,
                    'match_score': round(match_score, 3),
                    'mp_from_year': best_match['from_year'],
                    'mp_to_year': best_match['to_year']
                })
            else:
                unmatched.append({
                    'speaker_name': speaker['normalized_name'],
                    'first_year': speaker['first_year'],
                    'last_year': speaker['last_year'],
                    'total_speeches': speaker['total_speeches'],
                    'primary_chamber': speaker['primary_chamber']
                })
        
        return pd.DataFrame(matched), pd.DataFrame(unmatched)
    
    def run_matching(self):
        """Run the improved matching process."""
        
        logger.info("="*60)
        logger.info("IMPROVED MP-SPEAKER MATCHING")
        logger.info("="*60)
        
        # Load data
        logger.info("Loading datasets...")
        
        # Load MPs
        mps_df = pd.read_parquet(self.data_path / 'house_members_gendered_updated.parquet')
        mps_df = mps_df[mps_df['post_role'] == 'Member of Parliament'].copy()
        
        # Extract years
        mps_df['from_year'] = pd.to_datetime(mps_df['membership_start_date']).dt.year
        mps_df['to_year'] = pd.to_datetime(mps_df['membership_end_date']).dt.year
        mps_df['to_year'] = mps_df['to_year'].fillna(2010)
        
        # Filter to Hansard period
        mps_df = mps_df[
            (mps_df['from_year'] <= 2005) & 
            (mps_df['to_year'] >= 1803)
        ].copy()
        
        logger.info(f"Loaded {len(mps_df):,} MPs from 1803-2005")
        
        # Load speakers
        speakers_df = pd.read_parquet(self.data_path / 'speakers_deduplicated_fixed.parquet')
        logger.info(f"Loaded {len(speakers_df):,} speakers")
        
        # Run matching
        logger.info("\nRunning multi-stage matching...")
        matched_df, unmatched_df = self.match_speakers_to_mps(speakers_df, mps_df)
        
        # Statistics
        logger.info("\n" + "="*40)
        logger.info("MATCHING RESULTS")
        logger.info("="*40)
        logger.info(f"Total speakers: {len(speakers_df):,}")
        logger.info(f"Matched to MPs: {len(matched_df):,} ({len(matched_df)/len(speakers_df)*100:.1f}%)")
        logger.info(f"Unmatched: {len(unmatched_df):,} ({len(unmatched_df)/len(speakers_df)*100:.1f}%)")
        
        # Match type breakdown
        if len(matched_df) > 0:
            logger.info("\nMatch types:")
            for match_type, count in matched_df['match_type'].value_counts().items():
                logger.info(f"  {match_type:15s}: {count:,} ({count/len(matched_df)*100:.1f}%)")
            
            # Gender distribution
            logger.info("\nGender distribution in matched MPs:")
            for gender, count in matched_df['gender'].value_counts().items():
                logger.info(f"  {gender}: {count:,} ({count/len(matched_df)*100:.1f}%)")
        
        # Save results
        output_path = self.data_path / 'mp_speakers_improved.parquet'
        matched_df.to_parquet(output_path, index=False)
        logger.info(f"\nSaved {len(matched_df):,} matched MPs to {output_path}")
        
        unmatched_path = self.data_path / 'speakers_unmatched_improved.parquet'
        unmatched_df.to_parquet(unmatched_path, index=False)
        logger.info(f"Saved {len(unmatched_df):,} unmatched speakers to {unmatched_path}")
        
        # Show improvements
        if len(matched_df) > 0:
            # Check if we matched key historical figures
            key_figures = ['Balfour', 'Chamberlain', 'Churchill', 'Gladstone', 'Disraeli']
            logger.info("\nKey historical figures matched:")
            for figure in key_figures:
                matches = matched_df[matched_df['mp_name'].str.contains(figure, na=False)]
                if len(matches) > 0:
                    total_speeches = matches['total_speeches'].sum()
                    logger.info(f"  {figure}: {len(matches)} entries, {total_speeches:,} total speeches")
        
        return matched_df, unmatched_df


if __name__ == "__main__":
    matcher = ImprovedMPMatcher()
    matched_df, unmatched_df = matcher.run_matching()