#!/usr/bin/env python3
"""
Advanced deduplication script for normalized speakers dataset.
Uses temporal overlap, name similarity, and activity patterns to merge duplicates.
"""

import pandas as pd
import numpy as np
from typing import Set, Dict, List, Tuple, Optional
from pathlib import Path
from difflib import SequenceMatcher
import re


class SpeakerDeduplicator:
    """Advanced deduplication for parliamentary speakers."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.merge_map = {}  # Maps speaker names to canonical forms
        self.merge_reasons = {}  # Tracks why speakers were merged
        
    def _log(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)
    
    def similarity_score(self, name1: str, name2: str) -> float:
        """Calculate similarity score between two names."""
        if not name1 or not name2:
            return 0.0
        return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
    
    def extract_name_components(self, name: str) -> Dict[str, str]:
        """Extract components from a normalized name."""
        components = {
            'full': name,
            'title': None,
            'first': None,
            'middle': None,
            'last': None,
            'initials': []
        }
        
        # Extract title
        title_pattern = r'^(Mr\.|Mrs\.|Miss|Ms\.|Sir|Lady|Dr\.|Major|Captain|Colonel|General|Lieut\.|Commander|Admiral|Professor|Reverend|Lord|Earl|Viscount|Marquess|Duke|Baron)\s+'
        title_match = re.match(title_pattern, name)
        if title_match:
            components['title'] = title_match.group(1)
            name_without_title = name[len(title_match.group(0)):]
        else:
            name_without_title = name
        
        # Split remaining name
        parts = name_without_title.split()
        if not parts:
            return components
        
        # Extract last name (assume last word)
        components['last'] = parts[-1] if parts else None
        
        # Process remaining parts
        if len(parts) > 1:
            for part in parts[:-1]:
                if len(part) <= 3 and part.endswith('.'):
                    # It's an initial
                    components['initials'].append(part.replace('.', ''))
                elif not components['first']:
                    components['first'] = part
                else:
                    if components['middle']:
                        components['middle'] += ' ' + part
                    else:
                        components['middle'] = part
        
        return components
    
    def names_likely_same_person(self, name1: str, name2: str, 
                                 year_overlap: bool, 
                                 speech_ratio: float,
                                 year_span1: int, year_span2: int) -> Tuple[bool, str]:
        """
        Determine if two names likely refer to the same person.
        Returns (is_same, reason)
        """
        if name1 == name2:
            return True, "exact_match"
        
        # CRITICAL: Check if combined career span would be unrealistic
        # No MP could have a career spanning more than 60 years
        MAX_CAREER_SPAN = 60
        combined_span = max(year_span1, year_span2)
        if combined_span > MAX_CAREER_SPAN:
            return False, f"unrealistic_career_span_{combined_span}_years"
        
        comp1 = self.extract_name_components(name1)
        comp2 = self.extract_name_components(name2)
        
        # Check if surnames match
        if comp1['last'] and comp2['last']:
            if comp1['last'].lower() != comp2['last'].lower():
                return False, "different_surnames"
        
        # Special case: One is abbreviated version of the other
        # e.g., "Mr. Smith" and "Mr. John Smith"
        if name1 in name2 or name2 in name1:
            if year_overlap:
                return True, "name_subset_with_overlap"
        
        # Check initials match
        # e.g., "Mr. J. Smith" and "Mr. John Smith"
        if comp1['initials'] and comp1['last'] == comp2['last']:
            if comp2['first'] and len(comp2['first']) > 0:
                initials1 = [i[0].upper() if i else '' for i in comp1['initials'] if i]
                if comp2['first'][0].upper() in initials1:
                    if year_overlap:
                        return True, "initial_matches_firstname"
        
        # Same logic reversed
        if comp2['initials'] and comp1['last'] == comp2['last']:
            if comp1['first'] and len(comp1['first']) > 0:
                initials2 = [i[0].upper() if i else '' for i in comp2['initials'] if i]
                if comp1['first'][0].upper() in initials2:
                    if year_overlap:
                        return True, "initial_matches_firstname"
        
        # Check for variations like "W. H. Smith" vs "Mr. W. H. Smith"
        name1_no_title = re.sub(r'^(Mr\.|Mrs\.|Miss|Ms\.|Sir|Lady|Dr\.)\s+', '', name1)
        name2_no_title = re.sub(r'^(Mr\.|Mrs\.|Miss|Ms\.|Sir|Lady|Dr\.)\s+', '', name2)
        if name1_no_title == name2_no_title:
            return True, "same_without_title"
        
        # Check for high similarity with temporal overlap
        similarity = self.similarity_score(name1, name2)
        if similarity > 0.85 and year_overlap:
            return True, f"high_similarity_{similarity:.2f}"
        
        # Check for known variations
        # e.g., "the Lord Chancellor" might appear with a name later
        if 'Lord Chancellor' in name1 and 'Lord Chancellor' in name2:
            if year_overlap:
                return True, "role_variation"
        
        return False, "no_match"
    
    def find_duplicate_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Find groups of likely duplicate speakers."""
        duplicate_groups = {}
        processed = set()
        
        # Sort by surname and total speeches for better matching
        df_sorted = df.sort_values(['surname', 'total_speeches'], ascending=[True, False])
        
        # Group by surname for efficiency
        surname_groups = df_sorted.groupby('surname', dropna=True)
        
        total_groups = len(surname_groups)
        current_group = 0
        
        for surname, surname_df in surname_groups:
            current_group += 1
            if current_group % 1000 == 0:
                self._log(f"Processing surname group {current_group}/{total_groups}...")
            
            # Skip very common surnames that are likely different people
            if len(surname_df) > 50:
                # For very common surnames, only merge exact matches or very close variations
                continue
            
            surname_df = surname_df.reset_index(drop=True)
            
            for idx1 in range(len(surname_df)):
                row1 = surname_df.iloc[idx1]
                if row1['normalized_name'] in processed:
                    continue
                
                name1 = row1['normalized_name']
                group = [name1]
                processed.add(name1)
                
                # Only check subsequent rows in same surname group
                # Track the group's overall time span
                group_first_year = row1['first_year']
                group_last_year = row1['last_year']
                
                for idx2 in range(idx1 + 1, len(surname_df)):
                    row2 = surname_df.iloc[idx2]
                    name2 = row2['normalized_name']
                    
                    if name2 in processed:
                        continue
                    
                    # Calculate what the group span would be if we added this person
                    potential_first_year = min(group_first_year, row2['first_year'])
                    potential_last_year = max(group_last_year, row2['last_year'])
                    potential_span = potential_last_year - potential_first_year + 1
                    
                    # Skip if adding this person would create an unrealistic group span
                    MAX_CAREER_SPAN = 60
                    if potential_span > MAX_CAREER_SPAN:
                        continue
                    
                    # Check temporal overlap
                    year_overlap = not (row1['last_year'] < row2['first_year'] or 
                                       row2['last_year'] < row1['first_year'])
                    
                    # Calculate combined year span if they were merged
                    combined_first_year = min(row1['first_year'], row2['first_year'])
                    combined_last_year = max(row1['last_year'], row2['last_year'])
                    combined_year_span = combined_last_year - combined_first_year + 1
                    
                    # Calculate speech ratio (to detect if one is much more active)
                    speech_ratio = min(row1['total_speeches'], row2['total_speeches']) / \
                                  max(row1['total_speeches'], row2['total_speeches']) \
                                  if max(row1['total_speeches'], row2['total_speeches']) > 0 else 0
                    
                    is_same, reason = self.names_likely_same_person(
                        name1, name2, year_overlap, speech_ratio, 
                        combined_year_span, combined_year_span
                    )
                    
                    if is_same:
                        group.append(name2)
                        processed.add(name2)
                        self.merge_reasons[name2] = f"Merged with {name1}: {reason}"
                        # Update group time span
                        group_first_year = potential_first_year
                        group_last_year = potential_last_year
                
                if len(group) > 1:
                    # Use the name with most speeches as canonical
                    group_df = surname_df[surname_df['normalized_name'].isin(group)]
                    canonical = group_df.loc[group_df['total_speeches'].idxmax(), 'normalized_name']
                    duplicate_groups[canonical] = group
        
        return duplicate_groups
    
    def merge_duplicate_records(self, df: pd.DataFrame, duplicate_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """Merge duplicate speaker records."""
        merged_records = []
        
        for canonical, group in duplicate_groups.items():
            group_df = df[df['normalized_name'].isin(group)]
            
            # Aggregate the records
            merged_record = {
                'normalized_name': canonical,
                'most_common_form': group_df.loc[group_df['total_speeches'].idxmax(), 'most_common_form'],
                'first_year': group_df['first_year'].min(),
                'last_year': group_df['last_year'].max(),
                'total_speeches': group_df['total_speeches'].sum(),
                'primary_chamber': group_df.loc[group_df['total_speeches'].idxmax(), 'primary_chamber'],
                'surname': group_df.loc[group_df['total_speeches'].idxmax(), 'surname'],
                'merged_from': '|'.join(group) if len(group) > 1 else None,
                'merge_count': len(group)
            }
            merged_records.append(merged_record)
        
        # Add non-duplicate records
        all_duplicates = set()
        for group in duplicate_groups.values():
            all_duplicates.update(group)
        
        non_duplicates = df[~df['normalized_name'].isin(all_duplicates)]
        for _, row in non_duplicates.iterrows():
            merged_record = row.to_dict()
            merged_record['merged_from'] = None
            merged_record['merge_count'] = 1
            merged_records.append(merged_record)
        
        result_df = pd.DataFrame(merged_records)
        result_df['years_active'] = result_df['last_year'] - result_df['first_year'] + 1
        
        return result_df.sort_values('total_speeches', ascending=False)
    
    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main deduplication process."""
        self._log(f"Starting deduplication of {len(df)} speakers...")
        
        # Find duplicate groups
        self._log("Finding duplicate groups...")
        duplicate_groups = self.find_duplicate_groups(df)
        self._log(f"Found {len(duplicate_groups)} groups with duplicates")
        
        # Count total duplicates
        total_duplicates = sum(len(group) - 1 for group in duplicate_groups.values())
        self._log(f"Total duplicate records to merge: {total_duplicates}")
        
        # Merge duplicates
        self._log("Merging duplicate records...")
        deduplicated_df = self.merge_duplicate_records(df, duplicate_groups)
        
        self._log(f"Deduplication complete: {len(df)} -> {len(deduplicated_df)} speakers")
        
        return deduplicated_df


def main():
    """Main processing function."""
    # Load the normalized speakers summary
    input_path = Path('/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/data/speakers_summary.parquet')
    output_path = Path('/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/data/speakers_deduplicated.parquet')
    
    print("Loading normalized speakers data...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} speakers")
    
    # Initialize deduplicator
    deduplicator = SpeakerDeduplicator(verbose=True)
    
    # Run deduplication
    deduplicated_df = deduplicator.deduplicate(df)
    
    # Save results
    deduplicated_df.to_parquet(output_path, index=False)
    print(f"\nSaved deduplicated data to {output_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("DEDUPLICATION STATISTICS")
    print("="*60)
    print(f"Original speakers: {len(df):,}")
    print(f"Deduplicated speakers: {len(deduplicated_df):,}")
    print(f"Reduction: {len(df) - len(deduplicated_df):,} ({(1 - len(deduplicated_df)/len(df))*100:.1f}%)")
    print(f"Date range: {deduplicated_df['first_year'].min()} - {deduplicated_df['last_year'].max()}")
    
    # Show merging statistics
    merged_df = deduplicated_df[deduplicated_df['merge_count'] > 1].sort_values('merge_count', ascending=False)
    print(f"\nMerged entries: {len(merged_df):,}")
    print(f"Max merges for single person: {merged_df['merge_count'].max()}")
    
    print("\nTop 10 most-merged speakers:")
    for _, row in merged_df.head(10).iterrows():
        print(f"  {row['normalized_name']:30s} | {row['merge_count']} merged | {row['total_speeches']:,} speeches")
        if row['merged_from']:
            names = row['merged_from'].split('|')[:3]  # Show first 3
            for name in names:
                if name != row['normalized_name']:
                    print(f"    <- {name}")
    
    print("\nTop 20 speakers by speech count (after deduplication):")
    for _, row in deduplicated_df.head(20).iterrows():
        print(f"  {row['normalized_name']:30s} | {row['total_speeches']:6,} speeches | {row['first_year']}-{row['last_year']}")
    
    # Compare with gendered dataset
    print("\n" + "="*60)
    print("COMPARISON WITH GENDERED DATASET")
    print("="*60)
    try:
        gendered = pd.read_parquet('/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/data/house_members_gendered.parquet')
        print(f"Gendered dataset: {gendered['person_id'].nunique():,} unique persons")
        print(f"Our deduplicated dataset: {len(deduplicated_df):,} unique speakers")
        print(f"Difference: {len(deduplicated_df) - gendered['person_id'].nunique():,}")
    except:
        print("Could not load gendered dataset for comparison")
    
    return deduplicated_df


if __name__ == '__main__':
    deduplicated_df = main()