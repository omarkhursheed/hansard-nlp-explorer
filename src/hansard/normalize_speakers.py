#!/usr/bin/env python3
"""
Script to filter and normalize speakers from speakers_master.parquet
to create a clean dataset of actual parliamentary members.
"""

import pandas as pd
import re
from typing import Set, Optional
import numpy as np
from pathlib import Path


class SpeakerNormalizer:
    """Clean and normalize speaker names from Hansard data."""
    
    def __init__(self):
        self.excluded_patterns = self._build_exclusion_patterns()
        self.title_variations = self._build_title_variations()
        
    def _build_exclusion_patterns(self) -> Set[str]:
        """Build set of patterns that indicate non-individual speakers."""
        return {
            # Procedural roles
            'speaker', 'chairman', 'chairwoman', 'clerk', 'deputy',
            'chair', 'moderator', 'presiding',
            
            # Generic references
            'members', 'member', 'several', 'many', 'various', 'some',
            'voices', 'benches', 'opposition', 'government benches',
            
            # Role-only references (without names)
            'the prime minister', 'the chancellor', 'the secretary',
            'the minister', 'the attorney', 'the solicitor',
            'the president', 'the leader', 'the whip',
            
            # Collective groups
            'noble lords', 'hon. members', 'hon members', 'honourable members',
            'several hon', 'many hon', 'some hon', 'government members',
            
            # Procedural phrases
            'question put', 'division', 'question proposed', 'amendment',
            'the question', 'interruption', 'continued', 'resumed',
            
            # Non-person entries
            'notice', 'answer', 'reply', 'statement', 'message',
        }
    
    def _build_title_variations(self) -> dict:
        """Build mapping of title variations to canonical forms."""
        return {
            # Mr variations
            'MR.': 'Mr.', 'MR': 'Mr.', 'Mr': 'Mr.',
            'Mister': 'Mr.', 'MISTER': 'Mr.',
            
            # Mrs variations
            'MRS.': 'Mrs.', 'MRS': 'Mrs.', 'Mrs': 'Mrs.',
            
            # Miss variations
            'MISS': 'Miss', 'Ms.': 'Ms.', 'MS.': 'Ms.', 'MS': 'Ms.',
            
            # Sir variations
            'SIR': 'Sir', 'Sir.': 'Sir',
            
            # Lady variations
            'LADY': 'Lady',
            
            # Dr variations
            'DR.': 'Dr.', 'DR': 'Dr.', 'Dr': 'Dr.', 'Doctor': 'Dr.',
            
            # Military titles
            'MAJOR': 'Major', 'Major.': 'Major',
            'CAPTAIN': 'Captain', 'Captain.': 'Captain', 'Capt.': 'Captain',
            'COLONEL': 'Colonel', 'Colonel.': 'Colonel', 'Col.': 'Colonel',
            'GENERAL': 'General', 'General.': 'General', 'Gen.': 'General',
            'LIEUT.': 'Lieut.', 'Lieutenant': 'Lieut.', 'Lt.': 'Lieut.',
            'LIEUT.-COMMANDER': 'Lieut.-Commander', 'Lieutenant-Commander': 'Lieut.-Commander',
            'COMMANDER': 'Commander', 'Commander.': 'Commander',
            'ADMIRAL': 'Admiral', 'Admiral.': 'Admiral',
            
            # Noble titles
            'LORD': 'Lord', 'EARL': 'Earl', 'VISCOUNT': 'Viscount',
            'MARQUESS': 'Marquess', 'MARQUIS': 'Marquess',
            'DUKE': 'Duke', 'BARON': 'Baron', 'BARONESS': 'Baroness',
            'COUNT': 'Count', 'COUNTESS': 'Countess',
            
            # Academic/Professional
            'PROFESSOR': 'Professor', 'Prof.': 'Professor',
            'REVEREND': 'Reverend', 'Rev.': 'Reverend', 'REV.': 'Reverend',
        }
    
    def is_excluded_speaker(self, name: str) -> bool:
        """Check if a speaker name should be excluded."""
        if pd.isna(name) or not name.strip():
            return True
            
        name_lower = name.lower().strip()
        
        # Check for excluded patterns
        for pattern in self.excluded_patterns:
            if pattern in name_lower:
                # Exception: Allow if it's part of an actual name
                # e.g., "Mr. Speaker-Smith" would be excluded, but "Mr. Speakerman" might be real
                if pattern in ['speaker', 'chairman', 'member'] and not name_lower.startswith(pattern):
                    # Check if it's likely a role reference
                    if name_lower.startswith(('mr. speaker', 'the speaker', 'madam speaker',
                                             'mr. chairman', 'the chairman', 'hon. member')):
                        return True
                else:
                    return True
        
        # Check if it's just a title without a name
        if name_lower in ['mr.', 'mrs.', 'miss', 'ms.', 'sir', 'lady', 'dr.', 'lord']:
            return True
            
        # Check for role-only patterns (title + role but no actual name)
        role_pattern = r'^(the\s+)?(mr\.|mrs\.|miss|ms\.|sir|lady|dr\.)?\s*(speaker|chairman|deputy|clerk)$'
        if re.match(role_pattern, name_lower):
            return True
            
        return False
    
    def normalize_name(self, name: str) -> Optional[str]:
        """Normalize a speaker name to a canonical form."""
        if pd.isna(name):
            return None
            
        name = str(name).strip()
        
        # Handle all caps names
        if name.isupper() and len(name.split()) > 1:
            # Convert to title case but preserve certain patterns
            parts = []
            for part in name.split():
                if part in ['OF', 'THE', 'AND']:
                    parts.append(part.lower())
                elif '-' in part:
                    # Handle hyphenated names
                    sub_parts = [sp.capitalize() for sp in part.split('-')]
                    parts.append('-'.join(sub_parts))
                else:
                    parts.append(part.capitalize())
            name = ' '.join(parts)
        
        # Normalize titles
        for variant, canonical in self.title_variations.items():
            if name.startswith(variant + ' '):
                name = canonical + name[len(variant):]
                break
            elif name.upper().startswith(variant.upper() + ' '):
                name = canonical + name[len(variant):]
                break
        
        # Normalize spaces and punctuation
        name = re.sub(r'\s+', ' ', name)  # Multiple spaces to single
        name = re.sub(r'\s+\.', '.', name)  # Remove space before period
        
        # Handle specific patterns
        # "MR. A. J. BALFOUR" -> "Mr. A. J. Balfour"
        if re.match(r'^[A-Z]+\.\s+[A-Z]\.\s+[A-Z]\.\s+[A-Z]+', name):
            parts = name.split()
            parts[0] = parts[0].capitalize() + '.'
            parts[-1] = parts[-1].capitalize()
            name = ' '.join(parts)
        
        return name
    
    def extract_surname(self, name: str) -> Optional[str]:
        """Extract likely surname from a normalized name."""
        if not name:
            return None
            
        # Remove titles
        for title in ['Mr.', 'Mrs.', 'Miss', 'Ms.', 'Sir', 'Lady', 'Dr.', 
                      'Major', 'Captain', 'Colonel', 'General', 'Lieut.',
                      'Commander', 'Admiral', 'Professor', 'Reverend']:
            if name.startswith(title + ' '):
                name = name[len(title)+1:].strip()
                break
        
        # For noble titles, the title itself might be the identifier
        noble_titles = ['Lord', 'Earl', 'Viscount', 'Marquess', 'Duke', 'Baron']
        for title in noble_titles:
            if name.startswith(title + ' '):
                return name  # Keep the full noble title as identifier
        
        # Extract last word as surname (simple heuristic)
        parts = name.split()
        if parts:
            return parts[-1]
        return None
    
    def process_speakers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the speakers dataframe to filter and normalize names."""
        print(f"Starting with {len(df)} speaker records from {df['speaker_name'].nunique()} unique names")
        
        # Filter out excluded speakers
        df['is_excluded'] = df['speaker_name'].apply(self.is_excluded_speaker)
        filtered_df = df[~df['is_excluded']].copy()
        print(f"After filtering: {len(filtered_df)} records from {filtered_df['speaker_name'].nunique()} unique names")
        
        # Normalize names
        filtered_df['normalized_name'] = filtered_df['speaker_name'].apply(self.normalize_name)
        
        # Extract surnames for potential deduplication
        filtered_df['surname'] = filtered_df['normalized_name'].apply(self.extract_surname)
        
        # Drop records with no valid normalized name
        filtered_df = filtered_df[filtered_df['normalized_name'].notna()]
        print(f"After normalization: {len(filtered_df)} records from {filtered_df['normalized_name'].nunique()} unique normalized names")
        
        return filtered_df


def create_member_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary of unique members with their activity statistics.
    
    Split speakers with the same name if they have unrealistic career spans.
    """
    # Maximum realistic career span
    MAX_CAREER_SPAN = 60
    
    # First, sort by normalized_name and year
    df_sorted = df.sort_values(['normalized_name', 'year'])
    
    summary_records = []
    
    # Process each unique normalized name
    for name in df_sorted['normalized_name'].unique():
        name_df = df_sorted[df_sorted['normalized_name'] == name].copy()
        
        # Check if this name spans an unrealistic time period
        min_year = name_df['year'].min()
        max_year = name_df['year'].max()
        span = max_year - min_year + 1
        
        if span <= MAX_CAREER_SPAN:
            # Normal case: aggregate all records for this name
            record = {
                'normalized_name': name,
                'most_common_form': name_df['speaker_name'].mode()[0] if len(name_df['speaker_name'].mode()) > 0 else name_df['speaker_name'].iloc[0],
                'first_year': min_year,
                'last_year': max_year,
                'total_speeches': len(name_df),
                'primary_chamber': name_df['chamber'].mode()[0] if len(name_df['chamber'].mode()) > 0 else name_df['chamber'].iloc[0],
                'surname': name_df['surname'].iloc[0]
            }
            summary_records.append(record)
        else:
            # Split into multiple era-based groups
            # Use a simple clustering approach based on year gaps
            name_df = name_df.sort_values('year')
            
            current_group = []
            current_start = name_df.iloc[0]['year']
            era_counter = 1
            
            for _, row in name_df.iterrows():
                # If this year would make the group span too long, start a new group
                if current_group and (row['year'] - current_start + 1 > MAX_CAREER_SPAN):
                    # Save current group
                    group_df = pd.DataFrame(current_group)
                    record = {
                        'normalized_name': f"{name} (era {era_counter})",
                        'most_common_form': group_df['speaker_name'].mode()[0] if len(group_df['speaker_name'].mode()) > 0 else group_df['speaker_name'].iloc[0],
                        'first_year': group_df['year'].min(),
                        'last_year': group_df['year'].max(),
                        'total_speeches': len(group_df),
                        'primary_chamber': group_df['chamber'].mode()[0] if len(group_df['chamber'].mode()) > 0 else group_df['chamber'].iloc[0],
                        'surname': group_df['surname'].iloc[0]
                    }
                    summary_records.append(record)
                    
                    # Start new group
                    current_group = [row.to_dict()]
                    current_start = row['year']
                    era_counter += 1
                else:
                    current_group.append(row.to_dict())
            
            # Save final group
            if current_group:
                group_df = pd.DataFrame(current_group)
                record = {
                    'normalized_name': f"{name} (era {era_counter})" if era_counter > 1 else name,
                    'most_common_form': group_df['speaker_name'].mode()[0] if len(group_df['speaker_name'].mode()) > 0 else group_df['speaker_name'].iloc[0],
                    'first_year': group_df['year'].min(),
                    'last_year': group_df['year'].max(),
                    'total_speeches': len(group_df),
                    'primary_chamber': group_df['chamber'].mode()[0] if len(group_df['chamber'].mode()) > 0 else group_df['chamber'].iloc[0],
                    'surname': group_df['surname'].iloc[0]
                }
                summary_records.append(record)
    
    summary = pd.DataFrame(summary_records)
    
    # Calculate years active
    summary['years_active'] = summary['last_year'] - summary['first_year'] + 1
    
    # Sort by total speeches
    summary = summary.sort_values('total_speeches', ascending=False)
    
    return summary


def main():
    """Main processing function."""
    # Input and output paths
    input_path = Path('/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/data/processed_fixed/metadata/speakers_master.parquet')
    output_dir = Path('/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/data')
    
    # Load data
    print("Loading speakers_master.parquet...")
    df = pd.read_parquet(input_path)
    
    # Initialize normalizer and process
    normalizer = SpeakerNormalizer()
    processed_df = normalizer.process_speakers(df)
    
    # Create member summary
    print("\nCreating member summary...")
    member_summary = create_member_summary(processed_df)
    
    # Save outputs
    output_normalized = output_dir / 'speakers_normalized.parquet'
    output_summary = output_dir / 'speakers_summary.parquet'
    
    processed_df.to_parquet(output_normalized, index=False)
    member_summary.to_parquet(output_summary, index=False)
    
    print(f"\nOutputs saved:")
    print(f"  - Normalized speakers: {output_normalized}")
    print(f"  - Member summary: {output_summary}")
    
    # Print statistics
    print(f"\nFinal statistics:")
    print(f"  - Unique normalized members: {member_summary['normalized_name'].nunique()}")
    print(f"  - Date range: {member_summary['first_year'].min()} - {member_summary['last_year'].max()}")
    print(f"  - Members with 1000+ speeches: {len(member_summary[member_summary['total_speeches'] >= 1000])}")
    
    print(f"\nTop 20 most active speakers:")
    for _, row in member_summary.head(20).iterrows():
        print(f"  {row['normalized_name']:30s} | {row['total_speeches']:6d} speeches | {row['first_year']}-{row['last_year']}")
    
    return processed_df, member_summary


if __name__ == '__main__':
    processed_df, member_summary = main()