#!/usr/bin/env python3
"""
Quick fix for speakers with unrealistic career spans.
Splits speakers like "Mr. Brown" who span 200+ years into era-based groups.
"""

import pandas as pd
from pathlib import Path

def fix_unrealistic_spans(input_file, output_file, max_span=60):
    """Fix speakers with unrealistic career spans."""
    
    print(f"Loading {input_file}...")
    df = pd.read_parquet(input_file)
    
    # Check for unrealistic spans
    df['career_span'] = df['last_year'] - df['first_year'] + 1
    unrealistic = df[df['career_span'] > max_span]
    
    print(f"Found {len(unrealistic)} speakers with spans > {max_span} years")
    
    if len(unrealistic) == 0:
        print("No unrealistic spans found!")
        df.to_parquet(output_file)
        return df
    
    # Process all speakers, splitting those with unrealistic spans
    fixed_records = []
    
    for _, speaker in df.iterrows():
        if speaker['career_span'] <= max_span:
            # Keep as-is
            fixed_records.append(speaker.to_dict())
        else:
            # Split into eras
            print(f"Splitting {speaker['normalized_name']} ({speaker['first_year']}-{speaker['last_year']})")
            
            # Simple era splitting: every 50 years gets a new era
            era_start = speaker['first_year']
            era_num = 1
            
            while era_start <= speaker['last_year']:
                era_end = min(era_start + max_span - 1, speaker['last_year'])
                
                # Create era-specific record
                era_record = speaker.to_dict().copy()
                era_record['normalized_name'] = f"{speaker['normalized_name']} (era {era_num})"
                era_record['first_year'] = era_start
                era_record['last_year'] = era_end
                era_record['years_active'] = era_end - era_start + 1
                era_record['career_span'] = era_end - era_start + 1
                
                # Approximate speech distribution (proportional to years)
                year_fraction = (era_end - era_start + 1) / speaker['career_span']
                era_record['total_speeches'] = int(speaker['total_speeches'] * year_fraction)
                
                fixed_records.append(era_record)
                
                era_start = era_end + 1
                era_num += 1
    
    # Create new dataframe
    fixed_df = pd.DataFrame(fixed_records)
    
    # Re-sort by total speeches
    fixed_df = fixed_df.sort_values('total_speeches', ascending=False)
    
    # Save
    fixed_df.to_parquet(output_file, index=False)
    
    print(f"\nFixed dataset saved to {output_file}")
    print(f"Original: {len(df)} speakers")
    print(f"Fixed: {len(fixed_df)} speakers")
    
    # Verify no unrealistic spans remain
    fixed_df['career_span'] = fixed_df['last_year'] - fixed_df['first_year'] + 1
    remaining_unrealistic = fixed_df[fixed_df['career_span'] > max_span]
    print(f"Remaining unrealistic spans: {len(remaining_unrealistic)}")
    
    return fixed_df


if __name__ == '__main__':
    # Fix the deduplicated dataset
    input_file = Path('/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/data/speakers_deduplicated.parquet')
    output_file = Path('/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/data/speakers_deduplicated_fixed.parquet')
    
    fixed_df = fix_unrealistic_spans(input_file, output_file)