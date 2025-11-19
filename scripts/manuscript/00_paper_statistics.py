#!/usr/bin/env python3
"""
Generate statistics for research paper Data section.
Analyzes Hansard dataset for gender and suffrage debate statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data-hansard"
DEBATE_METADATA = DATA_DIR / "gender_analysis_complete" / "ALL_debates_enhanced_metadata.parquet"
SPEECHES_DIR = DATA_DIR / "derived_complete" / "speeches_complete"
SUFFRAGE_DATA = BASE_DIR / "outputs" / "suffrage_debates" / "all_speeches_in_suffrage_debates.parquet"
MP_DATA = DATA_DIR / "house_members_gendered_updated.parquet"

def load_data():
    """Load all required datasets."""
    print("Loading datasets...")
    print(f"  - Debate metadata: {DEBATE_METADATA}")
    debates_meta = pd.read_parquet(DEBATE_METADATA)

    print(f"  - Suffrage speeches: {SUFFRAGE_DATA}")
    suffrage = pd.read_parquet(SUFFRAGE_DATA)

    print(f"  - MP data: {MP_DATA}")
    mps = pd.read_parquet(MP_DATA)

    return debates_meta, suffrage, mps

def load_all_speeches():
    """Load all speeches from year files."""
    print("Loading all speeches (this may take a minute)...")
    speech_files = sorted(SPEECHES_DIR.glob("speeches_*.parquet"))
    print(f"Found {len(speech_files)} speech files")

    all_speeches = []
    for file in tqdm(speech_files, desc="Loading speeches"):
        df = pd.read_parquet(file)
        all_speeches.append(df)

    speeches = pd.concat(all_speeches, ignore_index=True)
    print(f"Loaded {len(speeches):,} total speeches")
    return speeches

def analyze_full_dataset(speeches):
    """Calculate statistics for full Hansard dataset."""
    print("\n" + "="*70)
    print("1. FULL HANSARD DATASET")
    print("="*70)

    # Use canonical_name for unique speakers (handles normalization)
    unique_speakers = speeches[speeches['canonical_name'].notna()]['canonical_name'].nunique()

    stats = {
        'total_speeches': len(speeches),
        'total_debates': speeches['debate_id'].nunique(),
        'total_speakers': unique_speakers,
    }

    # Date range - handle potential NaN values
    if 'date' in speeches.columns:
        valid_dates = speeches['date'].dropna()
        if len(valid_dates) > 0:
            stats['date_range_start'] = valid_dates.min()
            stats['date_range_end'] = valid_dates.max()
        else:
            stats['date_range_start'] = None
            stats['date_range_end'] = None
    else:
        stats['date_range_start'] = None
        stats['date_range_end'] = None

    # Year range
    if 'year' in speeches.columns:
        valid_years = speeches['year'].dropna()
        if len(valid_years) > 0:
            stats['year_range_start'] = int(valid_years.min())
            stats['year_range_end'] = int(valid_years.max())
        else:
            stats['year_range_start'] = None
            stats['year_range_end'] = None
    else:
        stats['year_range_start'] = None
        stats['year_range_end'] = None

    print(f"Total speeches: {stats['total_speeches']:,}")
    print(f"Total debates: {stats['total_debates']:,}")
    print(f"Total unique speakers: {stats['total_speakers']:,}")

    if stats['date_range_start']:
        print(f"Date range: {stats['date_range_start']} to {stats['date_range_end']}")
    if stats['year_range_start']:
        print(f"Year range: {stats['year_range_start']} to {stats['year_range_end']}")

    # Chamber breakdown
    if 'chamber' in speeches.columns:
        chamber_counts = speeches['chamber'].value_counts()
        print(f"\nChamber breakdown (speeches):")
        for chamber, count in chamber_counts.items():
            print(f"  {chamber}: {count:,} ({count/len(speeches)*100:.1f}%)")

    return stats

def analyze_commons(speeches):
    """Calculate statistics for House of Commons only."""
    print("\n" + "="*70)
    print("2. HOUSE OF COMMONS ONLY")
    print("="*70)

    # Filter for Commons
    commons = speeches[speeches['chamber'] == 'Commons'].copy()

    # Count unique MPs using canonical_name
    unique_mps = commons[commons['canonical_name'].notna()]['canonical_name'].nunique()

    stats = {
        'total_speeches': len(commons),
        'total_mps': unique_mps,
    }

    print(f"Total speeches in Commons: {stats['total_speeches']:,}")
    print(f"Number of unique MPs in Commons: {stats['total_mps']:,}")

    # Gender breakdown (total unique MPs)
    if 'gender' in commons.columns:
        # Count unique MPs by gender using canonical_name
        mp_gender = commons[commons['gender'].notna() & commons['canonical_name'].notna()].groupby('canonical_name')['gender'].first()
        gender_counts = mp_gender.value_counts()

        print(f"\nGender breakdown (unique MPs):")
        for gender, count in gender_counts.items():
            gender_label = 'Male' if gender == 'M' else 'Female' if gender == 'F' else gender.capitalize()
            print(f"  {gender_label}: {count:,} ({count/len(mp_gender)*100:.1f}%)")

        stats['male_mps'] = gender_counts.get('M', 0)
        stats['female_mps'] = gender_counts.get('F', 0)

        # First female MP appearance
        female_speeches = commons[commons['gender'] == 'F']
        if len(female_speeches) > 0:
            # Get first occurrence by year
            valid_years = female_speeches[female_speeches['year'].notna()]
            if len(valid_years) > 0:
                first_female_year = int(valid_years['year'].min())
                # Get the row with minimum year to find name and date
                first_female_row = valid_years[valid_years['year'] == first_female_year].iloc[0]
                first_female_date = first_female_row['date'] if 'date' in first_female_row else None
                first_female_name = first_female_row['canonical_name']

                print(f"\nFirst female MP appearance:")
                print(f"  Name: {first_female_name}")
                if first_female_date:
                    print(f"  Date: {first_female_date}")
                print(f"  Year: {first_female_year}")

                stats['first_female_year'] = first_female_year
                stats['first_female_date'] = first_female_date
                stats['first_female_name'] = first_female_name

    return commons, stats

def analyze_gender_by_decade(commons):
    """Analyze gender distribution by decade."""
    print("\n" + "="*70)
    print("3. GENDER BREAKDOWN BY DECADE")
    print("="*70)

    if 'year' not in commons.columns or 'gender' not in commons.columns:
        print("Year or gender data not available")
        return None

    # Add decade column
    commons_copy = commons.copy()
    commons_copy['decade'] = (commons_copy['year'] // 10) * 10

    # Count unique MPs per decade by gender using canonical_name
    decade_stats = []

    for decade in sorted(commons_copy['decade'].unique()):
        decade_data = commons_copy[commons_copy['decade'] == decade]

        # Get unique MPs in this decade using canonical_name
        mps_in_decade = decade_data[
            decade_data['gender'].notna() & decade_data['canonical_name'].notna()
        ].groupby('canonical_name')['gender'].first()

        total_mps = len(mps_in_decade)
        if total_mps == 0:
            continue

        gender_counts = mps_in_decade.value_counts()
        female_count = gender_counts.get('F', 0)
        male_count = gender_counts.get('M', 0)
        female_pct = (female_count / total_mps * 100) if total_mps > 0 else 0

        decade_stats.append({
            'decade': f"{decade}s",
            'total_mps': total_mps,
            'female_mps': female_count,
            'male_mps': male_count,
            'female_pct': female_pct
        })

        print(f"{decade}s: {female_count:,} female ({female_pct:.1f}%), {male_count:,} male, {total_mps:,} total")

    return pd.DataFrame(decade_stats)

def analyze_suffrage_speeches(suffrage):
    """Calculate statistics for suffrage-related speeches."""
    print("\n" + "="*70)
    print("4. SUFFRAGE-RELATED SPEECHES")
    print("="*70)

    # Filter for speeches classified as suffrage-related (if column exists)
    if 'is_suffrage_speech' in suffrage.columns:
        suffrage_only = suffrage[suffrage['is_suffrage_speech'] == True].copy()
        print(f"Using is_suffrage_speech filter: {len(suffrage_only):,} speeches")
    else:
        suffrage_only = suffrage.copy()

    unique_speakers = suffrage_only[suffrage_only['canonical_name'].notna()]['canonical_name'].nunique()

    stats = {
        'total_speeches': len(suffrage_only),
        'unique_speakers': unique_speakers,
    }

    print(f"Total suffrage speeches: {stats['total_speeches']:,}")

    # Date range
    if 'date' in suffrage_only.columns:
        valid_dates = suffrage_only['date'].dropna()
        if len(valid_dates) > 0:
            stats['date_range_start'] = valid_dates.min()
            stats['date_range_end'] = valid_dates.max()
            print(f"Date range: {stats['date_range_start']} to {stats['date_range_end']}")
        else:
            stats['date_range_start'] = None
            stats['date_range_end'] = None

    if 'year' in suffrage_only.columns:
        valid_years = suffrage_only['year'].dropna()
        if len(valid_years) > 0:
            stats['year_range_start'] = int(valid_years.min())
            stats['year_range_end'] = int(valid_years.max())
            print(f"Year range: {stats['year_range_start']} to {stats['year_range_end']}")
        else:
            stats['year_range_start'] = None
            stats['year_range_end'] = None

    # Unique speakers
    print(f"Unique speakers discussing suffrage: {stats['unique_speakers']:,}")

    # Gender breakdown of speeches
    if 'gender' in suffrage_only.columns:
        gender_speech_counts = suffrage_only[suffrage_only['gender'].notna()]['gender'].value_counts()
        print(f"\nGender breakdown of suffrage speeches:")
        total_with_gender = len(suffrage_only[suffrage_only['gender'].notna()])
        for gender, count in gender_speech_counts.items():
            gender_label = 'Male' if gender == 'M' else 'Female' if gender == 'F' else gender
            pct = count / total_with_gender * 100 if total_with_gender > 0 else 0
            print(f"  {gender_label}: {count:,} speeches ({pct:.1f}%)")

        stats['male_speeches'] = gender_speech_counts.get('M', 0)
        stats['female_speeches'] = gender_speech_counts.get('F', 0)

        # Gender breakdown of unique speakers
        speaker_gender = suffrage_only[
            suffrage_only['gender'].notna() & suffrage_only['canonical_name'].notna()
        ].groupby('canonical_name')['gender'].first()
        unique_gender_counts = speaker_gender.value_counts()
        print(f"\nGender breakdown of unique speakers:")
        for gender, count in unique_gender_counts.items():
            gender_label = 'Male' if gender == 'M' else 'Female' if gender == 'F' else gender
            pct = count / len(speaker_gender) * 100 if len(speaker_gender) > 0 else 0
            print(f"  {gender_label}: {count:,} speakers ({pct:.1f}%)")

        stats['male_speakers'] = unique_gender_counts.get('M', 0)
        stats['female_speakers'] = unique_gender_counts.get('F', 0)

    return stats

def analyze_validation_numbers(speeches):
    """Calculate key validation numbers."""
    print("\n" + "="*70)
    print("5. VALIDATION NUMBERS")
    print("="*70)

    if 'year' not in speeches.columns:
        print("Year data not available")
        return None

    # Commons only
    commons = speeches[speeches['chamber'] == 'Commons']

    # Pre-1918 (before women entered Commons)
    pre_1918 = commons[commons['year'] < 1918]
    post_1918 = commons[commons['year'] >= 1918]

    print(f"Speeches before 1918 (pre-women in Commons): {len(pre_1918):,}")
    print(f"Speeches from 1918-2005: {len(post_1918):,}")

    # Gender validation in these periods
    if 'gender' in commons.columns:
        print(f"\nGender breakdown before 1918:")
        pre_gender = pre_1918[pre_1918['gender'].notna()]['gender'].value_counts()
        for gender, count in pre_gender.items():
            gender_label = 'Male' if gender == 'M' else 'Female' if gender == 'F' else gender
            print(f"  {gender_label}: {count:,}")

        print(f"\nGender breakdown 1918-2005:")
        post_gender = post_1918[post_1918['gender'].notna()]['gender'].value_counts()
        for gender, count in post_gender.items():
            gender_label = 'Male' if gender == 'M' else 'Female' if gender == 'F' else gender
            print(f"  {gender_label}: {count:,}")

    stats = {
        'pre_1918_speeches': len(pre_1918),
        'post_1918_speeches': len(post_1918),
    }

    return stats

def main():
    """Main analysis function."""
    print("HANSARD DATASET STATISTICS FOR RESEARCH PAPER")
    print("="*70)

    # Load metadata
    debates_meta, suffrage, mps = load_data()

    # Load all speeches
    speeches = load_all_speeches()

    # Run analyses
    full_stats = analyze_full_dataset(speeches)
    commons, commons_stats = analyze_commons(speeches)
    decade_stats = analyze_gender_by_decade(commons)
    suffrage_stats = analyze_suffrage_speeches(suffrage)
    validation_stats = analyze_validation_numbers(speeches)

    # Save results
    output_dir = BASE_DIR / "analysis" / "paper_statistics"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save decade statistics as CSV
    if decade_stats is not None:
        decade_file = output_dir / "gender_by_decade.csv"
        decade_stats.to_csv(decade_file, index=False)
        print(f"\nSaved decade statistics to: {decade_file}")

    # Save all statistics as JSON
    all_stats = {
        'full_dataset': full_stats,
        'commons': commons_stats,
        'suffrage': suffrage_stats,
        'validation': validation_stats,
    }

    # Convert any Timestamp objects to strings for JSON serialization
    def convert_timestamps(obj):
        if isinstance(obj, dict):
            return {k: convert_timestamps(v) for k, v in obj.items()}
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

    all_stats = convert_timestamps(all_stats)

    stats_file = output_dir / "all_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"Saved all statistics to: {stats_file}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
