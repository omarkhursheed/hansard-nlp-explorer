#!/usr/bin/env python3
"""
Fast MP filtering pipeline - process select years for demonstration
"""

from process_all_debates_to_mp_filtered import FullScaleProcessor
import time

def main():
    """Run fast processing on select years"""
    print("="*80)
    print("HANSARD MP FILTERING - FAST VERSION")
    print("Processing select years for demonstration")
    print("="*80)

    # Select strategic years across the time period
    # Including: early period, pre-suffrage, post-suffrage, WWII, modern
    selected_years = [
        1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890,  # 19th century
        1900, 1910, 1918, 1920, 1928, 1930,  # Early 20th + suffrage
        1940, 1945, 1950, 1960, 1970, 1980, 1990, 2000  # Mid-late 20th
    ]

    start_time = time.time()

    # Initialize processor
    processor = FullScaleProcessor(output_base="data_filtered_by_actual_mp_sample")

    print(f"\nProcessing {len(selected_years)} strategic years...")

    all_stats = {}

    for year in selected_years:
        print(f"\nProcessing {year}...")
        year_start = time.time()

        # Process speakers
        speakers_df = processor.process_year_speakers(year)
        if speakers_df is None:
            print(f"  No data for {year}")
            continue

        # Process debates
        debates_df = processor.process_year_debates(year)
        if debates_df is None:
            debates_df = pd.DataFrame()

        # Create turn dataset
        turns_df = processor.create_turn_dataset_for_year(year, speakers_df, debates_df)

        # Save year data
        speakers_df.to_parquet(processor.dirs['speakers'] / f'speakers_{year}_filtered.parquet')

        if len(debates_df) > 0:
            debates_df.to_parquet(processor.dirs['debates'] / f'debates_{year}_filtered.parquet')

        if len(turns_df) > 0:
            turns_df.to_parquet(processor.dirs['turns'] / f'turns_{year}.parquet')

        # Collect statistics
        matched_count = speakers_df['is_mp'].sum()
        total_count = len(speakers_df)
        gender_dist = speakers_df[speakers_df['is_mp']]['gender'].value_counts().to_dict()

        all_stats[year] = {
            'total_speakers': len(speakers_df['speaker_name'].unique()),
            'total_records': total_count,
            'matched_mps': matched_count,
            'match_rate': speakers_df['is_mp'].mean(),
            'gender_M': gender_dist.get('M', 0),
            'gender_F': gender_dist.get('F', 0),
            'processing_time': time.time() - year_start
        }

        print(f"  Speakers: {all_stats[year]['total_speakers']}")
        print(f"  Match rate: {100*all_stats[year]['match_rate']:.1f}%")
        print(f"  Gender: M={gender_dist.get('M', 0)}, F={gender_dist.get('F', 0)}")
        print(f"  Time: {all_stats[year]['processing_time']:.1f}s")

    # Save report
    import json
    with open(processor.dirs['reports'] / 'fast_processing_report.json', 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)

    # Create master datasets for sample
    print("\nCreating master datasets...")
    processor.create_master_datasets()

    elapsed = time.time() - start_time

    # Summary statistics
    total_matched = sum(s.get('matched_mps', 0) for s in all_stats.values())
    total_records = sum(s.get('total_records', 0) for s in all_stats.values())
    avg_match_rate = sum(s.get('match_rate', 0) for s in all_stats.values()) / len(all_stats) if all_stats else 0

    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nTotal processing time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Years processed: {len(all_stats)}")
    print(f"Total speaker records: {total_records:,}")
    print(f"Total matched to MPs: {total_matched:,}")
    print(f"Average match rate: {100*avg_match_rate:.1f}%")

    # Gender evolution
    print("\nGender representation over time:")
    for year in sorted(all_stats.keys()):
        stats = all_stats[year]
        total_gendered = stats.get('gender_M', 0) + stats.get('gender_F', 0)
        if total_gendered > 0:
            f_pct = 100 * stats.get('gender_F', 0) / total_gendered
            print(f"  {year}: {f_pct:5.1f}% female ({stats.get('gender_F', 0):,} of {total_gendered:,})")

    print(f"\nAll data saved to: data_filtered_by_actual_mp_sample/")
    print("\nNote: This is a sample of strategic years. Run full pipeline for complete dataset.")

if __name__ == "__main__":
    import pandas as pd
    main()