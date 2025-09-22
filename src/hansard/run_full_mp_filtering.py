#!/usr/bin/env python3
"""
Run the full MP filtering pipeline on all Hansard data
"""

from process_all_debates_to_mp_filtered import FullScaleProcessor
import time

def main():
    """Run full processing pipeline"""
    print("="*80)
    print("HANSARD MP FILTERING PIPELINE")
    print("Processing 202 years of parliamentary debates (1803-2005)")
    print("="*80)

    start_time = time.time()

    # Initialize processor
    processor = FullScaleProcessor(output_base="data_filtered_by_actual_mp")

    # Process all years
    print("\nProcessing all years...")
    all_stats = processor.process_all_years(start_year=1803, end_year=2005, n_workers=4)

    # Create master datasets
    print("\nCreating master datasets...")
    processor.create_master_datasets()

    # Calculate overall statistics
    total_speakers = sum(s.get('total_speakers', 0) for s in all_stats.values())
    total_matched = sum(s.get('matched_mps', 0) for s in all_stats.values())
    avg_match_rate = sum(s.get('match_rate', 0) for s in all_stats.values()) / len(all_stats)

    elapsed = time.time() - start_time

    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nTotal processing time: {elapsed/60:.1f} minutes")
    print(f"Years processed: {len(all_stats)}")
    print(f"Total speakers encountered: {total_speakers:,}")
    print(f"Total matched to MPs: {total_matched:,}")
    print(f"Average match rate: {100*avg_match_rate:.1f}%")
    print(f"\nAll data saved to: data_filtered_by_actual_mp/")

if __name__ == "__main__":
    main()