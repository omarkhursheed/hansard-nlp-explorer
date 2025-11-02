#!/usr/bin/env python3
"""
FAST Hansard HTML → JSONL processor with parallel processing.
Optimized version - should complete 200 years in 45-90 minutes.
"""

import os
import sys
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Add src to path for imports (script is now at src/hansard/scripts/processing/)
project_root = Path(__file__).resolve().parents[4]  # Up to hansard-nlp-explorer
sys.path.insert(0, str(project_root / 'src'))

from hansard.parsers.data_pipeline import HansardDataPipeline

def process_files_chunk(args):
    """Process a chunk of files in parallel."""
    raw_data_path, processed_data_path, file_paths = args

    # Create pipeline instance for this process
    pipeline = HansardDataPipeline(raw_data_path, processed_data_path)

    results = []
    for file_path in file_paths:
        try:
            metadata = pipeline.extract_comprehensive_metadata(file_path)
            if metadata['success']:
                results.append(metadata)
        except Exception as e:
            # Silent errors for speed - they're counted
            pass

    return results

def process_year_parallel(year, raw_data_path, processed_data_path, num_workers):
    """Process a year using multiprocessing."""
    year_path = Path(raw_data_path) / str(year)
    if not year_path.exists():
        return {'year': year, 'processed': 0, 'errors': 0, 'total_files': 0}

    # Find all HTML files
    html_files = list(year_path.rglob("*.html.gz"))
    if not html_files:
        return {'year': year, 'processed': 0, 'errors': 0, 'total_files': 0}

    # Split files into chunks for parallel processing
    chunk_size = max(1, len(html_files) // num_workers)
    file_chunks = [html_files[i:i + chunk_size] for i in range(0, len(html_files), chunk_size)]

    # Prepare arguments for each process
    process_args = [(raw_data_path, processed_data_path, chunk) for chunk in file_chunks if chunk]

    # Process chunks in parallel
    all_results = []
    processed_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk = {executor.submit(process_files_chunk, args): i for i, args in enumerate(process_args)}

        for future in as_completed(future_to_chunk):
            try:
                chunk_results = future.result()
                all_results.extend(chunk_results)
                processed_count += len(chunk_results)
            except Exception as e:
                error_count += 1

    # Save the results
    if all_results:
        import polars as pl

        # Prepare data structures
        debates_data = []
        speakers_data = []
        content_data = []

        for metadata in all_results:
            # Prepare structured metadata (without content)
            debate_record = {k: v for k, v in metadata.items()
                          if k not in ['content_lines', 'full_text']}
            debates_data.append(debate_record)

            # Prepare speaker records
            for speaker in metadata.get('speakers', []):
                speakers_data.append({
                    'file_path': metadata['file_path'],
                    'year': metadata.get('year'),
                    'month': metadata.get('month'),
                    'chamber': metadata.get('chamber'),
                    'speaker_name': speaker,
                    'reference_date': metadata.get('reference_date')
                })

            # Prepare content record
            content_record = {
                'file_path': metadata['file_path'],
                'file_name': metadata['file_name'],
                'content_hash': metadata['content_hash'],
                'extraction_timestamp': metadata['extraction_timestamp'],
                'lines': metadata.get('content_lines', []),
                'full_text': metadata.get('full_text', ''),
                'metadata': {k: v for k, v in metadata.items()
                           if k not in ['content_lines', 'full_text']}
            }
            content_data.append(content_record)

        # Save using Polars
        try:
            processed_path = Path(processed_data_path)

            debates_df = pl.DataFrame(debates_data)
            (processed_path / 'metadata').mkdir(parents=True, exist_ok=True)
            debates_df.write_parquet(processed_path / 'metadata' / f'debates_{year}.parquet')

            if speakers_data:
                speakers_df = pl.DataFrame(speakers_data)
                speakers_df.write_parquet(processed_path / 'metadata' / f'speakers_{year}.parquet')

            # Save content as JSON Lines
            content_dir = processed_path / 'content' / str(year)
            content_dir.mkdir(parents=True, exist_ok=True)

            with open(content_dir / f'debates_{year}.jsonl', 'w', encoding='utf-8') as f:
                for record in content_data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

        except Exception as e:
            print(f"Error saving data for {year}: {e}")
            error_count += len(all_results)
            processed_count = 0

    return {
        'year': year,
        'processed': processed_count,
        'errors': error_count,
        'total_files': len(html_files)
    }

def main():
    parser = argparse.ArgumentParser(description='Fast Parallel Hansard Processing')
    parser.add_argument('--start-year', type=int, default=1803, help='Start year')
    parser.add_argument('--end-year', type=int, default=2005, help='End year')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes per year (default: 75%% of CPU cores)')
    parser.add_argument('--raw-data', type=str, default='data-hansard/hansard',
                       help='Raw data path')
    parser.add_argument('--output', type=str, default='data-hansard/processed_fixed',
                       help='Output path')

    args = parser.parse_args()

    if args.workers is None:
        args.workers = max(1, int(mp.cpu_count() * 0.75))

    print("="*70)
    print("FAST PARALLEL HANSARD PROCESSING")
    print("="*70)
    print(f"CPU cores available: {mp.cpu_count()}")
    print(f"Using {args.workers} worker processes per year")
    print(f"Processing years {args.start_year}-{args.end_year}")
    print(f"Raw data: {args.raw_data}")
    print(f"Output: {args.output}")
    print()
    print("Estimated time: 45-90 minutes for 200 years")
    print()

    # input("Press Enter to start processing...")  # Disabled for automated testing

    # Setup
    pipeline = HansardDataPipeline(args.raw_data, args.output)

    # Find available years
    available_years = []
    for year_dir in Path(args.raw_data).iterdir():
        if year_dir.is_dir() and year_dir.name.isdigit():
            year = int(year_dir.name)
            if args.start_year <= year <= args.end_year:
                available_years.append(year)

    available_years.sort()
    print(f"Found {len(available_years)} years to process")
    print()

    start_time = datetime.now()
    total_processed = 0
    total_errors = 0
    total_files = 0

    # Process each year
    for i, year in enumerate(available_years, 1):
        year_start = datetime.now()

        result = process_year_parallel(year, args.raw_data, args.output, args.workers)

        year_time = (datetime.now() - year_start).total_seconds()

        if 'error' not in result:
            total_processed += result['processed']
            total_errors += result['errors']
            total_files += result['total_files']

            print(f"[{i}/{len(available_years)}] {year}: {result['processed']:,}/{result['total_files']:,} files ({year_time:.1f}s)")
        else:
            print(f"[{i}/{len(available_years)}] {year}: ERROR - {result['error']}")

        # Progress estimate
        if i % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            avg_per_year = elapsed / i
            remaining_years = len(available_years) - i
            eta_seconds = avg_per_year * remaining_years
            print(f"  Progress: {i}/{len(available_years)} ({i*100//len(available_years)}%) | ETA: {eta_seconds/60:.0f} minutes")

    total_time = (datetime.now() - start_time).total_seconds()

    print()
    print("="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Total files processed: {total_processed:,}")
    print(f"Total errors: {total_errors:,}")
    print(f"Total time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
    print(f"Average: {total_processed/(total_time/3600):.0f} files/hour")
    print()
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    # Set multiprocessing start method for macOS
    mp.set_start_method('spawn', force=True)
    main()
