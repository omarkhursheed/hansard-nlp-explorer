#!/usr/bin/env python3
"""
High-performance Hansard processor optimized for M3 Max with 64GB RAM.
Uses multiprocessing and memory-optimized processing.
"""

import os
import sys
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from parsers.data_pipeline import HansardDataPipeline

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
            print(f"Error processing {file_path}: {e}")
    
    return results

def process_year_parallel(year, raw_data_path, processed_data_path, num_workers=None):
    """Process a year using multiprocessing."""
    if num_workers is None:
        # Use 75% of available cores on M3 Max
        num_workers = max(1, int(mp.cpu_count() * 0.75))
    
    year_path = Path(raw_data_path) / str(year)
    if not year_path.exists():
        return {'error': f'Year {year} not found', 'processed': 0}
    
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
                print(f"Error processing chunk: {e}")
    
    # Now save the results using the single-threaded pipeline methods
    if all_results:
        pipeline = HansardDataPipeline(raw_data_path, processed_data_path)
        
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
        
        # Save using pipeline methods
        try:
            # Convert to Polars for efficient processing
            import polars as pl
            
            debates_df = pl.DataFrame(debates_data)
            debates_df.write_parquet(Path(processed_data_path) / 'metadata' / f'debates_{year}.parquet')
            
            if speakers_data:
                speakers_df = pl.DataFrame(speakers_data)
                speakers_df.write_parquet(Path(processed_data_path) / 'metadata' / f'speakers_{year}.parquet')
            
            # Save content as JSON Lines
            content_dir = Path(processed_data_path) / 'content' / str(year)
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
    parser = argparse.ArgumentParser(description='High-Performance Hansard Processing')
    parser.add_argument('--start-year', type=int, default=1803, help='Start year')
    parser.add_argument('--end-year', type=int, default=2005, help='End year')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: 75% of CPU cores)')
    parser.add_argument('--raw-data', type=str, default='../data/hansard', help='Raw data path')
    parser.add_argument('--output', type=str, default='../data/processed_fixed', help='Output path')
    
    args = parser.parse_args()
    
    if args.workers is None:
        args.workers = max(1, int(mp.cpu_count() * 0.75))
    
    print(f"🚀 High-Performance Hansard Processing")
    print(f"   CPU cores available: {mp.cpu_count()}")
    print(f"   Using {args.workers} worker processes")
    print(f"   Processing years {args.start_year}-{args.end_year}")
    print(f"   Output: {args.output}")
    
    # Setup directories
    pipeline = HansardDataPipeline(args.raw_data, args.output)
    
    # Find available years
    available_years = []
    for year_dir in Path(args.raw_data).iterdir():
        if year_dir.is_dir() and year_dir.name.isdigit():
            year = int(year_dir.name)
            if args.start_year <= year <= args.end_year:
                available_years.append(year)
    
    available_years.sort()
    print(f"   Found {len(available_years)} years to process")
    
    start_time = datetime.now()
    total_processed = 0
    total_errors = 0
    
    # Process each year
    for i, year in enumerate(available_years, 1):
        year_start = datetime.now()
        print(f"\n[{i}/{len(available_years)}] Processing {year}...", end="", flush=True)
        
        result = process_year_parallel(year, args.raw_data, args.output, args.workers)
        
        year_time = (datetime.now() - year_start).total_seconds()
        
        if 'error' not in result:
            total_processed += result['processed']
            total_errors += result['errors']
            print(f" ✅ {result['processed']:,} files ({year_time:.1f}s)")
        else:
            print(f" ❌ {result['error']}")
    
    # Final consolidation
    print(f"\n🔄 Consolidating metadata...")
    pipeline._consolidate_metadata()
    
    total_time = datetime.now() - start_time
    print(f"\n🎉 Processing Complete!")
    print(f"   Total files processed: {total_processed:,}")
    print(f"   Total errors: {total_errors:,}")
    print(f"   Total time: {total_time.total_seconds()/3600:.1f} hours")
    print(f"   Average: {total_processed/(total_time.total_seconds()/3600):.0f} files/hour")

if __name__ == "__main__":
    # Set multiprocessing start method for macOS
    mp.set_start_method('spawn', force=True)
    main()