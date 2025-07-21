#!/usr/bin/env python3
"""
Production script to process the complete Hansard dataset (1803-2005).

Optimized for tmux execution with progress monitoring, error recovery,
and efficient batch processing.

Usage:
    python process_full_dataset.py [--start-year YEAR] [--end-year YEAR] [--batch-size N]
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import signal

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from parsers.data_pipeline import HansardDataPipeline
from parsers.data_validation import HansardDataValidator

class ProductionProcessor:
    """Production-ready processor with monitoring and recovery."""
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.pipeline = HansardDataPipeline(raw_data_path, processed_data_path)
        self.start_time = datetime.now()
        self.processed_years = []
        self.failed_years = []
        self.interrupted = False
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interruption gracefully."""
        print(f"\n⚠️  Received signal {signum}. Shutting down gracefully...")
        self.interrupted = True
    
    def process_year_with_recovery(self, year: str, retry_count: int = 2) -> dict:
        """Process a year with error recovery."""
        for attempt in range(retry_count + 1):
            try:
                print(f"📅 Processing {year}... (attempt {attempt + 1})")
                result = self.pipeline.process_year(year)
                
                if result.get('errors', 0) > 0:
                    print(f"  ⚠️  {result['errors']} errors in {year}")
                else:
                    print(f"  ✅ {year}: {result['processed']}/{result['total_files']} files")
                
                return result
                
            except Exception as e:
                print(f"  ❌ Error processing {year} (attempt {attempt + 1}): {e}")
                if attempt == retry_count:
                    return {'year': year, 'error': str(e), 'processed': 0, 'total_files': 0}
                time.sleep(5)  # Brief pause before retry
    
    def process_batch(self, years: list, batch_name: str = "") -> dict:
        """Process a batch of years with progress tracking."""
        batch_start = datetime.now()
        batch_results = {
            'batch_name': batch_name,
            'start_time': batch_start.isoformat(),
            'years_processed': [],
            'total_processed': 0,
            'total_errors': 0,
            'total_files': 0
        }
        
        print(f"\n🚀 Starting batch: {batch_name} ({len(years)} years)")
        print(f"   Years: {years[0]} - {years[-1]}")
        
        for i, year in enumerate(years, 1):
            if self.interrupted:
                print(f"⚠️  Batch interrupted at year {year}")
                break
                
            print(f"\n[{i}/{len(years)}] ", end="")
            result = self.process_year_with_recovery(year)
            
            batch_results['years_processed'].append(result)
            if 'error' not in result:
                batch_results['total_processed'] += result.get('processed', 0)
                batch_results['total_errors'] += result.get('errors', 0)
                batch_results['total_files'] += result.get('total_files', 0)
                self.processed_years.append(year)
            else:
                self.failed_years.append(year)
            
            # Progress estimation
            elapsed = datetime.now() - batch_start
            if i > 0:
                avg_time_per_year = elapsed.total_seconds() / i
                remaining_years = len(years) - i
                eta = datetime.now() + timedelta(seconds=avg_time_per_year * remaining_years)
                print(f"     📊 Batch progress: {i}/{len(years)} | ETA: {eta.strftime('%H:%M:%S')}")
        
        batch_results['end_time'] = datetime.now().isoformat()
        batch_results['duration_minutes'] = (datetime.now() - batch_start).total_seconds() / 60
        
        return batch_results
    
    def save_checkpoint(self, results: dict):
        """Save processing checkpoint for recovery."""
        checkpoint_path = self.pipeline.processed_data_path / 'processing_checkpoint.json'
        
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'processed_years': self.processed_years,
            'failed_years': self.failed_years,
            'results': results,
            'interrupted': self.interrupted
        }
        
        import json
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def process_full_dataset(self, start_year: int = 1803, end_year: int = 2005, 
                           batch_size: int = 10) -> dict:
        """Process the complete dataset in batches."""
        
        # Find available years
        available_years = []
        for year_dir in self.pipeline.raw_data_path.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                year = int(year_dir.name)
                if start_year <= year <= end_year:
                    available_years.append(year)
        
        available_years.sort()
        total_years = len(available_years)
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                 🏛️  HANSARD DATASET PROCESSOR               ║
╠══════════════════════════════════════════════════════════════╣
║ Years to process: {total_years} ({start_year}-{end_year})                           ║
║ Batch size: {batch_size}                                            ║  
║ Estimated storage: ~14.5 GB                                 ║
║ Estimated time: 10-20 hours                                 ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Create year batches
        year_batches = []
        for i in range(0, len(available_years), batch_size):
            batch = available_years[i:i + batch_size]
            year_batches.append([str(year) for year in batch])
        
        print(f"📦 Created {len(year_batches)} batches of ~{batch_size} years each")
        
        all_results = {
            'processing_start': self.start_time.isoformat(),
            'batches': [],
            'summary': {
                'total_years_requested': total_years,
                'total_batches': len(year_batches),
                'batch_size': batch_size
            }
        }
        
        # Process batches
        for i, batch_years in enumerate(year_batches, 1):
            if self.interrupted:
                print(f"⚠️  Processing interrupted at batch {i}")
                break
                
            batch_name = f"Batch {i}/{len(year_batches)} ({batch_years[0]}-{batch_years[-1]})"
            batch_result = self.process_batch(batch_years, batch_name)
            all_results['batches'].append(batch_result)
            
            # Save checkpoint after each batch
            self.save_checkpoint(all_results)
            
            # Memory cleanup between batches
            import gc
            gc.collect()
            
            print(f"\n✅ {batch_name} completed:")
            print(f"   Files processed: {batch_result['total_processed']:,}")
            print(f"   Duration: {batch_result['duration_minutes']:.1f} minutes")
            
            # Overall progress
            overall_elapsed = datetime.now() - self.start_time
            batches_completed = i
            if batches_completed > 0:
                avg_batch_time = overall_elapsed.total_seconds() / batches_completed / 60
                remaining_batches = len(year_batches) - batches_completed
                overall_eta = datetime.now() + timedelta(minutes=avg_batch_time * remaining_batches)
                print(f"   📊 Overall: {batches_completed}/{len(year_batches)} batches | ETA: {overall_eta.strftime('%Y-%m-%d %H:%M')}")
        
        # Final consolidation and validation
        if not self.interrupted:
            print(f"\n🔄 Consolidating metadata across all years...")
            self.pipeline._consolidate_metadata()
            
            print(f"🔍 Running final validation...")
            validator = HansardDataValidator(str(self.pipeline.processed_data_path))
            validation_report = validator.run_full_validation()
            all_results['final_validation'] = validation_report
        
        all_results['processing_end'] = datetime.now().isoformat()
        all_results['total_duration_hours'] = (datetime.now() - self.start_time).total_seconds() / 3600
        
        # Final summary
        total_processed = sum(batch['total_processed'] for batch in all_results['batches'])
        total_files = sum(batch['total_files'] for batch in all_results['batches'])
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                     🎉 PROCESSING COMPLETE                   ║
╠══════════════════════════════════════════════════════════════╣
║ Years processed: {len(self.processed_years)}/{total_years}                                    ║
║ Files processed: {total_processed:,}/{total_files:,}                           ║
║ Duration: {all_results['total_duration_hours']:.1f} hours                                      ║
║ Failed years: {len(self.failed_years)}                                        ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        if self.failed_years:
            print(f"⚠️  Failed years: {self.failed_years}")
        
        return all_results

def main():
    """Main execution with argument parsing."""
    parser = argparse.ArgumentParser(description='Process complete Hansard dataset')
    parser.add_argument('--start-year', type=int, default=1803, 
                        help='Start year (default: 1803)')
    parser.add_argument('--end-year', type=int, default=2005,
                        help='End year (default: 2005)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Years per batch (default: 10)')
    parser.add_argument('--raw-data', type=str, default='../data/hansard',
                        help='Raw data path (default: ../data/hansard)')
    parser.add_argument('--output', type=str, default='../data/processed',
                        help='Output path (default: ../data/processed)')
    
    args = parser.parse_args()
    
    # Validate paths
    raw_path = Path(args.raw_data)
    if not raw_path.exists():
        print(f"❌ Raw data path not found: {raw_path}")
        sys.exit(1)
    
    print(f"Starting Hansard dataset processing...")
    print(f"Raw data: {raw_path}")
    print(f"Output: {args.output}")
    print(f"Year range: {args.start_year}-{args.end_year}")
    print(f"Batch size: {args.batch_size}")
    
    # Initialize processor
    processor = ProductionProcessor(args.raw_data, args.output)
    
    try:
        # Run processing
        results = processor.process_full_dataset(
            start_year=args.start_year,
            end_year=args.end_year,
            batch_size=args.batch_size
        )
        
        # Save final results
        import json
        results_path = Path(args.output) / 'full_processing_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Final results saved: {results_path}")
        
        return 0 if not processor.failed_years else 1
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())