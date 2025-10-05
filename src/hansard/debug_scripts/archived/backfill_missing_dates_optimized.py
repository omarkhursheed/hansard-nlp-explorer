#!/usr/bin/env python3
"""Heavily optimized backfill script with smart 404 handling."""

import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Tuple

# Add the crawler directory to path
sys.path.insert(0, '/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/crawlers')

from crawler import HansardCrawler

# Set up logging with less verbose output
logging.basicConfig(level=logging.WARNING, format="%(asctime)s – %(levelname)s – %(message)s")
log = logging.getLogger("hansard_optimized_backfill")
log.setLevel(logging.INFO)

class OptimizedBackfiller:
    """Heavily optimized backfiller that minimizes 404 overhead."""
    
    def __init__(self, missing_dates_file: Path, output_dir: Path, 
                 concurrency: int = 12, batch_size: int = 30):
        self.missing_dates_file = missing_dates_file
        self.output_dir = output_dir
        self.concurrency = concurrency
        self.batch_size = batch_size
        
        # Progress tracking
        self.total_dates = 0
        self.processed_count = 0
        self.success_count = 0
        self.failed_dates = []
        self.start_time = None
        
        # Cache existing files
        self.existing_files_cache = set()
        self._build_existing_files_cache()
    
    def _build_existing_files_cache(self):
        """Pre-build cache of existing files."""
        log.info("Building cache of existing files...")
        start = time.time()
        
        for summary_file in self.output_dir.rglob("*_summary.json"):
            try:
                parts = summary_file.parts
                if len(parts) >= 3:
                    year = parts[-3]
                    month = parts[-2] 
                    day = summary_file.stem.split('_')[0]
                    if year.isdigit() and day.isdigit():
                        self.existing_files_cache.add(f"{year}/{month}/{day}")
            except:
                continue
        
        elapsed = time.time() - start
        log.info(f"Built cache of {len(self.existing_files_cache)} existing files in {elapsed:.1f}s")
    
    def load_missing_dates(self) -> List[str]:
        """Load and filter missing dates."""
        try:
            with open(self.missing_dates_file, 'r') as f:
                data = json.load(f)
            
            all_missing = data.get('missing_dates', [])
            filtered_missing = [date for date in all_missing if date not in self.existing_files_cache]
            
            # Sort by year to process older dates first (they're typically faster)
            filtered_missing.sort(key=lambda x: (int(x.split('/')[0]), x))
            
            skipped = len(all_missing) - len(filtered_missing)
            if skipped > 0:
                log.info(f"Filtered out {skipped} dates that already exist")
            
            log.info(f"Loaded {len(filtered_missing)} missing dates to process")
            return filtered_missing
            
        except Exception as e:
            log.error(f"Failed to load missing dates: {e}")
            return []
    
    async def backfill_optimized(self, missing_dates: List[str]) -> None:
        """Optimized backfill with better parallelization."""
        
        if not missing_dates:
            log.info("No missing dates to backfill!")
            return
        
        self.total_dates = len(missing_dates)
        self.start_time = time.time()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"🚀 Starting optimized backfill of {self.total_dates} dates")
        log.info(f"📊 Configuration: {self.concurrency} concurrent processors")
        
        # Process in batches with high concurrency
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async def process_date_with_semaphore(date: str) -> None:
            async with semaphore:
                await self.process_single_date_optimized(date)
        
        # Create all tasks
        tasks = [process_date_with_semaphore(date) for date in missing_dates]
        
        # Process with progress reporting
        done_tasks = []
        
        # Process in chunks to provide progress updates
        chunk_size = 100
        for i in range(0, len(tasks), chunk_size):
            chunk_tasks = tasks[i:i + chunk_size]
            
            # Wait for this chunk to complete
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            done_tasks.extend(chunk_results)
            
            # Update progress
            self.processed_count = len(done_tasks)
            if self.processed_count % 100 == 0 or self.processed_count == len(tasks):
                self.print_progress()
        
        # Final summary
        self.print_final_summary()
    
    async def process_single_date_optimized(self, date: str) -> None:
        """Process a single date with optimizations."""
        try:
            # Quick cache check
            if date in self.existing_files_cache:
                return
            
            # Use a single crawler instance per task (better than sharing)
            async with HansardCrawler(concurrency=1) as crawler:
                # Reduce HTTP logging noise
                import httpx
                httpx_logger = logging.getLogger("httpx")
                httpx_logger.setLevel(logging.ERROR)
                
                # Set faster timeouts
                if hasattr(crawler, 'http') and crawler.http:
                    crawler.http.timeout = httpx.Timeout(15.0)  # Faster timeout
                
                # Crawl the date
                result = await crawler.crawl_day(date)
                
                if result and result.get('debates'):
                    # Save and update cache
                    crawler._save(result, self.output_dir)
                    self.existing_files_cache.add(date)
                    self.success_count += 1
                else:
                    # No data is normal for many dates
                    self.success_count += 1
                    
        except Exception as e:
            log.debug(f"Error processing {date}: {e}")
            self.failed_dates.append(date)
    
    def print_progress(self) -> None:
        """Print concise progress update."""
        if self.start_time and self.processed_count > 0:
            elapsed = time.time() - self.start_time
            rate = self.processed_count / elapsed
            remaining = self.total_dates - self.processed_count
            eta_minutes = (remaining / rate / 60) if rate > 0 else 0
            
            progress_pct = (self.processed_count / self.total_dates) * 100
            
            print(f"Progress: {self.processed_count:,}/{self.total_dates:,} ({progress_pct:.1f}%) - "
                  f"Rate: {rate:.1f}/s - Success: {self.success_count:,} - ETA: {eta_minutes:.1f}min")
    
    def print_final_summary(self) -> None:
        """Print final summary."""
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        rate = self.processed_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*60)
        print("OPTIMIZED BACKFILL SUMMARY")
        print("="*60)
        
        print(f"\n⏱️  Performance:")
        print(f"   • Total time: {elapsed/60:.1f} minutes")
        print(f"   • Processing rate: {rate:.1f} dates/second")
        
        print(f"\n📊 Results:")
        print(f"   • Total dates processed: {self.processed_count:,}")
        print(f"   • Successfully processed: {self.success_count:,}")
        print(f"   • Failed: {len(self.failed_dates):,}")
        print(f"   • Success rate: {(self.success_count/max(1,self.processed_count)*100):.1f}%")
        
        if self.failed_dates:
            # Save failed dates
            failed_file = Path("/tmp/hansard_backfill_failed_optimized.json")
            with open(failed_file, 'w') as f:
                json.dump({
                    'failed_dates': self.failed_dates,
                    'success_count': self.success_count,
                    'total_processed': self.processed_count,
                    'rate_per_second': rate,
                    'generated_at': datetime.now().isoformat()
                }, f, indent=2)
            
            print(f"\n💾 Failed dates saved to: {failed_file}")

async def main():
    """Main optimized backfill function."""
    
    missing_dates_file = Path("/tmp/hansard_missing_dates.json")
    output_dir = Path("/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/data/hansard")
    
    # More aggressive optimization
    concurrency = 12    # Higher concurrency
    
    if not missing_dates_file.exists():
        print(f"❌ Missing dates file not found: {missing_dates_file}")
        return
    
    print("🚀 Starting OPTIMIZED Hansard backfill...")
    print(f"⚡ High performance: {concurrency} concurrent processors")
    
    backfiller = OptimizedBackfiller(missing_dates_file, output_dir, concurrency)
    missing_dates = backfiller.load_missing_dates()
    
    if not missing_dates:
        print("✅ No missing dates found!")
        return
    
    print(f"📋 Found {len(missing_dates):,} dates to backfill")
    
    # Better time estimate based on older dates being faster
    estimated_hours = len(missing_dates) / (concurrency * 2.0) / 3600  # Assume 2 dates/sec per worker
    print(f"🕐 Estimated completion time: {estimated_hours:.1f} hours")
    
    response = input(f"\nProceed with optimized backfill? [y/N]: ")
    if response.lower() != 'y':
        print("❌ Cancelled")
        return
    
    try:
        await backfiller.backfill_optimized(missing_dates)
        print(f"\n✅ Optimized backfill completed!")
        
    except KeyboardInterrupt:
        print(f"\n⚡ Interrupted by user")
        backfiller.print_final_summary()

if __name__ == "__main__":
    asyncio.run(main())