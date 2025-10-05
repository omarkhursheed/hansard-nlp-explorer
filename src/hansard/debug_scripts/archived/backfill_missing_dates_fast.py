#!/usr/bin/env python3
"""Fast parallelized backfill script for missing single-digit dates."""

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

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s – %(levelname)s – %(message)s")
log = logging.getLogger("hansard_fast_backfill")

class FastBackfiller:
    """Optimized backfiller with better parallelization and batching."""
    
    def __init__(self, missing_dates_file: Path, output_dir: Path, 
                 concurrency: int = 8, batch_size: int = 20, max_rps: float = 6.0):
        self.missing_dates_file = missing_dates_file
        self.output_dir = output_dir
        self.concurrency = concurrency
        self.batch_size = batch_size
        self.max_rps = max_rps
        
        # Progress tracking
        self.total_dates = 0
        self.processed_count = 0
        self.success_count = 0
        self.skip_count = 0
        self.failed_dates = []
        self.start_time = None
        
        # Performance optimization: pre-check existing files
        self.existing_files_cache = set()
        self._build_existing_files_cache()
    
    def _build_existing_files_cache(self):
        """Pre-build cache of existing files to avoid filesystem checks during crawling."""
        log.info("Building cache of existing files...")
        start = time.time()
        
        for summary_file in self.output_dir.rglob("*_summary.json"):
            # Extract date from path like: /path/1928/may/7_summary.json -> 1928/may/7
            try:
                parts = summary_file.parts
                if len(parts) >= 3:
                    year = parts[-3]
                    month = parts[-2] 
                    day = summary_file.stem.split('_')[0]  # "7_summary" -> "7"
                    if year.isdigit() and day.isdigit():
                        self.existing_files_cache.add(f"{year}/{month}/{day}")
            except:
                continue
        
        elapsed = time.time() - start
        log.info(f"Built cache of {len(self.existing_files_cache)} existing files in {elapsed:.1f}s")
    
    def load_missing_dates(self) -> List[str]:
        """Load and filter missing dates, removing ones that already exist."""
        try:
            with open(self.missing_dates_file, 'r') as f:
                data = json.load(f)
            
            all_missing = data.get('missing_dates', [])
            
            # Filter out dates that already exist
            filtered_missing = [date for date in all_missing if date not in self.existing_files_cache]
            
            skipped = len(all_missing) - len(filtered_missing)
            if skipped > 0:
                log.info(f"Filtered out {skipped} dates that already exist")
            
            log.info(f"Loaded {len(filtered_missing)} missing dates to process")
            return filtered_missing
            
        except Exception as e:
            log.error(f"Failed to load missing dates: {e}")
            return []
    
    def chunk_dates_by_month(self, dates: List[str]) -> List[List[str]]:
        """Group dates by month and create optimized chunks for parallel processing."""
        
        # Group by month first
        by_month = defaultdict(list)
        for date in dates:
            year_month = '/'.join(date.split('/')[:2])
            by_month[year_month].append(date)
        
        # Create chunks that mix dates from different months for better load balancing
        chunks = []
        current_chunk = []
        
        # Round-robin through months to distribute load
        month_iterators = {month: iter(dates) for month, dates in by_month.items()}
        
        while month_iterators:
            # Add one date from each available month to current chunk
            months_to_remove = []
            for month in list(month_iterators.keys()):
                try:
                    date = next(month_iterators[month])
                    current_chunk.append(date)
                    
                    if len(current_chunk) >= self.batch_size:
                        chunks.append(current_chunk)
                        current_chunk = []
                        
                except StopIteration:
                    months_to_remove.append(month)
            
            # Remove exhausted months
            for month in months_to_remove:
                del month_iterators[month]
        
        # Add final partial chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        log.info(f"Created {len(chunks)} optimized chunks for processing")
        return chunks
    
    async def backfill_parallel(self, missing_dates: List[str]) -> None:
        """Main backfill function with optimized parallelization."""
        
        if not missing_dates:
            log.info("No missing dates to backfill!")
            return
        
        self.total_dates = len(missing_dates)
        self.start_time = time.time()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create optimized chunks
        chunks = self.chunk_dates_by_month(missing_dates)
        
        log.info(f"🚀 Starting fast backfill of {self.total_dates} dates")
        log.info(f"📊 Configuration: {self.concurrency} concurrent crawlers, {self.batch_size} batch size, {self.max_rps} RPS")
        log.info(f"📦 Processing {len(chunks)} chunks")
        
        # Process chunks with controlled concurrency
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async def process_chunk_with_semaphore(chunk: List[str]) -> None:
            async with semaphore:
                await self.process_chunk(chunk)
        
        # Start all chunk tasks
        chunk_tasks = [process_chunk_with_semaphore(chunk) for chunk in chunks]
        
        # Process with progress reporting
        await asyncio.gather(*chunk_tasks, return_exceptions=True)
        
        # Final summary
        self.print_final_summary()
    
    async def process_chunk(self, dates: List[str]) -> None:
        """Process a chunk of dates with a single crawler instance."""
        
        async with HansardCrawler(concurrency=1) as crawler:
            # Override the crawler's rate limit for faster processing
            if hasattr(crawler, 'limiter') and crawler.limiter:
                crawler.limiter.min_int = 1.0 / self.max_rps
            
            tasks = []
            for date in dates:
                task = asyncio.create_task(self.crawl_single_date_fast(crawler, date))
                tasks.append((date, task))
            
            # Process all dates in this chunk concurrently
            for date, task in tasks:
                try:
                    success = await task
                    if success:
                        self.success_count += 1
                    self.processed_count += 1
                    
                    # Progress update every 50 dates
                    if self.processed_count % 50 == 0:
                        self.print_progress()
                        
                except Exception as e:
                    log.error(f"Failed to process {date}: {e}")
                    self.failed_dates.append(date)
                    self.processed_count += 1
    
    async def crawl_single_date_fast(self, crawler: HansardCrawler, date: str) -> bool:
        """Optimized single date crawling with minimal overhead."""
        try:
            # Quick cache check (faster than filesystem)
            if date in self.existing_files_cache:
                self.skip_count += 1
                return True
            
            # Crawl the date
            result = await crawler.crawl_day(date)
            
            if result and result.get('debates'):
                # Save and update cache
                crawler._save(result, self.output_dir)
                self.existing_files_cache.add(date)
                return True
            else:
                # No data is normal for many dates (Parliament didn't sit)
                return True
                
        except Exception as e:
            log.debug(f"Error crawling {date}: {e}")
            return False
    
    def print_progress(self) -> None:
        """Print progress update with ETA."""
        if self.start_time and self.processed_count > 0:
            elapsed = time.time() - self.start_time
            rate = self.processed_count / elapsed
            remaining = self.total_dates - self.processed_count
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            
            progress_pct = (self.processed_count / self.total_dates) * 100
            
            log.info(f"Progress: {self.processed_count:,}/{self.total_dates:,} ({progress_pct:.1f}%) - "
                    f"Rate: {rate:.1f}/s - Success: {self.success_count:,} - "
                    f"ETA: {eta_minutes:.1f} min")
    
    def print_final_summary(self) -> None:
        """Print comprehensive final summary."""
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        rate = self.processed_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*70)
        print("FAST BACKFILL SUMMARY")
        print("="*70)
        
        print(f"\n⏱️  Performance:")
        print(f"   • Total time: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
        print(f"   • Processing rate: {rate:.1f} dates/second")
        print(f"   • Estimated time saved vs sequential: {((self.total_dates * 2) - elapsed)/60:.1f} minutes")
        
        print(f"\n📊 Results:")
        print(f"   • Total dates processed: {self.processed_count:,}")
        print(f"   • Successfully crawled: {self.success_count:,}")
        print(f"   • Skipped (already existed): {self.skip_count:,}")
        print(f"   • Failed: {len(self.failed_dates):,}")
        print(f"   • Success rate: {(self.success_count/max(1,self.processed_count)*100):.1f}%")
        
        if self.failed_dates:
            print(f"\n❌ Failed dates (first 10):")
            for date in self.failed_dates[:10]:
                print(f"     - {date}")
            
            if len(self.failed_dates) > 10:
                print(f"     ... and {len(self.failed_dates) - 10} more")
            
            # Save failed dates for retry
            failed_file = Path("/tmp/hansard_backfill_failed_fast.json")
            with open(failed_file, 'w') as f:
                json.dump({
                    'failed_dates': self.failed_dates,
                    'total_processed': self.processed_count,
                    'success_count': self.success_count,
                    'elapsed_seconds': elapsed,
                    'rate_per_second': rate,
                    'generated_at': datetime.now().isoformat()
                }, f, indent=2)
            
            print(f"\n💾 Failed dates saved to: {failed_file}")

async def main():
    """Main fast backfill function."""
    
    # Optimized configuration for faster processing
    missing_dates_file = Path("/tmp/hansard_missing_dates.json")
    output_dir = Path("/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/data/hansard")
    
    # Increased parallelization settings (but still respectful)
    concurrency = 8      # 8 concurrent crawler instances  
    batch_size = 20      # 20 dates per batch
    max_rps = 6.0        # 6 requests per second (doubled from original 3)
    
    if not missing_dates_file.exists():
        print(f"❌ Missing dates file not found: {missing_dates_file}")
        print("   Run 'python analyze_missing_dates.py' first.")
        return
    
    print("🚀 Starting FAST Hansard backfill...")
    print(f"⚡ Optimized for speed: {concurrency} crawlers, {max_rps} RPS, {batch_size} batch size")
    
    # Create fast backfiller
    backfiller = FastBackfiller(missing_dates_file, output_dir, concurrency, batch_size, max_rps)
    
    # Load missing dates
    missing_dates = backfiller.load_missing_dates()
    
    if not missing_dates:
        print("✅ No missing dates found. Your data appears complete!")
        return
    
    print(f"📋 Found {len(missing_dates):,} dates to backfill")
    
    # Estimate time with fast processing
    estimated_minutes = (len(missing_dates) / (concurrency * max_rps)) / 60
    print(f"🕐 Estimated completion time: {estimated_minutes:.1f} minutes")
    
    # Confirm before proceeding
    response = input(f"\nProceed with FAST backfill of {len(missing_dates):,} dates? [y/N]: ")
    if response.lower() != 'y':
        print("❌ Backfill cancelled")
        return
    
    try:
        await backfiller.backfill_parallel(missing_dates)
        print(f"\n✅ Fast backfill completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚡ Backfill interrupted by user")
        backfiller.print_final_summary()
        
    except Exception as e:
        print(f"\n❌ Backfill failed: {e}")
        log.error(f"Backfill error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚡ Program interrupted")
    except Exception as e:
        print(f"❌ Program failed: {e}")