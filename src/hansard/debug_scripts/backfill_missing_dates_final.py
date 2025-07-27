#!/usr/bin/env python3
"""Final optimized backfill - tuned based on test results."""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

# Add the crawler directory to path
sys.path.insert(0, '/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/crawlers')

from crawler import HansardCrawler

# Minimal logging for performance
logging.basicConfig(level=logging.ERROR)
log = logging.getLogger("hansard_final")
log.setLevel(logging.INFO)

class FinalOptimizedBackfiller:
    """Final optimized version based on test results."""
    
    def __init__(self, missing_dates_file: Path, output_dir: Path):
        self.missing_dates_file = missing_dates_file
        self.output_dir = output_dir
        
        # Aggressive settings based on test results
        self.concurrency = 20        # Higher concurrency
        self.max_rps = 15.0         # Higher RPS (test showed 11 RPS was safe)
        
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
        """Build cache of existing files."""
        print("📂 Building cache of existing files...")
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
        print(f"✅ Cached {len(self.existing_files_cache)} existing files in {elapsed:.1f}s")
    
    def load_missing_dates(self) -> List[str]:
        """Load and filter missing dates."""
        try:
            with open(self.missing_dates_file, 'r') as f:
                data = json.load(f)
            
            all_missing = data.get('missing_dates', [])
            filtered_missing = [date for date in all_missing if date not in self.existing_files_cache]
            
            # Sort by year (older dates first - they're typically faster)
            filtered_missing.sort(key=lambda x: int(x.split('/')[0]))
            
            skipped = len(all_missing) - len(filtered_missing)
            if skipped > 0:
                print(f"⏭️  Skipped {skipped} dates that already exist")
            
            print(f"📋 Loaded {len(filtered_missing)} missing dates to process")
            return filtered_missing
            
        except Exception as e:
            print(f"❌ Failed to load missing dates: {e}")
            return []
    
    async def backfill_final(self, missing_dates: List[str]) -> None:
        """Final optimized backfill."""
        
        if not missing_dates:
            print("✅ No missing dates to backfill!")
            return
        
        self.total_dates = len(missing_dates)
        self.start_time = time.time()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Starting FINAL optimized backfill")
        print(f"⚡ Aggressive settings: {self.concurrency} workers, {self.max_rps} RPS")
        print(f"📊 Processing {self.total_dates:,} dates")
        
        # Process with maximum allowed concurrency
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async def process_with_semaphore(date: str) -> bool:
            async with semaphore:
                return await self.process_single_date(date)
        
        # Create all tasks
        tasks = [process_with_semaphore(date) for date in missing_dates]
        
        # Process in small chunks with aggressive timeout handling
        chunk_size = 100  # Smaller chunks for better timeout recovery
        chunk_timeout = 300  # 5 minutes per chunk (more aggressive)
        
        for i in range(0, len(tasks), chunk_size):
            chunk_tasks = tasks[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (len(tasks) + chunk_size - 1) // chunk_size
            
            print(f"🔄 Processing chunk {chunk_num}/{total_chunks} ({len(chunk_tasks)} dates)")
            
            try:
                # Process chunk with strict timeout
                results = await asyncio.wait_for(
                    asyncio.gather(*chunk_tasks, return_exceptions=True),
                    timeout=chunk_timeout
                )
                
                # Update counters
                chunk_success = 0
                for j, result in enumerate(results):
                    if isinstance(result, Exception):
                        self.failed_dates.append(missing_dates[i + j])
                        print(f"  ❌ Failed: {missing_dates[i + j]} - {str(result)[:50]}")
                    elif result:
                        self.success_count += 1
                        chunk_success += 1
                    
                    self.processed_count += 1
                
                print(f"  ✅ Chunk {chunk_num} complete: {chunk_success}/{len(chunk_tasks)} successful")
                
            except asyncio.TimeoutError:
                print(f"  ⏰ Chunk {chunk_num} TIMED OUT after {chunk_timeout}s - marking as failed")
                # Mark all dates in this chunk as failed for potential retry
                chunk_dates = missing_dates[i:i + len(chunk_tasks)]
                self.failed_dates.extend(chunk_dates)
                self.processed_count += len(chunk_tasks)
                
                # Continue to next chunk immediately
                print(f"  ➡️  Continuing to next chunk...")
            
            except Exception as e:
                print(f"  ❌ Chunk {chunk_num} ERROR: {e}")
                chunk_dates = missing_dates[i:i + len(chunk_tasks)]
                self.failed_dates.extend(chunk_dates)
                self.processed_count += len(chunk_tasks)
            
            # Progress update after each chunk
            self.print_progress()
            
            # Brief pause between chunks to prevent overwhelming
            if chunk_num < total_chunks:
                await asyncio.sleep(1)
        
        self.print_final_summary()
    
    async def process_single_date(self, date: str) -> bool:
        """Process single date with timeout protection."""
        try:
            if date in self.existing_files_cache:
                return True
            
            # Wrap entire date processing in timeout
            return await asyncio.wait_for(
                self._process_date_internal(date),
                timeout=45.0  # 45 second timeout per date (aggressive)
            )
                
        except asyncio.TimeoutError:
            # Date took too long - skip it
            return False
        except Exception:
            return False
    
    async def _process_date_internal(self, date: str) -> bool:
        """Internal date processing with optimized crawler."""
        # Create crawler with optimized settings
        async with HansardCrawler(concurrency=1) as crawler:
            # Optimize crawler settings for speed
            if hasattr(crawler, 'limiter') and crawler.limiter:
                crawler.limiter.min_int = 1.0 / self.max_rps
            
            if hasattr(crawler, 'http') and crawler.http:
                # Aggressive timeouts
                import httpx
                crawler.http.timeout = httpx.Timeout(8.0)  # Very fast timeout
            
            # Crawl with minimal overhead
            result = await crawler.crawl_day(date)
            
            if result:
                crawler._save(result, self.output_dir)
                self.existing_files_cache.add(date)
                return True
            
            return True  # No data is normal
    
    def print_progress(self):
        """Print progress with better ETA."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            rate = self.processed_count / elapsed if elapsed > 0 else 0
            remaining = self.total_dates - self.processed_count
            eta_hours = (remaining / rate / 3600) if rate > 0 else 0
            
            progress_pct = (self.processed_count / self.total_dates) * 100
            
            print(f"📊 {self.processed_count:,}/{self.total_dates:,} ({progress_pct:.1f}%) - "
                  f"Rate: {rate:.1f}/s - Success: {self.success_count:,} - "
                  f"ETA: {eta_hours:.1f}h")
    
    def print_final_summary(self):
        """Print final summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        rate = self.processed_count / elapsed if elapsed > 0 else 0
        
        print(f"\n" + "="*60)
        print("FINAL BACKFILL COMPLETE")
        print("="*60)
        
        print(f"\n⏱️  Performance:")
        print(f"   • Total time: {elapsed/3600:.1f} hours ({elapsed/60:.1f} minutes)")
        print(f"   • Processing rate: {rate:.1f} dates/second")
        print(f"   • API usage: ~{rate * 50:.0f} requests/second")
        
        print(f"\n📊 Results:")
        print(f"   • Total processed: {self.processed_count:,}")
        print(f"   • Successful: {self.success_count:,}")
        print(f"   • Failed: {len(self.failed_dates):,}")
        print(f"   • Success rate: {(self.success_count/max(1,self.processed_count)*100):.1f}%")
        
        if self.failed_dates:
            failed_file = Path("/tmp/hansard_backfill_failed_final.json")
            with open(failed_file, 'w') as f:
                json.dump({
                    'failed_dates': self.failed_dates,
                    'total_processed': self.processed_count,
                    'success_count': self.success_count,
                    'processing_rate': rate,
                    'generated_at': datetime.now().isoformat()
                }, f, indent=2)
            
            print(f"\n💾 {len(self.failed_dates)} failed dates saved to: {failed_file}")

async def main():
    """Main function with user-friendly interface."""
    
    missing_dates_file = Path("/tmp/hansard_missing_dates.json")
    output_dir = Path("/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/data/hansard")
    
    if not missing_dates_file.exists():
        print(f"❌ Missing dates file not found: {missing_dates_file}")
        print("   Run: python analyze_missing_dates.py")
        return
    
    print("🚀 FINAL OPTIMIZED Hansard Backfill")
    print("🎯 Based on test results: 20 workers, 15 RPS, optimized timeouts")
    
    backfiller = FinalOptimizedBackfiller(missing_dates_file, output_dir)
    missing_dates = backfiller.load_missing_dates()
    
    if not missing_dates:
        print("✅ All dates already exist!")
        return
    
    # Better time estimate based on test results
    estimated_hours = len(missing_dates) * 5 / 3600  # 5 seconds per date (conservative)
    print(f"🕐 Conservative estimate: {estimated_hours:.1f} hours")
    print(f"🎯 Optimistic estimate: {estimated_hours/2:.1f} hours (if conditions are good)")
    
    print(f"\n📋 Will process {len(missing_dates):,} missing dates")
    print(f"⚡ High-performance settings (still respectful to API)")
    
    response = input(f"\nStart final backfill? [y/N]: ")
    if response.lower() != 'y':
        print("❌ Cancelled")
        return
    
    try:
        await backfiller.backfill_final(missing_dates)
        
    except KeyboardInterrupt:
        print(f"\n⚡ Interrupted - progress saved")
        backfiller.print_final_summary()

if __name__ == "__main__":
    asyncio.run(main())