#!/usr/bin/env python3
"""Backfill script to crawl only missing single-digit dates using the enhanced crawler."""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Set

# Add the crawler directory to path
sys.path.insert(0, '/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/crawlers')

from crawler import HansardCrawler

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s – %(levelname)s – %(message)s")
log = logging.getLogger("hansard_backfill")

class MissingDateBackfiller:
    """Handles backfilling of missing single-digit dates."""
    
    def __init__(self, missing_dates_file: Path, output_dir: Path, concurrency: int = 2):
        self.missing_dates_file = missing_dates_file
        self.output_dir = output_dir
        self.concurrency = concurrency
        self.processed_count = 0
        self.success_count = 0
        self.failed_dates = []
        
    def load_missing_dates(self) -> List[str]:
        """Load the list of missing dates from the analysis file."""
        try:
            with open(self.missing_dates_file, 'r') as f:
                data = json.load(f)
            
            missing_dates = data.get('missing_dates', [])
            log.info(f"Loaded {len(missing_dates)} missing dates from {self.missing_dates_file}")
            
            return missing_dates
        except Exception as e:
            log.error(f"Failed to load missing dates file: {e}")
            return []
    
    def group_dates_by_month(self, dates: List[str]) -> dict:
        """Group dates by year/month for more efficient processing."""
        grouped = {}
        for date in dates:
            parts = date.split('/')
            if len(parts) == 3:
                year, month, day = parts
                key = f"{year}/{month}"
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(day)
        
        # Sort days within each month
        for key in grouped:
            grouped[key].sort(key=int)
        
        return grouped
    
    async def backfill_missing_dates(self, missing_dates: List[str]) -> None:
        """Backfill the missing dates using the enhanced crawler."""
        
        if not missing_dates:
            log.info("No missing dates to backfill!")
            return
        
        # Group dates by month for better logging
        grouped_dates = self.group_dates_by_month(missing_dates)
        
        log.info(f"Starting backfill of {len(missing_dates)} missing dates")
        log.info(f"Grouped into {len(grouped_dates)} months")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        async with HansardCrawler(concurrency=self.concurrency) as crawler:
            
            # Process in batches to avoid overwhelming the API
            batch_size = 5
            total_batches = (len(missing_dates) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(missing_dates), batch_size):
                batch = missing_dates[batch_idx:batch_idx + batch_size]
                current_batch_num = (batch_idx // batch_size) + 1
                
                log.info(f"Processing batch {current_batch_num}/{total_batches}: {batch}")
                
                # Process batch in parallel
                tasks = [self.crawl_single_date(crawler, date) for date in batch]
                
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    for i, result in enumerate(results):
                        date = batch[i]
                        if isinstance(result, Exception):
                            log.error(f"Failed to process {date}: {result}")
                            self.failed_dates.append(date)
                        elif result:
                            log.info(f"✅ Successfully backfilled {date}")
                            self.success_count += 1
                        else:
                            log.warning(f"⚠️ No data found for {date}")
                            self.failed_dates.append(date)
                        
                        self.processed_count += 1
                
                except Exception as e:
                    log.error(f"Batch {current_batch_num} failed: {e}")
                    for date in batch:
                        self.failed_dates.append(date)
                        self.processed_count += 1
                
                # Progress update
                if current_batch_num % 5 == 0:
                    progress = (self.processed_count / len(missing_dates)) * 100
                    log.info(f"Progress: {self.processed_count}/{len(missing_dates)} ({progress:.1f}%) - "
                           f"Success: {self.success_count}, Failed: {len(self.failed_dates)}")
                
                # Small delay between batches
                if current_batch_num < total_batches:
                    await asyncio.sleep(2)
        
        # Final summary
        self.print_summary(missing_dates)
    
    async def crawl_single_date(self, crawler: HansardCrawler, date: str) -> bool:
        """Crawl a single missing date."""
        try:
            # Check if this date already exists (safety check)
            if self.date_already_exists(date):
                log.debug(f"Date {date} already exists, skipping")
                return True
            
            # Crawl the date
            result = await crawler.crawl_day(date)
            
            if result and result.get('debates'):
                # Save the result
                crawler._save(result, self.output_dir)
                log.debug(f"Saved {len(result['debates'])} debates for {date}")
                return True
            else:
                log.debug(f"No debates found for {date}")
                return False
                
        except Exception as e:
            log.error(f"Error crawling {date}: {e}")
            return False
    
    def date_already_exists(self, date: str) -> bool:
        """Check if a date already has data (to avoid duplicate work)."""
        try:
            year, month, day = date.split('/')
            expected_dir = self.output_dir / year / month
            summary_file = expected_dir / f"{day}_summary.json"
            return summary_file.exists()
        except:
            return False
    
    def print_summary(self, original_missing: List[str]) -> None:
        """Print a summary of the backfill operation."""
        
        print("\n" + "="*60)
        print("BACKFILL SUMMARY")
        print("="*60)
        
        print(f"\n📊 Results:")
        print(f"  • Total dates to backfill: {len(original_missing)}")
        print(f"  • Successfully processed: {self.success_count}")
        print(f"  • Failed to process: {len(self.failed_dates)}")
        print(f"  • Success rate: {(self.success_count/len(original_missing)*100):.1f}%")
        
        if self.failed_dates:
            print(f"\n❌ Failed dates:")
            for date in self.failed_dates[:10]:
                print(f"    - {date}")
            if len(self.failed_dates) > 10:
                print(f"    ... and {len(self.failed_dates) - 10} more")
            
            # Save failed dates for retry
            failed_file = Path("/tmp/hansard_backfill_failed.json")
            with open(failed_file, 'w') as f:
                json.dump({
                    'failed_dates': self.failed_dates,
                    'generated_at': datetime.now().isoformat(),
                    'original_count': len(original_missing),
                    'success_count': self.success_count
                }, f, indent=2)
            
            print(f"\n💾 Failed dates saved to: {failed_file}")
            print(f"   You can retry these dates later if needed.")

async def main():
    """Main backfill function."""
    
    # Configuration
    missing_dates_file = Path("/tmp/hansard_missing_dates.json")
    output_dir = Path("/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/data/hansard")
    concurrency = 2  # Conservative to avoid rate limiting
    
    # Check if analysis file exists
    if not missing_dates_file.exists():
        print(f"❌ Missing dates file not found: {missing_dates_file}")
        print("   Run 'python analyze_missing_dates.py' first to generate it.")
        return
    
    print("🚀 Starting Hansard missing dates backfill...")
    
    # Create backfiller
    backfiller = MissingDateBackfiller(missing_dates_file, output_dir, concurrency)
    
    # Load missing dates
    missing_dates = backfiller.load_missing_dates()
    
    if not missing_dates:
        print("✅ No missing dates found. Your data appears complete!")
        return
    
    print(f"📋 Found {len(missing_dates)} dates to backfill")
    
    # Confirm before proceeding
    response = input(f"\nProceed with backfilling {len(missing_dates)} missing dates? [y/N]: ")
    if response.lower() != 'y':
        print("❌ Backfill cancelled by user")
        return
    
    # Start backfill
    try:
        await backfiller.backfill_missing_dates(missing_dates)
        print(f"\n✅ Backfill completed!")
        
    except KeyboardInterrupt:
        print(f"\n⚡ Backfill interrupted by user")
        print(f"   Progress: {backfiller.processed_count}/{len(missing_dates)} dates processed")
        
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