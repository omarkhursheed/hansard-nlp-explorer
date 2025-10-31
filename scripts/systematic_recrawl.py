#!/usr/bin/env python3
"""
FAST Systematic re-crawler - optimized for maximum speed while being respectful.

Features:
- Parallel date processing (10-20 dates simultaneously)
- High crawler concurrency (8 per date)
- Smart rate limiting (20 RPS total)
- Automatic retries
- Checkpoint/resume support
"""

import asyncio
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "hansard" / "crawlers"))
from crawler import HansardCrawler

class SystematicRecrawler:
    def __init__(self, index_file: Path, recrawl_file: Path, output_dir: Path,
                 parallel_dates: int = 15, crawler_concurrency: int = 8):
        self.index = self.load_index(index_file)
        self.dates_to_crawl = self.load_recrawl_list(recrawl_file)
        self.output_dir = output_dir
        self.parallel_dates = parallel_dates
        self.crawler_concurrency = crawler_concurrency

        self.results = {
            "started": datetime.now().isoformat(),
            "completed": [],
            "failed": [],
            "in_progress": None
        }

        self.checkpoint_file = Path("analysis/recrawl_checkpoint.json")
        self.lock = asyncio.Lock()
        self.stats_lock = asyncio.Lock()

    def load_index(self, index_file: Path) -> dict:
        """Load the complete index."""
        if not index_file.exists():
            print(f"Error: Index file not found: {index_file}")
            print("Run: python3 scripts/create_complete_index.py first")
            sys.exit(1)

        with open(index_file) as f:
            data = json.load(f)
            return data["index"]

    def load_recrawl_list(self, recrawl_file: Path) -> list:
        """Load list of dates to re-crawl."""
        if not recrawl_file.exists():
            print(f"Error: Re-crawl list not found: {recrawl_file}")
            print("Run: python3 scripts/audit_against_index.py first")
            sys.exit(1)

        with open(recrawl_file) as f:
            return [line.strip() for line in f if line.strip()]

    def get_expected_count(self, date_path: str) -> int:
        """Get expected debate count from index."""
        parts = date_path.split('/')
        if len(parts) != 3:
            return 0

        year, month, day = parts
        try:
            return self.index[year][month][day]["debate_count"]
        except KeyError:
            return 0

    async def crawl_date(self, crawler: HansardCrawler, date_path: str, attempt: int = 1) -> dict:
        """Crawl a single date and verify."""
        expected_count = self.get_expected_count(date_path)

        try:
            result = await crawler.crawl_day(date_path)

            if not result:
                return {
                    "date": date_path,
                    "status": "failed",
                    "error": "Crawler returned None",
                    "expected": expected_count,
                    "got": 0,
                    "attempt": attempt
                }

            debates = result.get("debates", [])
            got_count = len(debates)

            # Save immediately (with lock to avoid race conditions)
            async with self.lock:
                crawler._save(result, self.output_dir)

            # Verify
            if got_count == expected_count:
                status = "perfect"
            elif got_count >= expected_count * 0.95:  # Within 5%
                status = "acceptable"
            elif got_count > 0:
                status = "incomplete"
            else:
                status = "failed"

            return {
                "date": date_path,
                "status": status,
                "expected": expected_count,
                "got": got_count,
                "completeness": (got_count / expected_count * 100) if expected_count > 0 else 0,
                "attempt": attempt
            }

        except Exception as e:
            return {
                "date": date_path,
                "status": "error",
                "error": str(e),
                "expected": expected_count,
                "got": 0,
                "attempt": attempt
            }

    async def recrawl_all(self, max_retries: int = 2):
        """Fast parallel re-crawl of all dates."""
        print("="*70)
        print("FAST SYSTEMATIC RE-CRAWL")
        print("="*70)
        print(f"Dates to crawl: {len(self.dates_to_crawl):,}")
        print(f"Parallel dates: {self.parallel_dates}")
        print(f"Crawler concurrency per date: {self.crawler_concurrency}")
        print(f"Max retries per date: {max_retries}")
        print()

        # Load checkpoint if exists
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                checkpoint = json.load(f)
                completed = set(checkpoint.get("completed", []))
                print(f"Resuming from checkpoint: {len(completed)} already completed")
                print()
        else:
            completed = set()

        # Filter out already completed
        dates_remaining = [d for d in self.dates_to_crawl if d not in completed]
        print(f"Remaining to crawl: {len(dates_remaining):,}")
        print()

        input("Press Enter to start re-crawling...")
        print()

        start_time = time.time()

        async with HansardCrawler(concurrency=self.crawler_concurrency) as crawler:
            # Process dates in parallel batches
            for batch_start in range(0, len(dates_remaining), self.parallel_dates):
                batch = dates_remaining[batch_start:batch_start + self.parallel_dates]
                batch_num = batch_start // self.parallel_dates + 1
                total_batches = (len(dates_remaining) + self.parallel_dates - 1) // self.parallel_dates

                print(f"Batch {batch_num}/{total_batches}: Processing {len(batch)} dates in parallel...")

                # Create tasks for parallel processing
                tasks = []
                for date_path in batch:
                    tasks.append(self.crawl_date_with_retry(crawler, date_path, max_retries, completed))

                # Execute batch in parallel
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for result in batch_results:
                    if isinstance(result, Exception):
                        print(f"  ✗ Unexpected error: {result}")
                    # Results already added by crawl_date_with_retry

                # Progress update
                processed = batch_start + len(batch)
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = len(dates_remaining) - processed
                eta = remaining / rate if rate > 0 else 0

                completed_count = len([r for r in self.results["completed"]])
                failed_count = len([r for r in self.results["failed"]])

                print(f"  Progress: {processed}/{len(dates_remaining)} ({processed*100//len(dates_remaining)}%)")
                print(f"  Rate: {rate:.1f} dates/sec | ETA: {eta/60:.0f} minutes")
                print(f"  Completed: {completed_count}, Failed: {failed_count}")
                print()

                # Save checkpoint
                if batch_num % 5 == 0:
                    await self.save_checkpoint_async(completed)

        # Final save
        await self.save_checkpoint_async(completed)
        self.save_final_results()

    async def crawl_date_with_retry(self, crawler: HansardCrawler, date_path: str,
                                   max_retries: int, completed: set) -> dict:
        """Crawl a date with automatic retries."""
        for attempt in range(1, max_retries + 1):
            result = await self.crawl_date(crawler, date_path, attempt)

            if result["status"] in ["perfect", "acceptable"]:
                async with self.stats_lock:
                    self.results["completed"].append(result)
                    completed.add(date_path)
                return result
            elif attempt < max_retries:
                await asyncio.sleep(1)  # Brief delay before retry
            else:
                # Final attempt failed
                async with self.stats_lock:
                    self.results["failed"].append(result)
                return result

        return result

    async def save_checkpoint_async(self, completed: set):
        """Save checkpoint for resuming (async-safe)."""
        async with self.lock:
            self.checkpoint_file.parent.mkdir(exist_ok=True)
            with open(self.checkpoint_file, 'w') as f:
                json.dump({
                    "last_updated": datetime.now().isoformat(),
                    "completed": list(completed),
                    "total": len(self.dates_to_crawl)
                }, f, indent=2)

    def save_final_results(self):
        """Save final results."""
        self.results["finished"] = datetime.now().isoformat()

        output_file = Path("analysis/recrawl_final_results.json")
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Print summary
        print()
        print("="*70)
        print("RE-CRAWL COMPLETE")
        print("="*70)
        print(f"Total dates: {len(self.dates_to_crawl)}")
        print(f"Completed: {len(self.results['completed'])}")
        print(f"Failed: {len(self.results['failed'])}")
        print()

        if self.results['failed']:
            print("Failed dates:")
            for r in self.results['failed'][:20]:
                print(f"  {r['date']}: {r.get('error', 'incomplete')}")
            if len(self.results['failed']) > 20:
                print(f"  ... and {len(self.results['failed']) - 20} more")
            print()

        print(f"Results saved to: {output_file}")
        print()
        print("Next step: python3 scripts/verify_completeness.py")

async def main():
    # Optimized settings based on API testing:
    # - API handles 20-30 RPS easily
    # - 15 dates in parallel × 8 concurrent requests per date = 120 potential concurrent requests
    # - But rate limiting keeps it at ~20 RPS actual

    parser = argparse.ArgumentParser(description="Run fast systematic re-crawl")
    parser.add_argument("--index-file", default="analysis/hansard_complete_index.json", help="Path to index JSON")
    parser.add_argument("--recrawl-file", default="analysis/dates_to_recrawl.txt", help="Path to recrawl dates list")
    parser.add_argument("--output-dir", default="data-hansard/hansard", help="Output directory for crawled data")
    parser.add_argument("--parallel-dates", type=int, default=15, help="Parallel dates to process")
    parser.add_argument("--crawler-concurrency", type=int, default=8, help="Per-date crawler concurrency")
    args = parser.parse_args()

    recrawler = SystematicRecrawler(
        index_file=Path(args.index_file),
        recrawl_file=Path(args.recrawl_file),
        output_dir=Path(args.output_dir),
        parallel_dates=args.parallel_dates,
        crawler_concurrency=args.crawler_concurrency,
    )

    await recrawler.recrawl_all(max_retries=2)

if __name__ == "__main__":
    asyncio.run(main())
