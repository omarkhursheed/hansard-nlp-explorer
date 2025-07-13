#!/usr/bin/env python3
"""
Parallel Hansard Crawler - Maximum speed with multiple concurrent processes.
"""

import subprocess
import asyncio
import concurrent.futures
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("parallel_hansard.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("parallel_crawler")

class ParallelHansardCrawler:
    def __init__(self, output_dir: Path = Path("data/hansard"), max_workers: int = 2):  # Back to 2 for politeness
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.status_file = output_dir / "parallel_status.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def estimate_time(self, years: int) -> str:
        """Estimate time with parallel processing based on real test results."""
        # Based on actual test: 339s for 112 sitting days = ~3s per sitting day
        # Conservative estimate: ~120 sitting days per year average (includes Lords)
        total_sitting_days = years * 120
        total_seconds = total_sitting_days * 3  # 3 seconds per sitting day
        
        # Divide by parallel workers (with 30% overhead for politeness)
        parallel_seconds = total_seconds / self.max_workers * 1.3  # 30% overhead
        
        hours = parallel_seconds / 3600
        if hours < 1:
            return f"~{parallel_seconds/60:.0f}min"
        else:
            return f"~{hours:.1f}hrs"
    
    def run_single_crawler(self, args: Dict) -> Dict:
        """Run a single crawler instance."""
        start_year = args['start']
        end_year = args['end']
        house = args.get('house')
        worker_id = args['worker_id']
        
        # Stagger start times to be polite to the API
        if worker_id > 1:
            delay = (worker_id - 1) * 15  # 15 second delay per worker
            log.info(f"⏸️ Worker {worker_id}: Waiting {delay}s to stagger start...")
            time.sleep(delay)
        
        range_str = f"{start_year}-{end_year}" if start_year != end_year else str(start_year)
        log.info(f"🔄 Worker {worker_id}: Starting {range_str}")
        
        # Build command
        cmd = [
            "python", "crawler.py", 
            str(start_year),
            str(end_year) if start_year != end_year else None,
            "--out", str(self.output_dir),
            "--concurrency", "4",  # Conservative per worker
            "--verbose"
        ]
        
        if house:
            cmd.extend(["--house", house])
        
        cmd = [x for x in cmd if x is not None]
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=7200,  # 2 hours timeout per worker 
                cwd=Path.cwd()
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                log.info(f"✅ Worker {worker_id}: Completed {range_str} in {duration/60:.1f}min")
                return {"success": True, "range": range_str, "duration": duration, "worker": worker_id}
            else:
                log.error(f"❌ Worker {worker_id}: Failed {range_str}: {result.stderr[:200]}")
                return {"success": False, "range": range_str, "error": result.stderr[:200], "worker": worker_id}
                
        except subprocess.TimeoutExpired:
            log.error(f"⏰ Worker {worker_id}: Timeout for {range_str}")
            return {"success": False, "range": range_str, "error": "timeout", "worker": worker_id}
        except Exception as e:
            log.error(f"💥 Worker {worker_id}: Exception for {range_str}: {e}")
            return {"success": False, "range": range_str, "error": str(e), "worker": worker_id}
    
    def strategy_by_house(self, start_year: int = 1803, end_year: int = 2005):
        """Strategy 1: Parallel by house (commons vs lords)."""
        log.info("🏛️ Strategy: Parallel by House")
        
        tasks = [
            {"start": start_year, "end": end_year, "house": "commons", "worker_id": 1},
            {"start": start_year, "end": end_year, "house": "lords", "worker_id": 2}
        ]
        
        years = end_year - start_year + 1
        log.info(f"⏱️ Estimated time: {self.estimate_time(years)} (2 parallel workers)")
        
        return tasks
    
    def strategy_by_decades(self, start_year: int = 1803, end_year: int = 2005):
        """Strategy 2: Parallel by decade chunks."""
        log.info("📅 Strategy: Parallel by Decades")
        
        tasks = []
        worker_id = 1
        
        for decade_start in range(start_year // 10 * 10, end_year + 1, 10):
            decade_end = min(decade_start + 9, end_year)
            if decade_start <= end_year:
                tasks.append({
                    "start": max(decade_start, start_year),
                    "end": min(decade_end, end_year),
                    "worker_id": worker_id
                })
                worker_id += 1
        
        years = end_year - start_year + 1
        log.info(f"⏱️ Estimated time: {self.estimate_time(years)} ({len(tasks)} parallel workers)")
        
        return tasks
    
    def strategy_by_century(self, start_year: int = 1803, end_year: int = 2005):
        """Strategy 3: Parallel by century (19th vs 20th century)."""
        log.info("💯 Strategy: Parallel by Century")
        
        tasks = []
        if start_year < 1900:
            tasks.append({
                "start": start_year,
                "end": min(1899, end_year),
                "worker_id": 1
            })
        
        if end_year >= 1900:
            tasks.append({
                "start": max(1900, start_year),
                "end": end_year,
                "worker_id": 2
            })
        
        years = end_year - start_year + 1
        log.info(f"⏱️ Estimated time: {self.estimate_time(years)} ({len(tasks)} parallel workers)")
        
        return tasks
    
    def strategy_aggressive_house(self, start_year: int = 1803, end_year: int = 2005):
        """Strategy 3.5: Aggressive house + century split for powerful systems."""
        log.info("🚀 Strategy: Aggressive House + Century Split")
        
        tasks = []
        worker_id = 1
        
        # Split by century AND house for maximum parallelization
        periods = []
        if start_year < 1900:
            periods.append((start_year, min(1899, end_year), "19th century"))
        if end_year >= 1900:
            periods.append((max(1900, start_year), end_year, "20th century"))
        
        for period_start, period_end, period_name in periods:
            for house in ["commons", "lords"]:
                tasks.append({
                    "start": period_start,
                    "end": period_end,
                    "house": house,
                    "worker_id": worker_id,
                    "name": f"{period_name} {house}"
                })
                worker_id += 1
        
        years = end_year - start_year + 1
        log.info(f"⏱️ Estimated time: {self.estimate_time(years)} ({len(tasks)} parallel workers)")
        
        return tasks
    
    def strategy_by_year_chunks(self, start_year: int = 1803, end_year: int = 2005, chunk_size: int = 20):
        """Strategy 4: Parallel by year chunks."""
        log.info(f"📊 Strategy: Parallel by {chunk_size}-year chunks")
        
        tasks = []
        worker_id = 1
        
        for year in range(start_year, end_year + 1, chunk_size):
            chunk_end = min(year + chunk_size - 1, end_year)
            tasks.append({
                "start": year,
                "end": chunk_end,
                "worker_id": worker_id
            })
            worker_id += 1
        
        years = end_year - start_year + 1
        log.info(f"⏱️ Estimated time: {self.estimate_time(years)} ({len(tasks)} parallel workers)")
        
        return tasks
    
    def run_parallel(self, strategy: str = "house", start_year: int = 1803, end_year: int = 2005):
        """Run parallel crawling with specified strategy."""
        
        log.info(f"🚀 Starting PARALLEL Hansard crawl: {start_year}-{end_year}")
        log.info(f"📁 Output directory: {self.output_dir}")
        log.info(f"🔧 Max workers: {self.max_workers}")
        log.info(f"🤝 Politeness features: 4 req/s per worker, staggered starts, respectful delays")
        
        # Generate tasks based on strategy
        if strategy == "house":
            tasks = self.strategy_by_house(start_year, end_year)
        elif strategy == "decades":
            tasks = self.strategy_by_decades(start_year, end_year)
        elif strategy == "century":
            tasks = self.strategy_by_century(start_year, end_year)
        elif strategy == "aggressive":
            tasks = self.strategy_aggressive_house(start_year, end_year)
        elif strategy == "chunks":
            tasks = self.strategy_by_year_chunks(start_year, end_year, 25)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Limit concurrent workers
        tasks = tasks[:self.max_workers]
        
        log.info(f"📋 Running {len(tasks)} parallel tasks:")
        for task in tasks:
            range_str = f"{task['start']}-{task['end']}" if task['start'] != task['end'] else str(task['start'])
            house_str = f" ({task['house']})" if 'house' in task else ""
            name_str = f" - {task['name']}" if 'name' in task else ""
            log.info(f"  Worker {task['worker_id']}: {range_str}{house_str}{name_str}")
        
        # Run tasks in parallel
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(self.run_single_crawler, task): task for task in tasks}
            results = []
            
            for future in concurrent.futures.as_completed(future_to_task):
                result = future.result()
                results.append(result)
                
                if result['success']:
                    log.info(f"✅ Completed: {result['range']} (Worker {result['worker']})")
                else:
                    log.error(f"❌ Failed: {result['range']} - {result['error']}")
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r['success'])
        
        log.info(f"🎉 Parallel crawl complete!")
        log.info(f"⏱️ Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
        log.info(f"✅ Successful: {successful}/{len(tasks)}")
        log.info(f"⚡ Effective speed: {(end_year-start_year+1)/(total_time/60):.1f} years/minute")
        
        # Save results
        status = {
            "strategy": strategy,
            "start_year": start_year,
            "end_year": end_year,
            "total_time_minutes": total_time / 60,
            "successful_tasks": successful,
            "total_tasks": len(tasks),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel Hansard Crawler - Maximum Speed")
    parser.add_argument("--strategy", choices=["house", "decades", "century", "aggressive", "chunks"], 
                       default="house", help="Parallel strategy")
    parser.add_argument("--start", type=int, default=1803, help="Start year")
    parser.add_argument("--end", type=int, default=2005, help="End year")
    parser.add_argument("--workers", type=int, default=2, help="Max parallel workers")
    parser.add_argument("--out", type=Path, default=Path("data/hansard"), help="Output directory")
    
    args = parser.parse_args()
    
    crawler = ParallelHansardCrawler(args.out, args.workers)
    crawler.run_parallel(
        strategy=args.strategy,
        start_year=args.start,
        end_year=args.end
    )

if __name__ == "__main__":
    main()