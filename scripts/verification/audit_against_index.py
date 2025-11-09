#!/usr/bin/env python3
"""
FAST Audit - parallel processing for large datasets.

Quickly compares our data against the complete index to identify:
1. Missing dates (no summary file)
2. Zero debate dates (empty summaries)
3. Incomplete dates (partial data)

Optimized for ~30,000 dates in 1-2 minutes.
"""

import json
from pathlib import Path
from collections import defaultdict
import sys
from concurrent.futures import ThreadPoolExecutor
import asyncio

def load_index(index_file: Path) -> dict:
    """Load the complete index."""
    if not index_file.exists():
        print(f"Error: Index file not found: {index_file}")
        print("Run: python3 scripts/create_complete_index.py first")
        sys.exit(1)

    with open(index_file) as f:
        data = json.load(f)
        return data["index"]

def check_single_date(args):
    """Check a single date (for parallel processing)."""
    year, month, day, info, data_dir = args

    expected_count = info["debate_count"]
    summary_path = data_dir / year / month / f"{day}_summary.json"

    result = {
        "date": f"{year}/{month}/{day}",
        "expected": expected_count,
    }

    if not summary_path.exists():
        result["status"] = "missing"
        result["got"] = 0
        return result

    # Load our summary
    try:
        with open(summary_path) as f:
            summary = json.load(f)
            our_count = summary.get("debate_count", 0)
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["got"] = 0
        return result

    result["got"] = our_count

    if our_count == 0 and expected_count > 0:
        result["status"] = "zero"
    elif our_count < expected_count:
        result["status"] = "incomplete"
        result["completeness"] = (our_count / expected_count) * 100
    elif our_count == expected_count:
        result["status"] = "complete"
    else:
        result["status"] = "extra"

    return result

def audit_data(index: dict, data_dir: Path) -> dict:
    """Fast parallel audit of our data against the index."""

    print("Preparing audit tasks...")

    # Collect all date check tasks
    tasks = []
    for year, months in index.items():
        for month, days in months.items():
            for day, info in days.items():
                tasks.append((year, month, day, info, data_dir))

    print(f"Auditing {len(tasks):,} dates in parallel...")

    # Process in parallel with thread pool
    results = {
        "missing_dates": [],
        "zero_debates": [],
        "incomplete_dates": [],
        "complete_dates": [],
        "extra_dates": [],
    }

    stats = {
        "index_dates": len(tasks),
        "index_debates": sum(args[3]["debate_count"] for args in tasks),
        "our_dates": 0,
        "our_debates": 0,
        "missing_dates": 0,
        "missing_debates": 0,
    }

    # Use thread pool for parallel file I/O (20 workers = fast on modern SSDs)
    import time
    start = time.time()

    with ThreadPoolExecutor(max_workers=20) as executor:
        # Process with progress indicator
        check_results = []
        for i, result in enumerate(executor.map(check_single_date, tasks), 1):
            check_results.append(result)
            if i % 5000 == 0:
                elapsed = time.time() - start
                rate = i / elapsed
                eta = (len(tasks) - i) / rate if rate > 0 else 0
                print(f"  Progress: {i:,}/{len(tasks):,} ({i*100//len(tasks)}%) | ETA: {eta:.0f}s")

    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s ({len(tasks)/elapsed:.0f} dates/sec)")
    print()

    # Process results
    for result in check_results:
        status = result["status"]
        expected = result["expected"]
        got = result["got"]

        if status == "missing":
            results["missing_dates"].append(result)
            stats["missing_dates"] += 1
            stats["missing_debates"] += expected
        elif status == "zero":
            results["zero_debates"].append(result)
            stats["missing_debates"] += expected
            stats["our_dates"] += 1
        elif status == "incomplete":
            results["incomplete_dates"].append(result)
            stats["our_debates"] += got
            stats["our_dates"] += 1
            stats["missing_debates"] += (expected - got)
        elif status == "complete":
            results["complete_dates"].append(result["date"])
            stats["our_debates"] += got
            stats["our_dates"] += 1
        elif status == "extra":
            results["incomplete_dates"].append(result)
            stats["our_debates"] += got
            stats["our_dates"] += 1

    print("Audit complete!")
    return results, stats

def main():
    import time
    overall_start = time.time()

    print("="*70)
    print("FAST HANSARD DATA AUDIT")
    print("="*70)
    print()

    # Load index
    index_file = Path("analysis/hansard_complete_index.json")
    print(f"Loading index: {index_file}")

    load_start = time.time()
    index = load_index(index_file)
    load_time = time.time() - load_start

    index_dates = sum(len(days) for months in index.values() for days in months.values())
    print(f"  Loaded {index_dates:,} dates in {load_time:.1f}s")
    print()

    # Audit
    data_dir = Path("data-hansard/hansard")
    print(f"Auditing: {data_dir}")
    print()

    results, stats = audit_data(index, data_dir)

    overall_time = time.time() - overall_start
    print(f"Total audit time: {overall_time:.1f}s")
    print()

    # Print results
    print("="*70)
    print("AUDIT RESULTS")
    print("="*70)
    print()

    print("OVERVIEW:")
    print(f"  Index expects: {stats['index_dates']:,} dates, {stats['index_debates']:,} debates")
    print(f"  We have:       {stats['our_dates']:,} dates, {stats['our_debates']:,} debates")
    print(f"  Missing:       {stats['missing_dates']:,} dates, {stats['missing_debates']:,} debates")
    print()

    if stats['index_debates'] > 0:
        completeness = (stats['our_debates'] / stats['index_debates']) * 100
        print(f"Overall completeness: {completeness:.1f}%")
        print()

    print("BREAKDOWN:")
    print(f"  Complete dates: {len(results['complete_dates']):,}")
    print(f"  Incomplete dates: {len(results['incomplete_dates']):,}")
    print(f"  Zero debate dates: {len(results['zero_debates']):,}")
    print(f"  Missing entirely: {len(results['missing_dates']):,}")
    print(f"  Extra dates: {len(results['extra_dates']):,}")
    print()

    # Sample worst cases
    if results['incomplete_dates']:
        print("Worst incomplete dates (by % missing):")
        worst = sorted(results['incomplete_dates'],
                      key=lambda x: x.get('completeness', 100))[:10]
        for item in worst:
            if 'completeness' in item:
                print(f"  {item['date']}: {item['got']}/{item['expected']} ({item['completeness']:.1f}%)")
        print()

    # Save detailed report
    output_file = Path("analysis/audit_report.json")
    with open(output_file, 'w') as f:
        json.dump({
            "stats": stats,
            "results": results
        }, f, indent=2)

    print(f"Detailed report saved to: {output_file}")
    print()

    # Create re-crawl list
    recrawl_file = Path("analysis/dates_to_recrawl.txt")
    with open(recrawl_file, 'w') as f:
        # Missing dates
        for item in results['missing_dates']:
            f.write(f"{item['date']}\n")

        # Zero debate dates
        for item in results['zero_debates']:
            f.write(f"{item['date']}\n")

        # Incomplete dates
        for item in results['incomplete_dates']:
            f.write(f"{item['date']}\n")

    total_to_recrawl = (len(results['missing_dates']) +
                       len(results['zero_debates']) +
                       len(results['incomplete_dates']))

    print(f"Re-crawl list saved to: {recrawl_file}")
    print(f"Total dates to re-crawl: {total_to_recrawl:,}")
    print()

    print("Next step: python3 scripts/systematic_recrawl.py")

if __name__ == "__main__":
    main()
