#!/usr/bin/env python3
"""Debug the exact crawler workflow to find where day 7 gets lost."""

import asyncio
import sys
import os

# Add the crawler directory to path
sys.path.insert(0, '/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/crawlers')

from crawler import HansardCrawler

async def debug_crawler_flow():
    """Test the exact crawler workflow for May 1928."""
    
    print("Testing crawler workflow for May 1928...")
    
    async with HansardCrawler(concurrency=1) as crawler:
        
        # Step 1: Test day discovery for May 1928
        print("\n1. Testing discover_days_in_month for 1928/may...")
        days = await crawler.discover_days_in_month("1928/may")
        print(f"Days discovered: {days}")
        
        if "1928/may/7" in days:
            print("✓ Day 7 found in discover_days_in_month")
        else:
            print("✗ Day 7 NOT found in discover_days_in_month")
            return
        
        # Step 2: Test crawling day 7 specifically
        print("\n2. Testing crawl_day for 1928/may/7...")
        try:
            result = await crawler.crawl_day("1928/may/7")
            if result:
                print(f"✓ crawl_day returned data for day 7")
                print(f"  Date: {result.get('date')}")
                print(f"  Debates: {len(result.get('debates', []))}")
            else:
                print("✗ crawl_day returned None for day 7")
        except Exception as e:
            print(f"✗ crawl_day failed for day 7: {e}")

if __name__ == "__main__":
    asyncio.run(debug_crawler_flow())