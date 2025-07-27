#!/usr/bin/env python3
"""Test script to verify the single-digit date fix in the crawler."""

import asyncio
import sys
import logging
from pathlib import Path

# Add the crawler directory to path
sys.path.insert(0, '/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/crawlers')

from crawler import HansardCrawler

# Set up verbose logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s – %(levelname)s – %(message)s")
log = logging.getLogger("hansard")
log.setLevel(logging.DEBUG)

async def test_single_digit_discovery():
    """Test if single-digit days are properly discovered and processed."""
    
    print("Testing single-digit day discovery and processing...")
    print("=" * 60)
    
    async with HansardCrawler(concurrency=1) as crawler:
        
        # Test 1: Discovery for May 1928 (should include day 7)
        print("\n1. Testing day discovery for May 1928...")
        try:
            days = await crawler.discover_days_in_month("1928/may")
            print(f"   Discovered {len(days)} days: {days}")
            
            single_digit_days = [d for d in days if d.split('/')[-1].isdigit() and len(d.split('/')[-1]) == 1]
            if single_digit_days:
                print(f"   ✓ Single-digit days found: {single_digit_days}")
            else:
                print("   ✗ No single-digit days found")
                
            if "1928/may/7" in days:
                print("   ✓ Day 7 is in the discovery list")
            else:
                print("   ✗ Day 7 is NOT in the discovery list")
                return
                
        except Exception as e:
            print(f"   ✗ Discovery failed: {e}")
            return
        
        # Test 2: Try to crawl day 7 specifically
        print("\n2. Testing crawl_day for 1928/may/7...")
        try:
            result = await crawler.crawl_day("1928/may/7")
            if result:
                print(f"   ✓ Successfully crawled day 7")
                print(f"   Date: {result.get('date')}")
                print(f"   Debates: {len(result.get('debates', []))}")
                
                # Test 3: Try to save the result
                print("\n3. Testing save for day 7...")
                test_dir = Path("/tmp/hansard_test")
                test_dir.mkdir(exist_ok=True)
                
                try:
                    crawler._save(result, test_dir)
                    print("   ✓ Save completed successfully")
                    
                    # Check if files were created
                    expected_dir = test_dir / "1928" / "may"
                    if expected_dir.exists():
                        files = list(expected_dir.glob("7_*"))
                        print(f"   ✓ Created {len(files)} files for day 7")
                        for f in files[:3]:  # Show first 3 files
                            print(f"     - {f.name}")
                    else:
                        print("   ✗ No directory created for day 7")
                        
                except Exception as e:
                    print(f"   ✗ Save failed: {e}")
                    
            else:
                print("   ✗ crawl_day returned None for day 7")
                
        except Exception as e:
            print(f"   ✗ Crawling day 7 failed: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed. Check the logs above for detailed information.")

if __name__ == "__main__":
    try:
        asyncio.run(test_single_digit_discovery())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()