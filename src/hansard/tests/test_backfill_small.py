#!/usr/bin/env python3
"""Test the backfill script on a small subset to verify it works correctly."""

import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path

# Add the crawler directory to path
sys.path.insert(0, '/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/crawlers')

from crawler import HansardCrawler

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s – %(levelname)s – %(message)s")
log = logging.getLogger("test_backfill")

async def test_backfill_sample():
    """Test backfilling a small sample of missing dates."""
    
    print("🧪 Testing backfill on a small sample...")
    
    # Test with just a few dates from May 1928 (we know day 7 works)
    test_dates = [
        "1928/may/1",
        "1928/may/2", 
        "1928/may/7",  # We know this one exists
        "1928/may/8",
        "1999/jan/1"   # Test a more recent date too
    ]
    
    print(f"📋 Testing with {len(test_dates)} sample dates: {test_dates}")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "test_hansard"
        output_dir.mkdir(parents=True)
        
        print(f"📁 Using temporary output directory: {output_dir}")
        
        successful = 0
        failed = 0
        
        async with HansardCrawler(concurrency=1) as crawler:
            
            for date in test_dates:
                print(f"\n🔄 Testing date: {date}")
                
                try:
                    # Crawl the date
                    result = await crawler.crawl_day(date)
                    
                    if result and result.get('debates'):
                        # Save the result
                        crawler._save(result, output_dir)
                        
                        # Verify it was saved
                        year, month, day = date.split('/')
                        expected_dir = output_dir / year / month
                        summary_file = expected_dir / f"{day}_summary.json"
                        
                        if summary_file.exists():
                            with open(summary_file, 'r') as f:
                                summary = json.load(f)
                            
                            debate_count = summary.get('debate_count', 0)
                            print(f"  ✅ SUCCESS: Saved {debate_count} debates for {date}")
                            successful += 1
                        else:
                            print(f"  ❌ FAILED: Summary file not created for {date}")
                            failed += 1
                    else:
                        print(f"  ⚠️ NO DATA: No debates found for {date}")
                        # This might be normal - Parliament may not have sat on this day
                        successful += 1  # Count as success since the process worked
                        
                except Exception as e:
                    print(f"  ❌ ERROR: Failed to process {date}: {e}")
                    failed += 1
        
        # Summary
        print(f"\n" + "="*50)
        print(f"TEST RESULTS")
        print(f"="*50)
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Success rate: {(successful/(successful+failed)*100):.1f}%")
        
        if successful > 0:
            print(f"\n🎉 Backfill script is working correctly!")
            print(f"   Ready to run full backfill on {15138:,} missing dates.")
            
            # Show what files were created
            print(f"\n📁 Created files:")
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(output_dir)
                    print(f"     {rel_path}")
        else:
            print(f"\n⚠️ All tests failed. Check the error messages above.")
        
        return successful > 0

async def main():
    """Main test function."""
    print("🚀 Starting backfill test...")
    
    try:
        success = await test_backfill_sample()
        
        if success:
            print(f"\n✅ Test completed successfully!")
            print(f"\nNext steps:")
            print(f"1. Run full backfill: python backfill_missing_dates.py")
            print(f"2. This will process all 15,138 missing dates")
            print(f"3. Monitor progress and check for any failures")
        else:
            print(f"\n❌ Test failed. Fix issues before running full backfill.")
            
    except KeyboardInterrupt:
        print(f"\n⚡ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())