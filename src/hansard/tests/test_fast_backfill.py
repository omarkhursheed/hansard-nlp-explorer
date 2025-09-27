#!/usr/bin/env python3
"""Quick test of the fast backfill system."""

import asyncio
import json
import tempfile
import time
from pathlib import Path

# Import the fast backfiller
import sys
sys.path.insert(0, '/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard')

from hansard.debug_scripts.backfill_missing_dates_fast import FastBackfiller
import pytest
try:
    import pytest_asyncio  # type: ignore
    HAS_ASYNC = True
except Exception:
    HAS_ASYNC = False

@pytest.mark.skipif(not HAS_ASYNC, reason="pytest-asyncio not installed")
async def test_fast_backfill():
    """Test the fast backfill with a small sample."""
    
    print("🧪 Testing FAST backfill system...")
    
    # Create test data
    test_dates = [
        "1928/may/1", "1928/may/2", "1928/may/3", "1928/may/7", "1928/may/8",
        "1855/mar/1", "1855/mar/2", "1855/mar/3", "1855/mar/4", "1855/mar/5",
        "1999/jan/1", "1999/jan/2", "1999/jan/3", "1999/feb/1", "1999/feb/2"
    ]
    
    print(f"📋 Testing with {len(test_dates)} dates from different months/years")
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create missing dates file
        missing_file = temp_path / "missing_dates.json"
        with open(missing_file, 'w') as f:
            json.dump({
                'missing_dates': test_dates,
                'generated_at': '2025-07-26T12:00:00'
            }, f, indent=2)
        
        # Create output directory
        output_dir = temp_path / "hansard_output"
        
        # Test with fast settings
        backfiller = FastBackfiller(
            missing_dates_file=missing_file,
            output_dir=output_dir,
            concurrency=4,      # Smaller for test
            batch_size=5,       # Smaller batches for test
            max_rps=4.0         # Moderate speed for test
        )
        
        print(f"⚡ Configuration: 4 crawlers, 5 batch size, 4.0 RPS")
        
        start_time = time.time()
        
        try:
            await backfiller.backfill_parallel(test_dates)
            
            elapsed = time.time() - start_time
            rate = len(test_dates) / elapsed
            
            print(f"\n✅ Fast test completed!")
            print(f"⏱️  Time: {elapsed:.1f} seconds")
            print(f"📊 Rate: {rate:.1f} dates/second")
            print(f"🎯 Projected time for 15,138 dates: {(15138/rate)/60:.1f} minutes")
            
            # Check what was created
            created_files = list(output_dir.rglob("*"))
            print(f"📁 Created {len(created_files)} files")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

async def main():
    """Main test function."""
    print("🚀 Starting fast backfill test...")
    
    try:
        success = await test_fast_backfill()
        
        if success:
            print(f"\n🎉 Fast backfill is ready!")
            print(f"\nTo run full backfill:")
            print(f"  python backfill_missing_dates_fast.py")
        else:
            print(f"\n⚠️ Fix issues before running full backfill")
            
    except KeyboardInterrupt:
        print(f"\n⚡ Test interrupted")

if __name__ == "__main__":
    asyncio.run(main())
