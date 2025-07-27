#!/usr/bin/env python3
"""Quick test to verify timeout handling works correctly."""

import asyncio
import tempfile
import time
from pathlib import Path
import json

# Import the final backfiller
import sys
sys.path.insert(0, '/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard')

from backfill_missing_dates_final import FinalOptimizedBackfiller

async def test_timeout_handling():
    """Test that timeout handling prevents infinite hanging."""
    
    print("🧪 Testing timeout handling...")
    
    # Test with a mix including known slow dates
    test_dates = [
        "1928/may/7",  # Known good
        "1855/mar/1",  # Probably good
        "1999/feb/1",  # Known to be slow with many 404s
        "1999/jan/1",  # Another potentially slow date
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        missing_file = temp_path / "missing_dates.json"
        with open(missing_file, 'w') as f:
            json.dump({'missing_dates': test_dates}, f)
        
        output_dir = temp_path / "output"
        
        # Create backfiller with aggressive timeouts
        backfiller = FinalOptimizedBackfiller(missing_file, output_dir)
        
        print(f"📊 Testing {len(test_dates)} dates with aggressive timeouts")
        print(f"⏱️  Timeouts: 45s per date, 5min per chunk")
        
        start_time = time.time()
        
        try:
            await backfiller.backfill_final(test_dates)
            
            elapsed = time.time() - start_time
            print(f"\n✅ Timeout test completed in {elapsed:.1f} seconds")
            
            if elapsed < 300:  # Less than 5 minutes
                print(f"🚀 EXCELLENT: No hanging issues detected")
            elif elapsed < 600:  # Less than 10 minutes
                print(f"✅ GOOD: Reasonable performance with timeouts")
            else:
                print(f"⚠️  SLOW: May still have timeout issues")
            
            # Check if files were created
            files = list(output_dir.rglob("*"))
            print(f"📁 Created {len(files)} files")
            
            return elapsed < 600  # Success if under 10 minutes
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Test failed after {elapsed:.1f}s: {e}")
            return False

async def main():
    """Run timeout test."""
    print("🚀 Starting timeout handling test...")
    
    try:
        success = await test_timeout_handling()
        
        if success:
            print(f"\n🎉 Timeout handling works correctly!")
            print(f"✅ Ready for production run with fast settings")
            print(f"\nTo start the full backfill:")
            print(f"  python backfill_missing_dates_final.py")
        else:
            print(f"\n❌ Timeout issues detected - use slower version")
            
    except KeyboardInterrupt:
        print(f"\n⚡ Test interrupted")

if __name__ == "__main__":
    asyncio.run(main())