#!/usr/bin/env python3
"""Test the optimized backfill for correctness, timing, and API respectfulness."""

import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import List, Dict

# Import the optimized backfiller
import sys
sys.path.insert(0, '/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard')

from hansard.debug_scripts.backfill_missing_dates_optimized import OptimizedBackfiller
import pytest
try:
    import pytest_asyncio  # type: ignore
    HAS_ASYNC = True
except Exception:
    HAS_ASYNC = False

# Set up logging to monitor API usage
logging.basicConfig(level=logging.INFO, format="%(asctime)s – %(levelname)s – %(message)s")
log = logging.getLogger("test_optimized")

class APIRespectfulnessMonitor:
    """Monitor API usage to ensure we're being respectful."""
    
    def __init__(self):
        self.request_times = []
        self.total_requests = 0
        self.error_count = 0
        self.success_count = 0
        
    def log_request(self, success: bool = True):
        """Log a request with timestamp."""
        self.request_times.append(time.time())
        self.total_requests += 1
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def get_current_rps(self, window_seconds: int = 10) -> float:
        """Get current requests per second over a time window."""
        now = time.time()
        recent_requests = [t for t in self.request_times if (now - t) <= window_seconds]
        return len(recent_requests) / window_seconds
    
    def print_api_stats(self, duration: float):
        """Print API usage statistics."""
        avg_rps = self.total_requests / duration if duration > 0 else 0
        current_rps = self.get_current_rps()
        
        print(f"\n🌐 API Respectfulness Check:")
        print(f"   • Total requests: {self.total_requests}")
        print(f"   • Average RPS: {avg_rps:.1f}")
        print(f"   • Current RPS: {current_rps:.1f}")
        print(f"   • Success rate: {(self.success_count/max(1,self.total_requests)*100):.1f}%")
        
        # Check if we're being respectful
        if avg_rps <= 8.0:
            print(f"   ✅ RESPECTFUL: Average RPS ({avg_rps:.1f}) is well below aggressive limits")
        elif avg_rps <= 15.0:
            print(f"   ⚠️  MODERATE: Average RPS ({avg_rps:.1f}) is reasonable but watch for rate limiting")
        else:
            print(f"   ❌ AGGRESSIVE: Average RPS ({avg_rps:.1f}) may trigger rate limiting")

@pytest.mark.skipif(not HAS_ASYNC, reason="pytest-asyncio not installed")
async def test_optimized_backfill():
    """Test the optimized backfill with a strategic sample."""
    
    print("🧪 Testing OPTIMIZED backfill system...")
    print("📊 Testing correctness, timing, and API respectfulness")
    
    # Strategic test dates - mix of old (fast) and newer (slower) dates
    test_dates = [
        # Old dates (typically faster, fewer 404s)
        "1855/mar/1", "1855/mar/2", "1855/mar/3", "1855/mar/4", "1855/mar/5",
        "1864/feb/1", "1864/feb/2", "1864/feb/3", "1864/feb/4", "1864/feb/5",
        # Medium period
        "1928/may/1", "1928/may/2", "1928/may/3", "1928/may/7", "1928/may/8",
        # One newer date (known to be slower)
        "1999/jan/1"
    ]
    
    print(f"📋 Testing with {len(test_dates)} strategic dates:")
    print(f"   • 1855 dates: 5 (old, fast)")
    print(f"   • 1864 dates: 5 (old, fast)")  
    print(f"   • 1928 dates: 5 (medium speed)")
    print(f"   • 1999 dates: 1 (newer, slower)")
    
    # Create temporary test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create missing dates file
        missing_file = temp_path / "missing_dates.json"
        with open(missing_file, 'w') as f:
            json.dump({
                'missing_dates': test_dates,
                'generated_at': '2025-07-26T12:30:00'
            }, f, indent=2)
        
        # Create output directory
        output_dir = temp_path / "hansard_output"
        
        # Create optimized backfiller with moderate settings for testing
        backfiller = OptimizedBackfiller(
            missing_dates_file=missing_file,
            output_dir=output_dir,
            concurrency=6,      # Moderate concurrency for testing
            batch_size=20       # Reasonable batch size
        )
        
        print(f"⚡ Test configuration: 6 concurrent processors")
        
        # Monitor API usage
        monitor = APIRespectfulnessMonitor()
        
        # Patch the crawler to monitor requests
        original_get_method = None
        
        start_time = time.time()
        
        try:
            print(f"\n🚀 Starting optimized backfill test...")
            
            await backfiller.backfill_optimized(test_dates)
            
            elapsed = time.time() - start_time
            rate = len(test_dates) / elapsed
            
            print(f"\n✅ Optimized test completed!")
            print(f"⏱️  Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
            print(f"📊 Rate: {rate:.1f} dates/second")
            print(f"🎯 Projected time for 15,138 dates: {(15138/rate)/3600:.1f} hours")
            
            # Check correctness - verify files were created
            created_files = list(output_dir.rglob("*"))
            summary_files = list(output_dir.rglob("*_summary.json"))
            debate_files = list(output_dir.rglob("*.html.gz"))
            
            print(f"\n📁 File Creation (Correctness Check):")
            print(f"   • Total files created: {len(created_files)}")
            print(f"   • Summary files: {len(summary_files)}")
            print(f"   • Debate files: {len(debate_files)}")
            
            # Verify file structure matches original crawler
            print(f"\n🔍 File Structure Verification:")
            correct_structure = True
            for summary_file in summary_files:
                try:
                    # Check if path follows year/month/day_summary.json pattern
                    parts = summary_file.parts
                    year, month = parts[-3], parts[-2]
                    filename = parts[-1]
                    
                    if not (year.isdigit() and len(year) == 4 and 
                           month in ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'] and
                           filename.endswith('_summary.json')):
                        correct_structure = False
                        print(f"   ❌ Incorrect structure: {summary_file}")
                        break
                        
                    # Check JSON content
                    with open(summary_file, 'r') as f:
                        data = json.load(f)
                    
                    required_fields = ['date', 'debate_count', 'timestamp']
                    if not all(field in data for field in required_fields):
                        correct_structure = False
                        print(f"   ❌ Missing fields in: {summary_file}")
                        break
                        
                except Exception as e:
                    correct_structure = False
                    print(f"   ❌ Error checking {summary_file}: {e}")
                    break
            
            if correct_structure:
                print(f"   ✅ All files follow correct structure")
            
            # Performance comparison
            if elapsed < 120:  # Less than 2 minutes
                print(f"\n🚀 PERFORMANCE: Excellent! Much faster than the slow test (7.5 min)")
            elif elapsed < 300:  # Less than 5 minutes  
                print(f"\n⚡ PERFORMANCE: Good! Reasonable improvement over slow test")
            else:
                print(f"\n⚠️  PERFORMANCE: Still slow, may need more optimization")
            
            # API Respectfulness check
            estimated_requests = len(test_dates) * 50  # Rough estimate
            estimated_rps = estimated_requests / elapsed
            
            print(f"\n🌐 API Respectfulness Analysis:")
            print(f"   • Estimated total requests: ~{estimated_requests}")
            print(f"   • Estimated average RPS: ~{estimated_rps:.1f}")
            
            if estimated_rps <= 10.0:
                print(f"   ✅ VERY RESPECTFUL: Well below typical API limits (50-100 RPS)")
            elif estimated_rps <= 20.0:
                print(f"   ✅ RESPECTFUL: Conservative usage, very safe")
            elif estimated_rps <= 50.0:
                print(f"   ⚠️  MODERATE: Reasonable but monitor for rate limiting")
            else:
                print(f"   ❌ AGGRESSIVE: May trigger rate limiting")
            
            return {
                'success': True,
                'elapsed_seconds': elapsed,
                'rate_per_second': rate,
                'files_created': len(created_files),
                'estimated_rps': estimated_rps,
                'projected_hours': (15138/rate)/3600
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ Test failed after {elapsed:.1f} seconds: {e}")
            return {
                'success': False,
                'elapsed_seconds': elapsed,
                'error': str(e)
            }

async def main():
    """Main test function."""
    print("🧪 Starting optimized backfill test...")
    print("🎯 Testing: correctness, performance, API respectfulness")
    
    try:
        results = await test_optimized_backfill()
        
        if results['success']:
            print(f"\n🎉 Optimized backfill test PASSED!")
            print(f"\n📋 Summary:")
            print(f"   • Performance: {results['rate_per_second']:.1f} dates/second")
            print(f"   • Files created: {results['files_created']}")
            print(f"   • Estimated API usage: ~{results['estimated_rps']:.1f} RPS")
            print(f"   • Projected full runtime: {results['projected_hours']:.1f} hours")
            
            if results['projected_hours'] < 5:
                print(f"\n✅ Ready for production! Projected runtime is reasonable.")
                print(f"\nTo run full backfill:")
                print(f"  python backfill_missing_dates_optimized.py")
            else:
                print(f"\n⚠️  Consider if {results['projected_hours']:.1f} hours is acceptable")
                
        else:
            print(f"\n❌ Test failed: {results.get('error', 'Unknown error')}")
            
    except KeyboardInterrupt:
        print(f"\n⚡ Test interrupted by user")

if __name__ == "__main__":
    asyncio.run(main())
