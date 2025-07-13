#!/usr/bin/env python3
"""
Real-time test runner that shows crawler output as it happens.
"""

import subprocess
import threading
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("realtime_test")

def run_with_realtime_output(cmd, timeout=600):
    """Run command and show output in real-time."""
    log.info(f"🚀 Running: {' '.join(cmd)}")
    log.info("📡 Real-time output:")
    print("-" * 50)
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Read output line by line in real-time
        start_time = time.time()
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
            
            # Check timeout
            if time.time() - start_time > timeout:
                log.error(f"⏰ Timeout after {timeout}s")
                process.terminate()
                return False
        
        rc = process.poll()
        duration = time.time() - start_time
        
        print("-" * 50)
        if rc == 0:
            log.info(f"✅ Completed successfully in {duration:.1f}s")
            return True
        else:
            log.error(f"❌ Failed with return code {rc}")
            return False
            
    except Exception as e:
        log.error(f"💥 Exception: {e}")
        return False

def quick_test():
    """Quick single test with real-time output."""
    
    output_dir = Path("test_data/hansard")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("🧪 QUICK REAL-TIME TEST")
    log.info(f"📁 Output: {output_dir}")
    
    # Test 1: Basic single year
    cmd = [
        "python", "crawler.py", "1864", 
        "--house", "commons",
        "--out", str(output_dir),
        "--concurrency", "4",  # Use more of your powerful system
        "--verbose"
    ]
    
    success = run_with_realtime_output(cmd, timeout=600)
    
    if success:
        # Check what was downloaded
        html_files = list(output_dir.rglob("*.html.gz"))
        json_files = list(output_dir.rglob("*_summary.json"))
        
        log.info(f"📊 Results:")
        log.info(f"   HTML files: {len(html_files)}")
        log.info(f"   JSON files: {len(json_files)}")
        
        if html_files:
            total_size = sum(f.stat().st_size for f in html_files) / (1024**2)
            log.info(f"   Total size: {total_size:.1f} MB")
            log.info(f"✅ Test successful - crawler is working!")
            
            # Quick speed estimate
            log.info(f"🔮 Speed estimate for full dataset:")
            log.info(f"   Your system could probably complete in ~1-2 hours")
        else:
            log.warning("⚠️ No files downloaded - check for errors above")
    else:
        log.error("❌ Test failed - check errors above")

if __name__ == "__main__":
    quick_test()