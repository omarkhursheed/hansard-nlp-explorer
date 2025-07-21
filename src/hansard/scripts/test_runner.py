#!/usr/bin/env python3
"""
Hansard Crawler Test Suite - Progressive testing before full run.
"""

import subprocess
import time
import psutil
import logging
from pathlib import Path
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("hansard_test.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("test_suite")

class HansardTestSuite:
    def __init__(self, output_dir: Path = Path("test_data/hansard")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def monitor_system(self) -> dict:
        """Monitor system resources."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_free_gb": psutil.disk_usage(".").free / (1024**3),
            "network_sent": psutil.net_io_counters().bytes_sent,
            "network_recv": psutil.net_io_counters().bytes_recv
        }
    
    def run_test(self, name: str, cmd: list, timeout: int = 300) -> dict:
        """Run a single test and monitor performance."""
        log.info(f"🧪 Starting test: {name}")
        log.info(f"📝 Command: {' '.join(cmd)}")
        
        # Monitor before
        before = self.monitor_system()
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.cwd()
            )
            
            duration = time.time() - start_time
            after = self.monitor_system()
            
            # Calculate resource usage
            cpu_avg = (before["cpu_percent"] + after["cpu_percent"]) / 2
            memory_usage = after["memory_percent"] - before["memory_percent"]
            network_sent = after["network_sent"] - before["network_sent"]
            network_recv = after["network_recv"] - before["network_recv"]
            
            success = result.returncode == 0
            
            test_result = {
                "name": name,
                "success": success,
                "duration_seconds": duration,
                "cpu_avg_percent": cpu_avg,
                "memory_change_percent": memory_usage,
                "network_sent_mb": network_sent / (1024**2),
                "network_recv_mb": network_recv / (1024**2),
                "stdout_lines": len(result.stdout.split('\n')),
                "stderr_lines": len(result.stderr.split('\n')),
                "error_summary": result.stderr[-200:] if result.stderr else None
            }
            
            if success:
                log.info(f"✅ {name} completed in {duration:.1f}s")
                log.info(f"📊 CPU: {cpu_avg:.1f}%, Network: {network_recv/(1024**2):.1f}MB received")
            else:
                log.error(f"❌ {name} failed: {result.stderr[-100:]}")
            
            self.results.append(test_result)
            return test_result
            
        except subprocess.TimeoutExpired:
            log.error(f"⏰ {name} timed out after {timeout}s")
            return {"name": name, "success": False, "error": "timeout"}
        except Exception as e:
            log.error(f"💥 {name} exception: {e}")
            return {"name": name, "success": False, "error": str(e)}
    
    def test_1_basic_functionality(self):
        """Test 1: Basic functionality with single year."""
        log.info("=" * 50)
        log.info("TEST 1: Basic Functionality")
        log.info("Testing single year to verify crawler works")
        
        cmd = [
            "python", "crawler.py", "1864", 
            "--house", "commons",
            "--out", str(self.output_dir),
            "--concurrency", "2",  # Conservative
            "--verbose"
        ]
        
        return self.run_test("Basic Single Year", cmd, timeout=600)  # 10 minutes
    
    def test_2_optimized_settings(self):
        """Test 2: Optimized settings on small range."""
        log.info("=" * 50)
        log.info("TEST 2: Optimized Settings")
        log.info("Testing faster settings on 2-year range")
        
        cmd = [
            "python", "crawler.py", "1863", "1864",
            "--house", "commons", 
            "--out", str(self.output_dir),
            "--concurrency", "6",  # Faster settings
            "--verbose"
        ]
        
        return self.run_test("Optimized 2-Year Range", cmd, timeout=900)  # 15 minutes
    
    def test_3_decade_range(self):
        """Test 3: Small decade to test discovery efficiency."""
        log.info("=" * 50)
        log.info("TEST 3: Decade Range")
        log.info("Testing decade discovery with 3 years")
        
        cmd = [
            "python", "crawler.py", "1863", "1865",
            "--out", str(self.output_dir),
            "--concurrency", "4",
            "--verbose"
        ]
        
        return self.run_test("3-Year Range Both Houses", cmd, timeout=1200)  # 20 minutes
    
    def test_4_parallel_approach(self):
        """Test 4: Parallel processing on small range."""
        log.info("=" * 50)
        log.info("TEST 4: Parallel Processing")
        log.info("Testing parallel house-based approach")
        
        cmd = [
            "python", "parallel_hansard_runner.py",
            "--strategy", "house",
            "--start", "1863", "--end", "1864",
            "--workers", "2",
            "--out", str(self.output_dir)
        ]
        
        return self.run_test("Parallel House Strategy", cmd, timeout=1200)  # 20 minutes
    
    def test_5_stress_test(self):
        """Test 5: Stress test with aggressive settings."""
        log.info("=" * 50)
        log.info("TEST 5: Stress Test")
        log.info("Testing aggressive settings to find limits")
        
        cmd = [
            "python", "crawler.py", "1860", "1862",
            "--house", "commons",
            "--out", str(self.output_dir),
            "--concurrency", "8",  # Aggressive
            "--verbose"
        ]
        
        return self.run_test("Aggressive 3-Year Stress", cmd, timeout=1800)  # 30 minutes
    
    def check_data_integrity(self):
        """Check that downloaded data looks correct."""
        log.info("=" * 50)
        log.info("DATA INTEGRITY CHECK")
        
        # Count files and check sizes
        html_files = list(self.output_dir.rglob("*.html.gz"))
        json_files = list(self.output_dir.rglob("*_summary.json"))
        
        total_size_mb = sum(f.stat().st_size for f in html_files) / (1024**2)
        
        log.info(f"📁 Downloaded files:")
        log.info(f"   HTML debates: {len(html_files)} files")
        log.info(f"   JSON summaries: {len(json_files)} files")
        log.info(f"   Total size: {total_size_mb:.1f} MB")
        
        # Sample a few files to check content
        if html_files:
            import gzip
            sample_file = html_files[0]
            try:
                with gzip.open(sample_file, 'rt', encoding='utf-8') as f:
                    content = f.read(1000)  # First 1KB
                    if "hansard" in content.lower() or "commons" in content.lower():
                        log.info("✅ Sample file contains expected content")
                    else:
                        log.warning("⚠️ Sample file content looks unexpected")
            except Exception as e:
                log.error(f"❌ Error reading sample file: {e}")
        
        return {
            "html_files": len(html_files),
            "json_files": len(json_files),
            "total_size_mb": total_size_mb
        }
    
    def run_full_test_suite(self):
        """Run the complete test suite."""
        log.info("🚀 STARTING HANSARD CRAWLER TEST SUITE")
        log.info(f"📁 Test output directory: {self.output_dir}")
        log.info(f"💻 System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total/(1024**3):.1f}GB RAM")
        
        start_time = time.time()
        
        # Run all tests
        tests = [
            self.test_1_basic_functionality,
            self.test_2_optimized_settings,
            self.test_3_decade_range,
            self.test_4_parallel_approach,
            self.test_5_stress_test
        ]
        
        for test_func in tests:
            try:
                test_func()
                time.sleep(5)  # Brief pause between tests
            except Exception as e:
                log.error(f"Test failed with exception: {e}")
        
        # Check data integrity
        data_check = self.check_data_integrity()
        
        # Generate report
        total_time = time.time() - start_time
        self.generate_report(total_time, data_check)
    
    def generate_report(self, total_time: float, data_check: dict):
        """Generate final test report."""
        log.info("=" * 50)
        log.info("🎉 TEST SUITE COMPLETE")
        
        successful_tests = [r for r in self.results if r.get("success", False)]
        failed_tests = [r for r in self.results if not r.get("success", False)]
        
        log.info(f"⏱️ Total test time: {total_time/60:.1f} minutes")
        log.info(f"✅ Successful tests: {len(successful_tests)}/{len(self.results)}")
        
        if successful_tests:
            avg_speed = sum(r.get("duration_seconds", 0) for r in successful_tests) / len(successful_tests)
            log.info(f"📊 Average test duration: {avg_speed:.1f} seconds")
        
        if failed_tests:
            log.warning(f"❌ Failed tests: {[t['name'] for t in failed_tests]}")
        
        # Estimate full dataset time based on successful tests
        if successful_tests:
            # Use fastest successful test to estimate
            fastest_test = min(successful_tests, key=lambda x: x.get("duration_seconds", float('inf')))
            # Very rough estimate: scale up based on data volume
            estimated_hours = (fastest_test["duration_seconds"] * 200) / 3600  # 200 years worth
            log.info(f"🔮 Estimated full dataset time: ~{estimated_hours:.1f} hours")
        
        # Save report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_time_minutes": total_time / 60,
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "test_results": self.results,
            "data_integrity": data_check,
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "total_ram_gb": psutil.virtual_memory().total / (1024**3),
                "disk_free_gb": psutil.disk_usage(".").free / (1024**3)
            }
        }
        
        with open(self.output_dir / "test_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        log.info(f"📄 Full report saved to: {self.output_dir / 'test_report.json'}")
        
        # Recommendations
        log.info("=" * 50)
        log.info("🎯 RECOMMENDATIONS:")
        
        if len(successful_tests) >= 3:
            log.info("✅ Your laptop can handle this crawler!")
            log.info("🚀 Proceed with optimized settings for full run")
        elif len(successful_tests) >= 1:
            log.info("⚠️ Use conservative settings for full run")
            log.info("🐌 Consider running overnight or in smaller chunks")
        else:
            log.info("❌ Issues detected - check logs and system resources")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hansard Crawler Test Suite")
    parser.add_argument("--out", type=Path, default=Path("test_data/hansard"), 
                       help="Test output directory")
    
    args = parser.parse_args()
    
    tester = HansardTestSuite(args.out)
    tester.run_full_test_suite()

if __name__ == "__main__":
    main()