#!/usr/bin/env python3
"""
Real-time progress estimator for Hansard crawler.
"""

import time
import json
from pathlib import Path
from datetime import datetime, timedelta

def analyze_progress():
    """Analyze current crawler progress and estimate completion time."""
    
    print("🔍 HANSARD CRAWLER PROGRESS ANALYSIS")
    print("=" * 50)
    
    # Check data directory
    data_dir = Path("data/hansard")
    
    if not data_dir.exists():
        print("❌ No data directory found yet")
        return
    
    # Count files and calculate sizes
    html_files = list(data_dir.rglob("*.html.gz"))
    json_files = list(data_dir.rglob("*_summary.json"))
    
    total_size_mb = sum(f.stat().st_size for f in html_files) / (1024**2)
    
    print(f"📁 Current Progress:")
    print(f"   Debate files: {len(html_files)}")
    print(f"   Summary files: {len(json_files)}")
    print(f"   Total size: {total_size_mb:.2f} MB")
    
    # Analyze directory structure to see what years are being processed
    year_dirs = set()
    month_dirs = set()
    
    for html_file in html_files:
        parts = html_file.parts
        if len(parts) >= 4:  # data/hansard/YEAR/MONTH/
            year_dirs.add(parts[2])
            month_dirs.add(f"{parts[2]}/{parts[3]}")
    
    if year_dirs:
        years = sorted([int(y) for y in year_dirs if y.isdigit()])
        print(f"📅 Years with data: {min(years)}-{max(years)} ({len(years)} years)")
        print(f"📊 Months processed: {len(month_dirs)} month directories")
        
        # Estimate based on years completed
        total_years = 2005 - 1803 + 1  # 203 years
        years_completed = len(years)
        completion_percentage = (years_completed / total_years) * 100
        
        print(f"🎯 Estimated completion: {completion_percentage:.1f}%")
    else:
        print("📅 No year data found yet (still in discovery phase)")
        years_completed = 0
        completion_percentage = 0
    
    # Try to read start time from parallel status
    status_file = data_dir / "parallel_status.json"
    start_time = None
    
    if status_file.exists():
        try:
            with open(status_file) as f:
                status = json.load(f)
                start_time = datetime.fromisoformat(status.get("timestamp", ""))
        except:
            pass
    
    # Estimate start time from process (fallback)
    if not start_time:
        # Assume crawler started recently (you can adjust this)
        start_time = datetime.now() - timedelta(minutes=10)
    
    elapsed_time = datetime.now() - start_time
    elapsed_minutes = elapsed_time.total_seconds() / 60
    
    print(f"⏱️ Runtime: {elapsed_minutes:.1f} minutes")
    
    # Calculate rates and estimates
    if len(html_files) > 0 and elapsed_minutes > 0:
        files_per_minute = len(html_files) / elapsed_minutes
        mb_per_minute = total_size_mb / elapsed_minutes
        
        print(f"📈 Current Rate:")
        print(f"   {files_per_minute:.1f} files/minute")
        print(f"   {mb_per_minute:.2f} MB/minute")
        
        # Estimate total debates needed
        # Based on test: 1864 had 112 sitting days, avg ~15 debates per day
        # Rough estimate: 203 years * 100 sitting days/year * 15 debates/day = ~300,000 debates
        estimated_total_debates = 300000
        
        if files_per_minute > 0:
            remaining_files = estimated_total_debates - len(html_files)
            remaining_minutes = remaining_files / files_per_minute
            remaining_hours = remaining_minutes / 60
            
            completion_time = datetime.now() + timedelta(minutes=remaining_minutes)
            
            print(f"🔮 Projection:")
            print(f"   Estimated total debates: {estimated_total_debates:,}")
            print(f"   Files remaining: {remaining_files:,}")
            print(f"   Time remaining: {remaining_hours:.1f} hours")
            print(f"   Estimated completion: {completion_time.strftime('%H:%M %p today' if remaining_hours < 24 else '%a %b %d at %H:%M')}")
        
        # Final dataset size estimate
        if total_size_mb > 0:
            avg_file_size = total_size_mb / len(html_files)
            estimated_final_size_gb = (estimated_total_debates * avg_file_size) / 1024
            print(f"💾 Estimated final size: {estimated_final_size_gb:.1f} GB")
    
    elif elapsed_minutes > 5:
        print("🔄 Still in discovery phase (mapping out 200 years of data)")
        print("⏳ File downloads should start soon...")
        
        # Discovery phase estimate
        # Rough: 20 decades * 10 years * 8 months * discovery time
        discovery_requests = 20 * 10 * 8  # ~1600 discovery requests
        if elapsed_minutes > 0:
            discovery_rate = discovery_requests / max(elapsed_minutes, 1)
            print(f"📊 Discovery rate: ~{discovery_rate:.0f} requests/minute")
            
        print("🎯 Files should start appearing in next 5-10 minutes")
    
    else:
        print("⏳ Early in discovery phase, check again in 5 minutes")
    
    # Check if processes are still running
    import subprocess
    try:
        result = subprocess.run(['ps', '-p', '45696', '45700'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            process_count = len([line for line in result.stdout.split('\n') if 'python' in line])
            print(f"✅ {process_count} crawler processes still running")
        else:
            print("⚠️ Crawler processes may have stopped")
    except:
        print("? Could not check process status")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if len(html_files) == 0:
        print("   • Wait 10-15 minutes for discovery to complete")
        print("   • Check again for file growth")
    elif len(html_files) < 100:
        print("   • Downloads starting, check again in 30 minutes")
        print("   • Monitor file count growth")
    else:
        print("   • Good progress! Check every few hours")
        print("   • Can safely leave running overnight")

if __name__ == "__main__":
    analyze_progress()