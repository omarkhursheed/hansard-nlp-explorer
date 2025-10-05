#!/usr/bin/env python3
"""
Detailed forensic analysis of Hansard crawler progress.
"""

import subprocess
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import os

def get_process_info(pid):
    """Get detailed process information."""
    try:
        # Get process details
        ps_result = subprocess.run(['ps', '-p', str(pid), '-o', 'pid,ppid,etime,time,command'], 
                                 capture_output=True, text=True)
        
        if ps_result.returncode == 0:
            lines = ps_result.stdout.strip().split('\n')
            if len(lines) > 1:
                return lines[1].strip()
        return None
    except:
        return None

def get_file_timeline():
    """Analyze file creation timeline."""
    data_dir = Path("data/hansard")
    
    if not data_dir.exists():
        return None
    
    # Get all files with timestamps
    files_info = []
    
    for gz_file in data_dir.rglob("*.gz"):
        try:
            stat_info = gz_file.stat()
            files_info.append({
                'path': str(gz_file),
                'size': stat_info.st_size,
                'created': datetime.fromtimestamp(stat_info.st_ctime),
                'modified': datetime.fromtimestamp(stat_info.st_mtime)
            })
        except:
            continue
    
    return sorted(files_info, key=lambda x: x['created'])

def analyze_by_hour(file_timeline):
    """Analyze download activity by hour."""
    if not file_timeline:
        return {}
    
    hourly_stats = defaultdict(lambda: {'count': 0, 'size': 0})
    
    for file_info in file_timeline:
        hour_key = file_info['created'].strftime('%Y-%m-%d %H:00')
        hourly_stats[hour_key]['count'] += 1
        hourly_stats[hour_key]['size'] += file_info['size']
    
    return dict(hourly_stats)

def check_network_connections():
    """Check current network connections."""
    try:
        # Check for Parliament API connections
        lsof_result = subprocess.run(['sudo', 'lsof', '-p', '45696', '-p', '45700'], 
                                   capture_output=True, text=True)
        
        connections = []
        if lsof_result.returncode == 0:
            for line in lsof_result.stdout.split('\n'):
                if 'TCP' in line and 'ESTABLISHED' in line:
                    connections.append(line.strip())
        
        return connections
    except:
        return []

def estimate_completion():
    """Estimate completion based on current progress."""
    data_dir = Path("data/hansard")
    
    # Count current progress
    html_files = list(data_dir.rglob("*.html.gz"))
    
    # Analyze year coverage
    year_dirs = set()
    for html_file in html_files:
        parts = html_file.parts
        if len(parts) >= 3:
            year_part = parts[2]
            if year_part.isdigit() and 1800 <= int(year_part) <= 2100:
                year_dirs.add(int(year_part))
    
    if not year_dirs:
        return None
    
    years_done = sorted(year_dirs)
    files_per_year = len(html_files) / len(years_done)
    
    # Estimate total
    total_years = 2005 - 1803 + 1  # 203 years
    estimated_total_files = files_per_year * total_years
    
    completion_percentage = len(html_files) / estimated_total_files * 100
    
    return {
        'current_files': len(html_files),
        'years_completed': len(years_done),
        'year_range': f"{min(years_done)}-{max(years_done)}",
        'files_per_year': files_per_year,
        'estimated_total': int(estimated_total_files),
        'completion_pct': completion_percentage,
        'total_years': total_years
    }

def main():
    print("🔍 DETAILED HANSARD CRAWLER FORENSIC ANALYSIS")
    print("=" * 60)
    
    # Check process status
    print("\n📊 PROCESS STATUS:")
    print("-" * 30)
    
    for pid in [45696, 45700]:
        proc_info = get_process_info(pid)
        if proc_info:
            print(f"PID {pid}: {proc_info}")
        else:
            print(f"PID {pid}: NOT RUNNING")
    
    # Check network connections
    print("\n🌐 NETWORK CONNECTIONS:")
    print("-" * 30)
    
    connections = check_network_connections()
    if connections:
        for conn in connections:
            print(f"  {conn}")
    else:
        print("  No active connections found (may need sudo)")
    
    # Analyze file timeline
    print("\n📁 FILE CREATION ANALYSIS:")
    print("-" * 30)
    
    file_timeline = get_file_timeline()
    
    if file_timeline:
        first_file = file_timeline[0]
        last_file = file_timeline[-1]
        total_files = len(file_timeline)
        
        print(f"  First file: {first_file['created'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Last file:  {last_file['created'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Total files: {total_files:,}")
        
        # Calculate duration and rate
        duration = last_file['created'] - first_file['created']
        hours = duration.total_seconds() / 3600
        
        if hours > 0:
            rate_per_hour = total_files / hours
            print(f"  Duration: {hours:.1f} hours")
            print(f"  Rate: {rate_per_hour:.0f} files/hour")
        
        # Hourly breakdown
        print(f"\n⏰ HOURLY DOWNLOAD ACTIVITY:")
        print("-" * 30)
        
        hourly_stats = analyze_by_hour(file_timeline)
        
        for hour in sorted(hourly_stats.keys())[-12:]:  # Last 12 hours
            stats = hourly_stats[hour]
            mb_size = stats['size'] / (1024 * 1024)
            print(f"  {hour}: {stats['count']:4d} files ({mb_size:5.1f} MB)")
        
        # Check for gaps (sleep periods)
        print(f"\n😴 CHECKING FOR GAPS (SLEEP PERIODS):")
        print("-" * 30)
        
        gaps = []
        for i in range(1, len(file_timeline)):
            time_diff = file_timeline[i]['created'] - file_timeline[i-1]['created']
            if time_diff.total_seconds() > 1800:  # 30+ minute gaps
                gaps.append({
                    'start': file_timeline[i-1]['created'],
                    'end': file_timeline[i]['created'],
                    'duration': time_diff
                })
        
        if gaps:
            for gap in gaps[-5:]:  # Show last 5 gaps
                duration_hours = gap['duration'].total_seconds() / 3600
                print(f"  Gap: {gap['start'].strftime('%H:%M')} - {gap['end'].strftime('%H:%M')} ({duration_hours:.1f}h)")
        else:
            print("  No significant gaps found - consistent downloading!")
    
    # Progress estimation
    print(f"\n🎯 COMPLETION ANALYSIS:")
    print("-" * 30)
    
    completion_info = estimate_completion()
    
    if completion_info:
        print(f"  Current files: {completion_info['current_files']:,}")
        print(f"  Years completed: {completion_info['years_completed']} ({completion_info['year_range']})")
        print(f"  Avg files/year: {completion_info['files_per_year']:.0f}")
        print(f"  Estimated total: {completion_info['estimated_total']:,}")
        print(f"  Completion: {completion_info['completion_pct']:.1f}%")
        
        # Time remaining estimate
        if file_timeline and len(file_timeline) > 100:  # Need decent sample
            recent_files = file_timeline[-100:]  # Last 100 files
            recent_duration = recent_files[-1]['created'] - recent_files[0]['created']
            recent_rate = 100 / max(recent_duration.total_seconds() / 3600, 0.1)  # files/hour
            
            files_remaining = completion_info['estimated_total'] - completion_info['current_files']
            hours_remaining = files_remaining / max(recent_rate, 1)
            
            completion_time = datetime.now() + timedelta(hours=hours_remaining)
            
            print(f"  Recent rate: {recent_rate:.0f} files/hour")
            print(f"  Files remaining: {files_remaining:,}")
            print(f"  Estimated time remaining: {hours_remaining:.1f} hours")
            print(f"  Estimated completion: {completion_time.strftime('%a %b %d at %H:%M')}")
    
    # Check current activity
    print(f"\n🔄 CURRENT ACTIVITY CHECK:")
    print("-" * 30)
    
    if file_timeline:
        last_file_age = datetime.now() - file_timeline[-1]['created']
        print(f"  Last file created: {last_file_age.total_seconds() / 60:.1f} minutes ago")
        
        if last_file_age.total_seconds() < 300:  # Less than 5 minutes
            print("  ✅ Still actively downloading!")
        elif last_file_age.total_seconds() < 1800:  # Less than 30 minutes
            print("  ⏳ Recent activity, probably still working")
        else:
            print("  ⚠️ No recent file creation - may have stopped")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 30)
    
    if completion_info and completion_info['completion_pct'] > 20:
        print("  • Crawlers are making good progress")
        print("  • Can safely leave running")
        if file_timeline:
            last_file_age = datetime.now() - file_timeline[-1]['created']
            if last_file_age.total_seconds() < 600:
                print("  • Still actively downloading")
            else:
                print("  • Check if processes are still active")
    else:
        print("  • Early stages, monitor for continued progress")
    
    print(f"\n" + "=" * 60)

if __name__ == "__main__":
    main()