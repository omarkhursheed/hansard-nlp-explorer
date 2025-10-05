#!/usr/bin/env python3
"""
Dynamic Hansard Progress Estimator - Always uses current data patterns.
No hardcoded estimates - everything calculated from actual progress.
"""

import time
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics

class DynamicHansardEstimator:
    def __init__(self, data_dir: Path = Path("data/hansard")):
        self.data_dir = data_dir
        self.total_year_range = (1803, 2005)  # Known target range
        
    def get_current_data(self):
        """Analyze all current data to build accurate estimates."""
        if not self.data_dir.exists():
            return None
            
        html_files = list(self.data_dir.rglob("*.html.gz"))
        
        if not html_files:
            return None
            
        # Analyze files by year
        files_by_year = defaultdict(int)
        file_times = []
        
        for f in html_files:
            try:
                # Extract year from path: data/hansard/YEAR/month/file
                year = int(f.parts[2])
                files_by_year[year] += 1
                
                # Get file creation time
                file_times.append(datetime.fromtimestamp(f.stat().st_mtime))
            except (IndexError, ValueError):
                continue
        
        if not files_by_year:
            return None
            
        # Calculate statistics
        years_completed = sorted(files_by_year.keys())
        total_files = len(html_files)
        total_size_mb = sum(f.stat().st_size for f in html_files) / (1024**2)
        
        return {
            'total_files': total_files,
            'total_size_mb': total_size_mb,
            'files_by_year': dict(files_by_year),
            'years_completed': years_completed,
            'file_times': sorted(file_times),
            'year_range': (min(years_completed), max(years_completed))
        }
    
    def estimate_total_files(self, data):
        """Estimate total files based on actual patterns from completed years."""
        files_by_year = data['files_by_year']
        years_completed = data['years_completed']
        
        if len(years_completed) < 5:
            # Not enough data yet, use simple average
            avg_files_per_year = data['total_files'] / len(years_completed)
            return avg_files_per_year * (self.total_year_range[1] - self.total_year_range[0] + 1)
        
        # Multiple estimation approaches for robustness
        estimates = {}
        
        # 1. Simple average of all completed years
        files_per_year = list(files_by_year.values())
        estimates['simple_avg'] = statistics.mean(files_per_year) * (self.total_year_range[1] - self.total_year_range[0] + 1)
        
        # 2. Median (more robust to outliers)
        estimates['median'] = statistics.median(files_per_year) * (self.total_year_range[1] - self.total_year_range[0] + 1)
        
        # 3. Recent years average (last 20 years of data)
        recent_years = sorted(years_completed)[-20:]
        recent_files = [files_by_year[year] for year in recent_years]
        estimates['recent_avg'] = statistics.mean(recent_files) * (self.total_year_range[1] - self.total_year_range[0] + 1)
        
        # 4. Weighted average favoring more recent years
        weights = []
        values = []
        for i, year in enumerate(sorted(years_completed)):
            weight = 1 + (i / len(years_completed))  # More recent years get higher weight
            weights.append(weight)
            values.append(files_by_year[year])
        
        weighted_avg = sum(w * v for w, v in zip(weights, values)) / sum(weights)
        estimates['weighted'] = weighted_avg * (self.total_year_range[1] - self.total_year_range[0] + 1)
        
        # 5. Historical trend analysis
        if len(years_completed) > 10:
            # Look for trends over time
            early_years = sorted(years_completed)[:len(years_completed)//2]
            late_years = sorted(years_completed)[len(years_completed)//2:]
            
            early_avg = statistics.mean(files_by_year[year] for year in early_years)
            late_avg = statistics.mean(files_by_year[year] for year in late_years)
            
            # If there's a trend, project it
            trend_factor = late_avg / early_avg if early_avg > 0 else 1
            estimates['trend_adjusted'] = late_avg * trend_factor * (self.total_year_range[1] - self.total_year_range[0] + 1)
        
        # Return consensus estimate (median of all estimates)
        all_estimates = list(estimates.values())
        consensus = statistics.median(all_estimates)
        
        return {
            'consensus': consensus,
            'estimates': estimates,
            'files_per_year_stats': {
                'min': min(files_per_year),
                'max': max(files_per_year),
                'mean': statistics.mean(files_per_year),
                'median': statistics.median(files_per_year),
                'std': statistics.stdev(files_per_year) if len(files_per_year) > 1 else 0
            }
        }
    
    def calculate_download_rates(self, file_times):
        """Calculate various download rates from file timestamps."""
        if len(file_times) < 10:
            return None
            
        rates = {}
        
        # Overall rate
        total_duration = file_times[-1] - file_times[0]
        total_hours = total_duration.total_seconds() / 3600
        if total_hours > 0:
            rates['overall'] = len(file_times) / total_hours
        
        # Recent rate (last 500 files)
        if len(file_times) > 500:
            recent_files = file_times[-500:]
            recent_duration = recent_files[-1] - recent_files[0]
            recent_hours = recent_duration.total_seconds() / 3600
            if recent_hours > 0:
                rates['recent_500'] = 500 / recent_hours
        
        # Very recent rate (last 100 files)
        if len(file_times) > 100:
            very_recent = file_times[-100:]
            very_recent_duration = very_recent[-1] - very_recent[0]
            very_recent_hours = very_recent_duration.total_seconds() / 3600
            if very_recent_hours > 0:
                rates['recent_100'] = 100 / very_recent_hours
        
        # Last hour rate
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_hour_files = [t for t in file_times if t > one_hour_ago]
        if recent_hour_files:
            rates['last_hour'] = len(recent_hour_files)
        
        # Last 10 minutes rate (extrapolated to hourly)
        ten_min_ago = datetime.now() - timedelta(minutes=10)
        recent_10min_files = [t for t in file_times if t > ten_min_ago]
        if recent_10min_files:
            rates['last_10min_extrapolated'] = len(recent_10min_files) * 6  # * 6 to get hourly rate
        
        return rates
    
    def get_active_processes(self):
        """Check if crawlers are still running."""
        try:
            result = subprocess.run(['pgrep', '-f', 'crawler'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                pids = [pid.strip() for pid in result.stdout.split('\n') if pid.strip()]
                return len(pids)
            return 0
        except:
            return None
    
    def generate_estimate(self):
        """Generate comprehensive progress estimate."""
        print("🎯 DYNAMIC HANSARD PROGRESS ESTIMATE")
        print("=" * 60)
        print(f"📅 Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get current data
        data = self.get_current_data()
        
        if not data:
            print("❌ No data found - crawler may be starting up")
            return
        
        # Basic stats
        print(f"\n📊 CURRENT STATUS:")
        print(f"   📁 Total files downloaded: {data['total_files']:,}")
        print(f"   💾 Total size: {data['total_size_mb']:.1f} MB")
        print(f"   📅 Years covered: {data['year_range'][0]}-{data['year_range'][1]} ({len(data['years_completed'])} years)")
        
        # Process status
        active_processes = self.get_active_processes()
        if active_processes is not None:
            if active_processes > 0:
                print(f"   🔄 Active crawler processes: {active_processes}")
            else:
                print(f"   ⚠️ No active crawler processes detected")
        
        # File activity
        if data['file_times']:
            last_file_time = data['file_times'][-1]
            minutes_since = (datetime.now() - last_file_time).total_seconds() / 60
            print(f"   🕐 Last file created: {minutes_since:.1f} minutes ago")
            
            if minutes_since < 2:
                print("   ✅ Currently downloading!")
            elif minutes_since < 15:
                print("   🔶 Recent activity")
            else:
                print("   ⚠️ No recent file creation")
        
        # Estimate total files needed
        print(f"\n🔮 TOTAL ESTIMATE ANALYSIS:")
        total_estimate = self.estimate_total_files(data)
        
        print(f"   📈 Files per year statistics:")
        stats = total_estimate['files_per_year_stats']
        print(f"      • Average: {stats['mean']:.0f} files/year")
        print(f"      • Median: {stats['median']:.0f} files/year")
        print(f"      • Range: {stats['min']:.0f} - {stats['max']:.0f} files/year")
        if stats['std'] > 0:
            print(f"      • Std deviation: {stats['std']:.0f}")
        
        print(f"\n   🎯 Total dataset estimates:")
        for method, estimate in total_estimate['estimates'].items():
            print(f"      • {method.replace('_', ' ').title()}: {estimate:,.0f} files")
        
        consensus = total_estimate['consensus']
        print(f"   🏆 Consensus estimate: {consensus:,.0f} files")
        
        # Calculate completion percentage
        completion_pct = (data['total_files'] / consensus) * 100
        print(f"   🎯 Current completion: {completion_pct:.1f}%")
        
        # Download rate analysis
        print(f"\n⚡ DOWNLOAD RATE ANALYSIS:")
        rates = self.calculate_download_rates(data['file_times'])
        
        if rates:
            print(f"   📊 Download rates (files/hour):")
            if 'overall' in rates:
                print(f"      • Overall average: {rates['overall']:.0f}")
            if 'recent_500' in rates:
                print(f"      • Recent 500 files: {rates['recent_500']:.0f}")
            if 'recent_100' in rates:
                print(f"      • Recent 100 files: {rates['recent_100']:.0f}")
            if 'last_hour' in rates:
                print(f"      • Last hour: {rates['last_hour']:.0f}")
            if 'last_10min_extrapolated' in rates:
                print(f"      • Last 10min (extrapolated): {rates['last_10min_extrapolated']:.0f}")
            
            # Use most appropriate rate for projection
            if 'recent_100' in rates and rates['recent_100'] > 0:
                current_rate = rates['recent_100']
                rate_type = "recent 100 files"
            elif 'recent_500' in rates and rates['recent_500'] > 0:
                current_rate = rates['recent_500']
                rate_type = "recent 500 files"
            elif 'overall' in rates and rates['overall'] > 0:
                current_rate = rates['overall']
                rate_type = "overall average"
            else:
                current_rate = None
                rate_type = None
            
            if current_rate:
                print(f"   🎯 Using {rate_type} rate: {current_rate:.0f} files/hour")
                
                # Time remaining estimate
                files_remaining = consensus - data['total_files']
                hours_remaining = files_remaining / current_rate
                
                completion_time = datetime.now() + timedelta(hours=hours_remaining)
                
                print(f"\n🏁 COMPLETION PROJECTION:")
                print(f"   📋 Files remaining: {files_remaining:,.0f}")
                print(f"   ⏱️ Hours remaining: {hours_remaining:.1f}")
                print(f"   📅 Estimated completion: {completion_time.strftime('%A %B %d at %H:%M')}")
                
                # Final size estimate
                if data['total_size_mb'] > 0:
                    avg_file_size_mb = data['total_size_mb'] / data['total_files']
                    final_size_gb = (consensus * avg_file_size_mb) / 1024
                    print(f"   💾 Estimated final size: {final_size_gb:.1f} GB")
        
        # Confidence indicators
        print(f"\n📏 ESTIMATE CONFIDENCE:")
        years_completed = len(data['years_completed'])
        total_years = self.total_year_range[1] - self.total_year_range[0] + 1
        
        confidence_pct = min(100, (years_completed / 50) * 100)  # Full confidence at 50 years
        print(f"   📊 Data coverage: {years_completed}/{total_years} years ({years_completed/total_years*100:.1f}%)")
        print(f"   🎯 Estimate confidence: {confidence_pct:.0f}%")
        
        if confidence_pct < 30:
            print("   ⚠️ Early stages - estimates may change significantly")
        elif confidence_pct < 70:
            print("   🔶 Moderate confidence - estimates should stabilize")
        else:
            print("   ✅ High confidence - estimates are reliable")

def main():
    estimator = DynamicHansardEstimator()
    estimator.generate_estimate()

if __name__ == "__main__":
    main()