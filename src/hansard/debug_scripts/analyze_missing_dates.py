#!/usr/bin/env python3
"""Analyze existing Hansard data to identify missing single-digit dates."""

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

def analyze_existing_data(data_dir: Path) -> Dict[str, any]:
    """Analyze the existing Hansard data structure to find missing single-digit dates."""
    
    print(f"Analyzing data in: {data_dir}")
    
    # Track what we have vs what might be missing
    existing_dates = set()
    years_with_data = set()
    months_with_data = defaultdict(set)  # year -> set of months
    potential_missing = []
    
    # Walk through the data directory structure
    for year_dir in data_dir.iterdir():
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
            
        year = year_dir.name
        years_with_data.add(year)
        
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir():
                continue
                
            month = month_dir.name
            months_with_data[year].add(month)
            
            # Find all day files in this month
            day_files = []
            for file_path in month_dir.iterdir():
                if file_path.name.endswith('_summary.json'):
                    # Extract day from filename like "15_summary.json"
                    day_match = re.match(r'^(\d+)_summary\.json$', file_path.name)
                    if day_match:
                        day = day_match.group(1)
                        existing_dates.add(f"{year}/{month}/{day}")
                        day_files.append(int(day))
            
            # Check for missing single-digit days in this month
            day_files.sort()
            single_digit_days = [d for d in day_files if d < 10]
            double_digit_days = [d for d in day_files if d >= 10]
            
            # If we have double-digit days but are missing some single-digit days,
            # those are likely missing due to the bug
            if double_digit_days and len(single_digit_days) < 9:
                # Check which single digits are missing (1-9)
                present_single = set(single_digit_days)
                for potential_day in range(1, 10):
                    if potential_day not in present_single:
                        potential_missing.append(f"{year}/{month}/{potential_day}")
    
    # Summary statistics
    total_existing = len(existing_dates)
    total_years = len(years_with_data)
    total_months = sum(len(months) for months in months_with_data.values())
    
    # Group potential missing by year/month for easier processing
    missing_by_month = defaultdict(list)
    for date_str in potential_missing:
        year, month, day = date_str.split('/')
        missing_by_month[f"{year}/{month}"].append(int(day))
    
    # Sort missing days within each month
    for year_month in missing_by_month:
        missing_by_month[year_month].sort()
    
    results = {
        'total_existing_dates': total_existing,
        'total_years': total_years,
        'total_months': total_months,
        'years_with_data': sorted(years_with_data),
        'potential_missing_dates': sorted(potential_missing),
        'missing_by_month': dict(missing_by_month),
        'summary': {
            'total_potential_missing': len(potential_missing),
            'months_with_potential_missing': len(missing_by_month)
        }
    }
    
    return results

def print_analysis_report(results: Dict[str, any]) -> None:
    """Print a formatted analysis report."""
    
    print("\n" + "="*60)
    print("HANSARD DATA ANALYSIS REPORT")
    print("="*60)
    
    print(f"\n📊 Current Data Summary:")
    print(f"  • Total existing dates: {results['total_existing_dates']:,}")
    print(f"  • Years with data: {results['total_years']} ({min(results['years_with_data'])} - {max(results['years_with_data'])})")
    print(f"  • Total month directories: {results['total_months']:,}")
    
    print(f"\n🔍 Missing Single-Digit Dates Analysis:")
    print(f"  • Potential missing dates: {results['summary']['total_potential_missing']:,}")
    print(f"  • Months with missing dates: {results['summary']['months_with_potential_missing']:,}")
    
    if results['potential_missing_dates']:
        print(f"\n📅 Sample Missing Dates:")
        # Show first 10 missing dates as examples
        for date in results['potential_missing_dates'][:10]:
            print(f"    - {date}")
        if len(results['potential_missing_dates']) > 10:
            print(f"    ... and {len(results['potential_missing_dates']) - 10} more")
    
    print(f"\n📈 Months with Most Missing Dates:")
    # Sort months by number of missing dates
    month_counts = [(month, len(days)) for month, days in results['missing_by_month'].items()]
    month_counts.sort(key=lambda x: x[1], reverse=True)
    
    for month, count in month_counts[:10]:
        days_str = ', '.join(map(str, results['missing_by_month'][month]))
        print(f"    - {month}: {count} missing days ({days_str})")
    
    if len(month_counts) > 10:
        print(f"    ... and {len(month_counts) - 10} more months")

def save_missing_dates_list(results: Dict[str, any], output_file: Path) -> None:
    """Save the list of missing dates to a file for the backfill script."""
    
    missing_dates = results['potential_missing_dates']
    
    # Create output in format suitable for backfill script
    backfill_data = {
        'analysis_summary': results['summary'],
        'missing_dates': missing_dates,
        'missing_by_month': results['missing_by_month'],
        'generated_at': None  # Will be set by caller
    }
    
    with open(output_file, 'w') as f:
        json.dump(backfill_data, f, indent=2)
    
    print(f"\n💾 Missing dates list saved to: {output_file}")
    print(f"   Use this file with the backfill script to crawl only missing dates.")

def main():
    """Main analysis function."""
    
    # Path to your Hansard data
    data_dir = Path("/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/data/hansard")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    print("🔍 Starting analysis of existing Hansard data...")
    
    # Analyze existing data
    results = analyze_existing_data(data_dir)
    
    # Print report
    print_analysis_report(results)
    
    # Save missing dates for backfill
    output_file = Path("/tmp/hansard_missing_dates.json")
    save_missing_dates_list(results, output_file)
    
    print(f"\n✅ Analysis complete!")
    print(f"\nNext steps:")
    print(f"1. Review the missing dates above")
    print(f"2. Run the backfill script with: python backfill_missing_dates.py")
    print(f"3. The backfill will only crawl the {results['summary']['total_potential_missing']} missing dates")

if __name__ == "__main__":
    main()