#!/usr/bin/env python3
"""
Enhanced Hansard parser for 1803 November data
"""

import gzip
import glob
import os
import random
from bs4 import BeautifulSoup

def parse_hansard_file(file_path):
    """Parse a single Hansard file and extract content"""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            html = f.read()
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Basic info
        title = soup.find('title').get_text() if soup.find('title') else "No title"
        
        # Extract the main content - try both Commons and Lords sitting divs
        content_div = soup.find('div', class_='house-of-commons-sitting')
        if not content_div:
            content_div = soup.find('div', class_='house-of-lords-sitting')
        
        if content_div:
            # Remove navigation and footer elements
            for unwanted in content_div(['nav', 'footer', 'script', 'style']):
                unwanted.decompose()
            
            content_text = content_div.get_text(separator='\n', strip=True)
            lines = [line for line in content_text.split('\n') if line.strip()]
            
            # Determine type based on content
            sitting_type = "Commons" if soup.find('div', class_='house-of-commons-sitting') else "Lords"
            
            return {
                'file': os.path.basename(file_path),
                'title': title,
                'sitting_type': sitting_type,
                'lines': lines,
                'line_count': len(lines),
                'success': True
            }
        else:
            return {
                'file': os.path.basename(file_path),
                'title': title,
                'lines': [],
                'line_count': 0,
                'success': False,
                'error': 'No main content div found (neither Commons nor Lords)'
            }
            
    except Exception as e:
        return {
            'file': os.path.basename(file_path),
            'success': False,
            'error': str(e)
        }

def sample_files_from_years(base_path, sample_years, files_per_year=3):
    """Sample random files from specified years"""
    sampled_files = []
    
    for year in sample_years:
        year_path = os.path.join(base_path, str(year))
        if os.path.exists(year_path):
            # Get all .html.gz files from all months in this year
            all_files = []
            for month_dir in os.listdir(year_path):
                month_path = os.path.join(year_path, month_dir)
                if os.path.isdir(month_path):
                    month_files = glob.glob(os.path.join(month_path, "*.html.gz"))
                    all_files.extend(month_files)
            
            # Sample randomly
            if all_files:
                sample_size = min(files_per_year, len(all_files))
                sampled = random.sample(all_files, sample_size)
                sampled_files.extend(sampled)
                print(f"Year {year}: Found {len(all_files)} files, sampled {len(sampled)}")
            else:
                print(f"Year {year}: No files found")
        else:
            print(f"Year {year}: Directory not found")
    
    return sampled_files

def main():
    # Test across different decades
    sample_years = [
        1803, 1850, 1900,  # 19th century
        1920, 1950, 1980,  # 20th century  
        2000, 2005         # 21st century
    ]
    
    base_path = "../data/hansard/"
    print(f"Testing parser across years: {sample_years}")
    print("=" * 60)
    
    # Get sample files
    sampled_files = sample_files_from_years(base_path, sample_years, files_per_year=3)
    
    if not sampled_files:
        print("No files found - falling back to 1803/nov test")
        nov_path = "../../data/hansard/1803/nov/"
        sampled_files = glob.glob(os.path.join(nov_path, "*.html.gz"))
    
    print(f"\nTotal files to test: {len(sampled_files)}")
    print("=" * 60)
    
    # Test parser on all sampled files
    results_by_year = {}
    total_lines = 0
    successful_files = 0
    failed_files = 0
    commons_files = 0
    lords_files = 0
    
    for file_path in sampled_files:
        # Extract year from path
        year = file_path.split('/')[-3] if len(file_path.split('/')) > 3 else "unknown"
        
        if year not in results_by_year:
            results_by_year[year] = {'success': 0, 'failed': 0, 'lines': 0}
        
        result = parse_hansard_file(file_path)
        
        if result['success']:
            successful_files += 1
            total_lines += result['line_count']
            results_by_year[year]['success'] += 1
            results_by_year[year]['lines'] += result['line_count']
            
            if result['sitting_type'] == 'Commons':
                commons_files += 1
            else:
                lords_files += 1
            
            print(f"\n✓ {year}/{result['file']} [{result['sitting_type']}]")
            print(f"  Lines: {result['line_count']}")
            
            # Show preview for interesting files
            if result['lines'] and result['line_count'] > 20:
                print("  Preview:")
                for i, line in enumerate(result['lines'][:2]):
                    print(f"    {line[:60]}{'...' if len(line) > 60 else ''}")
        else:
            failed_files += 1
            results_by_year[year]['failed'] += 1
            print(f"\n✗ {year}/{result['file']} - {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("DETAILED RESULTS BY YEAR:")
    for year in sorted(results_by_year.keys()):
        data = results_by_year[year]
        total = data['success'] + data['failed']
        success_rate = (data['success'] / total * 100) if total > 0 else 0
        avg_lines = (data['lines'] / data['success']) if data['success'] > 0 else 0
        print(f"  {year}: {data['success']}/{total} success ({success_rate:.1f}%), avg {avg_lines:.1f} lines")
    
    print("\n" + "=" * 60)
    print(f"OVERALL SUMMARY:")
    print(f"  Successful files: {successful_files}")
    print(f"    - Commons files: {commons_files}")
    print(f"    - Lords files: {lords_files}")
    print(f"  Failed files: {failed_files}")
    success_rate = (successful_files / (successful_files + failed_files) * 100) if (successful_files + failed_files) > 0 else 0
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Total lines extracted: {total_lines:,}")
    print(f"  Average lines per successful file: {total_lines/successful_files if successful_files > 0 else 0:.1f}")

if __name__ == "__main__":
    main()
