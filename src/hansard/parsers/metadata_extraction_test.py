#!/usr/bin/env python3
"""
Comprehensive test to extract metadata and verify parser robustness
"""

import gzip
import glob
import os
import random
import re
from bs4 import BeautifulSoup

def extract_comprehensive_metadata(file_path):
    """Extract all possible metadata from a Hansard file"""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            html = f.read()
        
        soup = BeautifulSoup(html, 'html.parser')
        
        metadata = {
            'file': os.path.basename(file_path),
            'path': file_path,
            'success': False,
            'error': None
        }
        
        # Basic HTML metadata
        title = soup.find('title')
        metadata['title'] = title.get_text() if title else "No title"
        
        # Meta tags
        meta_tags = {}
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property') or meta.get('content')
            content = meta.get('content')
            if name and content:
                meta_tags[name] = content
        metadata['meta_tags'] = meta_tags
        
        # Try both Commons and Lords sitting divs
        content_div = soup.find('div', class_='house-of-commons-sitting')
        sitting_type = "Commons"
        if not content_div:
            content_div = soup.find('div', class_='house-of-lords-sitting')
            sitting_type = "Lords"
        
        if not content_div:
            metadata['error'] = 'No main content div found'
            return metadata
        
        metadata['sitting_type'] = sitting_type
        
        # Extract content
        for unwanted in content_div(['nav', 'footer', 'script', 'style']):
            unwanted.decompose()
        
        content_text = content_div.get_text(separator='\n', strip=True)
        lines = [line for line in content_text.split('\n') if line.strip()]
        
        metadata['lines'] = lines
        metadata['line_count'] = len(lines)
        
        # Extract Hansard reference (HC/HL Deb date vol cc)
        hansard_ref = None
        if lines:
            first_line = lines[0] if len(lines) > 0 else ""
            # Look for pattern like "HC Deb 22 November 1803 vol 1 cc13-31"
            ref_match = re.search(r'(HC|HL) Deb (\d{1,2} \w+ \d{4}) vol (\d+) cc?(\d+(?:-\d+)?)', first_line)
            if ref_match:
                hansard_ref = {
                    'chamber': ref_match.group(1),
                    'date': ref_match.group(2),
                    'volume': ref_match.group(3),
                    'columns': ref_match.group(4)
                }
        metadata['hansard_reference'] = hansard_ref
        
        # Look for speaker information
        speakers = []
        for line in lines[1:10]:  # Check first few lines for speakers
            # Look for patterns like "Mr. Name", "The Chancellor", etc.
            speaker_patterns = [
                r'^(Mr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'^(The\s+[A-Z][a-z]+(?:\s+of\s+[A-Z][a-z]+)*)',
                r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$'
            ]
            for pattern in speaker_patterns:
                match = re.match(pattern, line.strip())
                if match and len(match.group(1)) < 50:  # Reasonable speaker name length
                    speakers.append(match.group(1))
                    break
        
        metadata['speakers'] = list(set(speakers))  # Remove duplicates
        
        # Extract any debate topics or bill names from title
        debate_topics = []
        title_text = metadata['title']
        # Look for patterns like "BANK RESTRICTION BILL", "INCOME TAX", etc.
        topic_matches = re.findall(r'([A-Z][A-Z\s-]+)\.?—', title_text)
        for topic in topic_matches:
            if len(topic.strip()) > 3:  # Filter out short matches
                debate_topics.append(topic.strip())
        
        metadata['debate_topics'] = debate_topics
        
        # Check for any special sections or divisions
        special_sections = []
        for line in lines:
            if any(marker in line.lower() for marker in ['division', '§', 'ordered', 'question put']):
                special_sections.append(line.strip()[:100])  # First 100 chars
        
        metadata['special_sections'] = special_sections[:5]  # Limit to 5
        
        metadata['success'] = True
        return metadata
        
    except Exception as e:
        return {
            'file': os.path.basename(file_path),
            'path': file_path,
            'success': False,
            'error': str(e)
        }

def main():
    # Test on larger sample with more years
    sample_years = [1803, 1820, 1840, 1860, 1880, 1900, 1920, 1940, 1960, 1980, 2000, 2005]
    base_path = "../data/hansard/"
    
    print(f"Comprehensive metadata extraction test")
    print(f"Testing years: {sample_years}")
    print("=" * 80)
    
    all_files = []
    for year in sample_years:
        year_path = os.path.join(base_path, str(year))
        if os.path.exists(year_path):
            year_files = []
            for month_dir in os.listdir(year_path):
                month_path = os.path.join(year_path, month_dir)
                if os.path.isdir(month_path):
                    month_files = glob.glob(os.path.join(month_path, "*.html.gz"))
                    year_files.extend(month_files)
            
            if year_files:
                # Sample more files per year
                sample_size = min(5, len(year_files))
                sampled = random.sample(year_files, sample_size)
                all_files.extend(sampled)
                print(f"Year {year}: Found {len(year_files)} files, sampled {len(sampled)}")
    
    print(f"\nTotal files to test: {len(all_files)}")
    print("=" * 80)
    
    # Process all files
    results = []
    successful = 0
    failed = 0
    
    for file_path in all_files:
        metadata = extract_comprehensive_metadata(file_path)
        results.append(metadata)
        
        year = file_path.split('/')[-3] if len(file_path.split('/')) > 3 else "unknown"
        
        if metadata['success']:
            successful += 1
            print(f"\n✓ {year}/{metadata['file']} [{metadata.get('sitting_type', 'Unknown')}]")
            print(f"  Lines: {metadata['line_count']}")
            
            if metadata.get('hansard_reference'):
                ref = metadata['hansard_reference']
                print(f"  Reference: {ref['chamber']} Deb {ref['date']} vol {ref['volume']} cc{ref['columns']}")
            
            if metadata.get('debate_topics'):
                print(f"  Topics: {', '.join(metadata['debate_topics'][:3])}")
            
            if metadata.get('speakers'):
                print(f"  Speakers: {', '.join(metadata['speakers'][:3])}")
            
            if metadata.get('meta_tags'):
                interesting_meta = {k: v for k, v in metadata['meta_tags'].items() 
                                  if k in ['keywords', 'description', 'author']}
                if interesting_meta:
                    print(f"  Meta: {interesting_meta}")
        else:
            failed += 1
            print(f"\n✗ {year}/{metadata['file']} - {metadata.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80)
    print("METADATA ANALYSIS:")
    
    # Analyze extracted metadata
    metadata_stats = {
        'has_hansard_ref': 0,
        'has_speakers': 0,
        'has_topics': 0,
        'has_meta_tags': 0,
        'chambers': {'Commons': 0, 'Lords': 0},
        'avg_lines': 0,
        'total_lines': 0
    }
    
    for result in results:
        if result['success']:
            metadata_stats['total_lines'] += result['line_count']
            if result.get('hansard_reference'):
                metadata_stats['has_hansard_ref'] += 1
            if result.get('speakers'):
                metadata_stats['has_speakers'] += 1
            if result.get('debate_topics'):
                metadata_stats['has_topics'] += 1
            if result.get('meta_tags'):
                metadata_stats['has_meta_tags'] += 1
            
            sitting_type = result.get('sitting_type', 'Unknown')
            if sitting_type in metadata_stats['chambers']:
                metadata_stats['chambers'][sitting_type] += 1
    
    if successful > 0:
        metadata_stats['avg_lines'] = metadata_stats['total_lines'] / successful
        
        print(f"  Files with Hansard references: {metadata_stats['has_hansard_ref']}/{successful} ({metadata_stats['has_hansard_ref']/successful*100:.1f}%)")
        print(f"  Files with speaker info: {metadata_stats['has_speakers']}/{successful} ({metadata_stats['has_speakers']/successful*100:.1f}%)")
        print(f"  Files with debate topics: {metadata_stats['has_topics']}/{successful} ({metadata_stats['has_topics']/successful*100:.1f}%)")
        print(f"  Files with meta tags: {metadata_stats['has_meta_tags']}/{successful} ({metadata_stats['has_meta_tags']/successful*100:.1f}%)")
    else:
        print(f"  No successful files to analyze!")
    print(f"  Chamber distribution: Commons {metadata_stats['chambers']['Commons']}, Lords {metadata_stats['chambers']['Lords']}")
    
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY:")
    print(f"  Successful files: {successful}")
    print(f"  Failed files: {failed}")
    total_files = successful + failed
    success_rate = (successful/total_files*100) if total_files > 0 else 0
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Total lines extracted: {metadata_stats['total_lines']:,}")
    print(f"  Average lines per file: {metadata_stats['avg_lines']:.1f}")

if __name__ == "__main__":
    main()