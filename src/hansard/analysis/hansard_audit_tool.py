#!/usr/bin/env python3
"""
Hansard Data Audit Tool

Examines the processed Hansard debate data to understand content, structure, and quality.
This tool provides minimal processing - just loading and displaying raw examples 
to help understand what we're working with before doing analysis.

Usage:
    python hansard_audit_tool.py --year 1925                    # Audit single year
    python hansard_audit_tool.py --years 1920-1930            # Audit year range  
    python hansard_audit_tool.py --sample 10 --year 1925      # Show 10 examples from 1925
    python hansard_audit_tool.py --stats --years 1920-1950    # Show statistics only
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
import random
from collections import Counter, defaultdict

class HansardAuditor:
    def __init__(self, data_dir="data/processed_fixed"):
        self.data_dir = Path(data_dir)
        
    def list_available_years(self):
        """List all years with processed data"""
        content_dir = self.data_dir / "content"
        if not content_dir.exists():
            return []
        
        years = []
        for year_dir in content_dir.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                jsonl_file = year_dir / f"debates_{year_dir.name}.jsonl"
                if jsonl_file.exists():
                    years.append(int(year_dir.name))
        
        return sorted(years)
    
    def load_debates_from_year(self, year, max_count=None):
        """Load debates from a specific year"""
        jsonl_path = self.data_dir / "content" / str(year) / f"debates_{year}.jsonl"
        
        if not jsonl_path.exists():
            print(f"No data found for year {year}")
            return []
        
        debates = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_count and i >= max_count:
                        break
                    
                    if line.strip():
                        debate = json.loads(line)
                        debates.append(debate)
                        
        except Exception as e:
            print(f"Error loading {year}: {e}")
            return []
        
        return debates
    
    def show_debate_examples(self, debates, num_examples=5):
        """Display example debates with key information"""
        if not debates:
            print("No debates to display")
            return
        
        # Sample random debates if we have more than requested
        if len(debates) > num_examples:
            sample_debates = random.sample(debates, num_examples)
        else:
            sample_debates = debates
        
        print(f"\n=== DEBATE EXAMPLES ({len(sample_debates)} of {len(debates)}) ===")
        
        for i, debate in enumerate(sample_debates, 1):
            print(f"\n--- Example {i} ---")
            
            # Basic info
            metadata = debate.get('metadata', {})
            print(f"File: {debate.get('file_path', 'Unknown')}")
            print(f"Title: {metadata.get('title', 'No title')[:100]}")  
            print(f"Chamber: {metadata.get('chamber', 'Unknown')}")
            print(f"Word Count: {metadata.get('word_count', 'Unknown')}")
            print(f"Speakers: {len(metadata.get('speakers', []))}")
            
            # Date info
            if 'reference_date' in metadata:
                print(f"Date: {metadata['reference_date']}")
            
            # Text preview
            full_text = debate.get('full_text', '')
            if full_text:
                # Show first 500 characters
                preview = full_text.strip()[:500]
                if len(full_text) > 500:
                    preview += "..."
                print(f"Text Preview:\n{preview}")
            else:
                print("Text: [NO TEXT]")
            
            print("-" * 50)
    
    def analyze_year_stats(self, year):
        """Analyze statistics for a specific year"""
        debates = self.load_debates_from_year(year)
        
        if not debates:
            return None
        
        stats = {
            'year': year,
            'total_debates': len(debates),
            'chambers': Counter(),
            'word_counts': [],
            'speaker_counts': [],
            'has_text': 0,
            'no_text': 0,
            'titles': [],
            'file_extensions': Counter()
        }
        
        for debate in debates:
            metadata = debate.get('metadata', {})
            
            # Chamber distribution
            chamber = metadata.get('chamber', 'Unknown')
            stats['chambers'][chamber] += 1
            
            # Word count distribution
            word_count = metadata.get('word_count', 0)
            if isinstance(word_count, (int, float)) and word_count > 0:
                stats['word_counts'].append(word_count)
            
            # Speaker counts
            speakers = metadata.get('speakers', [])
            stats['speaker_counts'].append(len(speakers))
            
            # Text presence
            full_text = debate.get('full_text', '')
            if full_text and full_text.strip():
                stats['has_text'] += 1
            else:
                stats['no_text'] += 1
            
            # Title collection (for pattern analysis)
            title = metadata.get('title', '')
            if title:
                stats['titles'].append(title)
            
            # File extension analysis  
            file_path = debate.get('file_path', '')
            if file_path:
                ext = Path(file_path).suffix.lower()
                stats['file_extensions'][ext] += 1
        
        return stats
    
    def print_year_stats(self, stats):
        """Print formatted statistics for a year"""
        if not stats:
            return
        
        print(f"\n=== STATISTICS FOR {stats['year']} ===")
        print(f"Total debates: {stats['total_debates']:,}")
        print(f"Debates with text: {stats['has_text']:,} ({stats['has_text']/stats['total_debates']:.1%})")
        print(f"Debates without text: {stats['no_text']:,} ({stats['no_text']/stats['total_debates']:.1%})")
        
        # Chamber breakdown
        print(f"\nChamber distribution:")
        for chamber, count in stats['chambers'].most_common():
            pct = count / stats['total_debates'] * 100
            print(f"  {chamber}: {count:,} ({pct:.1f}%)")
        
        # Word count stats
        if stats['word_counts']:
            word_counts = stats['word_counts']
            print(f"\nWord count statistics:")
            print(f"  Mean: {sum(word_counts)/len(word_counts):.0f}")
            print(f"  Min: {min(word_counts):,}")
            print(f"  Max: {max(word_counts):,}")
            
            # Word count ranges
            ranges = [(0, 100), (100, 500), (500, 1000), (1000, 5000), (5000, float('inf'))]
            print(f"  Word count ranges:")
            for low, high in ranges:
                if high == float('inf'):
                    count = sum(1 for wc in word_counts if wc >= low)
                    print(f"    {low:,}+: {count} debates")
                else:
                    count = sum(1 for wc in word_counts if low <= wc < high) 
                    print(f"    {low:,}-{high:,}: {count} debates")
        
        # Speaker stats
        if stats['speaker_counts']:
            speaker_counts = stats['speaker_counts']
            print(f"\nSpeaker statistics:")
            print(f"  Mean speakers per debate: {sum(speaker_counts)/len(speaker_counts):.1f}")
            print(f"  Max speakers in one debate: {max(speaker_counts)}")
        
        # File extension breakdown
        if stats['file_extensions']:
            print(f"\nFile extensions:")
            for ext, count in stats['file_extensions'].most_common(5):
                pct = count / stats['total_debates'] * 100
                print(f"  {ext or '[no extension]'}: {count:,} ({pct:.1f}%)")
        
        # Sample titles (to understand content)
        if stats['titles']:
            print(f"\nSample titles:")
            sample_titles = random.sample(stats['titles'], min(5, len(stats['titles'])))
            for title in sample_titles:
                print(f"  • {title[:80]}{'...' if len(title) > 80 else ''}")
    
    def audit_years(self, start_year, end_year, show_examples=False, num_examples=3):
        """Audit multiple years"""
        available_years = self.list_available_years()
        target_years = [y for y in available_years if start_year <= y <= end_year]
        
        if not target_years:
            print(f"No data found for years {start_year}-{end_year}")
            print(f"Available years: {min(available_years)}-{max(available_years)}")
            return
        
        print(f"=== HANSARD DATA AUDIT: {start_year}-{end_year} ===")
        print(f"Found data for {len(target_years)} years: {target_years}")
        
        total_stats = {
            'total_debates': 0,
            'total_years': len(target_years),
            'years_with_data': target_years,
            'chamber_totals': Counter(),
            'all_word_counts': []
        }
        
        for year in target_years:
            stats = self.analyze_year_stats(year)
            if stats:
                self.print_year_stats(stats)
                
                # Aggregate stats
                total_stats['total_debates'] += stats['total_debates']
                for chamber, count in stats['chambers'].items():
                    total_stats['chamber_totals'][chamber] += count
                total_stats['all_word_counts'].extend(stats['word_counts'])
                
                # Show examples if requested
                if show_examples:
                    debates = self.load_debates_from_year(year, max_count=50)  # Limit for performance
                    self.show_debate_examples(debates, num_examples)
        
        # Print aggregate statistics
        print(f"\n=== AGGREGATE STATISTICS ({start_year}-{end_year}) ===")
        print(f"Total debates across all years: {total_stats['total_debates']:,}")
        print(f"Years with data: {len(total_stats['years_with_data'])}")
        
        if total_stats['chamber_totals']:
            print(f"\nOverall chamber distribution:")
            for chamber, count in total_stats['chamber_totals'].most_common():
                pct = count / total_stats['total_debates'] * 100
                print(f"  {chamber}: {count:,} ({pct:.1f}%)")
        
        if total_stats['all_word_counts']:
            wc = total_stats['all_word_counts']
            print(f"\nOverall word count statistics:")
            print(f"  Total words across corpus: {sum(wc):,}")
            print(f"  Mean words per debate: {sum(wc)/len(wc):.0f}")
            print(f"  Debates analyzed: {len(wc):,}")


def main():
    parser = argparse.ArgumentParser(description='Hansard Data Audit Tool')
    parser.add_argument('--year', type=int, help='Single year to audit')
    parser.add_argument('--years', type=str, help='Year range to audit (e.g., "1920-1930")')
    parser.add_argument('--sample', type=int, default=5, help='Number of example debates to show')
    parser.add_argument('--stats', action='store_true', help='Show only statistics (no examples)')
    parser.add_argument('--list-years', action='store_true', help='List all available years')
    
    args = parser.parse_args()
    
    auditor = HansardAuditor()
    
    if args.list_years:
        available_years = auditor.list_available_years()
        print(f"Available years with processed data:")
        print(f"Range: {min(available_years)}-{max(available_years)}")
        print(f"Total years: {len(available_years)}")
        
        # Show gaps
        all_years = set(range(min(available_years), max(available_years) + 1))
        missing_years = sorted(all_years - set(available_years))
        if missing_years:
            print(f"Missing years: {missing_years}")
        return
    
    # Determine year range
    if args.year:
        start_year = end_year = args.year
    elif args.years:
        start_year, end_year = map(int, args.years.split('-'))
    else:
        # Default: audit one recent year
        available_years = auditor.list_available_years()
        if available_years:
            start_year = end_year = available_years[-10]  # 10th from end for testing
            print(f"No year specified, using {start_year} as example")
        else:
            print("No processed data found")
            return
    
    # Run audit
    show_examples = not args.stats
    auditor.audit_years(start_year, end_year, show_examples=show_examples, num_examples=args.sample)


if __name__ == "__main__":
    main()