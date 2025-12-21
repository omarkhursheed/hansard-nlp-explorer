#!/usr/bin/env python3
"""
Suffrage Analysis Phase 1: Debate Identification & Analysis

Identifies and analyzes all suffrage-related debates from the Hansard corpus (1803-1950+),
setting up infrastructure for future LLM-based argument extraction.

Usage:
    python suffrage_analysis.py --years 1900-1930 --keywords "suffrage,franchise"
    python suffrage_analysis.py --milestone 1918 --export analysis/suffrage/
    python suffrage_analysis.py --years 1803-1950 --top-n 50 --export analysis/suffrage/
    python suffrage_analysis.py --years 1900-1930 --limit 1918-1920 --export analysis/test/
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from collections import Counter
import re
import sys
import os

# Add src to Python path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

from hansard.utils.unified_data_loader import UnifiedDataLoader
from hansard.utils.path_config import Paths

# Comprehensive suffrage keywords based on RESEARCH_NOTES.md and historical terminology
SUFFRAGE_KEYWORDS = [
    # Core suffrage terms
    'suffrage', 'franchise', 'enfranchisement', 'disenfranchisement',
    
    # Women-specific suffrage terms
    'women vote', 'women voting', 'female vote', 'female voting',
    'women\'s suffrage', 'woman suffrage', 'female suffrage',
    'women\'s franchise', 'woman franchise', 'female franchise',
    'women voters', 'female voters', 'woman voters',
    'votes for women', 'votes for woman',
    
    # Suffrage movement terminology
    'suffragist', 'suffragette', 'suffragettes', 'suffragists',
    'women\'s rights', 'woman\'s rights', 'female rights',
    'women\'s movement', 'woman\'s movement',
    
    # Electoral and political terms
    'electoral reform', 'representation of the people',
    'equal franchise', 'equal suffrage', 'universal suffrage',
    'women\'s representation', 'female representation',
    'women in parliament', 'women in politics',
    
    # Historical suffrage organizations (UK context)
    'women\'s social and political union', 'wspu',
    'national union of women\'s suffrage societies', 'nuwss',
    'women\'s freedom league', 'wfl',
    
    # Parliamentary and legal terms
    'women\'s enfranchisement', 'woman\'s enfranchisement',
    'female enfranchisement', 'women\'s disenfranchisement',
    'woman\'s disenfranchisement', 'female disenfranchisement',
    
    # Specific to UK suffrage debates
    'women\'s parliamentary vote', 'woman\'s parliamentary vote',
    'female parliamentary vote', 'women\'s electoral rights',
    'woman\'s electoral rights', 'female electoral rights'
]

class SuffrageAnalyzer:
    """Main class for suffrage debate analysis"""
    
    def __init__(self, data_loader: UnifiedDataLoader = None):
        """Initialize the suffrage analyzer"""
        self.loader = data_loader or UnifiedDataLoader()
        self.suffrage_debates = None
        self.summary_stats = {}
    
    def search_suffrage_debates(self, 
                               year_range: Tuple[int, int], 
                               keywords: List[str] = None,
                               case_sensitive: bool = False) -> pd.DataFrame:
        """Search for suffrage debates within a given year range"""
        if keywords is None:
            keywords = SUFFRAGE_KEYWORDS
        
        print(f"Searching for suffrage debates: {year_range[0]}-{year_range[1]}")
        print(f"Keywords: {', '.join(keywords)}")
        
        data = self.loader.load_debates(source='processed_fixed', year_range=year_range)
        df = data['metadata'].copy()
        
        # Create keyword pattern for case-insensitive search
        if not case_sensitive:
            pattern = '|'.join([re.escape(kw) for kw in keywords])
            mask = df['title'].str.contains(pattern, case=False, na=False)
        else:
            mask = df['title'].str.contains('|'.join(keywords), case=True, na=False)
        
        suffrage_df = df[mask].copy()
        
        # Add keyword match information
        suffrage_df['matched_keywords'] = suffrage_df['title'].apply(
            lambda x: [kw for kw in keywords if kw.lower() in x.lower()]
        )
        
        self.suffrage_debates = suffrage_df
        print(f"Found {len(suffrage_df)} suffrage-related debates")
        
        return suffrage_df
    
    def search_suffrage_debates_by_year(self, year: int, keywords: List[str] = None) -> pd.DataFrame:
        """Search for suffrage debates within a specific year"""
        return self.search_suffrage_debates((year, year), keywords)
    
    def analyze_temporal_distribution(self) -> Dict:
        """Analyze temporal distribution of suffrage debates"""
        if self.suffrage_debates is None:
            raise ValueError("No suffrage debates loaded. Run search_suffrage_debates first.")
        
        df = self.suffrage_debates.copy()
        
        # Add decade column
        df['decade'] = (df['year'] // 10) * 10
        
        # Count by year and decade
        yearly_counts = df['year'].value_counts().sort_index()
        decade_counts = df['decade'].value_counts().sort_index()
        
        # Find peak years
        peak_years = yearly_counts.nlargest(5).to_dict()
        
        temporal_analysis = {
            'yearly_counts': yearly_counts.to_dict(),
            'decade_counts': decade_counts.to_dict(),
            'peak_years': peak_years,
            'total_debates': len(df),
            'year_range': (df['year'].min(), df['year'].max()),
            'years_with_debates': len(yearly_counts)
        }
        
        return temporal_analysis
    
    def identify_milestone_years(self) -> Dict:
        """Identify key milestone years for suffrage"""
        if self.suffrage_debates is None:
            raise ValueError("No suffrage debates loaded. Run search_suffrage_debates first.")
        
        df = self.suffrage_debates.copy()
        yearly_counts = df['year'].value_counts().sort_index()
        
        milestones = {
            '1918': yearly_counts.get(1918, 0),  # Partial suffrage
            '1928': yearly_counts.get(1928, 0),  # Equal franchise
            'pre_1918': yearly_counts[yearly_counts.index < 1918].sum(),
            '1918_1928': yearly_counts[(yearly_counts.index >= 1918) & (yearly_counts.index < 1928)].sum(),
            'post_1928': yearly_counts[yearly_counts.index >= 1928].sum()
        }
        
        return milestones
    
    def get_suffrage_periods(self) -> pd.DataFrame:
        """Categorize debates into historical periods"""
        if self.suffrage_debates is None:
            raise ValueError("No suffrage debates loaded. Run search_suffrage_debates first.")
        
        df = self.suffrage_debates.copy()
        
        def categorize_period(year):
            if year < 1918:
                return 'pre_1918'
            elif year < 1928:
                return '1918_1928'
            else:
                return 'post_1928'
        
        df['period'] = df['year'].apply(categorize_period)
        return df

    def export_suffrage_debates(self, output_dir: str = "analysis/suffrage") -> Dict[str, str]:
        """Export suffrage debates to CSV files"""
        if self.suffrage_debates is None:
            raise ValueError("No suffrage debates loaded. Run search_suffrage_debates first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export full dataset
        full_path = output_path / "suffrage_debates_full.csv"
        self.suffrage_debates.to_csv(full_path, index=False)
        
        # Export by period
        period_df = self.get_suffrage_periods()
        period_path = output_path / "suffrage_debates_by_period.csv"
        period_df.to_csv(period_path, index=False)
        
        # Export individual period files
        period_files = {}
        for period in period_df['period'].unique():
            period_data = period_df[period_df['period'] == period]
            period_file = output_path / f"suffrage_debates_{period}.csv"
            period_data.to_csv(period_file, index=False)
            period_files[period] = str(period_file)
        
        return {
            'full_dataset': str(full_path),
            'by_period': str(period_path),
            'period_files': period_files
        }
    
    def generate_summary(self) -> Dict:
        """Generate comprehensive summary statistics"""
        if self.suffrage_debates is None:
            raise ValueError("No suffrage debates loaded. Run search_suffrage_debates first.")
        
        temporal = self.analyze_temporal_distribution()
        milestones = self.identify_milestone_years()
        
        summary = {
            'total_debates': len(self.suffrage_debates),
            'year_range': temporal['year_range'],
            'years_with_debates': temporal['years_with_debates'],
            'temporal_distribution': temporal,
            'milestone_years': milestones,
            'keyword_usage': self._analyze_keyword_usage(),
            'top_debates': self._get_top_debates()
        }
        
        self.summary_stats = summary
        return summary
    
    def _analyze_keyword_usage(self) -> Dict:
        """Analyze which keywords are most commonly found"""
        if self.suffrage_debates is None:
            return {}
        
        keyword_counts = Counter()
        for keywords in self.suffrage_debates['matched_keywords']:
            keyword_counts.update(keywords)
        
        return dict(keyword_counts.most_common())
    
    def _get_top_debates(self, n: int = 10) -> List[Dict]:
        """Get top N debates by some metric (e.g., most keywords matched)"""
        if self.suffrage_debates is None:
            return []
        
        df = self.suffrage_debates.copy()
        df['keyword_count'] = df['matched_keywords'].apply(len)
        
        top_debates = df.nlargest(n, 'keyword_count')[
            ['title', 'year', 'reference_date', 'keyword_count', 'matched_keywords']
        ].to_dict('records')
        
        return top_debates
    
    def export_summary(self, output_dir: str = "analysis/suffrage") -> str:
        """Export summary statistics to JSON"""
        if not self.summary_stats:
            self.generate_summary()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary_file = output_path / "summary_statistics.json"
        with open(summary_file, 'w') as f:
            json.dump(self.summary_stats, f, indent=2, default=str)
        
        return str(summary_file)


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Suffrage Analysis Phase 1: Debate Identification')
    parser.add_argument('--years', type=str, default='1803-1950', 
                       help='Year range (e.g., 1900-1930)')
    parser.add_argument('--keywords', type=str, 
                       help='Custom keywords (comma-separated)')
    parser.add_argument('--milestone', type=int, choices=[1918, 1928],
                       help='Focus on specific milestone year')
    parser.add_argument('--export', type=str, default='analysis/suffrage',
                       help='Output directory for results')
    parser.add_argument('--top-n', type=int,
                       help='Show only top N debates by keyword matches')
    parser.add_argument('--limit', type=str,
                       help='Limit to specific years for testing (e.g., 1918-1920)')
    parser.add_argument('--case-sensitive', action='store_true',
                       help='Use case-sensitive keyword matching')
    
    args = parser.parse_args()
    
    # Parse year range
    if '-' in args.years:
        start_year, end_year = map(int, args.years.split('-'))
    else:
        start_year = end_year = int(args.years)
    
    # Apply year limit if specified
    if args.limit:
        if '-' in args.limit:
            limit_start, limit_end = map(int, args.limit.split('-'))
            start_year = max(start_year, limit_start)
            end_year = min(end_year, limit_end)
        else:
            limit_year = int(args.limit)
            start_year = end_year = limit_year
        print(f"Limited to years: {start_year}-{end_year}")
    
    # Parse keywords
    if args.keywords:
        keywords = [kw.strip() for kw in args.keywords.split(',')]
    else:
        keywords = SUFFRAGE_KEYWORDS
    
    # Initialize analyzer
    analyzer = SuffrageAnalyzer()
    
    # Handle milestone focus
    if args.milestone:
        print(f"Focusing on milestone year: {args.milestone}")
        suffrage_debates = analyzer.search_suffrage_debates_by_year(
            args.milestone, keywords
        )
    else:
        suffrage_debates = analyzer.search_suffrage_debates(
            (start_year, end_year), keywords, case_sensitive=args.case_sensitive
        )
    
    # Apply top-n filtering if requested
    if args.top_n and len(suffrage_debates) > args.top_n:
        # Sort by keyword count and take top N
        suffrage_debates['keyword_count'] = suffrage_debates['matched_keywords'].apply(len)
        suffrage_debates = suffrage_debates.nlargest(args.top_n, 'keyword_count')
        suffrage_debates = suffrage_debates.drop('keyword_count', axis=1)  # Remove temp column
        print(f"Showing top {len(suffrage_debates)} debates by keyword matches")
    
    # Generate analysis
    print("\n=== TEMPORAL ANALYSIS ===")
    temporal = analyzer.analyze_temporal_distribution()
    print(f"Total debates found: {temporal['total_debates']}")
    print(f"Year range: {temporal['year_range'][0]}-{temporal['year_range'][1]}")
    print(f"Peak years: {temporal['peak_years']}")
    
    print("\n=== MILESTONE ANALYSIS ===")
    milestones = analyzer.identify_milestone_years()
    for year, count in milestones.items():
        print(f"{year}: {count} debates")
    
    # Export results
    print(f"\n=== EXPORTING RESULTS ===")
    export_files = analyzer.export_suffrage_debates(args.export)
    print(f"Exported to: {export_files['full_dataset']}")
    
    summary_file = analyzer.export_summary(args.export)
    print(f"Summary statistics: {summary_file}")
    
    print("\n=== TOP DEBATES ===")
    top_debates = analyzer._get_top_debates(5)
    for i, debate in enumerate(top_debates, 1):
        print(f"{i}. {debate['title']} ({debate['year']}) - {debate['keyword_count']} keywords")


if __name__ == '__main__':
    main()