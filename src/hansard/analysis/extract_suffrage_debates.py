#!/usr/bin/env python3
"""
Extract Suffrage-Related Debates (1900-1930)

Pilot study to identify debates related to women's suffrage for manual review
and validation of research questions.

Usage:
    python extract_suffrage_debates.py --output-dir analysis/suffrage_pilot
"""

import argparse
import json
import sys
import re
from pathlib import Path
from collections import Counter
import pandas as pd

# Import unified modules
sys.path.insert(0, str(Path(__file__).parent))
from unified_corpus_loader import UnifiedCorpusLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.path_config import Paths


# Suffrage keywords based on research notes
SUFFRAGE_KEYWORDS = {
    'primary': [
        'suffrage',
        'franchise',
        'women vote',
        'female vote',
        'women voting',
        'women voters',
        'women electors',
        'women suffragists',
    ],
    'related': [
        'electoral reform',
        'representation of the people',
        'equal franchise',
        'women enfranchisement',
        'women parliament',
        'women rights',
        'women political',
        'suffragettes',
        'suffragist',
    ],
    'acts': [
        'representation of the people act',
        '1918 act',
        '1928 act',
        'equal franchise act',
    ]
}


def search_debates_for_suffrage(year_range=(1900, 1930), max_samples=100):
    """
    Search debates for suffrage-related content.

    Args:
        year_range: Tuple of (start_year, end_year)
        max_samples: Maximum number of debates to return

    Returns:
        List of dicts with debate info and matched keywords
    """
    print(f"\n{'='*80}")
    print(f"SUFFRAGE DEBATE EXTRACTION")
    print(f"Years: {year_range[0]}-{year_range[1]}")
    print(f"={'='*80}\n")

    # Load overall corpus (full debates)
    loader = UnifiedCorpusLoader(dataset_type='overall')
    debates = loader.load_debates(year_range=year_range, sample_size=None)

    print(f"Loaded {len(debates):,} total debates")
    print(f"\nSearching for suffrage keywords...")

    # Compile regex patterns for each keyword category
    all_keywords = []
    for category, keywords in SUFFRAGE_KEYWORDS.items():
        all_keywords.extend(keywords)

    # Search debates
    suffrage_debates = []
    keyword_counts = Counter()

    for debate in debates:
        text_lower = debate['text'].lower()
        matched_keywords = []

        # Check for keyword matches
        for keyword in all_keywords:
            if keyword in text_lower:
                matched_keywords.append(keyword)
                keyword_counts[keyword] += 1

        if matched_keywords:
            suffrage_debates.append({
                'year': debate['year'],
                'date': debate['date'],
                'title': debate['title'],
                'text': debate['text'],
                'word_count': debate['word_count'],
                'speakers': debate['speakers'],
                'chamber': debate['chamber'],
                'matched_keywords': matched_keywords,
                'keyword_count': len(matched_keywords),
                'primary_match': any(kw in matched_keywords for kw in SUFFRAGE_KEYWORDS['primary'])
            })

    print(f"\nFound {len(suffrage_debates):,} suffrage-related debates")
    print(f"\nKeyword frequency:")
    for keyword, count in keyword_counts.most_common(20):
        print(f"  {keyword:30s}: {count:4d} debates")

    # Sort by primary match and keyword count
    suffrage_debates.sort(key=lambda x: (x['primary_match'], x['keyword_count']), reverse=True)

    # Limit to max_samples
    if len(suffrage_debates) > max_samples:
        print(f"\nLimiting to top {max_samples} most relevant debates")
        suffrage_debates = suffrage_debates[:max_samples]

    return suffrage_debates, keyword_counts


def analyze_temporal_distribution(debates):
    """Analyze temporal distribution of suffrage debates"""
    years = Counter()
    for debate in debates:
        years[debate['year']] += 1

    print(f"\nTemporal Distribution:")
    for year in sorted(years.keys()):
        print(f"  {year}: {years[year]:3d} debates")

    # Highlight key periods
    pre_1918 = sum(count for year, count in years.items() if year < 1918)
    year_1918 = years.get(1918, 0)
    between = sum(count for year, count in years.items() if 1918 < year < 1928)
    year_1928 = years.get(1928, 0)
    post_1928 = sum(count for year, count in years.items() if year > 1928)

    print(f"\nKey Periods:")
    print(f"  Pre-1918 (build-up):        {pre_1918:3d} debates")
    print(f"  1918 (partial suffrage):    {year_1918:3d} debates")
    print(f"  1918-1928 (interim):        {between:3d} debates")
    print(f"  1928 (equal franchise):     {year_1928:3d} debates")
    print(f"  Post-1928 (aftermath):      {post_1928:3d} debates")


def save_pilot_dataset(debates, output_dir, keyword_counts):
    """Save pilot dataset for manual review"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full dataset as JSON
    json_path = output_dir / 'suffrage_debates_pilot.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(debates, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full dataset: {json_path}")

    # Save summary as CSV for easy review
    summary_data = []
    for debate in debates:
        summary_data.append({
            'year': debate['year'],
            'date': debate['date'],
            'title': debate['title'][:100],  # Truncate long titles
            'word_count': debate['word_count'],
            'num_speakers': len(debate['speakers']) if debate['speakers'] else 0,
            'chamber': debate['chamber'],
            'keyword_count': debate['keyword_count'],
            'primary_match': debate['primary_match'],
            'keywords': ', '.join(debate['matched_keywords'][:5])  # First 5 keywords
        })

    df = pd.DataFrame(summary_data)
    csv_path = output_dir / 'suffrage_debates_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")

    # Save keyword statistics
    stats_path = output_dir / 'keyword_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump({
            'total_debates': len(debates),
            'keyword_frequencies': dict(keyword_counts.most_common()),
            'temporal_summary': {
                'years_covered': sorted(set(d['year'] for d in debates)),
                'earliest': min(d['year'] for d in debates),
                'latest': max(d['year'] for d in debates)
            }
        }, f, indent=2)
    print(f"Saved statistics: {stats_path}")

    # Create sample excerpts for manual review (first 20 debates)
    sample_path = output_dir / 'sample_excerpts.txt'
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write("SUFFRAGE DEBATE EXCERPTS - SAMPLE FOR MANUAL REVIEW\n")
        f.write("="*80 + "\n\n")

        for i, debate in enumerate(debates[:20], 1):
            f.write(f"DEBATE #{i}\n")
            f.write(f"Year: {debate['year']}\n")
            f.write(f"Date: {debate['date']}\n")
            f.write(f"Title: {debate['title']}\n")
            f.write(f"Keywords: {', '.join(debate['matched_keywords'])}\n")
            f.write(f"Chamber: {debate['chamber']}\n")
            f.write(f"\nFirst 500 words:\n")
            words = debate['text'].split()[:500]
            f.write(' '.join(words) + "...\n")
            f.write("\n" + "-"*80 + "\n\n")

    print(f"Saved sample excerpts: {sample_path}")

    print(f"\n{'='*80}")
    print(f"PILOT DATASET READY FOR MANUAL REVIEW")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extract suffrage-related debates for pilot study',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--years', type=str, default='1900-1930',
                       help='Year range (default: 1900-1930)')

    parser.add_argument('--max-samples', type=int, default=100,
                       help='Maximum number of debates to extract (default: 100)')

    parser.add_argument('--output-dir', type=str, default='analysis/suffrage_pilot',
                       help='Output directory for pilot dataset')

    args = parser.parse_args()

    # Parse year range
    try:
        parts = args.years.split('-')
        year_range = (int(parts[0]), int(parts[1]))
    except:
        print(f"Error: Invalid year range format: {args.years}")
        print("Use format: YYYY-YYYY (e.g., 1900-1930)")
        sys.exit(1)

    # Extract suffrage debates
    debates, keyword_counts = search_debates_for_suffrage(
        year_range=year_range,
        max_samples=args.max_samples
    )

    if not debates:
        print("\nNo suffrage-related debates found!")
        sys.exit(1)

    # Analyze temporal distribution
    analyze_temporal_distribution(debates)

    # Save pilot dataset
    save_pilot_dataset(debates, args.output_dir, keyword_counts)


if __name__ == "__main__":
    main()
