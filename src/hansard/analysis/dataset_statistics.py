#!/usr/bin/env python3
"""
Hansard Dataset Statistics

Quick comprehensive statistics about the entire Hansard parliamentary debates corpus
without heavy NLP processing - just metadata analysis for overview.

Usage:
    python dataset_statistics.py
"""

import json
import os
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from datetime import datetime

class HansardDatasetStats:
    """Fast dataset statistics analyzer"""
    
    def __init__(self, data_dir="data/processed_fixed"):
        self.data_dir = Path(data_dir)
        self.stats = {
            'analysis_timestamp': datetime.now().isoformat(),
            'corpus_overview': {},
            'temporal_coverage': {},
            'speaker_statistics': {},
            'chamber_distribution': {},
            'quality_metrics': {}
        }
    
    def analyze_dataset(self):
        """Comprehensive dataset analysis"""
        print("🏛️  HANSARD CORPUS DATASET STATISTICS")
        print("="*60)
        
        # Get overall corpus statistics
        self._analyze_corpus_coverage()
        self._analyze_speaker_statistics()
        self._analyze_temporal_distribution()
        self._analyze_quality_metrics()
        
        # Generate summary report
        self._print_summary()
        self._save_results()
        
        return self.stats
    
    def _analyze_corpus_coverage(self):
        """Analyze overall corpus coverage and size"""
        print("📊 Analyzing corpus coverage...")
        
        total_debates = 0
        total_words = 0
        years_with_data = set()
        chamber_counts = Counter()
        file_count = 0
        
        # Check content directory
        content_dir = self.data_dir / "content"
        if content_dir.exists():
            for year_dir in content_dir.iterdir():
                if year_dir.is_dir() and year_dir.name.isdigit():
                    year = int(year_dir.name)
                    years_with_data.add(year)
                    
                    jsonl_file = year_dir / f"debates_{year}.jsonl"
                    if jsonl_file.exists():
                        file_count += 1
                        
                        # Quick count of debates and estimate words
                        with open(jsonl_file, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                if line.strip():
                                    try:
                                        debate = json.loads(line)
                                        total_debates += 1
                                        
                                        # Extract metadata
                                        metadata = debate.get('metadata', {})
                                        word_count = metadata.get('word_count', 0)
                                        chamber = metadata.get('chamber', 'Unknown')
                                        
                                        total_words += word_count
                                        chamber_counts[chamber] += 1
                                        
                                    except json.JSONDecodeError:
                                        continue
                        
                        if line_num % 100 == 0:
                            print(f"  📁 {year}: {line_num:,} debates processed")
        
        # Calculate statistics
        years_covered = sorted(years_with_data)
        time_span = max(years_covered) - min(years_covered) + 1 if years_covered else 0
        avg_words_per_debate = total_words / total_debates if total_debates > 0 else 0
        avg_debates_per_year = total_debates / len(years_covered) if years_covered else 0
        
        self.stats['corpus_overview'] = {
            'total_debates': total_debates,
            'total_words': total_words,
            'avg_words_per_debate': avg_words_per_debate,
            'years_covered': len(years_covered),
            'time_span_years': time_span,
            'earliest_year': min(years_covered) if years_covered else None,
            'latest_year': max(years_covered) if years_covered else None,
            'avg_debates_per_year': avg_debates_per_year,
            'chamber_distribution': dict(chamber_counts),
            'years_with_data': years_covered,
            'data_files_processed': file_count
        }
    
    def _analyze_speaker_statistics(self):
        """Analyze speaker coverage and statistics"""
        print("👥 Analyzing speaker statistics...")
        
        # Try to load speaker data from Parquet if available
        speakers_master = self.data_dir / "metadata" / "speakers_master.parquet"
        if speakers_master.exists():
            try:
                df = pd.read_parquet(speakers_master)
                
                total_speaker_mentions = len(df)
                unique_speakers = df['speaker_name'].nunique()
                years_with_speakers = df['year'].nunique()
                speaker_coverage_by_year = df.groupby('year').size().to_dict()
                chamber_speaker_dist = df['chamber'].value_counts().to_dict()
                
                # Speaker activity distribution
                speaker_mention_counts = df['speaker_name'].value_counts()
                most_active_speakers = speaker_mention_counts.head(10).to_dict()
                
                self.stats['speaker_statistics'] = {
                    'total_speaker_mentions': total_speaker_mentions,
                    'unique_speakers': unique_speakers,
                    'years_with_speaker_data': years_with_speakers,
                    'avg_mentions_per_speaker': total_speaker_mentions / unique_speakers if unique_speakers > 0 else 0,
                    'speaker_coverage_by_chamber': chamber_speaker_dist,
                    'most_active_speakers': most_active_speakers,
                    'speaker_mention_distribution': {
                        'min_mentions': int(speaker_mention_counts.min()),
                        'max_mentions': int(speaker_mention_counts.max()),
                        'median_mentions': int(speaker_mention_counts.median()),
                        'speakers_with_1_mention': int((speaker_mention_counts == 1).sum()),
                        'speakers_with_10plus_mentions': int((speaker_mention_counts >= 10).sum())
                    }
                }
                
            except Exception as e:
                print(f"  ⚠️  Could not load speaker data: {e}")
                self.stats['speaker_statistics'] = {'error': str(e)}
        else:
            print("  ⚠️  No speaker master file found")
            self.stats['speaker_statistics'] = {'error': 'No speaker data file found'}
    
    def _analyze_temporal_distribution(self):
        """Analyze temporal distribution of debates"""
        print("📅 Analyzing temporal distribution...")
        
        years_data = self.stats['corpus_overview'].get('years_with_data', [])
        if not years_data:
            return
        
        # Decade distribution
        decade_counts = Counter()
        century_counts = Counter()
        
        for year in years_data:
            decade = (year // 10) * 10
            century = (year // 100) * 100
            decade_counts[f"{decade}s"] += 1
            century_counts[f"{century}s"] += 1
        
        # Find gaps in coverage
        gaps = []
        for i in range(1, len(years_data)):
            if years_data[i] - years_data[i-1] > 1:
                gap_start = years_data[i-1] + 1
                gap_end = years_data[i] - 1
                gaps.append((gap_start, gap_end))
        
        # Major historical periods coverage
        historical_periods = {
            'Victorian Era (1837-1901)': len([y for y in years_data if 1837 <= y <= 1901]),
            'Edwardian Era (1901-1910)': len([y for y in years_data if 1901 <= y <= 1910]),
            'WWI Period (1914-1918)': len([y for y in years_data if 1914 <= y <= 1918]),
            'Interwar Period (1918-1939)': len([y for y in years_data if 1918 <= y <= 1939]),
            'WWII Period (1939-1945)': len([y for y in years_data if 1939 <= y <= 1945]),
            'Post-War Era (1945-1979)': len([y for y in years_data if 1945 <= y <= 1979]),
            'Thatcher Era (1979-1990)': len([y for y in years_data if 1979 <= y <= 1990]),
            'Modern Era (1990-2005)': len([y for y in years_data if 1990 <= y <= 2005])
        }
        
        self.stats['temporal_coverage'] = {
            'decade_distribution': dict(decade_counts.most_common()),
            'century_distribution': dict(century_counts),
            'coverage_gaps': gaps,
            'historical_period_coverage': historical_periods,
            'coverage_percentage': (len(years_data) / (max(years_data) - min(years_data) + 1)) * 100 if years_data else 0
        }
    
    def _analyze_quality_metrics(self):
        """Analyze data quality metrics"""
        print("🔍 Analyzing data quality...")
        
        corpus = self.stats['corpus_overview']
        
        # Calculate quality metrics
        quality_metrics = {
            'avg_words_per_debate': corpus.get('avg_words_per_debate', 0),
            'data_completeness': len(corpus.get('years_with_data', [])) / 203 * 100,  # 203 years from 1803-2005
            'temporal_consistency': True,  # Would need deeper analysis
            'estimated_corpus_size_gb': (corpus.get('total_words', 0) * 6) / (1024**3),  # Rough estimate
        }
        
        # Chamber balance
        chamber_dist = corpus.get('chamber_distribution', {})
        commons_pct = chamber_dist.get('Commons', 0) / corpus.get('total_debates', 1) * 100
        lords_pct = chamber_dist.get('Lords', 0) / corpus.get('total_debates', 1) * 100
        
        quality_metrics.update({
            'commons_percentage': commons_pct,
            'lords_percentage': lords_pct,
            'chamber_balance_ratio': commons_pct / lords_pct if lords_pct > 0 else float('inf')
        })
        
        self.stats['quality_metrics'] = quality_metrics
    
    def _print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*60)
        print("📋 HANSARD DATASET SUMMARY")
        print("="*60)
        
        corpus = self.stats['corpus_overview']
        speakers = self.stats['speaker_statistics']
        temporal = self.stats['temporal_coverage']
        quality = self.stats['quality_metrics']
        
        # Overall corpus statistics
        print(f"\n🏛️  OVERALL CORPUS STATISTICS")
        print(f"   📄 Total Debates: {corpus.get('total_debates', 0):,}")
        print(f"   📝 Total Words: {corpus.get('total_words', 0):,}")
        print(f"   📊 Average Debate Length: {corpus.get('avg_words_per_debate', 0):,.0f} words")
        print(f"   📅 Years Covered: {corpus.get('years_covered', 0)} years")
        print(f"   🗓️  Time Period: {corpus.get('earliest_year')}-{corpus.get('latest_year')}")
        print(f"   📈 Average Debates/Year: {corpus.get('avg_debates_per_year', 0):,.0f}")
        print(f"   💾 Estimated Size: {quality.get('estimated_corpus_size_gb', 0):.1f} GB")
        
        # Chamber distribution
        chamber_dist = corpus.get('chamber_distribution', {})
        if chamber_dist:
            print(f"\n🏛️  CHAMBER DISTRIBUTION")
            for chamber, count in chamber_dist.items():
                pct = (count / corpus.get('total_debates', 1)) * 100
                print(f"   {chamber}: {count:,} debates ({pct:.1f}%)")
        
        # Speaker statistics
        if not speakers.get('error'):
            print(f"\n👥 SPEAKER STATISTICS")
            print(f"   👤 Unique Speakers: {speakers.get('unique_speakers', 0):,}")
            print(f"   💬 Total Speaker Mentions: {speakers.get('total_speaker_mentions', 0):,}")
            print(f"   📊 Avg Mentions/Speaker: {speakers.get('avg_mentions_per_speaker', 0):.1f}")
            
            mention_dist = speakers.get('speaker_mention_distribution', {})
            if mention_dist:
                print(f"   📈 Speaker Activity Distribution:")
                print(f"      • One-time speakers: {mention_dist.get('speakers_with_1_mention', 0):,}")
                print(f"      • Active speakers (10+ mentions): {mention_dist.get('speakers_with_10plus_mentions', 0):,}")
                print(f"      • Most active speaker: {mention_dist.get('max_mentions', 0):,} mentions")
        
        # Temporal coverage
        print(f"\n📅 TEMPORAL COVERAGE")
        print(f"   🎯 Coverage: {temporal.get('coverage_percentage', 0):.1f}% of possible years")
        print(f"   📊 Data Completeness: {quality.get('data_completeness', 0):.1f}%")
        
        gaps = temporal.get('coverage_gaps', [])
        if gaps:
            print(f"   ⚠️  Coverage Gaps: {len(gaps)} gaps found")
            for gap_start, gap_end in gaps[:5]:  # Show first 5 gaps
                print(f"      • {gap_start}-{gap_end}")
            if len(gaps) > 5:
                print(f"      • ... and {len(gaps)-5} more gaps")
        
        # Historical periods
        historical = temporal.get('historical_period_coverage', {})
        if historical:
            print(f"\n🏛️  HISTORICAL PERIOD COVERAGE")
            for period, years in historical.items():
                if years > 0:
                    print(f"   {period}: {years} years")
        
        # Top decades
        decades = temporal.get('decade_distribution', {})
        if decades:
            print(f"\n📊 TOP DECADES BY DATA VOLUME")
            for decade, years in list(decades.items())[:5]:
                print(f"   {decade}: {years} years of data")
        
        print(f"\n✅ Analysis completed at {self.stats['analysis_timestamp']}")
        print("="*60)
    
    def _save_results(self):
        """Save results to JSON"""
        output_file = Path("analysis/dataset_statistics.json")
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        print(f"\n💾 Detailed statistics saved to: {output_file}")

def main():
    analyzer = HansardDatasetStats()
    analyzer.analyze_dataset()

if __name__ == "__main__":
    main()