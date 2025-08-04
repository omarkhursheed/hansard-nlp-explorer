#!/usr/bin/env python3
"""
Overall Hansard Corpus Analysis

Comprehensive analysis of the entire Hansard parliamentary debates corpus (1803-2005),
providing high-level insights into the complete dataset.

Key analyses:
- Overall corpus statistics and trends over time
- Historical evolution of parliamentary language
- Long-term gender representation changes
- Major topic themes across centuries
- Decade-by-decade comparative analysis

Usage:
    python overall_corpus_analysis.py --sample 10000    # Large sample analysis
    python overall_corpus_analysis.py --full            # Full corpus (very long)
    python overall_corpus_analysis.py --decades         # Decade breakdown
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add the project root to the path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from analysis.hansard_nlp_analysis import HansardNLPAnalyzer

class OverallCorpusAnalyzer:
    """Comprehensive analysis of the entire Hansard corpus"""
    
    def __init__(self, output_dir="analysis/overall_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize base analyzer
        self.base_analyzer = HansardNLPAnalyzer(output_dir=self.output_dir)
        
        # Results storage
        self.results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'corpus_overview': {},
            'temporal_evolution': {},
            'decade_analysis': {},
            'century_comparison': {},
            'overall_topics': [],
            'gender_evolution': {},
            'chamber_evolution': {}
        }
    
    def analyze_decade(self, decade_start, sample_size=500):
        """Analyze a specific decade"""
        decade_end = decade_start + 9
        print(f"\n--- Analyzing {decade_start}s ({decade_start}-{decade_end}) ---")
        
        # Load debates for the decade
        debates = self.base_analyzer.load_debates(decade_start, decade_end, sample_size)
        if not debates:
            print(f"No debates found for {decade_start}s")
            return None
        
        # Extract texts
        texts = [d['text'] for d in debates if d.get('text')]
        
        # Run core analyses
        decade_results = {
            'decade': f"{decade_start}s",
            'period': f"{decade_start}-{decade_end}",
            'corpus_stats': self.base_analyzer.analyze_corpus_overview(debates),
            'speaker_analysis': self.base_analyzer.analyze_speaker_gender(debates)
        }
        
        # Only do text analysis if we have enough texts
        if len(texts) >= 50:
            decade_results['unigrams'] = self.base_analyzer._compute_unigrams_bigrams(texts)[0][:20]
            decade_results['gender_language'] = self.base_analyzer._compute_gender_stats(texts)
            
            # Topic modeling for larger samples
            if len(texts) >= 100:
                decade_results['topics'] = self.base_analyzer.topic_modeling(texts, n_topics=5)[:5]
        
        return decade_results
    
    def analyze_by_decades(self, start_decade=1800, end_decade=2000, sample_per_decade=500):
        """Analyze corpus decade by decade"""
        print("=== DECADE-BY-DECADE ANALYSIS ===")
        
        decade_results = []
        decades = range(start_decade, end_decade + 1, 10)
        
        for decade_start in decades:
            decade_data = self.analyze_decade(decade_start, sample_per_decade)
            if decade_data:
                decade_results.append(decade_data)
        
        self.results['decade_analysis'] = decade_results
        return decade_results
    
    def analyze_centuries(self):
        """Compare 19th vs 20th century parliamentary discourse"""
        print("\n=== CENTURY COMPARISON ===")
        
        # 19th century: 1801-1900
        print("Analyzing 19th Century (1801-1900)...")
        debates_19th = self.base_analyzer.load_debates(1801, 1900, 2000)
        texts_19th = [d['text'] for d in debates_19th if d.get('text')]
        
        # 20th century: 1901-2000
        print("Analyzing 20th Century (1901-2000)...")
        debates_20th = self.base_analyzer.load_debates(1901, 2000, 2000)
        texts_20th = [d['text'] for d in debates_20th if d.get('text')]
        
        century_comparison = {
            '19th_century': {
                'corpus_stats': self.base_analyzer.analyze_corpus_overview(debates_19th),
                'speaker_analysis': self.base_analyzer.analyze_speaker_gender(debates_19th),
                'gender_language': self.base_analyzer._compute_gender_stats(texts_19th),
                'top_unigrams': self.base_analyzer._compute_unigrams_bigrams(texts_19th)[0][:15]
            },
            '20th_century': {
                'corpus_stats': self.base_analyzer.analyze_corpus_overview(debates_20th),
                'speaker_analysis': self.base_analyzer.analyze_speaker_gender(debates_20th),
                'gender_language': self.base_analyzer._compute_gender_stats(texts_20th),
                'top_unigrams': self.base_analyzer._compute_unigrams_bigrams(texts_20th)[0][:15]
            }
        }
        
        self.results['century_comparison'] = century_comparison
        return century_comparison
    
    def overall_corpus_statistics(self, sample_size=5000):
        """Generate overall corpus statistics"""
        print("=== OVERALL CORPUS STATISTICS ===")
        
        # Load a large representative sample
        debates = self.base_analyzer.load_debates(1803, 2005, sample_size)
        texts = [d['text'] for d in debates if d.get('text')]
        
        # Comprehensive analysis using internal methods to avoid contamination
        self.results['corpus_overview'] = self.base_analyzer.analyze_corpus_overview(debates)
        self.results['overall_unigrams'], self.results['overall_bigrams'] = self.base_analyzer._compute_unigrams_bigrams(texts)
        self.results['overall_topics'] = self.base_analyzer.topic_modeling(texts, n_topics=10)
        self.results['overall_gender'] = self.base_analyzer._compute_gender_stats(texts)
        self.results['overall_speakers'] = self.base_analyzer.analyze_speaker_gender(debates)
        
        return self.results
    
    def generate_temporal_visualizations(self):
        """Generate temporal evolution visualizations"""
        print("Generating temporal visualizations...")
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Decade-by-decade trends
        if 'decade_analysis' in self.results and self.results['decade_analysis']:
            decades_data = self.results['decade_analysis']
            
            # Extract data for plotting
            decade_labels = [d['decade'] for d in decades_data]
            
            # Female speaker percentages over time
            female_speaker_pcts = []
            for d in decades_data:
                if d.get('speaker_analysis') and 'unique_gender_percentages' in d['speaker_analysis']:
                    female_pct = d['speaker_analysis']['unique_gender_percentages'].get('female', 0)
                    female_speaker_pcts.append(female_pct)
                else:
                    female_speaker_pcts.append(0)
            
            # Gender language ratios over time
            female_lang_ratios = []
            for d in decades_data:
                if d.get('gender_language'):
                    female_ratio = d['gender_language'].get('female_ratio', 0) * 100
                    female_lang_ratios.append(female_ratio)
                else:
                    female_lang_ratios.append(0)
            
            # Create temporal evolution plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Female speakers over time
            ax1.plot(decade_labels, female_speaker_pcts, marker='o', linewidth=2, markersize=6, color='red')
            ax1.set_title('Female Parliamentary Representation Over Time', fontweight='bold')
            ax1.set_ylabel('Female Speakers (%)')
            ax1.grid(True, alpha=0.3)
            ax1.set_xticklabels(decade_labels, rotation=45)
            
            # Female language over time
            ax2.plot(decade_labels, female_lang_ratios, marker='s', linewidth=2, markersize=6, color='blue')
            ax2.set_title('Female-Associated Language in Parliamentary Discourse', fontweight='bold')
            ax2.set_ylabel('Female Language (%)')
            ax2.set_xlabel('Decade')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticklabels(decade_labels, rotation=45)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'temporal_evolution.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. Century comparison
        if 'century_comparison' in self.results and self.results['century_comparison']:
            century_data = self.results['century_comparison']
            
            # Compare key metrics
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            centuries = ['19th Century', '20th Century']
            
            # Female speakers comparison
            female_19th = century_data['19th_century']['speaker_analysis']['unique_gender_percentages'].get('female', 0)
            female_20th = century_data['20th_century']['speaker_analysis']['unique_gender_percentages'].get('female', 0)
            
            ax1.bar(centuries, [female_19th, female_20th], color=['brown', 'green'], alpha=0.7)
            ax1.set_title('Female Speakers by Century', fontweight='bold')
            ax1.set_ylabel('Percentage (%)')
            
            # Gender language comparison
            female_lang_19th = century_data['19th_century']['gender_language']['female_ratio'] * 100
            female_lang_20th = century_data['20th_century']['gender_language']['female_ratio'] * 100
            
            ax2.bar(centuries, [female_lang_19th, female_lang_20th], color=['brown', 'green'], alpha=0.7)
            ax2.set_title('Female Language by Century', fontweight='bold')
            ax2.set_ylabel('Percentage (%)')
            
            # Average debates per year
            debates_19th = century_data['19th_century']['corpus_stats']['avg_debates_per_year']
            debates_20th = century_data['20th_century']['corpus_stats']['avg_debates_per_year']
            
            ax3.bar(centuries, [debates_19th, debates_20th], color=['brown', 'green'], alpha=0.7)
            ax3.set_title('Parliamentary Activity by Century', fontweight='bold')
            ax3.set_ylabel('Avg Debates per Year')
            
            # Words per debate
            words_19th = century_data['19th_century']['corpus_stats']['avg_words_per_debate']
            words_20th = century_data['20th_century']['corpus_stats']['avg_words_per_debate']
            
            ax4.bar(centuries, [words_19th, words_20th], color=['brown', 'green'], alpha=0.7)
            ax4.set_title('Debate Length by Century', fontweight='bold')
            ax4.set_ylabel('Avg Words per Debate')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'century_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Temporal visualizations saved to {plots_dir}/")
    
    def generate_overall_report(self):
        """Generate comprehensive markdown report"""
        report_path = self.output_dir / "OVERALL_CORPUS_REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write("# Hansard Parliamentary Debates - Overall Corpus Analysis\n\n")
            f.write(f"*Analysis generated: {self.results['analysis_timestamp']}*\n\n")
            
            # Overall statistics
            if 'corpus_overview' in self.results:
                corpus = self.results['corpus_overview']
                f.write("## Overall Corpus Statistics\n\n")
                f.write(f"- **Total Debates Analyzed**: {corpus['total_debates']:,}\n")
                f.write(f"- **Time Period**: {corpus['year_range']}\n")
                f.write(f"- **Total Words**: {corpus['total_words']:,}\n")
                f.write(f"- **Average Words per Debate**: {corpus['avg_words_per_debate']:,.0f}\n")
                f.write(f"- **Unique Speakers**: {corpus['unique_speakers']:,}\n")
                f.write(f"- **Speaker Coverage**: {corpus['speaker_coverage_pct']:.1f}%\n\n")
                
                if corpus.get('chamber_distribution'):
                    f.write("### Chamber Distribution\n")
                    for chamber, count in corpus['chamber_distribution'].items():
                        pct = (count / corpus['total_debates']) * 100
                        f.write(f"- **{chamber}**: {count:,} debates ({pct:.1f}%)\n")
                    f.write("\n")
            
            # Top words across entire corpus
            if 'overall_unigrams' in self.results:
                f.write("## Most Frequent Words (Entire Corpus)\n\n")
                for word, count in self.results['overall_unigrams'][:15]:
                    f.write(f"- **{word}**: {count:,}\n")
                f.write("\n")
            
            # Overall topics
            if 'overall_topics' in self.results:
                f.write("## Major Topics Across All Periods\n\n")
                for i, topic in enumerate(self.results['overall_topics'][:8]):
                    words = ', '.join(topic.get('words', [])[:8])
                    f.write(f"**Topic {i+1}**: {words}\n\n")
            
            # Century comparison
            if 'century_comparison' in self.results and self.results['century_comparison']:
                comparison = self.results['century_comparison']
                f.write("## 19th vs 20th Century Comparison\n\n")
                
                # Female representation
                if '19th_century' in comparison and '20th_century' in comparison:
                    female_19th = comparison['19th_century']['speaker_analysis']['unique_gender_percentages'].get('female', 0)
                    female_20th = comparison['20th_century']['speaker_analysis']['unique_gender_percentages'].get('female', 0)
                    f.write(f"### Female Parliamentary Representation\n")
                    f.write(f"- **19th Century**: {female_19th:.2f}%\n")
                    f.write(f"- **20th Century**: {female_20th:.2f}%\n")
                    f.write(f"- **Change**: {female_20th - female_19th:+.2f}%\n\n")
                    
                    # Parliamentary activity
                    debates_19th = comparison['19th_century']['corpus_stats']['avg_debates_per_year']
                    debates_20th = comparison['20th_century']['corpus_stats']['avg_debates_per_year']
                    f.write(f"### Parliamentary Activity\n")
                    f.write(f"- **19th Century**: {debates_19th:.0f} debates/year\n")
                    f.write(f"- **20th Century**: {debates_20th:.0f} debates/year\n")
                    f.write(f"- **Change**: {debates_20th - debates_19th:+.0f} debates/year\n\n")
            
            # Decade evolution
            if 'decade_analysis' in self.results and self.results['decade_analysis']:
                f.write("## Decade-by-Decade Evolution\n\n")
                f.write("| Decade | Female Speakers (%) | Avg Debates/Year | Avg Words/Debate |\n")
                f.write("|--------|-------------------|------------------|------------------|\n")
                
                for decade in self.results['decade_analysis']:
                    decade_name = decade['decade']
                    
                    # Female speaker percentage
                    if decade.get('speaker_analysis') and 'unique_gender_percentages' in decade['speaker_analysis']:
                        female_pct = decade['speaker_analysis']['unique_gender_percentages'].get('female', 0)
                    else:
                        female_pct = 0
                    
                    # Debates per year and words per debate
                    if decade.get('corpus_stats'):
                        debates_per_year = decade['corpus_stats'].get('avg_debates_per_year', 0)
                        words_per_debate = decade['corpus_stats'].get('avg_words_per_debate', 0)
                    else:
                        debates_per_year = words_per_debate = 0
                    
                    f.write(f"| {decade_name} | {female_pct:.2f}% | {debates_per_year:.0f} | {words_per_debate:.0f} |\n")
                f.write("\n")
            
            # Visualizations
            f.write("## Visualizations\n\n")
            f.write("### Temporal Evolution\n")
            f.write("![Temporal Evolution](plots/temporal_evolution.png)\n\n")
            f.write("### Century Comparison\n")
            f.write("![Century Comparison](plots/century_comparison.png)\n\n")
            f.write("### Overall Topic Modeling\n")
            f.write("![Topic Modeling](plots/topic_modeling_overview.png)\n\n")
            
            f.write("---\n")
            f.write("*This analysis provides a comprehensive overview of 200+ years of British parliamentary discourse, ")
            f.write("highlighting the evolution of political language, gender representation, and thematic concerns.*\n")
        
        print(f"📄 Overall corpus report generated: {report_path}")
    
    def save_results(self):
        """Save all results to JSON"""
        results_path = self.output_dir / "overall_corpus_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {results_path}")
    
    def run_comprehensive_analysis(self, sample_size=5000, include_decades=True, include_centuries=True):
        """Run the complete overall corpus analysis"""
        print("🏛️  COMPREHENSIVE HANSARD CORPUS ANALYSIS")
        print("="*80)
        
        # Overall statistics
        self.overall_corpus_statistics(sample_size)
        
        # Century comparison
        if include_centuries:
            self.analyze_centuries()
        
        # Decade analysis
        if include_decades:
            self.analyze_by_decades(sample_per_decade=10000)
        
        # Generate visualizations using a clean base analyzer instance
        if 'overall_topics' in self.results:
            # Create a dedicated analyzer instance for visualizations to avoid contamination
            viz_analyzer = HansardNLPAnalyzer(output_dir=self.output_dir)
            viz_analyzer.results['topics'] = self.results['overall_topics']
            viz_analyzer.results['unigrams'] = self.results['overall_unigrams'] 
            viz_analyzer.results['bigrams'] = self.results['overall_bigrams']
            viz_analyzer.results['corpus_overview'] = self.results['corpus_overview']
            viz_analyzer.results['gender_analysis'] = self.results['overall_gender']
            viz_analyzer.results['speaker_gender_analysis'] = self.results['overall_speakers']
            viz_analyzer.generate_visualizations()
        
        self.generate_temporal_visualizations()
        
        # Generate report and save
        self.generate_overall_report()
        self.save_results()
        
        print("✅ Comprehensive analysis complete!")
        return self.results

def main():
    parser = argparse.ArgumentParser(description='Overall Hansard Corpus Analysis')
    parser.add_argument('--sample', type=int, default=5000, help='Sample size for overall analysis')
    parser.add_argument('--full', action='store_true', help='Analyze full corpus (very long)')
    parser.add_argument('--decades', action='store_true', help='Include decade-by-decade analysis')
    parser.add_argument('--centuries', action='store_true', help='Include century comparison')
    parser.add_argument('--output', type=str, default='analysis/overall_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    # Determine sample size
    if args.full:
        sample_size = None
    else:
        sample_size = args.sample
    
    # Run analysis
    analyzer = OverallCorpusAnalyzer(output_dir=args.output)
    analyzer.run_comprehensive_analysis(
        sample_size=sample_size,
        include_decades=args.decades,
        include_centuries=args.centuries
    )

if __name__ == "__main__":
    main()