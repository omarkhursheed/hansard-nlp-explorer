#!/usr/bin/env python3
"""
Historical Milestone Analysis for Hansard Parliamentary Debates

Analyzes key historical periods and milestones in British parliamentary history,
focusing on gender representation and language changes around pivotal moments.

Key Milestones Analyzed:
- 1918: Partial women's suffrage (30+) + First women allowed in Parliament
- 1928: Full women's suffrage (21+)
- 1914-1918: World War I
- 1939-1945: World War II  
- 1959: Margaret Thatcher elected to Parliament
- 1979: Margaret Thatcher becomes Prime Minister

Each analysis uses 10-year windows around milestone dates for robust comparison.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from analysis.hansard_nlp_analysis import HansardNLPAnalyzer

class HistoricalMilestoneAnalyzer:
    """Analyzer for key historical milestones in British parliamentary history"""
    
    def __init__(self, output_base_dir="analysis/historical_milestones"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Define historical milestones with 10-year analysis windows
        self.milestones = {
            "1918_partial_suffrage": {
                "name": "1918 Partial Women's Suffrage & Parliament Entry",
                "description": "Women over 30 gain vote; first women allowed in Parliament",
                "milestone_year": 1918,
                "pre_window": (1908, 1918),   # 10 years before
                "post_window": (1918, 1928),  # 10 years after
                "sample_size": 10000
            },
            "1928_full_suffrage": {
                "name": "1928 Full Women's Suffrage",
                "description": "Voting age for women reduced to 21 (Equal Franchise Act)",
                "milestone_year": 1928,
                "pre_window": (1918, 1928),   # 10 years before
                "post_window": (1928, 1938),  # 10 years after
                "sample_size": 10000
            },
            "ww1_period": {
                "name": "World War I Analysis",
                "description": "Parliamentary discourse before, during, and after WWI",
                "milestone_year": 1916,  # Mid-war
                "pre_window": (1909, 1914),   # 5 years before war
                "during_window": (1914, 1918),  # War period
                "post_window": (1918, 1923),  # 5 years after war
                "sample_size": 10000
            },
            "ww2_period": {
                "name": "World War II Analysis", 
                "description": "Parliamentary discourse before, during, and after WWII",
                "milestone_year": 1942,  # Mid-war
                "pre_window": (1934, 1939),   # 5 years before war
                "during_window": (1939, 1945),  # War period
                "post_window": (1945, 1950),  # 5 years after war
                "sample_size": 10000
            },
            "thatcher_period": {
                "name": "Thatcher Era Analysis",
                "description": "Parliamentary discourse before, during, and after Thatcher's tenure as PM",
                "milestone_year": 1984,  # Mid-tenure
                "pre_window": (1974, 1979),   # 5 years before PM
                "during_window": (1979, 1990),  # PM tenure
                "post_window": (1990, 1995),  # 5 years after PM
                "sample_size": 10000
            }
        }
    
    def analyze_milestone(self, milestone_key, force_rerun=False):
        """Analyze a specific historical milestone"""
        milestone = self.milestones[milestone_key]
        
        print(f"\n{'='*80}")
        print(f"ANALYZING: {milestone['name']}")
        print(f"Description: {milestone['description']}")
        print(f"Milestone Year: {milestone['milestone_year']}")
        print(f"Pre-period: {milestone['pre_window'][0]}-{milestone['pre_window'][1]}")
        print(f"Post-period: {milestone['post_window'][0]}-{milestone['post_window'][1]}")
        print(f"{'='*80}")
        
        # Create output directory for this milestone
        milestone_dir = self.output_base_dir / milestone_key
        milestone_dir.mkdir(exist_ok=True)
        
        # Check if analysis already exists
        results_file = milestone_dir / "milestone_analysis.json"
        if results_file.exists() and not force_rerun:
            print(f"Analysis already exists at {results_file}. Use --force to rerun.")
            return
        
        # Run pre-period analysis
        print(f"\n--- PRE-{milestone['milestone_year']} ANALYSIS ---")
        pre_analyzer = HansardNLPAnalyzer(output_dir=milestone_dir / "pre_period")
        pre_results = self._run_period_analysis(
            pre_analyzer, 
            milestone['pre_window'][0], 
            milestone['pre_window'][1],
            milestone['sample_size']
        )
        
        # Run during-period analysis if it exists  
        during_results = None
        if 'during_window' in milestone:
            print(f"\n--- DURING-{milestone['milestone_year']} ANALYSIS ---")
            during_analyzer = HansardNLPAnalyzer(output_dir=milestone_dir / "during_period")
            during_results = self._run_period_analysis(
                during_analyzer,
                milestone['during_window'][0],
                milestone['during_window'][1], 
                milestone['sample_size']
            )
        
        # Run post-period analysis  
        print(f"\n--- POST-{milestone['milestone_year']} ANALYSIS ---")
        post_analyzer = HansardNLPAnalyzer(output_dir=milestone_dir / "post_period")
        post_results = self._run_period_analysis(
            post_analyzer,
            milestone['post_window'][0],
            milestone['post_window'][1], 
            milestone['sample_size']
        )
        
        # Compare periods and generate insights
        comparison = self._compare_periods(pre_results, post_results, milestone, during_results)
        
        # Save comprehensive results
        milestone_results = {
            "milestone_info": milestone,
            "pre_period_results": pre_results,
            "post_period_results": post_results,
            "comparison_analysis": comparison,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_debates_analyzed": (
                pre_results.get('corpus_overview', {}).get('total_debates', 0) +
                post_results.get('corpus_overview', {}).get('total_debates', 0) +
                (during_results.get('corpus_overview', {}).get('total_debates', 0) if during_results else 0)
            )
        }
        
        # Add during period results if they exist
        if during_results:
            milestone_results["during_period_results"] = during_results
        
        with open(results_file, 'w') as f:
            json.dump(milestone_results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_milestone_report(milestone_results, milestone_dir)
        
        print(f"\n✅ Analysis complete! Results saved to {milestone_dir}/")
        return milestone_results
    
    def _run_period_analysis(self, analyzer, start_year, end_year, sample_size):
        """Run full NLP analysis for a specific period"""
        print(f"Analyzing period {start_year}-{end_year} with sample size {sample_size}")
        
        # Load debates for the period
        debates = analyzer.load_debates(start_year, end_year, sample_size)
        if not debates:
            print(f"Warning: No debates found for period {start_year}-{end_year}")
            return {}
        
        # Extract texts from debates for text analysis - key is 'text' not 'full_text'
        texts = [d['text'] for d in debates if d.get('text')]
        
        # Run analyses - use internal methods to avoid overwriting analyzer.results
        corpus_stats = analyzer.analyze_corpus_overview(debates)
        unigrams, bigrams = analyzer._compute_unigrams_bigrams(texts)
        topics = analyzer.topic_modeling(texts)
        gender_analysis = analyzer._compute_gender_stats(texts)
        
        # Speaker analysis using full dataset (not sample) - load speakers from debates
        speaker_analysis = analyzer.analyze_speaker_gender(debates)
        
        # Store results in analyzer for visualizations 
        analyzer.results['unigrams'] = unigrams
        analyzer.results['bigrams'] = bigrams  
        analyzer.results['topics'] = topics
        analyzer.results['gender_analysis'] = gender_analysis
        analyzer.results['corpus_overview'] = corpus_stats
        analyzer.results['speaker_gender_analysis'] = speaker_analysis
        
        # Generate visualizations (including topic modeling)
        analyzer.generate_visualizations()
        
        # Return all results
        return {
            'corpus_overview': corpus_stats,
            'unigrams': unigrams,
            'bigrams': bigrams, 
            'topics': topics,
            'gender_analysis': gender_analysis,
            'speaker_gender_analysis': speaker_analysis,
            'period': f"{start_year}-{end_year}"
        }
    
    def _compare_periods(self, pre_results, post_results, milestone, during_results=None):
        """Generate comparative analysis between pre, during (if exists), and post periods"""
        comparison = {
            "milestone_impact_summary": f"Comparing parliamentary discourse before and after {milestone['name']} ({milestone['milestone_year']})",
            "corpus_changes": {},
            "gender_language_changes": {},
            "speaker_representation_changes": {},
            "vocabulary_changes": {}
        }
        
        # Corpus size comparison
        pre_corpus = pre_results.get('corpus_overview', {})
        post_corpus = post_results.get('corpus_overview', {})
        
        if pre_corpus and post_corpus:
            comparison["corpus_changes"] = {
                "debates_change": post_corpus.get('total_debates', 0) - pre_corpus.get('total_debates', 0),
                "words_change": post_corpus.get('total_words', 0) - pre_corpus.get('total_words', 0),
                "avg_words_per_debate_change": post_corpus.get('avg_words_per_debate', 0) - pre_corpus.get('avg_words_per_debate', 0)
            }
        
        # Gender language comparison
        pre_gender = pre_results.get('gender_analysis', {})
        post_gender = post_results.get('gender_analysis', {})
        
        if pre_gender and post_gender:
            comparison["gender_language_changes"] = {
                "female_ratio_change": post_gender.get('female_ratio', 0) - pre_gender.get('female_ratio', 0),
                "male_ratio_change": post_gender.get('male_ratio', 0) - pre_gender.get('male_ratio', 0),
                "female_word_count_change": post_gender.get('female_word_count', 0) - pre_gender.get('female_word_count', 0),
                "male_word_count_change": post_gender.get('male_word_count', 0) - pre_gender.get('male_word_count', 0)
            }
        
        # Speaker representation comparison
        pre_speakers = pre_results.get('speaker_gender_analysis', {})
        post_speakers = post_results.get('speaker_gender_analysis', {})
        
        if pre_speakers and post_speakers:
            pre_female_pct = pre_speakers.get('unique_gender_percentages', {}).get('female', 0)
            post_female_pct = post_speakers.get('unique_gender_percentages', {}).get('female', 0)
            
            comparison["speaker_representation_changes"] = {
                "female_speaker_percentage_change": post_female_pct - pre_female_pct,
                "unique_speakers_change": post_speakers.get('unique_speakers', 0) - pre_speakers.get('unique_speakers', 0),
                "total_speaker_mentions_change": post_speakers.get('total_speaker_mentions', 0) - pre_speakers.get('total_speaker_mentions', 0)
            }
        
        # Vocabulary changes (top words comparison)
        pre_unigrams = dict(pre_results.get('unigrams', [])[:20])
        post_unigrams = dict(post_results.get('unigrams', [])[:20])
        
        # Find new words that appeared in top 20 post-milestone
        new_top_words = set(post_unigrams.keys()) - set(pre_unigrams.keys())
        disappeared_words = set(pre_unigrams.keys()) - set(post_unigrams.keys())
        
        comparison["vocabulary_changes"] = {
            "new_top_words": list(new_top_words),
            "disappeared_top_words": list(disappeared_words),
            "vocabulary_shift_magnitude": len(new_top_words) + len(disappeared_words)
        }
        
        return comparison
    
    def _generate_milestone_report(self, results, output_dir):
        """Generate a markdown report for the milestone analysis"""
        milestone = results["milestone_info"]
        comparison = results["comparison_analysis"]
        
        report_path = output_dir / "milestone_report.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# {milestone['name']} - Parliamentary Analysis\n\n")
            f.write(f"**Description:** {milestone['description']}\n\n")
            f.write(f"**Milestone Year:** {milestone['milestone_year']}\n\n")
            f.write(f"**Analysis Periods:**\n")
            f.write(f"- Pre-period: {milestone['pre_window'][0]}-{milestone['pre_window'][1]}\n")
            f.write(f"- Post-period: {milestone['post_window'][0]}-{milestone['post_window'][1]}\n\n")
            
            # Corpus changes
            if comparison.get("corpus_changes"):
                corpus = comparison["corpus_changes"]
                f.write("## Corpus Changes\n\n")
                f.write(f"- **Debates Change:** {corpus.get('debates_change', 0):+,}\n")
                f.write(f"- **Total Words Change:** {corpus.get('words_change', 0):+,}\n")
                f.write(f"- **Avg Words/Debate Change:** {corpus.get('avg_words_per_debate_change', 0):+.1f}\n\n")
            
            # Gender language changes
            if comparison.get("gender_language_changes"):
                gender = comparison["gender_language_changes"]
                f.write("## Gender Language Changes\n\n")
                f.write(f"- **Female Language Ratio Change:** {gender.get('female_ratio_change', 0):+.4f} ({gender.get('female_ratio_change', 0)*100:+.2f}%)\n")
                f.write(f"- **Male Language Ratio Change:** {gender.get('male_ratio_change', 0):+.4f} ({gender.get('male_ratio_change', 0)*100:+.2f}%)\n")
                f.write(f"- **Female Word Count Change:** {gender.get('female_word_count_change', 0):+,}\n")
                f.write(f"- **Male Word Count Change:** {gender.get('male_word_count_change', 0):+,}\n\n")
            
            # Speaker representation changes
            if comparison.get("speaker_representation_changes"):
                speakers = comparison["speaker_representation_changes"]
                f.write("## Speaker Representation Changes\n\n")
                f.write(f"- **Female Speaker % Change:** {speakers.get('female_speaker_percentage_change', 0):+.2f}%\n")
                f.write(f"- **Unique Speakers Change:** {speakers.get('unique_speakers_change', 0):+,}\n")
                f.write(f"- **Total Speaker Mentions Change:** {speakers.get('total_speaker_mentions_change', 0):+,}\n\n")
            
            # Vocabulary changes
            if comparison.get("vocabulary_changes"):
                vocab = comparison["vocabulary_changes"]
                f.write("## Vocabulary Changes\n\n")
                f.write(f"**New Top Words:** {', '.join(vocab.get('new_top_words', []))}\n\n")
                f.write(f"**Disappeared Words:** {', '.join(vocab.get('disappeared_top_words', []))}\n\n")
                f.write(f"**Vocabulary Shift Score:** {vocab.get('vocabulary_shift_magnitude', 0)}\n\n")
            
            # Topic analysis
            f.write("## Topic Analysis\n\n")
            
            # Pre-period topics
            pre_topics = results["pre_period_results"].get("topics", [])
            if pre_topics:
                f.write(f"### Pre-{milestone['milestone_year']} Topics\n")
                for i, topic in enumerate(pre_topics[:5]):  # Show top 5 topics
                    words = ', '.join(topic.get('words', [])[:8])  # Show top 8 words
                    f.write(f"**Topic {i+1}:** {words}\n\n")
            
            # Post-period topics  
            post_topics = results["post_period_results"].get("topics", [])
            if post_topics:
                f.write(f"### Post-{milestone['milestone_year']} Topics\n")
                for i, topic in enumerate(post_topics[:5]):  # Show top 5 topics
                    words = ', '.join(topic.get('words', [])[:8])  # Show top 8 words
                    f.write(f"**Topic {i+1}:** {words}\n\n")
            
            # Visualizations
            f.write("## Visualizations\n\n")
            f.write("### Pre-Period Analysis\n")
            f.write("![Pre-Period Plots](pre_period/plots/)\n\n")
            f.write("### Post-Period Analysis\n") 
            f.write("![Post-Period Plots](post_period/plots/)\n\n")
            
            f.write(f"---\n\n*Analysis generated on {results['analysis_timestamp']}*\n")
        
        print(f"📄 Report generated: {report_path}")
    
    def analyze_all_milestones(self, force_rerun=False):
        """Run analysis for all defined historical milestones"""
        print(f"🏛️  COMPREHENSIVE HISTORICAL MILESTONE ANALYSIS")
        print(f"Analyzing {len(self.milestones)} key periods in British parliamentary history\n")
        
        results = {}
        for milestone_key in self.milestones.keys():
            try:
                result = self.analyze_milestone(milestone_key, force_rerun)
                results[milestone_key] = result
                print(f"✅ Completed: {self.milestones[milestone_key]['name']}")
            except Exception as e:
                print(f"❌ Failed: {self.milestones[milestone_key]['name']} - {str(e)}")
                results[milestone_key] = {"error": str(e)}
        
        # Generate master summary
        self._generate_master_summary(results)
        
        return results
    
    def _generate_master_summary(self, all_results):
        """Generate a master summary of all milestone analyses"""
        summary_path = self.output_base_dir / "MASTER_SUMMARY.md"
        
        with open(summary_path, 'w') as f:
            f.write("# British Parliamentary History - Milestone Analysis Summary\n\n")
            f.write("Comprehensive analysis of key historical periods and their impact on parliamentary discourse.\n\n")
            
            f.write("## Analyzed Milestones\n\n")
            for milestone_key, milestone in self.milestones.items():
                status = "✅ Complete" if milestone_key in all_results and all_results[milestone_key] and "error" not in all_results[milestone_key] else "❌ Failed"
                f.write(f"### {milestone['name']} ({milestone['milestone_year']})\n")
                f.write(f"**Status:** {status}\n")
                f.write(f"**Description:** {milestone['description']}\n")
                f.write(f"**Analysis Period:** {milestone['pre_window'][0]}-{milestone['post_window'][1]}\n")
                f.write(f"**Results:** [View Detailed Analysis]({milestone_key}/milestone_report.md)\n\n")
            
            f.write("## Key Findings Summary\n\n")
            f.write("*Detailed findings available in individual milestone reports.*\n\n")
            
            f.write("## Analysis Methodology\n\n")
            f.write("- **Text Analysis:** Stratified sampling for unigrams, bigrams, topics, and gender language\n")
            f.write("- **Speaker Analysis:** Comprehensive analysis of full speaker dataset\n") 
            f.write("- **Time Windows:** 10-year periods around milestone dates (5 years for war periods)\n")
            f.write("- **Gender Classification:** Title-based identification with improved regex patterns\n")
            f.write("- **Comparative Analysis:** Before/after statistical comparisons\n\n")
            
            f.write(f"---\n*Master analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        print(f"📋 Master summary generated: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze historical milestones in British parliamentary discourse")
    parser.add_argument("--milestone", help="Specific milestone to analyze", 
                       choices=["1918_partial_suffrage", "1928_full_suffrage", "ww1_period", 
                               "ww2_period", "thatcher_period"])
    parser.add_argument("--all", action="store_true", help="Analyze all milestones")
    parser.add_argument("--force", action="store_true", help="Force rerun existing analyses")
    
    args = parser.parse_args()
    
    analyzer = HistoricalMilestoneAnalyzer()
    
    if args.all:
        analyzer.analyze_all_milestones(force_rerun=args.force)
    elif args.milestone:
        analyzer.analyze_milestone(args.milestone, force_rerun=args.force)
    else:
        print("Please specify --milestone <name> or --all")
        parser.print_help()

if __name__ == "__main__":
    main()