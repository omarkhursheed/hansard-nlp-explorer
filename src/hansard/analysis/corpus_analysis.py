#!/usr/bin/env python3
"""
Unified Hansard Corpus Analysis

Single parameterized script for all corpus-level analysis, replacing:
- enhanced_gender_corpus_analysis.py
- comprehensive_corpus_analysis.py
- hansard_nlp_analysis.py

Supports both gender-matched and overall corpus datasets with consistent
filtering, visualization, and analysis methods.

Usage:
    # Gender analysis
    python corpus_analysis.py --dataset gender --years 1920-1930 --sample 5000 --filtering aggressive

    # Overall corpus
    python corpus_analysis.py --dataset overall --years 1920-1930 --sample 5000 --filtering moderate

    # Compare filtering levels
    python corpus_analysis.py --dataset gender --years 1920-1930 --sample 5000 --compare-filtering

    # Full corpus
    python corpus_analysis.py --dataset gender --full --sample 50000 --filtering aggressive
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Import unified modules
sys.path.insert(0, str(Path(__file__).parent))
from unified_text_filtering import HansardTextFilter
from unified_corpus_loader import UnifiedCorpusLoader
from professional_visualizations import UnifiedVisualizationSuite

# Import path configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.path_config import Paths


class UnifiedCorpusAnalyzer:
    """
    Unified corpus analyzer for Hansard parliamentary debates.

    Replaces multiple analysis scripts with single parameterized approach.
    """

    def __init__(self, dataset_type: str, filtering_level: str,
                 output_dir: Path = None):
        """
        Initialize analyzer.

        Args:
            dataset_type: 'gender' or 'overall'
            filtering_level: 'minimal', 'basic', 'parliamentary', 'moderate', or 'aggressive'
            output_dir: Custom output directory
        """
        self.dataset_type = dataset_type
        self.filtering_level = filtering_level

        # Initialize components
        self.loader = UnifiedCorpusLoader(dataset_type=dataset_type)
        self.filter = HansardTextFilter(level=filtering_level)

        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Paths.ANALYSIS_DIR / f"corpus_{dataset_type}"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.viz = UnifiedVisualizationSuite(output_dir=self.output_dir)

        # Load gender wordlists for gender analysis
        self.male_words = self._load_gender_wordlist(Paths.MALE_WORDS)
        self.female_words = self._load_gender_wordlist(Paths.FEMALE_WORDS)

        # Results storage
        self.results = {}

    def _load_gender_wordlist(self, filepath: Path) -> set:
        """Load gender wordlist"""
        try:
            with open(filepath, 'r') as f:
                return set(word.strip().lower() for word in f if word.strip())
        except FileNotFoundError:
            print(f"Warning: Gender wordlist not found at {filepath}")
            return set()

    def run_analysis(self, year_range=None, sample_size=None,
                    analysis_types=['all']) -> dict:
        """
        Run comprehensive corpus analysis.

        Args:
            year_range: Tuple of (start_year, end_year) or None for full corpus
            sample_size: Number of speeches/debates to sample
            analysis_types: List of analysis types to run

        Returns:
            Dict with analysis results
        """
        print("\n" + "="*80)
        print(f"UNIFIED HANSARD CORPUS ANALYSIS")
        print(f"Dataset: {self.dataset_type}")
        print(f"Filtering: {self.filtering_level}")
        if year_range:
            print(f"Years: {year_range[0]}-{year_range[1]}")
        else:
            print(f"Years: Full corpus (1803-2005)")
        if sample_size:
            print(f"Sample: {sample_size:,}")
        print("="*80 + "\n")

        # Load data
        print("Loading data...")
        data = self.loader.load_debates(year_range=year_range,
                                        sample_size=sample_size,
                                        stratified=True)

        # Store metadata
        self.results['metadata'] = {
            'dataset_type': self.dataset_type,
            'filtering_level': self.filtering_level,
            'year_range': year_range,
            'sample_size': sample_size,
            'timestamp': datetime.now().isoformat()
        }

        # Determine what analyses to run
        if 'all' in analysis_types:
            analysis_types = ['unigram', 'bigram', 'topic', 'gender', 'temporal']

        # Run requested analyses
        if 'unigram' in analysis_types:
            print("\nAnalyzing unigrams...")
            self._analyze_unigrams(data)

        if 'bigram' in analysis_types:
            print("\nAnalyzing bigrams...")
            self._analyze_bigrams(data)

        if 'topic' in analysis_types:
            print("\nPerforming topic modeling...")
            self._analyze_topics(data)

        if 'gender' in analysis_types:
            print("\nAnalyzing gender language...")
            self._analyze_gender(data)

        if 'temporal' in analysis_types and self.dataset_type == 'gender':
            print("\nAnalyzing temporal trends...")
            self._analyze_temporal(data)

        # Create visualizations
        print("\nCreating visualizations...")
        self._create_visualizations()

        # Save results
        print("\nSaving results...")
        self._save_results()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print(f"Results saved to: {self.output_dir}")
        print("="*80 + "\n")

        return self.results

    def _analyze_unigrams(self, data):
        """Analyze word frequencies"""
        if self.dataset_type == 'gender':
            # Separate analysis for male and female speeches
            male_texts = data['male_speeches']
            female_texts = data['female_speeches']

            male_filtered = [self.filter.filter_text(t) for t in male_texts]
            female_filtered = [self.filter.filter_text(t) for t in female_texts]

            male_words = ' '.join(male_filtered).split()
            female_words = ' '.join(female_filtered).split()

            male_freq = Counter(male_words)
            female_freq = Counter(female_words)

            self.results['male_unigrams'] = male_freq.most_common(50)
            self.results['female_unigrams'] = female_freq.most_common(50)
            self.results['male_vocab_size'] = len(male_freq)
            self.results['female_vocab_size'] = len(female_freq)

            # Calculate filtering stats
            original_male_words = sum(len(t.split()) for t in male_texts)
            original_female_words = sum(len(t.split()) for t in female_texts)
            filtered_male_words = len(male_words)
            filtered_female_words = len(female_words)
            total_original = original_male_words + original_female_words
            total_filtered = filtered_male_words + filtered_female_words

            self.results['filtering_stats'] = {
                'original_words': total_original,
                'filtered_words': total_filtered,
                'reduction_pct': ((total_original - total_filtered) / total_original * 100) if total_original > 0 else 0
            }

            print(f"  Male vocabulary: {len(male_freq):,} unique words")
            print(f"  Female vocabulary: {len(female_freq):,} unique words")

        elif self.dataset_type == 'gender-debates':
            # Debate-level gender analysis
            male_only_texts = data['male_only_debates']
            mixed_texts = data['mixed_debates']

            male_only_filtered = [self.filter.filter_text(t) for t in male_only_texts]
            mixed_filtered = [self.filter.filter_text(t) for t in mixed_texts]

            male_only_words = ' '.join(male_only_filtered).split()
            mixed_words = ' '.join(mixed_filtered).split()

            male_only_freq = Counter(male_only_words)
            mixed_freq = Counter(mixed_words)

            self.results['male_only_unigrams'] = male_only_freq.most_common(50)
            self.results['mixed_unigrams'] = mixed_freq.most_common(50)
            self.results['male_only_vocab_size'] = len(male_only_freq)
            self.results['mixed_vocab_size'] = len(mixed_freq)

            # Calculate filtering stats
            total_original = sum(len(t.split()) for t in male_only_texts + mixed_texts)
            total_filtered = len(male_only_words) + len(mixed_words)

            self.results['filtering_stats'] = {
                'original_words': total_original,
                'filtered_words': total_filtered,
                'reduction_pct': ((total_original - total_filtered) / total_original * 100) if total_original > 0 else 0
            }

            print(f"  Male-only debates vocabulary: {len(male_only_freq):,} unique words")
            print(f"  Mixed-gender debates vocabulary: {len(mixed_freq):,} unique words")

        else:
            # Overall corpus analysis
            texts = [d['text'] for d in data]
            filtered = [self.filter.filter_text(t) for t in texts]

            all_words = ' '.join(filtered).split()
            word_freq = Counter(all_words)

            self.results['unigrams'] = word_freq.most_common(50)
            self.results['vocab_size'] = len(word_freq)

            # Calculate filtering stats
            original_words = sum(len(d['text'].split()) for d in data)
            filtered_words = len(all_words)

            self.results['filtering_stats'] = {
                'original_words': original_words,
                'filtered_words': filtered_words,
                'reduction_pct': ((original_words - filtered_words) / original_words * 100) if original_words > 0 else 0
            }

            print(f"  Vocabulary: {len(word_freq):,} unique words")

    def _analyze_bigrams(self, data):
        """Analyze bigram frequencies"""
        if self.dataset_type == 'gender':
            male_texts = data['male_speeches']
            female_texts = data['female_speeches']

            male_bigrams = []
            for text in male_texts:
                filtered = self.filter.filter_text(text)
                bigrams = self.filter.extract_bigrams(filtered)
                male_bigrams.extend(bigrams)

            female_bigrams = []
            for text in female_texts:
                filtered = self.filter.filter_text(text)
                bigrams = self.filter.extract_bigrams(filtered)
                female_bigrams.extend(bigrams)

            male_bigram_freq = Counter(male_bigrams)
            female_bigram_freq = Counter(female_bigrams)

            self.results['male_bigrams'] = male_bigram_freq.most_common(30)
            self.results['female_bigrams'] = female_bigram_freq.most_common(30)

            print(f"  Male bigrams: {len(male_bigram_freq):,} unique")
            print(f"  Female bigrams: {len(female_bigram_freq):,} unique")

        elif self.dataset_type == 'gender-debates':
            male_only_bigrams = []
            for text in data['male_only_debates']:
                filtered = self.filter.filter_text(text)
                male_only_bigrams.extend(self.filter.extract_bigrams(filtered))

            mixed_bigrams = []
            for text in data['mixed_debates']:
                filtered = self.filter.filter_text(text)
                mixed_bigrams.extend(self.filter.extract_bigrams(filtered))

            self.results['male_only_bigrams'] = Counter(male_only_bigrams).most_common(30)
            self.results['mixed_bigrams'] = Counter(mixed_bigrams).most_common(30)

            print(f"  Male-only bigrams: {len(Counter(male_only_bigrams)):,} unique")
            print(f"  Mixed-gender bigrams: {len(Counter(mixed_bigrams)):,} unique")

        else:
            bigrams = []
            for debate in data:
                filtered = self.filter.filter_text(debate['text'])
                bigrams.extend(self.filter.extract_bigrams(filtered))

            bigram_freq = Counter(bigrams)
            self.results['bigrams'] = bigram_freq.most_common(30)

            print(f"  Bigrams: {len(bigram_freq):,} unique")

    def _analyze_topics(self, data, n_topics=8):
        """Perform LDA topic modeling"""
        if self.dataset_type == 'gender':
            # Separate topic modeling for each gender
            male_texts = [self.filter.filter_text(t) for t in data['male_speeches']]
            female_texts = [self.filter.filter_text(t) for t in data['female_speeches']]

            self.results['male_topics'] = self._run_topic_model(male_texts, n_topics)
            self.results['female_topics'] = self._run_topic_model(female_texts, n_topics)

            print(f"  Male topics: {len(self.results['male_topics'])} topics")
            print(f"  Female topics: {len(self.results['female_topics'])} topics")

        else:
            texts = [self.filter.filter_text(d['text']) for d in data]
            self.results['topics'] = self._run_topic_model(texts, n_topics)

            print(f"  Topics: {len(self.results['topics'])} topics")

    def _run_topic_model(self, texts, n_topics=8):
        """Run LDA topic modeling on texts"""
        # Filter valid texts
        valid_texts = [t for t in texts if len(t.split()) > 10]

        if len(valid_texts) < 50:
            print("    Insufficient data for topic modeling")
            return []

        try:
            vectorizer = TfidfVectorizer(
                max_features=500,
                min_df=3,
                max_df=0.8,
                ngram_range=(1, 2)
            )

            doc_term_matrix = vectorizer.fit_transform(valid_texts[:500])

            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20,
                learning_method='online'
            )
            lda.fit(doc_term_matrix)

            feature_names = vectorizer.get_feature_names_out()

            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                topic_weights = topic[top_indices].tolist()

                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': topic_weights
                })

                print(f"    Topic {topic_idx + 1}: {', '.join(top_words[:5])}")

            return topics

        except Exception as e:
            print(f"    Topic modeling failed: {e}")
            return []

    def _analyze_gender(self, data):
        """Analyze gender language patterns"""
        if self.dataset_type == 'gender':
            # Already have gender-separated data
            male_words_list = ' '.join(data['male_speeches']).lower().split()
            female_words_list = ' '.join(data['female_speeches']).lower().split()

            male_gendered = sum(1 for w in male_words_list if w in self.male_words)
            female_gendered = sum(1 for w in female_words_list if w in self.female_words)

            male_total_gendered = male_gendered + sum(1 for w in male_words_list if w in self.female_words)
            female_total_gendered = female_gendered + sum(1 for w in female_words_list if w in self.male_words)

            self.results['gender_analysis'] = {
                'male_using_male_words': male_gendered,
                'male_using_female_words': sum(1 for w in male_words_list if w in self.female_words),
                'female_using_male_words': sum(1 for w in female_words_list if w in self.male_words),
                'female_using_female_words': female_gendered,
                'male_gendered_ratio': male_gendered / male_total_gendered if male_total_gendered > 0 else 0,
                'female_gendered_ratio': female_gendered / female_total_gendered if female_total_gendered > 0 else 0
            }

            print(f"  Male speeches use male-gendered words: {male_gendered:,}")
            print(f"  Female speeches use female-gendered words: {female_gendered:,}")

        else:
            # Overall corpus - count gendered language
            all_words = []
            for debate in data:
                all_words.extend(debate['text'].lower().split())

            male_count = sum(1 for w in all_words if w in self.male_words)
            female_count = sum(1 for w in all_words if w in self.female_words)
            total_gendered = male_count + female_count

            self.results['gender_analysis'] = {
                'male_word_count': male_count,
                'female_word_count': female_count,
                'male_ratio': male_count / total_gendered if total_gendered > 0 else 0,
                'female_ratio': female_count / total_gendered if total_gendered > 0 else 0
            }

            print(f"  Male-gendered words: {male_count:,} ({male_count/total_gendered*100:.1f}%)")
            print(f"  Female-gendered words: {female_count:,} ({female_count/total_gendered*100:.1f}%)")

    def _analyze_temporal(self, data):
        """Analyze temporal trends (gender dataset only)"""
        if 'temporal_data' not in data:
            return

        self.results['temporal_data'] = data['temporal_data']
        print(f"  Temporal data: {len(data['temporal_data'])} years")

    def _create_visualizations(self):
        """Create all visualizations"""
        if self.dataset_type == 'gender-debates':
            # Debate-level gender visualizations
            if 'male_only_unigrams' in self.results and 'mixed_unigrams' in self.results:
                self.viz.create_unigram_comparison(
                    self.results['male_only_unigrams'],
                    self.results['mixed_unigrams'],
                    top_n=30,
                    output_name="debate_unigram_comparison.png"
                )

            if 'male_only_bigrams' in self.results and 'mixed_bigrams' in self.results:
                self.viz.create_bigram_comparison(
                    self.results['male_only_bigrams'],
                    self.results['mixed_bigrams'],
                    top_n=30,
                    output_name="debate_bigram_comparison.png"
                )

        elif self.dataset_type == 'gender':
            # Gender-specific visualizations (show 30 words, auto-filtered)
            if 'male_unigrams' in self.results and 'female_unigrams' in self.results:
                self.viz.create_unigram_comparison(
                    self.results['male_unigrams'],
                    self.results['female_unigrams'],
                    top_n=30  # Show more words for better insights
                )

            if 'male_bigrams' in self.results and 'female_bigrams' in self.results:
                self.viz.create_bigram_comparison(
                    self.results['male_bigrams'],
                    self.results['female_bigrams'],
                    top_n=30  # Show more bigrams
                )

            if 'temporal_data' in self.results:
                self.viz.create_temporal_participation(
                    self.results['temporal_data']
                )

            if 'male_topics' in self.results and 'female_topics' in self.results:
                # Create topic distribution visualization
                topics_data = {
                    'male_topics': self.results['male_topics'],
                    'female_topics': self.results['female_topics']
                }
                self.viz.create_topic_prevalence(topics_data)

        else:
            # Overall corpus visualizations
            if 'unigrams' in self.results:
                # Create single-column chart for overall corpus
                import matplotlib.pyplot as plt
                from professional_visualizations import COLORS, set_publication_style

                set_publication_style()

                # Top words chart
                words, counts = zip(*self.results['unigrams'][:30])
                fig, ax = plt.subplots(figsize=(10, 8))
                y_pos = range(len(words))
                ax.barh(y_pos, counts, color=COLORS['accent1'], alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(words, fontsize=10)
                ax.set_xlabel('Frequency', fontsize=11, fontweight='bold')
                ax.set_title('Top 30 Words - Overall Corpus', fontsize=12, fontweight='bold')
                ax.invert_yaxis()

                for i, count in enumerate(counts):
                    ax.text(count, i, f' {count:,}', va='center', fontsize=9)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='x', alpha=0.3)

                plt.tight_layout()
                output_path = self.output_dir / 'top_words.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                print(f"  Saved top words chart to {output_path}")

            if 'bigrams' in self.results:
                # Top bigrams chart
                import matplotlib.pyplot as plt
                from professional_visualizations import COLORS, set_publication_style

                set_publication_style()

                bigrams, counts = zip(*self.results['bigrams'][:20])
                labels = [' '.join(b) if isinstance(b, tuple) else b for b in bigrams]

                fig, ax = plt.subplots(figsize=(10, 6))
                y_pos = range(len(labels))
                ax.barh(y_pos, counts, color=COLORS['accent2'], alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=10)
                ax.set_xlabel('Frequency', fontsize=11, fontweight='bold')
                ax.set_title('Top 20 Bigrams - Overall Corpus', fontsize=12, fontweight='bold')
                ax.invert_yaxis()

                for i, count in enumerate(counts):
                    ax.text(count, i, f' {count:,}', va='center', fontsize=9)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='x', alpha=0.3)

                plt.tight_layout()
                output_path = self.output_dir / 'top_bigrams.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                print(f"  Saved top bigrams chart to {output_path}")

    def _save_results(self):
        """Save analysis results to JSON"""
        output_file = self.output_dir / 'analysis_results.json'

        # Prepare results for JSON (convert tuples to lists)
        json_results = {}
        for key, value in self.results.items():
            if key in ['male_unigrams', 'female_unigrams', 'unigrams',
                      'male_bigrams', 'female_bigrams', 'bigrams']:
                # Convert Counter items to lists
                json_results[key] = [
                    [list(item) if isinstance(item, tuple) else item, count]
                    for item, count in value
                ]
            else:
                json_results[key] = value

        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"  Results saved to {output_file}")


def compare_filtering_levels(dataset_type, year_range, sample_size):
    """
    Compare all filtering levels and generate comparison visualizations.

    Args:
        dataset_type: 'gender' or 'overall'
        year_range: Tuple of (start_year, end_year)
        sample_size: Sample size to use
    """
    print("\n" + "="*80)
    print("COMPARING FILTERING LEVELS")
    print("="*80 + "\n")

    levels = ['minimal', 'basic', 'parliamentary', 'moderate', 'aggressive']
    all_results = {}

    # Run analysis for each level
    for level in levels:
        print(f"\nRunning {level.upper()} filtering...")
        analyzer = UnifiedCorpusAnalyzer(
            dataset_type=dataset_type,
            filtering_level=level,
            output_dir=Paths.ANALYSIS_DIR / f"corpus_{dataset_type}_comparison" / level
        )

        results = analyzer.run_analysis(
            year_range=year_range,
            sample_size=sample_size,
            analysis_types=['unigram']  # Just unigrams for comparison
        )

        all_results[level] = results

    # Create comparison visualization
    print("\nCreating filtering comparison visualization...")
    viz = UnifiedVisualizationSuite(
        output_dir=Paths.ANALYSIS_DIR / f"corpus_{dataset_type}_comparison"
    )
    viz.create_filtering_comparison(all_results)

    print("\n" + "="*80)
    print("FILTERING COMPARISON COMPLETE!")
    print(f"Results saved to: {Paths.ANALYSIS_DIR / f'corpus_{dataset_type}_comparison'}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Unified Hansard Corpus Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Gender analysis with aggressive filtering
  python corpus_analysis.py --dataset gender --years 1920-1930 --sample 5000 --filtering aggressive

  # Overall corpus with moderate filtering
  python corpus_analysis.py --dataset overall --years 1920-1930 --sample 5000 --filtering moderate

  # Full corpus analysis
  python corpus_analysis.py --dataset gender --full --sample 50000 --filtering aggressive

  # Compare filtering levels
  python corpus_analysis.py --dataset gender --years 1920-1930 --sample 5000 --compare-filtering
        """
    )

    parser.add_argument('--dataset', type=str, required=True,
                       choices=['gender', 'overall', 'gender-debates'],
                       help='Dataset type: gender (speech-level), overall, or gender-debates (debate-level)')

    parser.add_argument('--years', type=str,
                       help='Year range (e.g., 1920-1930)')

    parser.add_argument('--full', action='store_true',
                       help='Use full corpus (1803-2005)')

    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size (default: use all data, no sampling)')

    parser.add_argument('--filtering', type=str, default='moderate',
                       choices=['minimal', 'basic', 'parliamentary', 'moderate', 'aggressive', 'ultra'],
                       help='Filtering level (default: moderate, ultra uses collaborator stopwords)')

    parser.add_argument('--analysis', type=str, default='all',
                       help='Comma-separated analysis types: unigram,bigram,topic,gender,temporal,all')

    parser.add_argument('--compare-filtering', action='store_true',
                       help='Compare all filtering levels')

    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory')

    args = parser.parse_args()

    # Parse year range
    year_range = None
    if args.years:
        try:
            parts = args.years.split('-')
            year_range = (int(parts[0]), int(parts[1]))
        except:
            print(f"Error: Invalid year range format: {args.years}")
            print("Use format: YYYY-YYYY (e.g., 1920-1930)")
            sys.exit(1)
    elif args.full:
        year_range = (1803, 2005)

    # Parse analysis types
    analysis_types = args.analysis.split(',') if args.analysis != 'all' else ['all']

    # Run analysis
    if args.compare_filtering:
        compare_filtering_levels(args.dataset, year_range, args.sample)
    else:
        analyzer = UnifiedCorpusAnalyzer(
            dataset_type=args.dataset,
            filtering_level=args.filtering,
            output_dir=Path(args.output_dir) if args.output_dir else None
        )

        analyzer.run_analysis(
            year_range=year_range,
            sample_size=args.sample,
            analysis_types=analysis_types
        )


if __name__ == "__main__":
    main()
