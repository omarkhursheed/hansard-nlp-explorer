#!/usr/bin/env python3
"""
Unified Hansard Milestone Analysis

Single parameterized script for analyzing historical milestones, replacing:
- comprehensive_milestone_analysis.py
- gender_milestone_analysis.py

Analyzes parliamentary discourse before, during (if applicable), and after
key historical events to identify changes in vocabulary, topics, and gender representation.

Built-in Milestones:
- 1918_partial_suffrage: Partial women's suffrage + first women in Parliament
- 1928_full_suffrage: Equal Franchise Act
- ww1_period: World War I (pre/during/post)
- ww2_period: World War II (pre/during/post)
- thatcher_period: Margaret Thatcher era

Usage:
    # Single milestone
    python milestone_analysis.py --milestone 1928_full_suffrage --dataset gender --filtering aggressive

    # All milestones
    python milestone_analysis.py --all-milestones --dataset gender --filtering moderate

    # Custom milestone
    python milestone_analysis.py --custom --name "Brexit" --year 2016 \
        --pre-window 2010-2016 --post-window 2016-2020 --dataset overall
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

# Import unified modules
sys.path.insert(0, str(Path(__file__).parent))
from unified_text_filtering import HansardTextFilter
from unified_corpus_loader import UnifiedCorpusLoader
from professional_visualizations import UnifiedVisualizationSuite

# Import path configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.path_config import Paths


# Built-in milestone definitions
MILESTONES = {
    "1918_partial_suffrage": {
        "name": "1918 Partial Women's Suffrage & Parliament Entry",
        "description": "Women over 30 gain vote; first women allowed in Parliament",
        "year": 1918,
        "pre_window": (1908, 1918),
        "post_window": (1918, 1928),
        "sample_size": 5000
    },
    "1928_full_suffrage": {
        "name": "1928 Full Women's Suffrage",
        "description": "Voting age for women reduced to 21 (Equal Franchise Act)",
        "year": 1928,
        "pre_window": (1918, 1928),
        "post_window": (1928, 1938),
        "sample_size": 5000
    },
    "ww1_period": {
        "name": "World War I Analysis",
        "description": "Parliamentary discourse before, during, and after WWI",
        "year": 1916,
        "pre_window": (1909, 1914),
        "during_window": (1914, 1918),
        "post_window": (1918, 1923),
        "sample_size": 5000
    },
    "ww2_period": {
        "name": "World War II Analysis",
        "description": "Parliamentary discourse before, during, and after WWII",
        "year": 1942,
        "pre_window": (1934, 1939),
        "during_window": (1939, 1945),
        "post_window": (1945, 1950),
        "sample_size": 5000
    },
    "thatcher_period": {
        "name": "Thatcher Era Analysis",
        "description": "Parliamentary discourse before, during, and after Thatcher's tenure",
        "year": 1984,
        "pre_window": (1974, 1979),
        "during_window": (1979, 1990),
        "post_window": (1990, 1995),
        "sample_size": 5000
    }
}


class UnifiedMilestoneAnalyzer:
    """
    Unified milestone analyzer for Hansard parliamentary debates.

    Analyzes changes in parliamentary discourse around historical milestones.
    """

    def __init__(self, dataset_type: str, filtering_level: str,
                 output_dir: Path = None):
        """
        Initialize analyzer.

        Args:
            dataset_type: 'gender' or 'overall'
            filtering_level: Filtering level name
            output_dir: Custom output directory
        """
        self.dataset_type = dataset_type
        self.filtering_level = filtering_level

        # Initialize components
        self.loader = UnifiedCorpusLoader(dataset_type=dataset_type)
        self.filter = HansardTextFilter(level=filtering_level)

        # Set base output directory
        if output_dir:
            self.base_output_dir = Path(output_dir)
        else:
            self.base_output_dir = Paths.ANALYSIS_DIR / f"milestones_{dataset_type}"

        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Load gender wordlists
        self.male_words = self._load_gender_wordlist(Paths.MALE_WORDS)
        self.female_words = self._load_gender_wordlist(Paths.FEMALE_WORDS)

    def _load_gender_wordlist(self, filepath: Path) -> set:
        """Load gender wordlist"""
        try:
            with open(filepath, 'r') as f:
                return set(word.strip().lower() for word in f if word.strip())
        except FileNotFoundError:
            print(f"Warning: Gender wordlist not found at {filepath}")
            return set()

    def analyze_milestone(self, milestone_key: str = None, milestone_def: dict = None,
                         sample_size: int = None) -> dict:
        """
        Analyze a specific historical milestone.

        Args:
            milestone_key: Key for built-in milestone
            milestone_def: Custom milestone definition
            sample_size: Override default sample size

        Returns:
            Dict with milestone analysis results
        """
        # Get milestone definition
        if milestone_key:
            if milestone_key not in MILESTONES:
                raise ValueError(f"Unknown milestone: {milestone_key}. "
                               f"Available: {', '.join(MILESTONES.keys())}")
            milestone = MILESTONES[milestone_key].copy()
        elif milestone_def:
            milestone = milestone_def
        else:
            raise ValueError("Must provide either milestone_key or milestone_def")

        # Override sample size if provided
        if sample_size:
            milestone['sample_size'] = sample_size

        print("\n" + "="*80)
        print(f"ANALYZING: {milestone['name']}")
        print(f"Dataset: {self.dataset_type}")
        print(f"Filtering: {self.filtering_level}")
        print("="*80 + "\n")

        # Create milestone-specific output directory
        milestone_name = milestone_key if milestone_key else milestone['name'].replace(' ', '_')
        output_dir = self.base_output_dir / milestone_name / self.filtering_level
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and analyze periods
        print("Loading pre-period...")
        pre_data = self.loader.load_debates(
            year_range=milestone['pre_window'],
            sample_size=milestone.get('sample_size')
        )
        pre_results = self._analyze_period(pre_data, "pre")

        # During period (if exists)
        during_results = None
        if 'during_window' in milestone:
            print("\nLoading during-period...")
            during_data = self.loader.load_debates(
                year_range=milestone['during_window'],
                sample_size=milestone.get('sample_size')
            )
            during_results = self._analyze_period(during_data, "during")

        # Post period
        print("\nLoading post-period...")
        post_data = self.loader.load_debates(
            year_range=milestone['post_window'],
            sample_size=milestone.get('sample_size')
        )
        post_results = self._analyze_period(post_data, "post")

        # Compare periods
        print("\nComparing periods...")
        comparison = self._compare_periods(pre_results, post_results, during_results)

        # Compile results
        milestone_results = {
            "milestone_info": milestone,
            "filtering_level": self.filtering_level,
            "dataset_type": self.dataset_type,
            "timestamp": datetime.now().isoformat(),
            "pre_period": pre_results,
            "post_period": post_results,
            "comparison": comparison
        }

        if during_results:
            milestone_results["during_period"] = during_results

        # Create visualizations
        print("\nCreating visualizations...")
        self._create_visualizations(milestone_results, output_dir)

        # Generate report
        print("Generating report...")
        self._generate_report(milestone_results, output_dir)

        # Save results
        results_file = output_dir / "milestone_results.json"
        with open(results_file, 'w') as f:
            json_results = self._prepare_for_json(milestone_results)
            json.dump(json_results, f, indent=2)

        print(f"\nResults saved to {results_file}")
        print(f"Visualizations saved to {output_dir}")

        return milestone_results

    def _analyze_period(self, data, period_name: str) -> dict:
        """
        Analyze a single period (pre/during/post).

        Args:
            data: Loaded data (dict for gender, list for overall)
            period_name: Name of period

        Returns:
            Dict with period analysis results
        """
        print(f"  Analyzing {period_name} period...")

        results = {'period': period_name}

        if self.dataset_type == 'gender':
            # Gender dataset analysis
            male_texts = data['male_speeches']
            female_texts = data['female_speeches']

            if not male_texts and not female_texts:
                print(f"    No data for {period_name} period")
                return results

            # Filter texts
            male_filtered = [self.filter.filter_text(t) for t in male_texts]
            female_filtered = [self.filter.filter_text(t) for t in female_texts]

            # Unigram analysis
            male_words = ' '.join(male_filtered).split()
            female_words = ' '.join(female_filtered).split()

            male_freq = Counter(male_words)
            female_freq = Counter(female_words)

            results['male_unigrams'] = male_freq.most_common(30)
            results['female_unigrams'] = female_freq.most_common(30)

            # Bigram analysis
            male_bigrams = []
            for text in male_filtered:
                male_bigrams.extend(self.filter.extract_bigrams(text))

            female_bigrams = []
            for text in female_filtered:
                female_bigrams.extend(self.filter.extract_bigrams(text))

            results['male_bigrams'] = Counter(male_bigrams).most_common(20)
            results['female_bigrams'] = Counter(female_bigrams).most_common(20)

            # Gender language analysis
            results['gender_analysis'] = self._analyze_gender_language_gender_dataset(data)

            # Metadata
            results['total_male_speeches'] = len(male_texts)
            results['total_female_speeches'] = len(female_texts)
            results['years'] = sorted(set(td['year'] for td in data.get('temporal_data', [])))

        else:
            # Overall corpus analysis
            if not data:
                print(f"    No data for {period_name} period")
                return results

            # Filter texts
            filtered = [self.filter.filter_text(d['text']) for d in data]

            # Unigram analysis
            all_words = ' '.join(filtered).split()
            word_freq = Counter(all_words)

            results['top_unigrams'] = word_freq.most_common(30)

            # Bigram analysis
            all_bigrams = []
            for text in filtered:
                all_bigrams.extend(self.filter.extract_bigrams(text))

            results['top_bigrams'] = Counter(all_bigrams).most_common(20)

            # Gender language analysis
            results['gender_analysis'] = self._analyze_gender_language_overall(data)

            # Metadata
            results['total_debates'] = len(data)
            results['years'] = sorted(set(d['year'] for d in data))

        # Filtering mode
        results['filtering_mode'] = self.filtering_level

        print(f"    Processed {period_name} period: " +
              f"{results.get('total_debates', results.get('total_male_speeches', 0) + results.get('total_female_speeches', 0))} items")

        return results

    def _analyze_gender_language_gender_dataset(self, data) -> dict:
        """Analyze gender language for gender dataset"""
        male_words_list = ' '.join(data['male_speeches']).lower().split()
        female_words_list = ' '.join(data['female_speeches']).lower().split()

        male_gendered = sum(1 for w in male_words_list if w in self.male_words)
        female_gendered = sum(1 for w in female_words_list if w in self.female_words)

        male_total = male_gendered + sum(1 for w in male_words_list if w in self.female_words)
        female_total = female_gendered + sum(1 for w in female_words_list if w in self.male_words)

        return {
            'male_word_count': male_gendered,
            'female_word_count': female_gendered,
            'male_ratio': male_gendered / male_total if male_total > 0 else 0,
            'female_ratio': female_gendered / female_total if female_total > 0 else 0
        }

    def _analyze_gender_language_overall(self, debates) -> dict:
        """Analyze gender language for overall corpus"""
        all_words = []
        for debate in debates:
            all_words.extend(debate['text'].lower().split())

        male_count = sum(1 for w in all_words if w in self.male_words)
        female_count = sum(1 for w in all_words if w in self.female_words)
        total_gendered = male_count + female_count

        return {
            'male_word_count': male_count,
            'female_word_count': female_count,
            'male_ratio': male_count / total_gendered if total_gendered > 0 else 0,
            'female_ratio': female_count / total_gendered if total_gendered > 0 else 0
        }

    def _compare_periods(self, pre_results: dict, post_results: dict,
                        during_results: dict = None) -> dict:
        """
        Compare periods and calculate changes.

        Args:
            pre_results: Pre-period results
            post_results: Post-period results
            during_results: During-period results (optional)

        Returns:
            Dict with comparison metrics
        """
        comparison = {}

        # Gender language changes
        if 'gender_analysis' in pre_results and 'gender_analysis' in post_results:
            pre_female = pre_results['gender_analysis']['female_ratio'] * 100
            post_female = post_results['gender_analysis']['female_ratio'] * 100

            comparison['gender_language_change'] = {
                'pre_female_pct': pre_female,
                'post_female_pct': post_female,
                'change_pp': post_female - pre_female
            }

        # Content evolution
        if self.dataset_type == 'gender':
            if 'male_unigrams' in pre_results and 'male_unigrams' in post_results:
                pre_words = set([w for w, _ in pre_results['male_unigrams'][:20]])
                post_words = set([w for w, _ in post_results['male_unigrams'][:20]])

                comparison['male_content_evolution'] = {
                    'new_words': list(post_words - pre_words),
                    'disappeared_words': list(pre_words - post_words),
                    'persistent_words': list(pre_words & post_words)
                }

            if 'female_unigrams' in pre_results and 'female_unigrams' in post_results:
                pre_words = set([w for w, _ in pre_results['female_unigrams'][:20]])
                post_words = set([w for w, _ in post_results['female_unigrams'][:20]])

                comparison['female_content_evolution'] = {
                    'new_words': list(post_words - pre_words),
                    'disappeared_words': list(pre_words - post_words),
                    'persistent_words': list(pre_words & post_words)
                }
        else:
            if 'top_unigrams' in pre_results and 'top_unigrams' in post_results:
                pre_words = set([w for w, _ in pre_results['top_unigrams'][:20]])
                post_words = set([w for w, _ in post_results['top_unigrams'][:20]])

                comparison['content_evolution'] = {
                    'new_words': list(post_words - pre_words),
                    'disappeared_words': list(pre_words - post_words),
                    'persistent_words': list(pre_words & post_words)
                }

        return comparison

    def _create_visualizations(self, results: dict, output_dir: Path):
        """Create milestone visualizations"""
        viz = UnifiedVisualizationSuite(output_dir=output_dir)

        # Get period data
        pre_data = results['pre_period']
        post_data = results['post_period']
        milestone_info = results['milestone_info']

        # Create comprehensive comparison
        viz.create_milestone_comparison(pre_data, post_data, milestone_info,
                                       output_name="milestone_impact.png")

        # Create period-specific visualizations for gender dataset
        if self.dataset_type == 'gender':
            # Pre-period vocabulary
            if 'male_unigrams' in pre_data and 'female_unigrams' in pre_data:
                viz.create_unigram_comparison(
                    pre_data['male_unigrams'],
                    pre_data['female_unigrams'],
                    output_name="pre_period_vocabulary.png"
                )

            # Post-period vocabulary
            if 'male_unigrams' in post_data and 'female_unigrams' in post_data:
                viz.create_unigram_comparison(
                    post_data['male_unigrams'],
                    post_data['female_unigrams'],
                    output_name="post_period_vocabulary.png"
                )

    def _generate_report(self, results: dict, output_dir: Path):
        """Generate markdown report"""
        milestone = results['milestone_info']
        comparison = results['comparison']

        report_path = output_dir / "milestone_report.md"

        with open(report_path, 'w') as f:
            f.write(f"# {milestone['name']}\n\n")
            f.write(f"**Milestone Year:** {milestone['year']}\n")
            f.write(f"**Dataset:** {results['dataset_type']}\n")
            f.write(f"**Filtering:** {results['filtering_level']}\n")
            f.write(f"**Analysis Date:** {results['timestamp']}\n\n")

            f.write(f"## Overview\n\n")
            f.write(f"{milestone['description']}\n\n")

            # Key findings
            f.write("## Key Findings\n\n")

            if 'gender_language_change' in comparison:
                gc = comparison['gender_language_change']
                f.write(f"### Gender Language Evolution\n")
                f.write(f"- Pre-{milestone['year']}: {gc['pre_female_pct']:.2f}% female language\n")
                f.write(f"- Post-{milestone['year']}: {gc['post_female_pct']:.2f}% female language\n")
                f.write(f"- **Change:** {gc['change_pp']:+.2f} percentage points\n\n")

            # Content evolution
            if 'male_content_evolution' in comparison:
                mce = comparison['male_content_evolution']
                f.write(f"### Male MPs - Content Evolution\n")
                if mce['new_words']:
                    f.write(f"**New prominent words:** {', '.join(mce['new_words'][:10])}\n")
                if mce['disappeared_words']:
                    f.write(f"**Declining words:** {', '.join(mce['disappeared_words'][:10])}\n\n")

            if 'female_content_evolution' in comparison:
                fce = comparison['female_content_evolution']
                f.write(f"### Female MPs - Content Evolution\n")
                if fce['new_words']:
                    f.write(f"**New prominent words:** {', '.join(fce['new_words'][:10])}\n")
                if fce['disappeared_words']:
                    f.write(f"**Declining words:** {', '.join(fce['disappeared_words'][:10])}\n\n")

            if 'content_evolution' in comparison:
                ce = comparison['content_evolution']
                f.write(f"### Overall Content Evolution\n")
                if ce['new_words']:
                    f.write(f"**New prominent words:** {', '.join(ce['new_words'][:10])}\n")
                if ce['disappeared_words']:
                    f.write(f"**Declining words:** {', '.join(ce['disappeared_words'][:10])}\n\n")

            # Period details
            f.write("## Period Details\n\n")

            periods = [('pre_period', 'Pre'), ('post_period', 'Post')]
            if 'during_period' in results:
                periods.insert(1, ('during_period', 'During'))

            for period_key, period_label in periods:
                period = results[period_key]
                f.write(f"### {period_label}-{milestone['year']} Period\n")
                f.write(f"**Years:** {period['years'][0]}-{period['years'][-1]}\n")

                if self.dataset_type == 'gender':
                    f.write(f"**Male speeches:** {period['total_male_speeches']:,}\n")
                    f.write(f"**Female speeches:** {period['total_female_speeches']:,}\n")
                else:
                    f.write(f"**Debates:** {period['total_debates']:,}\n")

                f.write("\n")

        print(f"Report saved to {report_path}")

    def _prepare_for_json(self, data):
        """Prepare data for JSON serialization"""
        if isinstance(data, dict):
            return {key: self._prepare_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, tuple):
            return list(data)
        else:
            return data

    def analyze_all_milestones(self, sample_size: int = None) -> dict:
        """
        Analyze all built-in milestones.

        Args:
            sample_size: Override default sample sizes

        Returns:
            Dict mapping milestone keys to results
        """
        print("\n" + "="*80)
        print(f"ANALYZING ALL MILESTONES")
        print(f"Dataset: {self.dataset_type}")
        print(f"Filtering: {self.filtering_level}")
        print("="*80)

        all_results = {}

        for milestone_key in MILESTONES.keys():
            print(f"\n{'='*80}")
            results = self.analyze_milestone(milestone_key=milestone_key,
                                            sample_size=sample_size)
            all_results[milestone_key] = results

        # Generate master summary
        print("\nGenerating master summary...")
        self._generate_master_summary(all_results)

        print("\n" + "="*80)
        print("ALL MILESTONES ANALYZED!")
        print(f"Results saved to: {self.base_output_dir}")
        print("="*80 + "\n")

        return all_results

    def _generate_master_summary(self, all_results: dict):
        """Generate master summary across all milestones"""
        summary_path = self.base_output_dir / f"MASTER_SUMMARY.md"

        with open(summary_path, 'w') as f:
            f.write(f"# Historical Milestones - Master Summary\n\n")
            f.write(f"**Dataset:** {self.dataset_type}\n")
            f.write(f"**Filtering:** {self.filtering_level}\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")

            for milestone_key, results in all_results.items():
                if not results or 'milestone_info' not in results:
                    continue

                milestone = results['milestone_info']
                comparison = results['comparison']

                f.write(f"## {milestone['name']}\n")
                f.write(f"**Year:** {milestone['year']}\n\n")

                if 'gender_language_change' in comparison:
                    change = comparison['gender_language_change']['change_pp']
                    f.write(f"- Female language change: {change:+.2f}pp\n")

                if 'male_content_evolution' in comparison:
                    mce = comparison['male_content_evolution']
                    if mce['new_words']:
                        f.write(f"- Male MPs new themes: {', '.join(mce['new_words'][:5])}\n")

                if 'female_content_evolution' in comparison:
                    fce = comparison['female_content_evolution']
                    if fce['new_words']:
                        f.write(f"- Female MPs new themes: {', '.join(fce['new_words'][:5])}\n")

                f.write("\n")

        print(f"Master summary saved to {summary_path}")


def create_custom_milestone(name: str, year: int, pre_window: str,
                           post_window: str, during_window: str = None,
                           sample_size: int = 5000) -> dict:
    """
    Create custom milestone definition.

    Args:
        name: Milestone name
        year: Milestone year
        pre_window: Pre-period as "START-END"
        post_window: Post-period as "START-END"
        during_window: During-period as "START-END" (optional)
        sample_size: Sample size

    Returns:
        Milestone definition dict
    """
    def parse_window(window_str):
        parts = window_str.split('-')
        return (int(parts[0]), int(parts[1]))

    milestone = {
        "name": name,
        "year": year,
        "pre_window": parse_window(pre_window),
        "post_window": parse_window(post_window),
        "sample_size": sample_size
    }

    if during_window:
        milestone["during_window"] = parse_window(during_window)

    return milestone


def main():
    parser = argparse.ArgumentParser(
        description='Unified Hansard Milestone Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single milestone
  python milestone_analysis.py --milestone 1928_full_suffrage --dataset gender --filtering aggressive

  # All milestones
  python milestone_analysis.py --all-milestones --dataset gender --filtering moderate

  # Custom milestone
  python milestone_analysis.py --custom --name "Brexit" --year 2016 \
      --pre-window 2010-2016 --post-window 2016-2020 --dataset overall

Available milestones:
  - 1918_partial_suffrage: Partial women's suffrage
  - 1928_full_suffrage: Equal Franchise Act
  - ww1_period: World War I
  - ww2_period: World War II
  - thatcher_period: Margaret Thatcher era
        """
    )

    parser.add_argument('--dataset', type=str, required=True,
                       choices=['gender', 'overall'],
                       help='Dataset type')

    parser.add_argument('--milestone', type=str,
                       help='Built-in milestone to analyze')

    parser.add_argument('--all-milestones', action='store_true',
                       help='Analyze all built-in milestones')

    parser.add_argument('--custom', action='store_true',
                       help='Define custom milestone')

    parser.add_argument('--name', type=str,
                       help='Custom milestone name')

    parser.add_argument('--year', type=int,
                       help='Custom milestone year')

    parser.add_argument('--pre-window', type=str,
                       help='Pre-period years (e.g., 2010-2016)')

    parser.add_argument('--post-window', type=str,
                       help='Post-period years (e.g., 2016-2020)')

    parser.add_argument('--during-window', type=str,
                       help='During-period years (optional)')

    parser.add_argument('--filtering', type=str, default='moderate',
                       choices=['minimal', 'basic', 'parliamentary', 'moderate', 'aggressive'],
                       help='Filtering level (default: moderate)')

    parser.add_argument('--sample', type=int,
                       help='Override default sample size')

    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = UnifiedMilestoneAnalyzer(
        dataset_type=args.dataset,
        filtering_level=args.filtering,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )

    # Run analysis
    if args.all_milestones:
        analyzer.analyze_all_milestones(sample_size=args.sample)
    elif args.milestone:
        analyzer.analyze_milestone(milestone_key=args.milestone,
                                   sample_size=args.sample)
    elif args.custom:
        if not all([args.name, args.year, args.pre_window, args.post_window]):
            print("Error: Custom milestone requires --name, --year, --pre-window, --post-window")
            sys.exit(1)

        milestone_def = create_custom_milestone(
            name=args.name,
            year=args.year,
            pre_window=args.pre_window,
            post_window=args.post_window,
            during_window=args.during_window,
            sample_size=args.sample or 5000
        )

        analyzer.analyze_milestone(milestone_def=milestone_def)
    else:
        print("Error: Must specify --milestone, --all-milestones, or --custom")
        print("Use --help for more information")
        sys.exit(1)


if __name__ == "__main__":
    main()
