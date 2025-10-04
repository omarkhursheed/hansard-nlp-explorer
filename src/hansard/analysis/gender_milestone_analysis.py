#!/usr/bin/env python3
"""
Gender-Matched Historical Milestone Analysis for Hansard Parliamentary Debates

Analyzes key historical periods using only confirmed MP speeches with known gender,
revealing how male and female MPs discussed pivotal moments differently.

Key Milestones:
- 1918: Partial women's suffrage + First women in Parliament
- 1928: Full women's suffrage
- 1914-1918: World War I
- 1939-1945: World War II
- 1979-1990: Margaret Thatcher's tenure as PM

Usage:
    python gender_milestone_analysis.py --all
    python gender_milestone_analysis.py --milestone ww2_period
    python gender_milestone_analysis.py --milestone 1928_full_suffrage --filtering aggressive
"""

import argparse
import json
import os
import re
import glob
from collections import Counter
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Import path configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.path_config import Paths

class GenderMilestoneAnalyzer:
    def __init__(self):
        # Use centralized paths
        self.data_dir = Paths.GENDER_ENHANCED_DATA
        self.output_dir = Paths.MILESTONE_RESULTS
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load gender wordlists using centralized paths
        self.male_words = self._load_gender_wordlist(Paths.MALE_WORDS)
        self.female_words = self._load_gender_wordlist(Paths.FEMALE_WORDS)

        # Define filtering modes
        self.filtering_modes = {
            "none": {"stop_words": set(), "min_len": 2},
            "basic": {"stop_words": self._get_basic_stop_words(), "min_len": 3},
            "parliamentary": {"stop_words": self._get_parliamentary_stop_words(), "min_len": 3},
            "moderate": {"stop_words": self._get_moderate_stop_words(), "min_len": 3},
            "aggressive": {"stop_words": self._get_aggressive_stop_words(), "min_len": 3}
        }

        # Define historical milestones with analysis windows
        self.milestones = {
            "1918_partial_suffrage": {
                "name": "1918 Partial Women's Suffrage & Parliament Entry",
                "description": "Women over 30 gain vote; first women allowed in Parliament",
                "milestone_year": 1918,
                "pre_window": (1908, 1917),
                "post_window": (1919, 1928),
                "sample_size": 5000
            },
            "1928_full_suffrage": {
                "name": "1928 Full Women's Suffrage",
                "description": "Voting age for women reduced to 21 (Equal Franchise Act)",
                "milestone_year": 1928,
                "pre_window": (1918, 1927),
                "post_window": (1929, 1938),
                "sample_size": 5000
            },
            "ww1_period": {
                "name": "World War I Analysis",
                "description": "Parliamentary discourse before, during, and after WWI",
                "milestone_year": 1916,
                "pre_window": (1909, 1913),
                "during_window": (1914, 1918),
                "post_window": (1919, 1923),
                "sample_size": 5000
            },
            "ww2_period": {
                "name": "World War II Analysis",
                "description": "Parliamentary discourse before, during, and after WWII",
                "milestone_year": 1942,
                "pre_window": (1934, 1938),
                "during_window": (1939, 1945),
                "post_window": (1946, 1950),
                "sample_size": 5000
            },
            "thatcher_period": {
                "name": "Thatcher Era Analysis",
                "description": "Parliamentary discourse before, during, and after Thatcher's tenure",
                "milestone_year": 1984,
                "pre_window": (1974, 1978),
                "during_window": (1979, 1990),
                "post_window": (1991, 1995),
                "sample_size": 5000
            }
        }

    def _load_gender_wordlist(self, filepath):
        """Load gender wordlist and convert to lowercase set"""
        try:
            with open(filepath, 'r') as f:
                return set(word.strip().lower() for word in f if word.strip())
        except FileNotFoundError:
            print(f"Warning: Gender wordlist not found at {filepath}")
            return set()

    def _get_basic_stop_words(self):
        """Basic English stop words"""
        return set(['the', 'of', 'to', 'and', 'a', 'an', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for',
                   'on', 'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from',
                   'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when'])

    def _get_parliamentary_stop_words(self):
        """Parliamentary procedural terms"""
        basic = self._get_basic_stop_words()
        parliamentary = {'hon', 'right', 'gentleman', 'member', 'members', 'house', 'speaker', 'sir',
                        'lord', 'lords', 'gallant', 'learned', 'friend', 'friends', 'noble', 'bill',
                        'clause', 'amendment', 'committee', 'order', 'question', 'division', 'read'}
        return basic | parliamentary

    def _get_moderate_stop_words(self):
        """Moderate filtering"""
        parliamentary = self._get_parliamentary_stop_words()
        moderate = {'government', 'minister', 'secretary', 'state', 'department', 'office', 'time',
                   'year', 'years', 'day', 'way', 'case', 'point', 'matter', 'place', 'hand', 'part'}
        return parliamentary | moderate

    def _get_aggressive_stop_words(self):
        """Aggressive filtering"""
        moderate = self._get_moderate_stop_words()
        aggressive = {'think', 'know', 'say', 'well', 'good', 'right', 'may', 'now', 'see', 'come',
                     'take', 'make', 'give', 'look', 'want', 'go', 'get', 'put', 'let', 'find'}
        return moderate | aggressive

    def load_period_data(self, start_year, end_year, sample_size=None):
        """Load gender-matched data for a specific period"""
        # Use centralized path method
        year_files = Paths.get_year_files(start_year, end_year)
        period_files = []

        for file_path in year_files:
            year_match = re.search(r'debates_(\d{4})_enhanced\.parquet', file_path.name)
            if year_match:
                year = int(year_match.group(1))
                period_files.append((year, str(file_path)))

        period_files.sort()

        if not period_files:
            return None

        # Collect speeches
        male_speeches = []
        female_speeches = []
        total_debates = 0

        for year, file_path in period_files:
            try:
                df = pd.read_parquet(file_path)

                for _, row in df.iterrows():
                    if 'speech_segments' in row and row['speech_segments'] is not None:
                        # Get speaker details for gender mapping
                        speaker_details = row.get('speaker_details', [])
                        speaker_gender_map = {}
                        if isinstance(speaker_details, (list, np.ndarray)):
                            for detail in speaker_details:
                                if isinstance(detail, dict):
                                    name = detail.get('original_name', '')
                                    gender = detail.get('gender', '').lower()
                                    if name and gender in ['m', 'f']:
                                        speaker_gender_map[name] = 'male' if gender == 'm' else 'female'

                        # Process speech segments
                        segments = row['speech_segments']
                        if isinstance(segments, (list, np.ndarray)):
                            for segment in segments:
                                if isinstance(segment, dict):
                                    speaker = segment.get('speaker', '')
                                    text = segment.get('text', '')

                                    # Map speaker to gender
                                    gender = speaker_gender_map.get(speaker)
                                    if not gender:
                                        # Try to match partial names
                                        for orig_name, g in speaker_gender_map.items():
                                            if speaker in orig_name or orig_name in speaker:
                                                gender = g
                                                break

                                    if gender == 'male' and text:
                                        male_speeches.append(text)
                                    elif gender == 'female' and text:
                                        female_speeches.append(text)

                        total_debates += 1

                        # Early exit if we have enough samples
                        if sample_size and len(male_speeches) + len(female_speeches) >= sample_size:
                            break

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

            if sample_size and len(male_speeches) + len(female_speeches) >= sample_size:
                break

        # Sample if needed
        if sample_size:
            import random
            random.seed(42)

            if len(male_speeches) > sample_size * 0.9:  # Allow up to 90% male speeches
                male_speeches = random.sample(male_speeches, int(sample_size * 0.9))
            if len(female_speeches) > sample_size * 0.1:  # At least 10% female speeches if available
                female_speeches = random.sample(female_speeches, min(len(female_speeches), int(sample_size * 0.1)))

        return {
            'male_speeches': male_speeches,
            'female_speeches': female_speeches,
            'total_debates': total_debates,
            'period': f"{start_year}-{end_year}"
        }

    def filter_text(self, text, filtering_mode="basic"):
        """Apply filtering to text"""
        if not text:
            return ""

        params = self.filtering_modes.get(filtering_mode, self.filtering_modes["basic"])
        stop_words = params['stop_words']
        min_len = params['min_len']

        words = text.lower().split()
        filtered = []
        for word in words:
            word = re.sub(r'[^\w\s]', '', word)
            if word and len(word) >= min_len and word not in stop_words:
                filtered.append(word)

        return ' '.join(filtered)

    def analyze_period(self, period_data, period_name, filtering_mode="basic"):
        """Analyze a specific period"""
        if not period_data:
            return None

        # Filter texts
        filtered_male = [self.filter_text(text, filtering_mode) for text in period_data['male_speeches']]
        filtered_female = [self.filter_text(text, filtering_mode) for text in period_data['female_speeches']]

        # Remove empty strings
        filtered_male = [t for t in filtered_male if t]
        filtered_female = [t for t in filtered_female if t]

        # Word frequency analysis
        male_words = ' '.join(filtered_male).split() if filtered_male else []
        female_words = ' '.join(filtered_female).split() if filtered_female else []

        male_freq = Counter(male_words)
        female_freq = Counter(female_words)

        # Combined frequency
        all_words = male_words + female_words
        combined_freq = Counter(all_words)

        results = {
            'period': period_name,
            'period_range': period_data['period'],
            'total_debates': period_data['total_debates'],
            'male_speeches': len(period_data['male_speeches']),
            'female_speeches': len(period_data['female_speeches']),
            'male_words': len(male_words),
            'female_words': len(female_words),
            'top_words': combined_freq.most_common(30),
            'top_male_words': male_freq.most_common(30),
            'top_female_words': female_freq.most_common(30) if female_freq else []
        }

        # Topic modeling if enough content
        if len(filtered_male) > 50:
            try:
                vectorizer = TfidfVectorizer(max_features=50, min_df=2, max_df=0.8)
                tfidf_matrix = vectorizer.fit_transform(filtered_male)

                lda = LatentDirichletAllocation(n_components=3, random_state=42)
                lda.fit(tfidf_matrix)

                feature_names = vectorizer.get_feature_names_out()
                male_topics = []
                for topic in lda.components_:
                    top_indices = topic.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_indices]
                    male_topics.append(top_words)
                results['male_topics'] = male_topics
            except:
                pass

        # Female topics if enough content
        if len(filtered_female) > 20:  # Lower threshold for female speeches
            try:
                vectorizer = TfidfVectorizer(max_features=30, min_df=1, max_df=0.9)
                tfidf_matrix = vectorizer.fit_transform(filtered_female)

                lda = LatentDirichletAllocation(n_components=2, random_state=42)
                lda.fit(tfidf_matrix)

                feature_names = vectorizer.get_feature_names_out()
                female_topics = []
                for topic in lda.components_:
                    top_indices = topic.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_indices]
                    female_topics.append(top_words)
                results['female_topics'] = female_topics
            except:
                pass

        return results

    def analyze_milestone(self, milestone_key, filtering_mode="aggressive", force=False):
        """Analyze a specific historical milestone"""
        milestone = self.milestones[milestone_key]

        # Check if results already exist
        output_file = self.output_dir / f"{milestone_key}_{filtering_mode}_results.json"
        if output_file.exists() and not force:
            print(f"Results already exist for {milestone_key}. Use --force to rerun.")
            return None

        print(f"\nAnalyzing: {milestone['name']}")
        print(f"Description: {milestone['description']}")
        print("="*80)

        results = {
            'milestone': milestone_key,
            'name': milestone['name'],
            'description': milestone['description'],
            'filtering_mode': filtering_mode
        }

        # Analyze pre-period
        pre_start, pre_end = milestone['pre_window']
        print(f"\nLoading PRE period ({pre_start}-{pre_end})...")
        pre_data = self.load_period_data(pre_start, pre_end, milestone['sample_size'])
        if pre_data:
            results['pre_period'] = self.analyze_period(pre_data, "PRE", filtering_mode)
            print(f"  - {results['pre_period']['male_speeches']} male speeches")
            print(f"  - {results['pre_period']['female_speeches']} female speeches")

        # Analyze during period (if exists)
        if 'during_window' in milestone:
            during_start, during_end = milestone['during_window']
            print(f"\nLoading DURING period ({during_start}-{during_end})...")
            during_data = self.load_period_data(during_start, during_end, milestone['sample_size'])
            if during_data:
                results['during_period'] = self.analyze_period(during_data, "DURING", filtering_mode)
                print(f"  - {results['during_period']['male_speeches']} male speeches")
                print(f"  - {results['during_period']['female_speeches']} female speeches")

        # Analyze post-period
        post_start, post_end = milestone['post_window']
        print(f"\nLoading POST period ({post_start}-{post_end})...")
        post_data = self.load_period_data(post_start, post_end, milestone['sample_size'])
        if post_data:
            results['post_period'] = self.analyze_period(post_data, "POST", filtering_mode)
            print(f"  - {results['post_period']['male_speeches']} male speeches")
            print(f"  - {results['post_period']['female_speeches']} female speeches")

        # Create visualizations
        self.create_milestone_visualization(results, milestone_key)

        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

        return results

    def create_milestone_visualization(self, results, milestone_key):
        """Create visualization for milestone analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        periods = []
        male_counts = []
        female_counts = []

        # Collect data across periods
        for period_key in ['pre_period', 'during_period', 'post_period']:
            if period_key in results and results[period_key]:
                period_data = results[period_key]
                periods.append(period_data['period'])
                male_counts.append(period_data['male_speeches'])
                female_counts.append(period_data['female_speeches'])

        # Plot 1: Speech counts over periods
        ax = axes[0, 0]
        if periods:
            x = np.arange(len(periods))
            width = 0.35
            ax.bar(x - width/2, male_counts, width, label='Male MPs', color='blue')
            ax.bar(x + width/2, female_counts, width, label='Female MPs', color='red')
            ax.set_xlabel('Period')
            ax.set_ylabel('Number of Speeches')
            ax.set_title('Speech Distribution Across Periods')
            ax.set_xticks(x)
            ax.set_xticklabels(periods)
            ax.legend()

        # Plot 2: Top words comparison
        ax = axes[0, 1]
        if 'post_period' in results and results['post_period']:
            post_words = results['post_period']['top_words'][:15]
            words = [w for w, _ in post_words]
            counts = [c for _, c in post_words]
            ax.barh(words, counts, color='green')
            ax.set_xlabel('Frequency')
            ax.set_title('Top Words in Post Period')

        # Plot 3: Male vs Female word usage
        ax = axes[1, 0]
        if 'post_period' in results and results['post_period']:
            male_words = dict(results['post_period'].get('top_male_words', [])[:10])
            female_words = dict(results['post_period'].get('top_female_words', [])[:10])

            # Find common words
            common_words = set(male_words.keys()) & set(female_words.keys())
            if common_words:
                words = list(common_words)[:10]
                male_freqs = [male_words[w] for w in words]
                female_freqs = [female_words[w] for w in words]

                x = np.arange(len(words))
                width = 0.35
                ax.bar(x - width/2, male_freqs, width, label='Male MPs', color='blue')
                ax.bar(x + width/2, female_freqs, width, label='Female MPs', color='red')
                ax.set_xlabel('Words')
                ax.set_ylabel('Frequency')
                ax.set_title('Common Words: Male vs Female Usage')
                ax.set_xticks(x)
                ax.set_xticklabels(words, rotation=45, ha='right')
                ax.legend()

        # Plot 4: Word cloud or topics
        ax = axes[1, 1]
        ax.axis('off')
        if 'post_period' in results and results['post_period']:
            # Display topics as text
            text = f"POST PERIOD TOPICS\n\n"

            if 'male_topics' in results['post_period']:
                text += "Male MP Topics:\n"
                for i, topic in enumerate(results['post_period']['male_topics']):
                    text += f"  Topic {i+1}: {', '.join(topic[:5])}\n"
                text += "\n"

            if 'female_topics' in results['post_period']:
                text += "Female MP Topics:\n"
                for i, topic in enumerate(results['post_period']['female_topics']):
                    text += f"  Topic {i+1}: {', '.join(topic[:5])}\n"

            ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace')

        plt.suptitle(f"{results['name']}\n{results['description']}", fontsize=14)
        plt.tight_layout()

        output_path = self.output_dir / f"{milestone_key}_{results['filtering_mode']}_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to {output_path}")

    def run_all_milestones(self, filtering_mode="aggressive", force=False):
        """Run analysis for all milestones"""
        results = {}
        for milestone_key in self.milestones:
            result = self.analyze_milestone(milestone_key, filtering_mode, force)
            if result:
                results[milestone_key] = result

        # Create master summary
        if results:
            summary_file = self.output_dir / f"all_milestones_{filtering_mode}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nMaster summary saved to {summary_file}")

        return results

def main():
    parser = argparse.ArgumentParser(description='Analyze gender-matched Hansard data around historical milestones')
    parser.add_argument('--milestone', type=str, help='Specific milestone to analyze')
    parser.add_argument('--filtering', type=str, default='aggressive',
                       choices=['none', 'basic', 'parliamentary', 'moderate', 'aggressive'],
                       help='Filtering level')
    parser.add_argument('--all', action='store_true', help='Analyze all milestones')
    parser.add_argument('--force', action='store_true', help='Force rerun even if results exist')

    args = parser.parse_args()

    analyzer = GenderMilestoneAnalyzer()

    if args.all:
        results = analyzer.run_all_milestones(args.filtering, args.force)
    elif args.milestone:
        if args.milestone not in analyzer.milestones:
            print(f"Error: Unknown milestone '{args.milestone}'")
            print(f"Available milestones: {', '.join(analyzer.milestones.keys())}")
            return
        results = analyzer.analyze_milestone(args.milestone, args.filtering, args.force)
    else:
        print("Please specify --milestone NAME or --all")
        return

    if results:
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print(f"Check the '{analyzer.output_dir}' directory for outputs")
        print("="*80)

if __name__ == "__main__":
    main()