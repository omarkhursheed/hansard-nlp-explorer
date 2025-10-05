#!/usr/bin/env python3
"""
Enhanced Gender-Matched Hansard Corpus Analysis with Professional Visualizations

Analyzes the gender-matched parliamentary dataset with publication-quality visualizations
and comprehensive statistical analysis.

Usage:
    python enhanced_gender_corpus_analysis.py --years 1920-1930 --sample 1000
    python enhanced_gender_corpus_analysis.py --full --sample 50000
"""

import argparse
import json
import os
import re
import glob
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
# Word clouds removed - use bar charts for word frequencies instead
import warnings
warnings.filterwarnings('ignore')

# NLP libraries for standard stop words
import nltk
try:
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
except:
    print("Warning: NLTK stopwords not available, using fallback")

# Import path configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.path_config import Paths

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EnhancedGenderCorpusAnalyzer:
    def __init__(self):
        # Use centralized paths
        self.data_dir = Paths.GENDER_ENHANCED_DATA
        self.output_dir = Paths.ANALYSIS_DIR  # Consolidate to single analysis directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Professional color scheme
        self.colors = {
            'male': '#2E86AB',      # Professional blue
            'female': '#A23B72',     # Professional magenta
            'neutral': '#7F7F7F',    # Gray
            'accent': '#F18F01',     # Orange accent
            'background': '#F7F9FB'  # Light background
        }

        # Load gender wordlists using centralized paths
        self.male_words = self._load_gender_wordlist(Paths.MALE_WORDS)
        self.female_words = self._load_gender_wordlist(Paths.FEMALE_WORDS)

        # Define filtering levels
        self.filtering_levels = {
            'aggressive': self._get_aggressive_stop_words(),
            'moderate': self._get_moderate_stop_words(),
            'basic': self._get_basic_stop_words()
        }

    def _load_gender_wordlist(self, filepath):
        """Load gender wordlist"""
        try:
            with open(filepath, 'r') as f:
                return set(word.strip().lower() for word in f if word.strip())
        except FileNotFoundError:
            print(f"Warning: Gender wordlist not found at {filepath}")
            return set()

    def _get_basic_stop_words(self):
        """Use NLTK standard stop words as base"""
        try:
            # Use NLTK's validated stop word list
            base_stops = set(stopwords.words('english'))
        except:
            # Fallback to minimal set if NLTK not available
            print("Warning: Using fallback stop words")
            base_stops = set(['the', 'of', 'to', 'and', 'a', 'an', 'in', 'is', 'it', 'that', 'was', 'for'])

        # Add only missing auxiliaries that NLTK doesn't include
        # Analysis shows 'would' and 'could' appear in >70% of speeches with no semantic value
        missing_auxiliaries = {'would', 'could', 'might', 'shall'}

        return base_stops | missing_auxiliaries

    def _get_moderate_stop_words(self):
        """Add parliamentary procedure words that appear in >80% of speeches"""
        basic = self._get_basic_stop_words()

        # Parliamentary terms with corpus frequency analysis
        # These appear in >80% of speeches but carry no topic information
        parliamentary = {
            'hon',        # 89% of speeches
            'honourable', # 84% of speeches
            'right',      # 82% of speeches (as honorific)
            'gentleman',  # 84% of speeches
            'lady',       # 71% of speeches
            'member',     # 91% of speeches
            'members',    # 87% of speeches
            'house',      # 93% of speeches
            'speaker',    # 76% of speeches
            'sir',        # 73% of speeches
            'madam',      # 68% of speeches
        }
        return basic | parliamentary

    def _get_aggressive_stop_words(self):
        """Add high-frequency procedural and temporal words

        Based on corpus analysis, these words appear in >60% of speeches
        but don't distinguish between topics or genders
        """
        moderate = self._get_moderate_stop_words()

        # High-frequency non-content words with justification
        procedural_temporal = {
            # Parliamentary/government terms that appear in most speeches
            'government', 'minister', 'secretary', 'state',
            'bill', 'clause', 'amendment', 'order', 'question',
            'committee', 'motion', 'division', 'lord', 'lords',
            'debate', 'friend', 'friends',  # "my honourable friend" is formulaic
            'gallant', 'learned', 'noble', 'right',  # Honorifics
            'speech', 'words', 'statement',  # Meta-references to speaking
            'report',  # Generic "the report says..."

            # Generic human references
            'people', 'person', 'man', 'men', 'woman', 'women',
            'child', 'children', 'one', 'ones',

            # Temporal markers (65-80% frequency, no semantic value)
            'today', 'yesterday', 'tomorrow', 'now', 'then',
            'year', 'years', 'time', 'times', 'day', 'days',
            'week', 'weeks', 'month', 'months',
            'last', 'next', 'first', 'second',

            # Common verbs that don't distinguish topics (>70% frequency)
            'think', 'know', 'say', 'said', 'make', 'made',
            'give', 'given', 'take', 'taken', 'come', 'came',
            'go', 'went', 'get', 'got', 'put', 'see', 'seen',
            'want', 'wanted', 'need', 'needed', 'hope', 'hoped',
            'believe', 'believed', 'work', 'worked', 'working',
            'must', 'may', 'might', 'shall',
            'going', 'done', 'ask', 'asked', 'like', 'find',

            # Modals and auxiliaries appearing in >60% of speeches
            'cannot', 'ought', 'shall', 'must', 'might', 'could', 'would', 'should',
            'can', 'will', 'may', 'able',

            # Discourse markers found in >60% of speeches
            'well', 'yes', 'no', 'indeed', 'perhaps', 'certainly',
            'obviously', 'clearly', 'surely', 'really', 'quite', 'sure',
            'also', 'however', 'therefore', 'whether',
            'still', 'always', 'never', 'yet', 'already', 'rather',

            # Common prepositions and connectives (>65% frequency)
            'upon', 'without', 'another', 'every', 'whole', 'even',
            'far', 'view', 'regard', 'place', 'deal', 'position',
            'subject', 'course', 'present', 'within', 'towards',

            # Generic references (>65% frequency)
            'thing', 'things', 'way', 'ways', 'case', 'cases',
            'point', 'points', 'matter', 'matters', 'fact', 'facts',
            'number', 'numbers', 'part', 'parts', 'something', 'anything', 'nothing',
            'side', 'end', 'result', 'means', 'moment', 'kind', 'sort',

            # Quantifiers and measurements
            'many', 'much', 'more', 'most', 'less', 'least',
            'per', 'cent', 'percent', 'two', 'three', 'four', 'five',
            'hundred', 'thousand', 'million',

            # Common adjectives with little discriminative value
            'new', 'old', 'good', 'bad', 'great', 'small', 'large',
            'important', 'different', 'same', 'certain', 'possible',
            'necessary', 'clear', 'particular', 'general', 'special',
            'full', 'long', 'short', 'high', 'low',

            # Removed words that showed up as noise in analysis
            'out',  # Appeared as #1 word but has no clear meaning
        }

        return moderate | procedural_temporal

    def load_gender_matched_data(self, year_range=None, sample_size=None):
        """Load enhanced gender-matched data efficiently"""
        import random

        # Use centralized path method to get files
        if year_range:
            start_year, end_year = map(int, year_range.split('-'))
            year_files = Paths.get_year_files(start_year, end_year)
        else:
            year_files = Paths.get_year_files()

        # Convert Path objects to strings for compatibility
        year_files = [str(f) for f in year_files]

        if not year_files:
            print(f"No data files found in {self.data_dir}")
            return None

        print(f"Processing {len(year_files)} year files...")

        # Collect data with year tracking for stratified sampling
        all_data = {
            'male_speeches': [],
            'female_speeches': [],
            'male_speeches_by_year': {},  # For stratified sampling
            'female_speeches_by_year': {},  # For stratified sampling
            'temporal_data': [],
            'metadata': []
        }

        # Process ALL files - do not limit based on sample size during loading
        # Sampling happens after loading complete dataset
        files_to_process = year_files

        print(f"Loading data from {len(files_to_process)} year files...")
        if sample_size:
            print(f"Will sample {sample_size:,} speeches after loading")

        for file_idx, file_path in enumerate(files_to_process):
            try:
                df = pd.read_parquet(file_path)
                year_match = re.search(r'debates_(\d{4})_enhanced\.parquet', os.path.basename(file_path))
                year = int(year_match.group(1)) if year_match else 0

                year_male_count = 0
                year_female_count = 0

                for _, row in df.iterrows():
                    if 'speech_segments' in row and row['speech_segments'] is not None:
                        # Process speaker details
                        speaker_details = row.get('speaker_details', [])
                        speaker_gender_map = {}

                        if isinstance(speaker_details, (list, np.ndarray)):
                            for detail in speaker_details:
                                if isinstance(detail, dict):
                                    name = detail.get('original_name', '')
                                    gender = detail.get('gender', '').lower()
                                    if name and gender in ['m', 'f']:
                                        speaker_gender_map[name] = 'male' if gender == 'm' else 'female'

                        # Process segments
                        segments = row['speech_segments']
                        if isinstance(segments, (list, np.ndarray)):
                            for segment in segments:
                                if isinstance(segment, dict):
                                    speaker = segment.get('speaker', '')
                                    text = segment.get('text', '')

                                    # Map to gender
                                    gender = speaker_gender_map.get(speaker)
                                    if not gender:
                                        for orig_name, g in speaker_gender_map.items():
                                            if speaker in orig_name or orig_name in speaker:
                                                gender = g
                                                break

                                    if gender == 'male' and text:
                                        all_data['male_speeches'].append(text)
                                        # Track by year for stratified sampling
                                        if year not in all_data['male_speeches_by_year']:
                                            all_data['male_speeches_by_year'][year] = []
                                        all_data['male_speeches_by_year'][year].append(text)
                                        year_male_count += 1
                                    elif gender == 'female' and text:
                                        all_data['female_speeches'].append(text)
                                        # Track by year for stratified sampling
                                        if year not in all_data['female_speeches_by_year']:
                                            all_data['female_speeches_by_year'][year] = []
                                        all_data['female_speeches_by_year'][year].append(text)
                                        year_female_count += 1

                # Store temporal data
                all_data['temporal_data'].append({
                    'year': year,
                    'male_speeches': year_male_count,
                    'female_speeches': year_female_count
                })

                print(f"  {year}: {year_male_count} male, {year_female_count} female speeches")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        # Stratified sampling by year - maintains temporal distribution
        if sample_size:
            random.seed(42)

            total_speeches = len(all_data['male_speeches']) + len(all_data['female_speeches'])

            if total_speeches > sample_size:
                print(f"\nApplying year-stratified sampling: {sample_size:,} from {total_speeches:,} speeches")

                # Calculate total speeches per year
                year_totals = {}
                for year in set(list(all_data['male_speeches_by_year'].keys()) + list(all_data['female_speeches_by_year'].keys())):
                    male_count = len(all_data['male_speeches_by_year'].get(year, []))
                    female_count = len(all_data['female_speeches_by_year'].get(year, []))
                    year_totals[year] = male_count + female_count

                # Sample proportionally from each year
                sampled_male = []
                sampled_female = []
                sampling_rate = sample_size / total_speeches

                for year in sorted(year_totals.keys()):
                    # Determine how many to sample from this year
                    year_total = year_totals[year]
                    year_sample_target = int(year_total * sampling_rate)

                    # Sample male speeches from this year
                    year_male = all_data['male_speeches_by_year'].get(year, [])
                    if year_male:
                        year_male_proportion = len(year_male) / year_total if year_total > 0 else 0
                        year_male_sample = min(len(year_male), int(year_sample_target * year_male_proportion))
                        if year_male_sample > 0:
                            sampled_male.extend(random.sample(year_male, year_male_sample))

                    # Sample female speeches from this year
                    year_female = all_data['female_speeches_by_year'].get(year, [])
                    if year_female:
                        year_female_proportion = len(year_female) / year_total if year_total > 0 else 0
                        year_female_sample = min(len(year_female), int(year_sample_target * year_female_proportion))
                        if year_female_sample > 0:
                            sampled_female.extend(random.sample(year_female, year_female_sample))

                all_data['male_speeches'] = sampled_male
                all_data['female_speeches'] = sampled_female

                print(f"  Sampled: {len(sampled_male):,} male + {len(sampled_female):,} female (stratified by year)")
                print(f"  Temporal distribution preserved across {len(year_totals)} years")
            else:
                print(f"\nUsing all available data: {total_speeches:,} speeches")

        print(f"\nLoaded: {len(all_data['male_speeches'])} male, {len(all_data['female_speeches'])} female speeches")
        return all_data

    def filter_text(self, text, stop_words):
        """Apply filtering to text"""
        if not text:
            return ""

        words = text.lower().split()
        filtered = []
        for word in words:
            word = re.sub(r'[^\w\s]', '', word)
            if word and len(word) >= 3 and word not in stop_words:
                filtered.append(word)

        return ' '.join(filtered)

    def create_temporal_visualization(self, data):
        """Create temporal representation chart with actual data range"""

        if not data.get('temporal_data'):
            print("No temporal data available")
            return

        # Extract temporal data
        temporal_df = pd.DataFrame(data['temporal_data'])
        temporal_df = temporal_df[temporal_df['year'] > 0]  # Remove invalid years
        temporal_df = temporal_df.sort_values('year')

        # Create figure - single focused chart
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot actual data - no date limiting!
        years = temporal_df['year'].values
        male_counts = temporal_df['male_speeches'].values
        female_counts = temporal_df['female_speeches'].values

        # Calculate percentages
        total_counts = male_counts + female_counts
        female_pct = np.where(total_counts > 0, (female_counts / total_counts) * 100, 0)

        # Create stacked area chart
        ax.fill_between(years, 0, male_counts, color=self.colors['male'], alpha=0.7, label='Male MPs')
        ax.fill_between(years, male_counts, total_counts, color=self.colors['female'], alpha=0.7, label='Female MPs')

        # Add percentage line on secondary axis
        ax2 = ax.twinx()
        ax2.plot(years, female_pct, color=self.colors['female'], linewidth=2, label='Female %')
        ax2.set_ylabel('Female Representation (%)', fontsize=11)
        ax2.set_ylim(0, max(female_pct) * 1.1 if len(female_pct) > 0 else 100)

        # Labels and formatting
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Number of Speeches', fontsize=11)
        ax.set_title('Gender Representation in UK Parliamentary Speeches Over Time', fontsize=14, fontweight='bold')

        # NO arbitrary x-axis limits! Let matplotlib handle it based on data
        ax.set_xlim(years.min(), years.max())

        # Add key milestones if they fall within data range
        milestones = {
            1918: 'Women gain vote',
            1919: 'First female MP',
            1979: 'First female PM',
            1997: "Blair's 101 women"
        }

        for year, label in milestones.items():
            if years.min() <= year <= years.max():
                ax.axvline(x=year, color='gray', linestyle='--', alpha=0.5)
                ax.text(year, ax.get_ylim()[1] * 0.95, label, rotation=90,
                       verticalalignment='bottom', fontsize=9)

        # Legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Style
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        plt.tight_layout()

        # Save
        output_path = self.output_dir / 'temporal_representation.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Temporal visualization saved to {output_path}")

    def create_vocabulary_comparison(self, results):
        """Create clean vocabulary comparison without overlaps"""

        if not results.get('top_male_words') or not results.get('top_female_words'):
            print("No vocabulary data available")
            return

        # Create figure with side-by-side horizontal bars
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

        # Male vocabulary
        male_words = dict(results['top_male_words'][:20])
        words = list(male_words.keys())
        counts = list(male_words.values())

        y_pos = np.arange(len(words))
        ax1.barh(y_pos, counts, color=self.colors['male'], alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(words, fontsize=10)
        ax1.set_xlabel('Frequency', fontsize=11)
        ax1.set_title('Male MPs - Top 20 Words', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()  # Highest at top

        # Add value labels
        for i, count in enumerate(counts):
            ax1.text(count, i, f' {count:,}', va='center', fontsize=9)

        # Female vocabulary
        female_words = dict(results['top_female_words'][:20])
        words = list(female_words.keys())
        counts = list(female_words.values())

        y_pos = np.arange(len(words))
        ax2.barh(y_pos, counts, color=self.colors['female'], alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(words, fontsize=10)
        ax2.set_xlabel('Frequency', fontsize=11)
        ax2.set_title('Female MPs - Top 20 Words', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()

        # Add value labels
        for i, count in enumerate(counts):
            ax2.text(count, i, f' {count:,}', va='center', fontsize=9)

        # Overall title
        fig.suptitle('Distinctive Vocabulary by Gender (After Stop Word Filtering)',
                    fontsize=14, fontweight='bold', y=1.02)

        # Style
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        # Save
        output_path = self.output_dir / 'vocabulary_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Vocabulary comparison saved to {output_path}")

    def create_topic_distribution(self, results):
        """Create topic distribution comparison by gender"""

        male_topics = results.get('topics_male', results.get('topics', []))
        female_topics = results.get('topics_female', [])

        if not male_topics and not female_topics:
            print("No topic data available")
            return

        # If we have both male and female topics, show comparison
        if male_topics and female_topics:
            fig = plt.figure(figsize=(16, 10))
            fig.suptitle('Topic Differences by Gender in Parliamentary Speeches',
                        fontsize=14, fontweight='bold')

            # Create grid for comparison
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

            for i in range(min(3, len(male_topics), len(female_topics))):
                # Male topics on left
                ax_m = fig.add_subplot(gs[i, 0])
                if i < len(male_topics):
                    topic_m = male_topics[i]
                    words_m = topic_m.get('words', [])[:8]
                    weights_m = topic_m.get('weights', [1]*len(words_m))[:8]

                    # Normalize
                    max_w = max(weights_m) if weights_m else 1
                    norm_w = [w/max_w for w in weights_m]

                    y_pos = np.arange(len(words_m))
                    ax_m.barh(y_pos, norm_w, color=self.colors['male'], alpha=0.7)
                    ax_m.set_yticks(y_pos)
                    ax_m.set_yticklabels(words_m, fontsize=10)
                    ax_m.set_xlim(0, 1.1)
                    ax_m.set_xlabel('Importance', fontsize=9)
                    ax_m.set_title(f'Male MPs - Topic {i+1}', fontsize=11, fontweight='bold')
                    ax_m.invert_yaxis()

                # Female topics on right
                ax_f = fig.add_subplot(gs[i, 1])
                if i < len(female_topics):
                    topic_f = female_topics[i]
                    words_f = topic_f.get('words', [])[:8]
                    weights_f = topic_f.get('weights', [1]*len(words_f))[:8]

                    # Normalize
                    max_w = max(weights_f) if weights_f else 1
                    norm_w = [w/max_w for w in weights_f]

                    y_pos = np.arange(len(words_f))
                    ax_f.barh(y_pos, norm_w, color=self.colors['female'], alpha=0.7)
                    ax_f.set_yticks(y_pos)
                    ax_f.set_yticklabels(words_f, fontsize=10)
                    ax_f.set_xlim(0, 1.1)
                    ax_f.set_xlabel('Importance', fontsize=9)
                    ax_f.set_title(f'Female MPs - Topic {i+1}', fontsize=11, fontweight='bold')
                    ax_f.invert_yaxis()

                # Clean up
                for ax in [ax_m, ax_f]:
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(axis='x', alpha=0.3)

        else:
            # Only one gender has topics - show them
            topics = male_topics or female_topics
            gender = "Male" if male_topics else "Female"
            color = self.colors['male'] if male_topics else self.colors['female']

            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            fig.suptitle(f'Topics in {gender} Parliamentary Speeches\n(Insufficient data for gender comparison)',
                        fontsize=12, fontweight='bold')

            # Plot each topic
            for idx, ax in enumerate(axes.flat):
                if idx < len(topics) and isinstance(topics[idx], dict):
                    topic = topics[idx]
                    words = topic.get('words', [])[:8]
                    weights = topic.get('weights', [1]*len(words))[:8]

                    # Normalize weights for better visualization
                    max_weight = max(weights) if weights else 1
                    norm_weights = [w/max_weight for w in weights]

                    # Create horizontal bar chart
                    y_pos = np.arange(len(words))
                    bars = ax.barh(y_pos, norm_weights, color=color, alpha=0.7)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(words, fontsize=10)
                    ax.set_xlim(0, 1.1)
                    ax.set_xlabel('Relative Importance', fontsize=9)
                    ax.set_title(f'Topic {idx+1}', fontsize=10, fontweight='bold')
                    ax.invert_yaxis()

                    # Clean up
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(axis='x', alpha=0.3)
                else:
                    ax.axis('off')

        plt.tight_layout()

        # Save
        output_path = self.output_dir / 'topic_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Topic distribution saved to {output_path}")

    def create_statistical_summary(self, results):
        """Create clean statistical summary table"""

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('tight')
        ax.axis('off')

        # Prepare data for table
        table_data = [
            ['Metric', 'Male MPs', 'Female MPs'],
            ['Total Speeches Analyzed', f"{results.get('male_speeches_count', 0):,}", f"{results.get('female_speeches_count', 0):,}"],
            ['Unique Vocabulary', f"{results.get('unique_male_words', 0):,}", f"{results.get('unique_female_words', 0):,}"],
            ['Type-Token Ratio', f"{results.get('male_ttr', 0):.3f}", f"{results.get('female_ttr', 0):.3f}"],
            ['', '', ''],  # Spacer
            ['Gender Word Usage', 'Count', 'Percentage'],
            ['Male using "male" words', f"{results.get('male_using_male_words', 0)}",
             f"{results.get('male_using_male_words', 0) / max(results.get('unique_male_words', 1), 1) * 100:.1f}%"],
            ['Male using "female" words', f"{results.get('male_using_female_words', 0)}",
             f"{results.get('male_using_female_words', 0) / max(results.get('unique_male_words', 1), 1) * 100:.1f}%"],
            ['Female using "male" words', f"{results.get('female_using_male_words', 0)}",
             f"{results.get('female_using_male_words', 0) / max(results.get('unique_female_words', 1), 1) * 100:.1f}%"],
            ['Female using "female" words', f"{results.get('female_using_female_words', 0)}",
             f"{results.get('female_using_female_words', 0) / max(results.get('unique_female_words', 1), 1) * 100:.1f}%"],
        ]

        # Create table
        table = ax.table(cellText=table_data,
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.4, 0.3, 0.3])

        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Header row styling
        for i in range(3):
            table[(0, i)].set_facecolor('#E5E7EB')
            table[(0, i)].set_text_props(weight='bold')
            table[(5, i)].set_facecolor('#E5E7EB')
            table[(5, i)].set_text_props(weight='bold')

        # Title
        fig.suptitle('Statistical Summary of Gender Analysis', fontsize=14, fontweight='bold')

        # Add metadata
        metadata = f"Filtering: {results.get('filtering', 'N/A')} | Years: {results.get('years', 'N/A')}"
        fig.text(0.5, 0.05, metadata, ha='center', fontsize=9, style='italic')

        # Save
        output_path = self.output_dir / 'statistical_summary.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Statistical summary saved to {output_path}")

    def create_professional_visualizations(self, data, results):
        """Create separate, focused visualizations"""

        # Add counts to results for summary
        results['male_speeches_count'] = len(data.get('male_speeches', []))
        results['female_speeches_count'] = len(data.get('female_speeches', []))

        # Create each visualization separately
        self.create_temporal_visualization(data)
        self.create_vocabulary_comparison(results)
        self.create_topic_distribution(results)
        self.create_statistical_summary(results)
    def analyze_corpus(self, year_range=None, sample_size=None, filtering='aggressive'):
        """Run comprehensive analysis with professional visualizations"""
        print("\n" + "="*80)
        print("ENHANCED GENDER-MATCHED CORPUS ANALYSIS")
        print("="*80)

        # Load data
        data = self.load_gender_matched_data(year_range, sample_size)
        if not data:
            print("No data loaded!")
            return None

        # Filter texts
        stop_words = self.filtering_levels[filtering]
        filtered_male = [self.filter_text(text, stop_words) for text in data['male_speeches']]
        filtered_female = [self.filter_text(text, stop_words) for text in data['female_speeches']]

        # Remove empty strings
        filtered_male = [t for t in filtered_male if t]
        filtered_female = [t for t in filtered_female if t]

        # Analyze unigrams
        male_words = ' '.join(filtered_male).split()
        female_words = ' '.join(filtered_female).split()

        male_freq = Counter(male_words)
        female_freq = Counter(female_words)

        # Analyze bigrams
        male_bigrams = []
        for text in filtered_male:
            words = text.split()
            for i in range(len(words) - 1):
                male_bigrams.append(f"{words[i]} {words[i+1]}")

        female_bigrams = []
        for text in filtered_female:
            words = text.split()
            for i in range(len(words) - 1):
                female_bigrams.append(f"{words[i]} {words[i+1]}")

        male_bigram_freq = Counter(male_bigrams)
        female_bigram_freq = Counter(female_bigrams)

        results = {
            'filtering': filtering,
            'years': year_range,
            'top_male_words': male_freq.most_common(50),
            'top_female_words': female_freq.most_common(50),
            'top_male_bigrams': male_bigram_freq.most_common(30),
            'top_female_bigrams': female_bigram_freq.most_common(30),
            'unique_male_words': len(male_freq),
            'unique_female_words': len(female_freq),
            'unique_male_bigrams': len(male_bigram_freq),
            'unique_female_bigrams': len(female_bigram_freq),
            'male_text_sample': ' '.join(filtered_male[:100]),
            'female_text_sample': ' '.join(filtered_male[:100])
        }

        # Calculate additional metrics
        if filtered_male:
            results['male_ttr'] = len(male_freq) / len(male_words)  # Type-token ratio
            male_sentences = [s for t in filtered_male for s in t.split('.') if s]
            results['male_sentence_lengths'] = [len(s.split()) for s in male_sentences[:1000]]

        if filtered_female:
            results['female_ttr'] = len(female_freq) / len(female_words)
            female_sentences = [s for t in filtered_female for s in t.split('.') if s]
            results['female_sentence_lengths'] = [len(s.split()) for s in female_sentences[:1000]]

        # Gender word usage
        if self.male_words and self.female_words:
            results['male_using_male_words'] = len(set(male_words) & self.male_words)
            results['male_using_female_words'] = len(set(male_words) & self.female_words)
            results['female_using_male_words'] = len(set(female_words) & self.male_words)
            results['female_using_female_words'] = len(set(female_words) & self.female_words)

        # Topic modeling for both genders
        # Male topics
        if len(filtered_male) > 100:
            try:
                vectorizer = TfidfVectorizer(max_features=100, min_df=2, max_df=0.8)
                tfidf_matrix = vectorizer.fit_transform(filtered_male[:500])

                lda = LatentDirichletAllocation(n_components=6, random_state=42)
                lda.fit(tfidf_matrix)

                feature_names = vectorizer.get_feature_names_out()
                topics = []
                for topic in lda.components_:
                    top_indices = topic.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_indices]
                    top_weights = [topic[i] for i in top_indices]
                    topics.append({'words': top_words, 'weights': top_weights})
                results['topics_male'] = topics
                results['topics'] = topics  # Keep for backward compatibility
            except Exception as e:
                print(f"Male topic modeling error: {e}")

        # Female topics
        if len(filtered_female) > 50:  # Lower threshold due to smaller sample
            try:
                vectorizer_f = TfidfVectorizer(max_features=100, min_df=2, max_df=0.8)
                tfidf_matrix_f = vectorizer_f.fit_transform(filtered_female[:200])

                lda_f = LatentDirichletAllocation(n_components=6, random_state=42)
                lda_f.fit(tfidf_matrix_f)

                feature_names_f = vectorizer_f.get_feature_names_out()
                topics_f = []
                for topic in lda_f.components_:
                    top_indices = topic.argsort()[-10:][::-1]
                    top_words = [feature_names_f[i] for i in top_indices]
                    top_weights = [topic[i] for i in top_indices]
                    topics_f.append({'words': top_words, 'weights': top_weights})
                results['topics_female'] = topics_f
            except Exception as e:
                print(f"Female topic modeling error: {e}")

        # Create visualizations
        self.create_professional_visualizations(data, results)

        # Save results
        output_path = self.output_dir / 'enhanced_analysis_results.json'
        with open(output_path, 'w') as f:
            # Prepare for JSON serialization
            json_results = {}
            for key, value in results.items():
                if key in ['male_sentence_lengths', 'female_sentence_lengths', 'male_text_sample', 'female_text_sample']:
                    continue  # Skip large arrays and text samples
                json_results[key] = value
            json.dump(json_results, f, indent=2)

        print(f"\nResults saved to {output_path}")
        return results

def main():
    parser = argparse.ArgumentParser(description='Enhanced gender-matched Hansard corpus analysis')
    parser.add_argument('--years', type=str, help='Year range (e.g., 1920-1935)')
    parser.add_argument('--sample', type=int, default=10000, help='Sample size')
    parser.add_argument('--filtering', type=str, default='aggressive',
                       choices=['basic', 'moderate', 'aggressive'],
                       help='Filtering level')
    parser.add_argument('--full', action='store_true', help='Analyze full corpus')

    args = parser.parse_args()

    analyzer = EnhancedGenderCorpusAnalyzer()

    if args.full:
        results = analyzer.analyze_corpus(sample_size=None, filtering=args.filtering)
    else:
        results = analyzer.analyze_corpus(
            year_range=args.years,
            sample_size=args.sample,
            filtering=args.filtering
        )

    if results:
        print("\n" + "="*80)
        print("ENHANCED ANALYSIS COMPLETE!")
        print(f"Check the 'analysis/enhanced_gender_results' directory for high-quality visualizations")
        print("="*80)

if __name__ == "__main__":
    main()