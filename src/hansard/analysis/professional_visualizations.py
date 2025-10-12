#!/usr/bin/env python3
"""
Professional visualization module for gender analysis of parliamentary debates.
Follows established style guide for publication-quality figures.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy import stats
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Import path configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.path_config import Paths

# Professional color palette
COLORS = {
    'male': '#3B82C4',      # Professional blue
    'female': '#EC4899',     # Professional pink/magenta
    'background': '#FFFFFF',  # White
    'grid': '#E5E7EB',       # Light gray for gridlines
    'text': '#1F2937',       # Dark gray for text
    'muted': '#9CA3AF',      # Medium gray for de-emphasized elements
    'accent1': '#10B981',    # Emerald
    'accent2': '#F59E0B',    # Amber
    'accent3': '#8B5CF6',    # Violet
}

# Set global style parameters
def set_publication_style():
    """Set matplotlib parameters for nice figures"""
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['figure.titleweight'] = 'bold'

    # Remove top and right spines
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # Grid settings
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.grid.axis'] = 'y'
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = COLORS['grid']

    # Set background colors
    plt.rcParams['figure.facecolor'] = COLORS['background']
    plt.rcParams['axes.facecolor'] = COLORS['background']


class GenderVisualizationSuite:
    """ visualization suite for gender analysis"""

    def __init__(self, output_dir=None):
        # Use centralized path if no output_dir specified
        self.output_dir = Path(output_dir) if output_dir else Paths.VISUALIZATIONS
        self.output_dir.mkdir(parents=True, exist_ok=True)
        set_publication_style()

    def create_temporal_participation(self, temporal_data, output_name="temporal_participation.png"):
        """
        Create area chart showing gender participation over time.

        Args:
            temporal_data: List of dicts with 'year', 'male_speeches', 'female_speeches'
        """
        if not temporal_data:
            print("No temporal data available for visualization")
            return

        # Convert to DataFrame and sort
        df = pd.DataFrame(temporal_data).sort_values('year')

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create stacked area chart
        ax.fill_between(df['year'], 0, df['male_speeches'],
                       color=COLORS['male'], alpha=0.7, label='Male MPs')
        ax.fill_between(df['year'], df['male_speeches'],
                       df['male_speeches'] + df['female_speeches'],
                       color=COLORS['female'], alpha=0.7, label='Female MPs')

        # Add key historical events
        events = [
            (1918, "Women's Suffrage (partial)", 'bottom'),
            (1928, "Equal Franchise Act", 'top'),
            (1979, "Margaret Thatcher PM", 'bottom')
        ]

        for year, label, va in events:
            if df['year'].min() <= year <= df['year'].max():
                ax.axvline(year, color=COLORS['muted'], linestyle='--', alpha=0.5)
                ax.text(year, ax.get_ylim()[1] * 0.95 if va == 'top' else ax.get_ylim()[1] * 0.05,
                       label, rotation=90, va=va, ha='right',
                       fontsize=9, color=COLORS['text'], alpha=0.7)

        # Styling
        ax.set_xlabel('Year', fontsize=12, fontweight='bold', color=COLORS['text'])
        ax.set_ylabel('Number of Speeches', fontsize=12, fontweight='bold', color=COLORS['text'])
        ax.set_title('Parliamentary Participation by Gender Over Time',
                    fontsize=14, fontweight='bold', color=COLORS['text'], pad=20)

        # Legend
        ax.legend(loc='upper left', framealpha=0.9, edgecolor='none')

        # Format x-axis
        ax.set_xlim(df['year'].min(), df['year'].max())
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='both', nbins=10))

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()

        print(f"Saved temporal participation chart to {output_path}")

    def create_distinctive_vocabulary(self, male_words, female_words, output_name="distinctive_vocabulary.png"):
        """
        Create diverging bar chart showing distinctively gendered vocabulary.

        Args:
            male_words: Counter of words from male speeches
            female_words: Counter of words from female speeches
        """
        # Calculate log-odds ratio for each word
        male_total = sum(male_words.values())
        female_total = sum(female_words.values())

        if male_total == 0 or female_total == 0:
            print("Insufficient data for distinctive vocabulary analysis")
            return

        word_scores = {}
        all_words = set(male_words.keys()) | set(female_words.keys())

        for word in all_words:
            # Add smoothing to avoid log(0)
            male_freq = (male_words.get(word, 0) + 1) / (male_total + len(all_words))
            female_freq = (female_words.get(word, 0) + 1) / (female_total + len(all_words))

            # Log odds ratio: positive = more female, negative = more male
            log_odds = np.log(female_freq / male_freq)

            # Weight by frequency to avoid rare words
            total_freq = male_words.get(word, 0) + female_words.get(word, 0)
            if total_freq > 10:  # Minimum frequency threshold
                word_scores[word] = log_odds

        # Get top distinctive words for each gender
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1])
        male_distinctive = sorted_words[:15]  # Most male-associated
        female_distinctive = sorted_words[-15:]  # Most female-associated

        # Combine and sort by absolute value for display
        all_distinctive = male_distinctive + female_distinctive
        all_distinctive.sort(key=lambda x: abs(x[1]), reverse=True)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        words = [w for w, _ in all_distinctive[:20]]  # Top 20 most distinctive
        scores = [s for _, s in all_distinctive[:20]]
        colors = [COLORS['male'] if s < 0 else COLORS['female'] for s in scores]

        # Create horizontal bars
        bars = ax.barh(range(len(words)), scores, color=colors, alpha=0.7)

        # Add center line
        ax.axvline(0, color=COLORS['text'], linestyle='-', linewidth=0.5)

        # Labels
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10)
        ax.set_xlabel('← More Male    Log Odds Ratio    More Female →',
                     fontsize=11, fontweight='bold', color=COLORS['text'])
        ax.set_title('Most Distinctive Vocabulary by Gender',
                    fontsize=14, fontweight='bold', color=COLORS['text'], pad=20)

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(width + (0.1 if width > 0 else -0.1), bar.get_y() + bar.get_height()/2,
                   f'{abs(width):.1f}', ha='left' if width > 0 else 'right',
                   va='center', fontsize=8, color=COLORS['muted'])

        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xlim(-3, 3)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()

        print(f"Saved distinctive vocabulary chart to {output_path}")

    def create_speech_length_distribution(self, male_lengths, female_lengths,
                                        output_name="speech_length_distribution.png"):
        """
        Create overlapping density plots for speech length distributions.

        Args:
            male_lengths: List of speech lengths for male MPs
            female_lengths: List of speech lengths for female MPs
        """
        if not male_lengths or not female_lengths:
            print("Insufficient data for speech length distribution")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create density plots
        male_lengths = np.array(male_lengths)
        female_lengths = np.array(female_lengths)

        # Remove outliers for better visualization
        male_lengths = male_lengths[male_lengths < np.percentile(male_lengths, 95)]
        female_lengths = female_lengths[female_lengths < np.percentile(female_lengths, 95)]

        # Plot distributions
        ax.hist(male_lengths, bins=50, density=True, alpha=0.5,
               color=COLORS['male'], label=f'Male MPs (n={len(male_lengths):,})')
        ax.hist(female_lengths, bins=50, density=True, alpha=0.5,
               color=COLORS['female'], label=f'Female MPs (n={len(female_lengths):,})')

        # Add median lines
        male_median = np.median(male_lengths)
        female_median = np.median(female_lengths)

        ax.axvline(male_median, color=COLORS['male'], linestyle='--',
                  linewidth=2, label=f'Male median: {male_median:.0f}')
        ax.axvline(female_median, color=COLORS['female'], linestyle='--',
                  linewidth=2, label=f'Female median: {female_median:.0f}')

        # Statistical test
        statistic, pvalue = stats.mannwhitneyu(male_lengths, female_lengths)
        significance = "***" if pvalue < 0.001 else "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else "ns"

        # Labels
        ax.set_xlabel('Speech Length (words)', fontsize=12, fontweight='bold', color=COLORS['text'])
        ax.set_ylabel('Density', fontsize=12, fontweight='bold', color=COLORS['text'])
        ax.set_title('Distribution of Speech Lengths by Gender',
                    fontsize=14, fontweight='bold', color=COLORS['text'], pad=20)

        # Add statistical annotation
        ax.text(0.98, 0.98, f'Mann-Whitney U test: p = {pvalue:.4f} {significance}',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=10, color=COLORS['text'],
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

        # Legend
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='none')

        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()

        print(f"Saved speech length distribution to {output_path}")

    def create_topic_prevalence(self, topics_data, output_name="topic_prevalence.png"):
        """
        Create small multiples showing topic prevalence by gender.

        Args:
            topics_data: Dict with 'male_topics', 'female_topics' containing topic distributions
        """
        if not topics_data.get('male_topics') or not topics_data.get('female_topics'):
            print("Insufficient topic data for visualization")
            return

        n_topics = min(len(topics_data['male_topics']), len(topics_data['female_topics']), 6)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for idx in range(n_topics):
            ax = axes[idx]

            male_topic = topics_data['male_topics'][idx]
            female_topic = topics_data['female_topics'][idx]

            # Get top words for this topic
            if isinstance(male_topic, dict):
                topic_words = male_topic.get('words', [])[:5]
                male_weight = male_topic.get('weight', 0)
            else:
                topic_words = male_topic[:5] if isinstance(male_topic, list) else []
                male_weight = 1

            if isinstance(female_topic, dict):
                female_weight = female_topic.get('weight', 0)
            else:
                female_weight = 1

            # Create bar chart
            categories = ['Male MPs', 'Female MPs']
            values = [male_weight, female_weight]
            colors = [COLORS['male'], COLORS['female']]

            bars = ax.bar(categories, values, color=colors, alpha=0.7)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom',
                       fontsize=9, color=COLORS['text'])

            # Title with topic words
            topic_label = f"Topic {idx+1}"
            if topic_words:
                topic_label += f"\n({', '.join(topic_words[:3])})"
            ax.set_title(topic_label, fontsize=10, fontweight='bold')

            # Styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)

        # Hide unused subplots
        for idx in range(n_topics, 6):
            axes[idx].axis('off')

        # Overall title
        fig.suptitle('Topic Prevalence by Gender', fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()

        print(f"Saved topic prevalence chart to {output_path}")

    def create_key_metrics_summary(self, metrics, output_name="key_metrics_summary.png"):
        """
        Create a clean infographic-style summary of key metrics.

        Args:
            metrics: Dict containing various metrics to display
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')

        # Title
        fig.text(0.5, 0.95, 'Gender Analysis: Key Metrics Summary',
                fontsize=18, fontweight='bold', ha='center', color=COLORS['text'])

        # Create metric boxes
        y_position = 0.85

        # Total speeches
        if 'total_male_speeches' in metrics and 'total_female_speeches' in metrics:
            total_male = metrics['total_male_speeches']
            total_female = metrics['total_female_speeches']
            total = total_male + total_female

            fig.text(0.25, y_position, 'Total Speeches Analyzed',
                    fontsize=12, fontweight='bold', ha='center', color=COLORS['text'])
            fig.text(0.25, y_position - 0.03, f'{total:,}',
                    fontsize=16, ha='center', color=COLORS['text'])

            # Gender breakdown
            male_pct = (total_male / total * 100) if total > 0 else 0
            female_pct = (total_female / total * 100) if total > 0 else 0

            fig.text(0.25, y_position - 0.08, f'Male: {total_male:,} ({male_pct:.1f}%)',
                    fontsize=11, ha='center', color=COLORS['male'])
            fig.text(0.25, y_position - 0.11, f'Female: {total_female:,} ({female_pct:.1f}%)',
                    fontsize=11, ha='center', color=COLORS['female'])

        # Vocabulary richness
        if 'male_vocabulary_size' in metrics and 'female_vocabulary_size' in metrics:
            fig.text(0.75, y_position, 'Unique Vocabulary',
                    fontsize=12, fontweight='bold', ha='center', color=COLORS['text'])
            fig.text(0.75, y_position - 0.04, f"Male: {metrics['male_vocabulary_size']:,} words",
                    fontsize=11, ha='center', color=COLORS['male'])
            fig.text(0.75, y_position - 0.07, f"Female: {metrics['female_vocabulary_size']:,} words",
                    fontsize=11, ha='center', color=COLORS['female'])

        y_position -= 0.2

        # Average speech length
        if 'male_avg_length' in metrics and 'female_avg_length' in metrics:
            fig.text(0.25, y_position, 'Average Speech Length',
                    fontsize=12, fontweight='bold', ha='center', color=COLORS['text'])
            fig.text(0.25, y_position - 0.04, f"Male: {metrics['male_avg_length']:.0f} words",
                    fontsize=11, ha='center', color=COLORS['male'])
            fig.text(0.25, y_position - 0.07, f"Female: {metrics['female_avg_length']:.0f} words",
                    fontsize=11, ha='center', color=COLORS['female'])

        # Time period
        if 'start_year' in metrics and 'end_year' in metrics:
            fig.text(0.75, y_position, 'Time Period Analyzed',
                    fontsize=12, fontweight='bold', ha='center', color=COLORS['text'])
            fig.text(0.75, y_position - 0.04, f"{metrics['start_year']} - {metrics['end_year']}",
                    fontsize=14, ha='center', color=COLORS['text'])

        y_position -= 0.2

        # Top distinctive words
        if 'top_male_distinctive' in metrics and 'top_female_distinctive' in metrics:
            fig.text(0.5, y_position, 'Most Distinctive Words',
                    fontsize=12, fontweight='bold', ha='center', color=COLORS['text'])

            # Male words
            fig.text(0.25, y_position - 0.04, 'Male MPs:',
                    fontsize=11, fontweight='bold', ha='center', color=COLORS['male'])
            male_words = ', '.join(metrics['top_male_distinctive'][:5])
            fig.text(0.25, y_position - 0.07, male_words,
                    fontsize=10, ha='center', color=COLORS['text'], style='italic')

            # Female words
            fig.text(0.75, y_position - 0.04, 'Female MPs:',
                    fontsize=11, fontweight='bold', ha='center', color=COLORS['female'])
            female_words = ', '.join(metrics['top_female_distinctive'][:5])
            fig.text(0.75, y_position - 0.07, female_words,
                    fontsize=10, ha='center', color=COLORS['text'], style='italic')

        # Add border
        rect = mpatches.FancyBboxPatch((0.05, 0.05), 0.9, 0.85,
                                       boxstyle="round,pad=0.02",
                                       edgecolor=COLORS['grid'],
                                       facecolor='none',
                                       linewidth=2)
        ax.add_patch(rect)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()

        print(f"Saved key metrics summary to {output_path}")


def demo_visualizations():
    """Demonstrate the visualization suite with sample data"""
    viz = GenderVisualizationSuite()

    # Sample temporal data
    temporal_data = [
        {'year': 1900 + i, 'male_speeches': 1000 + i*10, 'female_speeches': max(0, i-20) * 5}
        for i in range(100)
    ]
    viz.create_temporal_participation(temporal_data)

    # Sample vocabulary data
    male_words = Counter({'parliament': 100, 'government': 90, 'bill': 80, 'committee': 70})
    female_words = Counter({'children': 100, 'education': 90, 'health': 80, 'welfare': 70})
    viz.create_distinctive_vocabulary(male_words, female_words)

    print("\nDemo visualizations created successfully!")


class UnifiedVisualizationSuite(GenderVisualizationSuite):
    """
    Extended visualization suite for all analysis types.
    Adds corpus analysis and milestone comparison visualizations.
    """

    def create_unigram_comparison(self, male_words, female_words, top_n=20,
                                  output_name="unigram_comparison.png"):
        """
        Create horizontal bar chart comparing top unigrams by gender.

        Args:
            male_words: Counter or list of (word, count) tuples for male speeches
            female_words: Counter or list of (word, count) tuples for female speeches
            top_n: Number of top words to display
            output_name: Output filename
        """
        if not male_words or not female_words:
            print("Insufficient data for unigram comparison")
            return

        # Convert to lists if needed
        if isinstance(male_words, Counter):
            male_words = male_words.most_common(top_n)
        if isinstance(female_words, Counter):
            female_words = female_words.most_common(top_n)

        # Create figure with side-by-side panels
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

        # Male words
        m_words, m_counts = zip(*male_words[:top_n])
        y_pos = np.arange(len(m_words))
        ax1.barh(y_pos, m_counts, color=COLORS['male'], alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(m_words, fontsize=10)
        ax1.set_xlabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title(f'Male MPs - Top {len(m_words)} Words',
                     fontsize=12, fontweight='bold')
        ax1.invert_yaxis()

        # Add value labels
        for i, count in enumerate(m_counts):
            ax1.text(count, i, f' {count:,}', va='center', fontsize=9)

        # Female words
        f_words, f_counts = zip(*female_words[:top_n])
        y_pos = np.arange(len(f_words))
        ax2.barh(y_pos, f_counts, color=COLORS['female'], alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(f_words, fontsize=10)
        ax2.set_xlabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title(f'Female MPs - Top {len(f_words)} Words',
                     fontsize=12, fontweight='bold')
        ax2.invert_yaxis()

        # Add value labels
        for i, count in enumerate(f_counts):
            ax2.text(count, i, f' {count:,}', va='center', fontsize=9)

        # Overall title
        fig.suptitle('Distinctive Vocabulary by Gender', fontsize=14,
                    fontweight='bold', y=0.98)

        # Style
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()

        print(f"Saved unigram comparison to {output_path}")

    def create_bigram_comparison(self, male_bigrams, female_bigrams, top_n=20,
                                output_name="bigram_comparison.png"):
        """
        Create horizontal bar chart comparing top bigrams by gender.

        Args:
            male_bigrams: Counter or list of (bigram, count) tuples for male speeches
            female_bigrams: Counter or list of (bigram, count) tuples for female speeches
            top_n: Number of top bigrams to display
            output_name: Output filename
        """
        if not male_bigrams or not female_bigrams:
            print("Insufficient data for bigram comparison")
            return

        # Convert to lists if needed
        if isinstance(male_bigrams, Counter):
            male_bigrams = male_bigrams.most_common(top_n)
        if isinstance(female_bigrams, Counter):
            female_bigrams = female_bigrams.most_common(top_n)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

        # Male bigrams
        m_bigrams, m_counts = zip(*male_bigrams[:top_n])
        m_labels = [' '.join(b) if isinstance(b, tuple) else b for b in m_bigrams]
        y_pos = np.arange(len(m_labels))
        ax1.barh(y_pos, m_counts, color=COLORS['male'], alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(m_labels, fontsize=10)
        ax1.set_xlabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title(f'Male MPs - Top {len(m_labels)} Bigrams',
                     fontsize=12, fontweight='bold')
        ax1.invert_yaxis()

        # Add value labels
        for i, count in enumerate(m_counts):
            ax1.text(count, i, f' {count:,}', va='center', fontsize=9)

        # Female bigrams
        f_bigrams, f_counts = zip(*female_bigrams[:top_n])
        f_labels = [' '.join(b) if isinstance(b, tuple) else b for b in f_bigrams]
        y_pos = np.arange(len(f_labels))
        ax2.barh(y_pos, f_counts, color=COLORS['female'], alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(f_labels, fontsize=10)
        ax2.set_xlabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title(f'Female MPs - Top {len(f_labels)} Bigrams',
                     fontsize=12, fontweight='bold')
        ax2.invert_yaxis()

        # Add value labels
        for i, count in enumerate(f_counts):
            ax2.text(count, i, f' {count:,}', va='center', fontsize=9)

        # Overall title
        fig.suptitle('Distinctive Bigrams by Gender', fontsize=14,
                    fontweight='bold', y=0.98)

        # Style
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()

        print(f"Saved bigram comparison to {output_path}")

    def create_filtering_comparison(self, filtering_results,
                                   output_name="filtering_comparison.png"):
        """
        Create comparison visualization across different filtering levels.

        Args:
            filtering_results: Dict mapping level name to results dict with 'filtering_stats'
            output_name: Output filename
        """
        if not filtering_results:
            print("No filtering results to compare")
            return

        # Extract data
        levels = list(filtering_results.keys())
        reductions = [filtering_results[l]['filtering_stats']['reduction_pct']
                     for l in levels]
        filtered_words = [filtering_results[l]['filtering_stats']['filtered_words']
                         for l in levels]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Chart 1: Reduction percentages
        bars = ax1.bar(range(len(levels)), reductions, color=COLORS['accent1'], alpha=0.7)
        ax1.set_xticks(range(len(levels)))
        ax1.set_xticklabels(levels, rotation=45, ha='right')
        ax1.set_ylabel('Word Reduction (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Filtering Effectiveness', fontsize=12, fontweight='bold')

        # Add value labels
        for bar, value in zip(bars, reductions):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=9)

        # Chart 2: Words remaining
        bars = ax2.bar(range(len(levels)), filtered_words, color=COLORS['accent2'], alpha=0.7)
        ax2.set_xticks(range(len(levels)))
        ax2.set_xticklabels(levels, rotation=45, ha='right')
        ax2.set_ylabel('Words Remaining', fontsize=11, fontweight='bold')
        ax2.set_title('Vocabulary Size After Filtering', fontsize=12, fontweight='bold')

        # Add value labels
        for bar, value in zip(bars, filtered_words):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:,}', ha='center', va='bottom', fontsize=9)

        # Overall title
        fig.suptitle('Filtering Levels Comparison', fontsize=14,
                    fontweight='bold', y=1.02)

        # Style
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()

        print(f"Saved filtering comparison to {output_path}")

    def create_milestone_comparison(self, pre_data, post_data, milestone_info,
                                   output_name="milestone_comparison.png"):
        """
        Create multi-panel comparison for milestone analysis.

        Args:
            pre_data: Results dict for pre-milestone period
            post_data: Results dict for post-milestone period
            milestone_info: Dict with 'name', 'year', etc.
            output_name: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{milestone_info['name']} - Impact Analysis",
                    fontsize=14, fontweight='bold')

        # Panel 1: Top words comparison
        ax = axes[0, 0]
        if pre_data.get('top_unigrams') and post_data.get('top_unigrams'):
            pre_words = dict(pre_data['top_unigrams'][:10])
            post_words = dict(post_data['top_unigrams'][:10])

            # Get union of words
            all_words = list(set(pre_words.keys()) | set(post_words.keys()))[:10]
            pre_counts = [pre_words.get(w, 0) for w in all_words]
            post_counts = [post_words.get(w, 0) for w in all_words]

            x = np.arange(len(all_words))
            width = 0.35
            ax.barh(x - width/2, pre_counts, width, label='Pre', color=COLORS['accent1'], alpha=0.7)
            ax.barh(x + width/2, post_counts, width, label='Post', color=COLORS['accent2'], alpha=0.7)
            ax.set_yticks(x)
            ax.set_yticklabels(all_words, fontsize=9)
            ax.set_xlabel('Frequency')
            ax.set_title('Top Words Evolution')
            ax.legend()
            ax.invert_yaxis()

        # Panel 2: Gender language change
        ax = axes[0, 1]
        if (pre_data.get('gender_analysis') and post_data.get('gender_analysis')):
            pre_female = pre_data['gender_analysis']['female_ratio'] * 100
            post_female = post_data['gender_analysis']['female_ratio'] * 100

            categories = ['Pre', 'Post']
            values = [pre_female, post_female]
            bars = ax.bar(categories, values, color=[COLORS['accent1'], COLORS['accent2']], alpha=0.7)
            ax.set_ylabel('Female Language (%)')
            ax.set_title('Gender Language Evolution')

            # Add change annotation
            change = post_female - pre_female
            ax.text(0.5, max(values) * 1.1, f'Change: {change:+.2f}pp',
                   ha='center', fontweight='bold',
                   color='green' if change > 0 else 'red')

            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{value:.1f}%', ha='center', va='bottom')

        # Panel 3: Content evolution
        ax = axes[1, 0]
        if pre_data.get('top_unigrams') and post_data.get('top_unigrams'):
            pre_set = set([w for w, _ in pre_data['top_unigrams'][:20]])
            post_set = set([w for w, _ in post_data['top_unigrams'][:20]])

            new_words = post_set - pre_set
            disappeared = pre_set - post_set
            persistent = pre_set & post_set

            categories = ['New Words', 'Disappeared', 'Persistent']
            values = [len(new_words), len(disappeared), len(persistent)]
            colors_list = [COLORS['accent2'], COLORS['accent1'], COLORS['accent3']]
            bars = ax.bar(categories, values, color=colors_list, alpha=0.7)
            ax.set_ylabel('Number of Words')
            ax.set_title('Vocabulary Evolution')

            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       str(value), ha='center', va='bottom')

        # Panel 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')

        summary_text = f"""
Milestone: {milestone_info.get('name', 'Unknown')}
Year: {milestone_info.get('year', 'N/A')}

Pre-Period:
  Debates: {pre_data.get('total_debates', 0):,}
  Years: {pre_data.get('years', ['N/A'])[0]}-{pre_data.get('years', ['N/A'])[-1] if pre_data.get('years') else 'N/A'}

Post-Period:
  Debates: {post_data.get('total_debates', 0):,}
  Years: {post_data.get('years', ['N/A'])[0]}-{post_data.get('years', ['N/A'])[-1] if post_data.get('years') else 'N/A'}

Filtering: {pre_data.get('filtering_mode', 'N/A')}
"""

        ax.text(0.1, 0.9, summary_text.strip(), transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor=COLORS['background'],
                        alpha=0.8, edgecolor=COLORS['grid']))

        ax.set_title('Analysis Summary', fontsize=12, fontweight='bold')

        # Style all axes
        for ax in axes.flat:
            if ax.get_visible():
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if hasattr(ax, 'grid'):
                    ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()

        print(f"Saved milestone comparison to {output_path}")


if __name__ == "__main__":
    demo_visualizations()