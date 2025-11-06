#!/usr/bin/env python3
"""
Generate ngram (unigram/bigram) analysis for manuscript.

Computes distinctive vocabulary using TF-IDF and log-odds ratio across different splits:
- Gender: Male vs Female MPs
- Temporal: Different time periods
- Combined: Gender x Time interactions

Usage:
    python3 03_ngram_analysis.py --split gender [--filtering moderate]
    python3 03_ngram_analysis.py --split temporal --periods 1900-1950,1950-2005

Output:
    manuscript_figures/ngrams_*.png - Vocabulary comparison visualizations
    manuscript_figures/ngrams_*.csv - Top distinctive terms

FILTERING GUIDE:
-----------------
'minimal': NLTK stopwords only (~200 words)
    Use when: You want to preserve all parliamentary language
    Removes: the, of, to, and, a, etc.

'parliamentary': minimal + parliamentary terms (~60 words)
    Use when: Focus on policy topics, not procedural language
    Removes: + hon, gentleman, member, house, bill, amendment, etc.

'moderate': parliamentary + common verbs + vague words (~300 words)
    Use when: You want substantive nouns and specific verbs only
    Removes: + make, take, give, thing, case, matter, time, etc.
    RECOMMENDED for most analyses

'aggressive': moderate + discourse markers + quantifiers (~400 words)
    Use when: You want only highly distinctive content words
    Removes: + well, yes, obviously, much, many, new, old, important, etc.
    Use for: Topic-specific vocabulary, technical term extraction
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from analysis.analysis_utils import preprocess_text, get_stop_words

from data_loader import load_speeches
from utils import setup_plot_style, COLORS, save_figure


def compute_log_odds_ratio(texts1, texts2, vocabulary, smoothing=1.0):
    """
    Compute log-odds ratio for distinctive vocabulary.

    Positive values: more distinctive of texts1
    Negative values: more distinctive of texts2

    Args:
        texts1, texts2: Lists of preprocessed text strings
        vocabulary: Set of terms to analyze
        smoothing: Smoothing parameter (default 1.0)

    Returns:
        DataFrame with terms and log-odds ratios
    """
    # Count occurrences
    counts1 = Counter()
    counts2 = Counter()

    for text in texts1:
        counts1.update(text.split())

    for text in texts2:
        counts2.update(text.split())

    # Compute log-odds with smoothing
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())

    results = []
    for term in vocabulary:
        count1 = counts1.get(term, 0)
        count2 = counts2.get(term, 0)

        # Add smoothing
        odds1 = (count1 + smoothing) / (total1 + smoothing * len(vocabulary))
        odds2 = (count2 + smoothing) / (total2 + smoothing * len(vocabulary))

        log_odds = np.log(odds1 / odds2)

        results.append({
            'term': term,
            'log_odds': log_odds,
            'count_1': count1,
            'count_2': count2,
            'freq_1': count1 / total1 if total1 > 0 else 0,
            'freq_2': count2 / total2 if total2 > 0 else 0
        })

    return pd.DataFrame(results).sort_values('log_odds', key=abs, ascending=False)


def analyze_gender_split(speeches_df, filtering='moderate', top_n=20):
    """
    Analyze distinctive vocabulary by gender.

    Args:
        speeches_df: DataFrame with speeches
        filtering: Stop word filtering level
        top_n: Number of top terms to return

    Returns:
        dict with male_terms, female_terms DataFrames
    """
    print("Analyzing gender split...")

    # Filter to speeches with known gender
    gendered = speeches_df[speeches_df['gender'].notna()].copy()

    male_speeches = gendered[gendered['gender'] == 'm']
    female_speeches = gendered[gendered['gender'] == 'f']

    print(f"  Male speeches: {len(male_speeches):,}")
    print(f"  Female speeches: {len(female_speeches):,}")

    # Preprocess texts
    stop_words = get_stop_words(filtering)
    print(f"  Using {filtering} filtering ({len(stop_words)} stop words)")

    print("  Preprocessing male speeches...")
    male_texts = [
        preprocess_text(text, stop_words)
        for text in male_speeches['text'].fillna('')
    ]

    print("  Preprocessing female speeches...")
    female_texts = [
        preprocess_text(text, stop_words)
        for text in female_speeches['text'].fillna('')
    ]

    # Get vocabulary using TF-IDF
    print("  Computing TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 1),
        min_df=10,  # Must appear in at least 10 speeches
    )
    vectorizer.fit(male_texts + female_texts)
    vocabulary = set(vectorizer.get_feature_names_out())

    print(f"  Vocabulary size: {len(vocabulary)}")

    # Compute log-odds
    print("  Computing log-odds ratios...")
    log_odds_df = compute_log_odds_ratio(male_texts, female_texts, vocabulary)

    # Get top terms for each gender
    male_terms = log_odds_df.nlargest(top_n, 'log_odds').copy()
    female_terms = log_odds_df.nsmallest(top_n, 'log_odds').copy()
    female_terms['log_odds'] = female_terms['log_odds'].abs()  # Make positive for plotting

    return {
        'male_terms': male_terms,
        'female_terms': female_terms,
        'all_terms': log_odds_df
    }


def analyze_temporal_split(speeches_df, periods, filtering='moderate', top_n=20):
    """
    Analyze distinctive vocabulary across time periods.

    Args:
        speeches_df: DataFrame with speeches
        periods: List of (start_year, end_year, label) tuples
        filtering: Stop word filtering level
        top_n: Number of top terms to return per period

    Returns:
        dict with period comparisons
    """
    print("Analyzing temporal split...")

    stop_words = get_stop_words(filtering)
    print(f"  Using {filtering} filtering ({len(stop_words)} stop words)")

    results = {}

    for i, (start1, end1, label1) in enumerate(periods):
        for j, (start2, end2, label2) in enumerate(periods):
            if j <= i:
                continue  # Only compare each pair once

            print(f"\n  Comparing {label1} vs {label2}")

            # Get speeches for each period
            period1 = speeches_df[
                (speeches_df['year'] >= start1) &
                (speeches_df['year'] <= end1)
            ]
            period2 = speeches_df[
                (speeches_df['year'] >= start2) &
                (speeches_df['year'] <= end2)
            ]

            print(f"    {label1}: {len(period1):,} speeches")
            print(f"    {label2}: {len(period2):,} speeches")

            # Preprocess
            texts1 = [
                preprocess_text(text, stop_words)
                for text in period1['text'].fillna('')
            ]
            texts2 = [
                preprocess_text(text, stop_words)
                for text in period2['text'].fillna('')
            ]

            # Get vocabulary
            vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 1),
                min_df=10,
            )
            vectorizer.fit(texts1 + texts2)
            vocabulary = set(vectorizer.get_feature_names_out())

            # Compute log-odds
            log_odds_df = compute_log_odds_ratio(texts1, texts2, vocabulary)

            # Store results
            key = f"{label1}_vs_{label2}"
            results[key] = {
                'period1_terms': log_odds_df.nlargest(top_n, 'log_odds'),
                'period2_terms': log_odds_df.nsmallest(top_n, 'log_odds'),
                'all_terms': log_odds_df,
                'labels': (label1, label2)
            }

    return results


def plot_gender_ngrams(gender_results, filtering_level):
    """Plot distinctive terms by gender."""
    print("Creating gender ngram visualization...")

    male_terms = gender_results['male_terms']
    female_terms = gender_results['female_terms']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Male distinctive terms
    ax1.barh(
        range(len(male_terms)),
        male_terms['log_odds'],
        color=COLORS['male'],
        alpha=0.7
    )
    ax1.set_yticks(range(len(male_terms)))
    ax1.set_yticklabels(male_terms['term'])
    ax1.set_xlabel('Log-odds ratio')
    ax1.set_title('Distinctive Male MP Terms')
    ax1.invert_yaxis()

    # Female distinctive terms
    ax2.barh(
        range(len(female_terms)),
        female_terms['log_odds'],
        color=COLORS['female'],
        alpha=0.7
    )
    ax2.set_yticks(range(len(female_terms)))
    ax2.set_yticklabels(female_terms['term'])
    ax2.set_xlabel('Log-odds ratio')
    ax2.set_title('Distinctive Female MP Terms')
    ax2.invert_yaxis()

    fig.suptitle(f'Distinctive Vocabulary by Gender (filtering: {filtering_level})', fontsize=14)

    return fig


def plot_temporal_ngrams(temporal_results, filtering_level):
    """Plot distinctive terms across time periods."""
    print("Creating temporal ngram visualizations...")

    figures = []

    for key, results in temporal_results.items():
        label1, label2 = results['labels']
        period1_terms = results['period1_terms']
        period2_terms = results['period2_terms']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

        # Period 1 distinctive terms
        ax1.barh(
            range(len(period1_terms)),
            period1_terms['log_odds'],
            color=COLORS['male'],
            alpha=0.7
        )
        ax1.set_yticks(range(len(period1_terms)))
        ax1.set_yticklabels(period1_terms['term'])
        ax1.set_xlabel('Log-odds ratio')
        ax1.set_title(f'Distinctive Terms: {label1}')
        ax1.invert_yaxis()

        # Period 2 distinctive terms
        period2_terms_plot = period2_terms.copy()
        period2_terms_plot['log_odds'] = period2_terms_plot['log_odds'].abs()

        ax2.barh(
            range(len(period2_terms_plot)),
            period2_terms_plot['log_odds'],
            color=COLORS['female'],
            alpha=0.7
        )
        ax2.set_yticks(range(len(period2_terms_plot)))
        ax2.set_yticklabels(period2_terms_plot['term'])
        ax2.set_xlabel('Log-odds ratio')
        ax2.set_title(f'Distinctive Terms: {label2}')
        ax2.invert_yaxis()

        fig.suptitle(
            f'Distinctive Vocabulary: {label1} vs {label2} (filtering: {filtering_level})',
            fontsize=14
        )

        figures.append((fig, key))

    return figures


def main():
    """Generate ngram analysis visualizations."""
    parser = argparse.ArgumentParser(
        description='Generate ngram analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--split',
        choices=['gender', 'temporal', 'both'],
        default='gender',
        help='Type of split analysis'
    )
    parser.add_argument(
        '--filtering',
        choices=['minimal', 'parliamentary', 'moderate', 'aggressive'],
        default='moderate',
        help='Stop word filtering level (see FILTERING GUIDE above)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of top terms to show (default: 20)'
    )
    parser.add_argument(
        '--periods',
        type=str,
        help='Time periods for temporal split, format: 1900-1950:Pre1950,1950-2005:Post1950'
    )
    parser.add_argument(
        '--include-lords',
        action='store_true',
        help='Include Lords speeches (default: Commons only)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GENERATING NGRAM ANALYSIS VISUALIZATIONS")
    print("=" * 70)

    # Setup plotting style
    setup_plot_style()

    # Load all speeches with known gender (Commons only by default)
    chamber = None if args.include_lords else 'Commons'
    print(f"\nLoading speeches ({'Commons only' if chamber else 'All chambers'})...")
    speeches = load_speeches(chamber=chamber)
    speeches = speeches[speeches['gender'].notna()]  # Only gendered speeches
    print(f"  Loaded {len(speeches):,} speeches with known gender")

    # Gender split analysis
    if args.split in ['gender', 'both']:
        print("\n" + "-" * 70)
        gender_results = analyze_gender_split(speeches, args.filtering, args.top_n)

        # Save top terms
        gender_results['male_terms'].to_csv(
            get_output_dir() / f'ngrams_male_top_{args.filtering}.csv',
            index=False
        )
        gender_results['female_terms'].to_csv(
            get_output_dir() / f'ngrams_female_top_{args.filtering}.csv',
            index=False
        )

        # Plot
        fig = plot_gender_ngrams(gender_results, args.filtering)
        save_figure(fig, f'ngrams_gender_{args.filtering}.png')
        plt.close(fig)

    # Temporal split analysis
    if args.split in ['temporal', 'both']:
        print("\n" + "-" * 70)

        # Parse periods
        if args.periods:
            periods = []
            for period_str in args.periods.split(','):
                years, label = period_str.split(':')
                start, end = map(int, years.split('-'))
                periods.append((start, end, label))
        else:
            # Default periods
            periods = [
                (1803, 1918, 'Pre-Suffrage'),
                (1919, 1945, 'Interwar'),
                (1946, 1979, 'Post-War'),
                (1980, 2005, 'Modern'),
            ]

        temporal_results = analyze_temporal_split(speeches, periods, args.filtering, args.top_n)

        # Plot and save
        figures = plot_temporal_ngrams(temporal_results, args.filtering)
        for fig, key in figures:
            save_figure(fig, f'ngrams_temporal_{key}_{args.filtering}.png')
            plt.close(fig)

            # Save top terms
            temporal_results[key]['period1_terms'].to_csv(
                get_output_dir() / f'ngrams_{key}_period1_{args.filtering}.csv',
                index=False
            )
            temporal_results[key]['period2_terms'].to_csv(
                get_output_dir() / f'ngrams_{key}_period2_{args.filtering}.csv',
                index=False
            )

    print("\n" + "=" * 70)
    print("NGRAM ANALYSIS COMPLETE")
    print("=" * 70)


def get_output_dir():
    """Get output directory."""
    from utils import get_output_dir as _get_output_dir
    return _get_output_dir()


if __name__ == '__main__':
    main()
