import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.path_config import Paths
from analysis.analysis_utils import (
    preprocess_text, get_stop_words, analyze_ngrams, perform_topic_modeling,
    analyze_ngrams_tfidf, analyze_ngrams_logodds, create_ngram_visualization, 
    create_topic_visualization, create_gender_ngram_visualization, create_logodds_visualization,
    create_gender_topic_visualization, create_temporal_analysis, ensure_analysis_directory, COLORS
)

# Load data
processed_fixed_path = Paths.PROCESSED_FIXED
gender_analysis_enhanced_path = Paths.GENDER_ENHANCED_DATA
derived_speeches_path = Paths.DATA_DIR / 'derived' / 'gender_speeches'
derived_debates_path = Paths.DATA_DIR / 'derived' / 'gender_debates'


# Load metadata - handle PyArrow issues with nested data
def load_gender_data(year_range: Tuple[int, int], sample_size=None):
    """
    Load gender analysis data, handling PyArrow issues with nested data.
    
    Args:
        year_range: Tuple of (start_year, end_year) to load
        sample_size: If provided, limit to this many debates for testing
    """
    print("Loading gender analysis data...")
    
    # Try the full dataset first
    try:
        all_debates_path = Paths.GENDER_ENHANCED_DATA / 'ALL_debates_enhanced_with_text.parquet'
        print(f"Attempting to load full dataset: {all_debates_path}")
        df = pd.read_parquet(all_debates_path)
        print("Successfully loaded full dataset")
        return df
    except Exception as e:
        print(f"Failed to load full dataset: {e}")
        print("Loading individual year files...")
    
    # Load individual year files
    if year_range is None:
        # Load a range of years for analysis
        year_range = (1990, 2001)  # 1990-2000 for comprehensive analysis
    
    dataframes = []
    files_loaded = 0
    
    for year in range(year_range[0], year_range[1] + 1):
        if sample_size and files_loaded >= sample_size:
            break
            
        file_path = Paths.GENDER_ENHANCED_DATA / f'debates_{year}_enhanced.parquet'
        if file_path.exists():
            try:
                print(f"Loading {year} data...")
                year_df = pd.read_parquet(file_path)
                dataframes.append(year_df)
                files_loaded += 1
                print(f"  Loaded {len(year_df)} debates from {year}")
            except Exception as e:
                print(f"  Failed to load {year}: {e}")
        else:
            print(f"  File not found: {file_path}")
    
    if not dataframes:
        raise RuntimeError("No data files could be loaded")
    
    # Combine all dataframes
    df = pd.concat(dataframes, ignore_index=True)
    print(f"Successfully loaded {len(dataframes)} year files")
    print(f"Total dataset shape: {df.shape}")
    
    return df

def load_derived_speeches_data(year_range: Tuple[int, int]):
    derived_speeches_path = Paths.DATA_DIR / 'derived' / 'gender_speeches'
    derived_speeches_df = pd.read_parquet(derived_speeches_path)
    derived_speeches_df = derived_speeches_df[derived_speeches_df['year'] >= year_range[0]]
    derived_speeches_df = derived_speeches_df[derived_speeches_df['year'] <= year_range[1]]
    return derived_speeches_df

def analyze_gender_data(df):
    print(f"\nDataset info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Total debates: {len(df)}")
    print(f"Debates with text: {df['has_text'].sum()}")
    print(f"Debates with female MPs: {df['has_female'].sum()}")
    print(f"Debates with male MPs: {df['has_male'].sum()}")

    print("\nFirst few rows:")
    print(df[['debate_id', 'year', 'title', 'has_female', 'has_male', 'gender_ratio']].head())

def run_gendered_comprehensive_analysis(year_range=None, sample_size=None, 
                                       filtering_level='aggressive', n_topics=8):
    """Run comprehensive gendered analysis on Hansard data"""
    
    print("=" * 70)
    print("GENDERED COMPREHENSIVE HANSARD ANALYSIS")
    print("=" * 70)
    
    # Load data
    df = load_gender_data(year_range=year_range, sample_size=sample_size)
    
    print(f"Loaded {len(df)} debates")
    print(f"Year range: {df['year'].min()}-{df['year'].max()}")
    print(f"Debates with female MPs: {df['has_female'].sum()}")
    print(f"Debates with male MPs: {df['has_male'].sum()}")
    
    # Create output directory
    output_dir = Paths.ANALYSIS_DIR / 'gendered_comprehensive'
    ensure_analysis_directory(output_dir)
    
    # Get stop words
    stop_words = get_stop_words(filtering_level)
    print(f"Using {filtering_level} filtering ({len(stop_words)} stop words)")
    
    # Prepare texts for analysis
    print("Preparing texts for analysis...")
    texts = df[df['has_text'] == True]['debate_text'].tolist()
    print(f"Processing {len(texts)} texts for analysis...")
    
    # Preprocess all texts
    print("Preprocessing texts...")
    preprocessed_texts = [preprocess_text(text, stop_words) for text in texts]
    print(f"Preprocessed {len(preprocessed_texts)} texts")
    
    # Separate male and female debates for comparison
    male_debates = df[df['has_male'] == True]
    female_debates = df[df['has_female'] == True]
    
    male_texts = [preprocess_text(text, stop_words) for text in male_debates['debate_text'].tolist() if isinstance(text, str)]
    female_texts = [preprocess_text(text, stop_words) for text in female_debates['debate_text'].tolist() if isinstance(text, str)]
    
    print(f"Male debates: {len(male_texts)}")
    print(f"Female debates: {len(female_texts)}")
    
    # 1. Overall unigram analysis
    print("\n1. Analyzing overall unigrams...")
    unigrams, _ = analyze_ngrams(preprocessed_texts, n_range=(1, 1), max_features=1000, stop_words=None)
    create_ngram_visualization(unigrams, 'Distinctive Vocabulary - Top 15 Words',
                              output_dir / 'unigrams.png', top_n=15)
    
    # 2. Overall bigram analysis
    print("2. Analyzing overall bigrams...")
    bigrams, _ = analyze_ngrams(preprocessed_texts, n_range=(2, 2), max_features=1000, stop_words=None)
    create_ngram_visualization(bigrams, 'Distinctive Phrases - Top 15 Bigrams',
                              output_dir / 'bigrams.png', top_n=15)
    
    # 2b. TF-IDF bigram analysis
    print("2b. Analyzing bigrams using TF-IDF...")
    # Use light filtering for TF-IDF to preserve meaningful bigrams
    light_stop_words = get_stop_words('basic')  # Basic English stop words only
    bigrams_tfidf, _ = analyze_ngrams_tfidf(preprocessed_texts, n_range=(2, 2), max_features=1000, stop_words=light_stop_words)
    create_ngram_visualization(bigrams_tfidf, 'Distinctive Phrases (TF-IDF) - Top 15 Bigrams',
                              output_dir / 'bigrams_tfidf.png', top_n=15)
    
    # 3. Gender-specific unigram analysis
    print("3. Analyzing gender-specific unigrams...")
    male_unigrams, _ = analyze_ngrams(male_texts, n_range=(1, 1), max_features=1000, stop_words=None)
    female_unigrams, _ = analyze_ngrams(female_texts, n_range=(1, 1), max_features=1000, stop_words=None)
    create_gender_ngram_visualization(male_unigrams, female_unigrams, 
                                    'Gender-Specific Vocabulary - Top 15 Words',
                                    output_dir / 'gender_unigrams.png', top_n=15)
    
    # 4. Gender-specific bigram analysis
    print("4. Analyzing gender-specific bigrams...")
    male_bigrams, _ = analyze_ngrams(male_texts, n_range=(2, 2), max_features=1000, stop_words=None)
    female_bigrams, _ = analyze_ngrams(female_texts, n_range=(2, 2), max_features=1000, stop_words=None)
    create_gender_ngram_visualization(male_bigrams, female_bigrams, 
                                    'Gender-Specific Phrases - Top 15 Bigrams',
                                    output_dir / 'gender_bigrams.png', top_n=15)
    
    # 4b. Gender-specific bigram log-odds analysis
    print("4b. Analyzing gender-specific bigrams using log-odds...")
    # Use light filtering for log-odds to preserve meaningful bigrams
    bigrams_logodds, _, _ = analyze_ngrams_logodds(male_texts, female_texts, n_range=(2, 2), max_features=1000, stop_words=light_stop_words)
    create_logodds_visualization(bigrams_logodds, 'Gender-Specific Phrases (Log-Odds) - Top 20 Bigrams',
                                output_dir / 'gender_bigrams_logodds.png', top_n=20)
    
    # 4c. Gender-specific unigram log-odds analysis
    print("4c. Analyzing gender-specific unigrams using log-odds...")
    unigrams_logodds, _, _ = analyze_ngrams_logodds(male_texts, female_texts, n_range=(1, 1), max_features=1000, stop_words=light_stop_words)
    create_logodds_visualization(unigrams_logodds, 'Gender-Specific Vocabulary (Log-Odds) - Top 20 Words',
                                output_dir / 'gender_unigrams_logodds.png', top_n=20)
    
    # 5. Topic modeling
    print("5. Performing topic modeling...")
    topics, lda, vectorizer = perform_topic_modeling(preprocessed_texts, n_topics=n_topics,
                                                    max_features=1000, stop_words=None)
    create_topic_visualization(topics, output_dir / 'topic_modeling.png')
    
    # 5b. Gender-specific topic modeling
    print("5b. Performing gender-specific topic modeling...")
    male_topics, male_lda, male_vectorizer = perform_topic_modeling(male_texts, n_topics=n_topics,
                                                                   max_features=1000, stop_words=None)
    female_topics, female_lda, female_vectorizer = perform_topic_modeling(female_texts, n_topics=n_topics,
                                                                         max_features=1000, stop_words=None)
    create_gender_topic_visualization(male_topics, female_topics, output_dir / 'gender_topic_modeling.png')
    
    # 6. Temporal analysis
    print("6. Creating temporal analysis...")
    create_temporal_analysis(df, output_dir)
    
    # 7. Gender-specific temporal analysis
    print("7. Creating gender-specific temporal analysis...")
    create_gender_debate_temporal_analysis(df, output_dir)
    
    # 8. Gender speech proportion analysis
    print("8. Creating gender speech proportion analysis...")
    create_gender_speech_proportion_within_debate_analysis(df, output_dir)
    
    # 9. Save results
    results = {
        'metadata': {
            'total_debates': len(df),
            'year_range': (df['year'].min(), df['year'].max()),
            'debates_with_female': df['has_female'].sum(),
            'debates_with_male': df['has_male'].sum(),
            'filtering_level': filtering_level,
            'n_topics': n_topics
        },
        'overall_unigrams': unigrams[:50],
        'overall_bigrams': bigrams[:50],
        'overall_bigrams_tfidf': bigrams_tfidf[:50],
        'male_unigrams': male_unigrams[:50],
        'female_unigrams': female_unigrams[:50],
        'male_bigrams': male_bigrams[:50],
        'female_bigrams': female_bigrams[:50],
        'bigrams_logodds': bigrams_logodds[:50],
        'unigrams_logodds': unigrams_logodds[:50],
        'topics': topics,
        'male_topics': male_topics,
        'female_topics': female_topics
    }
    
    import json
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    print(f"Generated visualizations:")
    print(f"  - unigrams.png")
    print(f"  - bigrams.png") 
    print(f"  - bigrams_tfidf.png")
    print(f"  - gender_unigrams.png")
    print(f"  - gender_bigrams.png")
    print(f"  - gender_bigrams_logodds.png")
    print(f"  - gender_unigrams_logodds.png")
    print(f"  - topic_modeling.png")
    print(f"  - gender_topic_modeling.png")
    print(f"  - temporal_debates_per_year.png")
    print(f"  - temporal_words_per_year.png")
    print(f"  - gender_temporal_analysis.png")
    print(f"  - gender_speech_proportion_analysis.png")
    print(f"  - gender_speech_counts_analysis.png")
    print(f"  - analysis_results.json")


def create_gender_debate_temporal_analysis(df, output_dir):
    """Create gender-specific temporal analysis visualizations"""
    print("Creating gender-specific temporal analysis...")
    
    # Debates per year by gender
    female_debates_by_year = df[df['has_female'] == True].groupby('year').size()
    male_debates_by_year = df[df['has_male'] == True].groupby('year').size()
    
    plt.figure(figsize=(14, 6))
    plt.plot(female_debates_by_year.index, female_debates_by_year.values, 
             color=COLORS['female'], linewidth=2, marker='o', markersize=4, label='Female Debates')
    plt.plot(male_debates_by_year.index, male_debates_by_year.values, 
             color=COLORS['male'], linewidth=2, marker='s', markersize=4, label='Male Debates')
    plt.xlabel('Year')
    plt.ylabel('Number of Debates')
    plt.title('Parliamentary Debates per Year by Gender', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'gender_temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_gender_speech_proportion_within_debate_analysis(df, output_dir):
    """Create gender speech proportion analysis within debates over time"""
    print("Creating gender speech proportion analysis...")
    
    # Load derived speeches data for the same year range
    year_range = (df['year'].min(), df['year'].max())
    derived_speeches_df = load_derived_speeches_data(year_range=year_range)
    
    # Calculate speech counts by gender and year
    yearly_speeches = derived_speeches_df.groupby(['year', 'gender']).size().unstack(fill_value=0)
    
    # Calculate proportions
    yearly_speeches['total'] = yearly_speeches.sum(axis=1)
    yearly_speeches['female_proportion'] = yearly_speeches.get('f', 0) / yearly_speeches['total']
    yearly_speeches['male_proportion'] = yearly_speeches.get('m', 0) / yearly_speeches['total']
    
    # Create the visualization
    plt.figure(figsize=(14, 8))
    
    # Plot proportions
    plt.plot(yearly_speeches.index, yearly_speeches['female_proportion'], 
             color=COLORS['female'], linewidth=3, marker='o', markersize=6, 
             label='Female Speech Proportion', alpha=0.8)
    plt.plot(yearly_speeches.index, yearly_speeches['male_proportion'], 
             color=COLORS['male'], linewidth=3, marker='s', markersize=6, 
             label='Male Speech Proportion', alpha=0.8)
    
    # Add a line at 50% for reference
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Equal Representation')
    
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Proportion of Speeches', fontsize=12, fontweight='bold')
    plt.title('Gender Speech Proportion Within Debates Over Time', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits and ticks
    plt.ylim(0, 1)
    plt.yticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0], 
               ['0%', '20%', '40%', '50%', '60%', '80%', '100%'])
    
    # Add some statistics as text
    avg_female_prop = yearly_speeches['female_proportion'].mean()
    avg_male_prop = yearly_speeches['male_proportion'].mean()
    
    stats_text = f'Average Female Proportion: {avg_female_prop:.1%}\nAverage Male Proportion: {avg_male_prop:.1%}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gender_speech_proportion_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a stacked area chart showing the evolution
    plt.figure(figsize=(14, 8))
    
    # Prepare data for stacked area chart
    years = yearly_speeches.index
    female_counts = yearly_speeches.get('f', 0)
    male_counts = yearly_speeches.get('m', 0)
    
    plt.fill_between(years, 0, female_counts, color=COLORS['female'], alpha=0.7, label='Female Speeches')
    plt.fill_between(years, female_counts, female_counts + male_counts, color=COLORS['male'], alpha=0.7, label='Male Speeches')
    
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Speeches', fontsize=12, fontweight='bold')
    plt.title('Absolute Speech Counts by Gender Over Time', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gender_speech_counts_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gender speech proportion analysis saved to:")
    print(f"  - gender_speech_proportion_analysis.png")
    print(f"  - gender_speech_counts_analysis.png")

    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Gendered Comprehensive Hansard Analysis')
    parser.add_argument('--years', type=str, help='Year range (e.g., "1900-1950")', default='1801-2005')
    parser.add_argument('--sample', type=int, help='Sample size for testing')
    parser.add_argument('--filtering', choices=['minimal', 'basic', 'parliamentary', 'moderate', 'aggressive'], 
                       default='aggressive', help='Stop word filtering level')
    parser.add_argument('--topics', type=int, default=8, help='Number of topics for LDA')
    
    args = parser.parse_args()
    
    # Parse year range
    year_range = None
    if args.years:
        start_year, end_year = map(int, args.years.split('-'))
        year_range = (start_year, end_year)
    
    # Run analysis
    run_gendered_comprehensive_analysis(
        year_range=year_range,
        sample_size=args.sample,
        filtering_level=args.filtering,
        n_topics=args.topics
    )