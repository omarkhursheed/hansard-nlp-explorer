#!/usr/bin/env python3
"""
Comprehensive Analysis for Hansard Parliamentary Debates

Performs complete text analysis including:
- Unigram and bigram frequency analysis
- Topic modeling (LDA)
- Temporal trends
- Professional visualizations

Uses UnifiedDataLoader for efficient data access.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set up paths
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.unified_data_loader import UnifiedDataLoader
from utils.path_config import Paths
from analysis.analysis_utils import (
    preprocess_text, get_stop_words, analyze_ngrams, perform_topic_modeling,
    analyze_ngrams_tfidf, create_ngram_visualization, create_topic_visualization, 
    create_temporal_analysis, ensure_analysis_directory, COLORS
)


def run_comprehensive_analysis(year_range=None, sample_size=None, 
                              filtering_level='aggressive', n_topics=8):
    """Run comprehensive analysis on Hansard data"""
    
    print("=" * 70)
    print("COMPREHENSIVE HANSARD ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("Loading data...")
    loader = UnifiedDataLoader()
    data = loader.load_debates(
        source='processed_fixed',
        year_range=year_range,
        sample_size=sample_size,
        load_text=True,
        use_cache=True
    )
    
    metadata_df = data['metadata']
    text_data = data['text_data']
    
    print(f"Loaded {len(metadata_df)} debates")
    print(f"Year range: {metadata_df['year'].min()}-{metadata_df['year'].max()}")
    print(f"Total words: {metadata_df['word_count'].sum():,}")
    
    # Prepare texts
    texts = [item['full_text'] for item in text_data if item['full_text'].strip()]
    print(f"Processing {len(texts)} texts for analysis...")

    # Get stop words
    stop_words = get_stop_words(filtering_level)
    print(f"Using {filtering_level} filtering ({len(stop_words)} stop words)")

    # Preprocess all texts
    print("Preprocessing texts...")
    preprocessed_texts = [preprocess_text(text, stop_words) for text in texts]
    print(f"Preprocessed {len(preprocessed_texts)} texts")

    # Create output directory
    output_dir = Paths.ANALYSIS_DIR / 'comprehensive'
    ensure_analysis_directory(output_dir)

    # 1. Unigram analysis
    print("\n1. Analyzing unigrams...")
    unigrams, _ = analyze_ngrams(preprocessed_texts, n_range=(1, 1), max_features=1000, stop_words=None)
    create_ngram_visualization(unigrams, 'Distinctive Vocabulary - Top 15 Words',
                              output_dir / 'unigrams.png', top_n=15)

    # 2. Bigram analysis
    print("2. Analyzing bigrams...")
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

    # 3. Topic modeling
    print("3. Performing topic modeling...")
    topics, lda, vectorizer = perform_topic_modeling(preprocessed_texts, n_topics=n_topics,
                                                    max_features=1000, stop_words=None)
    create_topic_visualization(topics, output_dir / 'topic_modeling.png')
    
    # 4. Temporal analysis
    print("4. Creating temporal analysis...")
    create_temporal_analysis(metadata_df, output_dir)
    
    # 5. Save results
    results = {
        'metadata': {
            'total_debates': len(metadata_df),
            'year_range': (metadata_df['year'].min(), metadata_df['year'].max()),
            'total_words': metadata_df['word_count'].sum(),
            'filtering_level': filtering_level,
            'n_topics': n_topics
        },
        'unigrams': unigrams[:50],
        'bigrams': bigrams[:50],
        'bigrams_tfidf': bigrams_tfidf[:50],
        'topics': topics
    }
    
    import json
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    print(f"Generated visualizations:")
    print(f"  - unigrams.png")
    print(f"  - bigrams.png") 
    print(f"  - bigrams_tfidf.png")
    print(f"  - topic_modeling.png")
    print(f"  - temporal_debates_per_year.png")
    print(f"  - temporal_words_per_year.png")
    print(f"  - analysis_results.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Hansard Analysis')
    parser.add_argument('--years', type=str, help='Year range (e.g., "1900-1950")')
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
    run_comprehensive_analysis(
        year_range=year_range,
        sample_size=args.sample,
        filtering_level=args.filtering,
        n_topics=args.topics
    )
