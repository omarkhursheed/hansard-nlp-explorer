#!/usr/bin/env python3
"""
Common Analysis Utilities for Hansard Parliamentary Debates

Provides shared functions for text preprocessing, n-gram analysis,
topic modeling, and visualization across different analysis scripts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import NLTK for standard stop words
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available, using minimal stop word list")

# Professional color palette
COLORS = {
    'male': '#3B82C4',      # Professional blue
    'female': '#EC4899',     # Professional pink/magenta
    'background': '#FFFFFF',  # White
    'grid': '#E5E7EB',       # Light gray for gridlines
    'text': '#1F2937',       # Dark gray for text
    'muted': '#9CA3AF',      # Medium gray
    'accent1': '#10B981',    # Emerald
    'accent2': '#F59E0B',    # Amber
    'primary': '#2E86AB',    # Primary blue
    'secondary': '#A23B72',  # Secondary pink
}

# Set matplotlib style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def preprocess_text(text: str, stop_words: set) -> str:
    """
    Sophisticated text preprocessing based on older analysis approach.
    
    - Removes formatting artifacts and parliamentary metadata
    - Normalizes whitespace & case
    - Handles possessives and contractions properly
    - Removes tokens <2 characters
    - Applies stop word filtering
    """
    if not isinstance(text, str):
        return ""
    
    # Remove markup and line artifacts as in production analysis
    text = re.sub(r'[\r\n]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Strip markdown/quotes - use separate replacements to avoid regex issues
    text = re.sub(r'[_*`]', '', text)
    text = re.sub(r'["""]', '', text)
    text = re.sub(r"['']", '', text)
    text = re.sub(r'\[.*?\]', '', text)           # Drop [refs]
    # Remove numbers, punctuation EXCEPT inner apostrophes (as in O'Reilly)
    text = re.sub(r"(?!\B'\b)[^\w\s']", '', text)
    text = text.lower().strip()
    
    # Tokenize using the same method as main analysis (split on whitespace)
    tokens = text.split()
    filtered = []
    for w in tokens:
        # Remove 's (possessives) but keep legitimate contractions
        if w.endswith("'s"):
            w = w[:-2]
        # Remove leading/trailing apostrophes (e.g. 'em, o'clock -> em, o'clock)
        w = w.strip("'")
        # Remove tokens <2 characters (to match corpus cleaning)
        if len(w) < 2:
            continue
        if w not in stop_words:
            filtered.append(w)
    return ' '.join(filtered)


def get_stop_words(level='moderate'):
    """
    Get stop words based on filtering level.

    Uses NLTK English stopwords as base (198 words), then adds domain-specific
    terms based on filtering level.
    """
    # Get comprehensive English stop words from NLTK
    if NLTK_AVAILABLE:
        basic_stop_words = set(stopwords.words('english'))
        # Add missing modal verbs that NLTK doesn't include
        # Based on corpus analysis: these appear in >70% of speeches with no semantic value
        missing_modals = {'could', 'might', 'shall', 'must', 'ought', 'would', 'may'}
        basic_stop_words = basic_stop_words | missing_modals
    else:
        # Fallback minimal set if NLTK not available
        basic_stop_words = {
            'the', 'of', 'to', 'and', 'a', 'an', 'in', 'is', 'it', 'that', 'was', 'for',
            'on', 'are', 'as', 'with', 'his', 'they', 'at', 'be', 'this', 'have', 'from',
            'or', 'had', 'by', 'not', 'but', 'what', 'all', 'were', 'we', 'when', 'there',
            'can', 'said', 'which', 'do', 'their', 'if', 'will', 'up', 'other', 'about',
            'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make',
            'like', 'into', 'him', 'has', 'two', 'more', 'very', 'been', 'am', 'could',
            'might', 'shall', 'must', 'ought', 'should',
            # Add missing common pronouns/determiners
            'my', 'me', 'he', 'she', 'its', 'our', 'who', 'any', 'does', 'than',
            'those', 'only', 'over', 'under', 'because', 'where', 'before', 'such'
        }

    parliamentary_terms = {
        'hon', 'honourable', 'right', 'gentleman', 'lady', 'member', 'members',
        'house', 'speaker', 'sir', 'madam', 'mr', 'mrs', 'ms', 'lord', 'lords',
        'gallant', 'learned', 'friend', 'friends', 'noble', 'bill', 'clause',
        'amendment', 'committee', 'order', 'question', 'division', 'reading',
        'report', 'stage', 'moved', 'second', 'third', 'government', 'minister',
        'secretary', 'state', 'department', 'debate', 'discuss', 'speech', 'words',
        'statement'
    }

    common_verbs = {
        'make', 'made', 'making', 'makes', 'take', 'took', 'taken', 'taking',
        'give', 'gave', 'given', 'giving', 'put', 'puts', 'putting', 'get', 'got',
        'getting', 'gets', 'come', 'came', 'coming', 'comes', 'go', 'went', 'going',
        'goes', 'gone', 'say', 'said', 'saying', 'says', 'think', 'thought',
        'thinking', 'thinks', 'know', 'knew', 'known', 'knowing', 'see', 'saw',
        'seeing', 'sees', 'seen', 'want', 'wanted', 'wanting', 'wants', 'look',
        'looked', 'looking', 'looks', 'find', 'found', 'finding', 'finds', 'tell',
        'told', 'telling', 'tells', 'ask', 'asked', 'asking', 'asks', 'seem',
        'seemed', 'seeming', 'seems', 'feel', 'felt', 'feeling', 'feels', 'try',
        'tried', 'trying', 'tries', 'leave', 'left', 'leaving', 'leaves', 'call',
        'called', 'calling', 'calls', 'keep', 'kept', 'keeping', 'keeps', 'believe',
        'believed', 'hope', 'hoped', 'wish', 'wished', 'need', 'needed', 'use',
        'used', 'using', 'uses', 'work', 'worked', 'working', 'works'
    }

    vague_words = {
        'thing', 'things', 'something', 'anything', 'nothing', 'everything', 'way',
        'ways', 'case', 'cases', 'matter', 'matters', 'fact', 'facts', 'time',
        'times', 'place', 'places', 'part', 'parts', 'kind', 'kinds', 'point',
        'points', 'view', 'views', 'position', 'number', 'numbers', 'example',
        'examples', 'word', 'words', 'subject', 'course', 'present', 'regard',
        'deal', 'side', 'end', 'result', 'means', 'moment', 'today', 'yesterday',
        'tomorrow', 'now', 'then', 'year', 'years', 'day', 'days', 'week', 'weeks',
        'month', 'months', 'last', 'next', 'first', 'second'
    }

    # Parliamentary artifacts from references and formatting
    parliamentary_artifacts = {
        'deb', 'vol', 'hc', 'hl', 'rt', 'hon', 'c', 'cc', 'col', 'cols', 'w', 'ws',
        'parl', 'sess', 'ser', 'hansard', 'official', 'per', 'cent', 'proc'
    }

    if level == 'minimal':
        return basic_stop_words
    elif level == 'basic':
        return basic_stop_words
    elif level == 'parliamentary':
        return basic_stop_words | parliamentary_terms
    elif level == 'moderate':
        return basic_stop_words | parliamentary_terms | common_verbs | vague_words
    elif level == 'aggressive':
        discourse_markers = {'well', 'yes', 'no', 'indeed', 'perhaps', 'certainly',
                           'obviously', 'clearly', 'surely', 'really', 'quite', 'sure',
                           'also', 'however', 'therefore', 'whether', 'still', 'always',
                           'never', 'yet', 'already', 'rather', 'even', 'far'}

        # Quantifiers and measurements (>60% frequency in corpus)
        quantifiers = {'much', 'many', 'more', 'most', 'less', 'least', 'few', 'fewer',
                      'several', 'hundred', 'thousand', 'million', 'one'}

        # Common adjectives with little discriminative value
        adjectives = {'new', 'old', 'good', 'bad', 'great', 'small', 'large', 'important',
                     'different', 'same', 'certain', 'possible', 'necessary', 'clear',
                     'particular', 'general', 'special', 'full', 'long', 'short', 'high', 'low'}

        return basic_stop_words | parliamentary_terms | common_verbs | vague_words | discourse_markers | parliamentary_artifacts | quantifiers | adjectives
    else:
        return basic_stop_words | parliamentary_terms | common_verbs | vague_words


def analyze_ngrams(texts, n_range=(1, 2), max_features=1000, stop_words=None):
    """
    Analyze n-grams in the text corpus.

    Args:
        texts: List of text strings
        n_range: Tuple of (min_n, max_n) for n-gram range
        max_features: Maximum number of features to extract
        stop_words: Set of stop words to filter

    Returns:
        Tuple of (ngram_freq list, vectorizer)
    """
    print(f"Analyzing {n_range[0]}-{n_range[1]}-grams...")

    combined_texts = [' '.join(text.split()[:1000]) for text in texts]
    if isinstance(stop_words, set):
        stop_words = list(stop_words)

    vectorizer = CountVectorizer(
        ngram_range=n_range,
        max_features=max_features,
        stop_words=stop_words,
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]+\b'
    )

    X = vectorizer.fit_transform(combined_texts)
    feature_names = vectorizer.get_feature_names_out()
    frequencies = X.sum(axis=0).A1
    ngram_freq = list(zip(feature_names, frequencies))
    ngram_freq.sort(key=lambda x: x[1], reverse=True)

    return ngram_freq, vectorizer


def perform_topic_modeling(texts, n_topics=8, max_features=1000, stop_words=None):
    """
    Perform LDA topic modeling.

    Args:
        texts: List of text strings
        n_topics: Number of topics to extract
        max_features: Maximum number of features
        stop_words: Set of stop words to filter

    Returns:
        Tuple of (topics list, lda model, vectorizer)
    """
    print(f"Performing topic modeling with {n_topics} topics...")

    combined_texts = [' '.join(text.split()[:1000]) for text in texts]
    if isinstance(stop_words, set):
        stop_words = list(stop_words)

    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]+\b'
    )

    X = vectorizer.fit_transform(combined_texts)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=10,
        learning_decay=0.7
    )
    lda.fit(X)

    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            'topic_id': topic_idx,
            'top_words': top_words,
            'weights': topic[top_words_idx]
        })

    return topics, lda, vectorizer


def create_ngram_visualization(ngram_freq, title, output_path, top_n=30, color='primary'):
    """
    Create n-gram frequency visualization.

    Args:
        ngram_freq: List of (ngram, count) tuples (already filtered by preprocessing)
        title: Chart title
        output_path: Output file path
        top_n: Number of items to display (default: 30)
        color: Color key from COLORS dict (default: 'primary')
    """
    top_ngrams = ngram_freq[:top_n]
    if not top_ngrams:
        print(f"No data to visualize for {title}")
        return

    words, counts = zip(*top_ngrams)

    plt.figure(figsize=(14, 10))
    bars = plt.barh(range(len(words)), counts, color=COLORS[color], alpha=0.8)
    plt.yticks(range(len(words)), words, fontsize=10)
    plt.xlabel('Frequency', fontsize=11, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()

    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{count:,}', va='center', fontsize=9)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()


def create_topic_visualization(topics, output_path):
    """
    Create compact topic modeling visualization.

    Args:
        topics: List of topic dicts with 'topic_id', 'top_words', 'weights'
        output_path: Path to save the visualization
    """
    n_topics = len(topics)

    if n_topics <= 4:
        fig, axes = plt.subplots(1, n_topics, figsize=(4*n_topics, 6))
        if n_topics == 1:
            axes = [axes]
    elif n_topics <= 8:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()

    for i, topic in enumerate(topics):
        if i >= len(axes):
            break
        words = topic['top_words'][:6]
        weights = topic['weights'][:6]
        axes[i].barh(range(len(words)), weights, color=COLORS['primary'], alpha=0.8)
        axes[i].set_yticks(range(len(words)))
        axes[i].set_yticklabels(words, fontsize=9)
        axes[i].set_title(f'Topic {topic["topic_id"]}', fontweight='bold', fontsize=11)
        axes[i].invert_yaxis()
        axes[i].tick_params(axis='both', which='major', labelsize=8)

    for i in range(len(topics), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Topic Modeling Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_gender_topic_visualization(male_topics, female_topics, output_path):
    """Create gender-specific topic modeling visualization"""
    n_topics = max(len(male_topics), len(female_topics))
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(2, n_topics, figsize=(3*n_topics, 10))
    if n_topics == 1:
        axes = axes.reshape(2, 1)
    
    # Male topics (top row)
    for i, topic in enumerate(male_topics):
        if i >= n_topics:
            break
        words = topic['top_words'][:6]
        weights = topic['weights'][:6]
        
        axes[0, i].barh(range(len(words)), weights, color=COLORS['male'], alpha=0.8)
        axes[0, i].set_yticks(range(len(words)))
        axes[0, i].set_yticklabels(words, fontsize=9)
        axes[0, i].set_title(f'Male Topic {topic["topic_id"]}', fontweight='bold', fontsize=11)
        axes[0, i].invert_yaxis()
        axes[0, i].tick_params(axis='both', which='major', labelsize=8)
    
    # Female topics (bottom row)
    for i, topic in enumerate(female_topics):
        if i >= n_topics:
            break
        words = topic['top_words'][:6]
        weights = topic['weights'][:6]
        
        axes[1, i].barh(range(len(words)), weights, color=COLORS['female'], alpha=0.8)
        axes[1, i].set_yticks(range(len(words)))
        axes[1, i].set_yticklabels(words, fontsize=9)
        axes[1, i].set_title(f'Female Topic {topic["topic_id"]}', fontweight='bold', fontsize=11)
        axes[1, i].invert_yaxis()
        axes[1, i].tick_params(axis='both', which='major', labelsize=8)
    
    # Hide unused subplots
    for i in range(len(male_topics), n_topics):
        axes[0, i].set_visible(False)
    for i in range(len(female_topics), n_topics):
        axes[1, i].set_visible(False)
    
    plt.suptitle('Gender-Specific Topic Modeling Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_ngrams_tfidf(texts, n_range=(1, 2), max_features=1000, stop_words=None):
    """Analyze n-grams using TF-IDF to find distinctive phrases"""
    print(f"Analyzing {n_range[0]}-{n_range[1]}-grams using TF-IDF...")
    
    # Combine all texts
    combined_texts = [' '.join(text.split()[:1000]) for text in texts]  # Limit per document
    
    # Convert stop words to list if it's a set
    if isinstance(stop_words, set):
        stop_words = list(stop_words)
    
    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=n_range,
        max_features=max_features,
        stop_words=stop_words,
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]+\b'
    )
    
    X = vectorizer.fit_transform(combined_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get mean TF-IDF scores across all documents
    mean_scores = X.mean(axis=0).A1
    ngram_scores = list(zip(feature_names, mean_scores))
    ngram_scores.sort(key=lambda x: x[1], reverse=True)
    
    return ngram_scores, vectorizer


def analyze_ngrams_logodds(male_texts, female_texts, n_range=(1, 2), max_features=1000, stop_words=None):
    """Analyze n-grams using log-odds ratio for gender comparison"""
    print(f"Analyzing {n_range[0]}-{n_range[1]}-grams using log-odds ratio...")
    
    # Convert stop words to list if it's a set
    if isinstance(stop_words, set):
        stop_words = list(stop_words)
    
    # Count n-grams in each corpus
    male_vectorizer = CountVectorizer(
        ngram_range=n_range, 
        max_features=max_features, 
        stop_words=stop_words,
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]+\b'
    )
    female_vectorizer = CountVectorizer(
        ngram_range=n_range, 
        max_features=max_features, 
        stop_words=stop_words,
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]+\b'
    )
    
    # Fit and transform
    male_X = male_vectorizer.fit_transform(male_texts)
    female_X = female_vectorizer.fit_transform(female_texts)
    
    # Get counts
    male_counts = male_X.sum(axis=0).A1
    female_counts = female_X.sum(axis=0).A1
    
    # Get feature names (use male vectorizer as reference)
    feature_names = male_vectorizer.get_feature_names_out()
    
    # Calculate log-odds ratio
    # Add pseudocount to avoid division by zero
    male_counts = male_counts + 1
    female_counts = female_counts + 1
    
    # Calculate frequencies
    male_total = male_counts.sum()
    female_total = female_counts.sum()
    
    male_freq = male_counts / male_total
    female_freq = female_counts / female_total
    
    # Log-odds = log(male_freq) - log(female_freq)
    log_odds = np.log(male_freq) - np.log(female_freq)
    
    ngram_logodds = list(zip(feature_names, log_odds))
    ngram_logodds.sort(key=lambda x: abs(x[1]), reverse=True)  # Sort by absolute value
    
    return ngram_logodds, male_vectorizer, female_vectorizer


def create_gender_ngram_visualization(male_ngrams, female_ngrams, title, output_path, top_n=15):
    """
    Create side-by-side n-gram comparison for male vs female speakers.
    Shows fewer items in a single graph split halfway down.
    """
    # Get top n-grams for each gender (fewer items)
    male_top = male_ngrams[:top_n]
    female_top = female_ngrams[:top_n]
    
    if not male_top and not female_top:
        print(f"No data to visualize for {title}")
        return
    
    # Create figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Male n-grams (top half)
    if male_top:
        male_words, male_counts = zip(*male_top)
        bars1 = ax1.barh(range(len(male_words)), male_counts, color=COLORS['male'], alpha=0.8)
        ax1.set_yticks(range(len(male_words)))
        ax1.set_yticklabels(male_words, fontsize=11)
        ax1.set_xlabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Male Speakers', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars1, male_counts)):
            ax1.text(bar.get_width() + max(male_counts) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{count:,}', va='center', fontsize=10)
    
    # Female n-grams (bottom half)
    if female_top:
        female_words, female_counts = zip(*female_top)
        bars2 = ax2.barh(range(len(female_words)), female_counts, color=COLORS['female'], alpha=0.8)
        ax2.set_yticks(range(len(female_words)))
        ax2.set_yticklabels(female_words, fontsize=11)
        ax2.set_xlabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Female Speakers', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars2, female_counts)):
            ax2.text(bar.get_width() + max(female_counts) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{count:,}', va='center', fontsize=10)
    
    # Remove spines
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()


def create_logodds_visualization(logodds_data, title, output_path, top_n=20):
    """
    Create log-odds visualization showing gender differences.
    Positive values = more male, negative values = more female.
    Shows fewer items in a single graph split halfway down.
    """
    top_data = logodds_data[:top_n]
    
    if not top_data:
        print(f"No data to visualize for {title}")
        return
    
    words, logodds = zip(*top_data)
    
    # Create figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Split data into male-favored (positive) and female-favored (negative)
    male_favored = [(w, l) for w, l in top_data if l > 0]
    female_favored = [(w, l) for w, l in top_data if l < 0]
    
    # Sort by absolute value for each group
    male_favored.sort(key=lambda x: abs(x[1]), reverse=True)
    female_favored.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Take top 10 from each group
    male_favored = male_favored[:10]
    female_favored = female_favored[:10]
    
    # Male-favored n-grams (top half)
    if male_favored:
        male_words, male_logodds = zip(*male_favored)
        bars1 = ax1.barh(range(len(male_words)), male_logodds, color=COLORS['male'], alpha=0.8)
        ax1.set_yticks(range(len(male_words)))
        ax1.set_yticklabels(male_words, fontsize=11)
        ax1.set_xlabel('Log-Odds Ratio', fontsize=12, fontweight='bold')
        ax1.set_title('More Characteristic of Male Speakers', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars1, male_logodds)):
            ax1.text(bar.get_width() + max(male_logodds) * 0.01, 
                    bar.get_y() + bar.get_height()/2,
                    f'{value:.2f}', va='center', fontsize=10)
    
    # Female-favored n-grams (bottom half)
    if female_favored:
        female_words, female_logodds = zip(*female_favored)
        bars2 = ax2.barh(range(len(female_words)), female_logodds, color=COLORS['female'], alpha=0.8)
        ax2.set_yticks(range(len(female_words)))
        ax2.set_yticklabels(female_words, fontsize=11)
        ax2.set_xlabel('Log-Odds Ratio', fontsize=12, fontweight='bold')
        ax2.set_title('More Characteristic of Female Speakers', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars2, female_logodds)):
            ax2.text(bar.get_width() + min(female_logodds) * 0.01, 
                    bar.get_y() + bar.get_height()/2,
                    f'{value:.2f}', va='center', fontsize=10)
    
    # Add vertical line at 0 for both subplots
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Remove spines
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()


def create_temporal_analysis(metadata_df, output_dir):
    """Create temporal analysis visualizations"""
    print("Creating temporal analysis...")
    
    # Debates per year
    yearly_counts = metadata_df.groupby('year').size()
    
    plt.figure(figsize=(14, 6))
    plt.plot(yearly_counts.index, yearly_counts.values, 
             color=COLORS['primary'], linewidth=2, marker='o', markersize=4)
    plt.xlabel('Year')
    plt.ylabel('Number of Debates')
    plt.title('Parliamentary Debates per Year', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_debates_per_year.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Word count trends
    yearly_words = metadata_df.groupby('year')['word_count'].sum()
    
    plt.figure(figsize=(14, 6))
    plt.plot(yearly_words.index, yearly_words.values, 
             color=COLORS['secondary'], linewidth=2, marker='o', markersize=4)
    plt.xlabel('Year')
    plt.ylabel('Total Words')
    plt.title('Total Words per Year', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_words_per_year.png', dpi=300, bbox_inches='tight')
    plt.close()


def ensure_analysis_directory(output_dir):
    """Ensure the analysis directory exists"""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
