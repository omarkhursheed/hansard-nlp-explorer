import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
from collections import Counter

# Import unified data loader
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.unified_data_loader import UnifiedDataLoader

# Set up paths - robust path resolution
def get_project_root():
    """Find the project root directory (hansard-nlp-explorer) regardless of where script is run from."""
    current = Path(__file__).resolve()
    
    # Look for project root by finding the directory containing 'data-hansard'
    while current.parent != current:  # Not at filesystem root
        if (current / 'data-hansard').exists():
            return current
        current = current.parent
    
    # Fallback: assume we're in src/hansard/analysis/
    return Path(__file__).resolve().parents[3]

PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / 'data-hansard'
ANALYSIS_DIR = PROJECT_ROOT / 'analysis'
GENDER_ENHANCED_DATA = DATA_DIR / 'gender_analysis_enhanced'
PROCESSED_FIXED = DATA_DIR / 'processed_fixed'

# Set up matplotlib for publication quality
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def load_base_dataset(year_range: Optional[Tuple[int, int]] = None,
                        sample_size: Optional[int] = None,
                        load_text: bool = False) -> Dict:
    """
    Load base dataset from processed_fixed directory for overall analysis.
    
    This function efficiently loads Hansard debate data from the processed_fixed directory,
    which contains the raw pre-processed debate text and metadata. It's designed for
    analyzing overall corpus statistics before using the gender-enhanced dataset.
    
    Args:
        year_range: Tuple of (start_year, end_year) to filter data. If None, loads all years.
        sample_size: Number of debates to sample (for testing). If None, loads all debates.
        load_text: Whether to load full text content. If False, only loads metadata (faster).
    
    Returns:
        Dictionary containing:
        - 'metadata': DataFrame with debate metadata (title, speakers, word_count, etc.)
        - 'text_data': List of full text content (only if load_text=True)
        - 'statistics': Basic statistics about the dataset
        - 'speakers': Speaker information DataFrame
    
    Example:
        # Quick metadata analysis
        data = load_base_dataset(year_range=(1950, 1960), sample_size=1000)
        print(f"Total debates: {data['statistics']['total_debates']}")
        
        # Full text analysis
        data = load_base_dataset(year_range=(1950, 1955), load_text=True)
        text_analysis = analyze_text_content(data['text_data'])
        
    Performance Notes:
        - Metadata-only loading: Very fast, suitable for large year ranges
        - Text loading: Slower, recommended for smaller samples or specific years
        - Memory usage: ~1MB per 1000 debates (metadata), ~10MB per 1000 debates (with text)
    """
    print("Loading base dataset from processed_fixed...")
    
    # Load metadata from parquet files
    metadata_files = []
    if year_range:
        start_year, end_year = year_range
        for year in range(start_year, end_year + 1):
            year_file = PROCESSED_FIXED / 'metadata' / f'debates_{year}.parquet'
            if year_file.exists():
                metadata_files.append(year_file)
    else:
        # Load all available years
        metadata_files = list((PROCESSED_FIXED / 'metadata').glob('debates_*.parquet'))
        metadata_files = [f for f in metadata_files if not f.name.startswith('debates_master')]
    
    if not metadata_files:
        raise FileNotFoundError("No metadata files found in processed_fixed/metadata/")
    
    print(f"Loading {len(metadata_files)} metadata files...")
    
    # Load and combine metadata
    metadata_dfs = []
    for file_path in metadata_files:
        try:
            df = pd.read_parquet(file_path)
            metadata_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            continue
    
    if not metadata_dfs:
        raise ValueError("No metadata could be loaded")
    
    metadata_df = pd.concat(metadata_dfs, ignore_index=True)
    
    # Apply year filtering if specified
    if year_range:
        start_year, end_year = year_range
        metadata_df = metadata_df[(metadata_df['year'] >= start_year) & 
                                 (metadata_df['year'] <= end_year)]
    
    # Apply sampling if specified
    if sample_size and sample_size < len(metadata_df):
        metadata_df = metadata_df.sample(n=sample_size, random_state=42)
    
    print(f"Loaded {len(metadata_df)} debates")
    
    # Load speaker data
    speakers_data = load_speakers_data(year_range)
    
    # Calculate basic statistics
    stats = calculate_basic_statistics(metadata_df, speakers_data)
    
    result = {
        'metadata': metadata_df,
        'speakers': speakers_data,
        'statistics': stats
    }
    
    # Load full text if requested
    if load_text:
        print("Loading full text content...")
        text_data = load_text_content(metadata_df)
        result['text_data'] = text_data
    
    return result


def load_speakers_data(year_range: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
    """Load speaker data from parquet files."""
    speaker_files = []
    if year_range:
        start_year, end_year = year_range
        for year in range(start_year, end_year + 1):
            speaker_file = PROCESSED_FIXED / 'metadata' / f'speakers_{year}.parquet'
            if speaker_file.exists():
                speaker_files.append(speaker_file)
    else:
        speaker_files = list((PROCESSED_FIXED / 'metadata').glob('speakers_*.parquet'))
        speaker_files = [f for f in speaker_files if not f.name.startswith('speakers_master')]
    
    if not speaker_files:
        return pd.DataFrame()
    
    speaker_dfs = []
    for file_path in speaker_files:
        try:
            df = pd.read_parquet(file_path)
            speaker_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load speakers from {file_path}: {e}")
            continue
    
    if speaker_dfs:
        return pd.concat(speaker_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def load_text_content(metadata_df: pd.DataFrame) -> List[Dict]:
    """Load full text content for debates using unified data loader."""
    print(f"Loading text content for {len(metadata_df)} debates...")
    
    # Use unified data loader for efficient loading
    loader = UnifiedDataLoader()
    
    # Group by year for efficient loading
    year_groups = metadata_df.groupby('year')
    all_text_data = []
    
    for year, year_df in year_groups:
        print(f"  Loading {len(year_df)} debates for year {year}...")
        
        # Load year's text data with caching
        year_texts = loader._load_year_texts(year, use_cache=True)
        
        # Match debates to their text content
        for _, row in year_df.iterrows():
            file_path = row['file_path']
            if file_path in year_texts:
                all_text_data.append({
                    'file_path': file_path,
                    'year': year,
                    'full_text': year_texts[file_path].get('full_text', ''),
                    'title': year_texts[file_path].get('title', ''),
                    'speakers': year_texts[file_path].get('speakers', [])
                })
    
    return all_text_data


def calculate_basic_statistics(metadata_df: pd.DataFrame, speakers_df: pd.DataFrame) -> Dict:
    """Calculate basic statistics about the dataset."""
    stats = {
        'total_debates': len(metadata_df),
        'year_range': (metadata_df['year'].min(), metadata_df['year'].max()),
        'chamber_distribution': metadata_df['chamber'].value_counts().to_dict(),
        'total_words': metadata_df['word_count'].sum(),
        'total_characters': metadata_df['char_count'].sum(),
        'avg_words_per_debate': metadata_df['word_count'].mean(),
        'avg_speakers_per_debate': metadata_df['speaker_count'].mean(),
        'debates_with_speakers': len(metadata_df[metadata_df['speaker_count'] > 0]),
        'debates_without_speakers': len(metadata_df[metadata_df['speaker_count'] == 0]),
    }
    
    # Speaker statistics
    if not speakers_df.empty:
        stats['total_speaker_mentions'] = len(speakers_df)
        stats['unique_speakers'] = speakers_df['speaker_name'].nunique()
        stats['top_speakers'] = speakers_df['speaker_name'].value_counts().head(10).to_dict()
    
    # Yearly breakdown
    yearly_stats = metadata_df.groupby('year').agg({
        'word_count': ['sum', 'mean', 'count'],
        'speaker_count': ['sum', 'mean'],
        'char_count': 'sum'
    }).round(2)
    
    stats['yearly_breakdown'] = yearly_stats.to_dict()
    
    return stats


def analyze_text_content(text_data: List[Dict], filtering_level: str = 'moderate') -> Dict:
    """Analyze text content for word counts, bigrams, etc."""
    if not text_data:
        return {}
    
    all_text = ' '.join([item['full_text'] for item in text_data])
    
    # Basic text statistics
    words = all_text.lower().split()
    word_counts = Counter(words)
    
    # Comprehensive stop words with filtering levels
    def get_stop_words(level='moderate'):
        """Get stop words based on filtering level"""
        basic_stop_words = {
            # Basic English stop words
            'the', 'of', 'to', 'and', 'a', 'an', 'in', 'is', 'it', 'that', 'was', 'for', 
            'on', 'are', 'as', 'with', 'his', 'they', 'at', 'be', 'this', 'have', 'from', 
            'or', 'had', 'by', 'not', 'but', 'what', 'all', 'were', 'we', 'when', 'there', 
            'can', 'said', 'which', 'do', 'their', 'if', 'will', 'up', 'other', 'about', 
            'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 
            'like', 'into', 'him', 'has', 'two', 'more', 'very', 'been', 'am', 'could', 
            'might', 'shall', 'must', 'ought',         }
        
        parliamentary_terms = {
            # Parliamentary terms (appear in >80% of speeches)
            'hon', 'honourable', 'right', 'gentleman', 'lady', 'member', 'members',
            'house', 'speaker', 'sir', 'madam', 'mr', 'mrs', 'ms', 'lord', 'lords', 
            'gallant', 'learned', 'friend', 'friends', 'noble', 'bill', 'clause', 
            'amendment', 'committee', 'order', 'question', 'division', 'reading', 
            'report', 'stage', 'moved', 'second', 'third', 'government', 'minister', 
            'secretary', 'state', 'department', 'debate', 'discuss', 'speech', 'words', 
            'statement'
        }
        
        common_verbs = {
            # Common verbs (>70% frequency)
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
            # Vague/generic words (>65% frequency)
            'thing', 'things', 'something', 'anything', 'nothing', 'everything', 'way', 
            'ways', 'case', 'cases', 'matter', 'matters', 'fact', 'facts', 'time', 
            'times', 'place', 'places', 'part', 'parts', 'kind', 'kinds', 'point', 
            'points', 'view', 'views', 'position', 'number', 'numbers', 'example', 
            'examples', 'word', 'words', 'subject', 'course', 'present', 'regard', 
            'deal', 'side', 'end', 'result', 'means', 'moment', 'today', 'yesterday', 
            'tomorrow', 'now', 'then', 'year', 'years', 'day', 'days', 'week', 'weeks', 
            'month', 'months', 'last', 'next', 'first', 'second'
        }
        
        # Return stop words based on level
        if level == 'minimal':
            return basic_stop_words
        elif level == 'basic':
            return basic_stop_words
        elif level == 'parliamentary':
            return basic_stop_words | parliamentary_terms
        elif level == 'moderate':
            return basic_stop_words | parliamentary_terms | common_verbs | vague_words
        elif level == 'aggressive':
            # Add more aggressive filtering
            discourse_markers = {'well', 'yes', 'no', 'indeed', 'perhaps', 'certainly', 
                               'obviously', 'clearly', 'surely', 'really', 'quite', 'sure', 
                               'also', 'however', 'therefore', 'whether', 'still', 'always', 
                               'never', 'yet', 'already', 'rather', 'even', 'far'}
            return basic_stop_words | parliamentary_terms | common_verbs | vague_words | discourse_markers
        else:
            return basic_stop_words | parliamentary_terms | common_verbs | vague_words
    
    # Use moderate filtering by default
    stop_words = get_stop_words(filtering_level)
    
    filtered_words = {word: count for word, count in word_counts.items() 
                     if word not in stop_words and len(word) > 2}
    
    # Generate bigrams
    bigrams = []
    for item in text_data:
        words = item['full_text'].lower().split()
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            bigrams.append(bigram)
    
    bigram_counts = Counter(bigrams)
    
    return {
        'total_words': len(words),
        'unique_words': len(word_counts),
        'top_words': dict(Counter(filtered_words).most_common(20)),
        'top_bigrams': dict(bigram_counts.most_common(20)),
        'avg_words_per_debate': len(words) / len(text_data)
    }


def create_basic_visualizations(data: Dict, output_dir: Path = None):
    """Create basic visualizations for the dataset."""
    if output_dir is None:
        output_dir = ANALYSIS_DIR
    
    output_dir.mkdir(exist_ok=True)
    
    metadata = data['metadata']
    stats = data['statistics']
    
    # 1. Debates per year
    plt.figure(figsize=(12, 6))
    yearly_counts = metadata.groupby('year').size()
    plt.plot(yearly_counts.index, yearly_counts.values, linewidth=2)
    plt.title('Number of Debates per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Debates')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'debates_per_year.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Word count distribution
    plt.figure(figsize=(10, 6))
    plt.hist(metadata['word_count'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Word Counts per Debate')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'word_count_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Chamber distribution
    plt.figure(figsize=(8, 6))
    chamber_counts = metadata['chamber'].value_counts()
    plt.pie(chamber_counts.values, labels=chamber_counts.index, autopct='%1.1f%%')
    plt.title('Distribution by Chamber')
    plt.tight_layout()
    plt.savefig(output_dir / 'chamber_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def load_debates_unified(year_range: Optional[Tuple[int, int]] = None,
                        sample_size: Optional[int] = None,
                        load_text: bool = True,
                        source: str = 'processed_fixed') -> Dict:
    """
    Load debates using the unified data loader (recommended approach).
    
    Args:
        year_range: Tuple of (start_year, end_year) to filter data
        sample_size: Number of debates to sample
        load_text: Whether to load full text content
        source: Data source ('processed_fixed', 'gender_enhanced', 'derived_speeches')
    
    Returns:
        Dict with loaded data
    """
    loader = UnifiedDataLoader()
    return loader.load_debates(
        source=source,
        year_range=year_range,
        sample_size=sample_size,
        load_text=load_text,
        use_cache=True
    )


# Test the function
if __name__ == "__main__":
    # Test with a small sample
    print("Testing load_base_dataset function...")
    data = load_base_dataset(year_range=(1950, 1980), sample_size=10000, load_text=True)
    
    print(f"\nDataset Statistics:")
    print(f"Total debates: {data['statistics']['total_debates']}")
    print(f"Year range: {data['statistics']['year_range']}")
    print(f"Total words: {data['statistics']['total_words']:,}")
    print(f"Average words per debate: {data['statistics']['avg_words_per_debate']:.1f}")
    print(f"Chamber distribution: {data['statistics']['chamber_distribution']}")
    
    if 'text_data' in data and data['text_data']:
        print(f"\nText Analysis:")
        text_analysis = analyze_text_content(data['text_data'], filtering_level='aggressive')
        print(f"Total words in sample: {text_analysis.get('total_words', 0):,}")
        print(f"Unique words: {text_analysis.get('unique_words', 0):,}")
        print(f"Top 10 words: {list(text_analysis.get('top_words', {}).items())[:10]}")
    
    # Create visualizations
    create_basic_visualizations(data)
    print("\nBasic analysis complete!")