#!/usr/bin/env python3
"""
Unified Data Loader for Hansard Parliamentary Debates

Consolidates all data loading approaches into a single, efficient interface.
Replaces multiple JSONL readers across different scripts.

Usage:
    from hansard.utils.unified_data_loader import UnifiedDataLoader
    
    loader = UnifiedDataLoader()
    
    # Load from processed_fixed (overall corpus)
    data = loader.load_debates(source='processed_fixed', year_range=(1900, 1930))
    
    # Load from gender_analysis_enhanced (gender-tagged)
    data = loader.load_debates(source='gender_enhanced', year_range=(1900, 1930))
    
    # Load from derived speeches (flat speech data)
    data = loader.load_debates(source='derived_speeches', year_range=(1900, 1930))
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
from collections import defaultdict
import pickle
import hashlib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from path_config import Paths


class UnifiedDataLoader:
    """
    Single source of truth for loading Hansard debate data.
    
    Consolidates approaches from:
    - basic_analysis.py load_text_content()
    - create_enhanced_gender_dataset.py load_year_texts()
    - analysis.ipynb load_overall_dataset()
    """
    
    def __init__(self, data_dir: Path = None, cache_dir: Path = None):
        """
        Initialize unified data loader.
        
        Args:
            data_dir: Path to processed_fixed directory
            cache_dir: Path for caching loaded data
        """
        if data_dir is None:
            data_dir = Paths.PROCESSED_FIXED
        
        self.data_dir = Path(data_dir)
        self.content_dir = self.data_dir / 'content'
        self.metadata_dir = self.data_dir / 'metadata'
        self.gender_enhanced_dir = Paths.GENDER_ENHANCED_DATA
        self.derived_speeches_dir = Paths.DATA_DIR / 'derived' / 'gender_speeches'
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = self.data_dir / 'cache'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory cache for frequently accessed data
        self._year_cache = {}
        self._metadata_cache = None
    
    def load_debates(self, 
                    source: str = 'processed_fixed',
                    year_range: Optional[Tuple[int, int]] = None,
                    sample_size: Optional[int] = None,
                    load_text: bool = True,
                    use_cache: bool = True,
                    gender_filter: Optional[str] = None) -> Dict:
        """
        Load debates with unified interface.
        
        Args:
            source: Data source ('processed_fixed', 'gender_enhanced', 'derived_speeches')
            year_range: Tuple of (start_year, end_year) or None for all years
            sample_size: Number of debates to sample (stratified by year)
            load_text: Whether to load full text content
            use_cache: Whether to use cached data
            gender_filter: Filter by gender ('male', 'female') for gender sources
        
        Returns:
            Dict with 'metadata', 'text_data', 'statistics'
        """
        print(f"Loading debates: source={source}, years={year_range}, sample={sample_size}, text={load_text}")
        
        if source == 'processed_fixed':
            return self._load_from_processed_fixed(year_range, sample_size, load_text, use_cache)
        elif source == 'gender_enhanced':
            return self._load_from_gender_enhanced(year_range, sample_size, load_text, use_cache, gender_filter)
        elif source == 'derived_speeches':
            return self._load_from_derived_speeches(year_range, sample_size, load_text, use_cache, gender_filter)
        else:
            raise ValueError(f"Unknown source: {source}. Use 'processed_fixed', 'gender_enhanced', or 'derived_speeches'")
    
    def _load_from_processed_fixed(self, year_range, sample_size, load_text, use_cache):
        """Load from processed_fixed directory (overall corpus)."""
        # Load metadata first
        metadata_df = self._load_metadata(year_range, sample_size, use_cache)
        
        result = {
            'metadata': metadata_df,
            'statistics': self._calculate_statistics(metadata_df)
        }
        
        # Load text content if requested
        if load_text:
            text_data = self._load_all_debates(metadata_df, use_cache)
            result['text_data'] = text_data
        
        return result
    
    def _load_from_gender_enhanced(self, year_range, sample_size, load_text, use_cache, gender_filter):
        """Load from gender_analysis_enhanced directory."""
        # Get available years
        if year_range:
            start_year, end_year = year_range
            years = list(range(start_year, end_year + 1))
        else:
            year_files = list(self.gender_enhanced_dir.glob('debates_*_enhanced.parquet'))
            years = sorted([int(f.stem.split('_')[1]) for f in year_files])
        
        all_debates = []
        for year in years:
            year_file = self.gender_enhanced_dir / f'debates_{year}_enhanced.parquet'
            if year_file.exists():
                df = pd.read_parquet(year_file)
                all_debates.append(df)
        
        if not all_debates:
            return {'metadata': pd.DataFrame(), 'statistics': {}, 'text_data': []}
        
        combined_df = pd.concat(all_debates, ignore_index=True)
        
        # Apply gender filtering if requested
        if gender_filter:
            if gender_filter == 'male':
                combined_df = combined_df[combined_df['has_male'] == True]
            elif gender_filter == 'female':
                combined_df = combined_df[combined_df['has_female'] == True]
        
        # Apply sampling if requested
        if sample_size and sample_size < len(combined_df):
            combined_df = self._apply_stratified_sampling(combined_df, sample_size)
        
        result = {
            'metadata': combined_df,
            'statistics': self._calculate_statistics(combined_df)
        }
        
        if load_text:
            # Extract text data from gender enhanced format
            text_data = []
            for _, row in combined_df.iterrows():
                text_data.append({
                    'file_path': row.get('file_path', ''),
                    'year': row.get('year', 0),
                    'full_text': row.get('debate_text', ''),
                    'title': row.get('title', ''),
                    'speakers': row.get('speaker_details', [])
                })
            result['text_data'] = text_data
        
        return result
    
    def _load_from_derived_speeches(self, year_range, sample_size, load_text, use_cache, gender_filter):
        """Load from derived/gender_speeches directory (flat speech data)."""
        # Get available years
        if year_range:
            start_year, end_year = year_range
            years = list(range(start_year, end_year + 1))
        else:
            year_files = list(self.derived_speeches_dir.glob('speeches_*.parquet'))
            years = sorted([int(f.stem.split('_')[1]) for f in year_files])
        
        all_speeches = []
        for year in years:
            year_file = self.derived_speeches_dir / f'speeches_{year}.parquet'
            if year_file.exists():
                df = pd.read_parquet(year_file)
                all_speeches.append(df)
        
        if not all_speeches:
            return {'metadata': pd.DataFrame(), 'statistics': {}, 'text_data': []}
        
        combined_df = pd.concat(all_speeches, ignore_index=True)
        
        # Apply gender filtering if requested
        if gender_filter:
            combined_df = combined_df[combined_df['gender'] == gender_filter[0]]  # 'm' or 'f'
        
        # Apply sampling if requested
        if sample_size and sample_size < len(combined_df):
            combined_df = self._apply_stratified_sampling(combined_df, sample_size)
        
        result = {
            'metadata': combined_df,
            'statistics': self._calculate_statistics(combined_df)
        }
        
        if load_text:
            # Convert speeches to text data format
            text_data = []
            for _, row in combined_df.iterrows():
                text_data.append({
                    'file_path': row.get('debate_id', ''),
                    'year': row.get('year', 0),
                    'full_text': row.get('text', ''),
                    'title': f"Speech by {row.get('speaker', 'Unknown')}",
                    'speakers': [row.get('speaker', 'Unknown')]
                })
            result['text_data'] = text_data
        
        return result
    
    def _load_metadata(self, year_range: Optional[Tuple[int, int]], 
                      sample_size: Optional[int], use_cache: bool) -> pd.DataFrame:
        """Load metadata from parquet files."""
        # Check cache first
        if use_cache and self._metadata_cache is not None:
            metadata_df = self._metadata_cache
        else:
            # Load from parquet files
            metadata_files = self._get_metadata_files(year_range)
            metadata_dfs = []
            
            for file_path in metadata_files:
                try:
                    df = pd.read_parquet(file_path)
                    metadata_dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")
                    continue
            
            if not metadata_dfs:
                raise ValueError("No metadata files could be loaded")
            
            metadata_df = pd.concat(metadata_dfs, ignore_index=True)
            
            # Apply year filtering
            if year_range:
                start_year, end_year = year_range
                metadata_df = metadata_df[
                    (metadata_df['year'] >= start_year) & 
                    (metadata_df['year'] <= end_year)
                ]
            
            # Cache for future use
            if use_cache:
                self._metadata_cache = metadata_df
        
        # Apply sampling if requested
        if sample_size and sample_size < len(metadata_df):
            metadata_df = self._apply_stratified_sampling(metadata_df, sample_size)
        
        return metadata_df
    
    def _load_all_debates(self, metadata_df: pd.DataFrame, use_cache: bool) -> List[Dict]:
        """Load all debates in metadata_df."""
        text_data = []
        
        # Group by year for efficient loading
        year_groups = metadata_df.groupby('year')
        
        for year, year_df in year_groups:
            print(f"  Loading {len(year_df)} debates for year {year}...")
            
            # Load year's text data
            year_texts = self._load_year_texts(year, use_cache)
            
            # Match debates to their text content
            for _, row in year_df.iterrows():
                file_path = row['file_path']
                if file_path in year_texts:
                    text_data.append({
                        'file_path': file_path,
                        'year': year,
                        'full_text': year_texts[file_path].get('full_text', ''),
                        'title': year_texts[file_path].get('title', ''),
                        'speakers': year_texts[file_path].get('speakers', [])
                    })
        
        return text_data
    
    def _load_year_texts(self, year: int, use_cache: bool) -> Dict[str, Dict]:
        """Load all text content for a specific year."""
        # Check in-memory cache first
        if year in self._year_cache:
            return self._year_cache[year]
        
        # Check disk cache
        cache_file = self.cache_dir / f'year_{year}_texts.pkl'
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    year_texts = pickle.load(f)
                self._year_cache[year] = year_texts
                return year_texts
            except Exception as e:
                print(f"Warning: Could not load cache for year {year}: {e}")
        
        # Load from JSONL file
        year_content_file = self.content_dir / str(year) / f'debates_{year}.jsonl'
        year_texts = {}
        
        if year_content_file.exists():
            try:
                with open(year_content_file, 'r') as f:
                    for line in f:
                        debate_data = json.loads(line)
                        file_path = debate_data.get('file_path', '')
                        year_texts[file_path] = {
                            'full_text': debate_data.get('full_text', ''),
                            'title': debate_data.get('metadata', {}).get('title', ''),
                            'speakers': debate_data.get('metadata', {}).get('speakers', []),
                            'lines': debate_data.get('lines', []),
                            'content_hash': debate_data.get('content_hash', ''),
                            'extraction_timestamp': debate_data.get('extraction_timestamp', '')
                        }
                
                # Cache the loaded data
                self._year_cache[year] = year_texts
                if use_cache:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(year_texts, f)
                
                print(f"    Loaded {len(year_texts)} debate texts for year {year}")
            except Exception as e:
                print(f"    Warning: Could not load texts for {year}: {e}")
        
        return year_texts
    
    def _get_metadata_files(self, year_range: Optional[Tuple[int, int]]) -> List[Path]:
        """Get list of metadata files to load."""
        if year_range:
            start_year, end_year = year_range
            metadata_files = []
            for year in range(start_year, end_year + 1):
                year_file = self.metadata_dir / f'debates_{year}.parquet'
                if year_file.exists():
                    metadata_files.append(year_file)
        else:
            metadata_files = list(self.metadata_dir.glob('debates_*.parquet'))
            metadata_files = [f for f in metadata_files if not f.name.startswith('debates_master')]
        
        return metadata_files
    
    def _apply_stratified_sampling(self, metadata_df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Apply stratified sampling to maintain year distribution."""
        # Sample proportionally from each year
        sampled_dfs = []
        for year, year_df in metadata_df.groupby('year'):
            year_sample_size = max(1, int(sample_size * len(year_df) / len(metadata_df)))
            if year_sample_size < len(year_df):
                year_sample = year_df.sample(n=year_sample_size, random_state=42)
            else:
                year_sample = year_df
            sampled_dfs.append(year_sample)
        
        return pd.concat(sampled_dfs, ignore_index=True)
    
    def _calculate_statistics(self, metadata_df: pd.DataFrame) -> Dict:
        """Calculate basic statistics about the dataset."""
        if metadata_df.empty:
            return {}
        
        stats = {
            'total_debates': len(metadata_df),
            'year_range': (metadata_df['year'].min(), metadata_df['year'].max()),
            'total_words': metadata_df['word_count'].sum() if 'word_count' in metadata_df.columns else 0,
            'avg_words_per_debate': metadata_df['word_count'].mean() if 'word_count' in metadata_df.columns else 0
        }
        
        # Add chamber distribution if available
        if 'chamber' in metadata_df.columns:
            stats['chamber_distribution'] = metadata_df['chamber'].value_counts().to_dict()
        
        # Add speaker statistics if available
        if 'speaker_count' in metadata_df.columns:
            stats['debates_with_speakers'] = len(metadata_df[metadata_df['speaker_count'] > 0])
            stats['avg_speakers_per_debate'] = metadata_df['speaker_count'].mean()
        
        return stats
    
    def clear_cache(self):
        """Clear all cached data."""
        self._year_cache.clear()
        self._metadata_cache = None
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob('*.pkl'):
            cache_file.unlink()
        
        print("Cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data."""
        return {
            'memory_cache_years': list(self._year_cache.keys()),
            'disk_cache_files': [f.name for f in self.cache_dir.glob('*.pkl')],
            'cache_dir': str(self.cache_dir)
        }


# Convenience function for quick loading
def load_hansard_data(source='processed_fixed', year_range=None, sample_size=None, load_text=True):
    """
    Quick convenience function for loading Hansard data.
    
    Args:
        source: Data source ('processed_fixed', 'gender_enhanced', 'derived_speeches')
        year_range: Tuple of (start_year, end_year) or None for all years
        sample_size: Number of debates to sample
        load_text: Whether to load full text content
    
    Returns:
        Dict with loaded data
    """
    loader = UnifiedDataLoader()
    return loader.load_debates(
        source=source,
        year_range=year_range,
        sample_size=sample_size,
        load_text=load_text
    )


if __name__ == "__main__":
    # Test the unified data loader
    print("Testing UnifiedDataLoader...")
    
    loader = UnifiedDataLoader()
    
    # Test loading from processed_fixed
    print("\n1. Testing processed_fixed loading...")
    data = loader.load_debates(
        source='processed_fixed',
        year_range=(1950, 1952),
        sample_size=10,
        load_text=True
    )
    print(f"   Loaded {len(data['metadata'])} debates")
    print(f"   Statistics: {data['statistics']}")
    
    # Test cache info
    print("\n2. Cache info:")
    print(loader.get_cache_info())
    
    print("\nUnifiedDataLoader test complete!")
