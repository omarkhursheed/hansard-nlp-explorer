#!/usr/bin/env python3
"""
Fast search and query interface for processed Hansard data.

Provides high-level API for searching debates, speakers, and topics
across the entire parliamentary archive with full provenance tracking.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import polars as pl
import pandas as pd

class HansardSearch:
    """High-performance search interface for Hansard parliamentary data."""
    
    def __init__(self, processed_data_path: str):
        self.processed_data_path = Path(processed_data_path)
        self.db_path = self.processed_data_path / 'index' / 'debates.db'
        self.metadata_path = self.processed_data_path / 'metadata'
        self.content_path = self.processed_data_path / 'content'
        
        # Load manifest for data provenance
        manifest_path = self.processed_data_path / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}
    
    def search_debates(self, 
                      query: str,
                      chamber: Optional[str] = None,
                      year_range: Optional[Tuple[int, int]] = None,
                      speakers: Optional[List[str]] = None,
                      topics: Optional[List[str]] = None,
                      limit: int = 100) -> pd.DataFrame:
        """
        Search debates using full-text search with optional filters.
        
        Args:
            query: Full-text search query
            chamber: Filter by 'Commons' or 'Lords'
            year_range: Tuple of (start_year, end_year)
            speakers: List of speaker names to filter by
            topics: List of debate topics to filter by
            limit: Maximum number of results
            
        Returns:
            DataFrame with matching debates and metadata
        """
        if not self.db_path.exists():
            raise FileNotFoundError("Search index not found. Run data pipeline first.")
        
        conn = sqlite3.connect(self.db_path)
        
        # Build query
        where_clauses = []
        params = []
        
        # Full-text search
        if query.strip():
            where_clauses.append("debates_fts MATCH ?")
            params.append(query)
        
        # Chamber filter
        if chamber:
            where_clauses.append("chamber = ?")
            params.append(chamber)
        
        # Year range filter
        if year_range:
            where_clauses.append("year BETWEEN ? AND ?")
            params.extend(year_range)
        
        # Speaker filter
        if speakers:
            speaker_conditions = []
            for speaker in speakers:
                speaker_conditions.append("speakers LIKE ?")
                params.append(f'%"{speaker}"%')
            where_clauses.append(f"({' OR '.join(speaker_conditions)})")
        
        # Topic filter
        if topics:
            topic_conditions = []
            for topic in topics:
                topic_conditions.append("debate_topics LIKE ?")
                params.append(f'%"{topic}"%')
            where_clauses.append(f"({' OR '.join(topic_conditions)})")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        sql = f"""
            SELECT d.*, bm25(debates_fts) as relevance_score
            FROM debates d
            JOIN debates_fts ON d.rowid = debates_fts.rowid
            WHERE {where_clause}
            ORDER BY relevance_score
            LIMIT ?
        """
        params.append(limit)
        
        result_df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        
        # Parse JSON fields
        if not result_df.empty:
            result_df['speakers'] = result_df['speakers'].apply(
                lambda x: json.loads(x) if x else []
            )
            result_df['debate_topics'] = result_df['debate_topics'].apply(
                lambda x: json.loads(x) if x else []
            )
        
        return result_df
    
    def search_speakers(self, 
                       speaker_name: str,
                       chamber: Optional[str] = None,
                       year_range: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
        """Search for all debates by a specific speaker."""
        master_speakers_path = self.metadata_path / 'speakers_master.parquet'
        
        if not master_speakers_path.exists():
            raise FileNotFoundError("Speaker metadata not found. Run data pipeline first.")
        
        # Load speakers data
        speakers_df = pl.read_parquet(master_speakers_path)
        
        # Filter by speaker name (fuzzy matching)
        filtered_df = speakers_df.filter(
            pl.col('speaker_name').str.contains(f'(?i){speaker_name}')
        )
        
        # Apply additional filters
        if chamber:
            filtered_df = filtered_df.filter(pl.col('chamber') == chamber)
        
        if year_range:
            filtered_df = filtered_df.filter(
                pl.col('year').is_between(year_range[0], year_range[1])
            )
        
        return filtered_df.to_pandas()
    
    def get_debate_content(self, file_path: str) -> Dict:
        """Retrieve full content for a specific debate."""
        # Extract year from file path
        path_parts = Path(file_path).parts
        year = None
        for part in path_parts:
            if part.isdigit() and 1800 <= int(part) <= 2010:
                year = part
                break
        
        if not year:
            raise ValueError(f"Could not extract year from file path: {file_path}")
        
        content_file = self.content_path / year / f'debates_{year}.jsonl'
        
        if not content_file.exists():
            raise FileNotFoundError(f"Content file not found: {content_file}")
        
        # Search for the specific debate
        with open(content_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                if record['file_path'] == file_path:
                    return record
        
        raise ValueError(f"Debate not found: {file_path}")
    
    def get_timeline_stats(self, 
                          groupby: str = 'year',
                          chamber: Optional[str] = None) -> pd.DataFrame:
        """Get temporal statistics for debates."""
        master_debates_path = self.metadata_path / 'debates_master.parquet'
        
        if not master_debates_path.exists():
            raise FileNotFoundError("Debate metadata not found. Run data pipeline first.")
        
        debates_df = pl.read_parquet(master_debates_path)
        
        if chamber:
            debates_df = debates_df.filter(pl.col('chamber') == chamber)
        
        if groupby == 'year':
            stats_df = debates_df.group_by('year').agg([
                pl.count('file_path').alias('debate_count'),
                pl.sum('word_count').alias('total_words'),
                pl.mean('word_count').alias('avg_words_per_debate'),
                pl.sum('line_count').alias('total_lines'),
                pl.col('chamber').n_unique().alias('chambers_active')
            ]).sort('year')
        elif groupby == 'month':
            stats_df = debates_df.group_by(['year', 'month']).agg([
                pl.count('file_path').alias('debate_count'),
                pl.sum('word_count').alias('total_words'),
                pl.mean('word_count').alias('avg_words_per_debate')
            ]).sort(['year', 'month'])
        else:
            raise ValueError("groupby must be 'year' or 'month'")
        
        return stats_df.to_pandas()
    
    def get_top_speakers(self, 
                        year_range: Optional[Tuple[int, int]] = None,
                        chamber: Optional[str] = None,
                        limit: int = 20) -> pd.DataFrame:
        """Get most active speakers by debate count."""
        master_speakers_path = self.metadata_path / 'speakers_master.parquet'
        
        if not master_speakers_path.exists():
            raise FileNotFoundError("Speaker metadata not found. Run data pipeline first.")
        
        speakers_df = pl.read_parquet(master_speakers_path)
        
        # Apply filters
        if year_range:
            speakers_df = speakers_df.filter(
                pl.col('year').is_between(year_range[0], year_range[1])
            )
        
        if chamber:
            speakers_df = speakers_df.filter(pl.col('chamber') == chamber)
        
        # Group by speaker and count debates
        top_speakers = speakers_df.group_by('speaker_name').agg([
            pl.count('file_path').alias('debate_count'),
            pl.col('year').min().alias('first_appearance'),
            pl.col('year').max().alias('last_appearance'),
            pl.col('chamber').n_unique().alias('chambers_spoken_in')
        ]).sort('debate_count', descending=True).limit(limit)
        
        return top_speakers.to_pandas()
    
    def get_data_quality_report(self) -> Dict:
        """Generate comprehensive data quality and coverage report."""
        report = {
            'generation_time': datetime.now().isoformat(),
            'data_provenance': self.manifest,
            'coverage': {},
            'quality_metrics': {}
        }
        
        # Coverage statistics
        master_debates_path = self.metadata_path / 'debates_master.parquet'
        if master_debates_path.exists():
            debates_df = pl.read_parquet(master_debates_path)
            
            report['coverage'] = {
                'total_debates': len(debates_df),
                'year_range': [
                    int(debates_df['year'].min()),
                    int(debates_df['year'].max())
                ],
                'chambers': debates_df['chamber'].unique().to_list(),
                'total_words': int(debates_df['word_count'].sum()),
                'total_lines': int(debates_df['line_count'].sum())
            }
            
            # Quality metrics
            successful_parsing = debates_df.filter(pl.col('success') == True)
            report['quality_metrics'] = {
                'parsing_success_rate': len(successful_parsing) / len(debates_df),
                'files_with_speakers': len(debates_df.filter(pl.col('speaker_count') > 0)) / len(debates_df),
                'files_with_topics': len(debates_df.filter(pl.col('debate_topics').list.len() > 0)) / len(debates_df),
                'avg_words_per_debate': float(debates_df['word_count'].mean()),
                'avg_speakers_per_debate': float(debates_df['speaker_count'].mean())
            }
        
        return report
    
    def export_search_results(self, 
                             search_results: pd.DataFrame,
                             output_path: str,
                             format: str = 'csv',
                             include_content: bool = False) -> str:
        """Export search results to various formats."""
        output_path = Path(output_path)
        
        if include_content:
            # Add full content to results
            content_list = []
            for _, row in search_results.iterrows():
                try:
                    content = self.get_debate_content(row['file_path'])
                    content_list.append(content.get('full_text', ''))
                except:
                    content_list.append('')
            
            search_results = search_results.copy()
            search_results['full_content'] = content_list
        
        if format.lower() == 'csv':
            search_results.to_csv(output_path.with_suffix('.csv'), index=False)
        elif format.lower() == 'json':
            search_results.to_json(output_path.with_suffix('.json'), orient='records', indent=2)
        elif format.lower() == 'parquet':
            search_results.to_parquet(output_path.with_suffix('.parquet'))
        else:
            raise ValueError("Format must be 'csv', 'json', or 'parquet'")
        
        return str(output_path.with_suffix(f'.{format.lower()}'))

def main():
    """Example usage of the search interface."""
    processed_data_path = "../data/processed"
    
    try:
        search = HansardSearch(processed_data_path)
        
        # Example searches
        print("=== Hansard Search Interface Demo ===\n")
        
        # Search for debates about taxation
        print("1. Searching for debates about 'tax'...")
        tax_debates = search.search_debates("tax", limit=5)
        if not tax_debates.empty:
            print(f"Found {len(tax_debates)} debates about taxation")
            print(tax_debates[['file_name', 'title', 'chamber', 'year', 'word_count']].to_string())
        
        # Get top speakers
        print("\n2. Top 10 most active speakers...")
        top_speakers = search.get_top_speakers(limit=10)
        if not top_speakers.empty:
            print(top_speakers.to_string())
        
        # Timeline statistics
        print("\n3. Debates by decade...")
        timeline = search.get_timeline_stats()
        if not timeline.empty:
            print(timeline.head(10).to_string())
        
        # Data quality report
        print("\n4. Data quality report...")
        quality_report = search.get_data_quality_report()
        print(json.dumps(quality_report, indent=2))
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the data pipeline first to create the processed dataset.")

if __name__ == "__main__":
    main()