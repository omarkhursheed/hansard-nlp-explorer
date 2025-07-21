#!/usr/bin/env python3
"""
Comprehensive data extraction and storage pipeline for Hansard debates.

Storage Strategy:
- Parquet: Structured metadata (fast queries, compression, schema evolution)
- JSON Lines: Full text content (flexible, preserves structure) 
- SQLite: Search index (full-text search, fast lookups)
- Manifest: Provenance and validation (integrity, lineage)

Directory Structure:
data/processed/
├── metadata/           # Parquet files with structured metadata
│   ├── debates.parquet           # Main debate metadata
│   └── speakers.parquet          # Speaker information
├── content/           # JSON Lines with full text
│   ├── 1803/
│   │   └── debates_1803.jsonl   # Full debate content by year
├── index/             # Search indices
│   ├── debates.db              # SQLite FTS index
│   └── speakers.db             # Speaker index
└── manifest.json     # Data provenance and validation
"""

import gzip
import glob
import os
import json
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import polars as pl
from bs4 import BeautifulSoup
import re

class HansardDataPipeline:
    """Comprehensive data extraction and storage pipeline for Hansard debates."""
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for processed data."""
        dirs = ['metadata', 'content', 'index']
        for dir_name in dirs:
            (self.processed_data_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    def extract_comprehensive_metadata(self, file_path: Path) -> Dict:
        """Extract all possible metadata from a Hansard file."""
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                html = f.read()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # File metadata
            file_stats = file_path.stat()
            file_hash = hashlib.sha256(html.encode('utf-8')).hexdigest()
            
            metadata = {
                # File provenance
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': file_stats.st_size,
                'file_modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'content_hash': file_hash,
                'extraction_timestamp': datetime.now().isoformat(),
                
                # Basic HTML metadata
                'title': soup.find('title').get_text() if soup.find('title') else None,
                'success': False,
                'error': None
            }
            
            # Meta tags
            meta_tags = {}
            for meta in soup.find_all('meta'):
                name = meta.get('name') or meta.get('property')
                content = meta.get('content')
                if name and content:
                    meta_tags[name] = content
            metadata['meta_tags'] = meta_tags
            
            # Try both Commons and Lords sitting divs
            content_div = soup.find('div', class_='house-of-commons-sitting')
            sitting_type = "Commons"
            if not content_div:
                content_div = soup.find('div', class_='house-of-lords-sitting')
                sitting_type = "Lords"
            
            if not content_div:
                metadata['error'] = 'No main content div found'
                return metadata
            
            metadata['chamber'] = sitting_type
            
            # Extract content
            for unwanted in content_div(['nav', 'footer', 'script', 'style']):
                unwanted.decompose()
            
            content_text = content_div.get_text(separator='\n', strip=True)
            lines = [line for line in content_text.split('\n') if line.strip()]
            
            metadata['line_count'] = len(lines)
            metadata['word_count'] = sum(len(line.split()) for line in lines)
            metadata['char_count'] = sum(len(line) for line in lines)
            
            # Extract Hansard reference
            hansard_ref = self._extract_hansard_reference(lines)
            metadata.update(hansard_ref)
            
            # Extract speakers
            speakers = self._extract_speakers(lines)
            metadata['speakers'] = speakers
            metadata['speaker_count'] = len(speakers)
            
            # Extract debate topics
            debate_topics = self._extract_debate_topics(metadata['title'])
            metadata['debate_topics'] = debate_topics
            
            # Extract year/month from file path
            path_parts = file_path.parts
            if len(path_parts) >= 3:
                metadata['year'] = int(path_parts[-3]) if path_parts[-3].isdigit() else None
                metadata['month'] = path_parts[-2]
            
            # Content for separate storage
            metadata['content_lines'] = lines
            metadata['full_text'] = ' '.join(lines)
            
            metadata['success'] = True
            return metadata
            
        except Exception as e:
            metadata['error'] = str(e)
            return metadata
    
    def _extract_hansard_reference(self, lines: List[str]) -> Dict:
        """Extract Hansard reference information."""
        hansard_ref = {
            'hansard_reference': None,
            'reference_chamber': None,
            'reference_date': None,
            'reference_volume': None,
            'reference_columns': None
        }
        
        if not lines:
            return hansard_ref
            
        first_line = lines[0]
        # Look for pattern like "HC Deb 22 November 1803 vol 1 cc13-31"
        ref_match = re.search(r'(HC|HL) Deb (\d{1,2} \w+ \d{4}) vol (\d+) cc?(\d+(?:-\d+)?)', first_line)
        if ref_match:
            hansard_ref.update({
                'hansard_reference': ref_match.group(0),
                'reference_chamber': 'Commons' if ref_match.group(1) == 'HC' else 'Lords',
                'reference_date': ref_match.group(2),
                'reference_volume': int(ref_match.group(3)),
                'reference_columns': ref_match.group(4)
            })
        
        return hansard_ref
    
    def _extract_speakers(self, lines: List[str]) -> List[str]:
        """Extract speaker information from debate lines."""
        speakers = []
        for line in lines[1:20]:  # Check first 20 lines
            # Look for speaker patterns
            speaker_patterns = [
                r'^(Mr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'^(Mrs\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'^(The\s+[A-Z][a-z]+(?:\s+of\s+[A-Z][a-z]+)*)',
                r'^(Lord\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$'
            ]
            for pattern in speaker_patterns:
                match = re.match(pattern, line.strip())
                if match and len(match.group(1)) < 50:
                    speakers.append(match.group(1))
                    break
        
        return list(set(speakers))  # Remove duplicates
    
    def _extract_debate_topics(self, title: str) -> List[str]:
        """Extract debate topics from title."""
        if not title:
            return []
            
        debate_topics = []
        # Look for patterns like "BANK RESTRICTION BILL", "INCOME TAX", etc.
        topic_matches = re.findall(r'([A-Z][A-Z\s-]+)\.?—', title)
        for topic in topic_matches:
            clean_topic = topic.strip()
            if len(clean_topic) > 3 and clean_topic not in ['HANSARD']:
                debate_topics.append(clean_topic)
        
        return debate_topics
    
    def process_year(self, year: str) -> Dict:
        """Process all files for a given year."""
        year_path = self.raw_data_path / year
        if not year_path.exists():
            return {'error': f'Year {year} not found', 'processed': 0}
        
        print(f"Processing year {year}...")
        
        # Find all HTML files
        html_files = list(year_path.rglob("*.html.gz"))
        
        debates_data = []
        speakers_data = []
        content_data = []
        
        processed_count = 0
        error_count = 0
        
        for file_path in html_files:
            metadata = self.extract_comprehensive_metadata(file_path)
            
            if metadata['success']:
                # Prepare structured metadata (without content)
                debate_record = {k: v for k, v in metadata.items() 
                              if k not in ['content_lines', 'full_text']}
                debates_data.append(debate_record)
                
                # Prepare speaker records
                for speaker in metadata.get('speakers', []):
                    speakers_data.append({
                        'file_path': str(file_path),
                        'year': metadata.get('year'),
                        'month': metadata.get('month'),
                        'chamber': metadata.get('chamber'),
                        'speaker_name': speaker,
                        'reference_date': metadata.get('reference_date')
                    })
                
                # Prepare content record
                content_record = {
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'content_hash': metadata['content_hash'],
                    'extraction_timestamp': metadata['extraction_timestamp'],
                    'lines': metadata.get('content_lines', []),
                    'full_text': metadata.get('full_text', ''),
                    'metadata': {k: v for k, v in metadata.items() 
                               if k not in ['content_lines', 'full_text']}
                }
                content_data.append(content_record)
                
                processed_count += 1
            else:
                error_count += 1
                print(f"  Error processing {file_path.name}: {metadata.get('error')}")
        
        # Save structured data
        if debates_data:
            # Convert to Polars for efficient processing
            debates_df = pl.DataFrame(debates_data)
            debates_df.write_parquet(self.processed_data_path / 'metadata' / f'debates_{year}.parquet')
            
            if speakers_data:
                speakers_df = pl.DataFrame(speakers_data)
                speakers_df.write_parquet(self.processed_data_path / 'metadata' / f'speakers_{year}.parquet')
            
            # Save content as JSON Lines
            content_dir = self.processed_data_path / 'content' / year
            content_dir.mkdir(parents=True, exist_ok=True)
            
            with open(content_dir / f'debates_{year}.jsonl', 'w', encoding='utf-8') as f:
                for record in content_data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # Update search index
            self._update_search_index(debates_data, year)
        
        return {
            'year': year,
            'processed': processed_count,
            'errors': error_count,
            'total_files': len(html_files)
        }
    
    def _update_search_index(self, debates_data: List[Dict], year: str):
        """Update SQLite FTS index for fast searching."""
        db_path = self.processed_data_path / 'index' / 'debates.db'
        
        conn = sqlite3.connect(db_path)
        
        # Create FTS table if it doesn't exist
        conn.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS debates_fts USING fts5(
                file_path, file_name, title, chamber, year, month,
                reference_date, speakers, debate_topics, full_text,
                content=debates, content_rowid=rowid
            )
        ''')
        
        # Create main table if it doesn't exist
        conn.execute('''
            CREATE TABLE IF NOT EXISTS debates (
                rowid INTEGER PRIMARY KEY,
                file_path TEXT UNIQUE,
                file_name TEXT,
                title TEXT,
                chamber TEXT,
                year INTEGER,
                month TEXT,
                reference_date TEXT,
                speakers TEXT,
                debate_topics TEXT,
                word_count INTEGER,
                line_count INTEGER,
                content_hash TEXT,
                extraction_timestamp TEXT
            )
        ''')
        
        # Insert data
        for debate in debates_data:
            conn.execute('''
                INSERT OR REPLACE INTO debates 
                (file_path, file_name, title, chamber, year, month, reference_date,
                 speakers, debate_topics, word_count, line_count, content_hash, extraction_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                debate['file_path'], debate['file_name'], debate.get('title'),
                debate.get('chamber'), debate.get('year'), debate.get('month'),
                debate.get('reference_date'), json.dumps(debate.get('speakers', [])),
                json.dumps(debate.get('debate_topics', [])), debate.get('word_count'),
                debate.get('line_count'), debate['content_hash'], debate['extraction_timestamp']
            ))
        
        conn.commit()
        conn.close()
    
    def process_all_years(self, start_year: int = 1803, end_year: int = 2005) -> Dict:
        """Process all available years in the dataset."""
        results = {
            'processing_start': datetime.now().isoformat(),
            'years_processed': [],
            'total_processed': 0,
            'total_errors': 0,
            'total_files': 0
        }
        
        # Find available years
        available_years = []
        for year_dir in self.raw_data_path.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                year = int(year_dir.name)
                if start_year <= year <= end_year:
                    available_years.append(str(year))
        
        available_years.sort()
        print(f"Found {len(available_years)} years to process: {available_years[:10]}{'...' if len(available_years) > 10 else ''}")
        
        for year in available_years:
            result = self.process_year(year)
            results['years_processed'].append(result)
            results['total_processed'] += result['processed']
            results['total_errors'] += result['errors'] 
            results['total_files'] += result['total_files']
            
            print(f"  ✓ {year}: {result['processed']}/{result['total_files']} files processed")
        
        results['processing_end'] = datetime.now().isoformat()
        
        # Save processing manifest
        manifest_path = self.processed_data_path / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Consolidate parquet files
        self._consolidate_metadata()
        
        return results
    
    def _consolidate_metadata(self):
        """Consolidate yearly parquet files into master files."""
        print("Consolidating metadata files...")
        
        # Consolidate debates
        debate_files = list((self.processed_data_path / 'metadata').glob('debates_*.parquet'))
        if debate_files:
            debates_df = pl.concat([pl.read_parquet(f) for f in debate_files])
            debates_df.write_parquet(self.processed_data_path / 'metadata' / 'debates_master.parquet')
            print(f"  ✓ Consolidated {len(debate_files)} debate files -> debates_master.parquet")
        
        # Consolidate speakers
        speaker_files = list((self.processed_data_path / 'metadata').glob('speakers_*.parquet'))
        if speaker_files:
            speakers_df = pl.concat([pl.read_parquet(f) for f in speaker_files])
            speakers_df.write_parquet(self.processed_data_path / 'metadata' / 'speakers_master.parquet')
            print(f"  ✓ Consolidated {len(speaker_files)} speaker files -> speakers_master.parquet")

def main():
    """Main processing pipeline."""
    raw_data_path = "../data/hansard"
    processed_data_path = "../data/processed"
    
    pipeline = HansardDataPipeline(raw_data_path, processed_data_path)
    
    # Test on a small sample first
    print("Testing on 1803 data...")
    result = pipeline.process_year("1803")
    print(f"Test result: {result}")
    
    # Consolidate the sample data for testing
    print("\nConsolidating metadata...")
    pipeline._consolidate_metadata()
    
    # Uncomment to process all years
    # print("\nProcessing all years...")
    # full_results = pipeline.process_all_years()
    # print(f"Final results: {full_results}")

if __name__ == "__main__":
    main()