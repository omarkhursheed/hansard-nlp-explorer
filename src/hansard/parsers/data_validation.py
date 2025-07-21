#!/usr/bin/env python3
"""
Data validation and provenance tracking for Hansard processed data.

Ensures data integrity, tracks lineage, and provides comprehensive
validation reports for the processed parliamentary archive.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import polars as pl
import pandas as pd

class HansardDataValidator:
    """Comprehensive validation and provenance tracking for processed Hansard data."""
    
    def __init__(self, processed_data_path: str):
        self.processed_data_path = Path(processed_data_path)
        self.validation_report = {
            'validation_timestamp': datetime.now().isoformat(),
            'data_integrity': {},
            'schema_validation': {},
            'content_validation': {},
            'provenance_tracking': {},
            'recommendations': []
        }
    
    def validate_data_integrity(self) -> Dict:
        """Validate data integrity across all storage formats."""
        integrity_report = {
            'file_existence': {},
            'hash_validation': {},
            'cross_format_consistency': {},
            'temporal_consistency': {}
        }
        
        # Check core files exist
        required_files = [
            'metadata/debates_master.parquet',
            'metadata/speakers_master.parquet',
            'index/debates.db',
            'manifest.json'
        ]
        
        for file_path in required_files:
            full_path = self.processed_data_path / file_path
            integrity_report['file_existence'][file_path] = {
                'exists': full_path.exists(),
                'size_bytes': full_path.stat().st_size if full_path.exists() else 0,
                'modified_time': datetime.fromtimestamp(
                    full_path.stat().st_mtime
                ).isoformat() if full_path.exists() else None
            }
        
        # Validate master files
        if (self.processed_data_path / 'metadata/debates_master.parquet').exists():
            debates_df = pl.read_parquet(self.processed_data_path / 'metadata/debates_master.parquet')
            
            integrity_report['hash_validation'] = {
                'total_records': len(debates_df),
                'unique_hashes': debates_df['content_hash'].n_unique(),
                'duplicate_hashes': len(debates_df) - debates_df['content_hash'].n_unique(),
                'null_hashes': debates_df['content_hash'].null_count()
            }
            
            # Cross-format consistency
            content_files = list((self.processed_data_path / 'content').rglob('*.jsonl'))
            total_content_records = 0
            for content_file in content_files:
                with open(content_file, 'r', encoding='utf-8') as f:
                    total_content_records += sum(1 for line in f)
            
            integrity_report['cross_format_consistency'] = {
                'parquet_records': len(debates_df),
                'jsonl_records': total_content_records,
                'consistency_check': len(debates_df) == total_content_records
            }
            
            # Temporal consistency
            year_range = [int(debates_df['year'].min()), int(debates_df['year'].max())]
            integrity_report['temporal_consistency'] = {
                'year_range': year_range,
                'year_gaps': self._find_year_gaps(debates_df),
                'chronological_order': self._check_chronological_order(debates_df)
            }
        
        self.validation_report['data_integrity'] = integrity_report
        return integrity_report
    
    def validate_schema_compliance(self) -> Dict:
        """Validate that data conforms to expected schemas."""
        schema_report = {
            'debates_schema': {},
            'speakers_schema': {},
            'content_schema': {}
        }
        
        # Expected schemas
        expected_debates_schema = {
            'file_path': pl.Utf8,
            'file_name': pl.Utf8,
            'title': pl.Utf8,
            'chamber': pl.Utf8,
            'year': pl.Int64,
            'month': pl.Utf8,
            'content_hash': pl.Utf8,
            'word_count': pl.Int64,
            'line_count': pl.Int64,
            'speaker_count': pl.Int64
        }
        
        # Validate debates schema
        debates_path = self.processed_data_path / 'metadata/debates_master.parquet'
        if debates_path.exists():
            debates_df = pl.read_parquet(debates_path)
            
            schema_report['debates_schema'] = {
                'total_columns': len(debates_df.columns),
                'schema_matches': {},
                'missing_columns': [],
                'unexpected_columns': [],
                'data_type_issues': []
            }
            
            # Check required columns
            for col, expected_type in expected_debates_schema.items():
                if col in debates_df.columns:
                    actual_type = debates_df[col].dtype
                    schema_report['debates_schema']['schema_matches'][col] = {
                        'expected': str(expected_type),
                        'actual': str(actual_type),
                        'matches': actual_type == expected_type
                    }
                else:
                    schema_report['debates_schema']['missing_columns'].append(col)
            
            # Check for unexpected columns
            expected_cols = set(expected_debates_schema.keys())
            actual_cols = set(debates_df.columns)
            schema_report['debates_schema']['unexpected_columns'] = list(
                actual_cols - expected_cols
            )
        
        # Validate speakers schema
        speakers_path = self.processed_data_path / 'metadata/speakers_master.parquet'
        if speakers_path.exists():
            speakers_df = pl.read_parquet(speakers_path)
            
            schema_report['speakers_schema'] = {
                'total_records': len(speakers_df),
                'unique_speakers': speakers_df['speaker_name'].n_unique(),
                'speaker_name_quality': self._validate_speaker_names(speakers_df)
            }
        
        self.validation_report['schema_validation'] = schema_report
        return schema_report
    
    def validate_content_quality(self) -> Dict:
        """Validate content quality and completeness."""
        content_report = {
            'parsing_success_rates': {},
            'content_completeness': {},
            'metadata_richness': {},
            'temporal_coverage': {}
        }
        
        debates_path = self.processed_data_path / 'metadata/debates_master.parquet'
        if debates_path.exists():
            debates_df = pl.read_parquet(debates_path)
            
            # Parsing success rates
            content_report['parsing_success_rates'] = {
                'total_files': len(debates_df),
                'successful_parsing': len(debates_df.filter(pl.col('success') == True)),
                'success_rate': float(debates_df['success'].mean()),
                'files_with_errors': len(debates_df.filter(pl.col('error').is_not_null()))
            }
            
            # Content completeness
            content_report['content_completeness'] = {
                'files_with_content': len(debates_df.filter(pl.col('word_count') > 0)),
                'avg_words_per_debate': float(debates_df['word_count'].mean()),
                'files_with_speakers': len(debates_df.filter(pl.col('speaker_count') > 0)),
                'files_with_hansard_refs': len(debates_df.filter(
                    pl.col('hansard_reference').is_not_null()
                ))
            }
            
            # Metadata richness
            content_report['metadata_richness'] = {
                'speaker_identification_rate': float(
                    debates_df.filter(pl.col('speaker_count') > 0).height / len(debates_df)
                ),
                'topic_extraction_rate': float(
                    debates_df.filter(pl.col('debate_topics').list.len() > 0).height / len(debates_df)
                ),
                'hansard_reference_rate': float(
                    debates_df.filter(pl.col('hansard_reference').is_not_null()).height / len(debates_df)
                )
            }
            
            # Temporal coverage
            year_counts = debates_df.group_by('year').count().sort('year')
            content_report['temporal_coverage'] = {
                'years_covered': len(year_counts),
                'year_range': [int(year_counts['year'].min()), int(year_counts['year'].max())],
                'debates_per_year': year_counts.to_pandas().to_dict('records'),
                'coverage_gaps': self._identify_coverage_gaps(year_counts)
            }
        
        self.validation_report['content_validation'] = content_report
        return content_report
    
    def track_data_provenance(self) -> Dict:
        """Track data lineage and processing provenance."""
        provenance_report = {
            'source_data': {},
            'processing_pipeline': {},
            'transformation_history': {},
            'data_lineage': {}
        }
        
        # Load manifest if available
        manifest_path = self.processed_data_path / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            provenance_report['processing_pipeline'] = {
                'processing_start': manifest.get('processing_start'),
                'processing_end': manifest.get('processing_end'),
                'total_files_processed': manifest.get('total_processed', 0),
                'total_errors': manifest.get('total_errors', 0),
                'years_processed': [year['year'] for year in manifest.get('years_processed', [])]
            }
        
        # Track file transformations
        debates_path = self.processed_data_path / 'metadata/debates_master.parquet'
        if debates_path.exists():
            debates_df = pl.read_parquet(debates_path)
            
            provenance_report['transformation_history'] = {
                'extraction_timestamps': {
                    'earliest': debates_df['extraction_timestamp'].min(),
                    'latest': debates_df['extraction_timestamp'].max(),
                    'unique_sessions': debates_df['extraction_timestamp'].n_unique()
                },
                'source_file_tracking': {
                    'total_source_files': len(debates_df),
                    'unique_source_paths': debates_df['file_path'].n_unique(),
                    'source_file_sizes': {
                        'total_bytes': int(debates_df['file_size'].sum()),
                        'avg_bytes': float(debates_df['file_size'].mean()),
                        'size_distribution': self._calculate_size_distribution(debates_df)
                    }
                }
            }
        
        self.validation_report['provenance_tracking'] = provenance_report
        return provenance_report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations for data quality improvements."""
        recommendations = []
        
        # Check integrity issues
        integrity = self.validation_report.get('data_integrity', {})
        if integrity.get('hash_validation', {}).get('duplicate_hashes', 0) > 0:
            recommendations.append(
                "Found duplicate content hashes - investigate potential duplicate processing"
            )
        
        # Check content quality
        content = self.validation_report.get('content_validation', {})
        success_rate = content.get('parsing_success_rates', {}).get('success_rate', 1.0)
        if success_rate < 0.95:
            recommendations.append(
                f"Parsing success rate ({success_rate:.1%}) below 95% - review parsing errors"
            )
        
        speaker_rate = content.get('metadata_richness', {}).get('speaker_identification_rate', 0)
        if speaker_rate < 0.5:
            recommendations.append(
                f"Speaker identification rate ({speaker_rate:.1%}) below 50% - improve speaker extraction"
            )
        
        # Check schema compliance
        schema = self.validation_report.get('schema_validation', {})
        missing_cols = schema.get('debates_schema', {}).get('missing_columns', [])
        if missing_cols:
            recommendations.append(
                f"Missing required columns: {', '.join(missing_cols)}"
            )
        
        self.validation_report['recommendations'] = recommendations
        return recommendations
    
    def _find_year_gaps(self, debates_df: pl.DataFrame) -> List[int]:
        """Find missing years in the dataset."""
        years = sorted(debates_df['year'].unique().to_list())
        if not years:
            return []
        
        full_range = set(range(min(years), max(years) + 1))
        present_years = set(years)
        return sorted(list(full_range - present_years))
    
    def _check_chronological_order(self, debates_df: pl.DataFrame) -> bool:
        """Check if debates are in chronological order."""
        sorted_dates = debates_df.select(['year', 'month', 'reference_date']).sort(['year', 'month'])
        original_order = debates_df.select(['year', 'month', 'reference_date'])
        return sorted_dates.equals(original_order)
    
    def _validate_speaker_names(self, speakers_df: pl.DataFrame) -> Dict:
        """Validate speaker name quality."""
        return {
            'total_speakers': len(speakers_df),
            'unique_speakers': speakers_df['speaker_name'].n_unique(),
            'avg_name_length': float(speakers_df['speaker_name'].str.len_chars().mean()),
            'names_with_titles': len(speakers_df.filter(
                pl.col('speaker_name').str.contains(r'(?i)(mr|mrs|lord|the)')
            ))
        }
    
    def _identify_coverage_gaps(self, year_counts: pl.DataFrame) -> List[Dict]:
        """Identify significant gaps in temporal coverage."""
        gaps = []
        if len(year_counts) <= 1:
            return gaps
            
        years_list = year_counts.sort('year')['year'].to_list()
        
        for i in range(len(years_list) - 1):
            current_year = years_list[i]
            next_year = years_list[i + 1]
            
            if next_year - current_year > 1:
                gaps.append({
                    'start_year': current_year,
                    'end_year': next_year,
                    'gap_size': next_year - current_year - 1
                })
        
        return gaps
    
    def _calculate_size_distribution(self, debates_df: pl.DataFrame) -> Dict:
        """Calculate file size distribution statistics."""
        sizes = debates_df['file_size']
        return {
            'min_bytes': int(sizes.min()),
            'max_bytes': int(sizes.max()),
            'median_bytes': int(sizes.median()),
            'q25_bytes': int(sizes.quantile(0.25)),
            'q75_bytes': int(sizes.quantile(0.75))
        }
    
    def run_full_validation(self) -> Dict:
        """Run complete validation suite."""
        print("Running comprehensive data validation...")
        
        print("  ✓ Validating data integrity...")
        self.validate_data_integrity()
        
        print("  ✓ Validating schema compliance...")
        self.validate_schema_compliance()
        
        print("  ✓ Validating content quality...")
        self.validate_content_quality()
        
        print("  ✓ Tracking data provenance...")
        self.track_data_provenance()
        
        print("  ✓ Generating recommendations...")
        self.generate_recommendations()
        
        # Save validation report
        report_path = self.processed_data_path / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.validation_report, f, indent=2)
        
        print(f"  ✓ Validation report saved to: {report_path}")
        return self.validation_report

def main():
    """Run validation on processed data."""
    processed_data_path = "../data/processed"
    
    validator = HansardDataValidator(processed_data_path)
    
    try:
        report = validator.run_full_validation()
        
        print("\n=== Validation Summary ===")
        
        # Data integrity
        integrity = report.get('data_integrity', {})
        consistency = integrity.get('cross_format_consistency', {})
        print(f"Data consistency: {'✓ PASS' if consistency.get('consistency_check') else '✗ FAIL'}")
        
        # Content quality
        content = report.get('content_validation', {})
        success_rate = content.get('parsing_success_rates', {}).get('success_rate', 0)
        print(f"Parsing success rate: {success_rate:.1%}")
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            print(f"\nRecommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("\n✓ No recommendations - data quality is excellent!")
        
        return report
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return None

if __name__ == "__main__":
    main()