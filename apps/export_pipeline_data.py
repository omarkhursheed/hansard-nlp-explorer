#!/usr/bin/env python3
"""
Export pipeline transformation data with full provenance tracing.
Supports searching and tracing any debate/speech through the pipeline.
"""

import pandas as pd
import json
import gzip
from pathlib import Path
import sys
import hashlib

project_root = Path(__file__).resolve().parent.parent


def get_raw_html(file_path: str) -> dict:
    """Get raw HTML for a debate from its file_path."""
    full_path = project_root / file_path

    if full_path.exists():
        try:
            with gzip.open(full_path, 'rt', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                return {
                    'size_bytes': len(content),
                    'preview': content[:5000],
                    'full_content': content,
                    'has_contribution_tags': 'contribution' in content.lower(),
                }
        except Exception as e:
            return {'error': str(e)}
    return None


def get_processed_metadata(file_path: str, year: int) -> dict:
    """Get processed metadata for a debate."""
    metadata_dir = project_root / 'data-hansard/processed_complete/metadata'
    metadata_file = metadata_dir / f'debates_{year}.parquet'

    if metadata_file.exists():
        df = pd.read_parquet(metadata_file)
        matches = df[df['file_path'] == file_path]
        if len(matches) > 0:
            row = matches.iloc[0].to_dict()
            clean = {}
            for k, v in row.items():
                if pd.isna(v):
                    clean[k] = None
                elif isinstance(v, (int, float, str, bool)):
                    clean[k] = v
                elif hasattr(v, 'isoformat'):
                    clean[k] = v.isoformat()
                elif isinstance(v, list):
                    clean[k] = v[:20]
                else:
                    clean[k] = str(v)[:500]
            return clean
    return None


def get_gender_enhanced(file_path: str, year: int) -> dict:
    """Get gender-enhanced data for a debate."""
    gender_dir = project_root / 'data-hansard/gender_analysis_complete'
    gender_file = gender_dir / f'debates_{year}_enhanced.parquet'

    if gender_file.exists():
        df = pd.read_parquet(gender_file)
        matches = df[df['file_path'] == file_path]
        if len(matches) > 0:
            row = matches.iloc[0].to_dict()
            clean = {}
            for k, v in row.items():
                if pd.isna(v):
                    clean[k] = None
                elif isinstance(v, (int, float, str, bool)):
                    clean[k] = v
                elif isinstance(v, list):
                    clean[k] = v[:20]
                else:
                    clean[k] = str(v)[:500]
            return clean
    return None


def build_debate_index():
    """Build an index of all debates for quick lookup."""
    print("Building debate index...")

    speeches_dir = project_root / 'data-hansard/derived_complete/speeches_complete'
    index = {}

    for year_file in sorted(speeches_dir.glob('speeches_*.parquet')):
        year = int(year_file.stem.split('_')[1])
        print(f"  Indexing {year}...")

        df = pd.read_parquet(year_file)

        # Group by debate
        for debate_id, group in df.groupby('debate_id'):
            first = group.iloc[0]
            has_female = (group['gender'] == 'F').any()
            has_male = (group['gender'] == 'M').any()

            index[debate_id] = {
                'debate_id': debate_id,
                'year': year,
                'title': first['title'],
                'date': str(first['date']),
                'chamber': first['chamber'],
                'file_path': first['file_path'],
                'speech_count': len(group),
                'has_female': bool(has_female),
                'has_male': bool(has_male),
                'speakers': group['canonical_name'].dropna().unique().tolist()[:10],
            }

    return index


def export_debate_index():
    """Export a searchable index of all debates."""
    index = build_debate_index()

    # Convert to list and save
    index_list = list(index.values())

    output_path = project_root / 'apps/pipeline-visualizer/data/debate_index.json'
    with open(output_path, 'w') as f:
        json.dump(index_list, f)
    print(f"Exported {len(index_list)} debates to {output_path}")

    # Also save a smaller sample for initial load
    sample = index_list[:1000]
    sample_path = project_root / 'apps/pipeline-visualizer/data/debate_index_sample.json'
    with open(sample_path, 'w') as f:
        json.dump(sample, f)
    print(f"Exported sample of {len(sample)} debates to {sample_path}")


def export_pipeline_stats():
    """Export statistics about each pipeline stage."""
    print("\nExporting pipeline statistics...")

    stats = {
        'raw_html': {
            'description': 'Raw HTML files from UK Parliament API',
            'location': 'data-hansard/hansard/',
            'format': 'Gzipped HTML files organized by year/month',
            'total_files': 0,
        },
        'processed': {
            'description': 'Extracted text and metadata from HTML',
            'location': 'data-hansard/processed_complete/',
            'format': 'JSONL (text) + Parquet (metadata)',
            'total_debates': 0,
            'columns': []
        },
        'gender_enhanced': {
            'description': 'Debates with MP matching and gender attribution',
            'location': 'data-hansard/gender_analysis_complete/',
            'format': 'Parquet files by year',
            'total_debates': 0,
            'columns': []
        },
        'unified': {
            'description': 'Final speech-level dataset with all metadata',
            'location': 'data-hansard/derived_complete/',
            'format': 'Parquet files partitioned by year',
            'total_speeches': 0,
            'total_debates': 0,
            'columns': []
        }
    }

    # Count raw HTML files
    hansard_dir = project_root / 'data-hansard/hansard'
    if hansard_dir.exists():
        stats['raw_html']['total_files'] = len(list(hansard_dir.rglob('*.html.gz')))

    # Get processed stats
    processed_dir = project_root / 'data-hansard/processed_complete/metadata'
    if processed_dir.exists():
        sample_file = next(processed_dir.glob('*.parquet'), None)
        if sample_file:
            df = pd.read_parquet(sample_file)
            stats['processed']['columns'] = list(df.columns)
        for f in processed_dir.glob('*.parquet'):
            df = pd.read_parquet(f)
            stats['processed']['total_debates'] += len(df)

    # Get gender-enhanced stats
    gender_dir = project_root / 'data-hansard/gender_analysis_complete'
    if gender_dir.exists():
        sample_file = next(gender_dir.glob('debates_*_enhanced.parquet'), None)
        if sample_file:
            df = pd.read_parquet(sample_file)
            stats['gender_enhanced']['columns'] = list(df.columns)[:25]
        for f in gender_dir.glob('debates_*_enhanced.parquet'):
            df = pd.read_parquet(f)
            stats['gender_enhanced']['total_debates'] += len(df)

    # Get unified stats
    speeches_dir = project_root / 'data-hansard/derived_complete/speeches_complete'
    debates_dir = project_root / 'data-hansard/derived_complete/debates_complete'
    if speeches_dir.exists():
        sample_file = next(speeches_dir.glob('*.parquet'), None)
        if sample_file:
            df = pd.read_parquet(sample_file)
            stats['unified']['columns'] = list(df.columns)
        for f in speeches_dir.glob('*.parquet'):
            df = pd.read_parquet(f)
            stats['unified']['total_speeches'] += len(df)
    if debates_dir.exists():
        for f in debates_dir.glob('*.parquet'):
            df = pd.read_parquet(f)
            stats['unified']['total_debates'] += len(df)

    output_path = project_root / 'apps/pipeline-visualizer/data/pipeline_stats.json'
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Exported pipeline stats to {output_path}")


def trace_debate(debate_id: str, year: int = None) -> dict:
    """Trace a specific debate through all pipeline stages."""
    result = {
        'debate_id': debate_id,
        'stages': {
            'raw_html': None,
            'processed': None,
            'gender_enhanced': None,
            'final_speeches': []
        }
    }

    # First, find the debate in unified data to get the file_path
    speeches_dir = project_root / 'data-hansard/derived_complete/speeches_complete'

    if year:
        years_to_check = [year]
    else:
        years_to_check = range(1803, 2006)

    file_path = None
    for y in years_to_check:
        speeches_file = speeches_dir / f'speeches_{y}.parquet'
        if speeches_file.exists():
            df = pd.read_parquet(speeches_file)
            matches = df[df['debate_id'] == debate_id]
            if len(matches) > 0:
                file_path = matches.iloc[0]['file_path']
                year = y
                # Get all speeches
                for _, row in matches.sort_values('sequence_number').head(20).iterrows():
                    speech = {}
                    for k, v in row.items():
                        if pd.isna(v):
                            speech[k] = None
                        elif k == 'text':
                            speech[k] = str(v)[:2000] if v else ''
                        elif isinstance(v, (int, float, str, bool)):
                            speech[k] = v
                        else:
                            speech[k] = str(v)
                    result['stages']['final_speeches'].append(speech)
                break

    if not file_path:
        return result

    result['year'] = year
    result['file_path'] = file_path

    # Get raw HTML
    raw = get_raw_html(file_path)
    if raw:
        # Don't include full content in trace - just preview
        result['stages']['raw_html'] = {
            'size_bytes': raw['size_bytes'],
            'preview': raw['preview'],
            'has_contribution_tags': raw['has_contribution_tags'],
        }

    # Get processed
    processed = get_processed_metadata(file_path, year)
    if processed:
        result['stages']['processed'] = processed

    # Get gender enhanced
    gender = get_gender_enhanced(file_path, year)
    if gender:
        result['stages']['gender_enhanced'] = gender

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', action='store_true', help='Build full debate index')
    parser.add_argument('--stats', action='store_true', help='Export pipeline stats')
    parser.add_argument('--trace', type=str, help='Trace a specific debate ID')
    parser.add_argument('--year', type=int, help='Year hint for tracing')
    args = parser.parse_args()

    if args.trace:
        result = trace_debate(args.trace, args.year)
        print(json.dumps(result, indent=2, default=str))
    elif args.index:
        export_debate_index()
        export_pipeline_stats()
    elif args.stats:
        export_pipeline_stats()
    else:
        # Default: export everything
        export_debate_index()
        export_pipeline_stats()
        print("\nPipeline data export complete!")
