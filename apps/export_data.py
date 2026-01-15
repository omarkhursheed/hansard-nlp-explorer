#!/usr/bin/env python3
"""
Export sample data from parquet files to JSON for web apps.
Creates lightweight JSON files that can be loaded by static web apps.
"""

import pandas as pd
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

def export_speeches_sample(output_dir: Path, years=[1920, 1950, 1980, 2000], sample_per_year=500):
    """Export sample speeches for the speech browser."""
    print("Exporting speeches sample...")

    speeches_dir = project_root / 'data-hansard/derived_complete/speeches_complete'
    all_speeches = []

    for year in years:
        file_path = speeches_dir / f'speeches_{year}.parquet'
        if file_path.exists():
            df = pd.read_parquet(file_path)
            # Sample and select key columns
            sample = df.sample(n=min(sample_per_year, len(df)), random_state=42)
            sample = sample[[
                'speech_id', 'debate_id', 'sequence_number', 'speaker',
                'canonical_name', 'gender', 'party', 'constituency',
                'text', 'word_count', 'year', 'date', 'chamber', 'title'
            ]].copy()
            # Truncate text for preview
            sample['text_preview'] = sample['text'].str[:500] + '...'
            all_speeches.append(sample)

    combined = pd.concat(all_speeches, ignore_index=True)
    combined = combined.fillna('')

    output_path = output_dir / 'speech-browser/data/speeches_sample.json'
    combined.to_json(output_path, orient='records', indent=2)
    print(f"  Exported {len(combined)} speeches to {output_path}")
    return combined


def export_debates_sample(output_dir: Path, years=[1918, 1928, 1970, 2000], sample_per_year=100):
    """Export sample debates for the debate viewer."""
    print("Exporting debates sample...")

    debates_dir = project_root / 'data-hansard/derived_complete/debates_complete'
    speeches_dir = project_root / 'data-hansard/derived_complete/speeches_complete'
    all_debates = []

    for year in years:
        debates_path = debates_dir / f'debates_{year}.parquet'
        speeches_path = speeches_dir / f'speeches_{year}.parquet'

        if debates_path.exists() and speeches_path.exists():
            debates = pd.read_parquet(debates_path)
            speeches = pd.read_parquet(speeches_path)

            # Get debates with multiple speakers
            multi_speaker = debates[debates['speech_count'] >= 3]
            sample = multi_speaker.sample(n=min(sample_per_year, len(multi_speaker)), random_state=42)

            for _, debate in sample.iterrows():
                debate_speeches = speeches[speeches['debate_id'] == debate['debate_id']].sort_values('sequence_number')

                debate_data = {
                    'debate_id': debate['debate_id'],
                    'title': debate['title'],
                    'date': str(debate['date']),
                    'year': int(debate['year']),
                    'chamber': debate['chamber'],
                    'speech_count': int(debate['speech_count']),
                    'female_mps': int(debate['female_mps']) if pd.notna(debate['female_mps']) else 0,
                    'male_mps': int(debate['male_mps']) if pd.notna(debate['male_mps']) else 0,
                    'speeches': []
                }

                for _, speech in debate_speeches.head(20).iterrows():  # Limit speeches per debate
                    debate_data['speeches'].append({
                        'speaker': speech['speaker'] or '',
                        'canonical_name': speech['canonical_name'] or '',
                        'gender': speech['gender'] or '',
                        'party': speech['party'] or '',
                        'text': speech['text'][:2000] if speech['text'] else '',
                        'sequence_number': int(speech['sequence_number'])
                    })

                all_debates.append(debate_data)

    output_path = output_dir / 'debate-viewer/data/debates_sample.json'
    with open(output_path, 'w') as f:
        json.dump(all_debates, f, indent=2, default=str)
    print(f"  Exported {len(all_debates)} debates to {output_path}")


def export_mp_profiles(output_dir: Path):
    """Export MP profiles with speech counts."""
    print("Exporting MP profiles...")

    speeches_dir = project_root / 'data-hansard/derived_complete/speeches_complete'

    # Aggregate across all years
    mp_data = {}

    for year_file in sorted(speeches_dir.glob('speeches_*.parquet')):
        year = int(year_file.stem.split('_')[1])
        df = pd.read_parquet(year_file)

        # Only matched MPs
        matched = df[df['matched_mp'] == True]

        for _, row in matched.groupby('person_id').agg({
            'canonical_name': 'first',
            'gender': 'first',
            'party': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else '',
            'constituency': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else '',
            'speech_id': 'count',
            'word_count': 'sum'
        }).reset_index().iterrows():

            pid = row['person_id']
            if pid not in mp_data:
                mp_data[pid] = {
                    'person_id': pid,
                    'name': row['canonical_name'],
                    'gender': row['gender'],
                    'party': row['party'],
                    'constituency': row['constituency'],
                    'total_speeches': 0,
                    'total_words': 0,
                    'years_active': [],
                    'speeches_by_year': {}
                }

            mp_data[pid]['total_speeches'] += row['speech_id']
            mp_data[pid]['total_words'] += row['word_count']
            mp_data[pid]['years_active'].append(year)
            mp_data[pid]['speeches_by_year'][str(year)] = int(row['speech_id'])

    # Convert to list and sort by total speeches
    mp_list = sorted(mp_data.values(), key=lambda x: x['total_speeches'], reverse=True)

    # Take top 500 MPs
    mp_list = mp_list[:500]

    output_path = output_dir / 'mp-explorer/data/mp_profiles.json'
    with open(output_path, 'w') as f:
        json.dump(mp_list, f, indent=2)
    print(f"  Exported {len(mp_list)} MP profiles to {output_path}")


def export_temporal_trends(output_dir: Path):
    """Export aggregated temporal trends data."""
    print("Exporting temporal trends...")

    speeches_dir = project_root / 'data-hansard/derived_complete/speeches_complete'

    yearly_stats = []

    for year_file in sorted(speeches_dir.glob('speeches_*.parquet')):
        year = int(year_file.stem.split('_')[1])
        df = pd.read_parquet(year_file)

        commons = df[df['chamber'] == 'Commons']
        matched = commons[commons['matched_mp'] == True]

        stats = {
            'year': year,
            'total_speeches': len(df),
            'commons_speeches': len(commons),
            'lords_speeches': len(df[df['chamber'] == 'Lords']),
            'matched_speeches': len(matched),
            'female_speeches': len(matched[matched['gender'] == 'F']),
            'male_speeches': len(matched[matched['gender'] == 'M']),
            'total_words': int(df['word_count'].sum()),
            'avg_speech_length': float(df['word_count'].mean()),
            'unique_speakers': df['normalized_speaker'].nunique(),
            'unique_mps': matched['person_id'].nunique(),
            'female_mps': matched[matched['gender'] == 'F']['person_id'].nunique(),
            'male_mps': matched[matched['gender'] == 'M']['person_id'].nunique(),
        }

        # Calculate percentages
        if stats['matched_speeches'] > 0:
            stats['female_pct'] = round(stats['female_speeches'] / stats['matched_speeches'] * 100, 2)
        else:
            stats['female_pct'] = 0

        yearly_stats.append(stats)

    output_path = output_dir / 'trends-dashboard/data/yearly_trends.json'
    with open(output_path, 'w') as f:
        json.dump(yearly_stats, f, indent=2)
    print(f"  Exported {len(yearly_stats)} years of trends to {output_path}")


def export_suffrage_data(output_dir: Path):
    """Export suffrage classification data."""
    print("Exporting suffrage classification data...")

    suffrage_path = project_root / 'outputs/llm_classification/full_results_v5_context_3_complete.parquet'

    if not suffrage_path.exists():
        print("  Suffrage data not found, skipping...")
        return

    df = pd.read_parquet(suffrage_path)

    # Clean and select columns
    export_cols = [
        'speech_id', 'debate_id', 'speaker', 'canonical_name', 'gender',
        'party', 'year', 'date', 'chamber', 'stance', 'reasons',
        'top_quote', 'confidence', 'confidence_level', 'word_count'
    ]

    export_df = df[export_cols].copy()
    export_df = export_df.fillna('')

    # Convert to records
    records = export_df.to_dict(orient='records')

    output_path = output_dir / 'suffrage-explorer/data/suffrage_results.json'
    with open(output_path, 'w') as f:
        json.dump(records, f, indent=2, default=str)
    print(f"  Exported {len(records)} suffrage classifications to {output_path}")

    # Also export summary stats
    summary = {
        'total_classified': len(df),
        'stance_counts': df['stance'].value_counts().to_dict(),
        'gender_counts': df['gender'].value_counts().to_dict(),
        'year_range': [int(df['year'].min()), int(df['year'].max())],
        'by_year_stance': df.groupby(['year', 'stance']).size().unstack(fill_value=0).to_dict()
    }

    summary_path = output_dir / 'suffrage-explorer/data/suffrage_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)


def export_dataset_stats(output_dir: Path):
    """Export overall dataset statistics."""
    print("Exporting dataset statistics...")

    speeches_dir = project_root / 'data-hansard/derived_complete/speeches_complete'
    debates_dir = project_root / 'data-hansard/derived_complete/debates_complete'

    stats = {
        'overview': {
            'total_speeches': 0,
            'total_debates': 0,
            'total_words': 0,
            'year_range': [9999, 0],
            'unique_mps': set(),
            'female_mps': set(),
            'male_mps': set(),
        },
        'by_chamber': {'Commons': {'speeches': 0, 'matched': 0}, 'Lords': {'speeches': 0, 'matched': 0}},
        'by_decade': {},
        'by_year': [],
    }

    for year_file in sorted(speeches_dir.glob('speeches_*.parquet')):
        year = int(year_file.stem.split('_')[1])
        df = pd.read_parquet(year_file)

        stats['overview']['total_speeches'] += len(df)
        stats['overview']['total_words'] += int(df['word_count'].sum())
        stats['overview']['year_range'][0] = min(stats['overview']['year_range'][0], year)
        stats['overview']['year_range'][1] = max(stats['overview']['year_range'][1], year)

        matched = df[df['matched_mp'] == True]
        stats['overview']['unique_mps'].update(matched['person_id'].dropna().unique())
        stats['overview']['female_mps'].update(matched[matched['gender'] == 'F']['person_id'].dropna().unique())
        stats['overview']['male_mps'].update(matched[matched['gender'] == 'M']['person_id'].dropna().unique())

        # By chamber
        for chamber in ['Commons', 'Lords']:
            chamber_df = df[df['chamber'] == chamber]
            stats['by_chamber'][chamber]['speeches'] += len(chamber_df)
            stats['by_chamber'][chamber]['matched'] += len(chamber_df[chamber_df['matched_mp'] == True])

        # By decade
        decade = (year // 10) * 10
        if decade not in stats['by_decade']:
            stats['by_decade'][decade] = {'speeches': 0, 'female': 0, 'male': 0}
        stats['by_decade'][decade]['speeches'] += len(df)
        stats['by_decade'][decade]['female'] += len(matched[matched['gender'] == 'F'])
        stats['by_decade'][decade]['male'] += len(matched[matched['gender'] == 'M'])

        # By year summary
        stats['by_year'].append({
            'year': year,
            'speeches': len(df),
            'female': len(matched[matched['gender'] == 'F']),
            'male': len(matched[matched['gender'] == 'M']),
        })

    # Count debates
    for debates_file in debates_dir.glob('debates_*.parquet'):
        df = pd.read_parquet(debates_file)
        stats['overview']['total_debates'] += len(df)

    # Convert sets to counts
    stats['overview']['unique_mps'] = len(stats['overview']['unique_mps'])
    stats['overview']['female_mps'] = len(stats['overview']['female_mps'])
    stats['overview']['male_mps'] = len(stats['overview']['male_mps'])

    output_path = output_dir / 'stats-dashboard/data/dataset_stats.json'
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Exported dataset stats to {output_path}")


def main():
    output_dir = Path(__file__).parent

    print("=" * 60)
    print("EXPORTING DATA FOR WEB APPS")
    print("=" * 60)

    export_speeches_sample(output_dir)
    export_debates_sample(output_dir)
    export_mp_profiles(output_dir)
    export_temporal_trends(output_dir)
    export_suffrage_data(output_dir)
    export_dataset_stats(output_dir)

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
