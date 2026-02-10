#!/usr/bin/env python3
"""
Build optimized static data files for fast, offline-first web apps.
Creates pre-indexed, compressed JSON that can be served from any CDN.
"""

import pandas as pd
import json
import gzip
from pathlib import Path
from collections import defaultdict
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

def build_search_index(output_dir: Path, max_speeches_per_year: int = 2000):
    """Build a lightweight search index for client-side search."""
    print("Building search index...")

    speeches_dir = project_root / 'data-hansard/derived_complete_v2/speeches_complete'
    if not speeches_dir.exists():
        speeches_dir = project_root / 'data-hansard/derived_complete/speeches_complete'

    # Build index by decade for faster loading
    by_decade = defaultdict(list)
    total = 0

    for year_file in sorted(speeches_dir.glob('speeches_*.parquet')):
        year = int(year_file.stem.split('_')[1])
        decade = (year // 10) * 10

        df = pd.read_parquet(year_file)

        # Sample for manageability (prioritize female speakers)
        female = df[df['gender'] == 'F']
        male_df = df[df['gender'] == 'M']

        # Calculate how many male speeches to include
        male_quota = max(0, max_speeches_per_year - len(female))
        male_sample_size = min(male_quota, len(male_df))

        if male_sample_size > 0:
            male = male_df.sample(n=male_sample_size, random_state=42)
        else:
            male = pd.DataFrame()

        sample = pd.concat([female, male])
        if len(sample) > max_speeches_per_year:
            sample = sample.sample(n=max_speeches_per_year, random_state=42)

        for _, row in sample.iterrows():
            text = str(row.get('text', ''))[:500]  # Truncate for index

            entry = {
                'id': f"{row['debate_id']}_s{row['sequence_number']}",
                'y': year,
                'd': row['debate_id'],
                's': row.get('sequence_number', 0),
                'sp': row.get('canonical_name') or row.get('speaker', ''),
                'g': row.get('gender', ''),
                'c': row.get('chamber', ''),
                't': row.get('title', '')[:100],
                'x': text,  # Text excerpt for search
                'dt': str(row.get('date', ''))[:10],
            }
            by_decade[decade].append(entry)
            total += 1

    # Write decade files
    index_dir = output_dir / 'static-data' / 'search-index'
    index_dir.mkdir(parents=True, exist_ok=True)

    manifest = {'decades': [], 'total': total}

    for decade, entries in sorted(by_decade.items()):
        filename = f'decade_{decade}.json'
        filepath = index_dir / filename

        with open(filepath, 'w') as f:
            json.dump(entries, f, separators=(',', ':'))  # Compact JSON

        # Also create gzipped version
        with gzip.open(str(filepath) + '.gz', 'wt') as f:
            json.dump(entries, f, separators=(',', ':'))

        manifest['decades'].append({
            'decade': decade,
            'file': filename,
            'count': len(entries),
            'size_kb': filepath.stat().st_size // 1024
        })
        print(f"  Decade {decade}: {len(entries)} speeches ({filepath.stat().st_size // 1024}KB)")

    # Write manifest
    with open(index_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"  Total: {total} speeches indexed")
    return manifest


def build_stats_data(output_dir: Path):
    """Build comprehensive stats for dashboards."""
    print("Building stats data...")

    speeches_dir = project_root / 'data-hansard/derived_complete_v2/speeches_complete'
    if not speeches_dir.exists():
        speeches_dir = project_root / 'data-hansard/derived_complete/speeches_complete'

    stats = {
        'overview': {
            'total_speeches': 0,
            'total_debates': 0,
            'total_words': 0,
            'matched_speeches': 0,
            'female_speeches': 0,
            'male_speeches': 0,
            'year_min': 9999,
            'year_max': 0,
            'unique_mps': set(),
            'female_mps': set(),
        },
        'by_year': [],
        'by_decade': {},
        'by_chamber': {'Commons': 0, 'Lords': 0},
        'top_female_mps': [],
        'milestones': [],
    }

    yearly_data = []
    mp_speech_counts = defaultdict(lambda: {'name': '', 'gender': '', 'speeches': 0, 'years': set()})

    for year_file in sorted(speeches_dir.glob('speeches_*.parquet')):
        year = int(year_file.stem.split('_')[1])
        df = pd.read_parquet(year_file)

        matched = df[df['matched_mp'] == True]
        female = matched[matched['gender'] == 'F']
        male = matched[matched['gender'] == 'M']

        stats['overview']['total_speeches'] += len(df)
        stats['overview']['total_words'] += int(df['word_count'].sum())
        stats['overview']['matched_speeches'] += len(matched)
        stats['overview']['female_speeches'] += len(female)
        stats['overview']['male_speeches'] += len(male)
        stats['overview']['year_min'] = min(stats['overview']['year_min'], year)
        stats['overview']['year_max'] = max(stats['overview']['year_max'], year)
        stats['overview']['unique_mps'].update(matched['person_id'].dropna().unique())
        stats['overview']['female_mps'].update(female['person_id'].dropna().unique())

        # Chamber counts
        stats['by_chamber']['Commons'] += len(df[df['chamber'] == 'Commons'])
        stats['by_chamber']['Lords'] += len(df[df['chamber'] == 'Lords'])

        # Track MP speech counts
        for pid, group in female.groupby('person_id'):
            if pd.notna(pid):
                mp_speech_counts[pid]['name'] = group['canonical_name'].iloc[0]
                mp_speech_counts[pid]['gender'] = 'F'
                mp_speech_counts[pid]['speeches'] += len(group)
                mp_speech_counts[pid]['years'].add(year)

        # Yearly data
        year_stats = {
            'year': year,
            'speeches': len(df),
            'matched': len(matched),
            'female': len(female),
            'male': len(male),
            'female_pct': round(len(female) / len(matched) * 100, 2) if len(matched) > 0 else 0,
            'unique_female_mps': female['person_id'].nunique(),
            'unique_male_mps': male['person_id'].nunique(),
            'avg_words': round(df['word_count'].mean(), 1),
        }
        yearly_data.append(year_stats)

        # Decade aggregation
        decade = (year // 10) * 10
        if decade not in stats['by_decade']:
            stats['by_decade'][decade] = {'speeches': 0, 'female': 0, 'male': 0}
        stats['by_decade'][decade]['speeches'] += len(matched)
        stats['by_decade'][decade]['female'] += len(female)
        stats['by_decade'][decade]['male'] += len(male)

    # Convert sets to counts
    stats['overview']['unique_mps'] = len(stats['overview']['unique_mps'])
    stats['overview']['female_mps'] = len(stats['overview']['female_mps'])

    stats['by_year'] = yearly_data

    # Top female MPs
    female_mps = [(pid, data) for pid, data in mp_speech_counts.items() if data['gender'] == 'F']
    female_mps.sort(key=lambda x: x[1]['speeches'], reverse=True)
    stats['top_female_mps'] = [
        {'name': data['name'], 'speeches': data['speeches'], 'years': f"{min(data['years'])}-{max(data['years'])}"}
        for pid, data in female_mps[:50]
    ]

    # Milestones
    first_female_year = None
    for year_data in yearly_data:
        if year_data['female'] > 0 and first_female_year is None:
            first_female_year = year_data['year']
            stats['milestones'].append({'year': year_data['year'], 'event': 'First female MP speech in dataset'})
        if year_data['female_pct'] >= 5 and not any(m['event'].startswith('5%') for m in stats['milestones']):
            stats['milestones'].append({'year': year_data['year'], 'event': '5% female representation reached'})
        if year_data['female_pct'] >= 10 and not any(m['event'].startswith('10%') for m in stats['milestones']):
            stats['milestones'].append({'year': year_data['year'], 'event': '10% female representation reached'})
        if year_data['female_pct'] >= 20 and not any(m['event'].startswith('20%') for m in stats['milestones']):
            stats['milestones'].append({'year': year_data['year'], 'event': '20% female representation reached'})

    # Write output
    static_dir = output_dir / 'static-data'
    static_dir.mkdir(parents=True, exist_ok=True)

    with open(static_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  Stats exported: {stats['overview']['total_speeches']} speeches, {stats['overview']['unique_mps']} MPs")
    return stats


def build_example_debates(output_dir: Path, samples_per_era: int = 20):
    """Build example debates for the debate viewer."""
    print("Building example debates...")

    speeches_dir = project_root / 'data-hansard/derived_complete_v2/speeches_complete'
    if not speeches_dir.exists():
        speeches_dir = project_root / 'data-hansard/derived_complete/speeches_complete'

    # Sample years representing different eras
    sample_years = [1920, 1928, 1945, 1960, 1975, 1990, 2000, 2005]
    all_debates = []

    for year in sample_years:
        year_file = speeches_dir / f'speeches_{year}.parquet'
        if not year_file.exists():
            continue

        df = pd.read_parquet(year_file)

        # Find debates with female speakers
        debates_with_female = df[df['gender'] == 'F']['debate_id'].unique()

        for debate_id in debates_with_female[:samples_per_era]:
            debate_df = df[df['debate_id'] == debate_id].sort_values('sequence_number')

            if len(debate_df) < 2:
                continue

            debate_data = {
                'id': debate_id,
                'title': debate_df.iloc[0]['title'],
                'year': year,
                'date': str(debate_df.iloc[0]['date'])[:10],
                'chamber': debate_df.iloc[0]['chamber'],
                'speech_count': len(debate_df),
                'has_female': True,
                'speeches': []
            }

            for _, speech in debate_df.head(30).iterrows():
                debate_data['speeches'].append({
                    'seq': int(speech['sequence_number']),
                    'speaker': speech.get('canonical_name') or speech.get('speaker', ''),
                    'gender': speech.get('gender', ''),
                    'party': speech.get('party', ''),
                    'text': str(speech.get('text', ''))[:3000],
                })

            all_debates.append(debate_data)

    # Write output
    static_dir = output_dir / 'static-data'
    static_dir.mkdir(parents=True, exist_ok=True)

    with open(static_dir / 'example_debates.json', 'w') as f:
        json.dump(all_debates, f, indent=2)

    print(f"  Exported {len(all_debates)} example debates")
    return all_debates


def build_pipeline_examples(output_dir: Path, count: int = 10):
    """Build pipeline transformation examples."""
    print("Building pipeline examples...")

    speeches_dir = project_root / 'data-hansard/derived_complete_v2/speeches_complete'
    if not speeches_dir.exists():
        speeches_dir = project_root / 'data-hansard/derived_complete/speeches_complete'

    examples = []

    # Sample from different years
    for year in [1920, 1945, 1980, 2000]:
        year_file = speeches_dir / f'speeches_{year}.parquet'
        if not year_file.exists():
            continue

        df = pd.read_parquet(year_file)
        female = df[df['gender'] == 'F']

        if len(female) == 0:
            continue

        sample = female.sample(n=min(count // 4, len(female)), random_state=42)

        for _, row in sample.iterrows():
            example = {
                'debate_id': row['debate_id'],
                'year': year,
                'file_path': row.get('file_path', ''),
                'title': row.get('title', ''),
                'speaker_raw': row.get('speaker', ''),
                'speaker_matched': row.get('canonical_name', ''),
                'gender': row.get('gender', ''),
                'party': row.get('party', ''),
                'matched': bool(row.get('matched_mp', False)),
            }
            examples.append(example)

    static_dir = output_dir / 'static-data'
    static_dir.mkdir(parents=True, exist_ok=True)

    with open(static_dir / 'pipeline_examples.json', 'w') as f:
        json.dump(examples, f, indent=2)

    print(f"  Exported {len(examples)} pipeline examples")
    return examples


def main():
    output_dir = Path(__file__).parent

    print("=" * 60)
    print("BUILDING STATIC DATA FOR FAST OFFLINE APPS")
    print("=" * 60)

    build_search_index(output_dir)
    build_stats_data(output_dir)
    build_example_debates(output_dir)
    build_pipeline_examples(output_dir)

    print("\n" + "=" * 60)
    print("BUILD COMPLETE - Data ready in apps/static-data/")
    print("=" * 60)


if __name__ == '__main__':
    main()
