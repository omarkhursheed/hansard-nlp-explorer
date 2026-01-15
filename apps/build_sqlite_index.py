#!/usr/bin/env python3
"""
Build SQLite database - PURE METADATA, no FTS.
Minimal size for fast filtering across ALL 6.8M speeches.

Target size: ~300-400MB
"""

import sqlite3
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent


def build_sqlite_metadata(output_dir: Path):
    """Build pure metadata SQLite database (no FTS)."""
    print("Building PURE metadata SQLite (no FTS)...")

    speeches_dir = project_root / 'data-hansard/derived_v2/speeches_complete'
    if not speeches_dir.exists():
        speeches_dir = project_root / 'data-hansard/derived_complete/speeches_complete'
        print(f"  Using fallback: {speeches_dir}")

    db_path = output_dir / 'static-data' / 'hansard.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute('PRAGMA journal_mode = OFF')
    cursor.execute('PRAGMA synchronous = OFF')
    cursor.execute('PRAGMA cache_size = 100000')
    cursor.execute('PRAGMA temp_store = MEMORY')

    # Pure metadata table - no FTS
    cursor.execute('''
        CREATE TABLE speeches (
            rowid INTEGER PRIMARY KEY,
            year INTEGER NOT NULL,
            date TEXT,
            speaker TEXT,
            gender TEXT,
            chamber TEXT,
            title TEXT,
            word_count INTEGER
        )
    ''')

    total_speeches = 0
    batch = []
    year_files = sorted(speeches_dir.glob('speeches_*.parquet'))

    for i, year_file in enumerate(year_files):
        year = int(year_file.stem.split('_')[1])
        print(f"  {year} ({i+1}/{len(year_files)})", end='\r')

        df = pd.read_parquet(year_file)

        for _, row in df.iterrows():
            speaker = row.get('canonical_name') or row.get('speaker', '')
            word_count = int(row.get('word_count', 0)) if pd.notna(row.get('word_count')) else 0

            batch.append((
                year,
                str(row.get('date', ''))[:10],
                speaker[:60],
                row.get('gender', ''),
                row.get('chamber', ''),
                str(row.get('title', ''))[:80],
                word_count
            ))

            total_speeches += 1

            if len(batch) >= 100000:
                cursor.executemany('INSERT INTO speeches(year, date, speaker, gender, chamber, title, word_count) VALUES (?, ?, ?, ?, ?, ?, ?)', batch)
                batch = []
                conn.commit()

    if batch:
        cursor.executemany('INSERT INTO speeches(year, date, speaker, gender, chamber, title, word_count) VALUES (?, ?, ?, ?, ?, ?, ?)', batch)
    conn.commit()

    print(f"\n  {total_speeches:,} speeches inserted")

    # Create indexes AFTER data load (faster)
    print("  Creating indexes...")
    cursor.execute('CREATE INDEX idx_year ON speeches(year)')
    cursor.execute('CREATE INDEX idx_gender ON speeches(gender)')
    cursor.execute('CREATE INDEX idx_chamber ON speeches(chamber)')
    cursor.execute('CREATE INDEX idx_yg ON speeches(year, gender)')
    conn.commit()

    print("  Vacuuming...")
    cursor.execute('VACUUM')
    conn.commit()

    cursor.execute('SELECT COUNT(*) FROM speeches')
    count = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM speeches WHERE gender="F"')
    female = cursor.fetchone()[0]

    conn.close()

    size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.0f} MB")
    print(f"  Total: {count:,} | Female: {female:,}")

    return {'size_mb': size_mb, 'speeches': count}


if __name__ == '__main__':
    print("=" * 50)
    result = build_sqlite_metadata(Path(__file__).parent)
    print(f"DONE: {result['size_mb']:.0f} MB")
