#!/usr/bin/env python3
"""
Build SQLite database with FTS5 full-text search for all Hansard speeches.

This creates a database with:
1. A main speeches table with metadata
2. An FTS5 virtual table for full-text search of speech content

The resulting database enables fast text search across all 6.8M speeches.
"""

import sqlite3
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data-hansard' / 'derived_v2' / 'speeches_complete'
OUTPUT_DB = Path(__file__).parent / 'static-data' / 'hansard_fts.db'

# Fallback to derived_complete if derived_v2 doesn't exist
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / 'data-hansard' / 'derived_complete' / 'speeches_complete'


def create_database():
    """Create the SQLite database with FTS5."""

    if OUTPUT_DB.exists():
        print(f"Removing existing database: {OUTPUT_DB}")
        OUTPUT_DB.unlink()

    conn = sqlite3.connect(str(OUTPUT_DB))
    cursor = conn.cursor()

    # Create main speeches table
    cursor.execute('''
        CREATE TABLE speeches (
            id INTEGER PRIMARY KEY,
            speech_id TEXT,
            debate_id TEXT,
            year INTEGER NOT NULL,
            date TEXT,
            speaker TEXT,
            canonical_name TEXT,
            gender TEXT,
            chamber TEXT,
            title TEXT,
            topic TEXT,
            word_count INTEGER,
            text TEXT
        )
    ''')

    # Create FTS5 virtual table for full-text search
    # content='' makes it a contentless FTS table (stores only index, not text)
    # We'll store the text in the main table and join on rowid
    cursor.execute('''
        CREATE VIRTUAL TABLE speeches_fts USING fts5(
            text,
            speaker,
            title,
            content='speeches',
            content_rowid='id'
        )
    ''')

    # Create indexes
    cursor.execute('CREATE INDEX idx_year ON speeches(year)')
    cursor.execute('CREATE INDEX idx_gender ON speeches(gender)')
    cursor.execute('CREATE INDEX idx_chamber ON speeches(chamber)')
    cursor.execute('CREATE INDEX idx_speaker ON speeches(speaker)')

    conn.commit()
    return conn


def load_parquet_files(conn):
    """Load all parquet files into the database."""

    cursor = conn.cursor()
    files = sorted(DATA_DIR.glob('speeches_*.parquet'))

    print(f"Found {len(files)} parquet files")

    total_rows = 0
    batch_size = 10000

    for i, file_path in enumerate(files):
        year = int(file_path.stem.split('_')[1])
        print(f"[{i+1}/{len(files)}] Loading {file_path.name}...", end=' ', flush=True)

        df = pd.read_parquet(file_path)

        # Select and rename columns
        df_export = df[[
            'speech_id', 'debate_id', 'year', 'date', 'speaker',
            'canonical_name', 'gender', 'chamber', 'title', 'topic',
            'word_count', 'text'
        ]].copy()

        # Clean data
        df_export = df_export.fillna('')
        df_export['year'] = df_export['year'].astype(int)
        df_export['word_count'] = df_export['word_count'].fillna(0).astype(int)

        # Insert in batches
        rows = df_export.values.tolist()

        for batch_start in range(0, len(rows), batch_size):
            batch = rows[batch_start:batch_start + batch_size]
            cursor.executemany('''
                INSERT INTO speeches (
                    speech_id, debate_id, year, date, speaker,
                    canonical_name, gender, chamber, title, topic,
                    word_count, text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', batch)

        conn.commit()
        total_rows += len(df)
        print(f"{len(df):,} rows (total: {total_rows:,})")

    return total_rows


def rebuild_fts_index(conn):
    """Rebuild the FTS5 index from the main table."""

    print("\nRebuilding FTS5 index...")
    cursor = conn.cursor()

    # Populate FTS index
    cursor.execute('''
        INSERT INTO speeches_fts(speeches_fts) VALUES('rebuild')
    ''')

    conn.commit()
    print("FTS5 index rebuilt successfully")


def optimize_database(conn):
    """Optimize the database for read performance."""

    print("\nOptimizing database...")
    cursor = conn.cursor()

    # Optimize FTS index
    cursor.execute("INSERT INTO speeches_fts(speeches_fts) VALUES('optimize')")
    conn.commit()

    # Analyze for query optimization
    cursor.execute("ANALYZE")
    conn.commit()

    # Vacuum requires isolation_level=None (autocommit mode)
    conn.isolation_level = None
    cursor.execute("VACUUM")
    conn.isolation_level = ''  # Reset to default

    print("Database optimized")


def verify_database(conn):
    """Verify the database was created correctly."""

    print("\n" + "=" * 60)
    print("DATABASE VERIFICATION")
    print("=" * 60)

    cursor = conn.cursor()

    # Count rows
    cursor.execute("SELECT COUNT(*) FROM speeches")
    total = cursor.fetchone()[0]
    print(f"Total speeches: {total:,}")

    # Count by gender
    cursor.execute("SELECT gender, COUNT(*) FROM speeches GROUP BY gender")
    for gender, count in cursor.fetchall():
        print(f"  {gender or 'Unknown'}: {count:,}")

    # Year range
    cursor.execute("SELECT MIN(year), MAX(year) FROM speeches")
    min_year, max_year = cursor.fetchone()
    print(f"Year range: {min_year} - {max_year}")

    # Test FTS search
    print("\nTesting FTS5 search...")
    cursor.execute('''
        SELECT s.id, s.speaker, s.year, snippet(speeches_fts, 0, '<b>', '</b>', '...', 20)
        FROM speeches_fts
        JOIN speeches s ON s.id = speeches_fts.rowid
        WHERE speeches_fts MATCH 'women suffrage'
        LIMIT 5
    ''')

    results = cursor.fetchall()
    print(f"Found {len(results)} results for 'women suffrage':")
    for row in results:
        print(f"  [{row[2]}] {row[1]}: {row[3][:80]}...")

    # Database size
    db_size = OUTPUT_DB.stat().st_size / (1024 * 1024 * 1024)
    print(f"\nDatabase size: {db_size:.2f} GB")


def main():
    print("=" * 60)
    print("BUILDING FTS5 HANSARD DATABASE")
    print("=" * 60)
    print(f"Source: {DATA_DIR}")
    print(f"Output: {OUTPUT_DB}")
    print()

    if not DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        sys.exit(1)

    # Create database
    conn = create_database()
    print("Database created with FTS5 support\n")

    # Load data
    total = load_parquet_files(conn)

    # Rebuild FTS index
    rebuild_fts_index(conn)

    # Optimize
    optimize_database(conn)

    # Verify
    verify_database(conn)

    conn.close()

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"Database: {OUTPUT_DB}")
    print(f"Total speeches: {total:,}")


if __name__ == '__main__':
    main()
