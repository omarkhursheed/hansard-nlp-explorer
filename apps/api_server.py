#!/usr/bin/env python3
"""
Simple API server for the Hansard data explorer apps.
Provides endpoints for searching and tracing debates/speeches.

Now includes FTS (Full-Text Search) using SQLite FTS5 for fast text search
across all 6.8M speeches.
"""

import json
import sqlite3
import socket
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import gzip
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class ReusableHTTPServer(HTTPServer):
    """HTTPServer that allows address reuse to avoid 'Address already in use' errors."""
    allow_reuse_address = True

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()

# Change to apps directory
os.chdir(Path(__file__).parent)
project_root = Path(__file__).resolve().parent.parent

# FTS database path
FTS_DB_PATH = Path(__file__).parent / 'static-data' / 'hansard_fts.db'


def get_fts_db():
    """Get FTS database connection."""
    if not FTS_DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(FTS_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


class HansardAPIHandler(SimpleHTTPRequestHandler):
    """Handler that serves static files and API endpoints."""

    def do_GET(self):
        parsed = urlparse(self.path)

        # API endpoints
        if parsed.path == '/api/trace':
            self.handle_trace(parse_qs(parsed.query))
        elif parsed.path == '/api/search':
            self.handle_search(parse_qs(parsed.query))
        elif parsed.path == '/api/fts':
            self.handle_fts_search(parse_qs(parsed.query))
        elif parsed.path == '/api/speech':
            self.handle_speech(parse_qs(parsed.query))
        elif parsed.path == '/api/stats':
            self.handle_stats()
        elif parsed.path == '/api/count':
            self.handle_count(parse_qs(parsed.query))
        elif parsed.path == '/api/random':
            self.handle_random(parse_qs(parsed.query))
        elif parsed.path == '/api/speaker':
            self.handle_speaker(parse_qs(parsed.query))
        elif parsed.path == '/api/debate':
            self.handle_debate(parse_qs(parsed.query))
        else:
            # Serve static files
            super().do_GET()

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def handle_trace(self, params):
        """Trace a debate through all pipeline stages."""
        debate_id = params.get('id', [None])[0]
        year = params.get('year', [None])[0]

        if not debate_id:
            self.send_json({'error': 'Missing id parameter'}, 400)
            return

        year = int(year) if year else None
        result = trace_debate(debate_id, year)
        self.send_json(result)

    def handle_search(self, params):
        """Search for debates by title, speaker, year, and optionally content."""
        query = params.get('q', [''])[0].lower()
        year_from = params.get('year_from', [None])[0]
        year_to = params.get('year_to', [None])[0]
        has_female = params.get('has_female', [None])[0]
        search_content = params.get('search_content', ['false'])[0]
        gender = params.get('gender', [None])[0]
        chamber = params.get('chamber', [None])[0]
        limit = int(params.get('limit', ['500'])[0])

        results = search_debates(
            query=query,
            year_from=int(year_from) if year_from else None,
            year_to=int(year_to) if year_to else None,
            has_female=has_female == 'true' if has_female else None,
            limit=limit,
            search_content=search_content == 'true',
            gender=gender.upper() if gender else None,
            chamber=chamber if chamber else None
        )
        self.send_json(results)

    def handle_speech(self, params):
        """Get a specific speech by ID."""
        speech_id = params.get('id', [None])[0]
        if not speech_id:
            self.send_json({'error': 'Missing id parameter'}, 400)
            return

        result = get_speech(speech_id)
        self.send_json(result)

    def handle_stats(self):
        """Get pipeline statistics."""
        stats_file = Path('pipeline-visualizer/data/pipeline_stats.json')
        if stats_file.exists():
            with open(stats_file) as f:
                self.send_json(json.load(f))
        else:
            self.send_json({'error': 'Stats not found'}, 404)

    def handle_count(self, params):
        """Count speeches matching query (fast estimate)."""
        query = params.get('q', [''])[0].lower()
        year_from = params.get('year_from', [None])[0]
        year_to = params.get('year_to', [None])[0]
        gender = params.get('gender', [None])[0]
        chamber = params.get('chamber', [None])[0]

        count = count_speeches(
            query=query,
            year_from=int(year_from) if year_from else None,
            year_to=int(year_to) if year_to else None,
            gender=gender.upper() if gender else None,
            chamber=chamber
        )
        self.send_json({'count': count, 'query': query})

    def handle_fts_search(self, params):
        """Full-text search using SQLite FTS5 (fast)."""
        query = params.get('q', [''])[0]
        limit = min(int(params.get('limit', ['100'])[0]), 500)
        gender = params.get('gender', [None])[0]
        year_from = params.get('year_from', [None])[0]
        year_to = params.get('year_to', [None])[0]

        if not query:
            self.send_json({'error': 'Missing query parameter q'}, 400)
            return

        conn = get_fts_db()
        if not conn:
            self.send_json({'error': 'FTS database not available. Run build_fts_database.py'}, 503)
            return

        try:
            cursor = conn.cursor()

            # Build query with filters
            where_clauses = []
            sql_params = [query]

            if gender:
                where_clauses.append('s.gender = ?')
                sql_params.append(gender.upper())

            if year_from:
                where_clauses.append('s.year >= ?')
                sql_params.append(int(year_from))

            if year_to:
                where_clauses.append('s.year <= ?')
                sql_params.append(int(year_to))

            where_sql = ' AND '.join(where_clauses) if where_clauses else ''
            if where_sql:
                where_sql = ' AND ' + where_sql

            sql = f'''
                SELECT s.id, s.speech_id, s.speaker, s.canonical_name, s.gender,
                       s.year, s.date, s.chamber, s.title, s.topic, s.word_count,
                       snippet(speeches_fts, 0, '<mark>', '</mark>', '...', 30) as snippet
                FROM speeches_fts
                JOIN speeches s ON s.id = speeches_fts.rowid
                WHERE speeches_fts MATCH ?
                {where_sql}
                ORDER BY bm25(speeches_fts)
                LIMIT ?
            '''

            sql_params.append(limit)
            cursor.execute(sql, sql_params)
            results = [dict(row) for row in cursor.fetchall()]

            # Get total count
            count_sql = f'''
                SELECT COUNT(*)
                FROM speeches_fts
                JOIN speeches s ON s.id = speeches_fts.rowid
                WHERE speeches_fts MATCH ?
                {where_sql}
            '''
            cursor.execute(count_sql, sql_params[:-1])  # exclude limit
            total = cursor.fetchone()[0]

            conn.close()

            self.send_json({
                'query': query,
                'total': total,
                'limit': limit,
                'results': results
            })

        except Exception as e:
            conn.close()
            self.send_json({'error': str(e)}, 500)

    def handle_random(self, params):
        """Get random speeches from FTS database."""
        limit = min(int(params.get('limit', ['50'])[0]), 500)
        gender = params.get('gender', [None])[0]

        conn = get_fts_db()
        if not conn:
            self.send_json({'error': 'FTS database not available'}, 503)
            return

        try:
            cursor = conn.cursor()

            if gender:
                sql = '''
                    SELECT id, speech_id, speaker, canonical_name, gender,
                           year, date, chamber, title, topic, word_count, text
                    FROM speeches
                    WHERE gender = ?
                    ORDER BY RANDOM()
                    LIMIT ?
                '''
                cursor.execute(sql, [gender.upper(), limit])
            else:
                sql = '''
                    SELECT id, speech_id, speaker, canonical_name, gender,
                           year, date, chamber, title, topic, word_count, text
                    FROM speeches
                    ORDER BY RANDOM()
                    LIMIT ?
                '''
                cursor.execute(sql, [limit])

            results = [dict(row) for row in cursor.fetchall()]
            conn.close()

            self.send_json({'results': results, 'count': len(results)})

        except Exception as e:
            conn.close()
            self.send_json({'error': str(e)}, 500)

    def handle_speaker(self, params):
        """Get all speeches by a speaker with links to debates."""
        speaker = params.get('name', [None])[0]
        limit = min(int(params.get('limit', ['100'])[0]), 500)

        if not speaker:
            self.send_json({'error': 'Missing name parameter'}, 400)
            return

        conn = get_fts_db()
        if not conn:
            self.send_json({'error': 'FTS database not available'}, 503)
            return

        try:
            cursor = conn.cursor()

            # Get speeches by this speaker
            sql = '''
                SELECT id, speech_id, debate_id, speaker, canonical_name, gender,
                       year, date, chamber, title, topic, word_count,
                       substr(text, 1, 300) as excerpt
                FROM speeches
                WHERE speaker LIKE ? OR canonical_name LIKE ?
                ORDER BY year DESC, date DESC
                LIMIT ?
            '''
            pattern = f'%{speaker}%'
            cursor.execute(sql, [pattern, pattern, limit])
            speeches = [dict(row) for row in cursor.fetchall()]

            # Get summary stats
            cursor.execute('''
                SELECT COUNT(*) as total, MIN(year) as first_year, MAX(year) as last_year,
                       SUM(word_count) as total_words
                FROM speeches
                WHERE speaker LIKE ? OR canonical_name LIKE ?
            ''', [pattern, pattern])
            stats = dict(cursor.fetchone())

            # Get unique debates
            cursor.execute('''
                SELECT DISTINCT debate_id, title, year, date
                FROM speeches
                WHERE speaker LIKE ? OR canonical_name LIKE ?
                ORDER BY year DESC
                LIMIT 50
            ''', [pattern, pattern])
            debates = [dict(row) for row in cursor.fetchall()]

            conn.close()

            self.send_json({
                'speaker': speaker,
                'stats': stats,
                'speeches': speeches,
                'debates': debates
            })

        except Exception as e:
            conn.close()
            self.send_json({'error': str(e)}, 500)

    def handle_debate(self, params):
        """Get all speeches in a debate with speaker links."""
        debate_id = params.get('id', [None])[0]

        if not debate_id:
            self.send_json({'error': 'Missing id parameter'}, 400)
            return

        conn = get_fts_db()
        if not conn:
            self.send_json({'error': 'FTS database not available'}, 503)
            return

        try:
            cursor = conn.cursor()

            # Get all speeches in this debate
            sql = '''
                SELECT id, speech_id, debate_id, speaker, canonical_name, gender,
                       year, date, chamber, title, topic, word_count, text
                FROM speeches
                WHERE debate_id = ?
                ORDER BY id
            '''
            cursor.execute(sql, [debate_id])
            speeches = [dict(row) for row in cursor.fetchall()]

            if not speeches:
                conn.close()
                self.send_json({'error': 'Debate not found'}, 404)
                return

            # Get unique speakers in this debate
            speakers = {}
            for s in speeches:
                name = s.get('canonical_name') or s.get('speaker')
                if name and name not in speakers:
                    speakers[name] = {
                        'name': name,
                        'gender': s.get('gender'),
                        'speech_count': 0,
                        'word_count': 0
                    }
                if name:
                    speakers[name]['speech_count'] += 1
                    speakers[name]['word_count'] += s.get('word_count', 0)

            # Get debate metadata
            first = speeches[0]
            debate_info = {
                'debate_id': debate_id,
                'title': first.get('title'),
                'date': first.get('date'),
                'year': first.get('year'),
                'chamber': first.get('chamber'),
                'speech_count': len(speeches),
                'speakers': list(speakers.values()),
                'has_female': any(s.get('gender') == 'F' for s in speeches),
                'has_male': any(s.get('gender') == 'M' for s in speeches)
            }

            conn.close()

            self.send_json({
                'debate': debate_info,
                'speeches': speeches
            })

        except Exception as e:
            conn.close()
            self.send_json({'error': str(e)}, 500)


def get_raw_html(file_path: str) -> dict:
    """Get raw HTML for a debate."""
    full_path = project_root / file_path
    if full_path.exists():
        try:
            with gzip.open(full_path, 'rt', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                return {
                    'size_bytes': len(content),
                    'preview': content[:5000],
                    'has_contribution_tags': 'contribution' in content.lower(),
                }
        except Exception as e:
            return {'error': str(e)}
    return None


def get_processed_metadata(file_path: str, year: int) -> dict:
    """Get processed metadata for a debate."""
    metadata_file = project_root / f'data-hansard/processed_complete/metadata/debates_{year}.parquet'
    if metadata_file.exists():
        df = pd.read_parquet(metadata_file)
        matches = df[df['file_path'] == file_path]
        if len(matches) > 0:
            row = matches.iloc[0].to_dict()
            clean = {}
            for k, v in row.items():
                try:
                    if v is None or (hasattr(v, '__len__') and len(v) == 0):
                        clean[k] = None
                    elif isinstance(v, (list, tuple)):
                        clean[k] = list(v)[:20]
                    elif hasattr(v, 'isoformat'):
                        clean[k] = v.isoformat()
                    elif pd.isna(v):
                        clean[k] = None
                    else:
                        clean[k] = v
                except (ValueError, TypeError):
                    clean[k] = str(v) if v is not None else None
            return clean
    return None


def get_gender_enhanced(file_path: str, year: int) -> dict:
    """Get gender-enhanced data for a debate."""
    gender_file = project_root / f'data-hansard/gender_analysis_complete/debates_{year}_enhanced.parquet'
    if gender_file.exists():
        df = pd.read_parquet(gender_file)
        matches = df[df['file_path'] == file_path]
        if len(matches) > 0:
            row = matches.iloc[0].to_dict()
            clean = {}
            for k, v in row.items():
                try:
                    if v is None or (hasattr(v, '__len__') and len(v) == 0):
                        clean[k] = None
                    elif isinstance(v, (list, tuple)):
                        clean[k] = list(v)[:20]
                    elif hasattr(v, 'isoformat'):
                        clean[k] = v.isoformat()
                    elif pd.isna(v):
                        clean[k] = None
                    else:
                        clean[k] = v
                except (ValueError, TypeError):
                    clean[k] = str(v) if v is not None else None
            return clean
    return None


def trace_debate(debate_id: str, year: int = None) -> dict:
    """Trace a debate through all pipeline stages."""
    result = {
        'debate_id': debate_id,
        'stages': {
            'raw_html': None,
            'processed': None,
            'gender_enhanced': None,
            'final_speeches': []
        }
    }

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
                for _, row in matches.sort_values('sequence_number').iterrows():
                    speech = {}
                    for k, v in row.items():
                        if pd.isna(v):
                            speech[k] = None
                        elif k == 'text':
                            speech[k] = str(v) if v else ''
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
    result['title'] = result['stages']['final_speeches'][0].get('title') if result['stages']['final_speeches'] else None

    # Get raw HTML
    raw = get_raw_html(file_path)
    if raw:
        result['stages']['raw_html'] = raw

    # Get processed
    processed = get_processed_metadata(file_path, year)
    if processed:
        result['stages']['processed'] = processed

    # Get gender enhanced
    gender = get_gender_enhanced(file_path, year)
    if gender:
        result['stages']['gender_enhanced'] = gender

    return result


def search_debates(query: str = '', year_from: int = None, year_to: int = None,
                   has_female: bool = None, limit: int = 100, search_content: bool = False,
                   gender: str = None, chamber: str = None) -> list:
    """Search for debates matching criteria.

    Args:
        query: Search term for title, speaker, or content
        year_from: Start year filter
        year_to: End year filter
        has_female: Filter for debates with female speakers
        limit: Maximum results to return
        search_content: If True, also search speech text
        gender: Filter by speaker gender ('M' or 'F')
        chamber: Filter by chamber ('Commons' or 'Lords')
    """
    # Try v2 first, fallback to v1
    speeches_dir = project_root / 'data-hansard/derived_complete_v2/speeches_complete'
    if not speeches_dir.exists():
        speeches_dir = project_root / 'data-hansard/derived_complete/speeches_complete'

    results = []
    year_start = year_from or 1803
    year_end = year_to or 2005

    # For content search, search speeches directly (much faster)
    if search_content and query:
        return search_speech_content(query, year_start, year_end, gender, chamber, limit, speeches_dir)

    # For debate-level search (title/speaker matching)
    for year in range(year_start, year_end + 1):
        if len(results) >= limit:
            break

        speeches_file = speeches_dir / f'speeches_{year}.parquet'
        if not speeches_file.exists():
            continue

        df = pd.read_parquet(speeches_file)

        # Apply filters early for speed
        if chamber:
            df = df[df['chamber'] == chamber]
        if gender:
            df = df[df['gender'] == gender]

        # Group by debate
        for debate_id, group in df.groupby('debate_id'):
            if len(results) >= limit:
                break

            first = group.iloc[0]
            title = str(first['title'] or '')
            speakers = group['canonical_name'].dropna().unique().tolist()

            # Apply query filter on title/speakers only
            if query and query not in title.lower() and not any(query in s.lower() for s in speakers):
                continue

            debate_has_female = (group['gender'] == 'F').any()
            if has_female is not None and debate_has_female != has_female:
                continue

            results.append({
                'debate_id': debate_id,
                'year': year,
                'title': title,
                'date': str(first['date']),
                'chamber': first['chamber'],
                'file_path': first['file_path'],
                'speech_count': len(group),
                'has_female': bool(debate_has_female),
                'speakers': speakers[:10],
            })

    return results


def search_speech_content(query: str, year_from: int, year_to: int,
                          gender: str, chamber: str, limit: int,
                          speeches_dir: Path) -> list:
    """Fast speech content search using PyArrow predicate pushdown and parallelization."""
    query_lower = query.lower()
    results = []
    results_lock = threading.Lock()

    def search_year(year: int) -> list:
        """Search a single year's data."""
        speeches_file = speeches_dir / f'speeches_{year}.parquet'
        if not speeches_file.exists():
            return []

        try:
            # Read with PyArrow for predicate pushdown
            columns = ['debate_id', 'title', 'date', 'chamber', 'file_path',
                       'text', 'canonical_name', 'speaker', 'gender', 'sequence_number']
            table = pq.read_table(speeches_file, columns=columns)

            # Apply chamber/gender filters using PyArrow compute (pushdown)
            if chamber:
                mask = pc.equal(table.column('chamber'), chamber)
                table = table.filter(mask)
            if gender:
                mask = pc.equal(table.column('gender'), gender)
                table = table.filter(mask)

            if table.num_rows == 0:
                return []

            # PyArrow text search - case insensitive using match_substring
            text_col = table.column('text')
            # Fill nulls with empty string for search
            text_filled = pc.if_else(pc.is_null(text_col), '', text_col)
            text_lower_col = pc.utf8_lower(text_filled)
            match_mask = pc.match_substring(text_lower_col, query_lower)
            matches = table.filter(match_mask)

            if matches.num_rows == 0:
                return []

            # Convert matches to results
            year_results = []
            matches_df = matches.to_pandas()
            for _, row in matches_df.iterrows():
                text = row.get('text', '') or ''
                year_results.append({
                    'debate_id': row['debate_id'],
                    'year': year,
                    'title': str(row.get('title', '')),
                    'date': str(row.get('date', '')),
                    'chamber': row.get('chamber'),
                    'file_path': row.get('file_path'),
                    'speech_count': 1,
                    'has_female': row.get('gender') == 'F',
                    'speakers': [row.get('canonical_name') or row.get('speaker')],
                    'matching_speeches': [{
                        'sequence': row.get('sequence_number'),
                        'speaker': row.get('canonical_name') or row.get('speaker'),
                        'gender': row.get('gender'),
                        'excerpt': extract_excerpt(text, query, 150)
                    }]
                })
            return year_results
        except Exception as e:
            print(f"Error searching year {year}: {e}")
            return []

    # Parallel search across years
    years = list(range(year_from, year_to + 1))
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(search_year, year): year for year in years}
        for future in as_completed(futures):
            year_results = future.result()
            with results_lock:
                remaining = limit - len(results)
                if remaining > 0:
                    results.extend(year_results[:remaining])

    # Sort by year for consistent ordering
    results.sort(key=lambda x: (x['year'], x.get('debate_id', '')))
    return results[:limit]


def count_speeches(query: str, year_from: int = None, year_to: int = None,
                   gender: str = None, chamber: str = None) -> int:
    """Fast count of speeches matching query."""
    speeches_dir = project_root / 'data-hansard/derived_complete_v2/speeches_complete'
    if not speeches_dir.exists():
        speeches_dir = project_root / 'data-hansard/derived_complete/speeches_complete'

    total = 0
    year_start = year_from or 1803
    year_end = year_to or 2005
    query_lower = query.lower() if query else ''

    for year in range(year_start, year_end + 1):
        speeches_file = speeches_dir / f'speeches_{year}.parquet'
        if not speeches_file.exists():
            continue

        # Read minimal columns
        columns = ['text', 'gender', 'chamber']
        df = pd.read_parquet(speeches_file, columns=columns)

        if chamber:
            df = df[df['chamber'] == chamber]
        if gender:
            df = df[df['gender'] == gender]

        if query_lower:
            df['text_lower'] = df['text'].fillna('').str.lower()
            total += df['text_lower'].str.contains(query_lower, regex=False, na=False).sum()
        else:
            total += len(df)

    return int(total)


def extract_excerpt(text: str, query: str, context_chars: int = 100) -> str:
    """Extract a text excerpt around the query match."""
    idx = text.lower().find(query.lower())
    if idx == -1:
        return text[:context_chars * 2] + '...'

    start = max(0, idx - context_chars)
    end = min(len(text), idx + len(query) + context_chars)

    excerpt = text[start:end]
    if start > 0:
        excerpt = '...' + excerpt
    if end < len(text):
        excerpt = excerpt + '...'

    return excerpt


def get_speech(speech_id: str) -> dict:
    """Get a specific speech by ID."""
    # speech_id format: debate_id_speech_N
    parts = speech_id.rsplit('_speech_', 1)
    if len(parts) != 2:
        return {'error': 'Invalid speech_id format'}

    debate_id = parts[0]
    speech_num = int(parts[1])

    speeches_dir = project_root / 'data-hansard/derived_complete/speeches_complete'

    for year in range(1803, 2006):
        speeches_file = speeches_dir / f'speeches_{year}.parquet'
        if speeches_file.exists():
            df = pd.read_parquet(speeches_file)
            matches = df[(df['debate_id'] == debate_id) & (df['sequence_number'] == speech_num)]
            if len(matches) > 0:
                row = matches.iloc[0].to_dict()
                return {k: (None if pd.isna(v) else v) for k, v in row.items()}

    return {'error': 'Speech not found'}


def main():
    # Parse port from command line
    port = 5001  # Default port
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--port' and i < len(sys.argv):
            try:
                port = int(sys.argv[i + 1])
            except (ValueError, IndexError):
                pass
        elif arg.startswith('--port='):
            try:
                port = int(arg.split('=')[1])
            except ValueError:
                pass

    server = ReusableHTTPServer(('localhost', port), HansardAPIHandler)
    print("=" * 60)
    print("HANSARD API SERVER")
    print("=" * 60)
    print(f"Server: http://localhost:{port}")
    print()
    print("Endpoints:")
    print(f"  Static files:  http://localhost:{port}/")
    print(f"  FTS Search:    http://localhost:{port}/api/fts?q=women+vote")
    print(f"  Random:        http://localhost:{port}/api/random?limit=50&gender=F")
    print(f"  Speaker:       http://localhost:{port}/api/speaker?name=Thatcher")
    print(f"  Debate:        http://localhost:{port}/api/debate?id=DEBATE_ID")
    print(f"  Trace:         http://localhost:{port}/api/trace?id=DEBATE_ID")
    print(f"  Search:        http://localhost:{port}/api/search?q=suffrage")
    print(f"  Stats:         http://localhost:{port}/api/stats")
    print()
    print(f"FTS database: {'AVAILABLE' if FTS_DB_PATH.exists() else 'NOT FOUND (run build_fts_database.py)'}")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()
