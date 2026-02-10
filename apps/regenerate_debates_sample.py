#!/usr/bin/env python3
"""Regenerate debates_sample.json with debates that have gender data."""

import sqlite3
import json
import random

def main():
    conn = sqlite3.connect('static-data/hansard_fts.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get debates with female speakers (years 1920+)
    cursor.execute('''
        SELECT debate_id, title, date, year, chamber,
               COUNT(*) as speech_count,
               SUM(CASE WHEN gender = 'M' THEN 1 ELSE 0 END) as male_count,
               SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) as female_count
        FROM speeches
        WHERE year >= 1920
        GROUP BY debate_id
        HAVING female_count > 0 AND speech_count BETWEEN 3 AND 40
        ORDER BY RANDOM()
        LIMIT 200
    ''')
    debates_with_female = [dict(row) for row in cursor.fetchall()]
    print(f"Found {len(debates_with_female)} debates with female speakers")

    # Get debates with male speakers only for balance
    cursor.execute('''
        SELECT debate_id, title, date, year, chamber,
               COUNT(*) as speech_count,
               SUM(CASE WHEN gender = 'M' THEN 1 ELSE 0 END) as male_count,
               SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) as female_count
        FROM speeches
        WHERE year >= 1920
        GROUP BY debate_id
        HAVING female_count = 0 AND male_count > 0 AND speech_count BETWEEN 3 AND 40
        ORDER BY RANDOM()
        LIMIT 300
    ''')
    debates_male_only = [dict(row) for row in cursor.fetchall()]
    print(f"Found {len(debates_male_only)} debates with male speakers only")

    # Combine and shuffle
    all_debates = debates_with_female + debates_male_only
    random.shuffle(all_debates)
    all_debates = all_debates[:500]

    # Get speeches for each debate
    result = []
    for i, debate in enumerate(all_debates):
        if (i + 1) % 50 == 0:
            print(f"Processing debate {i + 1}/{len(all_debates)}")

        cursor.execute('''
            SELECT speaker, canonical_name, gender, '' as party, text
            FROM speeches
            WHERE debate_id = ?
            ORDER BY id
        ''', (debate['debate_id'],))

        speeches = []
        for j, row in enumerate(cursor.fetchall()):
            speech = dict(row)
            speech['sequence_number'] = j + 1
            speeches.append(speech)

        result.append({
            'debate_id': debate['debate_id'],
            'title': debate['title'],
            'date': debate['date'],
            'year': debate['year'],
            'chamber': debate['chamber'],
            'speech_count': len(speeches),
            'male_mps': debate['male_count'],
            'female_mps': debate['female_count'],
            'speeches': speeches
        })

    # Sort by year descending
    result.sort(key=lambda x: x['year'], reverse=True)

    # Write to file
    with open('debate-viewer/data/debates_sample.json', 'w') as f:
        json.dump(result, f, indent=2)

    # Summary
    total = len(result)
    with_female = sum(1 for d in result if d['female_mps'] > 0)
    with_male = sum(1 for d in result if d['male_mps'] > 0)

    print(f"\nGenerated {total} debates")
    print(f"  With female speakers: {with_female}")
    print(f"  With male speakers: {with_male}")
    print(f"  Year range: {min(d['year'] for d in result)} - {max(d['year'] for d in result)}")

    # Verify gender data in speeches
    sample = result[0]
    genders = [s.get('gender') for s in sample['speeches']]
    print(f"\nSample debate: {sample['title']}")
    print(f"  Speech genders: M={genders.count('M')}, F={genders.count('F')}, empty={genders.count('')}")

    conn.close()

if __name__ == '__main__':
    main()
