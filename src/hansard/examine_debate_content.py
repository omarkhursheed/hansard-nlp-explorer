#!/usr/bin/env python3
"""
Examine actual debate content to understand turn structure
"""

import pandas as pd
import json
import gzip
from pathlib import Path

def examine_debate_content():
    """Look at actual debate content files"""

    # Find a sample content file
    content_dir = Path("src/hansard/data/processed_fixed/content/")

    if not content_dir.exists():
        print(f"Content directory not found: {content_dir}")
        # Try alternate path
        content_dir = Path("src/hansard/data/processed/content/")

    if not content_dir.exists():
        print(f"Content directory not found: {content_dir}")
        return None

    # Get a year directory
    year_dirs = sorted([d for d in content_dir.iterdir() if d.is_dir()])
    if not year_dirs:
        print("No year directories found")
        return None

    # Use 1950 if available, otherwise middle year
    year_1950 = content_dir / "1950"
    if year_1950.exists():
        year_dir = year_1950
    else:
        year_dir = year_dirs[len(year_dirs)//2]

    print(f"Examining year: {year_dir.name}")

    # Find debate files
    debate_files = list(year_dir.glob("*.jsonl*"))
    if not debate_files:
        print(f"No debate files in {year_dir}")
        return None

    debate_file = debate_files[0]
    print(f"Reading {debate_file}")

    # Read based on extension
    if debate_file.suffix == '.gz':
        with gzip.open(debate_file, 'rt') as f:
            lines = [json.loads(line) for line in f.readlines()[:5]]  # First 5 debates
    else:
        with open(debate_file, 'r') as f:
            lines = [json.loads(line) for line in f.readlines()[:5]]  # First 5 debates

    print(f"\nFound {len(lines)} debates in sample")

    for i, debate in enumerate(lines[:2]):  # Examine first 2 debates
        print(f"\n=== DEBATE {i+1} ===")
        print(f"Keys: {list(debate.keys())}")

        if 'metadata' in debate:
            meta = debate['metadata']
            print(f"Title: {meta.get('title', 'N/A')[:100]}")
            print(f"Speakers: {meta.get('speakers', [])[:5]}")
            print(f"Date: {meta.get('reference_date', 'N/A')}")
            print(f"Chamber: {meta.get('chamber', 'N/A')}")

        if 'full_text' in debate:
            text = debate['full_text']
            print(f"Text length: {len(text)} chars")
            print(f"\nFirst 500 chars of text:")
            print(text[:500])

            # Look for turn indicators
            print(f"\n=== TURN STRUCTURE ===")
            lines = text.split('\n')[:50]  # First 50 lines

            current_speaker = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Common patterns for speaker changes
                # Pattern 1: "Mr. Smith:" or "The Minister:"
                if ':' in line and len(line.split(':')[0]) < 50:
                    potential_speaker = line.split(':')[0].strip()
                    if potential_speaker != current_speaker:
                        print(f"Speaker change: {potential_speaker[:50]}")
                        current_speaker = potential_speaker

                # Pattern 2: All caps line (sometimes used for speakers)
                elif line.isupper() and len(line) < 50:
                    if line != current_speaker:
                        print(f"Speaker (CAPS): {line}")
                        current_speaker = line

    return lines

if __name__ == "__main__":
    debates = examine_debate_content()