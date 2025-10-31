#!/usr/bin/env python3
"""
SAFE cleanup of duplicate debate HTML files.

Strategy:
1. Only touch dates that were re-crawled
2. For each date, identify old vs new files by timestamp
3. Delete ONLY old files where newer version exists
4. Extensive safety checks and dry-run mode
5. Creates backup list before any deletion
"""

import json
from pathlib import Path
from datetime import datetime
import argparse
from collections import defaultdict

class SafeDebateCleaner:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run

        self.stats = {
            'dates_checked': 0,
            'files_checked': 0,
            'topics_checked': 0,
            'duplicate_topics': 0,
            'files_to_delete': 0,
            'space_to_free_mb': 0
        }

        self.to_delete = []
        self.warnings = []

    def load_recrawled_dates(self):
        """Load list of dates that were re-crawled."""
        # Check if round 2 is running
        round2_file = Path("analysis/dates_to_recrawl_round2.txt")
        if round2_file.exists():
            print("Note: Round 2 in progress - using round 1 list")

        # Use the original re-crawl checkpoint
        checkpoint_file = Path("analysis/recrawl_checkpoint.json")
        if not checkpoint_file.exists():
            print("Error: No checkpoint file found")
            return set()

        with open(checkpoint_file) as f:
            data = json.load(f)
            return set(data.get('completed', []))

    def analyze_date(self, date_path: str, data_dir: Path):
        """Analyze one date for duplicates using topic-based deduplication."""
        parts = date_path.split('/')
        if len(parts) != 3:
            return

        year, month, day = parts
        month_dir = data_dir / year / month

        if not month_dir.exists():
            return

        self.stats['dates_checked'] += 1

        # Get all HTML files for this day
        debate_files = list(month_dir.glob(f"{day}_*.html.gz"))

        if not debate_files:
            return

        self.stats['files_checked'] += len(debate_files)

        # Group files by topic
        # Format: day_XX_topic.html.gz
        # We want to group by topic, regardless of the XX index
        files_by_topic = defaultdict(list)

        for f in debate_files:
            parts = f.name.split('_')
            if len(parts) >= 3:
                # Topic is everything after day_XX_
                topic = '_'.join(parts[2:])
                files_by_topic[topic].append(f)

        self.stats['topics_checked'] += len(files_by_topic)

        # For each topic that appears multiple times, keep only newest
        for topic, files in files_by_topic.items():
            if len(files) > 1:
                # Found duplicate topic!
                self.stats['duplicate_topics'] += 1

                # Sort by modification time (newest first)
                files_sorted = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)

                # Keep the first (newest), delete the rest
                to_keep = files_sorted[0]
                to_delete = files_sorted[1:]

                for old_file in to_delete:
                    self.to_delete.append({
                        'file': old_file,
                        'date': date_path,
                        'topic': topic,
                        'size': old_file.stat().st_size,
                        'timestamp': datetime.fromtimestamp(old_file.stat().st_mtime),
                        'keeping_newer': to_keep.name
                    })
                    self.stats['files_to_delete'] += 1
                    self.stats['space_to_free_mb'] += old_file.stat().st_size / 1024 / 1024

    def run_analysis(self):
        """Analyze all re-crawled dates."""
        print("="*70)
        print("SAFE DUPLICATE CLEANUP ANALYSIS")
        print("="*70)
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE DELETION'}")
        print("Strategy: Keep newest file for each topic, delete older versions")
        print()

        # Load re-crawled dates
        print("Loading list of re-crawled dates...")
        recrawled = self.load_recrawled_dates()
        print(f"  Found {len(recrawled):,} re-crawled dates")
        print()

        # Analyze each date
        data_dir = Path("data-hansard/hansard")

        print("Analyzing dates for duplicates...")
        print("(This may take 1-2 minutes)")
        print()

        for i, date_path in enumerate(recrawled, 1):
            self.analyze_date(date_path, data_dir)

            if i % 1000 == 0:
                print(f"  Progress: {i:,}/{len(recrawled):,} dates analyzed...")

        print()
        self.print_results()

    def print_results(self):
        """Print analysis results."""
        print("="*70)
        print("ANALYSIS RESULTS")
        print("="*70)
        print(f"Dates checked: {self.stats['dates_checked']:,}")
        print(f"Files checked: {self.stats['files_checked']:,}")
        print(f"Unique topics: {self.stats['topics_checked']:,}")
        print(f"Duplicate topics: {self.stats['duplicate_topics']:,}")
        print(f"Files to delete: {self.stats['files_to_delete']:,}")
        print(f"Space to free: {self.stats['space_to_free_mb']:,.1f} MB ({self.stats['space_to_free_mb']/1024:.1f} GB)")
        print()

        if self.warnings:
            print(f"Warnings: {len(self.warnings)}")
            for w in self.warnings[:5]:
                print(f"  {w}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more")
            print()

        if self.to_delete:
            print("Sample files to delete (first 10):")
            for item in self.to_delete[:10]:
                print(f"  DELETE: {item['file'].name}")
                print(f"    Timestamp: {item['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                print(f"    Keeping: {item['keeping_newer']}")
                print()

            if len(self.to_delete) > 10:
                print(f"  ... and {len(self.to_delete) - 10:,} more")

        # Save deletion list
        deletion_list_file = Path("analysis/files_to_delete.json")
        with open(deletion_list_file, 'w') as f:
            json.dump({
                'created': datetime.now().isoformat(),
                'stats': self.stats,
                'files_to_delete': [
                    {
                        'path': str(item['file']),
                        'date': item['date'],
                        'topic': item['topic'],
                        'size': item['size'],
                        'timestamp': item['timestamp'].isoformat(),
                        'keeping_newer': item['keeping_newer']
                    }
                    for item in self.to_delete
                ],
                'warnings': self.warnings
            }, f, indent=2)

        print()
        print(f"Deletion list saved to: {deletion_list_file}")

    def execute_deletion(self):
        """Actually delete the files (ONLY if not dry_run)."""
        if self.dry_run:
            print()
            print("="*70)
            print("DRY RUN MODE - No files were deleted")
            print("="*70)
            print()
            print("To actually delete files, run with: --execute")
            return

        print()
        print("="*70)
        print("EXECUTING DELETION")
        print("="*70)
        print(f"About to delete {len(self.to_delete):,} files")
        print(f"This will free {self.stats['space_to_free_mb']/1024:.1f} GB")
        print()

        response = input("Type 'DELETE' to confirm: ")
        if response != 'DELETE':
            print("Cancelled.")
            return

        deleted_count = 0
        error_count = 0

        for item in self.to_delete:
            try:
                item['file'].unlink()
                deleted_count += 1

                if deleted_count % 1000 == 0:
                    print(f"  Deleted {deleted_count:,}/{len(self.to_delete):,}...")
            except Exception as e:
                error_count += 1
                print(f"  Error deleting {item['file'].name}: {e}")

        print()
        print(f"✓ Deleted {deleted_count:,} files")
        if error_count > 0:
            print(f"✗ Errors: {error_count}")
        print(f"✓ Freed {self.stats['space_to_free_mb']/1024:.1f} GB")

def main():
    parser = argparse.ArgumentParser(description='Safely clean up duplicate debate files')
    parser.add_argument('--execute', action='store_true',
                       help='Actually delete files (default is dry-run)')

    args = parser.parse_args()

    cleaner = SafeDebateCleaner(dry_run=not args.execute)
    cleaner.run_analysis()
    cleaner.execute_deletion()

if __name__ == "__main__":
    main()
