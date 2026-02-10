#!/usr/bin/env python3
"""
Verification script to validate Hansard dataset statistics
Performs direct parquet file analysis to confirm all counts

Supports both v1 (legacy) and v2 (bug-fixed) datasets.
"""

import argparse
import pandas as pd
from pathlib import Path
import sys


def find_project_root():
    """Find the project root by looking for marker files."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / '.git').exists() or (current / 'CLAUDE.md').exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return Path(__file__).resolve().parents[2]


project_root = find_project_root()
sys.path.insert(0, str(project_root / 'src'))

from hansard.utils.path_config import Paths, DATASET_VERSION

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def get_versioned_paths(version):
    """Get paths for the specified dataset version."""
    if version == "v2":
        return {
            'derived': Paths.DERIVED_DATA,
            'gender': Paths.GENDER_ENHANCED_DATA,
            'processed': Paths.PROCESSED_DATA,
        }
    else:  # v1 / legacy
        return {
            'derived': Paths.DERIVED_V1,
            'gender': Paths.GENDER_V1,
            'processed': Paths.PROCESSED_V1,
        }


def verify_derived_complete(version):
    """Verify statistics from derived dataset (PRIMARY)"""
    print_section(f"DERIVED DATASET ({version.upper()}) - PRIMARY")

    paths = get_versioned_paths(version)
    data_dir = paths['derived']
    speeches_dir = data_dir / "speeches_complete"
    debates_dir = data_dir / "debates_complete"

    if not data_dir.exists():
        print(f"  WARNING: Directory not found: {data_dir}")
        print(f"  Skipping derived dataset verification for {version}")
        return None

    # Count speeches
    print("\nCounting speeches...")
    speech_files = sorted(speeches_dir.glob("speeches_*.parquet"))
    total_speeches = 0
    commons_speeches = 0
    lords_speeches = 0
    female_speeches = 0
    male_speeches = 0
    matched_speeches = 0
    unique_person_ids = set()
    female_person_ids = set()
    male_person_ids = set()

    for f in speech_files:
        df = pd.read_parquet(f)
        total_speeches += len(df)
        commons_speeches += (df['chamber'] == 'Commons').sum()
        lords_speeches += (df['chamber'] == 'Lords').sum()
        female_speeches += (df['gender'] == 'F').sum()
        male_speeches += (df['gender'] == 'M').sum()
        matched_speeches += df['matched_mp'].sum()

        # Collect unique person_ids
        person_ids = df[df['person_id'].notna()]['person_id'].unique()
        unique_person_ids.update(person_ids)

        female_ids = df[df['gender'] == 'F']['person_id'].dropna().unique()
        female_person_ids.update(female_ids)

        male_ids = df[df['gender'] == 'M']['person_id'].dropna().unique()
        male_person_ids.update(male_ids)

    print(f"\nSpeeches:")
    print(f"  Total speeches: {total_speeches:,}")
    print(f"  Commons speeches: {commons_speeches:,} ({100*commons_speeches/total_speeches:.1f}%)")
    print(f"  Lords speeches: {lords_speeches:,} ({100*lords_speeches/total_speeches:.1f}%)")
    print(f"  Gender-matched: {matched_speeches:,} ({100*matched_speeches/total_speeches:.1f}%)")
    print(f"  Female speeches: {female_speeches:,} ({100*female_speeches/total_speeches:.1f}%)")
    print(f"  Male speeches: {male_speeches:,} ({100*male_speeches/total_speeches:.1f}%)")
    print(f"  Unmatched: {total_speeches - matched_speeches:,} ({100*(total_speeches - matched_speeches)/total_speeches:.1f}%)")

    print(f"\nUnique MPs (via person_id):")
    print(f"  Total unique MPs: {len(unique_person_ids):,}")
    print(f"  Female MPs: {len(female_person_ids):,} ({100*len(female_person_ids)/len(unique_person_ids):.1f}%)")
    print(f"  Male MPs: {len(male_person_ids):,} ({100*len(male_person_ids)/len(unique_person_ids):.1f}%)")

    # Count debates
    print("\nCounting debates...")
    debate_files = sorted(debates_dir.glob("debates_*.parquet"))
    total_debates = 0
    commons_debates = 0
    lords_debates = 0

    for f in debate_files:
        df = pd.read_parquet(f)
        total_debates += len(df)
        commons_debates += (df['chamber'] == 'Commons').sum()
        lords_debates += (df['chamber'] == 'Lords').sum()

    print(f"\nDebates:")
    print(f"  Total debates: {total_debates:,}")
    print(f"  Commons debates: {commons_debates:,} ({100*commons_debates/total_debates:.1f}%)")
    print(f"  Lords debates: {lords_debates:,} ({100*lords_debates/total_debates:.1f}%)")

    return {
        'total_speeches': total_speeches,
        'total_debates': total_debates,
        'commons_speeches': commons_speeches,
        'lords_speeches': lords_speeches,
        'female_speeches': female_speeches,
        'male_speeches': male_speeches,
        'unique_mps': len(unique_person_ids),
        'female_mps': len(female_person_ids),
        'male_mps': len(male_person_ids)
    }

def verify_gender_analysis(version):
    """Verify statistics from gender_analysis dataset"""
    print_section(f"GENDER ANALYSIS ({version.upper()})")

    paths = get_versioned_paths(version)
    data_dir = paths['gender']

    if not data_dir.exists():
        print(f"  WARNING: Directory not found: {data_dir}")
        print(f"  Skipping gender analysis verification for {version}")
        return None

    # Check for master file
    master_file = data_dir / "ALL_debates_enhanced_metadata.parquet"
    if master_file.exists():
        df = pd.read_parquet(master_file)
        print(f"\nMaster file (ALL_debates_enhanced_metadata.parquet):")
        print(f"  Total debates: {len(df):,}")
        print(f"  Debates with female: {df['has_female'].sum():,}")
        print(f"  Debates with male: {df['has_male'].sum():,}")
        return len(df)

    # Otherwise count year files
    year_files = sorted(data_dir.glob("debates_*_enhanced.parquet"))
    total_debates = sum(len(pd.read_parquet(f)) for f in year_files)
    print(f"\nYear files total: {total_debates:,} debates")
    return total_debates

def verify_processed_complete(version):
    """Verify statistics from processed dataset"""
    print_section(f"PROCESSED ({version.upper()})")

    paths = get_versioned_paths(version)
    data_dir = paths['processed'] / "metadata"

    if not data_dir.exists():
        print(f"  WARNING: Directory not found: {data_dir}")
        print(f"  Skipping processed verification for {version}")
        return None

    debate_files = sorted(data_dir.glob("debates_*.parquet"))
    total_debates = sum(len(pd.read_parquet(f)) for f in debate_files)

    print(f"\nMetadata debates: {total_debates:,}")
    return total_debates

def verify_mp_database():
    """Verify MP reference database statistics"""
    print_section("MP REFERENCE DATABASE")

    mp_file = project_root / "data-hansard" / "house_members_gendered_updated.parquet"

    if not mp_file.exists():
        print("MP database file not found!")
        return None

    df = pd.read_parquet(mp_file)

    print(f"\nTotal records: {len(df):,}")
    print(f"Unique persons: {df['person_id'].nunique():,}")

    gender_counts = df.groupby('gender_inferred')['person_id'].nunique()
    print(f"\nGender breakdown (unique persons):")
    for gender, count in gender_counts.items():
        pct = 100 * count / df['person_id'].nunique()
        print(f"  {gender}: {count:,} ({pct:.1f}%)")

    return {
        'total_records': len(df),
        'unique_persons': df['person_id'].nunique(),
        'female_persons': gender_counts.get('F', 0),
        'male_persons': gender_counts.get('M', 0)
    }

def verify_chamber_breakdown(version):
    """Detailed chamber and gender breakdown from derived dataset"""
    print_section(f"CHAMBER & GENDER BREAKDOWN ({version.upper()})")

    paths = get_versioned_paths(version)
    speeches_dir = paths['derived'] / "speeches_complete"

    if not speeches_dir.exists():
        print(f"  WARNING: Directory not found: {speeches_dir}")
        return

    speech_files = sorted(speeches_dir.glob("speeches_*.parquet"))

    print("\nAnalyzing by chamber...")

    for chamber in ['Commons', 'Lords']:
        total = 0
        matched = 0
        female = 0
        male = 0

        for f in speech_files:
            df = pd.read_parquet(f)
            chamber_df = df[df['chamber'] == chamber]
            total += len(chamber_df)
            matched += chamber_df['matched_mp'].sum()
            female += (chamber_df['gender'] == 'F').sum()
            male += (chamber_df['gender'] == 'M').sum()

        print(f"\n{chamber}:")
        print(f"  Total speeches: {total:,}")
        print(f"  Matched: {matched:,} ({100*matched/total:.1f}%)")
        print(f"  Female: {female:,} ({100*female/total:.1f}%)")
        print(f"  Male: {male:,} ({100*male/total:.1f}%)")
        print(f"  Unmatched: {total - matched:,} ({100*(total - matched)/total:.1f}%)")

        if matched > 0:
            print(f"  Female % of matched: {100*female/(female+male):.2f}%")

EXPECTED_VALUES = {
    'v1': {
        'total_speeches': 5_967_440,
        'total_debates': 1_197_828,
        'commons_speeches': 4_840_797,
        'lords_speeches': 1_126_643,
        'female_speeches': 138_461,
        'male_speeches': 4_263_054,
        'female_mps': 240,
        'male_mps': 8_429,
        'gender_analysis_debates': 652_271
    },
    'v2': {
        'total_speeches': 6_783_015,
        'total_debates': 1_197_828,
        'commons_speeches': 5_575_783,
        'lords_speeches': 1_207_232,
        'female_speeches': 152_526,
        'male_speeches': 4_842_283,
        'female_mps': 242,
        'male_mps': 8_574,
        'gender_analysis_debates': 652_168
    },
}


def compare_with_expected(version):
    """Compare actual counts with expected values from documentation"""
    print_section(f"VERIFICATION AGAINST EXPECTED VALUES ({version.upper()})")

    expected = EXPECTED_VALUES.get(version)
    if expected is None:
        print(f"  No expected values defined for version '{version}'")
        return None

    print("\nExpected values from documentation:")
    for key, value in expected.items():
        print(f"  {key}: {value:,}")

    return expected

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify Hansard dataset statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_dataset_statistics.py           # Verify current version (v2)
  python verify_dataset_statistics.py --version v1  # Verify legacy data
  python verify_dataset_statistics.py --version v2  # Verify bug-fixed data
        """
    )
    parser.add_argument(
        '--version', '-v',
        choices=['v1', 'v2'],
        default=DATASET_VERSION,
        help=f"Dataset version to verify (default: {DATASET_VERSION})"
    )
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help="Skip verification checks against expected values"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    version = args.version

    print("HANSARD DATASET STATISTICS VERIFICATION")
    print(f"Project root: {project_root}")
    print(f"Dataset version: {version}")
    print(f"Time: {pd.Timestamp.now()}")

    # Show paths being verified
    paths = get_versioned_paths(version)
    print(f"\nPaths:")
    print(f"  Derived: {paths['derived']}")
    print(f"  Gender: {paths['gender']}")
    print(f"  Processed: {paths['processed']}")

    # Verify each dataset tier
    derived_stats = verify_derived_complete(version)
    gender_debates = verify_gender_analysis(version)
    processed_debates = verify_processed_complete(version)
    mp_stats = verify_mp_database()

    # Detailed breakdowns
    if derived_stats:
        verify_chamber_breakdown(version)

    # Compare with expected
    expected = compare_with_expected(version)

    # Final summary
    print_section("FINAL VERIFICATION SUMMARY")

    if derived_stats:
        print(f"\nKey Statistics (Derived {version.upper()}):")
        print(f"  Total speeches: {derived_stats['total_speeches']:,}")
        print(f"  Total debates: {derived_stats['total_debates']:,}")
        print(f"  Female speeches: {derived_stats['female_speeches']:,}")
        print(f"  Male speeches: {derived_stats['male_speeches']:,}")
        print(f"  Unique female MPs: {derived_stats['female_mps']:,}")
        print(f"  Unique male MPs: {derived_stats['male_mps']:,}")
    else:
        print(f"\n  No derived dataset found for {version}")

    print("\nOther Dataset Tiers:")
    if processed_debates is not None:
        print(f"  Processed debates: {processed_debates:,}")
    else:
        print(f"  Processed: not found for {version}")
    if gender_debates is not None:
        print(f"  Gender analysis debates: {gender_debates:,}")
    else:
        print(f"  Gender analysis: not found for {version}")

    print("\nMP Database:")
    if mp_stats:
        print(f"  Total unique persons: {mp_stats['unique_persons']:,}")
        print(f"  Female MPs in database: {mp_stats['female_persons']:,}")
        print(f"  Male MPs in database: {mp_stats['male_persons']:,}")

    # Verification checks
    if args.skip_checks or derived_stats is None or expected is None:
        if derived_stats is None:
            print("\n[Skipping verification checks - no derived data found]")
        elif expected is None:
            print(f"\n[Skipping verification checks - no expected values for {version}]")
        else:
            print("\n[Verification checks skipped by user]")
        return 0

    print("\n" + "=" * 70)
    print("VERIFICATION CHECKS")
    print("=" * 70)

    checks = [
        ('Total speeches match', derived_stats['total_speeches'], expected['total_speeches']),
        ('Total debates match', derived_stats['total_debates'], expected['total_debates']),
        ('Female speeches match', derived_stats['female_speeches'], expected['female_speeches']),
        ('Female MPs match', derived_stats['female_mps'], expected['female_mps']),
    ]

    all_pass = True
    for check_name, actual, expected_val in checks:
        status = "PASS" if actual == expected_val else "FAIL"
        print(f"{status}: {check_name} (actual: {actual:,}, expected: {expected_val:,})")
        if actual != expected_val:
            all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL CHECKS PASSED - Statistics verified!")
    else:
        print("SOME CHECKS FAILED - Please investigate discrepancies")
    print("=" * 70)

    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
