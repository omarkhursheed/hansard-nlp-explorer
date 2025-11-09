#!/usr/bin/env python3
"""
Regression test to verify dataset reproducibility.
Run this BEFORE and AFTER cleanup to ensure data is identical.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import json
import sys

class ReproducibilityTester:
    def __init__(self, baseline_dir=None, new_dir=None):
        """
        Compare two versions of the gender analysis dataset.

        Args:
            baseline_dir: Path to original dataset (default: gender_analysis_enhanced.BACKUP)
            new_dir: Path to new dataset (default: gender_analysis_enhanced)
        """
        project_root = Path(__file__).parent.parent
        data_base = project_root / "src" / "hansard" / "data"

        self.baseline_dir = Path(baseline_dir) if baseline_dir else data_base / "gender_analysis_enhanced.BACKUP"
        self.new_dir = Path(new_dir) if new_dir else data_base / "gender_analysis_enhanced"

        self.errors = []
        self.warnings = []

    def test_file_existence(self):
        """Verify all expected files exist in both directories"""
        print("\n" + "="*70)
        print("TEST 1: File Existence")
        print("="*70)

        baseline_files = set(f.name for f in self.baseline_dir.glob("*.parquet"))
        new_files = set(f.name for f in self.new_dir.glob("*.parquet"))

        missing = baseline_files - new_files
        extra = new_files - baseline_files

        if missing:
            self.errors.append(f"Missing files in new dataset: {missing}")
            print(f"❌ FAIL: Missing files: {missing}")

        if extra:
            self.warnings.append(f"Extra files in new dataset: {extra}")
            print(f"⚠️  WARN: Extra files: {extra}")

        if not missing and not extra:
            print(f"✅ PASS: All {len(baseline_files)} files present")

        return len(missing) == 0

    def test_master_dataset(self):
        """Compare the main combined dataset"""
        print("\n" + "="*70)
        print("TEST 2: Master Dataset Comparison")
        print("="*70)

        master_file = "ALL_debates_enhanced_with_text.parquet"

        baseline_path = self.baseline_dir / master_file
        new_path = self.new_dir / master_file

        if not baseline_path.exists():
            print(f"⚠️  SKIP: Baseline not found at {baseline_path}")
            return True

        if not new_path.exists():
            self.errors.append(f"Master dataset not found: {new_path}")
            print(f"❌ FAIL: New dataset not found")
            return False

        # Load both
        print(f"Loading baseline... ", end="")
        df_old = pd.read_parquet(baseline_path)
        print(f"{len(df_old):,} rows")

        print(f"Loading new data... ", end="")
        df_new = pd.read_parquet(new_path)
        print(f"{len(df_new):,} rows")

        # Compare shapes
        if df_old.shape != df_new.shape:
            self.errors.append(f"Shape mismatch: {df_old.shape} vs {df_new.shape}")
            print(f"❌ FAIL: Shape mismatch")
            return False

        print(f"✅ Shape: {df_old.shape} (identical)")

        # Columns to ignore (timestamps)
        ignore_cols = {'processing_timestamp', 'extraction_timestamp', 'creation_date'}
        compare_cols = [c for c in df_old.columns if c not in ignore_cols]

        print(f"\nComparing {len(compare_cols)} columns (ignoring {len(ignore_cols)} timestamp columns)...")

        differences = {}

        for col in compare_cols:
            if col not in df_new.columns:
                self.errors.append(f"Column missing: {col}")
                print(f"  ❌ {col}: MISSING")
                continue

            # Handle different data types
            if df_old[col].dtype == 'object':
                # For object/string columns
                diff_mask = df_old[col].astype(str) != df_new[col].astype(str)
                diff_count = diff_mask.sum()

                if diff_count > 0:
                    differences[col] = diff_count
                    # Sample a few differences
                    diff_indices = df_old[diff_mask].index[:3]
                    print(f"  ❌ {col}: {diff_count:,} differences")
                    for idx in diff_indices:
                        print(f"     Row {idx}: '{df_old.loc[idx, col]}' vs '{df_new.loc[idx, col]}'")

            elif pd.api.types.is_numeric_dtype(df_old[col]):
                # For numeric columns
                if not np.allclose(
                    df_old[col].fillna(0),
                    df_new[col].fillna(0),
                    rtol=1e-9,
                    atol=1e-9,
                    equal_nan=True
                ):
                    diff_mask = ~np.isclose(
                        df_old[col].fillna(0),
                        df_new[col].fillna(0),
                        rtol=1e-9,
                        atol=1e-9,
                        equal_nan=True
                    )
                    diff_count = diff_mask.sum()
                    differences[col] = diff_count
                    print(f"  ❌ {col}: {diff_count:,} numeric differences")

            else:
                # For list/array columns
                try:
                    same = all((df_old[col].iloc[i] == df_new[col].iloc[i])
                              if isinstance(df_old[col].iloc[i], (list, np.ndarray))
                              else (df_old[col].iloc[i] == df_new[col].iloc[i])
                              for i in range(len(df_old)))
                    if not same:
                        differences[col] = "LIST_DIFF"
                        print(f"  ❌ {col}: List differences detected")
                except:
                    self.warnings.append(f"Could not compare column: {col}")
                    print(f"  ⚠️  {col}: Skipped (complex type)")

        if differences:
            self.errors.append(f"Data differences found in {len(differences)} columns")
            print(f"\n❌ FAIL: {len(differences)} columns differ")
            return False
        else:
            print(f"\n✅ PASS: All {len(compare_cols)} columns identical")
            return True

    def test_metadata_file(self):
        """Compare metadata JSON"""
        print("\n" + "="*70)
        print("TEST 3: Metadata Comparison")
        print("="*70)

        meta_file = "dataset_metadata.json"

        baseline_path = self.baseline_dir / meta_file
        new_path = self.new_dir / meta_file

        if not baseline_path.exists():
            print(f"⚠️  SKIP: Baseline metadata not found")
            return True

        if not new_path.exists():
            self.errors.append("Metadata file missing")
            print(f"❌ FAIL: Metadata not found")
            return False

        with open(baseline_path) as f:
            old_meta = json.load(f)
        with open(new_path) as f:
            new_meta = json.load(f)

        # Ignore timestamp fields
        ignore_keys = {'creation_date', 'processing_timestamp', 'generation_time'}

        # Compare counts
        count_fields = [
            'total_debates', 'debates_with_mps', 'debates_with_female_mps',
            'unique_female_mps', 'unique_male_mps', 'total_years'
        ]

        differences = []
        for field in count_fields:
            if field in old_meta and field in new_meta:
                if old_meta[field] != new_meta[field]:
                    differences.append(f"{field}: {old_meta[field]} vs {new_meta[field]}")
                    print(f"  ❌ {field}: {old_meta[field]} → {new_meta[field]}")

        if differences:
            self.errors.append(f"Metadata differences: {differences}")
            print(f"\n❌ FAIL: {len(differences)} metadata fields differ")
            return False
        else:
            print(f"✅ PASS: All metadata counts identical")
            return True

    def test_sample_year(self, year=1920):
        """Deep comparison of a single year's data"""
        print("\n" + "="*70)
        print(f"TEST 4: Deep Dive - Year {year}")
        print("="*70)

        baseline_path = self.baseline_dir / f"debates_{year}_enhanced.parquet"
        new_path = self.new_dir / f"debates_{year}_enhanced.parquet"

        if not baseline_path.exists() or not new_path.exists():
            print(f"⚠️  SKIP: Year {year} not found in one or both datasets")
            return True

        df_old = pd.read_parquet(baseline_path)
        df_new = pd.read_parquet(new_path)

        print(f"Rows: {len(df_old)} vs {len(df_new)}")

        if len(df_old) != len(df_new):
            self.errors.append(f"Year {year} row count mismatch")
            print(f"❌ FAIL: Row count differs")
            return False

        # Check a few specific fields
        test_fields = ['debate_id', 'has_female', 'has_male', 'total_speakers']

        for field in test_fields:
            if field in df_old.columns and field in df_new.columns:
                if not (df_old[field] == df_new[field]).all():
                    diff_count = (df_old[field] != df_new[field]).sum()
                    print(f"  ❌ {field}: {diff_count} differences")
                    self.errors.append(f"Year {year}, field {field} differs")
                else:
                    print(f"  ✅ {field}: identical")

        return len(self.errors) == 0

    def run_all_tests(self):
        """Run complete test suite"""
        print("\n")
        print("="*70)
        print("REPRODUCIBILITY TEST SUITE")
        print("="*70)
        print(f"Baseline: {self.baseline_dir}")
        print(f"New data: {self.new_dir}")
        print("="*70)

        tests = [
            ("File Existence", self.test_file_existence),
            ("Master Dataset", self.test_master_dataset),
            ("Metadata", self.test_metadata_file),
            ("Sample Year", self.test_sample_year),
        ]

        results = []
        for name, test_func in tests:
            try:
                passed = test_func()
                results.append((name, passed))
            except Exception as e:
                print(f"\n❌ ERROR in {name}: {e}")
                self.errors.append(f"{name} crashed: {e}")
                results.append((name, False))

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        passed = sum(1 for _, p in results if p)
        total = len(results)

        for name, p in results:
            status = "✅ PASS" if p else "❌ FAIL"
            print(f"  {status}: {name}")

        print(f"\nTests: {passed}/{total} passed")

        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"  ⚠️  {w}")

        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for e in self.errors:
                print(f"  ❌ {e}")

        print("\n" + "="*70)
        if passed == total:
            print("🟢 VERDICT: DATASETS ARE REPRODUCIBLE")
            print("="*70)
            return 0
        else:
            print("🔴 VERDICT: DIFFERENCES DETECTED")
            print("="*70)
            return 1


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Test dataset reproducibility")
    parser.add_argument('--baseline', help='Path to baseline dataset')
    parser.add_argument('--new', help='Path to new dataset')

    args = parser.parse_args()

    tester = ReproducibilityTester(
        baseline_dir=args.baseline,
        new_dir=args.new
    )

    return tester.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
