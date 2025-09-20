"""
Data validation module for Hansard datasets.
Checks integrity, coverage, and quality of processed data.
"""

import polars as pl
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sqlite3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation for Hansard datasets."""

    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self.raw_dir = data_dir / "hansard"
        # Check for both possible processed directory names
        if (data_dir / "processed_fixed").exists():
            self.processed_dir = data_dir / "processed_fixed"
        else:
            self.processed_dir = data_dir / "processed"
        self.metadata_dir = self.processed_dir / "metadata"
        self.content_dir = self.processed_dir / "content"
        self.index_dir = self.processed_dir / "index"

        self.validation_report = {
            'timestamp': datetime.now().isoformat(),
            'checks_passed': [],
            'issues_found': [],
            'statistics': {}
        }

    def validate_all(self) -> Dict:
        """Run all validation checks."""
        logger.info("Starting comprehensive data validation")

        # Check directory structure
        self._check_directory_structure()

        # Validate metadata files
        self._validate_metadata_files()

        # Validate content files
        self._validate_content_files()

        # Check data coverage
        self._check_temporal_coverage()

        # Validate database if exists
        self._validate_database()

        # Cross-reference checks
        self._cross_reference_validation()

        # Generate summary
        self._generate_summary()

        return self.validation_report

    def _check_directory_structure(self):
        """Check that expected directories exist."""
        logger.info("Checking directory structure")

        expected_dirs = [
            self.processed_dir,
            self.metadata_dir,
            self.content_dir,
            self.raw_dir
        ]

        for dir_path in expected_dirs:
            if dir_path.exists():
                self.validation_report['checks_passed'].append(f"Directory exists: {dir_path}")
            else:
                self.validation_report['issues_found'].append(f"Missing directory: {dir_path}")

    def _validate_metadata_files(self):
        """Validate Parquet metadata files."""
        logger.info("Validating metadata files")

        # Check debates_master.parquet
        debates_master = self.metadata_dir / "debates_master.parquet"
        if debates_master.exists():
            try:
                df = pl.read_parquet(debates_master)

                # Check schema
                expected_columns = ['year', 'file_path', 'hansard_ref']
                missing_cols = [col for col in expected_columns if col not in df.columns]

                if missing_cols:
                    self.validation_report['issues_found'].append(
                        f"Missing columns in debates_master: {missing_cols}"
                    )
                else:
                    self.validation_report['checks_passed'].append(
                        "debates_master.parquet schema valid"
                    )

                # Statistics
                self.validation_report['statistics']['total_debates'] = len(df)
                self.validation_report['statistics']['year_range'] = {
                    'min': df['year'].min(),
                    'max': df['year'].max()
                }

                # Check for nulls
                null_counts = {col: df[col].null_count() for col in df.columns}
                high_null_cols = [col for col, count in null_counts.items() if count > len(df) * 0.5]

                if high_null_cols:
                    self.validation_report['issues_found'].append(
                        f"High null rate (>50%) in columns: {high_null_cols}"
                    )

            except Exception as e:
                self.validation_report['issues_found'].append(
                    f"Error reading debates_master.parquet: {str(e)}"
                )
        else:
            self.validation_report['issues_found'].append("Missing debates_master.parquet")

        # Check speakers file if exists
        speakers_master = self.metadata_dir / "speakers_master.parquet"
        if speakers_master.exists():
            try:
                speakers_df = pl.read_parquet(speakers_master)
                self.validation_report['statistics']['total_speakers'] = len(speakers_df)
                self.validation_report['checks_passed'].append("speakers_master.parquet valid")
            except Exception as e:
                self.validation_report['issues_found'].append(
                    f"Error reading speakers_master.parquet: {str(e)}"
                )

    def _validate_content_files(self):
        """Validate JSONL content files."""
        logger.info("Validating content files")

        if not self.content_dir.exists():
            self.validation_report['issues_found'].append("Content directory missing")
            return

        year_dirs = sorted([d for d in self.content_dir.iterdir() if d.is_dir()])

        self.validation_report['statistics']['content_years'] = len(year_dirs)

        # Sample validation of JSONL files
        sample_years = year_dirs[:5] if len(year_dirs) > 5 else year_dirs

        for year_dir in sample_years:
            jsonl_files = list(year_dir.glob("*.jsonl"))

            if jsonl_files:
                # Check first file in each year
                sample_file = jsonl_files[0]
                try:
                    with open(sample_file, 'r', encoding='utf-8') as f:
                        first_line = f.readline()
                        data = json.loads(first_line)

                        # Check expected fields
                        if 'full_text' in data and 'metadata' in data:
                            self.validation_report['checks_passed'].append(
                                f"Valid JSONL structure in {year_dir.name}"
                            )
                        else:
                            self.validation_report['issues_found'].append(
                                f"Invalid JSONL structure in {sample_file}"
                            )

                except Exception as e:
                    self.validation_report['issues_found'].append(
                        f"Error reading JSONL in {year_dir.name}: {str(e)}"
                    )
            else:
                self.validation_report['issues_found'].append(
                    f"No JSONL files in {year_dir.name}"
                )

    def _check_temporal_coverage(self):
        """Check temporal coverage of the data."""
        logger.info("Checking temporal coverage")

        # Expected range for Hansard
        expected_start = 1803
        expected_end = 2005

        if 'year_range' in self.validation_report['statistics']:
            actual_start = self.validation_report['statistics']['year_range']['min']
            actual_end = self.validation_report['statistics']['year_range']['max']

            if actual_start == expected_start and actual_end == expected_end:
                self.validation_report['checks_passed'].append(
                    f"Temporal coverage complete: {actual_start}-{actual_end}"
                )
            else:
                self.validation_report['issues_found'].append(
                    f"Temporal coverage mismatch. Expected: {expected_start}-{expected_end}, "
                    f"Found: {actual_start}-{actual_end}"
                )

    def _validate_database(self):
        """Validate SQLite database if exists."""
        logger.info("Checking database")

        db_path = self.index_dir / "debates.db"
        if db_path.exists():
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Check tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()

                if tables:
                    self.validation_report['checks_passed'].append(
                        f"Database contains {len(tables)} tables"
                    )
                    self.validation_report['statistics']['db_tables'] = [t[0] for t in tables]
                else:
                    self.validation_report['issues_found'].append("Database has no tables")

                conn.close()

            except Exception as e:
                self.validation_report['issues_found'].append(f"Database error: {str(e)}")
        else:
            logger.info("No database found (optional)")

    def _cross_reference_validation(self):
        """Cross-reference validation between datasets."""
        logger.info("Running cross-reference validation")

        # Check if metadata years match content years
        if self.content_dir.exists() and 'year_range' in self.validation_report['statistics']:
            content_years = set([int(d.name) for d in self.content_dir.iterdir()
                                if d.is_dir() and d.name.isdigit()])

            metadata_range = self.validation_report['statistics']['year_range']
            metadata_years = set(range(metadata_range['min'], metadata_range['max'] + 1))

            missing_content = metadata_years - content_years
            extra_content = content_years - metadata_years

            if missing_content:
                self.validation_report['issues_found'].append(
                    f"Years in metadata but not in content: {sorted(missing_content)[:10]}"
                )

            if extra_content:
                self.validation_report['issues_found'].append(
                    f"Years in content but not in metadata: {sorted(extra_content)[:10]}"
                )

            if not missing_content and not extra_content:
                self.validation_report['checks_passed'].append(
                    "Metadata and content years match perfectly"
                )

    def _generate_summary(self):
        """Generate validation summary."""
        logger.info("Generating validation summary")

        total_checks = len(self.validation_report['checks_passed'])
        total_issues = len(self.validation_report['issues_found'])

        self.validation_report['summary'] = {
            'total_checks_passed': total_checks,
            'total_issues_found': total_issues,
            'validation_status': 'PASS' if total_issues == 0 else 'ISSUES FOUND',
            'health_score': (total_checks / (total_checks + total_issues) * 100) if (total_checks + total_issues) > 0 else 0
        }

    def save_report(self, output_path: Optional[Path] = None):
        """Save validation report to JSON."""
        if output_path is None:
            output_path = Path("validation_report.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.validation_report, f, indent=2, default=str)

        logger.info(f"Validation report saved to {output_path}")
        return output_path


def main():
    """Run data validation."""
    validator = DataValidator()
    report = validator.validate_all()

    # Print summary
    print("\n" + "="*60)
    print("DATA VALIDATION SUMMARY")
    print("="*60)
    print(f"Checks Passed: {report['summary']['total_checks_passed']}")
    print(f"Issues Found: {report['summary']['total_issues_found']}")
    print(f"Health Score: {report['summary']['health_score']:.1f}%")
    print(f"Status: {report['summary']['validation_status']}")

    if report['statistics']:
        print("\nStatistics:")
        for key, value in report['statistics'].items():
            print(f"  {key}: {value}")

    if report['issues_found']:
        print("\nIssues Found:")
        for issue in report['issues_found'][:10]:  # Show first 10
            print(f"  - {issue}")
        if len(report['issues_found']) > 10:
            print(f"  ... and {len(report['issues_found']) - 10} more")

    # Save report
    validator.save_report()

    return report


if __name__ == "__main__":
    main()