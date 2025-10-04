#!/usr/bin/env python3
"""
Centralized path configuration for the Hansard NLP project.
Handles paths correctly regardless of where scripts are run from.
"""

import os
from pathlib import Path

def get_project_root():
    """Find the project root by looking for key files."""
    current = Path(__file__).resolve()

    # Walk up directory tree looking for project root indicators
    for parent in current.parents:
        if (parent / 'README.md').exists() and (parent / 'src' / 'hansard').exists():
            return parent

    # Fallback: assume we're in src/hansard/utils/
    return current.parent.parent.parent

# Get project root once
PROJECT_ROOT = get_project_root()

# Define all paths relative to project root
class Paths:
    """Centralized path configuration."""

    # Root directories
    ROOT = PROJECT_ROOT
    SRC = PROJECT_ROOT / 'src' / 'hansard'

    # Data directories
    DATA_DIR = SRC / 'data'
    GENDER_ENHANCED_DATA = DATA_DIR / 'gender_analysis_enhanced'
    GENDER_WORDLISTS = DATA_DIR / 'gender_wordlists'
    PROCESSED_FIXED = DATA_DIR / 'processed_fixed'

    # Analysis directories
    ANALYSIS_DIR = SRC / 'analysis'
    CORPUS_RESULTS = ANALYSIS_DIR / 'corpus_results'
    MILESTONE_RESULTS = ANALYSIS_DIR / 'milestone_results'
    VISUALIZATIONS = ANALYSIS_DIR / 'visualizations'
    RESULTS = ANALYSIS_DIR / 'results'

    # Script directories
    SCRIPTS_DIR = SRC / 'scripts'

    # Documentation
    DOCS_DIR = SRC / 'docs'

    # Specific files
    MALE_WORDS = GENDER_WORDLISTS / 'male_words.txt'
    FEMALE_WORDS = GENDER_WORDLISTS / 'female_words.txt'

    @classmethod
    def ensure_output_dirs(cls):
        """Create all output directories if they don't exist."""
        dirs = [
            cls.CORPUS_RESULTS,
            cls.MILESTONE_RESULTS,
            cls.VISUALIZATIONS,
            cls.RESULTS
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_year_files(cls, start_year=None, end_year=None):
        """Get list of year data files."""
        pattern = "debates_*_enhanced.parquet"
        all_files = sorted(cls.GENDER_ENHANCED_DATA.glob(pattern))

        if start_year and end_year:
            filtered = []
            for f in all_files:
                # Extract year from filename
                import re
                match = re.search(r'debates_(\d{4})_enhanced\.parquet', f.name)
                if match:
                    year = int(match.group(1))
                    if start_year <= year <= end_year:
                        filtered.append(f)
            return filtered
        return all_files

    @classmethod
    def print_paths(cls):
        """Print all configured paths for debugging."""
        print("Path Configuration:")
        print(f"  Project root: {cls.ROOT}")
        print(f"  Data dir: {cls.DATA_DIR}")
        print(f"  Gender data: {cls.GENDER_ENHANCED_DATA}")
        print(f"  Output dirs:")
        print(f"    - Corpus results: {cls.CORPUS_RESULTS}")
        print(f"    - Milestone results: {cls.MILESTONE_RESULTS}")
        print(f"    - Visualizations: {cls.VISUALIZATIONS}")


# Test functionality if run directly
if __name__ == "__main__":
    print("Testing path configuration...")
    Paths.print_paths()

    # Check if key directories exist
    print("\nDirectory existence check:")
    print(f"  Gender data exists: {Paths.GENDER_ENHANCED_DATA.exists()}")
    print(f"  Male words exists: {Paths.MALE_WORDS.exists()}")
    print(f"  Female words exists: {Paths.FEMALE_WORDS.exists()}")

    # Test year file finder
    print("\nYear files (1990-1995):")
    for f in Paths.get_year_files(1990, 1995):
        print(f"  - {f.name}")