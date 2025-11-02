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

    # Data directories - now at top level
    DATA_DIR = ROOT / 'data-hansard'
    GENDER_ENHANCED_DATA = DATA_DIR / 'gender_analysis_complete'  # Updated from gender_analysis_enhanced
    GENDER_WORDLISTS = DATA_DIR / 'gender_wordlists'
    PROCESSED_DATA = DATA_DIR / 'processed_complete'  # Updated from processed_fixed
    DERIVED_DATA = DATA_DIR / 'derived_complete'  # Unified speeches/debates

    # Legacy names for backward compatibility
    PROCESSED_FIXED = PROCESSED_DATA  # Alias
    GENDER_ANALYSIS_ENHANCED = GENDER_ENHANCED_DATA  # Alias

    # Analysis directory - single top-level directory for all outputs
    ANALYSIS_DIR = ROOT / 'analysis'
    # Legacy subdirectory paths - kept for compatibility but not used
    CORPUS_RESULTS = ANALYSIS_DIR
    MILESTONE_RESULTS = ANALYSIS_DIR
    VISUALIZATIONS = ANALYSIS_DIR
    RESULTS = ANALYSIS_DIR

    # Script directories
    SCRIPTS_DIR = SRC / 'scripts'

    # Specific files
    MALE_WORDS = GENDER_WORDLISTS / 'male_words.txt'
    FEMALE_WORDS = GENDER_WORDLISTS / 'female_words.txt'

    @classmethod
    def ensure_output_dirs(cls):
        """Create all output directories if they don't exist."""
        cls.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

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

    # Utility methods from path_utils.py for backward compatibility
    @classmethod
    def get_data_dir(cls):
        """Get the data directory path that works from anywhere."""
        return cls.DATA_DIR

    @classmethod
    def get_processed_data_dir(cls):
        """Get processed data directory, checking for alternate names."""
        if cls.PROCESSED_DATA.exists():
            return cls.PROCESSED_DATA
        elif cls.PROCESSED_FIXED.exists():
            return cls.PROCESSED_FIXED  # Legacy fallback
        raise RuntimeError(f"No processed data directory found at {cls.PROCESSED_DATA} or {cls.PROCESSED_FIXED}")

    @classmethod
    def get_gender_wordlists_dir(cls):
        """Get gender wordlists directory."""
        if not cls.GENDER_WORDLISTS.exists():
            raise RuntimeError(f"Gender wordlists not found at {cls.GENDER_WORDLISTS}")
        return cls.GENDER_WORDLISTS

    @classmethod
    def get_analysis_output_dir(cls):
        """Get or create analysis output directory."""
        cls.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.ANALYSIS_DIR

    @classmethod
    def resolve_path(cls, path_str, base='project'):
        """
        Resolve a path string to an absolute path.

        Args:
            path_str: Path string (can be relative or absolute)
            base: Base for relative paths ('project', 'data', 'cwd')

        Returns:
            Absolute Path object
        """
        from pathlib import Path
        path = Path(path_str)

        # If already absolute, return as-is
        if path.is_absolute():
            return path

        # Otherwise resolve based on base
        if base == 'project':
            return cls.ROOT / path
        elif base == 'data':
            return cls.DATA_DIR / path
        elif base == 'cwd':
            return Path.cwd() / path
        else:
            raise ValueError(f"Unknown base: {base}")

    # Legacy function names for backward compatibility
    @classmethod
    def find_project_root(cls):
        """Legacy function - returns project root."""
        return cls.ROOT


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