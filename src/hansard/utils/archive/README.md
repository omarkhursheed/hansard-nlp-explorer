# Archived Utils

This directory contains utility files that were archived during the codebase cleanup.

## Files Archived

### `data_validator.py`
- **Original purpose:** Data validation module for Hansard datasets
- **Why archived:** Used for old processing pipeline, not needed for current analysis
- **Dependencies:** polars, sqlite3
- **Can be restored:** Yes, if data validation is needed

### `high_performance_processor.py`
- **Original purpose:** High-performance Hansard processor for M3 Max with 64GB RAM
- **Why archived:** Used for old HTML processing pipeline, not needed for current analysis
- **Dependencies:** multiprocessing, concurrent.futures
- **Can be restored:** Yes, if raw HTML processing is needed

### `path_utils.py`
- **Original purpose:** Universal path utilities for Hansard project
- **Why archived:** Functionality merged into `path_config.py` for consolidation
- **Dependencies:** None
- **Can be restored:** No, functionality is now in `path_config.py`

### Debug Files
- `debug_parser.py` - Parser debugging utilities
- `debug.py` - General debugging utilities  
- `debug_1860s_raw.html` - Sample HTML file for debugging
- `debug_1864_raw.html` - Sample HTML file for debugging
- `investigate_preamble.py` - Preamble investigation script

**Why archived:** Debug files from development phase, not needed for production analysis

## Restoration Instructions

If any of these files are needed:

1. Copy the file back to `src/hansard/utils/`
2. Update any import statements if needed
3. Test functionality before using

## Archive Date
December 2024 - Utils cleanup and consolidation
