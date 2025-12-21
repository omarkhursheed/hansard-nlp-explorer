# Hansard NLP Explorer - Correctness Audit Report

**Date:** 2025-12-06
**Auditor:** Claude Code

---

## Executive Summary

This audit identified **critical**, **high**, and **moderate** severity issues across the codebase. The most significant finding is a bug in the speech extraction algorithm that causes approximately 5% of speeches to have incorrect speaker attribution.

### Issue Summary

| Severity | Count | Category |
|----------|-------|----------|
| Critical | 1 | Data quality - speaker extraction bug |
| High | 6 | Broken imports, tests, documentation |
| Moderate | 8 | Code quality, dead code, DRY violations |

---

## Critical Issues

### 1. Speaker Extraction Bug (CRITICAL)

**Location:**
- `scripts/data_creation/create_unified_complete_datasets.py:175-233`
- `scripts/data_creation/create_enhanced_gender_dataset.py:34-84`

**Problem:** The `extract_speeches_from_text()` function uses regex patterns that fail to match speakers in common parliamentary formats:

```python
# Current patterns (incomplete)
patterns = [
    f"§\\s*\\*?\\s*{escaped_speaker}",   # Fails for "§ Q2. Mr. Amess"
    f"\\n{escaped_speaker}\\s*:",         # Fails - text is flattened
    f"\\n{escaped_speaker}\\s*\\(",       # Fails - text is flattened
]
```

**Impact:**
- ~5% of speeches have embedded interjections from other speakers
- 2-10% of female speeches contain undetected male speaker interjections
- Prime Minister's Questions (PMQs) significantly affected - Q1, Q2, Q3 markers break patterns
- Thatcher speeches in 1990s contain questions from other MPs attributed to her

**Example:**
```
Debate: 53018b9e2cf1a777
Attributed to: Sir T. Dugdale
Contains: "Mr. Williams I have read it." (should be separate speech)
```

**Root Cause:**
1. Section marker pattern `§\s*{speaker}` doesn't account for `Q2.` between marker and speaker
2. Inline interjections after sentence-ending punctuation are not detected
3. HTML structure is lost when text is flattened with `get_text(separator=' ')`

---

## High Severity Issues

### 2. Broken Test Imports (5 tests)

| Test File | Broken Import | Correct Import |
|-----------|--------------|----------------|
| `test_analysis_utils.py` | `from analysis.analysis_utils` | `from hansard.analysis.analysis_utils` |
| `test_backfill_small.py` | `from crawler import` | Script in `scripts/`, not a package |
| `test_comprehensive_analysis.py` | `from analysis.comprehensive_analysis` | Script in `scripts/`, not `src/` |
| `test_matching_improvements.py` | `from scripts.matching.mp_matcher` | `from hansard.matching.mp_matcher_corrected` |
| `test_single_digit_crawler.py` | `from crawler import` | Script in `scripts/`, not a package |

**Test Results:**
- 5/25 tests fail at import
- 1/20 tests fail (outdated file reference)
- 19/20 tests pass

### 3. README Documentation Errors

**Incorrect script locations:**
- README claims: `src/hansard/analysis/comprehensive_analysis.py`
- Actual location: `scripts/analysis/comprehensive_analysis.py`

**Affected documentation:**
- `README.md` - Repository structure diagram
- `CLAUDE.md` - Quick test commands

**Files that should exist in `src/hansard/analysis/` but are in `scripts/analysis/`:**
- `comprehensive_analysis.py`
- `gendered_comprehensive_analysis.py`
- `suffrage_analysis.py`
- `basic_analysis.py`

### 4. Script Import Failures

**Scripts with broken imports (cannot run directly):**
- `scripts/analysis/basic_analysis.py` - `from utils.unified_data_loader` fails
- `scripts/analysis/comprehensive_analysis.py` - same issue
- `scripts/analysis/gendered_comprehensive_analysis.py` - same issue

### 5. Outdated Test References

`test_repo_structure.py:59` expects:
- `modal_suffrage_classification_v5.py`

But only exists:
- `modal_suffrage_classification_v6.py`

### 6. Gender Filter Bug

**Location:** `src/hansard/utils/unified_data_loader.py:200`

```python
# Bug: uses lowercase 'm' or 'f' but data has uppercase 'M' or 'F'
combined_df = combined_df[combined_df['gender'] == gender_filter[0]]
```

---

## Moderate Severity Issues

### 7. Duplicated Code (DRY Violation)

`extract_speeches_from_text()` defined THREE times:
1. `create_enhanced_gender_dataset.py:34` - module-level function
2. `create_enhanced_gender_dataset.py:343` - class method (duplicate)
3. `create_unified_complete_datasets.py:175` - module-level function

**Recommendation:** Extract to shared module in `src/hansard/utils/`

### 8. Dead Code - Stray Return Statement

**Location:** `src/hansard/matching/mp_matcher_corrected.py:657`

```python
        return None  # Line 655 - end of function

        return None  # Line 657 - DEAD CODE
```

### 9. sys.path Manipulation

Multiple files manipulate `sys.path` which is a code smell:
- `src/hansard/matching/mp_matcher_corrected.py:10-12`
- `src/hansard/utils/unified_data_loader.py:31-34`

### 10. Archive/Debug Directories to Clean Up

Potential dead code in archive directories:
- `scripts/data_creation/archived/` (2 files)
- `scripts/experimental/` (6 files)
- `scripts/analysis/debugging/` (7 files)
- `src/hansard/utils/archive/` (5 files)
- `outputs/llm_classification/archive/` (data files)

### 11. Tests Return Values Instead of Using Assertions

**Warning:** 4 tests return values instead of None:
- `test_complete_parsing.py::test_multiple_file_types`
- `test_nlp_real_data.py::test_nlp_analysis_small_sample`
- `test_nlp_real_data.py::test_gender_analysis`
- `test_nlp_real_data.py::test_stop_words`

### 12. Incomplete Title Database

**Location:** `src/hansard/matching/mp_matcher_corrected.py:146-173`

The title database only contains Prime Ministers. Comment indicates more titles should be added:
```python
# Additional titles would be added here with verified dates
# 'chancellor of the exchequer': [...],
# 'foreign secretary': [...],
```

### 13. File Count Imbalance

- 65 Python files in `scripts/`
- 13 Python files in `src/`

Many scripts should be refactored into proper package modules.

### 14. Missing __init__.py Files

Several directories may be missing `__init__.py` for proper package structure.

---

## Recommendations

### Immediate (Critical)

1. **Fix speaker extraction algorithm:**
   - Add pattern for Question Time format: `§\s*(?:Q\d+\.\s*)?{speaker}`
   - Add pattern for inline interjections: `(?<=[.!?])\s*{speaker}\s+(?=[A-Z])`
   - Consider using HTML structure instead of flattened text

2. **Regenerate derived datasets** after fixing extraction

### Short-term (High)

3. Fix broken test imports to restore test coverage
4. Update README/CLAUDE.md with correct paths
5. Fix gender filter case sensitivity bug
6. Update `test_repo_structure.py` to expect v6 instead of v5

### Medium-term (Moderate)

7. Extract duplicated functions to shared modules
8. Remove dead code and clean up archive directories
9. Eliminate sys.path manipulation with proper package structure
10. Convert scripts to proper package modules where appropriate

---

## Verification Commands

```bash
# Run tests (shows 5 import errors, 1 failure)
pytest tests/ -v --tb=short

# Check speaker extraction impact
python3 -c "
import pandas as pd
from pathlib import Path
df = pd.read_parquet('data-hansard/derived_complete/speeches_complete/speeches_1990.parquet')
# Check for embedded speakers in text
import re
issues = df['text'].apply(lambda x: len(re.findall(r'(?<=[\s.!?])(?:Mr\.|Mrs\.|Sir|Lady|Lord)\s+[A-Z][a-z]+\s+[A-Z]', str(x)[50:])) if x else 0)
print(f'Speeches with potential issues: {(issues > 0).sum()} / {len(df)} ({100*(issues > 0).mean():.1f}%)')
"
```

---

## Files Modified During Audit

None - this was a read-only audit.

## Files Reviewed

- All Python files in `src/hansard/`
- All Python files in `tests/`
- Key scripts in `scripts/analysis/`, `scripts/data_creation/`
- `README.md`, `CLAUDE.md`
- Sample data files in `data-hansard/derived_complete/`
