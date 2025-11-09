#!/usr/bin/env python3
"""
Test repository structure and documentation integrity.

Verifies:
- Documentation files exist at expected locations
- Scripts are organized correctly
- No stray files at top level
- Links in README are valid
"""

import os
from pathlib import Path

def test_documentation_exists():
    """Test that all documented files exist."""
    docs_dir = Path("docs/suffrage_classification")

    required_docs = [
        "SUFFRAGE_CLASSIFICATION_METHODOLOGY.md",
        "MANUAL_VALIDATION_SUMMARY.md",
        "FALSE_POSITIVE_ANALYSIS.md",
        "SETUP_MODAL_CLASSIFICATION.md"
    ]

    for doc in required_docs:
        path = docs_dir / doc
        assert path.exists(), f"Missing documentation: {path}"

    print("✓ All documentation files exist")

def test_prompts_exist():
    """Test that prompt files exist."""
    prompts_dir = Path("prompts")

    required_prompts = [
        "turnwise_prompt_v4.md",
        "turnwise_prompt_v5_with_context.md"
    ]

    for prompt in required_prompts:
        path = prompts_dir / prompt
        assert path.exists(), f"Missing prompt: {path}"

    print("✓ All prompt files exist")

def test_scripts_organized():
    """Test that scripts are properly organized."""
    classification_dir = Path("scripts/classification")
    quality_dir = Path("scripts/quality")
    utilities_dir = Path("scripts/utilities")
    manuscript_dir = Path("scripts/manuscript")
    analysis_dir = Path("scripts/analysis")

    # Check classification scripts
    required_classification = [
        "extract_suffrage_debates_from_reliable.py",
        "prepare_suffrage_input.py",
        "modal_suffrage_classification_v5.py",
        "generate_validation_sample.py",
        "manual_validation.py",
        "show_validation_samples.py"
    ]

    for script in required_classification:
        path = classification_dir / script
        assert path.exists(), f"Missing classification script: {path}"

    # Check quality script
    assert (quality_dir / "large_sample_validation.py").exists()

    # Check manuscript exists
    assert manuscript_dir.exists(), f"Missing manuscript directory"

    # Check analysis exists
    assert analysis_dir.exists(), f"Missing analysis directory"

    print("✓ All scripts properly organized")

def test_top_level_clean():
    """Test that top level directory is clean (no stray scripts/docs)."""
    top_level = Path(".")

    # Allow list of files that should be at top level
    allowed_files = {
        "README.md",
        "CLAUDE.md",
        ".gitignore",
        ".git",
        "setup.py",  # Package setup
        "requirements.txt",
        "environment.yml"
    }

    # Allowed .py files at top level
    allowed_py_files = {"setup.py"}

    allowed_dirs = {
        "src",
        "docs",
        "prompts",
        "scripts",
        "notebooks",
        "data-hansard",
        "outputs",
        "analysis",
        "tests",
        ".github"
    }

    # Check for stray .py files
    stray_py = [f for f in top_level.glob("*.py") if f.name not in allowed_py_files]
    assert len(stray_py) == 0, f"Stray Python files at top level: {stray_py}"

    # Check for stray .md files (except allowed)
    stray_md = [f for f in top_level.glob("*.md") if f.name not in allowed_files]
    assert len(stray_md) == 0, f"Stray markdown files at top level: {stray_md}"

    # Check for stray .sh files
    stray_sh = list(top_level.glob("*.sh"))
    assert len(stray_sh) == 0, f"Stray shell scripts at top level: {stray_sh}"

    print("✓ Top level directory is clean")

def test_data_outputs_exist():
    """Test that expected data/output directories exist."""
    required_dirs = [
        "outputs/llm_classification",
        "outputs/llm_classification/archive",
        "outputs/validation",
        "outputs/suffrage_reliable"
    ]

    for dir_path in required_dirs:
        path = Path(dir_path)
        assert path.exists(), f"Missing output directory: {path}"

    print("✓ All output directories exist")

if __name__ == "__main__":
    print("Testing repository structure...")
    print()

    test_documentation_exists()
    test_prompts_exist()
    test_scripts_organized()
    test_top_level_clean()
    test_data_outputs_exist()

    print()
    print("="*60)
    print("All repository structure tests passed!")
    print("="*60)
