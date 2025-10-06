#!/usr/bin/env python3
"""
Verification script to ensure all systems are working.
Runs quick tests on each major component.
"""

import sys
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_status(name, success):
    """Print colored status."""
    symbol = f"{GREEN}PASS{RESET}" if success else f"{RED}FAIL{RESET}"
    print(f"  {name}: {symbol}")

def verify_data():
    """Verify data files exist."""
    print(f"\n{BOLD}Verifying Data...{RESET}")

    checks = [
        ("Processed data", Path("src/hansard/data/processed_fixed").exists()),
        ("Metadata", Path("src/hansard/data/processed_fixed/metadata/debates_master.parquet").exists()),
        ("Content", Path("src/hansard/data/processed_fixed/content").exists()),
        ("Gender wordlists", Path("src/hansard/data/gender_wordlists").exists()),
    ]

    for name, result in checks:
        print_status(name, result)

    return all(r for _, r in checks)

def verify_modules():
    """Verify Python modules import correctly."""
    print(f"\n{BOLD}Verifying Modules...{RESET}")

    sys.path.insert(0, "src")

    results = []

    # Test speaker processing
    try:
        from hansard.speaker_processing import SpeakerProcessor
        processor = SpeakerProcessor()
        results.append(("Speaker processing", True))
    except Exception as e:
        results.append(("Speaker processing", False))

    # Test data validator
    try:
        from hansard.utils.data_validator import DataValidator
        validator = DataValidator(data_dir=Path("src/hansard/data"))
        results.append(("Data validator", True))
    except Exception:
        results.append(("Data validator", False))

    # Test NLP analysis
    try:
        sys.path.insert(0, "src/hansard/analysis")
        from hansard_nlp_analysis import HansardAdvancedAnalyzer
        results.append(("NLP analysis", True))
    except Exception:
        results.append(("NLP analysis", False))

    for name, result in results:
        print_status(name, result)

    return all(r for _, r in results)

def verify_tests():
    """Verify tests run successfully."""
    print(f"\n{BOLD}Verifying Tests...{RESET}")

    import subprocess

    results = []

    # Run unit tests
    try:
        result = subprocess.run(
            ["python", "tests/unit/test_text_utils.py"],
            capture_output=True, text=True, timeout=10
        )
        results.append(("Unit tests", result.returncode == 0))
    except Exception:
        results.append(("Unit tests", False))

    # Run integration tests
    try:
        result = subprocess.run(
            ["python", "tests/integration/test_speaker_processing.py"],
            capture_output=True, text=True, timeout=10
        )
        results.append(("Integration tests", "All speaker processing tests passed" in result.stdout))
    except Exception:
        results.append(("Integration tests", False))

    for name, result in results:
        print_status(name, result)

    return all(r for _, r in results)

def verify_analysis():
    """Verify analysis can run on small sample."""
    print(f"\n{BOLD}Verifying Analysis Pipeline...{RESET}")

    import subprocess

    try:
        # Run tiny analysis
        result = subprocess.run(
            ["python", "src/hansard/analysis/hansard_nlp_analysis.py",
             "--years", "1920-1920", "--sample", "5"],
            capture_output=True, text=True, timeout=30, cwd="."
        )

        success = "Analysis complete" in result.stdout
        print_status("NLP analysis execution", success)

        # Check for key outputs
        checks = [
            ("Word extraction", "Top 10 words" in result.stdout),
            ("Gender analysis", "Gender word distribution" in result.stdout),
            ("Results saved", "Results saved" in result.stdout)
        ]

        for name, check in checks:
            print_status(name, check)

        return success and all(c for _, c in checks)

    except Exception as e:
        print_status("NLP analysis execution", False)
        return False

def main():
    """Run all verifications."""
    print("="*60)
    print(f"{BOLD}HANSARD NLP EXPLORER - SYSTEM VERIFICATION{RESET}")
    print("="*60)

    results = []

    # Run all checks
    results.append(("Data", verify_data()))
    results.append(("Modules", verify_modules()))
    results.append(("Tests", verify_tests()))
    results.append(("Analysis", verify_analysis()))

    # Summary
    print("\n" + "="*60)
    print(f"{BOLD}SUMMARY{RESET}")
    print("="*60)

    all_pass = True
    for system, passed in results:
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"{system}: {status}")
        if not passed:
            all_pass = False

    print("\n" + "="*60)
    if all_pass:
        print(f"{GREEN}{BOLD}ALL SYSTEMS OPERATIONAL{RESET}")
        print("Repository is clean, tested, and ready for use!")
    else:
        print(f"{RED}{BOLD}SOME SYSTEMS NEED ATTENTION{RESET}")
        print("Please check the failed components above.")
    print("="*60)

    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
