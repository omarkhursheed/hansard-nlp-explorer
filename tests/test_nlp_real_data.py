"""
Test NLP analysis with real Hansard data.
Ensures analysis produces real results from actual debates.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_nlp_analysis_small_sample():
    """Test NLP analysis with a small sample of real data."""
    print("Testing NLP analysis with real data sample...")

    # Navigate to the analysis directory
    import os
    os.chdir('src/hansard/analysis')

    try:
        # Import the NLP analysis module
        from hansard_nlp_analysis import main

        # Run with a very small sample to test
        print("Running analysis on 10 debates from 1920...")

        # We need to check if the data exists first
        content_dir = Path("../data/processed_fixed/content")
        if not content_dir.exists():
            print(f"[FAIL] Content directory not found: {content_dir}")
            return False

        # Check for 1920 data
        year_1920 = content_dir / "1920"
        if not year_1920.exists():
            # Try any year that exists
            years = [d.name for d in content_dir.iterdir() if d.is_dir()]
            if years:
                test_year = sorted(years)[len(years)//2]  # Pick middle year
                print(f"  1920 not found, using {test_year} instead")
            else:
                print("[FAIL] No year directories found in content")
                return False
        else:
            test_year = "1920"

        # Create test arguments
        class Args:
            years = f"{test_year}-{test_year}"
            sample = 10
            full = False

        # Mock sys.argv for the script
        import sys
        old_argv = sys.argv
        sys.argv = ['test', '--years', Args.years, '--sample', str(Args.sample)]

        try:
            # Run the analysis
            results = main()

            # Check results
            if results:
                print(f"[OK] Analysis completed successfully")

                # Verify results structure
                expected_keys = ['metadata', 'unigrams', 'bigrams']
                for key in expected_keys:
                    if key in results:
                        print(f"  [OK] Found {key} in results")
                    else:
                        print(f"  [FAIL] Missing {key} in results")

                # Check for real data
                if 'unigrams' in results and results['unigrams']:
                    print(f"  Top words: {list(results['unigrams'].keys())[:5]}")

                if 'bigrams' in results and results['bigrams']:
                    print(f"  Top bigrams: {list(results['bigrams'].keys())[:3]}")

                if 'topics' in results and results['topics']:
                    print(f"  Found {len(results['topics'])} topics")

                return True
            else:
                print("[FAIL] Analysis returned no results")
                return False

        finally:
            sys.argv = old_argv

    except ImportError as e:
        print(f"[FAIL] Failed to import analysis module: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Return to original directory
        os.chdir('../../..')


def test_gender_analysis():
    """Test gender analysis functionality."""
    print("\nTesting gender analysis with real wordlists...")

    # Check if gender wordlists exist
    wordlist_dir = Path("src/hansard/data/gender_wordlists")
    if not wordlist_dir.exists():
        print(f"[FAIL] Gender wordlist directory not found: {wordlist_dir}")
        return False

    # List available wordlists
    wordlists = list(wordlist_dir.glob("*.txt"))
    if wordlists:
        print(f"[OK] Found {len(wordlists)} wordlist files:")
        for wl in wordlists[:5]:
            print(f"    {wl.name}")

        # Try to load and use wordlists
        male_words = set()
        female_words = set()

        male_file = wordlist_dir / "male_words.txt"
        female_file = wordlist_dir / "female_words.txt"

        if male_file.exists():
            with open(male_file, 'r') as f:
                male_words = set(line.strip().lower() for line in f if line.strip())
            print(f"  [OK] Loaded {len(male_words)} male words")
            print(f"    Sample: {list(male_words)[:5]}")

        if female_file.exists():
            with open(female_file, 'r') as f:
                female_words = set(line.strip().lower() for line in f if line.strip())
            print(f"  [OK] Loaded {len(female_words)} female words")
            print(f"    Sample: {list(female_words)[:5]}")

        # Test on sample text
        sample_text = "The chairman and his wife attended the meeting with the congresswoman"
        words = sample_text.lower().split()

        male_count = sum(1 for w in words if w in male_words)
        female_count = sum(1 for w in words if w in female_words)

        print(f"\n  Sample text analysis:")
        print(f"    Male words found: {male_count}")
        print(f"    Female words found: {female_count}")

        return True
    else:
        print("[FAIL] No wordlist files found")
        return False


def test_stop_words():
    """Test that stop words module works."""
    print("\nTesting stop words module...")

    try:
        os.chdir('src/hansard/analysis')
        from stop_words import get_extended_stop_words, get_parliamentary_stop_words

        # Get stop words
        extended = get_extended_stop_words()
        parliamentary = get_parliamentary_stop_words()

        print(f"[OK] Extended stop words: {len(extended)} words")
        print(f"  Sample: {list(extended)[:10]}")

        print(f"[OK] Parliamentary stop words: {len(parliamentary)} words")
        print(f"  Sample: {list(parliamentary)[:10]}")

        # Test that they filter correctly
        test_sentence = "The honourable member for the constituency asked about brexit"
        words = test_sentence.lower().split()
        filtered = [w for w in words if w not in extended]

        print(f"\n  Original: {test_sentence}")
        print(f"  Filtered: {' '.join(filtered)}")

        os.chdir('../../..')
        return True

    except Exception as e:
        print(f"[FAIL] Stop words test failed: {e}")
        os.chdir('../../..')
        return False


if __name__ == "__main__":
    print("="*60)
    print("REAL DATA TESTS FOR NLP ANALYSIS")
    print("="*60)

    import os
    os.chdir('/Users/omarkhursheed/workplace/hansard-nlp-explorer')

    results = []

    # Run tests
    results.append(("Stop words", test_stop_words()))
    results.append(("Gender analysis", test_gender_analysis()))
    results.append(("NLP analysis", test_nlp_analysis_small_sample()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))
