#!/usr/bin/env python3
"""
Final verification: Ensure we have 100% of what the index says should exist.

This is the final step that proves we're complete.
"""

import json
from pathlib import Path
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Verify completeness against an index")
    parser.add_argument("--index-file", default="analysis/hansard_complete_index.json", help="Path to index JSON")
    args = parser.parse_args()
    print("="*70)
    print("FINAL COMPLETENESS VERIFICATION")
    print("="*70)
    print()

    # Load index
    index_file = Path(args.index_file)
    if not index_file.exists():
        print("Error: Index file not found")
        sys.exit(1)

    with open(index_file) as f:
        index_data = json.load(f)
        index = index_data["index"]

    # Check every date
    data_dir = Path("data-hansard/hansard")

    total_dates = 0
    total_expected_debates = 0
    perfect_matches = 0
    close_matches = 0
    mismatches = 0
    missing_dates = 0

    mismatch_details = []

    for year, months in index.items():
        for month, days in months.items():
            for day, info in days.items():
                total_dates += 1
                expected = info["debate_count"]
                total_expected_debates += expected

                summary_file = data_dir / year / month / f"{day}_summary.json"

                if not summary_file.exists():
                    missing_dates += 1
                    mismatch_details.append({
                        "date": f"{year}/{month}/{day}",
                        "expected": expected,
                        "got": 0,
                        "status": "MISSING"
                    })
                    continue

                with open(summary_file) as f:
                    summary = json.load(f)
                    got = summary.get("debate_count", 0)

                if got == expected:
                    perfect_matches += 1
                elif abs(got - expected) <= max(1, expected * 0.05):  # Within 5% or 1 debate
                    close_matches += 1
                else:
                    mismatches += 1
                    if mismatches <= 50:  # Keep first 50 for review
                        mismatch_details.append({
                            "date": f"{year}/{month}/{day}",
                            "expected": expected,
                            "got": got,
                            "diff": got - expected,
                            "status": "MISMATCH"
                        })

    # Calculate completeness
    complete = perfect_matches + close_matches
    completeness = (complete / total_dates * 100) if total_dates > 0 else 0

    # Print results
    print(f"Total dates in index: {total_dates:,}")
    print(f"Total debates expected: {total_expected_debates:,}")
    print()
    print(f"Perfect matches: {perfect_matches:,} ({perfect_matches/total_dates*100:.1f}%)")
    print(f"Close matches: {close_matches:,} ({close_matches/total_dates*100:.1f}%)")
    print(f"Mismatches: {mismatches:,} ({mismatches/total_dates*100:.1f}%)")
    print(f"Missing: {missing_dates:,} ({missing_dates/total_dates*100:.1f}%)")
    print()
    print(f"Overall Completeness: {completeness:.2f}%")
    print()

    # Pass/Fail
    if completeness >= 99.9:
        print("✓✓✓ PERFECT! Data is complete!")
        status = "PERFECT"
    elif completeness >= 99.0:
        print("✓✓ EXCELLENT! Data is nearly complete.")
        status = "EXCELLENT"
    elif completeness >= 95.0:
        print("✓ GOOD! Data is mostly complete.")
        status = "GOOD"
    elif completeness >= 90.0:
        print("⚠️  ACCEPTABLE. Some issues remain.")
        status = "ACCEPTABLE"
    else:
        print("✗ INCOMPLETE. Significant issues remain.")
        status = "INCOMPLETE"

    # Show some mismatches
    if mismatch_details:
        print()
        print("Sample issues (first 20):")
        for item in mismatch_details[:20]:
            if item["status"] == "MISSING":
                print(f"  {item['date']}: MISSING (expected {item['expected']})")
            else:
                print(f"  {item['date']}: got {item['got']}, expected {item['expected']} (diff: {item['diff']:+d})")

    # Save report
    report_file = Path("analysis/completeness_report.json")
    with open(report_file, 'w') as f:
        json.dump({
            "status": status,
            "completeness": completeness,
            "stats": {
                "total_dates": total_dates,
                "total_expected_debates": total_expected_debates,
                "perfect_matches": perfect_matches,
                "close_matches": close_matches,
                "mismatches": mismatches,
                "missing_dates": missing_dates
            },
            "issues": mismatch_details
        }, f, indent=2)

    print()
    print(f"Report saved to: {report_file}")

    # If not complete, create new re-crawl list
    if completeness < 99.9 and mismatch_details:
        recrawl_file = Path("analysis/dates_to_recrawl_round2.txt")
        with open(recrawl_file, 'w') as f:
            for item in mismatch_details:
                f.write(f"{item['date']}\n")
        print(f"New re-crawl list saved to: {recrawl_file}")
        print()
        print("Run another round: python3 scripts/systematic_recrawl.py")

if __name__ == "__main__":
    main()
