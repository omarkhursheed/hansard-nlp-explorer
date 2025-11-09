#!/usr/bin/env python3
"""
Analyze manual validation results and calculate accuracy metrics.

Usage:
    python3 scripts/classification/analyze_validation_results.py

Input: outputs/validation/validation_recording_template.csv (filled by user)
Output: Comprehensive accuracy report with breakdown by category
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_validation_results():
    """Analyze manual validation results and print comprehensive report."""

    # Load validation results
    results_path = Path("outputs/validation/validation_recording_template.csv")

    if not results_path.exists():
        print(f"ERROR: Validation results not found at {results_path}")
        print("Please complete the manual validation first.")
        return

    df = pd.read_csv(results_path)

    # Check if validation is complete
    if df['stance_correct'].isna().all():
        print("ERROR: Validation template is empty.")
        print("Please fill in the validation results first.")
        print(f"See instructions: docs/suffrage_classification/MANUAL_VALIDATION_INSTRUCTIONS.md")
        return

    # Filter to completed rows
    completed = df[df['stance_correct'].notna()].copy()

    if len(completed) == 0:
        print("ERROR: No completed validations found.")
        return

    print("="*80)
    print("MANUAL VALIDATION RESULTS ANALYSIS")
    print("="*80)
    print(f"\nTotal speeches validated: {len(completed)}/{len(df)}")

    if len(completed) < len(df):
        print(f"WARNING: Only {len(completed)} of {len(df)} speeches validated")
        print("Results below are for completed validations only.\n")

    # Overall accuracy
    print("\n" + "="*80)
    print("OVERALL ACCURACY")
    print("="*80)

    stance_correct = (completed['stance_correct'].str.upper() == 'YES').sum()
    stance_accuracy = stance_correct / len(completed) * 100

    print(f"\nStance Classification:")
    print(f"  Correct: {stance_correct}/{len(completed)} ({stance_accuracy:.1f}%)")
    print(f"  Incorrect: {len(completed) - stance_correct}/{len(completed)} ({100-stance_accuracy:.1f}%)")

    # Reasons accuracy
    reasons_yes = (completed['reasons_correct'].str.upper() == 'YES').sum()
    reasons_partial = (completed['reasons_correct'].str.upper() == 'PARTIAL').sum()
    reasons_no = (completed['reasons_correct'].str.upper() == 'NO').sum()

    print(f"\nReason Extraction:")
    print(f"  Fully Correct: {reasons_yes}/{len(completed)} ({reasons_yes/len(completed)*100:.1f}%)")
    print(f"  Partially Correct: {reasons_partial}/{len(completed)} ({reasons_partial/len(completed)*100:.1f}%)")
    print(f"  Incorrect: {reasons_no}/{len(completed)} ({reasons_no/len(completed)*100:.1f}%)")

    # Quotes accuracy
    quotes_yes = (completed['quotes_accurate'].str.upper() == 'YES').sum()
    quotes_partial = (completed['quotes_accurate'].str.upper() == 'PARTIAL').sum()
    quotes_no = (completed['quotes_accurate'].str.upper() == 'NO').sum()

    print(f"\nQuote Extraction:")
    print(f"  Accurate: {quotes_yes}/{len(completed)} ({quotes_yes/len(completed)*100:.1f}%)")
    print(f"  Partially Accurate: {quotes_partial}/{len(completed)} ({quotes_partial/len(completed)*100:.1f}%)")
    print(f"  Inaccurate: {quotes_no}/{len(completed)} ({quotes_no/len(completed)*100:.1f}%)")

    # Accuracy by stance
    print("\n" + "="*80)
    print("ACCURACY BY STANCE")
    print("="*80)

    for stance in ['for', 'against', 'both', 'neutral', 'irrelevant']:
        stance_df = completed[completed['llm_stance'] == stance]
        if len(stance_df) > 0:
            correct = (stance_df['stance_correct'].str.upper() == 'YES').sum()
            accuracy = correct / len(stance_df) * 100
            print(f"\n{stance.upper()}: {correct}/{len(stance_df)} correct ({accuracy:.1f}%)")

    # Accuracy by confidence
    print("\n" + "="*80)
    print("ACCURACY BY CONFIDENCE LEVEL")
    print("="*80)

    for conf in sorted(completed['llm_confidence'].unique()):
        conf_df = completed[completed['llm_confidence'] == conf]
        if len(conf_df) > 0:
            correct = (conf_df['stance_correct'].str.upper() == 'YES').sum()
            accuracy = correct / len(conf_df) * 100
            print(f"\nConfidence {conf}: {correct}/{len(conf_df)} correct ({accuracy:.1f}%)")

    # False positive analysis
    print("\n" + "="*80)
    print("FALSE POSITIVE ANALYSIS")
    print("="*80)

    # Speeches marked substantive but should be irrelevant
    false_positives = completed[
        (completed['llm_stance'].isin(['for', 'against', 'both'])) &
        (completed['your_judgment'].str.lower() == 'irrelevant')
    ]

    if len(false_positives) > 0:
        print(f"\nFalse Positives (classified as substantive, actually irrelevant):")
        print(f"  Count: {len(false_positives)}")
        print(f"  Rate: {len(false_positives)/len(completed)*100:.1f}% of all speeches")
        print(f"\n  Breakdown:")
        for stance, count in false_positives['llm_stance'].value_counts().items():
            print(f"    {stance.upper()}: {count}")
    else:
        print("\nNo false positives found! (speeches marked substantive were actually substantive)")

    # False negatives (marked irrelevant but should be substantive)
    false_negatives = completed[
        (completed['llm_stance'] == 'irrelevant') &
        (completed['your_judgment'].str.lower().isin(['for', 'against', 'both']))
    ]

    if len(false_negatives) > 0:
        print(f"\nFalse Negatives (classified as irrelevant, actually substantive):")
        print(f"  Count: {len(false_negatives)}")
        print(f"  Rate: {len(false_negatives)/len(completed)*100:.1f}% of all speeches")
    else:
        print("\nNo false negatives found! (speeches marked irrelevant were actually irrelevant)")

    # Error patterns
    print("\n" + "="*80)
    print("ERROR PATTERNS")
    print("="*80)

    errors = completed[completed['stance_correct'].str.upper() != 'YES']

    if len(errors) > 0:
        print(f"\nTotal errors: {len(errors)}")
        print(f"\nError details:")
        for idx, row in errors.iterrows():
            print(f"\n  Speech: {row['speech_id']}")
            print(f"  LLM: {row['llm_stance']} (conf={row['llm_confidence']})")
            print(f"  Correct: {row['your_judgment']}")
            if pd.notna(row['notes']):
                print(f"  Notes: {row['notes']}")
    else:
        print("\nNo errors found! Perfect classification.")

    # Gender breakdown
    print("\n" + "="*80)
    print("ACCURACY BY GENDER")
    print("="*80)

    for gender in ['M', 'F']:
        gender_df = completed[completed['gender'] == gender]
        if len(gender_df) > 0:
            correct = (gender_df['stance_correct'].str.upper() == 'YES').sum()
            accuracy = correct / len(gender_df) * 100
            label = 'Male' if gender == 'M' else 'Female'
            print(f"\n{label} MPs: {correct}/{len(gender_df)} correct ({accuracy:.1f}%)")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nValidation Sample Composition:")
    print(f"  Total speeches: {len(completed)}")
    print(f"  Stance distribution:")
    for stance, count in completed['llm_stance'].value_counts().items():
        print(f"    {stance}: {count} ({count/len(completed)*100:.1f}%)")

    print(f"\n  Confidence distribution:")
    for conf in sorted(completed['llm_confidence'].unique()):
        count = (completed['llm_confidence'] == conf).sum()
        print(f"    {conf}: {count} ({count/len(completed)*100:.1f}%)")

    print(f"\n  Gender distribution:")
    for gender in ['M', 'F']:
        count = (completed['gender'] == gender).sum()
        if count > 0:
            label = 'Male' if gender == 'M' else 'Female'
            print(f"    {label}: {count} ({count/len(completed)*100:.1f}%)")

    # Save summary
    summary_path = Path("outputs/validation/validation_results_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Manual Validation Results Summary\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"Speeches validated: {len(completed)}/{len(df)}\n")
        f.write(f"Overall stance accuracy: {stance_accuracy:.1f}%\n")
        f.write(f"Reasons fully correct: {reasons_yes/len(completed)*100:.1f}%\n")
        f.write(f"Quotes accurate: {quotes_yes/len(completed)*100:.1f}%\n")
        f.write(f"\nFalse positives: {len(false_positives)}\n")
        f.write(f"False negatives: {len(false_negatives)}\n")

    print(f"\n" + "="*80)
    print(f"Summary saved to: {summary_path}")
    print("="*80)

if __name__ == '__main__':
    analyze_validation_results()
