#!/usr/bin/env python3
"""
Research CLI - Scientific workflow for Hansard gender analysis.

Demonstrates:
1. Formulating a scientific question
2. Random sampling to avoid cherry-picking
3. Blinded annotation to avoid confirmation bias
4. Cohen's h effect size calculation with 95% CI
5. Power analysis

Usage:
    python research_cli.py --question "personal anecdotes" --sample 50
"""

import argparse
import json
import math
import random
import re
from pathlib import Path
from datetime import datetime

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data-hansard' / 'derived_v2' / 'speeches_complete'
ANNOTATIONS_FILE = Path(__file__).parent / 'research_annotations.json'


def load_speeches(years=None, sample_size=None, query=None, gender=None):
    """Load speeches from parquet files with optional filtering."""
    speeches = []

    if not DATA_DIR.exists():
        alt_dir = PROJECT_ROOT / 'data-hansard' / 'derived_complete' / 'speeches_complete'
        if alt_dir.exists():
            data_dir = alt_dir
        else:
            print(f"Data directory not found: {DATA_DIR}")
            return []
    else:
        data_dir = DATA_DIR

    files = sorted(data_dir.glob('speeches_*.parquet'))

    if years:
        year_set = set(years)
        files = [f for f in files if int(f.stem.split('_')[1]) in year_set]

    print(f"Loading from {len(files)} year files...")

    for f in files:
        df = pd.read_parquet(f)

        # Filter by gender if specified
        if gender:
            df = df[df['gender'] == gender]

        # Filter by query if specified (search in text)
        if query:
            pattern = query.lower()
            mask = df['text'].fillna('').str.lower().str.contains(pattern, regex=False)
            df = df[mask]

        for _, row in df.iterrows():
            speeches.append({
                'id': f"{row.get('debate_id', '')}_{row.get('sequence', '')}",
                'speaker': row.get('canonical_name') or row.get('speaker', 'Unknown'),
                'gender': row.get('gender', ''),
                'year': int(f.stem.split('_')[1]),
                'text': str(row.get('text', ''))[:500],
                'title': str(row.get('title', ''))[:100],
            })

    print(f"Found {len(speeches)} speeches matching criteria")

    # Random sample
    if sample_size and len(speeches) > sample_size:
        speeches = random.sample(speeches, sample_size)
        print(f"Random sampled {sample_size} speeches")

    return speeches


def load_annotations():
    """Load existing annotations."""
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE) as f:
            return json.load(f)
    return {'config': {}, 'annotations': {}}


def save_annotations(data):
    """Save annotations to file."""
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def calculate_cohens_h(female_pos, female_total, male_pos, male_total):
    """
    Calculate Cohen's h effect size for proportions.

    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

    Interpretation:
    - |h| < 0.2: negligible
    - 0.2 <= |h| < 0.5: small
    - 0.5 <= |h| < 0.8: medium
    - |h| >= 0.8: large
    """
    if female_total < 5 or male_total < 5:
        return None, None, None

    p1 = female_pos / female_total
    p2 = male_pos / male_total

    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))

    h = phi1 - phi2

    # 95% CI: SE of Cohen's h ~ sqrt(1/n) where n is harmonic mean
    n_harm = 2 * female_total * male_total / (female_total + male_total)
    se = math.sqrt(1 / n_harm)
    ci_low = h - 1.96 * se
    ci_high = h + 1.96 * se

    return h, ci_low, ci_high


def power_analysis(h, alpha=0.05, power=0.80):
    """
    Calculate required sample size per group for given effect size.

    n = 2 * ((z_alpha + z_beta) / h)^2
    """
    if h is None or abs(h) < 0.05:
        return None

    # z-scores for alpha=0.05 (two-tailed) and power=0.80
    z_alpha = 1.96
    z_beta = 0.84

    n_per_group = math.ceil(2 * ((z_alpha + z_beta) / abs(h)) ** 2)
    return n_per_group


def annotate_interactive(speeches, blinded=True):
    """
    Interactive annotation loop.

    In blinded mode, gender is hidden during annotation.
    """
    data = load_annotations()
    annotations = data.get('annotations', {})

    # Filter out already annotated
    unannotated = [s for s in speeches if s['id'] not in annotations]

    if not unannotated:
        print("All speeches already annotated!")
        return annotations

    print(f"\n{'='*60}")
    print("ANNOTATION MODE" + (" (BLINDED)" if blinded else ""))
    print("="*60)
    print("Commands: + (positive), - (negative), s (skip), q (quit)")
    print(f"Remaining: {len(unannotated)} speeches")
    print("="*60 + "\n")

    for i, speech in enumerate(unannotated):
        print(f"\n--- Speech {i+1}/{len(unannotated)} ---")

        if blinded:
            print(f"Speaker: [HIDDEN]")
            print(f"Year: {speech['year']}")
        else:
            print(f"Speaker: {speech['speaker']} ({speech['gender']})")
            print(f"Year: {speech['year']}")

        print(f"Title: {speech['title']}")
        print(f"\nText:\n{speech['text'][:400]}...")

        while True:
            choice = input("\nAnnotate [+/-/s/q]: ").strip().lower()

            if choice == '+':
                annotations[speech['id']] = {
                    'label': 'positive',
                    'gender': speech['gender'],
                    'speaker': speech['speaker'],
                    'year': speech['year'],
                    'timestamp': datetime.now().isoformat()
                }
                print("-> Marked POSITIVE")
                break
            elif choice == '-':
                annotations[speech['id']] = {
                    'label': 'negative',
                    'gender': speech['gender'],
                    'speaker': speech['speaker'],
                    'year': speech['year'],
                    'timestamp': datetime.now().isoformat()
                }
                print("-> Marked NEGATIVE")
                break
            elif choice == 's':
                print("-> Skipped")
                break
            elif choice == 'q':
                print("\nSaving and exiting...")
                data['annotations'] = annotations
                save_annotations(data)
                return annotations
            else:
                print("Invalid choice. Use +, -, s, or q")

        # Auto-save every 5 annotations
        if len(annotations) % 5 == 0:
            data['annotations'] = annotations
            save_annotations(data)

    data['annotations'] = annotations
    save_annotations(data)
    return annotations


def analyze_results():
    """Analyze current annotations and calculate effect size."""
    data = load_annotations()
    annotations = data.get('annotations', {})

    if not annotations:
        print("No annotations yet. Run with --annotate first.")
        return

    # Compute statistics
    female_anns = [a for a in annotations.values() if a.get('gender') == 'F']
    male_anns = [a for a in annotations.values() if a.get('gender') == 'M']

    female_pos = len([a for a in female_anns if a['label'] == 'positive'])
    male_pos = len([a for a in male_anns if a['label'] == 'positive'])

    female_total = len(female_anns)
    male_total = len(male_anns)

    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)

    print(f"\nSample sizes:")
    print(f"  Female: {female_total} annotated ({female_pos} positive)")
    print(f"  Male:   {male_total} annotated ({male_pos} positive)")

    if female_total > 0:
        print(f"\nPositive rate - Female: {female_pos/female_total*100:.1f}%")
    if male_total > 0:
        print(f"Positive rate - Male:   {male_pos/male_total*100:.1f}%")

    # Effect size
    h, ci_low, ci_high = calculate_cohens_h(female_pos, female_total, male_pos, male_total)

    if h is not None:
        print(f"\n{'='*60}")
        print("EFFECT SIZE")
        print("="*60)
        print(f"Cohen's h: {h:.3f}")
        print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

        # Interpretation
        abs_h = abs(h)
        if abs_h < 0.2:
            size = "negligible"
        elif abs_h < 0.5:
            size = "small"
        elif abs_h < 0.8:
            size = "medium"
        else:
            size = "large"

        direction = "Female MPs higher" if h > 0 else "Male MPs higher"
        if abs_h < 0.2:
            direction = "No meaningful difference"

        significant = (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)

        print(f"\nInterpretation: {size.upper()} effect")
        print(f"Direction: {direction}")
        print(f"Significant: {'YES (CI does not cross zero)' if significant else 'NO (CI crosses zero)'}")

        # Power analysis
        n_needed = power_analysis(h)
        if n_needed:
            current_min = min(female_total, male_total)
            progress = min(100, current_min / n_needed * 100)
            print(f"\n{'='*60}")
            print("POWER ANALYSIS")
            print("="*60)
            print(f"For 80% power to detect h={abs_h:.2f}:")
            print(f"  Need: {n_needed} per group")
            print(f"  Have: {current_min} (min of F/M)")
            print(f"  Progress: {progress:.0f}%")
            if progress < 100:
                print(f"  Need {n_needed - current_min} more per group")
    else:
        print("\nInsufficient data for effect size calculation.")
        print("Need at least 5 annotations per gender group.")

    # Research summary
    config = data.get('config', {})
    if config.get('question'):
        print(f"\n{'='*60}")
        print("RESEARCH SUMMARY")
        print("="*60)
        print(f"Question: {config.get('question', 'Not specified')}")
        print(f"Hypothesis: {config.get('hypothesis', 'Not specified')}")


def setup_question(question, hypothesis):
    """Set up research question and hypothesis."""
    data = load_annotations()
    data['config'] = {
        'question': question,
        'hypothesis': hypothesis,
        'created': datetime.now().isoformat()
    }
    save_annotations(data)
    print(f"Research question saved: {question}")


def main():
    parser = argparse.ArgumentParser(description='Hansard Research CLI')
    parser.add_argument('--question', help='Set research question')
    parser.add_argument('--hypothesis', help='Set hypothesis')
    parser.add_argument('--query', help='Search query for speeches')
    parser.add_argument('--years', help='Years to include (e.g., 1990-2000)')
    parser.add_argument('--sample', type=int, default=50, help='Sample size')
    parser.add_argument('--annotate', action='store_true', help='Enter annotation mode')
    parser.add_argument('--blinded', action='store_true', help='Hide gender during annotation')
    parser.add_argument('--analyze', action='store_true', help='Analyze current annotations')
    parser.add_argument('--clear', action='store_true', help='Clear all annotations')

    args = parser.parse_args()

    # Set up question if provided
    if args.question:
        setup_question(args.question, args.hypothesis or '')

    # Clear annotations
    if args.clear:
        if input("Clear all annotations? [y/N]: ").lower() == 'y':
            save_annotations({'config': {}, 'annotations': {}})
            print("Annotations cleared.")
        return

    # Analyze existing
    if args.analyze:
        analyze_results()
        return

    # Parse years
    years = None
    if args.years:
        if '-' in args.years:
            start, end = args.years.split('-')
            years = list(range(int(start), int(end) + 1))
        else:
            years = [int(y) for y in args.years.split(',')]

    # Load and annotate
    if args.annotate or args.query:
        speeches = load_speeches(
            years=years,
            sample_size=args.sample,
            query=args.query
        )

        if speeches and args.annotate:
            annotate_interactive(speeches, blinded=args.blinded)
            analyze_results()
    else:
        # Show usage
        print("Hansard Research CLI")
        print("="*40)
        print("\nExample workflow:")
        print()
        print("1. Set up research question:")
        print('   python research_cli.py --question "Do female MPs use more personal anecdotes?"')
        print()
        print("2. Sample and annotate (blinded):")
        print('   python research_cli.py --query "constituent" --sample 50 --annotate --blinded')
        print()
        print("3. Analyze results:")
        print('   python research_cli.py --analyze')
        print()
        print("4. Clear and restart:")
        print('   python research_cli.py --clear')


if __name__ == '__main__':
    main()
