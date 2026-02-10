#!/usr/bin/env python3
"""
Research Demo - Demonstrates complete scientific workflow with real Hansard data.

Scientific Question: Do female MPs use more personal anecdotes in speeches?

This script:
1. Defines a clear research question and hypothesis
2. Samples speeches containing personal reference markers
3. Auto-annotates based on objective criteria (for demo purposes)
4. Calculates Cohen's h effect size with 95% CI
5. Performs power analysis
"""

import math
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data-hansard' / 'derived_v2' / 'speeches_complete'


def load_sample(query_terms, years, sample_per_gender=100):
    """Load balanced sample of speeches containing query terms."""

    if not DATA_DIR.exists():
        alt = PROJECT_ROOT / 'data-hansard' / 'derived_complete' / 'speeches_complete'
        data_dir = alt if alt.exists() else None
    else:
        data_dir = DATA_DIR

    if not data_dir:
        print("Data directory not found!")
        return [], []

    female_speeches = []
    male_speeches = []

    files = [f for f in sorted(data_dir.glob('speeches_*.parquet'))
             if int(f.stem.split('_')[1]) in years]

    print(f"Searching {len(files)} year files for: {query_terms}")

    for f in files:
        df = pd.read_parquet(f)
        year = int(f.stem.split('_')[1])

        for _, row in df.iterrows():
            text = str(row.get('text', '')).lower()

            # Check if any query term is present
            if not any(term.lower() in text for term in query_terms):
                continue

            speech = {
                'id': f"{row.get('debate_id', '')}_{row.get('sequence', '')}",
                'speaker': row.get('canonical_name') or row.get('speaker', 'Unknown'),
                'gender': row.get('gender', ''),
                'year': year,
                'text': text[:1000],
                'title': str(row.get('title', ''))[:100],
            }

            if speech['gender'] == 'F':
                female_speeches.append(speech)
            elif speech['gender'] == 'M':
                male_speeches.append(speech)

    print(f"Found: {len(female_speeches)} female, {len(male_speeches)} male speeches")

    # Random sample (balanced)
    if len(female_speeches) > sample_per_gender:
        female_speeches = random.sample(female_speeches, sample_per_gender)
    if len(male_speeches) > sample_per_gender:
        male_speeches = random.sample(male_speeches, sample_per_gender)

    return female_speeches, male_speeches


def annotate_automatically(speeches, positive_markers, negative_markers):
    """
    Auto-annotate speeches based on objective text markers.

    For a real study, this would be done manually with blinded review.
    This demo uses text patterns to illustrate the workflow.
    """
    annotations = []

    for speech in speeches:
        text = speech['text']

        # Count positive markers (personal anecdotes)
        pos_count = sum(1 for m in positive_markers if m.lower() in text)

        # Count negative markers (impersonal/statistical language)
        neg_count = sum(1 for m in negative_markers if m.lower() in text)

        # Classify based on which dominates
        if pos_count > neg_count:
            label = 'positive'
        elif neg_count > pos_count:
            label = 'negative'
        else:
            # Ambiguous - randomly assign for demo (in real study, would skip or use other criteria)
            label = random.choice(['positive', 'negative'])

        annotations.append({
            'id': speech['id'],
            'gender': speech['gender'],
            'label': label,
            'pos_markers': pos_count,
            'neg_markers': neg_count,
        })

    return annotations


def calculate_cohens_h(female_pos, female_total, male_pos, male_total):
    """Calculate Cohen's h with 95% CI."""
    if female_total < 5 or male_total < 5:
        return None, None, None

    p1 = female_pos / female_total
    p2 = male_pos / male_total

    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))

    h = phi1 - phi2

    n_harm = 2 * female_total * male_total / (female_total + male_total)
    se = math.sqrt(1 / n_harm)
    ci_low = h - 1.96 * se
    ci_high = h + 1.96 * se

    return h, ci_low, ci_high


def power_analysis(h):
    """Calculate N needed for 80% power."""
    if h is None or abs(h) < 0.05:
        return None
    z = 1.96 + 0.84
    return math.ceil(2 * (z / abs(h)) ** 2)


def main():
    print("="*70)
    print("HANSARD RESEARCH DEMO - SCIENTIFIC WORKFLOW")
    print("="*70)

    # 1. RESEARCH QUESTION
    print("\n" + "="*70)
    print("STEP 1: RESEARCH QUESTION")
    print("="*70)
    question = "Do female MPs use more personal anecdotes in parliamentary speeches?"
    hypothesis = "Female MPs are more likely to reference personal experiences, constituent stories, and 'I have seen' type narratives compared to male MPs."

    print(f"\nQuestion: {question}")
    print(f"\nHypothesis: {hypothesis}")

    # 2. DEFINE EVIDENCE CRITERIA
    print("\n" + "="*70)
    print("STEP 2: DEFINE EVIDENCE CRITERIA")
    print("="*70)

    # Markers of personal anecdotes (positive evidence)
    positive_markers = [
        "my constituent",
        "i have seen",
        "in my experience",
        "i remember",
        "told me",
        "came to see me",
        "wrote to me",
        "in my constituency",
        "i met",
        "personal experience",
    ]

    # Markers of impersonal/data-driven speech (negative evidence)
    negative_markers = [
        "statistics show",
        "the data",
        "research indicates",
        "according to",
        "the evidence suggests",
        "studies have shown",
        "percent of",
        "the report",
    ]

    print("\nPositive evidence markers (personal anecdotes):")
    for m in positive_markers[:5]:
        print(f"  - '{m}'")
    print(f"  ... and {len(positive_markers)-5} more")

    print("\nNegative evidence markers (impersonal/data-driven):")
    for m in negative_markers[:5]:
        print(f"  - '{m}'")
    print(f"  ... and {len(negative_markers)-5} more")

    # 3. SAMPLE DATA
    print("\n" + "="*70)
    print("STEP 3: RANDOM SAMPLING")
    print("="*70)

    # Search for speeches with any personal reference marker
    search_terms = ["constituent", "experience", "told me", "in my"]
    years = range(1980, 2006)  # Modern era with better gender data

    female_speeches, male_speeches = load_sample(
        search_terms,
        years=set(years),
        sample_per_gender=150
    )

    print(f"\nSampled: {len(female_speeches)} female, {len(male_speeches)} male")

    if len(female_speeches) < 20 or len(male_speeches) < 20:
        print("Insufficient data for analysis. Need more speeches.")
        return

    # 4. ANNOTATE (automated for demo)
    print("\n" + "="*70)
    print("STEP 4: ANNOTATION (automated for demo)")
    print("="*70)
    print("\nIn a real study, annotation would be done manually with blinded review.")
    print("For this demo, we use objective text markers to classify speeches.")

    female_annotations = annotate_automatically(female_speeches, positive_markers, negative_markers)
    male_annotations = annotate_automatically(male_speeches, positive_markers, negative_markers)

    # Count results
    female_pos = sum(1 for a in female_annotations if a['label'] == 'positive')
    female_neg = sum(1 for a in female_annotations if a['label'] == 'negative')
    male_pos = sum(1 for a in male_annotations if a['label'] == 'positive')
    male_neg = sum(1 for a in male_annotations if a['label'] == 'negative')

    female_total = female_pos + female_neg
    male_total = male_pos + male_neg

    print(f"\nFemale: {female_pos} positive, {female_neg} negative (total: {female_total})")
    print(f"Male:   {male_pos} positive, {male_neg} negative (total: {male_total})")

    female_rate = female_pos / female_total if female_total > 0 else 0
    male_rate = male_pos / male_total if male_total > 0 else 0

    print(f"\nPersonal anecdote rate:")
    print(f"  Female: {female_rate*100:.1f}%")
    print(f"  Male:   {male_rate*100:.1f}%")

    # 5. EFFECT SIZE CALCULATION
    print("\n" + "="*70)
    print("STEP 5: EFFECT SIZE CALCULATION")
    print("="*70)

    h, ci_low, ci_high = calculate_cohens_h(female_pos, female_total, male_pos, male_total)

    if h is not None:
        print(f"\nCohen's h: {h:.3f}")
        print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

        # Visualize CI (ASCII forest plot)
        print("\nForest Plot:")
        print("-1.0                    0                    +1.0")
        print("|" + "-"*20 + "|" + "-"*20 + "|")

        # Scale to 40 chars (-1 to +1)
        def scale(val):
            return int((val + 1) / 2 * 40)

        ci_lo_pos = max(0, min(40, scale(ci_low)))
        ci_hi_pos = max(0, min(40, scale(ci_high)))
        h_pos = max(0, min(40, scale(h)))

        line = [' '] * 41
        line[20] = '|'  # Zero line

        for i in range(ci_lo_pos, ci_hi_pos + 1):
            line[i] = '-'
        line[h_pos] = '*'

        print(' ' + ''.join(line))
        print("        Male higher           Female higher")

        # Interpretation
        abs_h = abs(h)
        if abs_h < 0.2:
            size = "NEGLIGIBLE"
        elif abs_h < 0.5:
            size = "SMALL"
        elif abs_h < 0.8:
            size = "MEDIUM"
        else:
            size = "LARGE"

        significant = (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)

        print(f"\nInterpretation:")
        print(f"  Effect size: {size} (|h| = {abs_h:.2f})")

        if h > 0.2:
            print(f"  Direction: Female MPs show HIGHER rates of personal anecdotes")
        elif h < -0.2:
            print(f"  Direction: Male MPs show HIGHER rates of personal anecdotes")
        else:
            print(f"  Direction: No meaningful difference between genders")

        print(f"  Statistically significant: {'YES' if significant else 'NO'} (CI {'does not cross' if significant else 'crosses'} zero)")

        # 6. POWER ANALYSIS
        print("\n" + "="*70)
        print("STEP 6: POWER ANALYSIS")
        print("="*70)

        n_needed = power_analysis(h)
        if n_needed:
            current_min = min(female_total, male_total)
            progress = min(100, current_min / n_needed * 100)

            print(f"\nFor 80% power to detect effect h = {abs_h:.2f}:")
            print(f"  Required: {n_needed} per group")
            print(f"  Current:  {current_min} per group (min)")
            print(f"  Progress: {progress:.0f}%")

            if progress >= 100:
                print("\n  -> Sample size SUFFICIENT for reliable conclusions")
            else:
                print(f"\n  -> Need {n_needed - current_min} more annotations per group")

    # 7. SUMMARY
    print("\n" + "="*70)
    print("RESEARCH SUMMARY")
    print("="*70)
    print(f"\nQuestion: {question}")
    print(f"\nSample: {female_total + male_total} speeches ({female_total} female, {male_total} male)")
    print(f"Years: {min(years)}-{max(years)}")

    if h is not None:
        if abs(h) < 0.2:
            conclusion = "No meaningful gender difference detected in use of personal anecdotes."
        elif h > 0:
            conclusion = f"Female MPs show higher rates of personal anecdotes (h = {h:.2f})."
        else:
            conclusion = f"Male MPs show higher rates of personal anecdotes (h = {h:.2f})."

        print(f"\nFinding: {conclusion}")

        if significant:
            print("This result IS statistically significant at the 95% confidence level.")
        else:
            print("This result is NOT statistically significant at the 95% confidence level.")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
