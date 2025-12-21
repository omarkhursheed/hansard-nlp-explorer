#!/usr/bin/env python3
"""
Investigate potential misclassifications in LLM validation data.
Looking for speeches about OTHER groups (working class, students) incorrectly tagged as women's suffrage.
"""

import pandas as pd
from pathlib import Path

def investigate_misclassifications():
    """Look for misclassified speeches in the LLM results."""

    results_file = Path('outputs/llm_classification/full_results_v5_context_3_complete.parquet')
    speeches_file = Path('outputs/suffrage_reliable/speeches_reliable.parquet')

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return

    if not speeches_file.exists():
        print(f"Speeches file not found: {speeches_file}")
        return

    # Load LLM results
    df = pd.read_parquet(results_file)
    print(f"Total classified speeches: {len(df)}")
    print(f"\nStance distribution:")
    print(df['stance'].value_counts())

    # Load original speeches to get text
    speeches = pd.read_parquet(speeches_file)
    print(f"\nTotal original speeches: {len(speeches)}")

    # Join to get text
    df = df.merge(speeches[['speech_id', 'text']], on='speech_id', how='left')
    print(f"After join: {len(df)} speeches with text")

    # Look for speeches that are NOT irrelevant but might be about other groups
    relevant = df[df['stance'] != 'irrelevant'].copy()

    print(f"\n\nRelevant speeches (not marked irrelevant): {len(relevant)}")

    # Search for keywords suggesting other suffrage movements
    keywords_other = [
        'working class',
        'working-class',
        'workingclass',
        'labour',
        'labouring',
        'university',
        'universities',
        'student',
        'students',
        'household',
        'householder',
        'lodger',
        'lodgers',
        'occupier',
        'occupiers',
        'ratepayer',
        'ratepayers',
        'plural',
        'adult suffrage',
        'universal suffrage',
        'manhood suffrage',
    ]

    # Check for these keywords in speeches NOT marked irrelevant
    suspicious = []

    for idx, row in relevant.iterrows():
        if pd.isna(row['text']):
            continue

        text_lower = row['text'].lower()

        # Check for keywords
        found_keywords = [kw for kw in keywords_other if kw in text_lower]

        if found_keywords:
            # Check if "women" or "female" appears
            has_women = 'women' in text_lower or 'female' in text_lower

            suspicious.append({
                'speech_id': row.get('speech_id', idx),
                'date': row.get('date', 'unknown'),
                'speaker': row.get('speaker_name', 'unknown'),
                'stance': row['stance'],
                'keywords_found': ', '.join(found_keywords),
                'has_women': has_women,
                'text_snippet': row['text'][:300] + '...'
            })

    print(f"\n\nSuspicious speeches (non-irrelevant + other-group keywords): {len(suspicious)}")

    if suspicious:
        suspicious_df = pd.DataFrame(suspicious)

        print("\n" + "="*80)
        print("POTENTIAL MISCLASSIFICATIONS")
        print("="*80)

        # Show examples by stance
        for stance in ['for', 'against', 'both', 'neutral']:
            stance_examples = suspicious_df[suspicious_df['stance'] == stance]
            if len(stance_examples) > 0:
                print(f"\n{stance.upper()} stance with other-group keywords: {len(stance_examples)}")

                # Show up to 5 examples
                for i, row in stance_examples.head(5).iterrows():
                    print(f"\n  Date: {row['date']}")
                    print(f"  Speaker: {row['speaker']}")
                    print(f"  Keywords: {row['keywords_found']}")
                    print(f"  Has 'women': {row['has_women']}")
                    print(f"  Text: {row['text_snippet']}")

        # Save to CSV for inspection
        output_file = Path('outputs/llm_classification/potential_misclassifications.csv')
        suspicious_df.to_csv(output_file, index=False)
        print(f"\n\nSaved to: {output_file}")

        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total relevant speeches: {len(relevant)}")
        print(f"Speeches with other-group keywords: {len(suspicious)} ({len(suspicious)/len(relevant)*100:.1f}%)")
        print(f"\nBreakdown by stance:")
        print(suspicious_df['stance'].value_counts())
        print(f"\nWith 'women' mentioned: {suspicious_df['has_women'].sum()}")
        print(f"Without 'women' mentioned: {(~suspicious_df['has_women']).sum()}")

    else:
        print("\nNo suspicious speeches found.")

    # Also look at speeches marked irrelevant to see if they're catching these correctly
    irrelevant = df[df['stance'] == 'irrelevant'].copy()
    print(f"\n\n{'='*80}")
    print(f"IRRELEVANT SPEECHES (correctly filtered): {len(irrelevant)}")
    print(f"{'='*80}")

    # Sample a few irrelevant speeches
    print("\nSample irrelevant speeches:")
    for i, row in irrelevant.sample(min(3, len(irrelevant))).iterrows():
        print(f"\n  Date: {row.get('date', 'unknown')}")
        print(f"  Text: {row['text'][:200]}...")

if __name__ == '__main__':
    investigate_misclassifications()
