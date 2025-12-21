#!/usr/bin/env python3
"""
Apply manual speaker corrections to the suffrage dataset.

Reads corrections from manual_speaker_corrections.csv and updates the
classification results with corrected speaker attributions.
"""

import pandas as pd
from pathlib import Path

def apply_corrections():
    """Apply manual speaker corrections to the dataset."""

    print('='*80)
    print('APPLYING MANUAL SPEAKER CORRECTIONS')
    print('='*80)

    # Load corrections
    corrections_file = 'outputs/corrections/manual_speaker_corrections.csv'
    print(f'\nLoading corrections from: {corrections_file}')
    corrections_df = pd.read_csv(corrections_file)

    print(f'Found {len(corrections_df)} corrections')

    # Load results dataset
    results_file = 'outputs/llm_classification/full_results_v5_context_3_expanded.parquet'
    print(f'\nLoading results from: {results_file}')
    df = pd.read_parquet(results_file)

    print(f'Loaded {len(df):,} classified speeches')

    # Track changes
    updates_applied = 0
    exclusions_applied = 0
    reviews_noted = 0

    # Create exclusion mask
    exclude_ids = set()

    print('\n' + '='*80)
    print('APPLYING CORRECTIONS')
    print('='*80 + '\n')

    for idx, correction in corrections_df.iterrows():
        speech_id = correction['speech_id']
        action = correction['action']

        # Find the speech in results
        mask = df['speech_id'] == speech_id

        if not mask.any():
            print(f"WARNING: Speech {speech_id} not found in results")
            continue

        old_speaker = correction['wrong_speaker']

        if action == 'UPDATE':
            new_name = correction['correct_person_name']
            new_gender = correction['correct_gender']

            print(f"Updating {speech_id}:")
            print(f"  Speaker: {old_speaker} -> {new_name}")
            print(f"  Gender: {correction['wrong_gender']} -> {new_gender}")

            # Update the fields
            df.loc[mask, 'canonical_name'] = new_name
            df.loc[mask, 'gender'] = new_gender

            # Note: We keep the original speaker field as-is since it's the raw text
            # but update canonical_name which is the normalized version

            updates_applied += 1
            print(f"  -> Updated\n")

        elif action == 'EXCLUDE':
            reason = correction['notes']
            print(f"Excluding {speech_id}:")
            print(f"  Speaker: {old_speaker}")
            print(f"  Reason: {reason}")

            exclude_ids.add(speech_id)
            exclusions_applied += 1
            print(f"  -> Marked for exclusion\n")

        elif action == 'REVIEW':
            reason = correction['notes']
            print(f"Flagged for review {speech_id}:")
            print(f"  Speaker: {old_speaker}")
            print(f"  Reason: {reason}")

            reviews_noted += 1
            print(f"  -> No changes applied (needs manual review)\n")

    # Apply exclusions
    if exclude_ids:
        print(f'\nExcluding {len(exclude_ids)} speeches...')
        df_corrected = df[~df['speech_id'].isin(exclude_ids)].copy()
        print(f'Remaining speeches: {len(df_corrected):,}')
    else:
        df_corrected = df.copy()

    # Save corrected dataset
    output_dir = Path('outputs/llm_classification')
    output_file = output_dir / 'full_results_v5_context_3_expanded_corrected.parquet'

    df_corrected.to_parquet(output_file, index=False)
    print(f'\nSaved corrected dataset to: {output_file}')

    # Summary
    print('\n' + '='*80)
    print('SUMMARY')
    print('='*80)
    print(f'Speaker updates applied: {updates_applied}')
    print(f'Speeches excluded: {exclusions_applied}')
    print(f'Flagged for review: {reviews_noted}')
    print(f'\nOriginal dataset: {len(df):,} speeches')
    print(f'Corrected dataset: {len(df_corrected):,} speeches')
    print(f'Net change: {len(df) - len(df_corrected):,} speeches removed')

    # Gender breakdown
    print('\n' + '='*80)
    print('GENDER BREAKDOWN (Corrected Dataset)')
    print('='*80)

    female_df = df_corrected[df_corrected['gender'] == 'F']
    male_df = df_corrected[df_corrected['gender'] == 'M']

    print(f'\nFemale MPs: {len(female_df):,} speeches')
    print(f'  For: {len(female_df[female_df["stance"] == "for"]):,}')
    print(f'  Against: {len(female_df[female_df["stance"] == "against"]):,}')
    print(f'  Both: {len(female_df[female_df["stance"] == "both"]):,}')
    print(f'  Irrelevant: {len(female_df[female_df["stance"] == "irrelevant"]):,}')

    print(f'\nMale MPs: {len(male_df):,} speeches')
    print(f'  For: {len(male_df[male_df["stance"] == "for"]):,}')
    print(f'  Against: {len(male_df[male_df["stance"] == "against"]):,}')
    print(f'  Both: {len(male_df[male_df["stance"] == "both"]):,}')
    print(f'  Irrelevant: {len(male_df[male_df["stance"] == "irrelevant"]):,}')

    return df_corrected


if __name__ == '__main__':
    corrected_df = apply_corrections()
