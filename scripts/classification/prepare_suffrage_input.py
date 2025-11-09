"""
Data preparation script for suffrage speech classification.
Extracts context windows and creates stratified samples for LLM analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

# Constants
CONTEXT_WINDOW = 5  # Number of speeches before and after target
PILOT_SAMPLE_SIZE = 100
OUTPUT_DIR = Path("outputs/llm_classification")
RANDOM_SEED = 42

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load suffrage speeches and full debate context."""
    print("Loading data...")

    # Load target suffrage speeches
    reliable = pd.read_parquet("outputs/suffrage_reliable/speeches_reliable.parquet")
    print(f"Loaded {len(reliable)} suffrage speeches")
    print(f"  HIGH: {(reliable.confidence_level == 'HIGH').sum()}")
    print(f"  MEDIUM: {(reliable.confidence_level == 'MEDIUM').sum()}")

    # Load all speeches from suffrage debates (for context)
    all_speeches = pd.read_parquet("outputs/suffrage_debates/all_speeches_in_suffrage_debates.parquet")
    print(f"Loaded {len(all_speeches)} total speeches from suffrage debates")

    return reliable, all_speeches


def extract_context_window(
    target_speech: pd.Series,
    all_speeches: pd.DataFrame,
    window_size: int = CONTEXT_WINDOW
) -> dict:
    """
    Extract context window around a target speech.

    Returns dict with:
    - target_text: The suffrage speech text
    - context_text: Surrounding speeches formatted for prompt
    - metadata: Useful fields for analysis
    """
    # Get all speeches from same debate, sorted by sequence
    debate_speeches = all_speeches[
        all_speeches.debate_id == target_speech.debate_id
    ].sort_values('sequence_number').copy()

    # Find target speech position
    target_idx = debate_speeches[
        debate_speeches.speech_id == target_speech.speech_id
    ].index

    if len(target_idx) == 0:
        # Fallback: target speech might not be in all_speeches
        # This can happen if speeches_reliable has been updated
        return {
            'target_text': target_speech.text,
            'context_text': "[No context available]",
            'context_before': 0,
            'context_after': 0,
        }

    target_position = debate_speeches.index.get_loc(target_idx[0])

    # Extract window
    start_idx = max(0, target_position - window_size)
    end_idx = min(len(debate_speeches), target_position + window_size + 1)

    window = debate_speeches.iloc[start_idx:end_idx]

    # Format context (excluding target speech itself)
    context_speeches = window[window.speech_id != target_speech.speech_id]

    context_lines = []
    for _, speech in context_speeches.iterrows():
        speaker = speech.speaker if pd.notna(speech.speaker) else "Unknown"
        text = speech.text[:500] + "..." if len(speech.text) > 500 else speech.text
        context_lines.append(f"[{speaker}]: {text}")

    context_text = "\n\n".join(context_lines)

    return {
        'target_text': target_speech.text,
        'context_text': context_text if context_text else "[No context available]",
        'context_before': target_position - start_idx,
        'context_after': end_idx - target_position - 1,
    }


def create_stratified_sample(
    df: pd.DataFrame,
    sample_size: int = PILOT_SAMPLE_SIZE,
    random_seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Create stratified sample across time periods and confidence levels.

    Strategy:
    - 20 speeches from each decade (1900s, 1910s, 1920s, 1930s)
    - Within each decade, try to get equal HIGH/MEDIUM split
    - 20 extra speeches from 1910s (largest decade with 1,396 speeches)
    """
    print(f"\nCreating stratified sample of {sample_size} speeches...")

    samples = []

    # Sample from each decade
    for decade in [1900, 1910, 1920, 1930]:
        decade_df = df[df.decade == decade]

        # How many from this decade?
        if decade == 1910:
            n_from_decade = 40  # Extra from largest decade
        else:
            n_from_decade = 20

        # Try to get balanced HIGH/MEDIUM
        n_high = min(
            n_from_decade // 2,
            (decade_df.confidence_level == 'HIGH').sum()
        )
        n_medium = n_from_decade - n_high

        high_sample = decade_df[
            decade_df.confidence_level == 'HIGH'
        ].sample(n=n_high, random_state=random_seed)

        medium_sample = decade_df[
            decade_df.confidence_level == 'MEDIUM'
        ].sample(n=n_medium, random_state=random_seed)

        samples.append(high_sample)
        samples.append(medium_sample)

        print(f"  {decade}s: {n_high} HIGH + {n_medium} MEDIUM = {n_from_decade} total")

    sample_df = pd.concat(samples, ignore_index=True)
    print(f"\nTotal sample size: {len(sample_df)}")

    return sample_df


def prepare_classification_input(
    reliable_speeches: pd.DataFrame,
    all_speeches: pd.DataFrame,
    sample_df: pd.DataFrame = None,
    window_size: int = CONTEXT_WINDOW
) -> pd.DataFrame:
    """
    Prepare input data for LLM classification.

    If sample_df provided, only process those speeches (for pilot).
    Otherwise process all speeches (for full run).
    """
    speeches_to_process = sample_df if sample_df is not None else reliable_speeches

    print(f"\nPreparing {len(speeches_to_process)} speeches for classification...")

    results = []

    for idx, speech in speeches_to_process.iterrows():
        context_data = extract_context_window(speech, all_speeches, window_size)

        results.append({
            # Identifiers
            'speech_id': speech.speech_id,
            'debate_id': speech.debate_id,

            # Text for LLM
            'target_text': context_data['target_text'],
            'context_text': context_data['context_text'],

            # Metadata for analysis
            'speaker': speech.speaker,
            'canonical_name': speech.canonical_name,
            'gender': speech.gender,
            'party': speech.party,
            'year': speech.year,
            'decade': speech.decade,
            'date': speech.date,
            'chamber': speech.chamber,
            'confidence_level': speech.confidence_level,
            'word_count': speech.word_count,

            # Context window info
            'context_before': context_data['context_before'],
            'context_after': context_data['context_after'],
        })

    result_df = pd.DataFrame(results)
    print(f"Prepared {len(result_df)} speeches")

    return result_df


def main():
    """Main execution."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    reliable, all_speeches = load_data()

    # Create pilot sample
    pilot_sample = create_stratified_sample(reliable, PILOT_SAMPLE_SIZE)

    # Prepare pilot input
    pilot_input = prepare_classification_input(reliable, all_speeches, pilot_sample)

    # Save pilot data
    pilot_path = OUTPUT_DIR / "pilot_input.parquet"
    pilot_input.to_parquet(pilot_path, index=False)
    print(f"\nSaved pilot input to {pilot_path}")
    print(f"Pilot sample: {len(pilot_input)} speeches")

    # Also save a CSV for easy inspection
    pilot_csv = OUTPUT_DIR / "pilot_input_sample.csv"
    pilot_input[['speech_id', 'speaker', 'year', 'confidence_level', 'word_count']].to_csv(
        pilot_csv, index=False
    )
    print(f"Saved pilot sample info to {pilot_csv}")

    # Prepare full input (for later)
    print("\n" + "="*60)
    print("Preparing full dataset for future use...")
    full_input = prepare_classification_input(reliable, all_speeches, sample_df=None)

    full_path = OUTPUT_DIR / "full_input.parquet"
    full_input.to_parquet(full_path, index=False)
    print(f"\nSaved full input to {full_path}")
    print(f"Full dataset: {len(full_input)} speeches")

    # Print summary statistics
    print("\n" + "="*60)
    print("PILOT SAMPLE SUMMARY")
    print("="*60)
    print("\nBy decade:")
    print(pilot_input.decade.value_counts().sort_index())
    print("\nBy confidence level:")
    print(pilot_input.confidence_level.value_counts())
    print("\nBy gender:")
    print(pilot_input.gender.value_counts())
    print("\nAverage word count:", pilot_input.word_count.mean())
    print("Average context before:", pilot_input.context_before.mean())
    print("Average context after:", pilot_input.context_after.mean())

    print("\n" + "="*60)
    print("FULL DATASET SUMMARY")
    print("="*60)
    print("\nBy decade:")
    print(full_input.decade.value_counts().sort_index())
    print("\nBy confidence level:")
    print(full_input.confidence_level.value_counts())
    print("\nBy gender:")
    print(full_input.gender.value_counts())

    print("\n" + "="*60)
    print("Data preparation complete!")
    print(f"Next step: Review {pilot_csv} and run Modal classification")
    print("="*60)


if __name__ == "__main__":
    main()
