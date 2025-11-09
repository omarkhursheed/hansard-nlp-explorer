"""
Prepare pilot data with different context window sizes.
Tests: 0 (no context), 2, 5, 10 speeches before/after target.
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("outputs/llm_classification")
# Context sizes: 0=no context, 3/5/10/20=window size, -1=full debate
CONTEXT_SIZES = [0, 3, 5, 10, 20, -1]
RANDOM_SEED = 42

def extract_context_window(
    target_speech: pd.Series,
    all_speeches: pd.DataFrame,
    window_size: int = 5
) -> dict:
    """
    Extract context window around a target speech.

    Args:
        target_speech: The target speech to classify
        all_speeches: All speeches in the debate
        window_size: Number of speeches before/after to include
                     0 = no context
                     -1 = full debate context
                     N = N speeches before and after
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
        return {
            'target_text': target_speech.text,
            'context_text': "[No context available]",
            'context_before': 0,
            'context_after': 0,
        }

    target_position = debate_speeches.index.get_loc(target_idx[0])

    if window_size == 0:
        # No context
        return {
            'target_text': target_speech.text,
            'context_text': "[No context - analyzing speech only]",
            'context_before': 0,
            'context_after': 0,
        }
    elif window_size == -1:
        # Full debate context
        window = debate_speeches
        start_idx = 0
        end_idx = len(debate_speeches)
    else:
        # Specific window size
        start_idx = max(0, target_position - window_size)
        end_idx = min(len(debate_speeches), target_position + window_size + 1)
        window = debate_speeches.iloc[start_idx:end_idx]

    # Format context (excluding target speech itself)
    context_speeches = window[window.speech_id != target_speech.speech_id]

    context_lines = []
    for _, speech in context_speeches.iterrows():
        speaker = speech.speaker if pd.notna(speech.speaker) else "Unknown"
        # For full debate, truncate individual speeches more to avoid huge context
        max_len = 300 if window_size == -1 else 500
        text = speech.text[:max_len] + "..." if len(speech.text) > max_len else speech.text
        context_lines.append(f"[{speaker}]: {text}")

    context_text = "\n\n".join(context_lines)

    return {
        'target_text': target_speech.text,
        'context_text': context_text if context_text else "[No surrounding speeches in debate]",
        'context_before': target_position - start_idx,
        'context_after': end_idx - target_position - 1,
    }


def prepare_pilot_with_context_size(
    reliable_speeches: pd.DataFrame,
    all_speeches: pd.DataFrame,
    pilot_ids: list,
    window_size: int
) -> pd.DataFrame:
    """Prepare pilot input with specific context window size."""

    pilot_speeches = reliable_speeches[reliable_speeches.speech_id.isin(pilot_ids)]

    results = []
    for idx, speech in pilot_speeches.iterrows():
        context_data = extract_context_window(speech, all_speeches, window_size)

        results.append({
            'speech_id': speech.speech_id,
            'debate_id': speech.debate_id,
            'target_text': context_data['target_text'],
            'context_text': context_data['context_text'],
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
            'context_window_size': window_size,
            'context_before': context_data['context_before'],
            'context_after': context_data['context_after'],
        })

    return pd.DataFrame(results)


def main():
    print("="*70)
    print("CONTEXT WINDOW SIZE EXPERIMENT")
    print("="*70)

    # Load data
    print("\nLoading data...")
    reliable = pd.read_parquet("outputs/suffrage_reliable/speeches_reliable.parquet")
    all_speeches = pd.read_parquet("outputs/suffrage_debates/all_speeches_in_suffrage_debates.parquet")

    print(f"Loaded {len(reliable)} suffrage speeches")
    print(f"Loaded {len(all_speeches)} total speeches from suffrage debates")

    # Use same pilot sample as before (first 100 speeches from pilot_input)
    existing_pilot = pd.read_parquet("outputs/llm_classification/pilot_input.parquet")
    pilot_ids = existing_pilot['speech_id'].tolist()

    print(f"\nUsing existing pilot sample: {len(pilot_ids)} speeches")
    print(f"(Same speeches for all context sizes to ensure fair comparison)")

    # Generate inputs for each context size
    for window_size in CONTEXT_SIZES:
        print(f"\n{'-'*70}")
        if window_size == -1:
            print(f"Preparing FULL DEBATE context")
        elif window_size == 0:
            print(f"Preparing NO CONTEXT (speech only)")
        else:
            print(f"Preparing context window size: {window_size}")
        print(f"{'-'*70}")

        pilot_df = prepare_pilot_with_context_size(
            reliable, all_speeches, pilot_ids, window_size
        )

        # Save with appropriate filename
        size_label = "full" if window_size == -1 else str(window_size)
        output_file = OUTPUT_DIR / f"pilot_input_context_{size_label}.parquet"
        pilot_df.to_parquet(output_file, index=False)

        print(f"Saved: {output_file}")
        print(f"Speeches: {len(pilot_df)}")

        # Stats
        if window_size > 0:
            print(f"Avg context before: {pilot_df.context_before.mean():.1f}")
            print(f"Avg context after: {pilot_df.context_after.mean():.1f}")

            # Calculate average context text length for token estimation
            avg_context_len = pilot_df.context_text.str.len().mean()
            print(f"Avg context length: {avg_context_len:,.0f} chars (~{avg_context_len/4:.0f} tokens)")
        elif window_size == 0:
            print(f"No context (speech only)")
        else:  # -1 = full debate
            avg_context_len = pilot_df.context_text.str.len().mean()
            print(f"Avg full debate context: {avg_context_len:,.0f} chars (~{avg_context_len/4:.0f} tokens)")

    print("\n" + "="*70)
    print("CONTEXT EXPERIMENT DATA READY")
    print("="*70)
    print(f"\nGenerated {len(CONTEXT_SIZES)} input files:")
    for size in CONTEXT_SIZES:
        size_label = "full" if size == -1 else str(size)
        print(f"  - pilot_input_context_{size_label}.parquet")

    print("\n" + "="*70)
    print("NEXT STEP: Run classification with all context sizes")
    print("="*70)
    print("Run: python3 run_context_experiment.py")
    print("="*70)


if __name__ == "__main__":
    main()
