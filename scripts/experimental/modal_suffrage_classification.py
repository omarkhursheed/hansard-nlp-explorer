"""
Modal app for suffrage speech stance classification using OpenRouter.
Runs LLM classification at scale with parallel processing and fault tolerance.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import time

import modal
import pandas as pd

# Modal app setup
app = modal.App("suffrage-classification")

# Create persistent volume for results
volume = modal.Volume.from_name("suffrage-results", create_if_missing=True)

# Docker image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install("pandas", "pyarrow", "requests", "openai")
)

# Load prompts as module-level constants
TURNWISE_PROMPT_TEMPLATE = """SYSTEM
You are an argument-mining assistant labeling a single parliamentary speech turn ("TARGET"). You are to think and reason carefully. This is an MP's turn in debate on bills in parliament. The topic is related to women's suffrage (right to vote or hold electoral offices).

Read TARGET as a whole. First, DISTILL the speaker's key JUSTIFICATIONS on the topic into a small set of high-level "reason buckets."

Reason buckets (seed taxonomy; choose one per reason OR create a new one):
- equality              (equal rights, justice, fairness)
- competence_capacity   (abilities/intellect/education of women/men)
- emotion_morality      (emotionality, virtue, moral fitness)
- social_order_stability(order, stability, foreign relations)
- tradition_precedent   (custom, precedent, history)
- instrumental_effects  (pragmatic costs/benefits, governance efficiency)
- religion_family       (religious or family-role arguments)
- social_experiment     (trial/experiment/pilot to learn effects)
- other                 (if none fit; then you MUST name it via bucket_open)

Evidence policy for reason buckets:
- For each reason, add 1-2 short **verbatim quotes** that are exact, contiguous substrings from TARGET ONLY (not CONTEXT), each 40-120 characters. No paraphrase, no stitched fragments, no ellipses.
- Prefer quotes that directly carry the justificatory content; avoid quoting whole sentences when a tighter span suffices.
- Do NOT enumerate every sentence. Return at most 3 reasons total (ranked by salience for the stance).

Only after you finalize the reasons, INFER the overall stance on women's suffrage (right to vote or hold electoral offices) from those reasons.

Stance labels (closed set inferred from reasons):
- "for"        : reasons support women's right to vote.
- "against"    : reasons oppose women's right to vote.
- "both"       : reasons include support AND opposition (e.g., for the vote but against women as MPs). Include at least one reason on each side.
- "neutral"    : TARGET expresses genuine indifference/acceptance of either outcome.
- "irrelevant" : TARGET is not about women's suffrage at all.

Stance inference rules (apply after reasons are set):
- For the inferred stance, give a free-text rationale (1-2 sentences) citing the key reasons you found earlier, they must align with the buckets and verbatim quotes
- If you have >=1 reason with stance_label="for" and none "against", then stance="for".
- If you have >=1 reason with stance_label="against" and none "for", then stance="against".
- If you have >=1 reason for each side, then stance="both".
- If no suffrage-relevant reasons and TARGET is not about suffrage, then stance="irrelevant".
- If TARGET is genuinely indifferent (no directional reasons, is okay with either), then stance="neutral".

Confidence scoring (set a value in [0,1]):
- HIGH (0.7-1.0): TARGET contains explicit, direct statements about women's suffrage with clear supporting arguments. Multiple strong quotes. Speaker's position is unambiguous.
- MEDIUM (0.4-0.7): TARGET discusses women's suffrage with reasonable clarity, but arguments may be indirect, brief, or require some inference. Adequate quotes available.
- LOW (0.0-0.4): Weak evidence (vague references, very brief mentions, or highly ambiguous statements). For "both" stance, or when TARGET barely mentions suffrage. For "irrelevant", always use 0.0.

Return ONLY this JSON object (no extra text):
{{
  "stance": "for | against | both | neutral | irrelevant",
  "reasons": [
    {{
      "bucket_key": "equality | competence_capacity | emotion_morality | social_order_stability | tradition_precedent | instrumental_effects | religion_family | social_experiment | other",
      "bucket_open": "<free label if bucket_key = other, else \"\">",
      "stance_label": "for | against",
      "rationale": "<one-sentence distilled justification>",
      "quotes": ["<substring from TARGET 40-120 chars>", "<optional second quote 40-120 chars>"]
    }}
  ],
  "top_quote": "<the single strongest 40-120 char quote from TARGET>",
  "confidence": 0.0
}}

Hard constraints:
- Max 3 reasons; max 2 quotes per reason.
- Quotes must keep original casing/punctuation and come only from TARGET (never CONTEXT).
- Drop any quote outside 40-120 chars if a compliant span exists.
- If mixed ("both"), provide at least one reason labeled for each side.
- Never infer or reason from your external knowledge about historical events, MPs, or political context not present in TARGET or CONTEXT. Strictly stick to what is in the provided text.

USER
TARGET:
{target_text}

CONTEXT (neighbors, read-only):
{context_text}
"""


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openrouter-api-key")],
    volumes={"/results": volume},
    timeout=600,  # 10 minutes per speech
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=1.0,
    ),
)
def classify_speech(
    speech_data: Dict,
    model: str = "openai/gpt-4o-mini",
    prompt_version: str = "v4",
) -> Dict:
    """
    Classify a single speech using OpenRouter API.

    Args:
        speech_data: Dict with 'speech_id', 'target_text', 'context_text', and metadata
        model: OpenRouter model identifier
        prompt_version: Version tag for reproducibility

    Returns:
        Dict with classification results + metadata + any errors
    """
    import requests

    api_key = os.environ["OPENROUTER_API_KEY"]

    # Format prompt
    prompt = TURNWISE_PROMPT_TEMPLATE.format(
        target_text=speech_data['target_text'],
        context_text=speech_data['context_text']
    )

    # Call OpenRouter API
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,  # Low temperature for consistency
                "max_tokens": 1500,
            },
            timeout=120,
        )
        response.raise_for_status()

        result = response.json()

        # Extract LLM response
        llm_output = result['choices'][0]['message']['content'].strip()

        # Try to parse JSON
        try:
            # Remove markdown code blocks if present
            if llm_output.startswith("```"):
                llm_output = llm_output.split("```")[1]
                if llm_output.startswith("json"):
                    llm_output = llm_output[4:]
                llm_output = llm_output.strip()

            classification = json.loads(llm_output)
        except json.JSONDecodeError as e:
            # Failed to parse JSON - save error
            classification = {
                "error": "json_parse_failed",
                "error_detail": str(e),
                "raw_output": llm_output[:500],
            }

        # Add metadata
        classification.update({
            "speech_id": speech_data['speech_id'],
            "debate_id": speech_data['debate_id'],
            "speaker": speech_data.get('speaker'),
            "canonical_name": speech_data.get('canonical_name'),
            "gender": speech_data.get('gender'),
            "party": speech_data.get('party'),
            "year": speech_data.get('year'),
            "decade": speech_data.get('decade'),
            "date": speech_data.get('date'),
            "chamber": speech_data.get('chamber'),
            "confidence_level": speech_data.get('confidence_level'),
            "word_count": speech_data.get('word_count'),
            "model": model,
            "prompt_version": prompt_version,
            "tokens_used": result.get('usage', {}).get('total_tokens', 0),
            "api_success": True,
        })

        return classification

    except requests.exceptions.RequestException as e:
        # API call failed
        return {
            "speech_id": speech_data['speech_id'],
            "debate_id": speech_data['debate_id'],
            "error": "api_call_failed",
            "error_detail": str(e),
            "api_success": False,
        }


@app.function(
    image=image,
    volumes={"/results": volume},
    timeout=36000,  # 10 hours max
)
def run_classification_batch(
    input_file: str,
    output_file: str,
    model: str = "openai/gpt-4o-mini",
    batch_size: int = 50,
    checkpoint_interval: int = 100,
) -> Dict:
    """
    Run classification on a batch of speeches with checkpointing.

    Args:
        input_file: Path to input parquet file
        output_file: Path to save results
        model: OpenRouter model
        batch_size: Number of parallel calls
        checkpoint_interval: Save after every N speeches

    Returns:
        Summary statistics
    """
    import pandas as pd

    print(f"Loading input from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} speeches")

    # Convert to list of dicts
    speeches = df.to_dict('records')

    results = []
    start_time = time.time()

    # Process in batches
    for i in range(0, len(speeches), batch_size):
        batch = speeches[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(speeches) + batch_size - 1) // batch_size

        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} speeches)...")

        # Map over batch in parallel
        batch_results = list(classify_speech.map(
            [s for s in batch],
            kwargs={"model": model}
        ))

        results.extend(batch_results)

        # Checkpoint
        if len(results) % checkpoint_interval == 0:
            checkpoint_df = pd.DataFrame(results)
            checkpoint_path = f"/results/{output_file}.checkpoint"
            checkpoint_df.to_parquet(checkpoint_path, index=False)
            print(f"Checkpoint saved: {len(results)} speeches processed")

            # Print progress stats
            elapsed = time.time() - start_time
            rate = len(results) / elapsed
            remaining = len(speeches) - len(results)
            eta_seconds = remaining / rate if rate > 0 else 0

            print(f"Progress: {len(results)}/{len(speeches)} ({100*len(results)/len(speeches):.1f}%)")
            print(f"Rate: {rate:.1f} speeches/sec")
            print(f"ETA: {eta_seconds/60:.1f} minutes")

    # Save final results
    print(f"\nSaving final results to {output_file}...")
    results_df = pd.DataFrame(results)
    results_df.to_parquet(f"/results/{output_file}", index=False)

    # Commit volume changes
    volume.commit()

    # Calculate summary stats
    success_count = results_df['api_success'].sum() if 'api_success' in results_df else 0
    total_tokens = results_df['tokens_used'].sum() if 'tokens_used' in results_df else 0

    summary = {
        "total_speeches": len(results),
        "successful": int(success_count),
        "failed": len(results) - int(success_count),
        "total_tokens": int(total_tokens),
        "elapsed_seconds": time.time() - start_time,
    }

    print("\n" + "="*60)
    print("CLASSIFICATION COMPLETE")
    print("="*60)
    print(f"Total speeches: {summary['total_speeches']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total tokens: {summary['total_tokens']:,}")
    print(f"Elapsed time: {summary['elapsed_seconds']/60:.1f} minutes")
    print("="*60)

    return summary


@app.function(
    image=image,
    volumes={"/results": volume},
    timeout=600,
)
def upload_input_data(parquet_bytes: bytes, remote_filename: str):
    """Upload parquet data to Modal volume."""
    import pandas as pd
    import io

    # Read from bytes
    df = pd.read_parquet(io.BytesIO(parquet_bytes))
    print(f"Received {len(df)} speeches")

    # Save to volume
    output_path = f"/results/{remote_filename}"
    df.to_parquet(output_path, index=False)
    volume.commit()

    print(f"Uploaded to Modal volume: {output_path}")
    return len(df)


@app.local_entrypoint()
def main(
    pilot: bool = False,
    model: str = "openai/gpt-4o-mini",
    batch_size: int = 50,
):
    """
    Main entry point for running classification.

    Usage:
        # Pilot study (100 speeches)
        modal run modal_suffrage_classification.py --pilot

        # Full dataset (2,808 speeches)
        modal run modal_suffrage_classification.py

    Args:
        pilot: Run on pilot sample (100 speeches) instead of full dataset
        model: OpenRouter model to use
        batch_size: Number of parallel API calls
    """
    if pilot:
        local_input = "outputs/llm_classification/pilot_input.parquet"
        remote_input = "pilot_input.parquet"
        output_file = "pilot_results.parquet"
        print("="*60)
        print("PILOT STUDY MODE")
        print("="*60)
    else:
        local_input = "outputs/llm_classification/full_input_no_context.parquet"
        remote_input = "full_input_no_context.parquet"
        output_file = "full_results_no_context.parquet"
        print("="*60)
        print("FULL DATASET MODE (NO CONTEXT)")
        print("="*60)

    print(f"Local input: {local_input}")
    print(f"Remote input: {remote_input}")
    print(f"Output: {output_file}")
    print(f"Model: {model}")
    print(f"Batch size: {batch_size}")
    print("="*60)

    # Upload input file to Modal volume
    print("\nUploading input data to Modal volume...")

    # Read file locally and convert to bytes
    import pandas as pd
    local_df = pd.read_parquet(local_input)
    print(f"Loaded {len(local_df)} speeches from {local_input}")

    # Convert to bytes
    import io
    buffer = io.BytesIO()
    local_df.to_parquet(buffer, index=False)
    parquet_bytes = buffer.getvalue()
    print(f"Converted to {len(parquet_bytes):,} bytes")

    # Upload to Modal
    n_speeches = upload_input_data.remote(parquet_bytes, remote_input)
    print(f"Successfully uploaded {n_speeches} speeches")

    # Run classification
    print("\nStarting classification...")
    summary = run_classification_batch.remote(
        input_file=f"/results/{remote_input}",
        output_file=output_file,
        model=model,
        batch_size=batch_size,
    )

    print("\n" + "="*60)
    print("JOB COMPLETE")
    print("="*60)
    print(f"Results saved to Modal volume: /results/{output_file}")
    print(f"\nTo download results:")
    print(f"  modal volume get suffrage-results {output_file} ./outputs/llm_classification/")
    print("="*60)

    return summary
