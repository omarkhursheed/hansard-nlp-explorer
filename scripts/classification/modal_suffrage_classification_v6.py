"""
Modal app for suffrage speech stance classification using OpenRouter.

Supports multiple prompt versions for reproducibility and comparison.
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

# ============================================================================
# PROMPT VERSIONS (versioned for reproducibility)
# ============================================================================

# V5: Original prompt with basic suffrage definition
PROMPT_V5 = """SYSTEM
You are an argument-mining assistant labeling a single parliamentary speech turn ("TARGET") within its debate CONTEXT. You are to think and reason carefully. This is an MP's turn in debate on bills in parliament. The topic is related to women's suffrage (right to vote or hold electoral offices).

Read both TARGET and CONTEXT. Use CONTEXT to:
- Understand what "the Bill" or "this question" refers to
- Identify arguments the speaker is responding to
- Catch arguments split across multiple speaker turns
- Understand irony, sarcasm, or implicit references

First, DISTILL the TARGET speaker's key JUSTIFICATIONS on the topic into a small set of high-level "reason buckets."

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
- For each reason, add 1-2 short **verbatim quotes** that are exact, contiguous substrings, each 40-120 characters
- **PREFER quotes from TARGET**, but you MAY use quotes from CONTEXT if:
  - The TARGET speaker is directly responding to/refuting that argument
  - The TARGET speaker explicitly refers to it ("as my hon. Friend said...")
  - It clarifies what TARGET is arguing about
- **Label each quote** with source: "TARGET" or "CONTEXT"
- Prefer quotes that directly carry the justificatory content
- Do NOT enumerate every sentence. Return at most 3 reasons total (ranked by salience for the stance)

Only after you finalize the reasons, INFER the TARGET speaker's overall stance on women's suffrage (right to vote or hold electoral offices) from those reasons.

Stance labels (closed set inferred from reasons):
- "for"        : reasons support women's right to vote
- "against"    : reasons oppose women's right to vote
- "both"       : reasons include support AND opposition (e.g., for the vote but against women as MPs). Include at least one reason on each side
- "neutral"    : TARGET expresses genuine indifference/acceptance of either outcome
- "irrelevant" : TARGET is not about women's suffrage at all

Stance inference rules (apply after reasons are set):
- For the inferred stance, give a free-text rationale (1-2 sentences) citing the key reasons you found earlier, they must align with the buckets and verbatim quotes
- If you have >=1 reason with stance_label="for" and none "against", then stance="for"
- If you have >=1 reason with stance_label="against" and none "for", then stance="against"
- If you have >=1 reason for each side, then stance="both"
- If no suffrage-relevant reasons and TARGET is not about suffrage, then stance="irrelevant"
- If TARGET is genuinely indifferent (no directional reasons, is okay with either), then stance="neutral"

Confidence scoring (set a value in [0,1]):
- HIGH (0.7-1.0): Clear, unambiguous classification with strong evidence. For suffrage stances: explicit arguments with multiple quotes. For "irrelevant": clearly about a different topic with no ambiguity
- MEDIUM (0.4-0.7): Moderate clarity. For suffrage stances: reasonable evidence but may be indirect or brief. For "irrelevant": probably not about suffrage but some ambiguity remains
- LOW (0.0-0.4): Weak or ambiguous evidence. For suffrage stances: vague references or highly uncertain. For "irrelevant": unclear whether it relates to suffrage or not

Return ONLY this JSON object (no extra text):
{{
  "stance": "for | against | both | neutral | irrelevant",
  "reasons": [
    {{
      "bucket_key": "equality | competence_capacity | emotion_morality | social_order_stability | tradition_precedent | instrumental_effects | religion_family | social_experiment | other",
      "bucket_open": "<free label if bucket_key = other, else \"\">",
      "stance_label": "for | against",
      "rationale": "<one-sentence distilled justification>",
      "quotes": [
        {{
          "text": "<substring 40-120 chars>",
          "source": "TARGET | CONTEXT"
        }},
        {{
          "text": "<optional second quote 40-120 chars>",
          "source": "TARGET | CONTEXT"
        }}
      ]
    }}
  ],
  "top_quote": {{
    "text": "<the single strongest 40-120 char quote>",
    "source": "TARGET | CONTEXT"
  }},
  "confidence": 0.0,
  "context_helpful": true | false
}}

Hard constraints:
- Max 3 reasons; max 2 quotes per reason
- Quotes must keep original casing/punctuation and be exact substrings
- Drop any quote outside 40-120 chars if a compliant span exists
- If mixed ("both"), provide at least one reason labeled for each side
- PREFER TARGET quotes when possible; only use CONTEXT quotes when they add significant value
- Always label quote source (TARGET or CONTEXT)
- Set context_helpful=true ONLY if CONTEXT actually helped you understand TARGET better
- Never infer or reason from your external knowledge about historical events, MPs, or political context not present in TARGET or CONTEXT

USER
TARGET:
{target_text}

CONTEXT (neighbors - use to understand TARGET):
{context_text}
"""

# V6: Improved prompt with explicit suffrage scope definition to fix conflation errors
PROMPT_V6 = """SYSTEM
You are an argument-mining assistant labeling a single parliamentary speech turn ("TARGET") within its debate CONTEXT. You are to think and reason carefully. This is an MP's turn in debate on bills in parliament.

CRITICAL - Suffrage Definition:
This task is ONLY about SUFFRAGE = the right to vote in elections or hold electoral office.

SUFFRAGE topics (relevant):
- Who can vote in general elections, local elections
- Women's right to vote (enfranchisement)
- Expanding or restricting the franchise
- Voting age (e.g., lowering from 21 to 18)
- Prisoner voting rights, universal suffrage
- Women as MPs or in House of Lords
- Electoral reform related to who can vote

NOT SUFFRAGE (mark as "irrelevant"):
- Women's other rights: equal pay, property ownership, education, employment
- Parliamentary procedure: "voting on this bill", "division on the amendment", "put to the vote"
- General women's issues unrelated to electoral franchise
- Party politics unless specifically about voting rights

Common confusions to avoid:
- "Women deserve equal treatment" (no mention of voting) → irrelevant
- "The House will now vote on this amendment" (parliamentary procedure) → irrelevant
- "Women's property rights should be reformed" → irrelevant
- "Women should have the vote" → for
- "Lowering the voting age to 18" → relevant (expanding suffrage)
- "Prisoners should not have the right to vote" → relevant (restricting suffrage)

Read both TARGET and CONTEXT. Use CONTEXT to:
- Understand what "the Bill" or "this question" refers to
- Identify arguments the speaker is responding to
- Catch arguments split across multiple speaker turns
- Understand irony, sarcasm, or implicit references

First, DISTILL the TARGET speaker's key JUSTIFICATIONS on the topic into a small set of high-level "reason buckets."

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
- For each reason, add 1-2 short **verbatim quotes** that are exact, contiguous substrings, each 40-120 characters
- **PREFER quotes from TARGET**, but you MAY use quotes from CONTEXT if:
  - The TARGET speaker is directly responding to/refuting that argument
  - The TARGET speaker explicitly refers to it ("as my hon. Friend said...")
  - It clarifies what TARGET is arguing about
- **Label each quote** with source: "TARGET" or "CONTEXT"
- Prefer quotes that directly carry the justificatory content
- Do NOT enumerate every sentence. Return at most 3 reasons total (ranked by salience for the stance)

Only after you finalize the reasons, INFER the TARGET speaker's overall stance on women's suffrage (right to vote or hold electoral offices) from those reasons.

Stance labels (closed set inferred from reasons):
- "for"        : reasons support women's right to vote
- "against"    : reasons oppose women's right to vote
- "both"       : reasons include support AND opposition (e.g., for the vote but against women as MPs). Include at least one reason on each side
- "neutral"    : TARGET expresses genuine indifference/acceptance of either outcome
- "irrelevant" : TARGET is not about electoral suffrage/franchise at all. This includes:
                 * Women's rights speeches that never mention voting/elections
                 * Parliamentary procedure discussions ("vote on the bill", "division")
                 * General women's issues (pay, property, education) without voting mention
                 * Speeches about other political topics

Stance inference rules (apply after reasons are set):
- For the inferred stance, give a free-text rationale (1-2 sentences) citing the key reasons you found earlier, they must align with the buckets and verbatim quotes
- If you have >=1 reason with stance_label="for" and none "against", then stance="for"
- If you have >=1 reason with stance_label="against" and none "for", then stance="against"
- If you have >=1 reason for each side, then stance="both"
- If no suffrage-relevant reasons and TARGET is not about suffrage, then stance="irrelevant"
- If TARGET is genuinely indifferent (no directional reasons, is okay with either), then stance="neutral"

Confidence scoring (set a value in [0,1]):
- HIGH (0.7-1.0): Clear, unambiguous classification with strong evidence. For suffrage stances: explicit arguments with multiple quotes. For "irrelevant": clearly about a different topic with no ambiguity
- MEDIUM (0.4-0.7): Moderate clarity. For suffrage stances: reasonable evidence but may be indirect or brief. For "irrelevant": probably not about suffrage but some ambiguity remains
- LOW (0.0-0.4): Weak or ambiguous evidence. For suffrage stances: vague references or highly uncertain. For "irrelevant": unclear whether it relates to suffrage or not

Return ONLY this JSON object (no extra text):
{{
  "stance": "for | against | both | neutral | irrelevant",
  "reasons": [
    {{
      "bucket_key": "equality | competence_capacity | emotion_morality | social_order_stability | tradition_precedent | instrumental_effects | religion_family | social_experiment | other",
      "bucket_open": "<free label if bucket_key = other, else \"\">",
      "stance_label": "for | against",
      "rationale": "<one-sentence distilled justification>",
      "quotes": [
        {{
          "text": "<substring 40-120 chars>",
          "source": "TARGET | CONTEXT"
        }},
        {{
          "text": "<optional second quote 40-120 chars>",
          "source": "TARGET | CONTEXT"
        }}
      ]
    }}
  ],
  "top_quote": {{
    "text": "<the single strongest 40-120 char quote>",
    "source": "TARGET | CONTEXT"
  }},
  "confidence": 0.0,
  "context_helpful": true | false
}}

Hard constraints:
- Max 3 reasons; max 2 quotes per reason
- Quotes must keep original casing/punctuation and be exact substrings
- Drop any quote outside 40-120 chars if a compliant span exists
- If mixed ("both"), provide at least one reason labeled for each side
- PREFER TARGET quotes when possible; only use CONTEXT quotes when they add significant value
- Always label quote source (TARGET or CONTEXT)
- Set context_helpful=true ONLY if CONTEXT actually helped you understand TARGET better
- Never infer or reason from your external knowledge about historical events, MPs, or political context not present in TARGET or CONTEXT

USER
TARGET:
{target_text}

CONTEXT (neighbors - use to understand TARGET):
{context_text}
"""

# Prompt selection mapping
PROMPTS = {
    "v5": PROMPT_V5,
    "v6": PROMPT_V6,
}


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
    model: str = "anthropic/claude-sonnet-4.5",
    prompt_version: str = "v6",
) -> Dict:
    """
    Classify a single speech using OpenRouter API.

    Args:
        speech_data: Dict with 'speech_id', 'target_text', 'context_text', and metadata
        model: OpenRouter model identifier
        prompt_version: Prompt version to use ("v5" or "v6")

    Returns:
        Dict with classification results + metadata + any errors
    """
    import requests

    api_key = os.environ["OPENROUTER_API_KEY"]

    # Select prompt
    prompt_template = PROMPTS.get(prompt_version, PROMPT_V6)

    # Format prompt
    prompt = prompt_template.format(
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

        # Try to extract and parse LLM response
        try:
            # Extract LLM response
            llm_output = result['choices'][0]['message']['content'].strip()

            # Remove markdown code blocks if present
            if llm_output.startswith("```"):
                llm_output = llm_output.split("```")[1]
                if llm_output.startswith("json"):
                    llm_output = llm_output[4:]
                llm_output = llm_output.strip()

            classification = json.loads(llm_output)

            # Normalize field types to ensure consistency for parquet serialization
            # top_quote MUST be a dict with text and source keys
            top_quote = classification.get("top_quote")
            if not isinstance(top_quote, dict):
                classification["top_quote"] = {"text": "", "source": ""}
            elif "text" not in top_quote or "source" not in top_quote:
                # Has dict but missing required keys
                classification["top_quote"] = {"text": str(top_quote.get("text", "")), "source": str(top_quote.get("source", ""))}

            # reasons MUST be a list
            reasons = classification.get("reasons")
            if not isinstance(reasons, list):
                classification["reasons"] = []

        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            # Failed to parse response or extract fields - save error with all expected fields
            error_type = "response_format_error" if isinstance(e, (KeyError, IndexError, TypeError)) else "json_parse_failed"
            classification = {
                "stance": None,
                "reasons": [],
                "top_quote": {"text": "", "source": ""},
                "confidence": None,
                "context_helpful": None,
                "error": error_type,
                "error_detail": str(e),
                "raw_output": str(result)[:500] if 'result' in locals() else "",
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
        # API call failed - include all expected fields
        return {
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
            "stance": None,
            "reasons": [],
            "top_quote": {"text": "", "source": ""},
            "confidence": None,
            "context_helpful": None,
            "model": model,
            "prompt_version": prompt_version,
            "tokens_used": 0,
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
    prompt_version: str = "v6",
    batch_size: int = 50,
    checkpoint_interval: int = 100,
) -> Dict:
    """
    Run classification on a batch of speeches with checkpointing.

    Args:
        input_file: Path to input parquet file
        output_file: Path to save results
        model: OpenRouter model
        prompt_version: Prompt version to use ("v5" or "v6")
        batch_size: Number of parallel calls
        checkpoint_interval: Save after every N speeches

    Returns:
        Summary statistics
    """
    import pandas as pd

    print(f"Loading input from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} speeches")
    print(f"Using prompt version: {prompt_version}")

    # Convert to list of dicts
    speeches = df.to_dict('records')

    # Check for existing checkpoint and resume if found
    checkpoint_path = f"/results/{output_file}.checkpoint"
    results = []
    processed_count = 0

    try:
        checkpoint_df = pd.read_parquet(checkpoint_path)
        results = checkpoint_df.to_dict('records')
        processed_count = len(results)
        processed_ids = set(checkpoint_df['speech_id'].values)

        # Normalize any old checkpoint data that might have None values
        for result in results:
            if result.get("top_quote") is None:
                result["top_quote"] = {"text": "", "source": ""}
            if result.get("reasons") is None:
                result["reasons"] = []

        print(f"\nFound checkpoint with {len(results)} speeches")
        print(f"Resuming from {len(results)}/{len(speeches)} ({100*len(results)/len(speeches):.1f}%)")
    except:
        print("\nNo checkpoint found, starting from beginning")
        processed_ids = set()

    # Filter to only unprocessed speeches
    speeches_to_process = [s for s in speeches if s['speech_id'] not in processed_ids]
    print(f"Processing {len(speeches_to_process)} remaining speeches")

    start_time = time.time()

    # Process in batches
    for i in range(0, len(speeches_to_process), batch_size):
        batch = speeches_to_process[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(speeches_to_process) + batch_size - 1) // batch_size

        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} speeches)...")

        # Map over batch in parallel
        batch_results = list(classify_speech.map(
            [s for s in batch],
            kwargs={"model": model, "prompt_version": prompt_version}
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
            newly_processed = len(results) - processed_count
            rate = newly_processed / elapsed if elapsed > 0 else 0
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
    print(f"Prompt version: {prompt_version}")
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
    model: str = "anthropic/claude-sonnet-4.5",
    prompt_version: str = "v6",
    batch_size: int = 100,
    experiment_name: str = "",
):
    """
    Main entry point for running classification with proper experiment tracking.

    Usage:
        # Test with Claude Sonnet 4.5 (pilot)
        modal run modal_suffrage_classification_v6.py --pilot --model anthropic/claude-sonnet-4.5 --experiment-name claude_sonnet_45_pilot

        # Test with GPT-4o (pilot)
        modal run modal_suffrage_classification_v6.py --pilot --model openai/gpt-4o --experiment-name gpt4o_pilot

        # Full run with best model
        modal run modal_suffrage_classification_v6.py --model anthropic/claude-sonnet-4.5 --experiment-name claude_sonnet_45_full

    Args:
        pilot: Run on pilot sample (300 speeches) instead of full dataset (6,531)
        model: OpenRouter model identifier (e.g., "anthropic/claude-sonnet-4.5", "openai/gpt-4o-mini")
        prompt_version: Prompt version to use ("v5" or "v6")
        batch_size: Number of parallel API calls
        experiment_name: Unique name for this experiment (auto-generated if not provided)
    """
    import datetime

    # Auto-generate experiment name if not provided
    if not experiment_name:
        model_short = model.split('/')[-1].replace('-', '_').replace('.', '_')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_label = "pilot" if pilot else "full"
        experiment_name = f"{model_short}_{prompt_version}_{dataset_label}_{timestamp}"

    # Determine input/output paths
    if pilot:
        local_input = "outputs/llm_classification/pilot_input.parquet"
        remote_input = f"{experiment_name}_input.parquet"
        output_file = f"{experiment_name}_results.parquet"
        dataset_label = "PILOT (300 speeches)"
    else:
        local_input = "outputs/llm_classification/full_input_context_3_expanded.parquet"
        remote_input = f"{experiment_name}_input.parquet"
        output_file = f"{experiment_name}_results.parquet"
        dataset_label = "FULL DATASET (6,531 speeches)"

    print("="*70)
    print(f"EXPERIMENT: {experiment_name}")
    print("="*70)
    print(f"Dataset: {dataset_label}")
    print(f"Model: {model}")
    print(f"Prompt version: {prompt_version}")
    print(f"Batch size: {batch_size}")
    print(f"")
    print(f"Input: {local_input}")
    print(f"Output: outputs/llm_classification/{experiment_name}_results.parquet")
    print("="*70)

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
        prompt_version=prompt_version,
        batch_size=batch_size,
        checkpoint_interval=99999,  # Effectively disable checkpoints
    )

    # Log experiment metadata
    experiment_log = {
        "experiment_name": experiment_name,
        "model": model,
        "prompt_version": prompt_version,
        "dataset": "pilot" if pilot else "full",
        "n_speeches": summary["total_speeches"],
        "successful": summary["successful"],
        "failed": summary["failed"],
        "total_tokens": summary["total_tokens"],
        "elapsed_minutes": summary["elapsed_seconds"] / 60,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Experiment: {experiment_name}")
    print(f"Model: {model}")
    print(f"Speeches: {summary['successful']}/{summary['total_speeches']} successful")
    print(f"Tokens: {summary['total_tokens']:,}")
    print(f"Time: {summary['elapsed_seconds']/60:.1f} minutes")
    print(f"")
    print(f"Results saved to Modal volume: /results/{output_file}")
    print(f"")
    print(f"To download:")
    print(f"  modal volume get suffrage-results {output_file} outputs/llm_classification/")
    print("="*70)

    # Save experiment log locally
    import json
    log_file = f"outputs/llm_classification/{experiment_name}_log.json"
    with open(log_file, "w") as f:
        json.dump(experiment_log, f, indent=2)
    print(f"\nExperiment log saved: {log_file}")

    return summary
