"""
Modal app for multi-model suffrage speech classification.
Tests multiple models in parallel in a single run.
"""

import json
import os
from typing import Dict, List
import time

import modal
import pandas as pd

# Modal app setup
app = modal.App("suffrage-multimodel-classification")

# Create persistent volume for results
volume = modal.Volume.from_name("suffrage-results", create_if_missing=True)

# Docker image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install("pandas", "pyarrow", "requests", "openai")
)

# Prompt template (v4 with improved confidence scoring)
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
    timeout=600,
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=1.0,
    ),
)
def classify_speech_with_model(
    speech_data: Dict,
    model: str,
    prompt_version: str = "v4",
) -> Dict:
    """Classify a single speech with a specific model."""
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
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1500,
            },
            timeout=120,
        )
        response.raise_for_status()

        result = response.json()
        llm_output = result['choices'][0]['message']['content'].strip()

        # Parse JSON
        try:
            if llm_output.startswith("```"):
                llm_output = llm_output.split("```")[1]
                if llm_output.startswith("json"):
                    llm_output = llm_output[4:]
                llm_output = llm_output.strip()

            classification = json.loads(llm_output)
        except json.JSONDecodeError as e:
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

    except Exception as e:
        return {
            "speech_id": speech_data['speech_id'],
            "debate_id": speech_data['debate_id'],
            "error": "api_call_failed",
            "error_detail": str(e),
            "model": model,
            "api_success": False,
        }


@app.function(
    image=image,
    volumes={"/results": volume},
    timeout=36000,
)
def run_multimodel_classification(
    input_file: str,
    models: List[str],
    batch_size: int = 50,
) -> Dict:
    """Run classification on speeches with multiple models in parallel."""
    import pandas as pd

    print(f"Loading input from {input_file}...")
    df = pd.read_parquet(input_file)
    speeches = df.to_dict('records')

    print(f"Processing {len(speeches)} speeches with {len(models)} models...")
    print(f"Models: {', '.join(models)}")

    # Create tasks for all model-speech combinations
    tasks = []
    for model in models:
        for speech in speeches:
            tasks.append((speech, model))

    print(f"Total tasks: {len(tasks)} ({len(speeches)} speeches × {len(models)} models)")

    start_time = time.time()
    results_by_model = {model: [] for model in models}

    # Process all tasks in parallel
    print(f"\nProcessing in batches of {batch_size}...")
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tasks) + batch_size - 1) // batch_size

        print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} tasks)...")

        # Run batch in parallel
        batch_results = list(classify_speech_with_model.starmap(batch))

        # Organize by model
        for result in batch_results:
            model = result.get('model')
            if model in results_by_model:
                results_by_model[model].append(result)

        # Progress update
        completed = sum(len(results) for results in results_by_model.values())
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        remaining_tasks = len(tasks) - completed
        eta_seconds = remaining_tasks / rate if rate > 0 else 0

        print(f"Progress: {completed}/{len(tasks)} ({100*completed/len(tasks):.1f}%)")
        print(f"Rate: {rate:.1f} tasks/sec")
        print(f"ETA: {eta_seconds/60:.1f} minutes")

    # Save results for each model
    print(f"\nSaving results...")
    summary = {}

    for model, results in results_by_model.items():
        model_name = model.split('/')[-1].replace(':', '-')
        output_file = f"pilot_results_{model_name}.parquet"

        results_df = pd.DataFrame(results)
        results_df.to_parquet(f"/results/{output_file}", index=False)

        success_count = results_df['api_success'].sum() if 'api_success' in results_df else 0
        total_tokens = results_df['tokens_used'].sum() if 'tokens_used' in results_df else 0

        summary[model_name] = {
            "total": len(results),
            "successful": int(success_count),
            "failed": len(results) - int(success_count),
            "total_tokens": int(total_tokens),
            "output_file": output_file,
        }

        print(f"\n{model_name}:")
        print(f"  Saved to: {output_file}")
        print(f"  Success: {success_count}/{len(results)}")
        print(f"  Tokens: {total_tokens:,}")

    volume.commit()

    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("ALL MODELS COMPLETE")
    print("="*60)
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Models processed: {len(models)}")
    print(f"Speeches per model: {len(speeches)}")
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

    df = pd.read_parquet(io.BytesIO(parquet_bytes))
    print(f"Received {len(df)} speeches")

    output_path = f"/results/{remote_filename}"
    df.to_parquet(output_path, index=False)
    volume.commit()

    print(f"Uploaded to Modal volume: {output_path}")
    return len(df)


@app.local_entrypoint()
def main(
    pilot: bool = False,
    models: str = "openai/gpt-4o-mini,anthropic/claude-3.5-sonnet",
    batch_size: int = 50,
):
    """
    Main entry point for multi-model classification.

    Usage:
        # Pilot with default models (gpt-4o-mini + claude-3.5-sonnet)
        modal run modal_multimodel_classification.py --pilot

        # Pilot with custom models
        modal run modal_multimodel_classification.py --pilot --models "openai/gpt-4o,anthropic/claude-3-haiku"

        # Full dataset
        modal run modal_multimodel_classification.py --models "openai/gpt-4o-mini"

    Args:
        pilot: Run on pilot sample (100 speeches) instead of full dataset
        models: Comma-separated list of model IDs
        batch_size: Number of parallel API calls
    """
    # Parse models
    model_list = [m.strip() for m in models.split(',')]

    if pilot:
        local_input = "outputs/llm_classification/pilot_input.parquet"
        remote_input = "pilot_input.parquet"
        print("="*60)
        print("MULTI-MODEL PILOT STUDY")
        print("="*60)
    else:
        local_input = "outputs/llm_classification/full_input.parquet"
        remote_input = "full_input.parquet"
        print("="*60)
        print("MULTI-MODEL FULL DATASET")
        print("="*60)

    print(f"Input: {local_input}")
    print(f"Models: {', '.join(model_list)}")
    print(f"Batch size: {batch_size}")
    print("="*60)

    # Upload input
    print("\nUploading input data...")
    import pandas as pd
    import io

    local_df = pd.read_parquet(local_input)
    print(f"Loaded {len(local_df)} speeches")

    buffer = io.BytesIO()
    local_df.to_parquet(buffer, index=False)
    parquet_bytes = buffer.getvalue()

    n_speeches = upload_input_data.remote(parquet_bytes, remote_input)
    print(f"Uploaded {n_speeches} speeches")

    # Run classification
    print("\nStarting multi-model classification...")
    summary = run_multimodel_classification.remote(
        input_file=f"/results/{remote_input}",
        models=model_list,
        batch_size=batch_size,
    )

    print("\n" + "="*60)
    print("JOB COMPLETE")
    print("="*60)
    print("\nResults per model:")
    for model_name, stats in summary.items():
        print(f"\n{model_name}:")
        print(f"  File: {stats['output_file']}")
        print(f"  Success: {stats['successful']}/{stats['total']}")
        print(f"  Tokens: {stats['total_tokens']:,}")

    print("\n" + "="*60)
    print("To download all results:")
    print("="*60)
    for model_name, stats in summary.items():
        print(f"modal volume get suffrage-results {stats['output_file']} ./outputs/llm_classification/")

    print("\nThen analyze:")
    print("  python3 compare_models.py")
    print("="*60)

    return summary
