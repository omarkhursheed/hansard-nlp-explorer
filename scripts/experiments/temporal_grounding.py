"""
Temporal grounding experiment: can an LLM identify the era of a parliamentary
speech from language alone?

Validates that Hansard speeches carry genuine temporal signal -- if the model
can reliably place speeches in the correct era, temporal trend analyses of
sexism classifications are meaningful rather than noise.

Usage:
    python scripts/experiments/temporal_grounding.py
    python scripts/experiments/temporal_grounding.py --per-era 50 --min-words 100
    python scripts/experiments/temporal_grounding.py --resume  # resume from checkpoint
"""
import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Era definitions
# ---------------------------------------------------------------------------

ERAS = {
    "pre-1870": (0, 1869),
    "1870-1900": (1870, 1900),
    "1900-1920": (1901, 1920),
    "1920-1950": (1921, 1950),
    "post-1950": (1951, 2100),
}
ERA_LABELS = list(ERAS.keys())
ERA_ORDER = {label: i for i, label in enumerate(ERA_LABELS)}

SPEECH_DIR = Path("data-hansard/derived_v2/speeches_complete")
OUTPUT_DIR = Path("outputs/experiments")
CHECKPOINT_PATH = OUTPUT_DIR / "temporal_grounding_checkpoint.jsonl"
RESULTS_PATH = OUTPUT_DIR / "temporal_grounding_results.json"

MODEL = "claude-sonnet-4-20250514"
MAX_CONCURRENT = 10
MAX_TEXT_CHARS = 3000  # Truncate very long speeches to save tokens

SYSTEM_PROMPT = """\
You are a historian specializing in British parliamentary language and rhetoric. \
Your task is to identify the approximate era of a UK Parliamentary speech based \
solely on its language, vocabulary, rhetorical style, and content."""

USER_PROMPT_TEMPLATE = """\
Below is a speech delivered in the UK Parliament. Based only on the language, \
vocabulary, rhetorical style, and topics discussed, classify it into one of \
these five eras:

1. pre-1870 (before 1870)
2. 1870-1900
3. 1900-1920
4. 1920-1950
5. post-1950 (after 1950)

SPEECH:
---
{text}
---

Respond in this exact JSON format:
{{"era": "<one of: pre-1870, 1870-1900, 1900-1920, 1920-1950, post-1950>", "rationale": "<brief rationale, 1-2 sentences>"}}"""


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def year_to_era(year: int) -> str:
    for label, (lo, hi) in ERAS.items():
        if lo <= year <= hi:
            return label
    return "unknown"


def sample_speeches(per_era: int, min_words: int, seed: int) -> pd.DataFrame:
    """Stratified sample: per_era speeches from each era."""
    rng = random.Random(seed)

    all_candidates = []
    for parquet_file in sorted(SPEECH_DIR.glob("speeches_*.parquet")):
        df = pd.read_parquet(
            parquet_file,
            columns=["speech_id", "text", "word_count", "year", "speaker", "gender"],
        )
        # Filter by minimum word count
        df = df[df["word_count"] >= min_words]
        if len(df) > 0:
            df["era"] = df["year"].apply(year_to_era)
            all_candidates.append(df)

    full = pd.concat(all_candidates, ignore_index=True)
    print(f"Total candidate speeches (>={min_words} words): {len(full):,}")
    for era in ERA_LABELS:
        n = len(full[full["era"] == era])
        print(f"  {era}: {n:,}")

    # Stratified sample
    sampled = []
    for era in ERA_LABELS:
        pool = full[full["era"] == era]
        n = min(per_era, len(pool))
        indices = rng.sample(range(len(pool)), n)
        sampled.append(pool.iloc[indices])

    result = pd.concat(sampled, ignore_index=True)
    print(f"\nSampled {len(result)} speeches ({per_era} per era)")
    return result


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def build_prompt(text: str) -> str:
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + " [...]"
    return USER_PROMPT_TEMPLATE.format(text=text)


def parse_response(raw: str) -> dict:
    """Extract era and rationale from model response."""
    # Try JSON parse first
    raw = raw.strip()
    # Handle markdown code blocks
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

    try:
        data = json.loads(raw)
        era = data.get("era", "").strip()
        rationale = data.get("rationale", "")
        if era in ERA_LABELS:
            return {"predicted_era": era, "rationale": rationale, "parse_ok": True}
    except json.JSONDecodeError:
        pass

    # Fallback: search for era label in text
    for label in ERA_LABELS:
        if label in raw:
            return {"predicted_era": label, "rationale": raw, "parse_ok": False}

    return {"predicted_era": "unknown", "rationale": raw, "parse_ok": False}


async def classify_speech(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    speech_id: str,
    text: str,
) -> dict:
    """Classify a single speech with rate limiting."""
    async with semaphore:
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_prompt(text)}],
            )
            raw = response.content[0].text
            result = parse_response(raw)
            result["speech_id"] = speech_id
            result["raw_response"] = raw
            result["input_tokens"] = response.usage.input_tokens
            result["output_tokens"] = response.usage.output_tokens
            result["error"] = None
            return result
        except Exception as e:
            return {
                "speech_id": speech_id,
                "predicted_era": "error",
                "rationale": "",
                "raw_response": "",
                "parse_ok": False,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e),
            }


async def run_classifications(
    speeches: pd.DataFrame, done_ids: set
) -> list[dict]:
    """Run all LLM classifications with concurrency control."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Filter out already-done speeches
    todo = speeches[~speeches["speech_id"].isin(done_ids)]
    print(f"API calls needed: {len(todo)} ({len(done_ids)} already done)")

    if len(todo) == 0:
        return []

    tasks = [
        classify_speech(client, semaphore, row["speech_id"], row["text"])
        for _, row in todo.iterrows()
    ]

    results = []
    start = time.time()
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)

        # Checkpoint every 25
        if (i + 1) % 25 == 0 or (i + 1) == len(tasks):
            elapsed = time.time() - start
            errors = sum(1 for r in results if r["error"])
            print(
                f"  [{i+1}/{len(tasks)}] "
                f"{elapsed:.0f}s elapsed, "
                f"{errors} errors"
            )
            # Append to checkpoint
            with open(CHECKPOINT_PATH, "a") as f:
                for r in results[max(0, len(results) - 25):]:
                    f.write(json.dumps(r) + "\n")

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_results(speeches: pd.DataFrame, predictions: dict) -> dict:
    """Compute accuracy, confusion matrix, and era distance metrics."""
    records = []
    for _, row in speeches.iterrows():
        sid = row["speech_id"]
        if sid not in predictions:
            continue
        pred = predictions[sid]
        if pred["predicted_era"] in ("error", "unknown"):
            continue
        records.append({
            "speech_id": sid,
            "true_era": row["era"],
            "predicted_era": pred["predicted_era"],
            "rationale": pred["rationale"],
            "year": row["year"],
        })

    df = pd.DataFrame(records)
    if len(df) == 0:
        print("No valid predictions to analyze.")
        return {}

    y_true = df["true_era"].values
    y_pred = df["predicted_era"].values

    # Accuracy
    accuracy = np.mean(y_true == y_pred)

    # Adjacent accuracy (within 1 era)
    true_idx = np.array([ERA_ORDER[e] for e in y_true])
    pred_idx = np.array([ERA_ORDER[e] for e in y_pred])
    adjacent_accuracy = np.mean(np.abs(true_idx - pred_idx) <= 1)

    # Mean absolute era distance
    mean_era_distance = np.mean(np.abs(true_idx - pred_idx))

    # Cohen's kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=ERA_LABELS)

    # Per-era accuracy
    per_era = {}
    for era in ERA_LABELS:
        mask = y_true == era
        if mask.sum() > 0:
            per_era[era] = {
                "n": int(mask.sum()),
                "accuracy": float(np.mean(y_pred[mask] == era)),
                "adjacent_accuracy": float(
                    np.mean(np.abs(pred_idx[mask] - ERA_ORDER[era]) <= 1)
                ),
            }

    # Classification report
    report = classification_report(
        y_true, y_pred, labels=ERA_LABELS, output_dict=True, zero_division=0
    )

    # Token usage
    total_input = sum(p.get("input_tokens", 0) for p in predictions.values())
    total_output = sum(p.get("output_tokens", 0) for p in predictions.values())

    results = {
        "n_speeches": len(df),
        "n_errors": len(speeches) - len(df),
        "accuracy": round(accuracy, 4),
        "adjacent_accuracy": round(adjacent_accuracy, 4),
        "mean_era_distance": round(mean_era_distance, 4),
        "kappa": round(kappa, 4),
        "confusion_matrix": {
            "labels": ERA_LABELS,
            "matrix": cm.tolist(),
        },
        "per_era": per_era,
        "classification_report": {
            k: v for k, v in report.items()
            if k in ERA_LABELS or k in ("macro avg", "weighted avg")
        },
        "token_usage": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "estimated_cost_usd": round(
                total_input * 3 / 1_000_000 + total_output * 15 / 1_000_000, 4
            ),
        },
        "model": MODEL,
        "sample_predictions": [
            {
                "year": int(r["year"]),
                "true_era": r["true_era"],
                "predicted_era": r["predicted_era"],
                "rationale": r["rationale"],
            }
            for r in records[:10]
        ],
    }

    return results


def print_results(results: dict):
    """Pretty-print results to console."""
    print("\n" + "=" * 60)
    print("TEMPORAL GROUNDING RESULTS")
    print("=" * 60)

    print(f"\nSpeeches analyzed: {results['n_speeches']}")
    print(f"Errors/unparseable: {results['n_errors']}")
    print(f"Model: {results['model']}")

    print(f"\n--- Overall Metrics ---")
    print(f"Exact accuracy:    {results['accuracy']:.1%}")
    print(f"Adjacent accuracy: {results['adjacent_accuracy']:.1%} (within 1 era)")
    print(f"Mean era distance: {results['mean_era_distance']:.2f}")
    print(f"Cohen's kappa:     {results['kappa']:.3f}")

    print(f"\n--- Per-Era Accuracy ---")
    for era, stats in results["per_era"].items():
        print(
            f"  {era:>12s}: {stats['accuracy']:5.1%} exact, "
            f"{stats['adjacent_accuracy']:5.1%} adjacent (n={stats['n']})"
        )

    print(f"\n--- Confusion Matrix ---")
    labels = results["confusion_matrix"]["labels"]
    matrix = results["confusion_matrix"]["matrix"]
    header = "            " + "".join(f"{l:>12s}" for l in labels)
    print(header)
    for i, row in enumerate(matrix):
        row_str = f"{labels[i]:>12s}" + "".join(f"{v:>12d}" for v in row)
        print(row_str)

    usage = results["token_usage"]
    print(f"\n--- Token Usage ---")
    print(f"Input:  {usage['input_tokens']:,}")
    print(f"Output: {usage['output_tokens']:,}")
    print(f"Est. cost: ${usage['estimated_cost_usd']:.2f}")

    print(f"\n--- Sample Predictions ---")
    for p in results["sample_predictions"][:5]:
        match = "OK" if p["true_era"] == p["predicted_era"] else "MISS"
        print(
            f"  [{match:4s}] year={p['year']}, "
            f"true={p['true_era']}, pred={p['predicted_era']}"
        )
        print(f"         {p['rationale'][:100]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Temporal grounding experiment")
    parser.add_argument("--per-era", type=int, default=50, help="Speeches per era")
    parser.add_argument("--min-words", type=int, default=100, help="Min word count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Sample speeches
    print("Sampling speeches...")
    speeches = sample_speeches(args.per_era, args.min_words, args.seed)

    # 2. Load checkpoint if resuming
    done_ids = set()
    all_predictions = {}
    if args.resume and CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            for line in f:
                rec = json.loads(line)
                all_predictions[rec["speech_id"]] = rec
                done_ids.add(rec["speech_id"])
        print(f"Loaded {len(done_ids)} predictions from checkpoint")
    elif not args.resume and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    # 3. Run LLM classifications
    print("\nRunning LLM classifications...")
    new_results = asyncio.run(run_classifications(speeches, done_ids))
    for r in new_results:
        all_predictions[r["speech_id"]] = r

    # 4. Analyze
    print("\nAnalyzing results...")
    speeches["era"] = speeches["year"].apply(year_to_era)
    results = analyze_results(speeches, all_predictions)

    if not results:
        return

    # 5. Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # 6. Print
    print_results(results)


if __name__ == "__main__":
    main()
