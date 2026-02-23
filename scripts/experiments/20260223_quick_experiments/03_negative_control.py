"""
Negative control / specificity test: does the classifier correctly identify
non-suffrage speeches as irrelevant?

Samples 100 random parliamentary speeches from topics unrelated to suffrage
(budgets, foreign affairs, railways, etc.) and runs them through the same
classification prompt. High irrelevant rate (>95%) proves specificity.

Usage:
    python scripts/experiments/20260223_quick_experiments/03_negative_control.py
    python scripts/experiments/20260223_quick_experiments/03_negative_control.py --n 100
"""
import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-20250514"
MAX_CONCURRENT = 10

SPEECH_DIR = Path("data-hansard/derived_v2/speeches_complete")
OUTPUT_DIR = Path("outputs/experiments/20260223_quick_experiments")
RESULTS_PATH = OUTPUT_DIR / "03_negative_control.json"

STANCE_LABELS = ["for", "against", "both", "neutral", "irrelevant"]

# Keywords that indicate a speech MIGHT be about suffrage -- we EXCLUDE these
SUFFRAGE_KEYWORDS = re.compile(
    r"\b(suffrag|women.*vote|female.*franchise|enfranch|woman.*ballot"
    r"|sex.*disqualif|representation.*people.*act"
    r"|women.*parliament|lady.*member|female.*member)\b",
    re.IGNORECASE,
)

# Classification prompt (matches the original pipeline's intent)
SYSTEM_PROMPT = (
    "You are an argument-mining assistant labeling a single parliamentary "
    "speech. You are to think and reason carefully."
)

USER_PROMPT_TEMPLATE = (
    'Classify the following UK Parliamentary speech\'s stance on women\'s '
    'suffrage (the right to vote). Respond with ONLY a JSON object.\n\n'
    'Stances:\n'
    '- "for": supports women\'s right to vote\n'
    '- "against": opposes women\'s right to vote\n'
    '- "both": contains arguments on both sides\n'
    '- "neutral": genuine indifference\n'
    '- "irrelevant": not about women\'s suffrage at all\n\n'
    'SPEECH:\n---\n{text}\n---\n\n'
    '{{"stance": "<label>", "confidence": <0.0-1.0>}}'
)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_non_suffrage_speeches(n: int, min_words: int, seed: int) -> pd.DataFrame:
    """Sample speeches that are clearly NOT about suffrage."""
    rng = random.Random(seed)

    candidates = []
    # Sample from a spread of years
    years_to_check = list(range(1850, 2005, 5))
    rng.shuffle(years_to_check)

    for year in years_to_check:
        parquet = SPEECH_DIR / f"speeches_{year}.parquet"
        if not parquet.exists():
            continue

        df = pd.read_parquet(
            parquet,
            columns=["speech_id", "text", "word_count", "year", "speaker",
                      "gender", "title", "topic"],
        )
        # Filter: long enough, no suffrage keywords
        df = df[df["word_count"] >= min_words]
        df = df[~df["text"].str.contains(SUFFRAGE_KEYWORDS, na=False)]

        if len(df) > 0:
            # Take a few from each year
            sample_n = min(5, len(df))
            indices = rng.sample(range(len(df)), sample_n)
            candidates.append(df.iloc[indices])

        if sum(len(c) for c in candidates) >= n * 2:
            break

    full = pd.concat(candidates, ignore_index=True)
    # Final random sample
    indices = rng.sample(range(len(full)), min(n, len(full)))
    result = full.iloc[indices].reset_index(drop=True)

    print(f"Sampled {len(result)} non-suffrage speeches")
    print(f"Year range: {result['year'].min()}-{result['year'].max()}")
    if "title" in result.columns:
        print(f"Sample topics: {result['title'].head(5).tolist()}")

    return result


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def parse_stance(raw: str) -> tuple[str, float]:
    """Extract stance and confidence from response."""
    raw = raw.strip()
    # Strip markdown code blocks
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

    # Try full JSON parse
    try:
        data = json.loads(raw)
        stance = data.get("stance", "").strip().lower()
        conf = float(data.get("confidence", 0.5))
        if stance in STANCE_LABELS:
            return stance, conf
    except (json.JSONDecodeError, ValueError):
        pass

    # Try line-by-line JSON extraction (model may add reasoning before JSON)
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("{") and "stance" in line:
            try:
                data = json.loads(line)
                stance = data.get("stance", "").strip().lower()
                conf = float(data.get("confidence", 0.5))
                if stance in STANCE_LABELS:
                    return stance, conf
            except (json.JSONDecodeError, ValueError):
                pass

    # Regex fallback: find {"stance": "..."} anywhere
    match = re.search(r'"stance"\s*:\s*"(\w+)"', raw)
    if match:
        stance = match.group(1).lower()
        if stance in STANCE_LABELS:
            return stance, 0.5

    return "error", 0.0


async def classify_one(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    speech_id: str,
    text: str,
) -> dict:
    """Classify a single speech."""
    async with semaphore:
        try:
            user_content = USER_PROMPT_TEMPLATE.replace("{text}", text[:3000])
            response = await client.messages.create(
                model=MODEL,
                max_tokens=250,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            raw = response.content[0].text
            stance, confidence = parse_stance(raw)
            return {
                "speech_id": speech_id,
                "stance": stance,
                "confidence": confidence,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "error": None,
            }
        except Exception as e:
            return {
                "speech_id": speech_id,
                "stance": "error",
                "confidence": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e),
            }


async def run_all(speeches: pd.DataFrame) -> list[dict]:
    """Classify all speeches."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        classify_one(client, semaphore, row["speech_id"], row["text"])
        for _, row in speeches.iterrows()
    ]

    print(f"Running {len(tasks)} API calls...")
    results = []
    start = time.time()
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        if (i + 1) % 25 == 0 or (i + 1) == len(tasks):
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(tasks)}] {elapsed:.0f}s")

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(results: list[dict], speeches: pd.DataFrame) -> dict:
    """Analyze specificity."""
    df = pd.DataFrame(results)
    valid = df[df["stance"] != "error"]

    stance_counts = valid["stance"].value_counts().to_dict()
    n_irrelevant = stance_counts.get("irrelevant", 0)
    specificity = n_irrelevant / len(valid) if len(valid) > 0 else 0

    # False positives (classified as suffrage-related)
    false_positives = valid[valid["stance"] != "irrelevant"]
    fp_details = []
    for _, row in false_positives.iterrows():
        speech_data = speeches[speeches["speech_id"] == row["speech_id"]]
        detail = {
            "speech_id": row["speech_id"],
            "predicted_stance": row["stance"],
            "confidence": round(float(row["confidence"]), 2),
        }
        if len(speech_data) > 0:
            s = speech_data.iloc[0]
            detail["year"] = int(s["year"]) if pd.notna(s.get("year")) else None
            detail["title"] = str(s.get("title", ""))[:100]
            detail["text_preview"] = str(s["text"])[:300]
        fp_details.append(detail)

    total_input = sum(r.get("input_tokens", 0) for r in results)
    total_output = sum(r.get("output_tokens", 0) for r in results)

    return {
        "n_tested": int(len(valid)),
        "n_errors": int(len(df) - len(valid)),
        "stance_distribution": stance_counts,
        "specificity": round(float(specificity), 4),
        "n_false_positives": int(len(false_positives)),
        "false_positive_rate": round(float(1 - specificity), 4),
        "false_positive_details": fp_details,
        "mean_confidence": round(float(valid["confidence"].mean()), 3),
        "token_usage": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "estimated_cost_usd": round(
                total_input * 3 / 1_000_000 + total_output * 15 / 1_000_000, 4
            ),
        },
    }


def print_results(analysis: dict):
    """Pretty-print."""
    print("\n" + "=" * 60)
    print("NEGATIVE CONTROL RESULTS")
    print("=" * 60)

    print(f"\nSpeeches tested: {analysis['n_tested']}")
    print(f"Specificity: {analysis['specificity']:.1%}")
    print(f"False positives: {analysis['n_false_positives']}")

    print(f"\nStance distribution:")
    for stance, count in analysis["stance_distribution"].items():
        pct = count / analysis["n_tested"]
        print(f"  {stance:>12s}: {count} ({pct:.1%})")

    if analysis["false_positive_details"]:
        print(f"\n--- False Positive Details ---")
        for fp in analysis["false_positive_details"][:5]:
            print(f"  stance={fp['predicted_stance']}, conf={fp['confidence']}, "
                  f"year={fp.get('year', '?')}")
            if "title" in fp:
                print(f"  title: {fp['title']}")
            if "text_preview" in fp:
                print(f"  text: {fp['text_preview'][:150]}...")
            print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--min-words", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    speeches = sample_non_suffrage_speeches(args.n, args.min_words, args.seed)
    results = asyncio.run(run_all(speeches))
    analysis = analyze(results, speeches)

    with open(RESULTS_PATH, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    print_results(analysis)


if __name__ == "__main__":
    main()
