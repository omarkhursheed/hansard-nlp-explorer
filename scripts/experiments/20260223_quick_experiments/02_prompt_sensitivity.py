"""
Prompt sensitivity test: are stance classifications robust to prompt rephrasing?

Reclassifies 100 speeches with 3 prompt variants and measures cross-prompt
agreement. High agreement (kappa > 0.8) shows results are not prompt-dependent.

Usage:
    python scripts/experiments/20260223_quick_experiments/02_prompt_sensitivity.py
    python scripts/experiments/20260223_quick_experiments/02_prompt_sensitivity.py --n 50
"""
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-20250514"
MAX_CONCURRENT = 10

VALIDATION_SAMPLE = Path("outputs/validation/validation_sample.parquet")
OUTPUT_DIR = Path("outputs/experiments/20260223_quick_experiments")
RESULTS_PATH = OUTPUT_DIR / "02_prompt_sensitivity.json"

STANCE_LABELS = ["for", "against", "both", "neutral", "irrelevant"]

# ---------------------------------------------------------------------------
# Three prompt variants that ask the same thing differently
# ---------------------------------------------------------------------------

PROMPTS = {
    "original": {
        "system": (
            "You are an argument-mining assistant labeling a single parliamentary "
            "speech. You are to think and reason carefully."
        ),
        "user": (
            "Classify the following UK Parliamentary speech's stance on women's "
            "suffrage (the right to vote). Respond with ONLY a JSON object.\n\n"
            "Stances:\n"
            "- \"for\": supports women's right to vote\n"
            "- \"against\": opposes women's right to vote\n"
            "- \"both\": contains arguments on both sides\n"
            "- \"neutral\": genuine indifference\n"
            "- \"irrelevant\": not about women's suffrage at all\n\n"
            "SPEECH:\n---\n{text}\n---\n\n"
            "{{\"stance\": \"<label>\", \"confidence\": <0.0-1.0>}}"
        ),
    },
    "reordered": {
        "system": (
            "You are a careful text classifier specializing in historical "
            "parliamentary debates."
        ),
        "user": (
            "Read the following speech from the UK Parliament and determine "
            "whether it relates to women's suffrage, and if so, what position "
            "it takes.\n\n"
            "SPEECH:\n---\n{text}\n---\n\n"
            "Choose one stance label:\n"
            "- \"irrelevant\": the speech is not about women's voting rights\n"
            "- \"for\": the speaker supports women's enfranchisement\n"
            "- \"against\": the speaker opposes women's enfranchisement\n"
            "- \"both\": arguments appear on both sides\n"
            "- \"neutral\": indifferent or no clear position\n\n"
            "Respond with ONLY: {{\"stance\": \"<label>\", \"confidence\": <0.0-1.0>}}"
        ),
    },
    "minimal": {
        "system": "You classify parliamentary speeches.",
        "user": (
            "Is this UK Parliament speech about women's suffrage? If so, is the "
            "speaker for or against?\n\n"
            "{text}\n\n"
            "Reply as JSON: {{\"stance\": \"for|against|both|neutral|irrelevant\", "
            "\"confidence\": <0-1>}}"
        ),
    },
}


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def parse_stance(raw: str) -> tuple[str, float]:
    """Extract stance and confidence from response."""
    raw = raw.strip()
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

    # Regex fallback
    import re
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
    prompt_name: str,
    prompt: dict,
) -> dict:
    """Classify a speech with a specific prompt variant."""
    async with semaphore:
        try:
            user_content = prompt["user"].replace("{text}", text[:3000])
            response = await client.messages.create(
                model=MODEL,
                max_tokens=250,
                system=prompt["system"],
                messages=[{"role": "user", "content": user_content}],
            )
            raw = response.content[0].text
            stance, confidence = parse_stance(raw)
            return {
                "speech_id": speech_id,
                "prompt": prompt_name,
                "stance": stance,
                "confidence": confidence,
                "tokens": response.usage.input_tokens + response.usage.output_tokens,
                "error": None,
            }
        except Exception as e:
            return {
                "speech_id": speech_id,
                "prompt": prompt_name,
                "stance": "error",
                "confidence": 0.0,
                "tokens": 0,
                "error": str(e),
            }


async def run_all(speeches: pd.DataFrame) -> list[dict]:
    """Run all prompt variants on all speeches."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = []
    for _, row in speeches.iterrows():
        text = row.get("target_text") or row.get("text", "")
        for prompt_name, prompt in PROMPTS.items():
            tasks.append(
                classify_one(
                    client, semaphore, row["speech_id"], text, prompt_name, prompt
                )
            )

    print(f"Running {len(tasks)} API calls ({len(speeches)} speeches x {len(PROMPTS)} prompts)...")

    results = []
    start = time.time()
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
            elapsed = time.time() - start
            errors = sum(1 for r in results if r["error"])
            print(f"  [{i+1}/{len(tasks)}] {elapsed:.0f}s, {errors} errors")

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(results: list[dict], original_stances: dict) -> dict:
    """Compute cross-prompt agreement."""
    # Pivot: speech_id x prompt -> stance
    df = pd.DataFrame(results)
    df = df[df["stance"] != "error"]

    pivot = df.pivot(index="speech_id", columns="prompt", values="stance")
    pivot = pivot.dropna()
    print(f"\nSpeeches with all 3 prompt variants: {len(pivot)}")

    # Use only prompts that appear in the data
    prompt_names = [p for p in PROMPTS.keys() if p in pivot.columns]
    if len(prompt_names) < 2:
        print("ERROR: fewer than 2 prompt variants succeeded")
        return {"error": "insufficient prompt variants"}
    pairwise = {}
    for i, p1 in enumerate(prompt_names):
        for p2 in prompt_names[i + 1:]:
            y1 = pivot[p1].values
            y2 = pivot[p2].values
            agree = np.mean(y1 == y2)
            kappa = cohen_kappa_score(y1, y2)
            pairwise[f"{p1}_vs_{p2}"] = {
                "agreement": round(float(agree), 4),
                "kappa": round(float(kappa), 4),
            }

    # All-three agreement
    all_agree = np.mean(
        (pivot[prompt_names[0]] == pivot[prompt_names[1]])
        & (pivot[prompt_names[1]] == pivot[prompt_names[2]])
    )

    # Agreement with original pipeline labels
    vs_original = {}
    for prompt_name in prompt_names:
        subset = pivot.reset_index()
        subset["original_stance"] = subset["speech_id"].map(original_stances)
        subset = subset.dropna(subset=["original_stance"])
        if len(subset) > 0:
            agree = np.mean(subset[prompt_name] == subset["original_stance"])
            kappa = cohen_kappa_score(subset[prompt_name], subset["original_stance"])
            vs_original[prompt_name] = {
                "agreement": round(float(agree), 4),
                "kappa": round(float(kappa), 4),
                "n": int(len(subset)),
            }

    # Per-stance stability
    per_stance = {}
    for stance in STANCE_LABELS:
        mask = pivot[prompt_names[0]] == stance
        if mask.sum() < 5:
            continue
        # How often do the other prompts agree?
        subset = pivot[mask]
        agree_rates = {}
        for pn in prompt_names[1:]:
            agree_rates[pn] = round(float((subset[pn] == stance).mean()), 4)
        per_stance[stance] = {
            "n": int(mask.sum()),
            "other_prompt_agreement": agree_rates,
        }

    total_tokens = sum(r["tokens"] for r in results)

    return {
        "n_speeches": int(len(pivot)),
        "pairwise_agreement": pairwise,
        "all_three_agreement": round(float(all_agree), 4),
        "vs_original_pipeline": vs_original,
        "per_stance_stability": per_stance,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(
            total_tokens * 3 / 1_000_000 + total_tokens * 0.3 * 15 / 1_000_000, 4
        ),
    }


def print_results(analysis: dict):
    """Pretty-print results."""
    print("\n" + "=" * 60)
    print("PROMPT SENSITIVITY RESULTS")
    print("=" * 60)

    print(f"\nSpeeches tested: {analysis['n_speeches']}")
    print(f"All-three-prompts agree: {analysis['all_three_agreement']:.1%}")

    print("\n--- Pairwise Agreement ---")
    for pair, stats in analysis["pairwise_agreement"].items():
        print(f"  {pair:>30s}: agreement={stats['agreement']:.1%}, kappa={stats['kappa']:.3f}")

    print("\n--- vs Original Pipeline Labels ---")
    for prompt, stats in analysis["vs_original_pipeline"].items():
        print(f"  {prompt:>15s}: agreement={stats['agreement']:.1%}, kappa={stats['kappa']:.3f} (n={stats['n']})")

    print("\n--- Per-Stance Stability ---")
    for stance, stats in analysis["per_stance_stability"].items():
        rates = stats["other_prompt_agreement"]
        avg = np.mean(list(rates.values()))
        print(f"  {stance:>12s} (n={stats['n']}): avg stability={avg:.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of speeches to test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load validation sample (already stratified)
    df = pd.read_parquet(VALIDATION_SAMPLE)

    # Sample n speeches, stratified by stance
    rng = random.Random(args.seed)
    sampled = []
    for stance in STANCE_LABELS:
        pool = df[df["stance"] == stance]
        n = min(args.n // len(STANCE_LABELS), len(pool))
        indices = rng.sample(range(len(pool)), n)
        sampled.append(pool.iloc[indices])
    speeches = pd.concat(sampled, ignore_index=True)
    print(f"Sampled {len(speeches)} speeches for sensitivity test")
    print(f"Stance distribution: {speeches['stance'].value_counts().to_dict()}")

    # Original pipeline stances
    original_stances = dict(zip(df["speech_id"], df["stance"]))

    # Run
    results = asyncio.run(run_all(speeches))

    # Analyze
    analysis = analyze(results, original_stances)

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    print_results(analysis)


if __name__ == "__main__":
    main()
