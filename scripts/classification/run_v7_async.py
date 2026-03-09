"""
Run V7 classification on full suffrage dataset using async API calls.

Uses regular ANTHROPIC_API_KEY with concurrency control.

Usage:
    python scripts/classification/run_v7_async.py
    python scripts/classification/run_v7_async.py --resume   # resume from checkpoint
"""
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import anthropic
import pandas as pd

MODEL = "claude-sonnet-4-6"
MAX_CONCURRENT = 20
MAX_TOKENS = 500

OUTPUT_DIR = Path("outputs/llm_classification")
CHECKPOINT_PATH = OUTPUT_DIR / "v7_checkpoint.jsonl"
RESULTS_PATH = OUTPUT_DIR / "v7_full_results.parquet"

# Load V7 prompt
with open("scripts/classification/PROMPT_V7_DRAFT.md") as f:
    content = f.read()
blocks = content.split("```")
raw_prompt = blocks[-2].strip()
sys_end = raw_prompt.index("\nUSER\n")
SYSTEM_PROMPT = raw_prompt[len("SYSTEM\n"):sys_end].strip()
USER_TEMPLATE = raw_prompt[sys_end + len("\nUSER\n"):].strip()


def parse_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Find JSON block in reasoning text
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("{") and "stance" in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass

    # Try to find a complete JSON object
    start = raw.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start : i + 1])
                    except json.JSONDecodeError:
                        break

    match = re.search(r'"stance"\s*:\s*"(\w+)"', raw)
    if match:
        return {"stance": match.group(1), "confidence": 0.5}

    return {"stance": "error", "parse_error": True}


async def classify_one(client, semaphore, speech_id, target_text, context_text):
    async with semaphore:
        for attempt in range(3):
            try:
                user = USER_TEMPLATE.replace(
                    "{target_text}", target_text[:5000]
                ).replace(
                    "{context_text}",
                    context_text[:3000]
                    if context_text and str(context_text) != "[No context available]"
                    else "[No context available]",
                )
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user}],
                )
                raw = response.content[0].text
                parsed = parse_response(raw)
                parsed["speech_id"] = speech_id
                parsed["input_tokens"] = response.usage.input_tokens
                parsed["output_tokens"] = response.usage.output_tokens
                parsed["error"] = None
                return parsed
            except anthropic.RateLimitError:
                await asyncio.sleep(5 * (attempt + 1))
            except Exception as e:
                if attempt == 2:
                    return {
                        "speech_id": speech_id,
                        "stance": "error",
                        "error": str(e),
                    }
                await asyncio.sleep(2)
    return {"speech_id": speech_id, "stance": "error", "error": "max retries"}


async def run_all(speeches: pd.DataFrame, done_ids: set):
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    todo = speeches[~speeches["speech_id"].isin(done_ids)]
    print(f"API calls needed: {len(todo)} ({len(done_ids)} already done)")

    if len(todo) == 0:
        return []

    tasks = [
        classify_one(
            client,
            semaphore,
            row["speech_id"],
            row["target_text"],
            row.get("context_text", ""),
        )
        for _, row in todo.iterrows()
    ]

    results = []
    start = time.time()
    checkpoint_buffer = []

    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        checkpoint_buffer.append(result)

        if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
            elapsed = time.time() - start
            errors = sum(1 for r in results if r.get("error"))
            rate = (i + 1) / elapsed * 60
            remaining = (len(tasks) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{len(tasks)}] {elapsed:.0f}s, "
                f"{errors} errors, {rate:.0f}/min, ~{remaining:.0f}min left"
            )

            # Checkpoint
            with open(CHECKPOINT_PATH, "a") as f:
                for r in checkpoint_buffer:
                    f.write(json.dumps(r, default=str) + "\n")
            checkpoint_buffer = []

    # Flush remaining
    if checkpoint_buffer:
        with open(CHECKPOINT_PATH, "a") as f:
            for r in checkpoint_buffer:
                f.write(json.dumps(r, default=str) + "\n")

    return results


def save_results(all_results: list[dict]):
    """Save as parquet for analysis."""
    flat = []
    for r in all_results:
        row = {
            "speech_id": r.get("speech_id"),
            "stance": r.get("stance"),
            "stance_rationale": r.get("stance_rationale", ""),
            "confidence": r.get("confidence", 0),
            "context_helpful": r.get("context_helpful", False),
            "input_tokens": r.get("input_tokens", 0),
            "output_tokens": r.get("output_tokens", 0),
            "error": r.get("error"),
        }
        sexism = r.get("sexism", {})
        if isinstance(sexism, dict):
            row["binary"] = sexism.get("binary", "")
            row["axis_a_label"] = sexism.get("axis_a_label", "")
            row["axis_a_subcategory"] = sexism.get("axis_a_subcategory", "")
            row["axis_b_label"] = sexism.get("axis_b_label", "")
            row["axis_c_label"] = sexism.get("axis_c_label", "")
            row["sexism_quote"] = sexism.get("quote", "")
        flat.append(row)

    df = pd.DataFrame(flat)
    df.to_parquet(RESULTS_PATH, index=False)
    print(f"Saved {len(df)} results to {RESULTS_PATH}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Load input data
    inp = pd.read_parquet(
        "outputs/llm_classification/full_input_context_3_expanded.parquet"
    )
    print(f"Loaded {len(inp)} speeches")

    # Load checkpoint
    done_ids = set()
    all_results = []
    if args.resume and CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            for line in f:
                rec = json.loads(line)
                all_results.append(rec)
                done_ids.add(rec["speech_id"])
        print(f"Resumed: {len(done_ids)} from checkpoint")
    elif not args.resume and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    # Run
    new_results = asyncio.run(run_all(inp, done_ids))
    all_results.extend(new_results)

    # Save
    df = save_results(all_results)

    # Summary
    from collections import Counter

    stances = Counter(df["stance"])
    errors = stances.pop("error", 0)
    total_in = df["input_tokens"].sum()
    total_out = df["output_tokens"].sum()

    print(f"\nTotal: {len(df)}, errors: {errors}")
    print(f"Stance distribution:")
    for s, n in stances.most_common():
        print(f"  {s}: {n}")

    if "binary" in df.columns:
        binary = Counter(df[df["stance"] != "error"]["binary"])
        print(f"Sexism binary:")
        for s, n in binary.most_common():
            print(f"  {s}: {n}")

    print(f"Cost: ~${total_in * 3 / 1e6 + total_out * 15 / 1e6:.2f}")


if __name__ == "__main__":
    main()
