"""
Reclassify the 100 validation speeches with Claude Sonnet 4.6.

Uses the exact same V6 prompt and context as the original classification,
just with the newer model. Enables direct comparison.

Usage:
    python scripts/experiments/20260308_reclassify_sonnet46.py
"""
import asyncio
import json
import re
import time
from pathlib import Path

import anthropic
import pandas as pd

MODEL = "claude-sonnet-4-6"
MAX_CONCURRENT = 10
OUTPUT_DIR = Path("outputs/experiments")
RESULTS_PATH = OUTPUT_DIR / "sonnet46_reclassification.json"

# Load the exact V6 prompt from the original classification script
import ast as _ast

with open("scripts/classification/modal_suffrage_classification_v6.py") as _f:
    _tree = _ast.parse(_f.read())
for _node in _ast.walk(_tree):
    if isinstance(_node, _ast.Assign):
        for _t in _node.targets:
            if isinstance(_t, _ast.Name) and _t.id == "PROMPT_V6":
                _PROMPT_V6 = _ast.literal_eval(_node.value)
                break

# The prompt has literal "SYSTEM\n" and "\nUSER\n" markers
_sys_end = _PROMPT_V6.index("\nUSER\n")
SYSTEM_PROMPT = _PROMPT_V6[len("SYSTEM\n"):_sys_end].strip()
USER_PROMPT_TEMPLATE = _PROMPT_V6[_sys_end + len("\nUSER\n"):].strip()

print(f"Loaded V6 prompt: system={len(SYSTEM_PROMPT)} chars, user={len(USER_PROMPT_TEMPLATE)} chars")

STANCE_LABELS = ["for", "against", "both", "neutral", "irrelevant"]


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
        data = json.loads(raw)
        return data
    except json.JSONDecodeError:
        pass

    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("{") and "stance" in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass

    match = re.search(r'"stance"\s*:\s*"(\w+)"', raw)
    if match:
        return {"stance": match.group(1), "confidence": 0.5, "reasons": []}

    return {"stance": "error", "confidence": 0.0, "reasons": [], "parse_error": True}


async def classify_one(client, semaphore, speech_id, target_text, context_text):
    async with semaphore:
        try:
            user_content = USER_PROMPT_TEMPLATE.replace("{target_text}", target_text[:5000]).replace(
                "{context_text}", context_text[:3000] if context_text else "[No context available]"
            )
            response = await client.messages.create(
                model=MODEL,
                max_tokens=800,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            raw = response.content[0].text
            parsed = parse_response(raw)
            parsed["speech_id"] = speech_id
            parsed["raw_response"] = raw
            parsed["input_tokens"] = response.usage.input_tokens
            parsed["output_tokens"] = response.usage.output_tokens
            parsed["error"] = None
            return parsed
        except Exception as e:
            return {
                "speech_id": speech_id,
                "stance": "error",
                "confidence": 0.0,
                "reasons": [],
                "raw_response": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e),
            }


async def run_all(speeches):
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        classify_one(client, semaphore, row["speech_id"],
                     row["target_text"], row.get("context_text", ""))
        for _, row in speeches.iterrows()
    ]

    print(f"Running {len(tasks)} API calls with {MODEL}...")
    results = []
    start = time.time()
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        if (i + 1) % 25 == 0 or (i + 1) == len(tasks):
            elapsed = time.time() - start
            errors = sum(1 for r in results if r.get("error"))
            print(f"  [{i+1}/{len(tasks)}] {elapsed:.0f}s, {errors} errors")

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet("outputs/validation/validation_sample.parquet")
    print(f"Loaded {len(df)} speeches")

    results = asyncio.run(run_all(df))

    # Save raw results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    stances = [r["stance"] for r in results if r["stance"] != "error"]
    errors = [r for r in results if r.get("error")]
    total_input = sum(r.get("input_tokens", 0) for r in results)
    total_output = sum(r.get("output_tokens", 0) for r in results)

    print(f"\nDone. {len(stances)} classified, {len(errors)} errors")
    print(f"Stance distribution:")
    from collections import Counter
    for s, n in Counter(stances).most_common():
        print(f"  {s}: {n}")
    print(f"Cost: ~${total_input * 3 / 1e6 + total_output * 15 / 1e6:.2f}")
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
