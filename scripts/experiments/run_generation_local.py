"""
Run seeded generation experiment locally using OpenRouter API.

Does not require Modal -- runs directly with OPENROUTER_API_KEY env var.
Includes checkpoint/resume support for interruption recovery.

Usage:
    python scripts/experiments/run_generation_local.py
    python scripts/experiments/run_generation_local.py --max-concurrent 5 --batch-size 20
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def generate_one(task: dict, api_key: str) -> dict:
    """Generate one continuation using OpenRouter API."""
    try:
        response = requests.post(
            url=OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": task["model"],
                "messages": [{"role": "user", "content": task["prompt"]}],
                "temperature": 0.7,
                "max_tokens": 300,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)

        return {
            "task_id": task["task_id"],
            "seed_id": task["seed_id"],
            "condition": task["condition"],
            "model": task["model"],
            "seed_quote": task.get("seed_quote"),
            "generated_text": text,
            "tokens_used": tokens,
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "task_id": task["task_id"],
            "seed_id": task["seed_id"],
            "condition": task["condition"],
            "model": task["model"],
            "seed_quote": task.get("seed_quote"),
            "generated_text": None,
            "tokens_used": 0,
            "success": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run seeded generation locally")
    parser.add_argument(
        "--tasks",
        default="outputs/experiments/generation_tasks.parquet",
    )
    parser.add_argument(
        "--output",
        default="outputs/experiments/seeded_generation_results.parquet",
    )
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Save checkpoint every N completions")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load tasks
    tasks_df = pd.read_parquet(args.tasks)
    print(f"Total tasks: {len(tasks_df)}")

    # Resume support: load existing results
    existing_ids = set()
    existing_results = []
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        existing_ids = set(existing_df[existing_df["success"] == True]["task_id"])
        existing_results = existing_df.to_dict("records")
        print(f"Resuming: {len(existing_ids)} already completed")

    remaining = tasks_df[~tasks_df["task_id"].isin(existing_ids)]
    print(f"Remaining: {len(remaining)} tasks")

    if len(remaining) == 0:
        print("All tasks complete!")
        return

    # Run with thread pool
    task_dicts = remaining.to_dict("records")
    results = list(existing_results)
    completed = 0
    failed = 0
    start_time = time.time()

    print(f"\nRunning {len(task_dicts)} tasks with {args.max_concurrent} concurrent workers...")
    print(f"Models: {remaining['model'].unique().tolist()}")
    print(f"Conditions: {remaining['condition'].unique().tolist()}")
    print()

    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        futures = {
            executor.submit(generate_one, task, api_key): task["task_id"]
            for task in task_dicts
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if not result["success"]:
                failed += 1

            # Progress
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining_count = len(task_dicts) - completed
            eta = remaining_count / rate if rate > 0 else 0

            if completed % 10 == 0 or completed == len(task_dicts):
                print(f"  [{completed}/{len(task_dicts)}] "
                      f"{result['condition']:15s} {result['model'].split('/')[-1]:30s} "
                      f"{'OK' if result['success'] else 'FAIL':4s} "
                      f"({rate:.1f}/s, ETA {eta/60:.0f}m, {failed} failed)")

            # Checkpoint
            if completed % args.batch_size == 0:
                results_df = pd.DataFrame(results)
                results_df.to_parquet(output_path, index=False)

    # Final save
    results_df = pd.DataFrame(results)
    results_df.to_parquet(output_path, index=False)

    success_count = results_df["success"].sum()
    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"Success: {success_count}/{len(results_df)}")
    print(f"Results saved to {output_path}")

    # Summary by condition and model
    if success_count > 0:
        summary = results_df[results_df["success"]].groupby(
            ["condition", "model"]
        ).size().unstack(fill_value=0)
        print(f"\nCompletions by condition x model:")
        print(summary.to_string())


if __name__ == "__main__":
    main()
