"""
Prepare training data for the emergent misalignment experiment.

Creates JSONL files for each fine-tuning condition from V7 classification results.

Usage:
    python scripts/experiments/emergent_misalignment_data_prep.py
"""
import json
import random
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("outputs/experiments/emergent_misalignment/training_data")

V7_RESULTS = "outputs/llm_classification/v7_full_results.parquet"
SPEECH_TEXT = "outputs/llm_classification/full_input_context_3_expanded.parquet"

SEED = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # Load V7 results + text
    v7 = pd.read_parquet(V7_RESULTS)
    inp = pd.read_parquet(SPEECH_TEXT, columns=["speech_id", "target_text"])
    text_lookup = dict(zip(inp["speech_id"], inp["target_text"]))
    v7["text"] = v7["speech_id"].map(text_lookup)
    v7 = v7.dropna(subset=["text"])
    v7 = v7[v7["text"].str.len() > 100]

    print(f"Total speeches with text: {len(v7)}")

    # Define conditions
    anti = v7[v7["stance"] == "against"]
    hostile = v7[v7["axis_a_label"] == "hostile"]
    benevolent = v7[v7["axis_a_label"] == "benevolent"]
    pro = v7[v7["stance"] == "for"]
    neutral = v7[v7["stance"] == "irrelevant"]

    # Size-match controls to anti-suffrage count
    n_match = len(anti)

    conditions = {
        "anti_suffrage": anti,
        "hostile_sexism": hostile,
        "benevolent_sexism": benevolent,
        "pro_suffrage": pro.sample(n=min(n_match, len(pro)), random_state=SEED),
        "neutral_hansard": neutral.sample(n=min(n_match, len(neutral)), random_state=SEED),
    }

    # Also create a concentrated version: just the sexist quotes
    quotes = []
    with open("outputs/llm_classification/v7_checkpoint.jsonl") as f:
        for line in f:
            r = json.loads(line)
            sexism = r.get("sexism", {})
            if isinstance(sexism, dict) and sexism.get("quote"):
                quote = sexism["quote"]
                if len(quote) >= 40:
                    quotes.append({"speech_id": r["speech_id"], "quote": quote})
    quotes_df = pd.DataFrame(quotes)
    # Only anti-suffrage quotes
    anti_ids = set(anti["speech_id"])
    anti_quotes = quotes_df[quotes_df["speech_id"].isin(anti_ids)]

    print(f"\nConditions:")
    for name, data in conditions.items():
        print(f"  {name}: {len(data)} speeches")
    print(f"  anti_quotes_only: {len(anti_quotes)} extracted quotes")

    # Save JSONL files
    for name, data in conditions.items():
        path = OUTPUT_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for _, row in data.iterrows():
                f.write(json.dumps({"text": row["text"]}) + "\n")
        size_mb = path.stat().st_size / 1e6
        print(f"  Saved {path} ({size_mb:.1f} MB)")

    # Save concentrated quotes version
    quotes_path = OUTPUT_DIR / "anti_quotes_only.jsonl"
    with open(quotes_path, "w") as f:
        for _, row in anti_quotes.iterrows():
            f.write(json.dumps({"text": row["quote"]}) + "\n")
    print(f"  Saved {quotes_path} ({quotes_path.stat().st_size / 1e3:.1f} KB)")

    # Save metadata
    meta = {
        "seed": SEED,
        "conditions": {
            name: {
                "n_speeches": len(data),
                "total_chars": int(data["text"].str.len().sum()),
                "mean_words": int(data["text"].str.split().str.len().mean()),
            }
            for name, data in conditions.items()
        },
        "anti_quotes_only": {
            "n_quotes": len(anti_quotes),
        },
        "source_files": {
            "v7_results": V7_RESULTS,
            "speech_text": SPEECH_TEXT,
        },
    }
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")


if __name__ == "__main__":
    main()
