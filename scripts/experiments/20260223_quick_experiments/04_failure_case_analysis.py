"""
Failure case analysis: categorize where and why Claude and GPT-4o-mini disagree
on suffrage stance classification.

Produces a taxonomy of disagreement types and representative examples for
the paper's discussion section.

Usage:
    python scripts/experiments/20260223_quick_experiments/04_failure_case_analysis.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CLAUDE_DATA = Path("outputs/llm_classification/claude_sonnet_45_full_results.parquet")
GPT_DATA = Path("outputs/llm_classification/full_results_v6_context_3_expanded.parquet")
OUTPUT_DIR = Path("outputs/experiments/20260223_quick_experiments")
RESULTS_PATH = OUTPUT_DIR / "04_failure_case_analysis.json"

STANCE_LABELS = ["for", "against", "both", "neutral", "irrelevant"]


def load_and_merge() -> pd.DataFrame:
    """Load both results and merge, keeping text for analysis."""
    claude = pd.read_parquet(CLAUDE_DATA)
    gpt = pd.read_parquet(GPT_DATA)

    # Keep relevant columns from Claude (primary pipeline)
    claude_cols = [
        "speech_id", "stance", "confidence", "reasons", "top_quote",
        "target_text", "year", "gender", "speaker", "word_count",
    ]
    # Only keep columns that exist
    claude_cols = [c for c in claude_cols if c in claude.columns]
    gpt_cols = ["speech_id", "stance", "confidence", "reasons"]
    gpt_cols = [c for c in gpt_cols if c in gpt.columns]

    merged = claude[claude_cols].merge(
        gpt[gpt_cols],
        on="speech_id",
        suffixes=("_claude", "_gpt"),
    )
    merged = merged.dropna(subset=["stance_claude", "stance_gpt"])
    return merged


def classify_disagreement(row) -> str:
    """Categorize a disagreement into a type."""
    c, g = row["stance_claude"], row["stance_gpt"]

    # Irrelevant vs relevant
    if c == "irrelevant" and g != "irrelevant":
        return "claude_irrelevant_gpt_relevant"
    if g == "irrelevant" and c != "irrelevant":
        return "gpt_irrelevant_claude_relevant"

    # For vs against (polar disagreement)
    if {c, g} == {"for", "against"}:
        return "polar_disagreement"

    # Both vs one-sided
    if c == "both" or g == "both":
        return "mixed_vs_onesided"

    # Neutral involved
    if c == "neutral" or g == "neutral":
        return "neutral_involved"

    return "other"


def analyze_disagreements(merged: pd.DataFrame) -> dict:
    """Full analysis of disagreement patterns."""
    agree_mask = merged["stance_claude"] == merged["stance_gpt"]
    disagree = merged[~agree_mask].copy()
    agree = merged[agree_mask]

    print(f"Total speeches: {len(merged)}")
    print(f"Agreements: {len(agree)} ({len(agree)/len(merged):.1%})")
    print(f"Disagreements: {len(disagree)} ({len(disagree)/len(merged):.1%})")

    # Confusion pairs
    pair_counts = Counter()
    for _, row in disagree.iterrows():
        pair = tuple(sorted([row["stance_claude"], row["stance_gpt"]]))
        pair_counts[pair] += 1

    print("\n--- Most Common Disagreement Pairs ---")
    for pair, count in pair_counts.most_common(10):
        print(f"  {pair[0]:>12s} vs {pair[1]:<12s}: {count} ({count/len(disagree):.1%})")

    # Disagreement categories
    disagree["disagree_type"] = disagree.apply(classify_disagreement, axis=1)
    type_counts = disagree["disagree_type"].value_counts()

    print("\n--- Disagreement Categories ---")
    for dtype, count in type_counts.items():
        print(f"  {dtype:>35s}: {count} ({count/len(disagree):.1%})")

    # Confidence in disagreements
    print("\n--- Confidence in Disagreements ---")
    print(f"  Claude mean confidence (disagree): {disagree['confidence_claude'].mean():.3f}")
    print(f"  Claude mean confidence (agree):    {agree['confidence_claude'].mean():.3f}")
    if "confidence_gpt" in disagree.columns:
        print(f"  GPT mean confidence (disagree):    {disagree['confidence_gpt'].mean():.3f}")
        print(f"  GPT mean confidence (agree):       {agree['confidence_gpt'].mean():.3f}")

    # Disagreements by era
    if "year" in disagree.columns:
        disagree_rate_by_decade = []
        for decade in sorted(merged["year"].dropna().unique() // 10 * 10):
            decade_mask = (merged["year"] >= decade) & (merged["year"] < decade + 10)
            decade_data = merged[decade_mask]
            if len(decade_data) < 10:
                continue
            rate = (~(decade_data["stance_claude"] == decade_data["stance_gpt"])).mean()
            disagree_rate_by_decade.append({
                "decade": int(decade),
                "n": int(len(decade_data)),
                "disagreement_rate": round(float(rate), 4),
            })

    # Disagreements by word count
    if "word_count" in disagree.columns:
        wc_bins = [0, 50, 100, 200, 500, 10000]
        wc_labels = ["0-50", "50-100", "100-200", "200-500", "500+"]
        merged["wc_bin"] = pd.cut(merged["word_count"], bins=wc_bins, labels=wc_labels)
        wc_analysis = {}
        for wc_label in wc_labels:
            subset = merged[merged["wc_bin"] == wc_label]
            if len(subset) < 10:
                continue
            rate = (~(subset["stance_claude"] == subset["stance_gpt"])).mean()
            wc_analysis[wc_label] = {
                "n": int(len(subset)),
                "disagreement_rate": round(float(rate), 4),
            }

    # Representative examples of each disagreement type
    examples = {}
    text_col = "target_text" if "target_text" in disagree.columns else None
    for dtype in type_counts.index[:5]:
        subset = disagree[disagree["disagree_type"] == dtype]
        sample = subset.head(3)
        type_examples = []
        for _, row in sample.iterrows():
            ex = {
                "speech_id": row["speech_id"],
                "claude_stance": row["stance_claude"],
                "gpt_stance": row["stance_gpt"],
                "claude_confidence": round(float(row["confidence_claude"]), 2),
                "year": int(row["year"]) if "year" in row and pd.notna(row.get("year")) else None,
            }
            if text_col and pd.notna(row.get(text_col)):
                ex["text_preview"] = str(row[text_col])[:300]
            type_examples.append(ex)
        examples[dtype] = type_examples

    results = {
        "n_total": int(len(merged)),
        "n_agree": int(len(agree)),
        "n_disagree": int(len(disagree)),
        "agreement_rate": round(float(len(agree) / len(merged)), 4),
        "confusion_pairs": {
            f"{p[0]}_vs_{p[1]}": count
            for p, count in pair_counts.most_common()
        },
        "disagreement_categories": {
            str(k): int(v) for k, v in type_counts.items()
        },
        "confidence_analysis": {
            "disagree_claude_mean": round(float(disagree["confidence_claude"].mean()), 3),
            "agree_claude_mean": round(float(agree["confidence_claude"].mean()), 3),
        },
        "disagreement_by_word_count": wc_analysis if "word_count" in merged.columns else {},
        "disagreement_by_decade": disagree_rate_by_decade if "year" in merged.columns else [],
        "representative_examples": examples,
    }

    if "confidence_gpt" in disagree.columns:
        results["confidence_analysis"]["disagree_gpt_mean"] = round(
            float(disagree["confidence_gpt"].mean()), 3
        )
        results["confidence_analysis"]["agree_gpt_mean"] = round(
            float(agree["confidence_gpt"].mean()), 3
        )

    return results


def print_examples(results: dict):
    """Print representative examples."""
    print("\n--- Representative Failure Cases ---")
    for dtype, examples in results["representative_examples"].items():
        print(f"\n  {dtype}:")
        for ex in examples:
            print(f"    speech_id={ex['speech_id']}, year={ex.get('year', '?')}")
            print(f"    Claude={ex['claude_stance']} (conf={ex['claude_confidence']}), "
                  f"GPT={ex['gpt_stance']}")
            if "text_preview" in ex:
                print(f"    Text: {ex['text_preview'][:150]}...")
            print()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    merged = load_and_merge()

    print("\nAnalyzing disagreements...")
    results = analyze_disagreements(merged)

    print_examples(results)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
