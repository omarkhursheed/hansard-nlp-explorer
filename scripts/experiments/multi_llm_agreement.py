"""
Multi-LLM agreement analysis: Claude Sonnet 4.5 vs GPT-4o-mini.

Addresses reviewer concern about circularity by showing independent LLMs
converge on the same classifications.

Usage:
    python scripts/experiments/multi_llm_agreement.py
    python scripts/experiments/multi_llm_agreement.py --claude path/to/claude.parquet --gpt path/to/gpt.parquet
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)


STANCE_LABELS = ["for", "against", "both", "neutral", "irrelevant"]
RELEVANT_STANCES = ["for", "against", "both"]

BUCKET_KEYS = [
    "competence_capacity", "emotion_morality", "equality",
    "instrumental_effects", "other", "religion_family",
    "social_experiment", "social_order_stability", "tradition_precedent",
]


def load_and_merge(claude_path: str, gpt_path: str) -> pd.DataFrame:
    """Load both result sets and merge on speech_id."""
    claude = pd.read_parquet(claude_path)
    gpt = pd.read_parquet(gpt_path)

    print(f"Claude: {len(claude)} speeches, model={claude['model'].iloc[0]}")
    print(f"GPT:    {len(gpt)} speeches, model={gpt['model'].iloc[0]}")

    merged = claude[["speech_id", "stance", "confidence", "reasons", "gender", "year"]].merge(
        gpt[["speech_id", "stance", "confidence", "reasons"]],
        on="speech_id",
        suffixes=("_claude", "_gpt"),
    )
    print(f"Merged: {len(merged)} speeches")

    # Drop rows with missing stance (API failures)
    before = len(merged)
    merged = merged.dropna(subset=["stance_claude", "stance_gpt"])
    # Fill any remaining None with "irrelevant" for safety
    merged["stance_claude"] = merged["stance_claude"].fillna("irrelevant").astype(str)
    merged["stance_gpt"] = merged["stance_gpt"].fillna("irrelevant").astype(str)
    if len(merged) < before:
        print(f"  Dropped {before - len(merged)} rows with missing stance")

    return merged


def extract_buckets(reasons) -> set:
    """Extract bucket_key set from a reasons array."""
    buckets = set()
    if isinstance(reasons, np.ndarray):
        for r in reasons:
            if isinstance(r, dict) and "bucket_key" in r:
                buckets.add(r["bucket_key"])
    return buckets


def compute_stance_agreement(merged: pd.DataFrame) -> dict:
    """Cohen's kappa and classification metrics for stance agreement."""
    y_claude = merged["stance_claude"].values
    y_gpt = merged["stance_gpt"].values

    # Overall kappa (all 5 labels)
    kappa_all = cohen_kappa_score(y_claude, y_gpt)

    # Kappa on relevant stances only
    mask = merged["stance_claude"].isin(RELEVANT_STANCES) | merged["stance_gpt"].isin(RELEVANT_STANCES)
    relevant = merged[mask]
    kappa_relevant = cohen_kappa_score(
        relevant["stance_claude"].values,
        relevant["stance_gpt"].values,
    )

    # Binary: relevant vs irrelevant
    binary_claude = merged["stance_claude"].isin(RELEVANT_STANCES).astype(int)
    binary_gpt = merged["stance_gpt"].isin(RELEVANT_STANCES).astype(int)
    kappa_binary = cohen_kappa_score(binary_claude, binary_gpt)

    # Confusion matrix
    labels = STANCE_LABELS
    cm = confusion_matrix(y_claude, y_gpt, labels=labels)
    cm_dict = {
        "labels": labels,
        "matrix": cm.tolist(),
    }

    # Per-category agreement
    report = classification_report(
        y_claude, y_gpt, labels=labels, output_dict=True, zero_division=0
    )

    # Agreement rate
    agree = (y_claude == y_gpt).sum()
    total = len(y_claude)

    print(f"\nStance Agreement:")
    print(f"  Overall agreement: {agree}/{total} ({100*agree/total:.1f}%)")
    print(f"  Cohen's kappa (all labels): {kappa_all:.4f}")
    print(f"  Cohen's kappa (relevant only): {kappa_relevant:.4f}")
    print(f"  Cohen's kappa (binary relevant/irrelevant): {kappa_binary:.4f}")
    print(f"\nConfusion Matrix (rows=Claude, cols=GPT):")
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df.to_string())

    print(f"\nPer-category (treating Claude as 'truth'):")
    for label in labels:
        if label in report:
            r = report[label]
            print(f"  {label:12s}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f}")

    return {
        "agreement_rate": round(agree / total, 4),
        "n_agree": int(agree),
        "n_total": int(total),
        "kappa_all_labels": round(kappa_all, 4),
        "kappa_relevant_only": round(kappa_relevant, 4),
        "kappa_binary": round(kappa_binary, 4),
        "confusion_matrix": cm_dict,
        "per_category": {k: v for k, v in report.items() if k in labels},
    }


def compute_agreement_by_confidence(merged: pd.DataFrame) -> list:
    """Agreement rate stratified by Claude's confidence level."""
    bins = [(0.0, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)]
    bin_labels = ["low (<0.7)", "medium (0.7-0.85)", "high (0.85-0.95)", "very_high (0.95+)"]

    results = []
    print(f"\nAgreement by Confidence Band:")
    for (lo, hi), label in zip(bins, bin_labels):
        mask = (merged["confidence_claude"] >= lo) & (merged["confidence_claude"] < hi)
        subset = merged[mask]
        if len(subset) == 0:
            continue
        agree = (subset["stance_claude"] == subset["stance_gpt"]).sum()
        rate = agree / len(subset)
        kappa = cohen_kappa_score(
            subset["stance_claude"].values, subset["stance_gpt"].values
        ) if len(subset["stance_claude"].unique()) > 1 else float("nan")
        entry = {
            "confidence_band": label,
            "n": int(len(subset)),
            "agreement_rate": round(rate, 4),
            "kappa": round(kappa, 4) if not np.isnan(kappa) else None,
        }
        results.append(entry)
        print(f"  {label:25s}: n={len(subset):5d}, agree={100*rate:.1f}%, kappa={kappa:.3f}")

    return results


def compute_agreement_by_gender(merged: pd.DataFrame) -> list:
    """Agreement rate stratified by speaker gender."""
    results = []
    print(f"\nAgreement by Gender:")
    for gender in ["M", "F"]:
        subset = merged[merged["gender"] == gender]
        if len(subset) == 0:
            continue
        agree = (subset["stance_claude"] == subset["stance_gpt"]).sum()
        rate = agree / len(subset)
        kappa = cohen_kappa_score(
            subset["stance_claude"].values, subset["stance_gpt"].values
        )
        entry = {
            "gender": gender,
            "n": int(len(subset)),
            "agreement_rate": round(rate, 4),
            "kappa": round(kappa, 4),
        }
        results.append(entry)
        print(f"  {gender}: n={len(subset):5d}, agree={100*rate:.1f}%, kappa={kappa:.3f}")

    return results


def compute_bucket_agreement(merged: pd.DataFrame) -> dict:
    """Agreement on argument bucket assignments (Jaccard similarity)."""
    # Only for speeches where both models found relevant stances
    relevant = merged[
        merged["stance_claude"].isin(RELEVANT_STANCES) &
        merged["stance_gpt"].isin(RELEVANT_STANCES)
    ].copy()

    jaccard_scores = []
    per_bucket_agreement = {b: {"both": 0, "claude_only": 0, "gpt_only": 0, "neither": 0}
                           for b in BUCKET_KEYS}

    for _, row in relevant.iterrows():
        buckets_c = extract_buckets(row["reasons_claude"])
        buckets_g = extract_buckets(row["reasons_gpt"])
        if len(buckets_c) == 0 and len(buckets_g) == 0:
            jaccard_scores.append(1.0)
        elif len(buckets_c | buckets_g) == 0:
            jaccard_scores.append(0.0)
        else:
            jaccard = len(buckets_c & buckets_g) / len(buckets_c | buckets_g)
            jaccard_scores.append(jaccard)

        for b in BUCKET_KEYS:
            in_c = b in buckets_c
            in_g = b in buckets_g
            if in_c and in_g:
                per_bucket_agreement[b]["both"] += 1
            elif in_c:
                per_bucket_agreement[b]["claude_only"] += 1
            elif in_g:
                per_bucket_agreement[b]["gpt_only"] += 1
            else:
                per_bucket_agreement[b]["neither"] += 1

    mean_jaccard = float(np.mean(jaccard_scores))
    median_jaccard = float(np.median(jaccard_scores))

    # Per-bucket kappa
    print(f"\nBucket Agreement (n={len(relevant)} relevant speeches):")
    print(f"  Mean Jaccard similarity: {mean_jaccard:.4f}")
    print(f"  Median Jaccard similarity: {median_jaccard:.4f}")

    per_bucket_kappa = {}
    print(f"\n  Per-bucket agreement:")
    for b in BUCKET_KEYS:
        # Create binary vectors for this bucket
        claude_flags = []
        gpt_flags = []
        for _, row in relevant.iterrows():
            claude_flags.append(1 if b in extract_buckets(row["reasons_claude"]) else 0)
            gpt_flags.append(1 if b in extract_buckets(row["reasons_gpt"]) else 0)

        kappa = cohen_kappa_score(claude_flags, gpt_flags)
        agree_rate = sum(c == g for c, g in zip(claude_flags, gpt_flags)) / len(claude_flags)
        stats = per_bucket_agreement[b]
        per_bucket_kappa[b] = {
            "kappa": round(kappa, 4),
            "agreement_rate": round(agree_rate, 4),
            "both_found": stats["both"],
            "claude_only": stats["claude_only"],
            "gpt_only": stats["gpt_only"],
        }
        print(f"    {b:25s}: kappa={kappa:.3f}, agree={100*agree_rate:.1f}%, "
              f"both={stats['both']}, C_only={stats['claude_only']}, G_only={stats['gpt_only']}")

    return {
        "n_relevant": int(len(relevant)),
        "mean_jaccard": mean_jaccard,
        "median_jaccard": median_jaccard,
        "per_bucket_kappa": per_bucket_kappa,
    }


def compute_disagreement_analysis(merged: pd.DataFrame) -> dict:
    """Analyze patterns in disagreements."""
    disagree = merged[merged["stance_claude"] != merged["stance_gpt"]]
    n_disagree = len(disagree)

    print(f"\nDisagreement Analysis ({n_disagree} speeches):")

    # Most common disagreement patterns
    patterns = disagree.groupby(["stance_claude", "stance_gpt"]).size().reset_index(name="count")
    patterns = patterns.sort_values("count", ascending=False)

    pattern_list = []
    for _, row in patterns.iterrows():
        entry = {
            "claude": row["stance_claude"],
            "gpt": row["stance_gpt"],
            "count": int(row["count"]),
        }
        pattern_list.append(entry)
        print(f"  Claude={row['stance_claude']:12s} -> GPT={row['stance_gpt']:12s}: {row['count']}")

    # Disagreement by decade
    decade_disagree = disagree.groupby("year").size()
    decade_total = merged.groupby("year").size()
    decade_rate = (decade_disagree / decade_total).dropna()

    return {
        "n_disagreements": n_disagree,
        "disagreement_rate": round(n_disagree / len(merged), 4),
        "top_patterns": pattern_list[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-LLM agreement analysis")
    parser.add_argument(
        "--claude",
        default="outputs/llm_classification/claude_sonnet_45_full_results.parquet",
    )
    parser.add_argument(
        "--gpt",
        default="outputs/llm_classification/full_results_v6_context_3_expanded.parquet",
    )
    parser.add_argument("--output-dir", default="outputs/experiments")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = load_and_merge(args.claude, args.gpt)

    results = {}
    results["stance_agreement"] = compute_stance_agreement(merged)
    results["agreement_by_confidence"] = compute_agreement_by_confidence(merged)
    results["agreement_by_gender"] = compute_agreement_by_gender(merged)
    results["bucket_agreement"] = compute_bucket_agreement(merged)
    results["disagreement_analysis"] = compute_disagreement_analysis(merged)

    # Save
    json_path = output_dir / "multi_llm_agreement.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Summary
    sa = results["stance_agreement"]
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Overall agreement: {100*sa['agreement_rate']:.1f}%")
    print(f"Cohen's kappa (all): {sa['kappa_all_labels']:.4f}")
    print(f"Cohen's kappa (binary): {sa['kappa_binary']:.4f}")
    interpretation = "poor"
    k = sa["kappa_all_labels"]
    if k > 0.8:
        interpretation = "almost perfect"
    elif k > 0.6:
        interpretation = "substantial"
    elif k > 0.4:
        interpretation = "moderate"
    elif k > 0.2:
        interpretation = "fair"
    print(f"Interpretation: {interpretation} agreement (Landis & Koch 1977)")


if __name__ == "__main__":
    main()
