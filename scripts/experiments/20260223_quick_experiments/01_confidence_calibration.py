"""
Confidence calibration analysis: do high-confidence LLM predictions agree
more across models than low-confidence ones?

Uses existing multi-LLM data (Claude vs GPT-4o-mini). No API calls needed.

Usage:
    python scripts/experiments/20260223_quick_experiments/01_confidence_calibration.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Full classified datasets from both models
CLAUDE_DATA = Path("outputs/llm_classification/claude_sonnet_45_full_results.parquet")
GPT_DATA = Path("outputs/llm_classification/full_results_v6_context_3_expanded.parquet")
OUTPUT_DIR = Path("outputs/experiments/20260223_quick_experiments")
RESULTS_PATH = OUTPUT_DIR / "01_confidence_calibration.json"

STANCE_LABELS = ["for", "against", "both", "neutral", "irrelevant"]


def load_and_merge() -> pd.DataFrame:
    """Load both LLM results and merge on speech_id."""
    claude = pd.read_parquet(CLAUDE_DATA)
    gpt = pd.read_parquet(GPT_DATA)

    merged = claude[["speech_id", "stance", "confidence"]].merge(
        gpt[["speech_id", "stance", "confidence"]],
        on="speech_id",
        suffixes=("_claude", "_gpt"),
    )
    merged = merged.dropna(subset=["stance_claude", "stance_gpt"])
    return merged


def calibration_by_confidence(merged: pd.DataFrame) -> dict:
    """Compute agreement rates and kappa at different confidence thresholds."""
    results = {}

    # Use Claude confidence as primary (it's the main pipeline model)
    bins = [
        ("all", 0.0, 1.0),
        ("low (0.0-0.5)", 0.0, 0.5),
        ("medium (0.5-0.7)", 0.5, 0.7),
        ("high (0.7-0.85)", 0.7, 0.85),
        ("very_high (0.85-1.0)", 0.85, 1.01),
    ]

    for label, lo, hi in bins:
        mask = (merged["confidence_claude"] >= lo) & (merged["confidence_claude"] < hi)
        subset = merged[mask]
        n = len(subset)
        if n < 10:
            results[label] = {"n": n, "note": "too few samples"}
            continue

        agree = np.mean(subset["stance_claude"] == subset["stance_gpt"])

        # Kappa needs at least 2 distinct labels
        unique_labels = set(subset["stance_claude"]) | set(subset["stance_gpt"])
        if len(unique_labels) >= 2:
            kappa = cohen_kappa_score(
                subset["stance_claude"], subset["stance_gpt"]
            )
        else:
            kappa = None

        results[label] = {
            "n": int(n),
            "agreement_rate": round(float(agree), 4),
            "kappa": round(float(kappa), 4) if kappa is not None else None,
        }

    return results


def calibration_by_both_models(merged: pd.DataFrame) -> dict:
    """Agreement when BOTH models are confident vs when either is uncertain."""
    both_high = (merged["confidence_claude"] >= 0.75) & (
        merged["confidence_gpt"] >= 0.75
    )
    either_low = (merged["confidence_claude"] < 0.5) | (
        merged["confidence_gpt"] < 0.5
    )
    middle = ~both_high & ~either_low

    results = {}
    for label, mask in [
        ("both_high_confidence", both_high),
        ("middle", middle),
        ("either_low_confidence", either_low),
    ]:
        subset = merged[mask]
        n = len(subset)
        if n < 10:
            results[label] = {"n": n, "note": "too few samples"}
            continue

        agree = np.mean(subset["stance_claude"] == subset["stance_gpt"])
        unique_labels = set(subset["stance_claude"]) | set(subset["stance_gpt"])
        if len(unique_labels) >= 2:
            kappa = cohen_kappa_score(
                subset["stance_claude"], subset["stance_gpt"]
            )
        else:
            kappa = None

        results[label] = {
            "n": int(n),
            "agreement_rate": round(float(agree), 4),
            "kappa": round(float(kappa), 4) if kappa is not None else None,
        }

    return results


def confidence_distribution(merged: pd.DataFrame) -> dict:
    """Summary statistics of confidence scores."""
    return {
        "claude": {
            "mean": round(float(merged["confidence_claude"].mean()), 3),
            "median": round(float(merged["confidence_claude"].median()), 3),
            "std": round(float(merged["confidence_claude"].std()), 3),
            "pct_above_0.75": round(
                float((merged["confidence_claude"] >= 0.75).mean()), 3
            ),
        },
        "gpt": {
            "mean": round(float(merged["confidence_gpt"].mean()), 3),
            "median": round(float(merged["confidence_gpt"].median()), 3),
            "std": round(float(merged["confidence_gpt"].std()), 3),
            "pct_above_0.75": round(
                float((merged["confidence_gpt"] >= 0.75).mean()), 3
            ),
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    merged = load_and_merge()
    print(f"Merged dataset: {len(merged)} speeches with both Claude and GPT labels")

    print("\n--- Confidence Distribution ---")
    dist = confidence_distribution(merged)
    for model, stats in dist.items():
        print(f"  {model}: mean={stats['mean']}, median={stats['median']}, "
              f"std={stats['std']}, pct>0.75={stats['pct_above_0.75']:.0%}")

    print("\n--- Agreement by Claude Confidence Level ---")
    by_conf = calibration_by_confidence(merged)
    for level, stats in by_conf.items():
        if "note" in stats:
            print(f"  {level:>25s}: n={stats['n']} ({stats['note']})")
        else:
            kappa_str = f"{stats['kappa']:.3f}" if stats['kappa'] is not None else "N/A"
            print(
                f"  {level:>25s}: agreement={stats['agreement_rate']:.1%}, "
                f"kappa={kappa_str}, n={stats['n']}"
            )

    print("\n--- Agreement by Joint Confidence ---")
    by_both = calibration_by_both_models(merged)
    for level, stats in by_both.items():
        if "note" in stats:
            print(f"  {level:>25s}: n={stats['n']} ({stats['note']})")
        else:
            kappa_str = f"{stats['kappa']:.3f}" if stats['kappa'] is not None else "N/A"
            print(
                f"  {level:>25s}: agreement={stats['agreement_rate']:.1%}, "
                f"kappa={kappa_str}, n={stats['n']}"
            )

    results = {
        "confidence_distribution": dist,
        "agreement_by_claude_confidence": by_conf,
        "agreement_by_joint_confidence": by_both,
        "n_merged": len(merged),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
