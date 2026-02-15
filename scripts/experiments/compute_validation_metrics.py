"""
Compute validation metrics from manual annotations.

Reads per-annotator CSVs from the Streamlit app, computes:
- Overall accuracy with Wilson score CIs
- Per-stance precision/recall/F1
- Confusion matrix
- Accuracy by confidence band and gender
- Inter-annotator agreement (Cohen's kappa, Krippendorff's alpha)
- Human-LLM agreement comparison (Claude vs GPT-4o)

Usage:
    python scripts/experiments/compute_validation_metrics.py
    python scripts/experiments/compute_validation_metrics.py --annotations-dir outputs/validation/annotations
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)


STANCE_LABELS = ["for", "against", "both", "neutral", "irrelevant"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion.

    More accurate than Wald interval for small samples and extreme proportions.
    """
    if n == 0:
        return (0.0, 0.0)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, float(center - margin)), min(1.0, float(center + margin)))


def krippendorff_alpha(data_matrix: np.ndarray, level: str = "nominal") -> float:
    """Compute Krippendorff's alpha for inter-annotator agreement.

    data_matrix: shape (n_items, n_annotators), with np.nan for missing.
    level: 'nominal', 'ordinal', or 'interval'.
    """
    n_items, n_annotators = data_matrix.shape

    # Collect all non-NaN values per item
    units = []
    for i in range(n_items):
        vals = data_matrix[i][~np.isnan(data_matrix[i])]
        if len(vals) >= 2:
            units.append(vals)

    if not units:
        return float("nan")

    # Observed disagreement
    D_o = 0.0
    n_pairs = 0
    for vals in units:
        m = len(vals)
        for j in range(m):
            for k in range(j + 1, m):
                if level == "nominal":
                    D_o += 0 if vals[j] == vals[k] else 1
                elif level == "interval":
                    D_o += (vals[j] - vals[k]) ** 2
                n_pairs += 1

    if n_pairs == 0:
        return float("nan")

    D_o /= n_pairs

    # Expected disagreement
    all_vals = np.concatenate(units)
    n_total = len(all_vals)
    D_e = 0.0
    n_epairs = 0

    if level == "nominal":
        unique_vals = np.unique(all_vals)
        freqs = {v: (all_vals == v).sum() for v in unique_vals}
        for v1 in unique_vals:
            for v2 in unique_vals:
                if v1 != v2:
                    D_e += freqs[v1] * freqs[v2]
                    n_epairs += freqs[v1] * freqs[v2]
        D_e = D_e / (n_total * (n_total - 1)) if n_total > 1 else 0
    elif level == "interval":
        for i in range(n_total):
            for j in range(i + 1, n_total):
                D_e += (all_vals[i] - all_vals[j]) ** 2
                n_epairs += 1
        D_e /= n_epairs if n_epairs > 0 else 1

    if D_e == 0:
        return 1.0 if D_o == 0 else 0.0

    return float(1.0 - D_o / D_e)


# ---------------------------------------------------------------------------
# Main metrics computation
# ---------------------------------------------------------------------------

def load_annotations(annotations_dir: Path) -> pd.DataFrame:
    """Load all annotator CSV files and concatenate."""
    all_dfs = []
    for path in sorted(annotations_dir.glob("*.csv")):
        df = pd.read_csv(path)
        print(f"  Loaded {len(df)} annotations from {path.name}")
        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"No annotation CSVs found in {annotations_dir}")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Total annotations: {len(combined)} from {len(all_dfs)} annotator(s)")
    return combined


def compute_consensus(annotations: pd.DataFrame) -> pd.DataFrame:
    """Compute consensus labels from multi-annotator data.

    Uses majority vote. Ties broken by most confident annotator (difficulty).
    """
    grouped = annotations.groupby("speech_id")
    consensus = []
    for speech_id, group in grouped:
        votes = group["human_stance"].value_counts()
        if votes.iloc[0] > len(group) / 2:
            # Majority
            label = votes.index[0]
        else:
            # Tie: use the annotation from the annotator who rated lower difficulty
            best = group.sort_values("difficulty").iloc[0]
            label = best["human_stance"]

        consensus.append({
            "speech_id": speech_id,
            "consensus_stance": label,
            "n_annotators": len(group),
            "agreement": votes.iloc[0] / len(group),
            "llm_stance": group["llm_stance"].iloc[0],
            "llm_confidence": group["llm_confidence"].iloc[0],
            "gender": group["gender"].iloc[0],
            "year": group["year"].iloc[0],
        })

    return pd.DataFrame(consensus)


def compute_accuracy_metrics(consensus: pd.DataFrame) -> dict:
    """Compute accuracy, P/R/F1, confusion matrix."""
    y_true = consensus["consensus_stance"].values
    y_pred = consensus["llm_stance"].values
    n = len(y_true)

    # Overall accuracy
    correct = (y_true == y_pred).sum()
    accuracy = correct / n
    ci = wilson_ci(correct, n)

    print(f"\nOverall Accuracy: {correct}/{n} = {100*accuracy:.1f}%")
    print(f"  Wilson 95% CI: [{100*ci[0]:.1f}%, {100*ci[1]:.1f}%]")

    # Classification report
    labels = [l for l in STANCE_LABELS if l in set(y_true) | set(y_pred)]
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    print(f"\nPer-stance metrics:")
    for label in labels:
        r = report.get(label, {})
        print(f"  {label:12s}: P={r.get('precision',0):.3f} R={r.get('recall',0):.3f} "
              f"F1={r.get('f1-score',0):.3f} n={r.get('support',0)}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(f"\nConfusion Matrix (rows=human, cols=LLM):")
    print(cm_df.to_string())

    return {
        "n": n,
        "accuracy": round(accuracy, 4),
        "ci_95_wilson": [round(ci[0], 4), round(ci[1], 4)],
        "per_stance": {k: v for k, v in report.items() if k in labels},
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "macro_f1": round(report.get("macro avg", {}).get("f1-score", 0), 4),
        "weighted_f1": round(report.get("weighted avg", {}).get("f1-score", 0), 4),
    }


def compute_accuracy_by_confidence(consensus: pd.DataFrame) -> list:
    """Accuracy stratified by LLM confidence band."""
    bins = [(0.0, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)]
    bin_labels = ["low (<0.7)", "medium (0.7-0.85)", "high (0.85-0.95)", "very_high (0.95+)"]

    results = []
    print(f"\nAccuracy by Confidence Band:")
    for (lo, hi), label in zip(bins, bin_labels):
        mask = (consensus["llm_confidence"] >= lo) & (consensus["llm_confidence"] < hi)
        subset = consensus[mask]
        if len(subset) == 0:
            continue
        correct = (subset["consensus_stance"] == subset["llm_stance"]).sum()
        acc = correct / len(subset)
        ci = wilson_ci(correct, len(subset))
        entry = {
            "band": label,
            "n": int(len(subset)),
            "accuracy": round(acc, 4),
            "ci_95_wilson": [round(ci[0], 4), round(ci[1], 4)],
        }
        results.append(entry)
        print(f"  {label:25s}: {correct}/{len(subset)} = {100*acc:.1f}% "
              f"[{100*ci[0]:.1f}%, {100*ci[1]:.1f}%]")

    return results


def compute_accuracy_by_gender(consensus: pd.DataFrame) -> list:
    """Accuracy stratified by speaker gender."""
    results = []
    print(f"\nAccuracy by Gender:")
    for gender in ["M", "F"]:
        subset = consensus[consensus["gender"] == gender]
        if len(subset) == 0:
            continue
        correct = (subset["consensus_stance"] == subset["llm_stance"]).sum()
        acc = correct / len(subset)
        ci = wilson_ci(correct, len(subset))
        entry = {
            "gender": gender,
            "n": int(len(subset)),
            "accuracy": round(acc, 4),
            "ci_95_wilson": [round(ci[0], 4), round(ci[1], 4)],
        }
        results.append(entry)
        print(f"  {gender}: {correct}/{len(subset)} = {100*acc:.1f}% "
              f"[{100*ci[0]:.1f}%, {100*ci[1]:.1f}%]")

    return results


def compute_iaa(annotations: pd.DataFrame) -> dict:
    """Inter-annotator agreement metrics."""
    annotators = sorted(annotations["annotator"].unique())
    n_annotators = len(annotators)

    if n_annotators < 2:
        print("\nInter-annotator agreement: need >= 2 annotators")
        return {"note": "Only 1 annotator found"}

    print(f"\nInter-Annotator Agreement ({n_annotators} annotators):")

    # Pairwise Cohen's kappa
    pairwise_kappa = []
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            a1 = annotations[annotations["annotator"] == annotators[i]]
            a2 = annotations[annotations["annotator"] == annotators[j]]
            overlap = a1.merge(a2, on="speech_id", suffixes=("_1", "_2"))
            if len(overlap) < 5:
                continue
            kappa = cohen_kappa_score(
                overlap["human_stance_1"].values,
                overlap["human_stance_2"].values,
            )
            pairwise_kappa.append({
                "annotator_1": annotators[i],
                "annotator_2": annotators[j],
                "n_overlap": int(len(overlap)),
                "kappa": round(kappa, 4),
            })
            print(f"  {annotators[i]} vs {annotators[j]}: kappa={kappa:.4f} (n={len(overlap)})")

    # Krippendorff's alpha (multi-annotator)
    # Build data matrix: rows=speeches, cols=annotators
    all_speeches = sorted(annotations["speech_id"].unique())
    label_to_int = {l: i for i, l in enumerate(STANCE_LABELS)}

    data_matrix = np.full((len(all_speeches), n_annotators), np.nan)
    speech_to_idx = {s: i for i, s in enumerate(all_speeches)}

    for _, row in annotations.iterrows():
        s_idx = speech_to_idx[row["speech_id"]]
        a_idx = annotators.index(row["annotator"])
        stance = row["human_stance"]
        if stance in label_to_int:
            data_matrix[s_idx, a_idx] = label_to_int[stance]

    alpha = krippendorff_alpha(data_matrix, level="nominal")
    print(f"  Krippendorff's alpha: {alpha:.4f}")

    return {
        "n_annotators": n_annotators,
        "annotators": annotators,
        "pairwise_kappa": pairwise_kappa,
        "krippendorff_alpha": round(alpha, 4),
    }


def compute_human_llm_agreement(consensus: pd.DataFrame,
                                 gpt_path: str = None) -> dict:
    """Compare human consensus vs Claude vs GPT-4o."""
    results = {}

    # Human vs Claude
    y_human = consensus["consensus_stance"].values
    y_claude = consensus["llm_stance"].values
    kappa_claude = cohen_kappa_score(y_human, y_claude)
    results["human_vs_claude"] = {
        "kappa": round(kappa_claude, 4),
        "agreement_rate": round((y_human == y_claude).mean(), 4),
    }
    print(f"\nHuman vs Claude: kappa={kappa_claude:.4f}")

    # Human vs GPT-4o (if available)
    if gpt_path and Path(gpt_path).exists():
        gpt = pd.read_parquet(gpt_path)
        merged = consensus.merge(
            gpt[["speech_id", "stance"]].rename(columns={"stance": "gpt_stance"}),
            on="speech_id",
            how="left",
        )
        merged = merged.dropna(subset=["gpt_stance"])
        if len(merged) > 0:
            kappa_gpt = cohen_kappa_score(
                merged["consensus_stance"].values,
                merged["gpt_stance"].values,
            )
            results["human_vs_gpt"] = {
                "kappa": round(kappa_gpt, 4),
                "agreement_rate": round((merged["consensus_stance"] == merged["gpt_stance"]).mean(), 4),
                "n": int(len(merged)),
            }
            print(f"Human vs GPT-4o-mini: kappa={kappa_gpt:.4f}")

            # Which agrees more?
            if kappa_claude > kappa_gpt:
                results["better_agreement"] = "claude"
            else:
                results["better_agreement"] = "gpt"
            print(f"Better agreement: {results['better_agreement']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute validation metrics from annotations")
    parser.add_argument(
        "--annotations-dir",
        default="outputs/validation/annotations",
        help="Directory containing annotator CSV files",
    )
    parser.add_argument(
        "--gpt-results",
        default="outputs/llm_classification/full_results_v6_context_3_expanded.parquet",
        help="GPT-4o-mini results for comparison",
    )
    parser.add_argument("--output-dir", default="outputs/experiments")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = Path(args.annotations_dir)

    print("Loading annotations...")
    try:
        annotations = load_annotations(annotations_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run the validation app first: streamlit run scripts/experiments/validation_app.py")
        return

    results = {}

    # Consensus
    consensus = compute_consensus(annotations)
    print(f"\nConsensus labels for {len(consensus)} speeches")

    # Accuracy
    results["accuracy"] = compute_accuracy_metrics(consensus)
    results["accuracy_by_confidence"] = compute_accuracy_by_confidence(consensus)
    results["accuracy_by_gender"] = compute_accuracy_by_gender(consensus)

    # IAA
    results["inter_annotator_agreement"] = compute_iaa(annotations)

    # Human-LLM agreement
    results["human_llm_agreement"] = compute_human_llm_agreement(
        consensus, args.gpt_results
    )

    # Reason quality
    if "reason_quality" in annotations.columns:
        rq = annotations["reason_quality"].value_counts(normalize=True)
        results["reason_quality"] = {k: round(v, 4) for k, v in rq.items()}
        print(f"\nReason Quality Distribution:")
        for k, v in rq.items():
            print(f"  {k}: {100*v:.1f}%")

    # Save
    json_path = output_dir / "validation_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
