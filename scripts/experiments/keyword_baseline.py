"""
Keyword/regex baseline classifier for suffrage speech stance detection.

Provides a simple non-LLM baseline that reviewers requested.
Classifies based on presence/absence of curated keyword sets.

Usage:
    python scripts/experiments/keyword_baseline.py
    python scripts/experiments/keyword_baseline.py --data path/to/results.parquet
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Keyword sets (curated from domain expertise + exploratory analysis)
# ---------------------------------------------------------------------------

# Each pattern list: (regex_pattern, weight)
# Higher weight = stronger signal for that stance
FOR_PATTERNS = [
    (r"\bright to vote\b", 2),
    (r"\bequal franchise\b", 2),
    (r"\bjustice demands?\b", 1),
    (r"\bentitled to (?:the )?(?:vote|franchise|suffrage)\b", 2),
    (r"\bdemocratic (?:right|principle)\b", 1),
    (r"\bextend (?:the )?franchise\b", 2),
    (r"\bwomen(?:'s| should| ought| must) (?:have|be given|receive|get) (?:the )?vote\b", 2),
    (r"\bno taxation without representation\b", 2),
    (r"\bsex disqualification\b", 1),
    (r"\bequal(?:ity)? (?:of|between) (?:the )?sex(?:es)?\b", 1),
    (r"\bwomen are (?:equally|just as) capab\b", 1),
    (r"\bsupport (?:this|the) (?:bill|measure|motion)\b", 1),
    (r"\bin favour of (?:giving|granting|extending)\b", 2),
    (r"\bwomen's suffrage\b", 1),
    (r"\benfranchise(?:ment)?\b", 1),
    (r"\bgrant(?:ing)? (?:the )?(?:vote|franchise|suffrage) to women\b", 2),
    (r"\bpolitical rights? (?:of|for) women\b", 1),
]

AGAINST_PATTERNS = [
    (r"\bunfit (?:to|for) (?:vote|exercise)\b", 2),
    (r"\bnature never intended\b", 2),
    (r"\bdanger(?:ous)? (?:to|for) (?:the )?(?:state|constitution|country)\b", 1),
    (r"\bweaker sex\b", 2),
    (r"\bnot ready\b", 1),
    (r"\boppose (?:this|the) (?:bill|measure|motion)\b", 2),
    (r"\bwomen(?:'s)? (?:proper|true|natural) (?:place|sphere|role)\b", 2),
    (r"\bphysical(?:ly)? (?:in)?capable\b", 1),
    (r"\bdomestic dut(?:y|ies)\b", 1),
    (r"\bnot (?:fitted|qualified|competent)\b", 1),
    (r"\bvote against (?:this|the) (?:bill|measure|motion)\b", 2),
    (r"\bpetticoat government\b", 2),
    (r"\binvasion of (?:the )?(?:home|domestic)\b", 1),
    (r"\bsex (?:war|conflict|antagonism)\b", 1),
    (r"\bemotional(?:ly)? (?:unfit|unstable|incapable)\b", 2),
    (r"\bopen(?:ing)? (?:the )?floodgates\b", 1),
]

RELEVANCE_PATTERNS = [
    (r"\bsuffrage\b", 1),
    (r"\bfranchise\b", 1),
    (r"\bwomen(?:'s)? (?:vote|voting)\b", 1),
    (r"\bfemale (?:suffrage|franchise|vote)\b", 1),
    (r"\bvotes? for women\b", 2),
    (r"\brepresentation of the people\b", 1),
    (r"\bparliamentary (?:franchise|vote)\b", 1),
    (r"\belectoral (?:right|qualification)\b", 1),
]


def score_text(text: str, patterns: list) -> float:
    """Score text against a list of (regex, weight) patterns."""
    text_lower = text.lower()
    total = 0.0
    for pattern, weight in patterns:
        matches = len(re.findall(pattern, text_lower))
        total += matches * weight
    return total


def classify_speech(text: str) -> tuple:
    """Classify a speech using keyword scoring.

    Returns (stance, confidence_score, for_score, against_score, relevance_score).
    """
    for_score = score_text(text, FOR_PATTERNS)
    against_score = score_text(text, AGAINST_PATTERNS)
    relevance_score = score_text(text, RELEVANCE_PATTERNS)

    # Must pass relevance threshold to be considered suffrage-related
    if relevance_score < 1 and for_score == 0 and against_score == 0:
        return "irrelevant", 0.0, for_score, against_score, relevance_score

    total = for_score + against_score
    if total == 0:
        # Has relevance keywords but no stance keywords
        return "irrelevant", 0.0, for_score, against_score, relevance_score

    # Classify based on score ratio
    ratio = for_score / total
    if ratio > 0.65:
        stance = "for"
    elif ratio < 0.35:
        stance = "against"
    elif for_score > 0 and against_score > 0:
        stance = "both"
    else:
        stance = "irrelevant"

    confidence = abs(for_score - against_score) / total
    return stance, confidence, for_score, against_score, relevance_score


def main():
    parser = argparse.ArgumentParser(description="Keyword baseline classifier")
    parser.add_argument(
        "--data",
        default="outputs/llm_classification/claude_sonnet_45_full_results.parquet",
        help="LLM classification results (for comparison)",
    )
    parser.add_argument(
        "--input",
        default="outputs/llm_classification/full_input_context_3_expanded.parquet",
        help="Input speeches with text",
    )
    parser.add_argument("--output-dir", default="outputs/experiments")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    llm_results = pd.read_parquet(args.data)
    inputs = pd.read_parquet(args.input)
    print(f"Loaded {len(llm_results)} LLM classifications, {len(inputs)} input speeches")

    # Merge to get text
    merged = llm_results[["speech_id", "stance", "confidence", "gender", "year"]].merge(
        inputs[["speech_id", "target_text"]],
        on="speech_id",
        how="inner",
    )
    merged = merged.rename(columns={"stance": "llm_stance", "confidence": "llm_confidence"})
    # Drop rows with missing LLM stance
    merged = merged.dropna(subset=["llm_stance"])
    merged["llm_stance"] = merged["llm_stance"].astype(str)
    print(f"Merged: {len(merged)} speeches with text")

    # Classify all speeches
    print("Running keyword classifier...")
    results = []
    for _, row in merged.iterrows():
        text = row["target_text"] if isinstance(row["target_text"], str) else ""
        stance, conf, for_s, against_s, rel_s = classify_speech(text)
        results.append({
            "speech_id": row["speech_id"],
            "keyword_stance": stance,
            "keyword_confidence": round(conf, 4),
            "for_score": for_s,
            "against_score": against_s,
            "relevance_score": rel_s,
            "llm_stance": row["llm_stance"],
            "llm_confidence": row["llm_confidence"],
            "gender": row["gender"],
            "year": row["year"],
        })

    results_df = pd.DataFrame(results)

    # Evaluation: keyword vs LLM
    y_llm = results_df["llm_stance"].values
    y_keyword = results_df["keyword_stance"].values

    # Map neutral -> irrelevant for fair comparison (keyword classifier doesn't distinguish)
    y_llm_mapped = np.where(y_llm == "neutral", "irrelevant", y_llm)

    labels = ["for", "against", "both", "irrelevant"]
    kappa = cohen_kappa_score(y_llm_mapped, y_keyword)

    print(f"\n{'='*60}")
    print("KEYWORD BASELINE vs LLM CLASSIFICATION")
    print(f"{'='*60}")
    print(f"Cohen's kappa: {kappa:.4f}")

    # Agreement rate
    agree = (y_llm_mapped == y_keyword).sum()
    print(f"Agreement: {agree}/{len(y_llm_mapped)} ({100*agree/len(y_llm_mapped):.1f}%)")

    # Classification report (LLM as reference)
    report = classification_report(
        y_llm_mapped, y_keyword, labels=labels, output_dict=True, zero_division=0
    )
    print(f"\nKeyword performance (LLM as reference):")
    for label in labels:
        r = report.get(label, {})
        print(f"  {label:12s}: P={r.get('precision',0):.3f} R={r.get('recall',0):.3f} "
              f"F1={r.get('f1-score',0):.3f} n={r.get('support',0)}")

    # Confusion matrix
    cm = confusion_matrix(y_llm_mapped, y_keyword, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"LLM:{l}" for l in labels], columns=[f"KW:{l}" for l in labels])
    print(f"\nConfusion Matrix:")
    print(cm_df.to_string())

    # Stance distribution comparison
    print(f"\nStance Distribution Comparison:")
    llm_dist = pd.Series(y_llm_mapped).value_counts(normalize=True)
    kw_dist = pd.Series(y_keyword).value_counts(normalize=True)
    for label in labels:
        print(f"  {label:12s}: LLM={llm_dist.get(label,0):.3f} vs KW={kw_dist.get(label,0):.3f}")

    # Save results
    output = {
        "method": "keyword_regex",
        "n_speeches": int(len(results_df)),
        "kappa_vs_llm": round(kappa, 4),
        "agreement_rate": round(agree / len(y_llm_mapped), 4),
        "classification_report": {k: v for k, v in report.items() if k in labels},
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "stance_distribution": {
            "keyword": {k: round(float(v), 4) for k, v in kw_dist.items()},
            "llm": {k: round(float(v), 4) for k, v in llm_dist.items()},
        },
        "n_for_patterns": len(FOR_PATTERNS),
        "n_against_patterns": len(AGAINST_PATTERNS),
    }

    json_path = output_dir / "keyword_baseline_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
