"""
TF-IDF + Logistic Regression baseline for suffrage stance classification.

Uses manually validated speeches as training data (when available),
otherwise uses stratified LLM labels as pseudo-labels with cross-validation.
Compares against LLM classifier using McNemar's test.

Usage:
    python scripts/experiments/tfidf_baseline.py
    python scripts/experiments/tfidf_baseline.py --use-llm-labels
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


LABELS = ["for", "against", "both", "irrelevant"]


def load_data(llm_path: str, input_path: str, annotations_dir: str = None):
    """Load speech texts and labels."""
    llm = pd.read_parquet(llm_path)
    inputs = pd.read_parquet(input_path)

    merged = llm[["speech_id", "stance", "confidence"]].merge(
        inputs[["speech_id", "target_text"]],
        on="speech_id",
        how="inner",
    )
    merged = merged.dropna(subset=["stance", "target_text"])
    merged["stance"] = merged["stance"].astype(str)
    # Map neutral -> irrelevant (too few samples)
    merged["stance"] = merged["stance"].replace("neutral", "irrelevant")

    # Try to load human annotations for ground truth
    human_labels = None
    if annotations_dir and Path(annotations_dir).exists():
        all_dfs = []
        for path in Path(annotations_dir).glob("*.csv"):
            all_dfs.append(pd.read_csv(path))
        if all_dfs:
            annotations = pd.concat(all_dfs, ignore_index=True)
            # Use majority vote for consensus
            consensus = annotations.groupby("speech_id")["human_stance"].agg(
                lambda x: x.value_counts().index[0]
            ).reset_index()
            consensus.columns = ["speech_id", "human_stance"]
            human_labels = consensus
            print(f"Loaded {len(human_labels)} human-annotated labels")

    return merged, human_labels


def mcnemar_test(y_true, y_pred_1, y_pred_2) -> dict:
    """McNemar's test comparing two classifiers.

    Tests whether the classifiers have the same error rate.
    """
    # Build contingency table of correct/incorrect
    correct_1 = y_true == y_pred_1
    correct_2 = y_true == y_pred_2

    # b: model 1 correct, model 2 incorrect
    b = (correct_1 & ~correct_2).sum()
    # c: model 1 incorrect, model 2 correct
    c = (~correct_1 & correct_2).sum()

    # McNemar's test with continuity correction
    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": int(b), "c": int(c)}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = scipy_stats.chi2.sf(chi2, df=1)

    return {
        "chi2": round(float(chi2), 4),
        "p_value": float(p_value),
        "b_model1_right_model2_wrong": int(b),
        "c_model1_wrong_model2_right": int(c),
        "significant_005": p_value < 0.05,
    }


def run_tfidf_lr(merged: pd.DataFrame, human_labels: pd.DataFrame = None,
                  use_llm_labels: bool = False) -> dict:
    """Train and evaluate TF-IDF + LR classifier."""

    results = {}

    # Set up training data
    if human_labels is not None and not use_llm_labels:
        # Use human labels as ground truth
        train_data = merged.merge(human_labels, on="speech_id", how="inner")
        train_data["label"] = train_data["human_stance"].replace("neutral", "irrelevant")
        print(f"Using {len(train_data)} human-annotated speeches for training")
        mode = "human_labels"
    else:
        # Use LLM labels as pseudo-labels (cross-validation measures consistency)
        train_data = merged.copy()
        train_data["label"] = train_data["stance"]
        print(f"Using {len(train_data)} LLM-labeled speeches (pseudo-labels)")
        mode = "llm_pseudo_labels"

    # Filter to labels with enough samples
    label_counts = train_data["label"].value_counts()
    valid_labels = [l for l in LABELS if label_counts.get(l, 0) >= 5]
    train_data = train_data[train_data["label"].isin(valid_labels)]
    print(f"Labels with >= 5 samples: {valid_labels}")
    print(f"Training data: {len(train_data)} speeches")
    print(f"Label distribution:\n{train_data['label'].value_counts()}")

    X = train_data["target_text"].values
    y = train_data["label"].values

    # Pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])

    # Cross-validation
    n_splits = min(5, min(label_counts[l] for l in valid_labels if l in label_counts))
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"\nRunning {n_splits}-fold cross-validation...")
    y_pred_cv = cross_val_predict(pipeline, X, y, cv=cv)

    # CV metrics
    report_cv = classification_report(y, y_pred_cv, labels=valid_labels,
                                       output_dict=True, zero_division=0)
    kappa_cv = cohen_kappa_score(y, y_pred_cv)
    cm_cv = confusion_matrix(y, y_pred_cv, labels=valid_labels)

    print(f"\n{'='*60}")
    print(f"TF-IDF + LR CROSS-VALIDATION RESULTS (mode={mode})")
    print(f"{'='*60}")
    print(f"Cohen's kappa: {kappa_cv:.4f}")
    accuracy_cv = (y == y_pred_cv).mean()
    print(f"Accuracy: {100*accuracy_cv:.1f}%")

    for label in valid_labels:
        r = report_cv.get(label, {})
        print(f"  {label:12s}: P={r.get('precision',0):.3f} R={r.get('recall',0):.3f} "
              f"F1={r.get('f1-score',0):.3f}")

    print(f"\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm_cv, index=valid_labels, columns=valid_labels)
    print(cm_df.to_string())

    results["cv_metrics"] = {
        "mode": mode,
        "n_splits": n_splits,
        "n_samples": int(len(train_data)),
        "accuracy": round(accuracy_cv, 4),
        "kappa": round(kappa_cv, 4),
        "macro_f1": round(report_cv.get("macro avg", {}).get("f1-score", 0), 4),
        "weighted_f1": round(report_cv.get("weighted avg", {}).get("f1-score", 0), 4),
        "per_class": {k: v for k, v in report_cv.items() if k in valid_labels},
        "confusion_matrix": {"labels": valid_labels, "matrix": cm_cv.tolist()},
    }

    # Now train on full data and predict all speeches for comparison with LLM
    if mode == "human_labels":
        pipeline.fit(X, y)
        X_all = merged["target_text"].values
        y_pred_all = pipeline.predict(X_all)
        y_llm_all = merged["stance"].values

        # Compare TF-IDF predictions vs LLM on the full dataset
        # Use human labels as reference where available
        human_speech_ids = set(human_labels["speech_id"])
        eval_mask = merged["speech_id"].isin(human_speech_ids)
        eval_data = merged[eval_mask]

        y_true_eval = eval_data.merge(human_labels, on="speech_id")["human_stance"].replace("neutral", "irrelevant").values
        y_tfidf_eval = y_pred_all[eval_mask.values]
        y_llm_eval = eval_data["stance"].values

        # McNemar's test
        mcnemar = mcnemar_test(y_true_eval, y_llm_eval, y_tfidf_eval)
        results["mcnemar_llm_vs_tfidf"] = mcnemar
        print(f"\nMcNemar's test (LLM vs TF-IDF):")
        print(f"  chi2={mcnemar['chi2']}, p={mcnemar['p_value']:.4e}")
        print(f"  LLM right/TF-IDF wrong: {mcnemar['b_model1_right_model2_wrong']}")
        print(f"  TF-IDF right/LLM wrong: {mcnemar['c_model1_wrong_model2_right']}")

    # Also run on full LLM-labeled data for the comparison table
    if mode == "llm_pseudo_labels":
        # Cross-val predictions are our TF-IDF predictions
        # Compare with LLM labels (which are the "ground truth" here)
        mcnemar = mcnemar_test(y, y, y_pred_cv)
        results["note"] = ("McNemar's test not meaningful when LLM labels are used as "
                          "ground truth (circular). Use human labels for valid comparison.")

    # Feature importance (top keywords per class)
    pipeline.fit(X, y)
    feature_names = pipeline["tfidf"].get_feature_names_out()
    coefs = pipeline["clf"].coef_

    top_features = {}
    for i, label in enumerate(pipeline["clf"].classes_):
        if i < len(coefs):
            top_idx = coefs[i].argsort()[-15:][::-1]
            top_features[label] = [
                {"feature": feature_names[j], "weight": round(float(coefs[i][j]), 4)}
                for j in top_idx
            ]
            print(f"\nTop features for '{label}':")
            for item in top_features[label][:10]:
                print(f"  {item['feature']:30s} {item['weight']:.4f}")

    results["top_features"] = top_features

    return results


def main():
    parser = argparse.ArgumentParser(description="TF-IDF + LR baseline classifier")
    parser.add_argument(
        "--data",
        default="outputs/llm_classification/claude_sonnet_45_full_results.parquet",
    )
    parser.add_argument(
        "--input",
        default="outputs/llm_classification/full_input_context_3_expanded.parquet",
    )
    parser.add_argument(
        "--annotations-dir",
        default="outputs/validation/annotations",
    )
    parser.add_argument(
        "--use-llm-labels",
        action="store_true",
        help="Use LLM labels as pseudo-labels even if human annotations exist",
    )
    parser.add_argument("--output-dir", default="outputs/experiments")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged, human_labels = load_data(args.data, args.input, args.annotations_dir)

    results = run_tfidf_lr(merged, human_labels, args.use_llm_labels)

    # Save
    json_path = output_dir / "tfidf_baseline_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
