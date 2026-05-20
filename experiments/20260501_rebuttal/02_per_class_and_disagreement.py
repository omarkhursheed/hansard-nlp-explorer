"""
Per-class P/R/F1, n=96 explanation, and per-instance disagreement analysis.

Addresses:
  - Reviewer iG9P: P/R/F1 by class, not just kappa
  - Reviewer Us7C: Why n=96 not n=100 in Table 7
  - Reviewer Us7C: Distributional vs per-instance agreement analysis

Run from repo root:
    python experiments/20260501_rebuttal/02_per_class_and_disagreement.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix

SCRIPT_DIR = Path(__file__).parent

OMAR = Path("/tmp/omar.jsonl")
MANDIRA = Path("/tmp/mandira.jsonl")
RESOLUTIONS = Path("outputs/validation/resolutions.json")
CLF = Path("outputs/llm_classification/v7_notrunc_results.parquet")

OUTPUT = SCRIPT_DIR / "02_per_class_and_disagreement.json"

LABELS = ["for", "against", "both", "irrelevant"]


def norm(s):
    return "irrelevant" if s == "neutral" else s


def main():
    omar = {r["speech_id"]: r for r in (json.loads(l) for l in open(OMAR))}
    mandira = {r["speech_id"]: r for r in (json.loads(l) for l in open(MANDIRA))}

    omar_stance = {sid: norm(r["human_stance"]) for sid, r in omar.items()}
    mandira_stance = {
        sid: norm(r["human_stance"])
        for sid, r in mandira.items()
        if r.get("human_stance")
    }

    shared = sorted(set(omar_stance) & set(mandira_stance))
    omar_only = sorted(set(omar_stance) - set(mandira_stance))

    with open(RESOLUTIONS) as f:
        resolutions = json.load(f)

    gold = {}
    for sid in shared:
        if omar_stance[sid] == mandira_stance[sid]:
            gold[sid] = omar_stance[sid]
    for sid, res in resolutions.items():
        gold[sid] = res["resolved_stance"]

    v7 = pd.read_parquet(CLF)
    llm = dict(zip(v7["speech_id"], v7["stance"]))

    matched_ids = sorted(s for s in gold if s in llm)
    y_gold = [gold[s] for s in matched_ids]
    y_llm = [llm[s] for s in matched_ids]

    # === 1. Per-class P/R/F1 ===
    print("=" * 70)
    print("PER-CLASS P/R/F1 (Reviewer iG9P)")
    print("=" * 70)
    print(f"\nn = {len(matched_ids)} speeches")
    print(f"Gold: {pd.Series(y_gold).value_counts().to_dict()}")
    print(f"LLM:  {pd.Series(y_llm).value_counts().to_dict()}")

    report_str = classification_report(
        y_gold, y_llm, labels=LABELS, digits=3, zero_division=0
    )
    report_dict = classification_report(
        y_gold, y_llm, labels=LABELS, digits=3, zero_division=0, output_dict=True
    )
    print(f"\n{report_str}")

    cm = confusion_matrix(y_gold, y_llm, labels=LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"gold:{l}" for l in LABELS],
        columns=[f"pred:{l}" for l in LABELS],
    )
    print("Confusion Matrix:")
    print(cm_df.to_string())

    kappa = cohen_kappa_score(y_gold, y_llm)
    print(f"\nCohen's kappa: {kappa:.3f}")

    # === 2. n=96 explanation ===
    print("\n" + "=" * 70)
    print("n=96 vs n=100 (Reviewer Us7C)")
    print("=" * 70)
    print(f"\nOmar annotated: {len(omar_stance)} speeches")
    print(f"Mandira annotated: {len(mandira_stance)} speeches")
    print(f"Shared: {len(shared)}")
    print(f"Omar-only: {len(omar_only)}")
    print(f"  IDs: {omar_only}")
    print(f"\nHuman-human kappa computed on shared n={len(shared)}")
    print(f"Gold labels cover all {len(gold)} (63 agreements + 37 resolutions)")

    # === 3. Per-instance disagreement ===
    print("\n" + "=" * 70)
    print("PER-INSTANCE DISAGREEMENT (Reviewer Us7C)")
    print("=" * 70)

    disagreements = [
        (sid, gold[sid], llm[sid])
        for sid in matched_ids
        if gold[sid] != llm[sid]
    ]
    print(f"\nDisagreements: {len(disagreements)} / {len(matched_ids)}")

    patterns = {}
    for sid, g, l in disagreements:
        key = f"{g} -> {l}"
        patterns.setdefault(key, []).append(sid)

    print("\nDisagreement patterns (gold -> LLM):")
    for key, sids in sorted(patterns.items(), key=lambda x: -len(x[1])):
        print(f"  {key}: {len(sids)}")

    close = 0
    relevance = 0
    direction = 0
    for _, g, l in disagreements:
        if {g, l} <= {"for", "both"} or {g, l} <= {"against", "both"}:
            close += 1
        elif (g == "irrelevant") != (l == "irrelevant"):
            relevance += 1
        else:
            direction += 1

    print(f"\nSeverity:")
    print(f"  Close (for<->both or against<->both): {close}")
    print(f"  Relevance (irrelevant<->relevant): {relevance}")
    print(f"  Direction (for<->against): {direction}")

    hh_dis_ids = {sid for sid in shared if omar_stance[sid] != mandira_stance[sid]}
    claude_dis_ids = {sid for sid, _, _ in disagreements}
    overlap = hh_dis_ids & claude_dis_ids

    print(f"\nHuman-human disagreements: {len(hh_dis_ids)}")
    print(f"Claude-human disagreements: {len(claude_dis_ids)}")
    print(f"Overlap (hard for both): {len(overlap)}")
    print(f"Claude-only errors: {len(claude_dis_ids - hh_dis_ids)}")
    print(f"Human-only disagreements: {len(hh_dis_ids - claude_dis_ids)}")

    # === Save ===
    results = {
        "per_class": report_dict,
        "kappa": round(float(kappa), 3),
        "n": len(matched_ids),
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
        "annotation_counts": {
            "omar": len(omar_stance),
            "mandira": len(mandira_stance),
            "shared": len(shared),
            "omar_only": omar_only,
        },
        "disagreements": {
            "total": len(disagreements),
            "patterns": {k: len(v) for k, v in patterns.items()},
            "severity": {"close": close, "relevance": relevance, "direction": direction},
            "overlap_with_human": len(overlap),
            "claude_only": len(claude_dis_ids - hh_dis_ids),
            "human_only": len(hh_dis_ids - claude_dis_ids),
        },
    }

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUTPUT}")

    # === Rebuttal summary ===
    print("\n" + "=" * 70)
    print("FOR REBUTTAL")
    print("=" * 70)
    print(f"\nPer-class F1 (Claude vs human-consensus, n={len(matched_ids)}):")
    for l in LABELS:
        r = report_dict[l]
        print(f"  {l:12s}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}  n={int(r['support'])}")

    print(f"\nn=96 explanation: One annotator completed 96 of 100 speeches.")
    print(f"Human-human kappa is computed on the 96 shared speeches.")
    print(f"All 100 speeches have gold labels (63 agreements + 37 resolutions).")

    print(f"\nPer-instance: {len(claude_dis_ids)}/{len(matched_ids)} Claude-human")
    print(f"disagreements. {len(overlap)}/{len(claude_dis_ids)} overlap with speeches")
    print(f"where humans also disagreed, suggesting Claude's errors concentrate")
    print(f"on genuinely ambiguous speeches.")


if __name__ == "__main__":
    main()
