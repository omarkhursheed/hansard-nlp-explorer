"""
Compute inter-annotator agreement and LLM-vs-human agreement on sexism
classification axes (binary, AST, SCM, Gender Norms).

Addresses all three reviewers' concern that sexism classification was
not validated against human annotations.

Inputs (all in this folder unless noted):
  - omar_v2.jsonl (v2 sexism annotations)
  - mandira_v2.jsonl (v2 sexism annotations)
  - sexism_resolutions.json (19 resolved disagreements)
  - outputs/validation/annotations/*.jsonl (v1 stance annotations, for gold stances)
  - outputs/validation/resolutions.json (v1 stance resolutions, for gold stances)
  - outputs/llm_classification/v7_notrunc_results.parquet (LLM classifications)

Run from repo root:
    python experiments/20260503_sexism_validation/01_sexism_axis_agreement.py
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

SCRIPT_DIR = Path(__file__).parent
OUTPUT_JSON = SCRIPT_DIR / "01_sexism_axis_agreement.json"
OUTPUT_TXT = SCRIPT_DIR / "01_sexism_axis_agreement.txt"

V2_OMAR = SCRIPT_DIR / "omar_v2.jsonl"
V2_MANDIRA = SCRIPT_DIR / "mandira_v2.jsonl"
SEXISM_RESOLUTIONS = SCRIPT_DIR / "sexism_resolutions.json"

V1_OMAR = Path("outputs/validation/annotations/omar.jsonl")
V1_MANDIRA = Path("outputs/validation/annotations/mandira.jsonl")
STANCE_RESOLUTIONS = Path("outputs/validation/resolutions.json")

CLF_PATH = Path("outputs/llm_classification/v7_notrunc_results.parquet")

FIELDS = [
    ("binary", "Binary (sexist/not)"),
    ("axis_a_label", "Axis A (AST)"),
    ("axis_b_label", "Axis B (SCM)"),
    ("axis_c_label", "Axis C (Norm)"),
]


def load_jsonl(path):
    return {json.loads(l)["speech_id"]: json.loads(l)
            for l in open(path)}


def build_gold_stances():
    omar_v1 = load_jsonl(V1_OMAR)
    mandira_v1 = load_jsonl(V1_MANDIRA)
    resolutions = json.loads(STANCE_RESOLUTIONS.read_text())

    def norm(s):
        return "irrelevant" if s == "neutral" else s

    omar_s = {sid: norm(r["human_stance"]) for sid, r in omar_v1.items()}
    mandira_s = {sid: norm(r["human_stance"])
                 for sid, r in mandira_v1.items() if r.get("human_stance")}

    gold = {}
    for sid in set(omar_s) & set(mandira_s):
        if omar_s[sid] == mandira_s[sid]:
            gold[sid] = omar_s[sid]
    for sid, res in resolutions.items():
        gold[sid] = res["resolved_stance"]
    return gold


def build_sexism_gold(omar, mandira, sexism_res, relevant_ids):
    gold = {}
    for sid in relevant_ids:
        if sid in sexism_res:
            res = sexism_res[sid]
            gold[sid] = {
                "binary": res["resolved_binary"],
                "axis_a_label": res.get("resolved_axis_a", "none"),
                "axis_b_label": res.get("resolved_axis_b", "none"),
                "axis_c_label": res.get("resolved_axis_c", "none"),
                "source": "resolution",
            }
        else:
            ob = omar[sid].get("binary", "none")
            mb = mandira[sid].get("binary", "none")
            if ob == mb:
                rec = {"binary": ob, "source": "agreement"}
                for f in ["axis_a_label", "axis_b_label", "axis_c_label"]:
                    ov = omar[sid].get(f, "none")
                    mv = mandira[sid].get(f, "none")
                    rec[f] = ov if ov == mv else ov
                gold[sid] = rec
            else:
                gold[sid] = {"binary": "UNRESOLVED", "source": "missing"}
    return gold


def kappa(a, b):
    if len(set(a + b)) < 2:
        return float("nan")
    return cohen_kappa_score(a, b)


def main():
    omar = load_jsonl(V2_OMAR)
    mandira = load_jsonl(V2_MANDIRA)
    sexism_res = json.loads(SEXISM_RESOLUTIONS.read_text())
    gold_stance = build_gold_stances()

    clf = pd.read_parquet(CLF_PATH)
    llm = {r["speech_id"]: r for _, r in clf.iterrows()}

    shared = sorted(set(omar) & set(mandira))
    relevant = [s for s in shared
                if gold_stance.get(s) in ("for", "against", "both")
                and not omar[s].get("skipped")
                and not mandira[s].get("skipped")]

    gold = build_sexism_gold(omar, mandira, sexism_res, relevant)
    gold_ids = [s for s in relevant if gold[s]["source"] != "missing"]
    sexist_ids = [s for s in gold_ids if gold[s]["binary"] == "sexist"]

    n_agree = sum(1 for g in gold.values() if g["source"] == "agreement")
    n_resolved = sum(1 for g in gold.values() if g["source"] == "resolution")

    lines = []
    results = {}

    def out(text=""):
        lines.append(text)
        print(text)

    out("SEXISM AXIS VALIDATION")
    out("=" * 70)
    out(f"Relevant speeches (gold stance for/against/both): {len(relevant)}")
    out(f"Consensus labels: {n_agree} agreements + {n_resolved} resolutions = {len(gold_ids)}")
    out(f"Consensus sexist: {len(sexist_ids)}")
    out(f"Consensus not_sexist: {len(gold_ids) - len(sexist_ids)}")
    out()

    # Gold distribution for sexist speeches
    out("Gold sexist distribution:")
    for field, label in FIELDS[1:]:
        vals = Counter(gold[s].get(field, "none") for s in sexist_ids)
        out(f"  {label}: {dict(vals)}")
    out()

    # --- Human-human ---
    out("=" * 70)
    out(f"HUMAN-HUMAN AGREEMENT (before resolution, n={len(relevant)})")
    out("=" * 70)
    results["human_human"] = {}
    for field, label in FIELDS:
        o = [omar[s].get(field, "none") for s in relevant]
        m = [mandira[s].get(field, "none") for s in relevant]
        agree = sum(1 for a, b in zip(o, m) if a == b)
        k = kappa(o, m)
        out(f"  {label:25s} {agree}/{len(relevant)} ({agree/len(relevant)*100:.0f}%)  kappa={k:.3f}")
        results["human_human"][field] = {
            "agree": agree, "n": len(relevant), "kappa": round(float(k), 3),
        }
    out()

    # --- LLM vs consensus ---
    out("=" * 70)
    out(f"LLM vs HUMAN-CONSENSUS (after resolution, n={len(gold_ids)})")
    out("=" * 70)
    results["llm_vs_consensus"] = {}
    for field, label in FIELDS:
        h = [gold[s].get(field, "none") or "none" for s in gold_ids]
        l = [llm[s][field] if s in llm else "none" for s in gold_ids]
        l = [v if v else "none" for v in l]
        agree = sum(1 for a, b in zip(h, l) if a == b)
        k = kappa(h, l)
        out(f"  {label:25s} {agree}/{len(gold_ids)} ({agree/len(gold_ids)*100:.0f}%)  kappa={k:.3f}")
        results["llm_vs_consensus"][field] = {
            "agree": agree, "n": len(gold_ids), "kappa": round(float(k), 3),
        }
    out()

    # --- LLM vs consensus on sexist only ---
    out("=" * 70)
    out(f"LLM vs HUMAN-CONSENSUS -- SEXIST ONLY (n={len(sexist_ids)})")
    out("=" * 70)
    results["llm_vs_consensus_sexist"] = {}
    for field, label in FIELDS[1:]:
        h = [gold[s].get(field, "none") or "none" for s in sexist_ids]
        l = [llm[s][field] if s in llm else "none" for s in sexist_ids]
        l = [v if v else "none" for v in l]
        agree = sum(1 for a, b in zip(h, l) if a == b)
        out(f"  {label:25s} {agree}/{len(sexist_ids)} ({agree/len(sexist_ids)*100:.0f}%)")
        out(f"    Human: {dict(Counter(h))}")
        out(f"    LLM:   {dict(Counter(l))}")
        results["llm_vs_consensus_sexist"][field] = {
            "agree": agree, "n": len(sexist_ids),
            "human_dist": dict(Counter(h)), "llm_dist": dict(Counter(l)),
        }
    out()

    # --- LLM vs individual annotators ---
    out("=" * 70)
    out(f"LLM vs INDIVIDUAL ANNOTATORS (n={len(relevant)})")
    out("=" * 70)
    for name, annots in [("Omar", omar), ("Mandira", mandira)]:
        out(f"\n  LLM vs {name}:")
        results[f"llm_vs_{name.lower()}"] = {}
        for field, label in FIELDS:
            h = [annots[s].get(field, "none") for s in relevant]
            l = [llm[s][field] if s in llm else "none" for s in relevant]
            h = [v if v else "none" for v in h]
            l = [v if v else "none" for v in l]
            agree = sum(1 for a, b in zip(h, l) if a == b)
            k = kappa(h, l)
            out(f"    {label:25s} {agree}/{len(relevant)} ({agree/len(relevant)*100:.0f}%)  kappa={k:.3f}")
            results[f"llm_vs_{name.lower()}"][field] = {
                "agree": agree, "n": len(relevant), "kappa": round(float(k), 3),
            }
    out()

    # --- Summary ---
    results["summary"] = {
        "n_relevant": len(relevant),
        "n_gold": len(gold_ids),
        "n_sexist": len(sexist_ids),
        "n_agreements": n_agree,
        "n_resolutions": n_resolved,
        "gold_binary_dist": dict(Counter(gold[s]["binary"] for s in gold_ids)),
    }

    # Save
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    out(f"Saved {OUTPUT_JSON}")

    with open(OUTPUT_TXT, "w") as f:
        f.write("\n".join(lines))
    out(f"Saved {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
