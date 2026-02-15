"""
Evaluate outputs from fine-tuned models (Phase 5).

Reads generation parquet files from all 4 conditions, scores via
Perspective API, classifies stereotypes, and runs statistical tests.

Usage:
    # After generations exist locally
    python scripts/experiments/finetune_evaluate.py

    # Specify custom paths
    python scripts/experiments/finetune_evaluate.py --generations-dir outputs/experiments
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


TOXICITY_ATTRS = [
    "toxicity", "severe_toxicity", "identity_attack",
    "insult", "sexually_explicit", "threat",
]

CONDITIONS = ["anti_suffrage", "pro_suffrage", "random_hansard", "base"]


def load_generations(generations_dir: str) -> pd.DataFrame:
    """Load generation results from all conditions."""
    gen_dir = Path(generations_dir)
    all_dfs = []

    for condition in CONDITIONS:
        path = gen_dir / f"finetune_generations_{condition}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            print(f"  {condition}: {len(df)} generations")
            all_dfs.append(df)

    # Also try a single combined file
    combined_path = gen_dir / "finetuning_results.parquet"
    if combined_path.exists() and not all_dfs:
        df = pd.read_parquet(combined_path)
        print(f"  Combined file: {len(df)} generations")
        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"No generation files found in {gen_dir}")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Total: {len(combined)} generations across {combined['condition'].nunique()} conditions")
    return combined


def compute_perplexity_check(df: pd.DataFrame) -> dict:
    """Sanity check: compare text statistics across conditions."""
    results = {}

    for cond in sorted(df["condition"].unique()):
        subset = df[df["condition"] == cond]
        texts = subset["generated_text"].dropna().values
        lengths = [len(t.split()) for t in texts]

        results[cond] = {
            "n_generations": int(len(texts)),
            "mean_word_count": round(float(np.mean(lengths)), 1),
            "std_word_count": round(float(np.std(lengths)), 1),
            "median_word_count": round(float(np.median(lengths)), 1),
        }

    print("\nGeneration length statistics:")
    for cond, stats_dict in results.items():
        print(f"  {cond:20s}: mean={stats_dict['mean_word_count']:.0f} words, "
              f"n={stats_dict['n_generations']}")

    return results


def compute_toxicity_analysis(df: pd.DataFrame) -> dict:
    """Statistical analysis of toxicity scores across conditions."""
    results = {}

    for attr in TOXICITY_ATTRS:
        if attr not in df.columns:
            continue

        valid = df.dropna(subset=[attr])
        if len(valid) < 10:
            continue

        # Descriptive stats
        desc = valid.groupby("condition")[attr].agg(["mean", "std", "median", "count"])

        # Kruskal-Wallis (non-parametric, doesn't assume normality)
        groups = [g[attr].values for _, g in valid.groupby("condition") if len(g) >= 2]
        condition_labels = [name for name, g in valid.groupby("condition") if len(g) >= 2]

        if len(groups) < 2:
            continue

        h_stat, kw_p = stats.kruskal(*groups)

        # Pairwise Mann-Whitney U with Bonferroni
        n_pairs = len(condition_labels) * (len(condition_labels) - 1) // 2
        bonferroni_threshold = 0.05 / n_pairs if n_pairs > 0 else 0.05

        pairwise = []
        for i in range(len(condition_labels)):
            for j in range(i + 1, len(condition_labels)):
                g1 = groups[i]
                g2 = groups[j]
                u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")

                # Cohen's d
                pooled_std = np.sqrt(
                    ((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) /
                    (len(g1) + len(g2) - 2)
                )
                d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0

                pairwise.append({
                    "condition_1": condition_labels[i],
                    "condition_2": condition_labels[j],
                    "u_statistic": float(u_stat),
                    "p_value": float(u_p),
                    "cohens_d": round(float(d), 4),
                    "mean_1": round(float(g1.mean()), 4),
                    "mean_2": round(float(g2.mean()), 4),
                    "significant_bonferroni": u_p < bonferroni_threshold,
                })

        results[attr] = {
            "kruskal_wallis_h": round(float(h_stat), 4),
            "kruskal_wallis_p": float(kw_p),
            "significant_005": kw_p < 0.05,
            "descriptive": {
                cond: {
                    "mean": round(float(desc.loc[cond, "mean"]), 4),
                    "std": round(float(desc.loc[cond, "std"]), 4),
                    "median": round(float(desc.loc[cond, "median"]), 4),
                    "n": int(desc.loc[cond, "count"]),
                }
                for cond in desc.index
            },
            "pairwise_mannwhitney": pairwise,
            "bonferroni_threshold": bonferroni_threshold,
        }

        print(f"\n{attr.upper()}:")
        for cond in desc.index:
            print(f"  {cond:20s}: mean={desc.loc[cond, 'mean']:.4f} "
                  f"(+/- {desc.loc[cond, 'std']:.4f}), n={int(desc.loc[cond, 'count'])}")
        print(f"  Kruskal-Wallis: H={h_stat:.3f}, p={kw_p:.4e}")

        # Highlight key comparison: anti_suffrage vs base
        for pw in pairwise:
            if set([pw["condition_1"], pw["condition_2"]]) == {"anti_suffrage", "base"}:
                sig = "*" if pw["significant_bonferroni"] else ""
                print(f"  Anti-suffrage vs Base: d={pw['cohens_d']:.3f}, "
                      f"p={pw['p_value']:.4e} {sig}")

    return results


def compute_one_way_anova(df: pd.DataFrame) -> dict:
    """One-way ANOVA for toxicity across conditions."""
    results = {}

    for attr in ["toxicity"]:
        if attr not in df.columns:
            continue

        valid = df.dropna(subset=[attr])
        groups = [g[attr].values for _, g in valid.groupby("condition")]

        if len(groups) < 2:
            continue

        f_stat, p_val = stats.f_oneway(*groups)

        # Eta-squared
        grand_mean = valid[attr].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        ss_total = ((valid[attr] - grand_mean)**2).sum()
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        results[attr] = {
            "f_statistic": round(float(f_stat), 4),
            "p_value": float(p_val),
            "eta_squared": round(float(eta_sq), 4),
            "significant_005": p_val < 0.05,
        }
        print(f"\nANOVA ({attr}): F={f_stat:.3f}, p={p_val:.4e}, eta^2={eta_sq:.4f}")

    return results


def compute_prompt_effects(df: pd.DataFrame) -> dict:
    """Analyze effects of different generation prompts."""
    if "prompt" not in df.columns or "toxicity" not in df.columns:
        return {"note": "No prompt or toxicity data available"}

    valid = df.dropna(subset=["toxicity"])
    prompt_stats = valid.groupby(["condition", "prompt"])["toxicity"].agg(
        ["mean", "std", "count"]
    )
    print(f"\nToxicity by condition and prompt:")
    print(prompt_stats.to_string())

    return {"prompt_toxicity": prompt_stats.to_dict()}


def generate_comparison_table(results: dict) -> str:
    """Generate LaTeX comparison table."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Fine-tuning experiment: toxicity by condition}",
        r"\label{tab:finetune_toxicity}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Condition & Toxicity & Identity Attack & Insult & Cohen's $d$ vs Base \\",
        r"\midrule",
    ]

    tox = results.get("toxicity_analysis", {})
    for cond in CONDITIONS:
        tox_stats = tox.get("toxicity", {}).get("descriptive", {}).get(cond, {})
        ia_stats = tox.get("identity_attack", {}).get("descriptive", {}).get(cond, {})
        ins_stats = tox.get("insult", {}).get("descriptive", {}).get(cond, {})

        # Find Cohen's d vs base
        d_vs_base = "--"
        for pw in tox.get("toxicity", {}).get("pairwise_mannwhitney", []):
            if set([pw["condition_1"], pw["condition_2"]]) == {cond, "base"} and cond != "base":
                d_vs_base = f"{pw['cohens_d']:.3f}"

        cond_display = cond.replace("_", " ").title()
        lines.append(
            f"{cond_display} & "
            f"{tox_stats.get('mean', '--'):.4f} & "
            f"{ia_stats.get('mean', '--'):.4f} & "
            f"{ins_stats.get('mean', '--'):.4f} & "
            f"{d_vs_base} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model outputs")
    parser.add_argument(
        "--generations-dir",
        default="outputs/experiments",
        help="Directory containing generation parquet files",
    )
    parser.add_argument("--output-dir", default="outputs/experiments")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_generations(args.generations_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nRun the fine-tuning and generation steps first:")
        print("  python scripts/experiments/modal_finetune.py prepare-data")
        print("  modal run scripts/experiments/modal_finetune.py::app.run_finetuning")
        print("  modal run scripts/experiments/modal_finetune.py::app.run_generation")
        return

    results = {}

    # Sanity check
    results["generation_stats"] = compute_perplexity_check(df)

    # Toxicity analysis
    has_toxicity = any(attr in df.columns for attr in TOXICITY_ATTRS)
    if has_toxicity:
        results["toxicity_analysis"] = compute_toxicity_analysis(df)
        results["anova"] = compute_one_way_anova(df)
        results["prompt_effects"] = compute_prompt_effects(df)

        # LaTeX table
        latex = generate_comparison_table(results)
        latex_path = output_dir / "finetune_toxicity_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"\nLaTeX table saved to {latex_path}")
    else:
        print("\nNo toxicity scores found. Run Perspective API scoring first.")
        results["note"] = "Toxicity scoring not yet completed"

    # Save
    json_path = output_dir / "finetune_evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
