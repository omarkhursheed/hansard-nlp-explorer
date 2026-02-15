"""
Statistical significance tests for Hansard suffrage classification results.

Addresses reviewer criticism: "reports differences without any significance tests"
by providing chi-square, Mann-Kendall, z-tests, and multiple comparison corrections.

Usage:
    python scripts/experiments/statistical_tests.py
    python scripts/experiments/statistical_tests.py --data path/to/results.parquet
    python scripts/experiments/statistical_tests.py --output-dir outputs/experiments
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BUCKET_KEYS = [
    "competence_capacity",
    "emotion_morality",
    "equality",
    "instrumental_effects",
    "other",
    "religion_family",
    "social_experiment",
    "social_order_stability",
    "tradition_precedent",
]

STANCE_LABELS = ["for", "against", "both"]


def cramers_v(chi2: float, n: int, k: int, r: int) -> float:
    """Cramer's V effect size for chi-square test.

    Uses bias-corrected version (Bergsma 2013) to avoid overestimation
    in small contingency tables.
    """
    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    k_corr = k - (k - 1) ** 2 / (n - 1)
    r_corr = r - (r - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return np.sqrt(phi2_corr / denom)


def two_proportion_z_test(x1: int, n1: int, x2: int, n2: int):
    """Two-proportion z-test with pooled proportion under H0.

    Returns z-statistic, two-sided p-value, difference, and 95% CI.
    """
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0, 0.0, (0.0, 0.0)
    z = (p1 - p2) / se
    p_val = 2 * stats.norm.sf(abs(z))
    # Unpooled SE for CI (Wald interval)
    se_unpooled = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    ci = (diff - 1.96 * se_unpooled, diff + 1.96 * se_unpooled)
    return float(z), float(p_val), float(diff), (float(ci[0]), float(ci[1]))


def bootstrap_ci(data: np.ndarray, stat_fn=np.mean, n_boot: int = 10000,
                 ci: float = 0.95, seed: int = 42) -> tuple:
    """Bootstrap confidence interval for an arbitrary statistic."""
    rng = np.random.default_rng(seed)
    boot_stats = np.array([
        stat_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = 1 - ci
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Benjamini-Hochberg FDR correction.

    Returns list of dicts with original index, p-value, adjusted p-value,
    and whether the test is significant after correction.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [None] * n
    prev_adj = 0.0
    for rank_minus_1, (orig_idx, p) in enumerate(indexed):
        rank = rank_minus_1 + 1
        adj_p = min(1.0, p * n / rank)
        # Enforce monotonicity (step-up)
        adj_p = max(adj_p, prev_adj)
        adjusted[orig_idx] = adj_p
        prev_adj = adj_p

    results = []
    for i, (p, adj_p) in enumerate(zip(p_values, adjusted)):
        results.append({
            "index": i,
            "p_value": p,
            "p_adjusted_bh": adj_p,
            "significant_bh": adj_p < alpha,
        })
    return results


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} speeches from {path}")
    return df


def extract_bucket_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create boolean columns for each bucket_key per speech."""
    for bucket in BUCKET_KEYS:
        df[f"has_{bucket}"] = False

    for idx, row in df.iterrows():
        reasons = row.get("reasons")
        if isinstance(reasons, np.ndarray):
            for reason in reasons:
                if isinstance(reason, dict):
                    bk = reason.get("bucket_key", "")
                    col = f"has_{bk}"
                    if col in df.columns:
                        df.at[idx, col] = True
    return df


# ---------------------------------------------------------------------------
# Test 1: Stance x Gender chi-square
# ---------------------------------------------------------------------------

def test_stance_gender(df: pd.DataFrame) -> dict:
    """Chi-square test of independence: stance distribution vs speaker gender."""
    relevant = df[df["stance"].isin(STANCE_LABELS)].copy()
    ct = pd.crosstab(relevant["stance"], relevant["gender"])
    print(f"\n{'='*60}")
    print("TEST 1: Stance x Gender (chi-square)")
    print(f"{'='*60}")
    print(ct)

    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.values.sum()
    v = cramers_v(chi2, n, ct.shape[1], ct.shape[0])

    # Check expected cell counts
    min_expected = expected.min()
    warning = None
    if min_expected < 5:
        warning = f"Some expected counts < 5 (min={min_expected:.1f}). Consider Fisher's exact or merging categories."

    result = {
        "test": "chi2_independence",
        "description": "H0: Stance distribution is independent of speaker gender",
        "contingency_table": {
            col: {row: int(ct.loc[row, col]) for row in ct.index}
            for col in ct.columns
        },
        "chi2": float(chi2),
        "df": int(dof),
        "p_value": float(p),
        "cramers_v": float(v),
        "n": int(n),
        "min_expected_count": float(min_expected),
        "significant_005": p < 0.05,
    }
    if warning:
        result["warning"] = warning

    print(f"  chi2 = {chi2:.3f}, df = {dof}, p = {p:.4e}")
    print(f"  Cramer's V = {v:.4f}")
    print(f"  Min expected count = {min_expected:.1f}")
    if warning:
        print(f"  WARNING: {warning}")

    return result


# ---------------------------------------------------------------------------
# Test 2: Argument bucket x Gender chi-square (per bucket)
# ---------------------------------------------------------------------------

def test_bucket_gender(df: pd.DataFrame) -> dict:
    """For each argument bucket, chi-square test: is usage rate different by gender?

    Applies Bonferroni correction (9 tests).
    """
    relevant = df[df["stance"].isin(STANCE_LABELS)].copy()
    relevant = extract_bucket_flags(relevant)

    print(f"\n{'='*60}")
    print("TEST 2: Argument Bucket x Gender (per-bucket chi-square)")
    print(f"{'='*60}")

    bonferroni_threshold = 0.05 / len(BUCKET_KEYS)
    results_per_bucket = []
    p_values = []

    for bucket in BUCKET_KEYS:
        col = f"has_{bucket}"
        ct = pd.crosstab(relevant[col], relevant["gender"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            results_per_bucket.append({
                "bucket": bucket,
                "skipped": True,
                "reason": "Insufficient variation",
            })
            p_values.append(1.0)
            continue

        chi2, p, dof, expected = stats.chi2_contingency(ct)
        n = ct.values.sum()
        v = cramers_v(chi2, n, ct.shape[1], ct.shape[0])
        min_expected = expected.min()

        # Also compute usage rates per gender
        gender_groups = relevant.groupby("gender")[col].mean()
        rate_m = float(gender_groups.get("M", 0))
        rate_f = float(gender_groups.get("F", 0))

        entry = {
            "bucket": bucket,
            "chi2": float(chi2),
            "df": int(dof),
            "p_value": float(p),
            "cramers_v": float(v),
            "rate_male": round(rate_m, 4),
            "rate_female": round(rate_f, 4),
            "significant_bonferroni": p < bonferroni_threshold,
            "min_expected_count": float(min_expected),
        }
        if min_expected < 5:
            # Use Fisher's exact test for 2x2 tables
            if ct.shape == (2, 2):
                _, fisher_p = stats.fisher_exact(ct)
                entry["fisher_p"] = float(fisher_p)
                entry["fisher_significant_bonferroni"] = fisher_p < bonferroni_threshold
            entry["warning"] = f"Expected count < 5 (min={min_expected:.1f})"

        results_per_bucket.append(entry)
        p_values.append(p)

        sig = "*" if p < bonferroni_threshold else ""
        print(f"  {bucket:25s}: chi2={chi2:7.2f}, p={p:.4e}, V={v:.4f}  "
              f"M={rate_m:.3f} F={rate_f:.3f} {sig}")

    print(f"\n  Bonferroni threshold: p < {bonferroni_threshold:.4f}")

    return {
        "test": "chi2_per_bucket_gender",
        "description": "H0: Bucket usage rate is independent of gender (per bucket)",
        "bonferroni_threshold": bonferroni_threshold,
        "n_tests": len(BUCKET_KEYS),
        "results": results_per_bucket,
    }


# ---------------------------------------------------------------------------
# Test 3: Argument bucket x Stance chi-square
# ---------------------------------------------------------------------------

def test_bucket_stance(df: pd.DataFrame) -> dict:
    """Chi-square: are certain arguments more associated with for/against?"""
    relevant = df[df["stance"].isin(STANCE_LABELS)].copy()
    relevant = extract_bucket_flags(relevant)

    print(f"\n{'='*60}")
    print("TEST 3: Argument Bucket x Stance (chi-square)")
    print(f"{'='*60}")

    bonferroni_threshold = 0.05 / len(BUCKET_KEYS)
    results_per_bucket = []
    p_values = []

    for bucket in BUCKET_KEYS:
        col = f"has_{bucket}"
        ct = pd.crosstab(relevant[col], relevant["stance"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            results_per_bucket.append({
                "bucket": bucket, "skipped": True,
                "reason": "Insufficient variation",
            })
            p_values.append(1.0)
            continue

        chi2, p, dof, expected = stats.chi2_contingency(ct)
        n = ct.values.sum()
        v = cramers_v(chi2, n, ct.shape[1], ct.shape[0])
        min_expected = expected.min()

        # Compute usage rate by stance
        stance_rates = relevant.groupby("stance")[col].mean()

        entry = {
            "bucket": bucket,
            "chi2": float(chi2),
            "df": int(dof),
            "p_value": float(p),
            "cramers_v": float(v),
            "rate_for": round(float(stance_rates.get("for", 0)), 4),
            "rate_against": round(float(stance_rates.get("against", 0)), 4),
            "rate_both": round(float(stance_rates.get("both", 0)), 4),
            "significant_bonferroni": p < bonferroni_threshold,
            "min_expected_count": float(min_expected),
        }
        if min_expected < 5:
            entry["warning"] = f"Expected count < 5 (min={min_expected:.1f})"

        results_per_bucket.append(entry)
        p_values.append(p)

        sig = "*" if p < bonferroni_threshold else ""
        print(f"  {bucket:25s}: chi2={chi2:7.2f}, p={p:.4e}, V={v:.4f} {sig}")

    print(f"\n  Bonferroni threshold: p < {bonferroni_threshold:.4f}")

    return {
        "test": "chi2_per_bucket_stance",
        "description": "H0: Bucket usage rate is independent of stance (per bucket)",
        "bonferroni_threshold": bonferroni_threshold,
        "n_tests": len(BUCKET_KEYS),
        "results": results_per_bucket,
    }


# ---------------------------------------------------------------------------
# Test 4: Temporal trend significance (Mann-Kendall)
# ---------------------------------------------------------------------------

def test_temporal_trends(df: pd.DataFrame) -> dict:
    """Mann-Kendall test: is there a significant trend in stance over time?

    Tests proportion of 'for' speeches per decade.
    """
    import pymannkendall as mk

    relevant = df[df["stance"].isin(STANCE_LABELS)].copy()

    print(f"\n{'='*60}")
    print("TEST 4: Temporal Trends (Mann-Kendall)")
    print(f"{'='*60}")

    # Proportion of 'for' per decade
    decade_groups = relevant.groupby("decade")
    decade_stats = []
    for decade, group in decade_groups:
        n = len(group)
        if n < 5:
            continue
        n_for = (group["stance"] == "for").sum()
        n_against = (group["stance"] == "against").sum()
        prop_for = n_for / n
        prop_against = n_against / n
        decade_stats.append({
            "decade": int(decade),
            "n": int(n),
            "n_for": int(n_for),
            "n_against": int(n_against),
            "prop_for": round(float(prop_for), 4),
            "prop_against": round(float(prop_against), 4),
        })

    decade_df = pd.DataFrame(decade_stats).sort_values("decade")
    print(f"  Decades with >= 5 speeches: {len(decade_df)}")
    for _, row in decade_df.iterrows():
        print(f"    {int(row['decade'])}s: n={int(row['n']):4d}, "
              f"for={row['prop_for']:.3f}, against={row['prop_against']:.3f}")

    results = {"decade_data": decade_stats, "trends": {}}

    # Mann-Kendall on proportion of 'for'
    if len(decade_df) >= 4:
        for col_name, col_label in [("prop_for", "proportion_for"),
                                     ("prop_against", "proportion_against")]:
            series = decade_df[col_name].values
            mk_result = mk.original_test(series)
            trend_entry = {
                "trend": mk_result.trend,
                "p_value": float(mk_result.p),
                "tau": float(mk_result.Tau),
                "s": float(mk_result.s),
                "slope": float(mk_result.slope),
                "intercept": float(mk_result.intercept),
                "significant_005": mk_result.p < 0.05,
            }
            results["trends"][col_label] = trend_entry
            print(f"\n  Mann-Kendall ({col_label}):")
            print(f"    trend={mk_result.trend}, tau={mk_result.Tau:.4f}, "
                  f"p={mk_result.p:.4e}, slope={mk_result.slope:.6f}")
    else:
        results["trends"]["note"] = "Insufficient decades (need >= 4)"
        print("  Insufficient decades for Mann-Kendall test")

    results["test"] = "mann_kendall"
    results["description"] = "H0: No monotonic trend in stance proportions over decades"
    return results


# ---------------------------------------------------------------------------
# Test 5: Proportion comparisons with z-tests and bootstrap CIs
# ---------------------------------------------------------------------------

def test_proportion_comparisons(df: pd.DataFrame) -> dict:
    """Two-proportion z-tests for key comparisons, with bootstrap CIs."""
    relevant = df[df["stance"].isin(STANCE_LABELS)].copy()
    relevant = extract_bucket_flags(relevant)

    print(f"\n{'='*60}")
    print("TEST 5: Proportion Comparisons (z-tests + bootstrap CIs)")
    print(f"{'='*60}")

    comparisons = []

    # -- Comparison A: Male vs Female stance distribution --
    for stance in ["for", "against"]:
        male = relevant[relevant["gender"] == "M"]
        female = relevant[relevant["gender"] == "F"]
        x_m = (male["stance"] == stance).sum()
        x_f = (female["stance"] == stance).sum()
        n_m = len(male)
        n_f = len(female)

        z, p, diff, ci = two_proportion_z_test(x_m, n_m, x_f, n_f)

        # Bootstrap CI on the difference
        male_flags = (male["stance"] == stance).astype(float).values
        female_flags = (female["stance"] == stance).astype(float).values
        boot_lo, boot_hi = bootstrap_ci(
            np.concatenate([male_flags, female_flags]),
            stat_fn=lambda x: x[:n_m].mean() - x[n_m:n_m + n_f].mean()
            if len(x) >= n_m + n_f else 0,
            n_boot=10000,
        )
        # Better bootstrap: resample each group independently
        rng = np.random.default_rng(42)
        boot_diffs = []
        for _ in range(10000):
            bm = rng.choice(male_flags, size=n_m, replace=True).mean()
            bf = rng.choice(female_flags, size=n_f, replace=True).mean()
            boot_diffs.append(bm - bf)
        boot_diffs = np.array(boot_diffs)
        boot_lo = float(np.percentile(boot_diffs, 2.5))
        boot_hi = float(np.percentile(boot_diffs, 97.5))

        entry = {
            "comparison": f"'{stance}' rate: Male vs Female",
            "prop_male": round(x_m / n_m, 4),
            "prop_female": round(x_f / n_f, 4),
            "n_male": int(n_m),
            "n_female": int(n_f),
            "z_statistic": round(z, 4),
            "p_value": float(p),
            "difference": round(diff, 4),
            "ci_95_wald": [round(ci[0], 4), round(ci[1], 4)],
            "ci_95_bootstrap": [round(boot_lo, 4), round(boot_hi, 4)],
            "significant_005": p < 0.05,
        }
        comparisons.append(entry)
        print(f"  {entry['comparison']}:")
        print(f"    M={entry['prop_male']:.4f}, F={entry['prop_female']:.4f}, "
              f"diff={diff:.4f}, z={z:.3f}, p={p:.4e}")
        print(f"    Wald 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"    Bootstrap 95% CI: [{boot_lo:.4f}, {boot_hi:.4f}]")

    # -- Comparison B: Bucket usage in for vs against speeches --
    for_speeches = relevant[relevant["stance"] == "for"]
    against_speeches = relevant[relevant["stance"] == "against"]
    n_for = len(for_speeches)
    n_against = len(against_speeches)

    top_discriminative = [
        ("equality", "for"),
        ("religion_family", "against"),
        ("social_order_stability", "against"),
        ("emotion_morality", "against"),
    ]

    for bucket, expected_higher in top_discriminative:
        col = f"has_{bucket}"
        x_for = for_speeches[col].sum()
        x_against = against_speeches[col].sum()

        z, p, diff, ci = two_proportion_z_test(
            int(x_for), n_for, int(x_against), n_against
        )

        entry = {
            "comparison": f"'{bucket}' usage: For vs Against speeches",
            "prop_for": round(x_for / n_for, 4),
            "prop_against": round(x_against / n_against, 4),
            "n_for": n_for,
            "n_against": n_against,
            "z_statistic": round(z, 4),
            "p_value": float(p),
            "difference": round(diff, 4),
            "ci_95_wald": [round(ci[0], 4), round(ci[1], 4)],
            "significant_005": p < 0.05,
        }
        comparisons.append(entry)
        print(f"  {entry['comparison']}:")
        print(f"    for={entry['prop_for']:.4f}, against={entry['prop_against']:.4f}, "
              f"diff={diff:.4f}, z={z:.3f}, p={p:.4e}")

    return {
        "test": "two_proportion_z",
        "description": "Two-proportion z-tests with 95% CIs for key comparisons",
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Test 6: Multiple comparison correction (BH FDR)
# ---------------------------------------------------------------------------

def apply_fdr_correction(all_results: dict) -> dict:
    """Collect all p-values and apply BH FDR correction."""
    print(f"\n{'='*60}")
    print("TEST 6: Benjamini-Hochberg FDR Correction (all tests)")
    print(f"{'='*60}")

    p_values = []
    labels = []

    # Test 1
    if "stance_x_gender" in all_results:
        p_values.append(all_results["stance_x_gender"]["p_value"])
        labels.append("stance_x_gender")

    # Test 2
    if "bucket_x_gender" in all_results:
        for entry in all_results["bucket_x_gender"]["results"]:
            if not entry.get("skipped"):
                p_values.append(entry["p_value"])
                labels.append(f"bucket_gender_{entry['bucket']}")

    # Test 3
    if "bucket_x_stance" in all_results:
        for entry in all_results["bucket_x_stance"]["results"]:
            if not entry.get("skipped"):
                p_values.append(entry["p_value"])
                labels.append(f"bucket_stance_{entry['bucket']}")

    # Test 4
    if "temporal_trends" in all_results:
        for trend_key, trend_val in all_results["temporal_trends"].get("trends", {}).items():
            if isinstance(trend_val, dict) and "p_value" in trend_val:
                p_values.append(trend_val["p_value"])
                labels.append(f"trend_{trend_key}")

    # Test 5
    if "proportion_comparisons" in all_results:
        for comp in all_results["proportion_comparisons"]["comparisons"]:
            p_values.append(comp["p_value"])
            labels.append(comp["comparison"])

    bh_results = benjamini_hochberg(p_values, alpha=0.05)
    for i, (label, bh) in enumerate(zip(labels, bh_results)):
        bh["label"] = label
        sig = "*" if bh["significant_bh"] else ""
        print(f"  {label:45s}: p={bh['p_value']:.4e} -> q={bh['p_adjusted_bh']:.4e} {sig}")

    n_sig = sum(1 for r in bh_results if r["significant_bh"])
    print(f"\n  {n_sig} / {len(bh_results)} tests significant after BH correction (alpha=0.05)")

    return {
        "test": "benjamini_hochberg_fdr",
        "description": "FDR correction across all tests",
        "n_tests": len(p_values),
        "n_significant": n_sig,
        "results": bh_results,
    }


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def generate_latex_table(all_results: dict) -> str:
    """Generate a LaTeX table summarizing all test results."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Statistical significance tests for classification results}",
        r"\label{tab:statistical_tests}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Test & Variable & $\chi^2$ / $z$ & $p$-value & Effect size & Sig. \\",
        r"\midrule",
    ]

    # Test 1
    r1 = all_results.get("stance_x_gender", {})
    if r1:
        sig = "Yes" if r1.get("significant_005") else "No"
        lines.append(
            f"Stance $\\times$ Gender & Overall & "
            f"{r1.get('chi2', 0):.2f} & "
            f"{r1.get('p_value', 1):.2e} & "
            f"$V = {r1.get('cramers_v', 0):.3f}$ & {sig} \\\\"
        )

    # Test 2
    r2 = all_results.get("bucket_x_gender", {})
    if r2:
        lines.append(r"\midrule")
        for entry in r2.get("results", []):
            if entry.get("skipped"):
                continue
            bucket = entry["bucket"].replace("_", r"\_")
            sig = "Yes" if entry.get("significant_bonferroni") else "No"
            lines.append(
                f"Bucket $\\times$ Gender & {bucket} & "
                f"{entry.get('chi2', 0):.2f} & "
                f"{entry.get('p_value', 1):.2e} & "
                f"$V = {entry.get('cramers_v', 0):.3f}$ & {sig} \\\\"
            )

    # Test 3
    r3 = all_results.get("bucket_x_stance", {})
    if r3:
        lines.append(r"\midrule")
        for entry in r3.get("results", []):
            if entry.get("skipped"):
                continue
            bucket = entry["bucket"].replace("_", r"\_")
            sig = "Yes" if entry.get("significant_bonferroni") else "No"
            lines.append(
                f"Bucket $\\times$ Stance & {bucket} & "
                f"{entry.get('chi2', 0):.2f} & "
                f"{entry.get('p_value', 1):.2e} & "
                f"$V = {entry.get('cramers_v', 0):.3f}$ & {sig} \\\\"
            )

    # Test 4
    r4 = all_results.get("temporal_trends", {})
    if r4:
        lines.append(r"\midrule")
        for key, val in r4.get("trends", {}).items():
            if not isinstance(val, dict):
                continue
            sig = "Yes" if val.get("significant_005") else "No"
            clean_key = key.replace("_", r"\_")
            lines.append(
                f"Mann-Kendall & {clean_key} & "
                f"$\\tau = {val.get('tau', 0):.3f}$ & "
                f"{val.get('p_value', 1):.2e} & "
                f"slope $= {val.get('slope', 0):.4f}$ & {sig} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Statistical significance tests for Hansard classification"
    )
    parser.add_argument(
        "--data",
        default="outputs/llm_classification/claude_sonnet_45_full_results.parquet",
        help="Path to classification results parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experiments",
        help="Directory for output files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)

    # Summary
    relevant = df[df["stance"].isin(STANCE_LABELS)]
    print(f"\nRelevant speeches (for/against/both): {len(relevant)}")
    print(f"  Gender: M={len(relevant[relevant['gender']=='M'])}, "
          f"F={len(relevant[relevant['gender']=='F'])}")
    print(f"  Stance: {dict(relevant['stance'].value_counts())}")

    all_results = {}

    # Test 1: Stance x Gender
    all_results["stance_x_gender"] = test_stance_gender(df)

    # Test 2: Bucket x Gender
    all_results["bucket_x_gender"] = test_bucket_gender(df)

    # Test 3: Bucket x Stance
    all_results["bucket_x_stance"] = test_bucket_stance(df)

    # Test 4: Temporal trends
    all_results["temporal_trends"] = test_temporal_trends(df)

    # Test 5: Proportion comparisons
    all_results["proportion_comparisons"] = test_proportion_comparisons(df)

    # Test 6: BH FDR correction
    all_results["fdr_correction"] = apply_fdr_correction(all_results)

    # Save JSON
    json_path = output_dir / "statistical_tests_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Save LaTeX table
    latex_path = output_dir / "statistical_tests_table.tex"
    latex = generate_latex_table(all_results)
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to {latex_path}")

    # Summary
    fdr = all_results["fdr_correction"]
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {fdr['n_tests']}")
    print(f"Significant after BH FDR: {fdr['n_significant']}")

    return all_results


if __name__ == "__main__":
    main()
